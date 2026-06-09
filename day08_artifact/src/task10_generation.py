"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


def has_real_api_key(value: str, placeholders: set[str]) -> bool:
    """True nếu env var có vẻ là API key thật, không phải placeholder demo."""
    value = (value or "").strip()
    return bool(value) and value.lower() not in placeholders and not value.endswith("xxx")


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    back.reverse()
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        chunk_index = metadata.get("chunk_index", "N/A")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type} | Chunk: {chunk_index}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if isinstance(context_chunks, int):
        top_k = context_chunks
        context_chunks = None

    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not reordered:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if has_real_api_key(openai_key, {"sk-xxx", "xxx"}):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or ""
            return {
                "answer": answer.strip(),
                "sources": chunks,
                "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
            }
        except Exception as exc:
            print(f"OpenAI generation failed ({exc.__class__.__name__}); trying Gemini fallback.")

    gemini_answer = generate_with_gemini(user_message)
    if gemini_answer:
        return {
            "answer": gemini_answer,
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        }

    return {
        "answer": build_extractive_answer(query, reordered),
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


def generate_with_gemini(user_message: str) -> str:
    """Call Gemini generateContent as fallback when OpenAI is unavailable."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not has_real_api_key(gemini_key, {"gemini_xxx", "google_xxx", "xxx"}):
        return ""

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "topP": TOP_P,
        },
    }

    for model in gemini_model_candidates():
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        try:
            response = requests.post(
                url,
                params={"key": gemini_key},
                json=payload,
                timeout=45,
            )
            response.raise_for_status()
            data = response.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            answer = "\n".join(part.get("text", "") for part in parts).strip()
            if answer:
                return answer
        except requests.HTTPError:
            message = extract_google_error_message(response)
            print(f"Gemini model {model} failed: {message}")
        except Exception as exc:
            print(f"Gemini model {model} failed ({exc.__class__.__name__}).")

    print("All Gemini models failed; using extractive fallback.")
    return ""


def gemini_model_candidates() -> list[str]:
    """Return primary Gemini model plus fallback models, preserving order."""
    primary = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    fallback = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite")
    models = [primary] + [model.strip() for model in fallback.split(",") if model.strip()]
    deduped = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def extract_google_error_message(response) -> str:
    """Compact Google API error message without leaking request details."""
    try:
        error = response.json().get("error", {})
        status = error.get("status", "HTTP_ERROR")
        message = error.get("message", response.text[:200])
        return f"{response.status_code} {status}: {message[:180]}"
    except Exception:
        return f"{response.status_code} HTTP_ERROR"


def build_extractive_answer(query: str, chunks: list[dict]) -> str:
    """Fallback generation không cần API: trích câu liên quan và gắn citation."""
    query_terms = set(re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE))
    cited_sentences = []
    for chunk in chunks[:TOP_K]:
        source_label = citation_label(chunk)
        sentences = split_sentences(chunk.get("content", ""))
        best_sentence = max(
            sentences,
            key=lambda sent: len(query_terms & set(re.findall(r"[\wÀ-ỹ]+", sent.lower(), flags=re.UNICODE))),
            default="",
        )
        best_sentence = best_sentence.strip()
        if best_sentence and len(best_sentence) > 20:
            cited_sentences.append(f"{best_sentence} [{source_label}]")
        if len(cited_sentences) >= 3:
            break

    if not cited_sentences:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    intro = "Dựa trên các nguồn đã truy xuất, các thông tin liên quan nhất là:"
    return intro + "\n\n" + "\n\n".join(cited_sentences)


def split_sentences(text: str) -> list[str]:
    """Tách câu đơn giản cho tiếng Việt và markdown."""
    cleaned = re.sub(r"\s+", " ", text.replace("#", " ")).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def citation_label(chunk: dict) -> str:
    """Tạo citation dạng [Nguồn, Năm]."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "Nguồn")
    year_match = re.search(r"(20\d{2}|19\d{2})", source)
    if not year_match:
        year_match = re.search(r"(20\d{2}|19\d{2})", chunk.get("content", ""))
    year = year_match.group(1) if year_match else "n.d."
    source_name = source.replace(".md", "").replace(".pdf", "").replace("_", "-")
    source_name = source_name[:80]
    return f"{source_name}, {year}"


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
