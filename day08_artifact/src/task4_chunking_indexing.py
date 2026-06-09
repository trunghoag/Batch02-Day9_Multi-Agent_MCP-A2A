"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
VECTOR_STORE_PATH = INDEX_DIR / "drug_law_chunks.json"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive splitter ổn định cho cả văn bản luật dài và bài báo ngắn; nó ưu
# tiên ranh giới đoạn/câu trước khi cắt theo ký tự nên ít làm vỡ ngữ cảnh.
CHUNK_SIZE = 500        # Đủ nhỏ để retrieval chính xác, vẫn giữ được 1-2 đoạn.
CHUNK_OVERLAP = 50      # 10% overlap để không mất ý ở ranh giới chunk.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Local hashing embedding giúp repo chạy offline. Nếu muốn embedding API đúng
# production hơn, có thể thay embed_text() bằng OpenAI text-embedding-3-small
# hoặc BGE-M3 theo README.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local-hashing-vietnamese-384")
EMBEDDING_DIM = 384

# Lưu index local JSON để không cần Docker/Weaviate Cloud khi demo. Có thể đổi
# sang Weaviate Cloud sau bằng WEAVIATE_URL và WEAVIATE_API_KEY.
VECTOR_STORE = "local_json"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        split_text = simple_split_text

    chunks = []
    for doc_index, doc in enumerate(documents):
        splits = [s.strip() for s in split_text(doc["content"]) if s.strip()]
        for chunk_index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "id": f"{doc_index:04d}-{chunk_index:04d}",
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def simple_split_text(text: str) -> list[str]:
    """Fallback splitter nếu langchain-text-splitters chưa được cài."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["content"])
    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, giữ chữ tiếng Việt và số."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Dense embedding local bằng feature hashing.

    Đây không thay thế semantic model production, nhưng đủ deterministic để
    demo/retrieval offline. Có thể nâng cấp sang OpenAI hoặc BGE-M3 sau.
    """
    vector = [0.0] * dim
    tokens = tokenize(text)
    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity cho hai vector đã có cùng dimension."""
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return float(sum(left[i] * right[i] for i in range(size)))


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunking_method": CHUNKING_METHOD,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "vector_store": VECTOR_STORE,
        },
        "chunks": chunks,
    }
    VECTOR_STORE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return VECTOR_STORE_PATH


def load_index(rebuild_if_missing: bool = True) -> list[dict]:
    """Load chunks đã index; tự build nếu thiếu."""
    if not VECTOR_STORE_PATH.exists() and rebuild_if_missing:
        run_pipeline()
    if not VECTOR_STORE_PATH.exists():
        return []
    payload = json.loads(VECTOR_STORE_PATH.read_text(encoding="utf-8"))
    return payload.get("chunks", [])


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
