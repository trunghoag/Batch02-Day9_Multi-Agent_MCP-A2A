"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

from .task4_chunking_indexing import cosine_similarity, embed_text, tokenize

load_dotenv()


def has_real_api_key(value: str, placeholders: set[str]) -> bool:
    """True nếu env var có vẻ là API key thật, không phải placeholder demo."""
    value = (value or "").strip()
    return bool(value) and value.lower() not in placeholders and not value.endswith("xxx")


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    jina_key = os.getenv("JINA_API_KEY", "")
    use_jina = os.getenv("USE_JINA_RERANKER", "false").lower() == "true"
    if use_jina and has_real_api_key(jina_key, {"jina_xxx", "xxx"}):
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k,
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = response.json()["results"]
            return [
                {**candidates[item["index"]], "score": float(item["relevance_score"])}
                for item in reranked
            ]
        except Exception as exc:
            print(f"Jina reranker failed ({exc.__class__.__name__}); using local reranker.")

    query_terms = set(tokenize(query))
    query_embedding = embed_text(query)
    scored = []
    for rank, candidate in enumerate(candidates, 1):
        content_terms = set(tokenize(candidate.get("content", "")))
        overlap = len(query_terms & content_terms) / max(1, len(query_terms))
        dense = cosine_similarity(
            query_embedding,
            candidate.get("embedding") or embed_text(candidate.get("content", "")),
        )
        prior = float(candidate.get("score", 0.0))
        rerank_score = 0.55 * dense + 0.35 * overlap + 0.10 * (1 / (rank + 1))
        scored.append({**candidate, "score": float(max(rerank_score, prior * 0.01))})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding") or embed_text(
                candidates[idx].get("content", "")
            )
            relevance = cosine_similarity(query_embedding, candidate_embedding)
            max_sim_to_selected = 0.0
            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding") or embed_text(
                    candidates[selected_idx].get("content", "")
                )
                sim = cosine_similarity(candidate_embedding, selected_embedding)
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(item.get("score", 0.0))
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("metadata", {}).get("chunk_index")
            key = f"{item.get('metadata', {}).get('source', '')}:{key}:{item.get('content', '')[:80]}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    results = []
    for key, score in sorted_items[:top_k]:
        item = content_map[key].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr(embed_text(query), candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
