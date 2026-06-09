"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math
from collections import Counter

from .task4_chunking_indexing import load_index, tokenize

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return SimpleBM25Okapi(tokenized_corpus)


class SimpleBM25Okapi:
    """Small BM25 implementation used by the embedded Day 08 artifact."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        doc_freq: Counter[str] = Counter()
        for doc in tokenized_corpus:
            doc_freq.update(set(doc))
        doc_count = max(1, len(tokenized_corpus))
        self.idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for idx, freqs in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[idx] or 1
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores.append(float(score))
        return scores


def get_corpus_and_index():
    """Lazy-load corpus từ local vector index."""
    global CORPUS, BM25_INDEX
    if not CORPUS:
        CORPUS = load_index(rebuild_if_missing=True)
    if BM25_INDEX is None and CORPUS:
        BM25_INDEX = build_bm25_index(CORPUS)
    return CORPUS, BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus, bm25 = get_corpus_and_index()
    if not corpus or bm25 is None or not query.strip():
        return []

    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    results = []
    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        if score <= 0:
            continue
        item = corpus[idx]
        results.append(
            {
                "content": item["content"],
                "score": score,
                "metadata": item.get("metadata", {}),
                "embedding": item.get("embedding", []),
            }
        )
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
