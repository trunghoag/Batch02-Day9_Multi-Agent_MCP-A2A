"""Task 8 - PageIndex vectorless RAG capability.

This embedded Day 08 module is used by the Day 09 MCP-style worker. It is strict:
if PageIndex is not configured, it raises a clear error instead of silently using
a local fallback.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
PAGEINDEX_STATE_PATH = Path(__file__).parent.parent / "data" / "index" / "pageindex_documents.json"


def has_real_api_key(value: str, placeholders: set[str]) -> bool:
    value = (value or "").strip()
    return bool(value) and value.lower() not in placeholders and not value.endswith("xxx")


def require_pageindex_client():
    """Return a PageIndex client or raise a clear setup error."""
    if not has_real_api_key(PAGEINDEX_API_KEY, {"pi_xxx", "xxx"}):
        raise RuntimeError("PAGEINDEX_API_KEY is required for the Day 09 MCP PageIndex capability.")
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:
        raise RuntimeError("The pageindex package is required. Install it before running the MCP capability.") from exc
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def upload_documents():
    """Upload legal PDFs to PageIndex and store returned document IDs."""
    client = require_pageindex_client()
    uploaded = []
    for pdf_file in sorted(LANDING_LEGAL_DIR.glob("*.pdf")):
        if pdf_file.stat().st_size <= 1024:
            continue
        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id") or response.get("document_id") or response.get("id")
        uploaded.append(
            {
                "doc_id": doc_id,
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "type": "legal",
                "raw_response": response,
            }
        )
        print(f"Uploaded: {pdf_file.name}")

    PAGEINDEX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_STATE_PATH.write_text(
        json.dumps(uploaded, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"status": "uploaded", "files": uploaded}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Query PageIndex. Raises if PageIndex is not configured or returns no results."""
    client = require_pageindex_client()
    doc_records = load_pageindex_records()
    if not doc_records:
        raise RuntimeError("No PageIndex document records found. Run upload_documents() first.")

    collected = []
    errors = []
    for record in doc_records:
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        try:
            response = client.submit_query(doc_id=doc_id, query=query)
            response = resolve_pageindex_retrieval(client, response)
            collected.extend(parse_pageindex_response(response, record))
        except Exception as exc:
            errors.append(f"{record.get('filename', 'unknown')}: {exc}")

    collected.sort(key=lambda item: item["score"], reverse=True)
    if collected:
        return collected[:top_k]
    detail = "; ".join(errors) if errors else "PageIndex returned no results."
    raise RuntimeError(f"PageIndex search failed: {detail}")


def load_pageindex_records() -> list[dict]:
    if not PAGEINDEX_STATE_PATH.exists():
        return []
    return json.loads(PAGEINDEX_STATE_PATH.read_text(encoding="utf-8"))


def parse_pageindex_response(response: dict, record: dict) -> list[dict]:
    candidates = []
    for key in ("results", "retrieval", "contexts", "chunks"):
        value = response.get(key)
        if isinstance(value, list):
            candidates = value
            break
    if not candidates and response.get("answer"):
        candidates = [{"text": response["answer"], "score": 1.0}]

    parsed = []
    for index, item in enumerate(candidates):
        if isinstance(item, str):
            text = item
            score = 1.0 / (index + 1)
            metadata = {}
        else:
            text = item.get("text") or item.get("content") or item.get("chunk") or str(item)
            score = float(item.get("score", 1.0 / (index + 1)))
            metadata = item.get("metadata", {})
        parsed.append(
            {
                "content": text,
                "score": score,
                "metadata": {
                    **metadata,
                    "source": record.get("filename", "pageindex"),
                    "doc_id": record.get("doc_id"),
                },
                "source": "pageindex",
            }
        )
    return parsed


def resolve_pageindex_retrieval(client, response: dict, attempts: int = 6, delay: float = 1.0) -> dict:
    retrieval_id = response.get("retrieval_id")
    if not retrieval_id:
        return response

    last_response = response
    for _ in range(attempts):
        time.sleep(delay)
        retrieval = client.get_retrieval(retrieval_id)
        last_response = retrieval
        if any(isinstance(retrieval.get(key), list) for key in ("results", "retrieval", "contexts", "chunks")):
            return retrieval
        if retrieval.get("answer"):
            return retrieval
    return last_response


if __name__ == "__main__":
    print("Uploading documents to PageIndex...")
    upload_documents()
    print("Testing PageIndex query...")
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
