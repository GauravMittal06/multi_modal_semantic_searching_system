# src/bm25_index.py
"""
BM25 index for lexical retrieval.
Built at ingest time, persisted to disk, loaded at query time.
Used alongside dense ANN retrieval in RRF hybrid search.
"""

import os
import json
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is required: pip install rank-bm25")

# ── Index storage path ────────────────────────────────────────────────────────
# Stored next to the source document, keyed by source_document name.
# Default: project root / bm25_cache / <source_document>.pkl

_CACHE_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "bm25_cache",
)


def _cache_path(source_document: str) -> str:
    safe_name = re.sub(r"[^\w\-.]", "_", source_document)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{safe_name}.bm25.pkl")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Lowercase, split on whitespace and punctuation, preserve numbers.
    Keeps tokens like '400m', 'mumbai', 'hyderabad', '60%', 'fy2025'.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:[%$][a-z0-9]*)?", text)
    return [t for t in tokens if len(t) > 1]


# ── Build ─────────────────────────────────────────────────────────────────────

def build_bm25_index(
    elements: List[Dict[str, Any]],
    source_document: str,
) -> None:
    """
    Build and persist a BM25 index from a list of ingested elements.
    Called at the end of ingest_document(), after upsert_elements().

    Uses the same text selection logic as embed_passages in ingest.py:
    - images: vision_summary or content
    - all others: content

    Stores element_id, type, page_number, section_heading alongside
    the BM25 index so search_bm25() can return full payloads.
    """
    texts = []
    metadata = []

    for elem in elements:
        if elem["type"] == "image":
            text = elem.get("vision_summary") or elem.get("content") or ""
        else:
            text = elem.get("content") or ""

        if not text.strip():
            text = f"{elem['type']} on page {elem['page_number']}"

        texts.append(text)
        metadata.append({
            "element_id": elem.get("id") or elem.get("element_id", ""),
            "type": elem["type"],
            "page_number": elem["page_number"],
            "section_heading": elem.get("section_heading", ""),
            "source_document": elem.get("source_document", source_document),
            "content": elem.get("content", ""),
            "vision_summary": elem.get("vision_summary", ""),
            "related_elements": elem.get("related_elements", []),
            "keywords": elem.get("keywords", []),
            "chart_facts": elem.get("chart_facts", ""),
            "metadata": elem.get("metadata", {}),
        })

    tokenized = [_tokenize(t) for t in texts]
    index = BM25Okapi(tokenized)

    payload = {
        "index": index,
        "metadata": metadata,
        "texts": texts,
    }

    path = _cache_path(source_document)
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[bm25] Index built: {len(texts)} elements → {path}")


# ── Load ──────────────────────────────────────────────────────────────────────

_loaded_indexes: Dict[str, Dict] = {}


def _load_index(source_document: str) -> Optional[Dict]:
    if source_document in _loaded_indexes:
        return _loaded_indexes[source_document]

    path = _cache_path(source_document)
    if not os.path.exists(path):
        print(f"[bm25] No index found for '{source_document}' at {path}")
        return None

    with open(path, "rb") as f:
        payload = pickle.load(f)

    _loaded_indexes[source_document] = payload
    print(f"[bm25] Index loaded: {len(payload['metadata'])} elements for '{source_document}'")
    return payload


def delete_bm25_index(source_document: str) -> None:
    """Call before re-ingestion to invalidate the old index."""
    path = _cache_path(source_document)
    if os.path.exists(path):
        os.remove(path)
        print(f"[bm25] Deleted index: {path}")
    _loaded_indexes.pop(source_document, None)


# ── Search ────────────────────────────────────────────────────────────────────

def search_bm25(
    query: str,
    source_document: str,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    BM25 lexical search over a single source document's index.

    Returns up to k results as payload dicts with a 'bm25_score' field.
    Results are sorted descending by BM25 score.
    Returns [] if no index exists (graceful degradation).
    """
    payload = _load_index(source_document)
    if payload is None:
        return []

    index: BM25Okapi = payload["index"]
    metadata: List[Dict] = payload["metadata"]

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = index.get_scores(tokens)

    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for idx, score in scored[:k]:
        if score <= 0.0:
            break  # BM25 scores of 0 mean no token overlap — not useful
        entry = dict(metadata[idx])
        entry["bm25_score"] = float(score)
        results.append(entry)

    return results