# src/rag_core.py
"""
Qdrant-backed vector store for multi-modal document elements.
Each element (paragraph, table, image, heading) is embedded and stored
with full metadata including related_elements for cross-element retrieval.
"""

import os
import uuid
import time
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except Exception:
    UnexpectedResponse = Exception

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GEMINI_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_DISTANCE

# ─── Qdrant client ───────────────────────────────────────────────────────────

_qdrant = None
try:
    if QDRANT_API_KEY:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        _qdrant = QdrantClient(url=QDRANT_URL)
except Exception as e:
    print(f"[rag_core] Qdrant init failed: {e}")

# ─── sentence transformer client ────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("[rag_core] Sentence transformer loaded.")

from sentence_transformers import CrossEncoder
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_pairs(query: str, candidates: List[str]) -> List[float]:
    """
    Cross-encoder reranking.

    Input:
        query = user question
        candidates = retrieved text/image/table contents

    Output:
        relevance scores
    """
    if not candidates:
        return []

    pairs = [[query, c] for c in candidates]

    try:
        scores = _reranker.predict(pairs)
        return scores.tolist()
    except Exception as e:
        print(f"[rag_core] rerank_pairs failed: {e}")
        return [0.0] * len(candidates)

# ─── Collection management ───────────────────────────────────────────────────

def _ensure_collection(dim: int):
    if _qdrant is None:
        raise RuntimeError("Qdrant not configured. Check QDRANT_URL and QDRANT_API_KEY.")
    try:
        _qdrant.get_collection(QDRANT_COLLECTION)
    except Exception:
        distance = qmodels.Distance.COSINE if QDRANT_DISTANCE.lower().startswith("cos") else qmodels.Distance.EUCLID
        try:
            _qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=distance),
            )
        except Exception:
            _qdrant.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=dim, distance=distance),
            )

    # Ensure payload indexes for filtering
    for field in ["source_document", "type", "section_heading", "element_id"]:
        try:
            _qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
    try:
        _qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="page_number",
            field_schema=qmodels.PayloadSchemaType.INTEGER,
        )
    except Exception:
        pass


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_passages(passages: List[str]) -> List[List[float]]:
    if not passages:
        return []
    return _embed_model.encode(passages, show_progress_bar=True).tolist()


def embed_query(text: str) -> List[float]:
    return _embed_model.encode([text]).tolist()[0]


# ─── Upsert ───────────────────────────────────────────────────────────────────

def upsert_elements(elements: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Upsert document elements into Qdrant.
    Each element's full metadata is stored as payload for citation retrieval.
    """
    if not elements or not embeddings:
        return
    if len(elements) != len(embeddings):
        raise ValueError(f"elements ({len(elements)}) and embeddings ({len(embeddings)}) count mismatch")

    _ensure_collection(len(embeddings[0]))

    points = []
    for elem, vec in zip(elements, embeddings):
        payload = {
            "element_id": elem["id"],
            "type": elem["type"],
            "content": elem.get("content", ""),
            "page_number": elem["page_number"],
            "section_heading": elem.get("section_heading", ""),
            "source_document": elem["source_document"],
            "related_elements": elem.get("related_elements", []),
            "related_edges": elem.get("related_edges", []),
            "vision_summary": elem.get("vision_summary", ""),
            "keywords": elem.get("keywords", []),
            "metadata": elem.get("metadata", {}),
            "chart_facts": elem.get("chart_facts", ""),
        }
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, elem["id"]))
        points.append(qmodels.PointStruct(id=point_id, vector=vec, payload=payload))

    _qdrant.upload_points(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    print(f"[rag_core] Upserted {len(points)} elements to Qdrant.")


# ─── Search ───────────────────────────────────────────────────────────────────

def search_elements(
    query_vector: List[float],
    k: int = 6,
    source_document: Optional[str] = None,
    element_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for relevant elements.
    Returns list of payload dicts with distance score.
    Supports optional filtering by source_document and element_type.
    """
    if _qdrant is None:
        raise RuntimeError("Qdrant not configured.")
    
    try:
        _qdrant.get_collection(QDRANT_COLLECTION)
    except Exception as e:
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION}' unavailable: {e}"
        )

    must_conditions = []
    if source_document:
        must_conditions.append(
            qmodels.FieldCondition(key="source_document", match=qmodels.MatchValue(value=source_document))
        )
    if element_types:
        must_conditions.append(
            qmodels.FieldCondition(key="type", match=qmodels.MatchAny(any=element_types))
        )

    query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

    try:
        results = _qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
            query_filter=query_filter,
        )
    except Exception as e:
        print(f"[rag_core] Search with filter failed ({e}). Raising exception to prevent cross-document leakage.")
        raise e

    output = []
    for hit in results:
        payload = hit.payload or {}
        score = getattr(hit, "score", None)
        output.append({**payload, "score": float(score) if score is not None else 0.0})

    return output


def fetch_elements_by_ids(element_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch specific elements from Qdrant by their element_id field (not Qdrant UUID).
    Used for cross-element retrieval via related_elements links.
    """
    if _qdrant is None or not element_ids:
        return []
    try:
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="element_id",
                match=qmodels.MatchAny(any=element_ids)
            )]
        )
        results, _ = _qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=len(element_ids) * 2,
            with_payload=True,
        )
        return [r.payload for r in results if r.payload]
    except Exception as e:
        print(f"[rag_core] fetch_elements_by_ids failed: {e}")
        return []


def fetch_all_elements_for_document(source_document: str, limit: int = 5000) -> List[Dict[str, Any]]:
    """
    Fetch every element belonging to a given source document, regardless of
    relevance to any query. Used for the full-document graph explorer view,
    where we need the complete node/edge set rather than a query-scoped one.
    """
    if _qdrant is None or not source_document:
        return []
    try:
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="source_document",
                match=qmodels.MatchValue(value=source_document)
            )]
        )
        results, _ = _qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return [r.payload for r in results if r.payload]
    except Exception as e:
        print(f"[rag_core] fetch_all_elements_for_document failed: {e}")
        return []


def delete_by_source(source_document: str):
    if _qdrant is None:
        return
    try:
        _qdrant.get_collection(QDRANT_COLLECTION)
    except Exception:
        print("[rag_core] Collection does not exist yet, skipping delete.")
        return
    try:
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="source_document",
                match=qmodels.MatchValue(value=source_document)
            )]
        )
        _qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=qmodels.FilterSelector(filter=flt),
        )
        print(f"[rag_core] Deleted existing elements for: {source_document}")
    except Exception as e:
        print(f"[rag_core] delete_by_source failed: {e}")