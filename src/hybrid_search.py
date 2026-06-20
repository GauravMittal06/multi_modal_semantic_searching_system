# src/hybrid_search.py
"""
Reciprocal Rank Fusion (RRF) combining BM25 and dense ANN results.
Drop-in replacement for the raw search_elements() call in retrieve_context().

Usage in retrieval.py:
    Replace:
        primary_hits = search_elements(query_vector=q_vec, k=100, source_document=source_document)
    With:
        primary_hits = hybrid_search(question=question, query_vector=q_vec, source_document=source_document)
"""

from typing import List, Dict, Any, Optional

try:
    from .rag_core import search_elements
    from .bm25_index import search_bm25
except ImportError:
    from rag_core import search_elements
    from bm25_index import search_bm25

# ── RRF constant ──────────────────────────────────────────────────────────────
# k=60 is the standard value from the original RRF paper (Cormack et al. 2009).
# Higher k reduces the impact of top ranks; lower k amplifies them.
# Do not tune this until you have evidence it needs changing.
_RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)  # rank is 0-indexed


def hybrid_search(
    question: str,
    query_vector: List[float],
    source_document: Optional[str] = None,
    dense_k: int = 60,
    bm25_k: int = 60,
    top_n: int = 60,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: dense ANN + BM25, merged via Reciprocal Rank Fusion.

    Args:
        question:        Raw query string (used for BM25 tokenization).
        query_vector:    Pre-computed dense embedding of the question.
        source_document: Filter both retrievers to this document. Recommended.
        dense_k:         How many candidates to pull from ANN.
        bm25_k:          How many candidates to pull from BM25.
        top_n:           Final merged candidate count passed to cross-encoder.

    Returns:
        List of payload dicts sorted by descending RRF score.
        Each dict has 'rrf_score', 'dense_rank' (or -1), 'bm25_rank' (or -1),
        and all standard payload fields (content, type, page_number, etc.)
        plus a 'score' field set to rrf_score for compatibility with
        existing retrieval.py logic.
    """

    # ── Dense ANN retrieval ───────────────────────────────────────────────────
    dense_hits = search_elements(
        query_vector=query_vector,
        k=dense_k,
        source_document=source_document,
    )
    print(f"[hybrid] dense hits: {len(dense_hits)}")

    # ── BM25 lexical retrieval ────────────────────────────────────────────────
    bm25_hits = []
    if source_document:
        bm25_hits = search_bm25(
            query=question,
            source_document=source_document,
            k=bm25_k,
        )
        print(f"[hybrid] bm25 hits: {len(bm25_hits)} (non-zero score)")
    else:
        print("[hybrid] No source_document specified — skipping BM25 (cross-document BM25 not supported)")

    # ── Build element_id → payload map ───────────────────────────────────────
    # Dense hits use element_id from payload.
    # BM25 hits use element_id from metadata.
    # Merge by element_id; dense payload is authoritative (has full Qdrant payload).

    all_elements: Dict[str, Dict[str, Any]] = {}

    for hit in dense_hits:
        eid = hit.get("element_id", "")
        if eid:
            all_elements[eid] = dict(hit)

    for hit in bm25_hits:
        eid = hit.get("element_id", "")
        if eid and eid not in all_elements:
            # BM25 found something dense missed — include it with bm25_score
            all_elements[eid] = dict(hit)

    # ── Build rank maps ───────────────────────────────────────────────────────
    dense_rank: Dict[str, int] = {
        hit.get("element_id", ""): i
        for i, hit in enumerate(dense_hits)
        if hit.get("element_id")
    }

    bm25_rank: Dict[str, int] = {
        hit.get("element_id", ""): i
        for i, hit in enumerate(bm25_hits)
        if hit.get("element_id")
    }

    # ── Compute RRF scores ────────────────────────────────────────────────────
    rrf_scores: Dict[str, float] = {}

    for eid in all_elements:
        score = 0.0
        if eid in dense_rank:
            score += _rrf_score(dense_rank[eid])
        if eid in bm25_rank:
            score += _rrf_score(bm25_rank[eid])
        rrf_scores[eid] = score

    # ── Sort and assemble output ──────────────────────────────────────────────
    sorted_eids = sorted(rrf_scores, key=lambda e: rrf_scores[e], reverse=True)

    results = []
    for eid in sorted_eids[:top_n]:
        elem = dict(all_elements[eid])
        elem["rrf_score"] = rrf_scores[eid]
        elem["dense_rank"] = dense_rank.get(eid, -1)
        elem["bm25_rank"] = bm25_rank.get(eid, -1)
        elem["score"] = rrf_scores[eid]  # compatibility: retrieval.py reads hit.get("score")
        results.append(elem)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    _log_hybrid_diagnostics(results, dense_rank, bm25_rank, rrf_scores)

    return results


def _log_hybrid_diagnostics(
    results: List[Dict[str, Any]],
    dense_rank: Dict[str, int],
    bm25_rank: Dict[str, int],
    rrf_scores: Dict[str, float],
) -> None:
    print("\n[hybrid] ===== RRF MERGE DIAGNOSTICS =====")
    dense_only = sum(1 for e in results if e["dense_rank"] >= 0 and e["bm25_rank"] < 0)
    bm25_only  = sum(1 for e in results if e["bm25_rank"] >= 0 and e["dense_rank"] < 0)
    both       = sum(1 for e in results if e["dense_rank"] >= 0 and e["bm25_rank"] >= 0)
    print(f"[hybrid] top-{len(results)} breakdown: dense_only={dense_only} bm25_only={bm25_only} both={both}")

    print("[hybrid] top-20 merged candidates:")
    for i, elem in enumerate(results[:20]):
        dr = elem["dense_rank"]
        br = elem["bm25_rank"]
        print(
            f"[hybrid]   rrf_rank={i} "
            f"rrf_score={elem['rrf_score']:.4f} "
            f"dense_rank={'--' if dr < 0 else dr} "
            f"bm25_rank={'--' if br < 0 else br} "
            f"type={elem.get('type')} "
            f"page={elem.get('page_number')} "
            f"section={elem.get('section_heading', '')!r:.40} "
            f"contains_priority2={'Priority 2' in (elem.get('content') or elem.get('vision_summary') or '')}"
        )
    print("[hybrid] ===== END RRF DIAGNOSTICS =====\n")