# src/retrieval.py
"""
Cross-Element Retrieval — Module 4
Retrieves semantically relevant elements AND their related elements
(tables, images, paragraphs) to build a complete distributed context.
"""
import re
from typing import List, Dict, Any, Optional

try:
    from .rag_core import embed_query, search_elements, fetch_elements_by_ids, rerank_pairs
    from .graph_view import relationships_used_from_subgraph
except ImportError:
    from rag_core import embed_query, search_elements, fetch_elements_by_ids, rerank_pairs
    from graph_view import relationships_used_from_subgraph

TABLE_THRESHOLD = -1.5
IMAGE_THRESHOLD = -1.0
CITATION_THRESHOLD = -0.5
HIGH_CONFIDENCE = 2.5
MEDIUM_CONFIDENCE = 0.0
ANSWER_CITATION_THRESHOLD = -0.5
ANSWER_ATTRIBUTION_THRESHOLD = -0.5
PRIMARY_PROMOTION_THRESHOLD = 0.5

def _scoring_text(elem: Dict[str, Any]) -> str:
    """
    Returns clean text for cross-encoder scoring.
    Strips [TABLE COLUMNS:] / [KEY VALUES:] prefixes from tables,
    then normalises pipe-delimited structure to prose-like tokens.
    For images uses vision_summary. For all others uses content.
    """
    if elem.get("type") == "table":
        raw = elem.get("content", "")
        lines = raw.splitlines()
        clean_lines = [
            l for l in lines
            if not l.startswith("[TABLE COLUMNS:") and not l.startswith("[KEY VALUES:")
        ]
        cleaned = "\n".join(clean_lines).strip()
        cleaned = re.sub(r"\s*\|\s*", " ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned
    return elem.get("vision_summary") or elem.get("content", "")

def retrieve_context(
    question: str,
    source_document: Optional[str] = None,
    top_k: int = 5,
    expand_related: bool = True,
) -> List[Dict[str, Any]]:

    q_vec = embed_query(question)
    ref = _extract_reference(question)
    if not q_vec:
        print("[retrieval] embed_query returned empty vector")
        return []

    print(f"[retrieval] Searching with source_document filter: '{source_document}'")
    primary_hits = search_elements(
        query_vector=q_vec,
        k=40,
        source_document=source_document,
    )
    print(f"[retrieval] primary_hits count: {len(primary_hits)}")
    print("\n========== PRIMARY HITS ==========")
    for i, hit in enumerate(primary_hits):
        print(
            f"[{i+1}] type={hit.get('type')} | "
            f"score={round(hit.get('score', 0.0), 3)} | "
            f"page={hit.get('page_number')} | "
            f"section={hit.get('section_heading', '')}"
        )
    print("==================================\n")

    if not primary_hits:
        print("[retrieval] No hits with filter — trying unfiltered search to diagnose...")
        unfiltered = search_elements(query_vector=q_vec, k=top_k, source_document=None)
        print(f"[retrieval] Unfiltered hits: {len(unfiltered)}")
        for h in unfiltered:
            print(f"  → source_document in payload: '{h.get('source_document')}'")
        return []

    # Deprioritize page snapshots before cross-encoder — they flood short queries
    for h in primary_hits:
        if h.get("metadata", {}).get("is_page_render") or h.get("metadata", {}).get("image_type") == "page_snapshot":
            h["score"] = h.get("score", 0.0) * 0.5

    primary_texts = [h.get("vision_summary") or h.get("content", "") for h in primary_hits]
    primary_scores = rerank_pairs(question, primary_texts)

    for h, s in zip(primary_hits, primary_scores):

        vector_score = h.get("score", 0.0)

        h["reranked_score"] = (
            0.35 * vector_score
            + 0.65 * s
        )

        print(
            f"[fusion] "
            f"vector={vector_score:.3f} "
            f"cross={s:.3f} "
            f"final={h['reranked_score']:.3f}"
        )
        
        
    # ── Figure/Table reference boost ───────────────────────────────

    if ref:
        target = f"{ref['kind']} {ref['number']}".lower()

        print(f"[retrieval] Detected reference query: {target}")

        for hit in primary_hits:
            text = (
                (hit.get("content", "") or "")
                + " "
                + (hit.get("vision_summary", "") or "")
            ).lower()

            if target in text:
                hit["reranked_score"] += 100.0

                print(
                    f"[retrieval] Boosted reference match "
                    f"page={hit.get('page_number')} "
                    f"type={hit.get('type')}"
                )

    primary_hits.sort(key=lambda h: h["reranked_score"], reverse=True)
    print("\n===== TOP 20 AFTER RERANK =====")

    for hit in primary_hits[:20]:
        print(
            f"\nPAGE {hit.get('page_number')}"
            f"\nTYPE {hit.get('type')}"
            f"\nSECTION {hit.get('section_heading')}"
            f"\nSCORE {hit.get('reranked_score'):.3f}"
        )
    
        print(
            (hit.get("content") or hit.get("vision_summary") or "")[:150]
        )
    expanded_seed_hits = primary_hits[:15]
    primary_hits = primary_hits[:8]

    seen_ids = set()
    context_elements = []

    for hit in expanded_seed_hits:
        eid = hit.get("element_id", "")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            hit["is_primary"] = True
            context_elements.append(hit)

    # ── Determine if query is page-focused ───────────────────────────────────
    page_counts = {}
    for h in primary_hits:
        p = h.get("page_number")
        page_counts[p] = page_counts.get(p, 0) + 1
    is_focused_query = max(page_counts.values(), default=0) >= 4

    # ── Forced table fetch — always ensure tables in context ─────────────────
    # Get pages already covered by primary hits
    covered_pages = set(h.get("page_number") for h in primary_hits)

    table_hits = search_elements(
        query_vector=q_vec,
        k=5,
        source_document=source_document,
        element_types=["table"],
    )
    table_texts = [_scoring_text(h) for h in table_hits]
    table_scores = rerank_pairs(question, table_texts)
    for hit, reranked in zip(table_hits, table_scores):

        vector_score = hit.get("score", 0.0)

        reranked = (
            0.35 * vector_score
            + 0.65 * reranked
        )
        cross_score = table_scores[table_hits.index(hit)]

        print(
            f"[table-rank] "
            f"page={hit.get('page_number')} "
            f"vector={vector_score:.3f} "
            f"cross={cross_score:.3f} "
            f"final={reranked:.3f}"
        )
        eid = hit.get("element_id", "")
        if eid and eid not in seen_ids:
            if reranked < TABLE_THRESHOLD:
                continue
        
            seen_ids.add(eid)
            hit["is_primary"] = True
            hit["reranked_score"] = reranked
            context_elements.append(hit)

    # ── Forced image fetch — skip low-score images on focused queries ─────────
    image_hits = search_elements(
        query_vector=q_vec,
        k=10,
        source_document=source_document,
        element_types=["image"],
    )
    image_texts = [h.get("vision_summary") or h.get("content", "") for h in image_hits]
    image_scores = rerank_pairs(question, image_texts)
    for hit, reranked in zip(image_hits, image_scores):

        vector_score = hit.get("score", 0.0)

        reranked = (
            0.35 * vector_score
            + 0.65 * reranked
        )
        # print(f"[calibration] image score={reranked:.4f}")
        eid = hit.get("element_id", "")
        if eid and eid not in seen_ids:
            if reranked < IMAGE_THRESHOLD:
                continue
            seen_ids.add(eid)
            hit["is_primary"] = True
            hit["reranked_score"] = reranked
            context_elements.append(hit)

    if not expand_related:
        return context_elements

    # ── Related expansion: cap per primary hit, skip headings ────────────────
    MAX_RELATED_PER_HIT = 2
    related_ids_to_fetch = []

    for hit in primary_hits:
        # if hit.get("reranked_score", 0) < 0:
        #     continue

        added_for_this_hit = 0
        for rel_id in hit.get("related_elements", []):
            if added_for_this_hit >= MAX_RELATED_PER_HIT:
                break
            if rel_id not in seen_ids:
                related_ids_to_fetch.append(rel_id)
                seen_ids.add(rel_id)
                added_for_this_hit += 1

    related_ids_to_fetch = related_ids_to_fetch[:12]

    print(f"[retrieval] related_ids_to_fetch count: {len(related_ids_to_fetch)}")

    if related_ids_to_fetch:
        related_hits = fetch_elements_by_ids(related_ids_to_fetch)
        print(f"[retrieval] related_hits returned: {len(related_hits)}")
        primary_pages = set(h.get("page_number") for h in primary_hits)
        primary_sections = set(h.get("section_heading", "") for h in primary_hits if h.get("section_heading"))
        for hit in related_hits:
            if hit.get("type") == "heading":
                continue
            hit_page = hit.get("page_number")
            hit_section = hit.get("section_heading", "")
            if hit_page not in primary_pages and hit_section not in primary_sections:
                continue
            hit["is_primary"] = False
            hit["score"] = 0.0
            parent_score = max(
                (
                    h.get("reranked_score", 0)
                    for h in primary_hits
                    if hit_page == h.get("page_number")
                ),
                default=0.0,
            )

            hit["reranked_score"] = parent_score * 0.8
            context_elements.append(hit)

    # ── Promote high-scoring related elements to primary ──────────────────────
    for elem in context_elements:
        if not elem.get("is_primary", False):
            if elem.get("reranked_score", -999) >= PRIMARY_PROMOTION_THRESHOLD:
                elem["is_primary"] = True

    # ── Final dedup by element_id ─────────────────────────────────────────────
    final_deduped = []
    final_seen = set()
    for elem in context_elements:
        eid = elem.get("element_id", "")
        if eid and eid in final_seen:
            continue
        final_seen.add(eid)
        final_deduped.append(elem)

    # ── Type-balanced context cap ─────────────────────────────────────────────
    TYPE_LIMITS = {"paragraph": 8, "table": 5, "image": 5, "heading": 2}
    type_counts = {t: 0 for t in TYPE_LIMITS}
    balanced = []

    for elem in final_deduped:
        if not elem.get("is_primary"):
            continue
        t = elem.get("type", "paragraph")
        limit = TYPE_LIMITS.get(t, 3)
        if type_counts.get(t, 0) < limit:
            type_counts[t] = type_counts.get(t, 0) + 1
            balanced.append(elem)

    for elem in final_deduped:
        if elem.get("is_primary"):
            continue
        t = elem.get("type", "paragraph")
        limit = TYPE_LIMITS.get(t, 3)
        if type_counts.get(t, 0) < limit:
            type_counts[t] = type_counts.get(t, 0) + 1
            balanced.append(elem)

    balanced.sort(key=lambda x: (not x.get("is_primary", False), -x.get("reranked_score", 0.0)))

    # ── Modality diagnostics ────────────────────────────────────────
    modality_counts = {
        "paragraph": 0,
        "table": 0,
        "image": 0,
        "heading": 0,
    }

    for elem in balanced:
        t = elem.get("type", "unknown")
        modality_counts[t] = modality_counts.get(t, 0) + 1

    print("\n========== MODALITY SUMMARY ==========")
    for modality, count in modality_counts.items():
        print(f"{modality}: {count}")
    print("======================================")

    print("\n========== FINAL CONTEXT ==========")
    for elem in balanced:
        print(
            f"type={elem.get('type')} | "
            f"primary={elem.get('is_primary')} | "
            f"score={round(elem.get('reranked_score', 0.0), 3)} | "
            f"page={elem.get('page_number')} | "
            f"section={elem.get('section_heading','')}"
        )
    print("===================================\n")

    # ── Section distribution diagnostics ─────────────────────────────
    section_counts = {}

    for elem in balanced:
        sec = elem.get("section_heading", "") or "NO_SECTION"
        section_counts[sec] = section_counts.get(sec, 0) + 1

    print("========== SECTION DISTRIBUTION ==========")

    for sec, count in sorted(
        section_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{count}x | {sec}")

    print("==========================================\n")

    return balanced

def _extract_reference(query: str):
    m = re.search(
        r"(figure|fig\.?|table)\s+(\d+(?:\.\d+)?)",
        query,
        re.I,
    )

    if not m:
        return None

    return {
        "kind": m.group(1).lower(),
        "number": m.group(2),
    }

def format_context_for_llm(context_elements: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved elements into a structured context string for the LLM prompt.
    Each element is labeled with its type and source metadata.
    """
    if not context_elements:
        return "No context available."

    parts = []
    for i, elem in enumerate(context_elements):
        elem_type = elem.get("type", "unknown").upper()
        page = elem.get("page_number", "?")
        section = elem.get("section_heading", "")
        source = elem.get("source_document", "")
        is_primary = elem.get("is_primary", False)

        # Content: for images use vision_summary; for others use content
        if elem_type == "IMAGE":
            content = elem.get("vision_summary") or elem.get("content", "")
            chart_facts = elem.get("chart_facts", "")
            if chart_facts and chart_facts != "N/A":
                content += f"\nChart facts: {chart_facts}"
            keywords = elem.get("keywords", [])
            if keywords:
                content += f"\nKeywords: {', '.join(keywords)}"
            is_page_render = elem.get("metadata", {}).get("is_page_render", False)
            display_type = "PAGE SNAPSHOT" if is_page_render else "FIGURE"
        else:
            content = elem.get("content", "")
            display_type = elem_type

        if not content.strip():
            continue

        label = f"[{i+1}] {'★' if is_primary else '○'} {display_type}"
        meta = f"Source: {source} | Page: {page} | Section: {section}"
        parts.append(f"{label}\n{meta}\n{content}")

    return "\n\n---\n\n".join(parts)

def build_citations(
    context_elements: List[Dict[str, Any]],
    answer_text: str = "",
) -> List[Dict[str, Any]]:
    citations = []
    seen = set()
    print("\n========== BUILDING CITATIONS ==========")

    # Gate 1: primary, non-heading, above retrieval threshold
    candidates = [
        elem for elem in context_elements
        if elem.get("is_primary", False)
        and elem.get("type", "unknown") != "heading"
        and elem.get("reranked_score", 1.0) >= CITATION_THRESHOLD
    ]

    # Gate 2: answer attribution
    if answer_text.strip() and candidates:
        candidate_texts = [_scoring_text(elem) for elem in candidates]
        attribution_scores = rerank_pairs(answer_text, candidate_texts)
    else:
        attribution_scores = [1.0] * len(candidates)

    modality_counts: Dict[str, int] = {}

    for elem, attr_score in zip(candidates, attribution_scores):
        if attr_score < ANSWER_CITATION_THRESHOLD:
            print(
                f"[citation] EXCLUDED by attribution | "
                f"attr={attr_score:.3f} | "
                f"page={elem.get('page_number')} | "
                f"section={elem.get('section_heading', '')}"
            )
            continue

        elem_type = elem.get("type", "unknown")
        page = elem.get("page_number", "?")
        section = elem.get("section_heading", "")
        source = elem.get("source_document", "")

        key = f"{source}|{page}|{elem_type}|{elem.get('element_id', '')}"
        if key in seen:
            continue
        seen.add(key)

        elem["_attribution_score"] = attr_score
        modality_counts[elem_type] = modality_counts.get(elem_type, 0) + 1

        print(
            f"[citation] type={elem_type} | "
            f"page={page} | "
            f"retrieval={round(elem.get('reranked_score', 0.0), 3)} | "
            f"attribution={round(attr_score, 3)} | "
            f"section={section}"
        )

        citations.append({
            "source_document": source,
            "page_number": page,
            "element_type": elem_type,
            "section_name": section,
        })

    print(f"[citation] modality summary:")
    for t, c in modality_counts.items():
        print(f"  {t}: {c}")
    print(f"[citation] total citations: {len(citations)}")
    print("========================================\n")

    return citations


def build_explainability(
    context_elements: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    answer_text: str = "",
) -> Dict[str, Any]:
    """
    Judge-friendly, non-technical summary of how the answer was formed.
    Evidence is built from answer attribution, independent of citations.
    """
    _seen_evidence_keys = set()
    evidence = []

    primary_candidates = [
        e for e in context_elements
        if e.get("is_primary", False)
        and e.get("type") != "heading"
        and ((e.get("content") or "").strip() or (e.get("vision_summary") or "").strip())
    ]

    if answer_text.strip() and primary_candidates:
        candidate_texts = [_scoring_text(e) for e in primary_candidates]
        attr_scores = rerank_pairs(answer_text, candidate_texts)
    else:
        attr_scores = [1.0] * len(primary_candidates)

    for e, attr_score in zip(primary_candidates, attr_scores):
        if attr_score < ANSWER_ATTRIBUTION_THRESHOLD:
            continue
        eid = e.get("element_id", "")
        dedup_key = eid if eid else f"{e.get('source_document','')}|{e.get('page_number','')}|{e.get('type','')}|{(e.get('content') or e.get('vision_summary') or '')[:120]}"
        if dedup_key in _seen_evidence_keys:
            continue
        _seen_evidence_keys.add(dedup_key)
        e["_attribution_score"] = attr_score
        evidence.append(e)

    evidence_used = [
        {
            "type": e.get("type", "unknown").capitalize(),
            "page_number": e.get("page_number", "?"),
            "section_heading": e.get("section_heading", ""),
        }
        for e in evidence
    ]

    # ── Relationships used (real graph edges between evidence elements) ───────
    # Uses the persisted related_edges (typed, directed) from relationships.py
    # rather than a same-section heuristic — see graph_view.py for the
    # subgraph reconstruction this is built on.
    try:
        relationships_used = relationships_used_from_subgraph(context_elements)
    except Exception as ex:
        print(f"[explainability] relationships_used_from_subgraph failed: {ex}")
        relationships_used = []

    # ── Modalities used ────────────────────────────────────────────────────────
    types_present = set(e.get("type") for e in evidence)
    modalities_used = {
        "text": "paragraph" in types_present,
        "table": "table" in types_present,
        "image": "image" in types_present,
    }

    # ── Retrieval summary ──────────────────────────────────────────────────────
    pages = set(e.get("page_number") for e in evidence)
    type_counts: Dict[str, int] = {}
    for e in evidence:
        t = e.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    retrieval_summary = {
        "total_elements": len(evidence),
        "page_count": len(pages),
        "type_counts": type_counts,
    }

    # ── Confidence ──────────────────────────────────────────────────────────────
    modality_count = sum(1 for v in modalities_used.values() if v)
    n = len(evidence)
    avg_score = sum(
        e.get("reranked_score", 0.0) for e in context_elements if e.get("is_primary")
    ) / max(sum(1 for e in context_elements if e.get("is_primary")), 1)

    if avg_score >= HIGH_CONFIDENCE and citations:
        confidence = "High"
        if modality_count < 2:
            confidence = "Medium"
    elif avg_score >= MEDIUM_CONFIDENCE:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "evidence_used": evidence_used,
        "relationships_used": relationships_used,
        "modalities_used": modalities_used,
        "retrieval_summary": retrieval_summary,
        "confidence": confidence,
    }