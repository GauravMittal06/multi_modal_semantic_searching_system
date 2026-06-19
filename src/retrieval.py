# src/retrieval.py
"""
Cross-Element Retrieval — Module 4
Retrieves semantically relevant elements AND their related elements
(tables, images, paragraphs) to build a complete distributed context.
"""

from typing import List, Dict, Any, Optional

try:
    from .rag_core import embed_query, search_elements, fetch_elements_by_ids
except ImportError:
    from rag_core import embed_query, search_elements, fetch_elements_by_ids


def retrieve_context(
    question: str,
    source_document: Optional[str] = None,
    top_k: int = 5,
    expand_related: bool = True,
) -> List[Dict[str, Any]]:

    q_vec = embed_query(question)
    if not q_vec:
        print("[retrieval] embed_query returned empty vector")
        return []

    print(f"[retrieval] Searching with source_document filter: '{source_document}'")
    primary_hits = search_elements(
        query_vector=q_vec,
        k=top_k,
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

    # ── Type weights for reranking ────────────────────────────────────────────
    TYPE_WEIGHT = {
        "table":     1.10,
        "paragraph": 1.00,
        "image":     0.80,
        "heading":   0.70,
    }

    def _rerank_score(hit):
        base = hit.get("score", 0.0)
        type_mult = TYPE_WEIGHT.get(hit.get("type", "paragraph"), 1.00)
        # Removed the is_page_render penalty so we don't drop charts
        return base * type_mult

    primary_hits.sort(key=_rerank_score, reverse=True)

    seen_ids = set()
    context_elements = []

    for hit in primary_hits:
        eid = hit.get("element_id", "")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            hit["is_primary"] = True
            hit["reranked_score"] = _rerank_score(hit)
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
    for hit in table_hits:
        eid = hit.get("element_id", "")
        if eid and eid not in seen_ids:
            reranked = _rerank_score(hit)
            # Always include tables from pages already in context
            # For off-page tables, require minimum relevance score
            hit_page = hit.get("page_number")
            if hit_page not in covered_pages and reranked < 0.45:
                continue
            seen_ids.add(eid)
            hit["is_primary"] = True
            hit["reranked_score"] = reranked
            context_elements.append(hit)

    # ── Forced image fetch — skip low-score images on focused queries ─────────
    image_hits = search_elements(
        query_vector=q_vec,
        k=10,  # Increased search radius from 3 to 10
        source_document=source_document,
        element_types=["image"],
    )
    for hit in image_hits:
        eid = hit.get("element_id", "")
        if eid and eid not in seen_ids:
            reranked = _rerank_score(hit)
            # Lowered flat threshold so charts survive the cut
            if reranked < 0.15:
                continue
            seen_ids.add(eid)
            hit["is_primary"] = True
            hit["reranked_score"] = reranked
            context_elements.append(hit)

    if not expand_related:
        return context_elements

    # ── Related expansion: cap per primary hit, skip headings ────────────────
    MAX_RELATED_PER_HIT = 5
    related_ids_to_fetch = []

    for hit in primary_hits:
        # if hit.get("type") == "heading":
            # continue
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
            print(
                f"  -> type={hit.get('type')} | "
                f"page={hit.get('page_number')} | "
                f"section={hit.get('section_heading','')}"
            )
            if hit.get("type") == "heading":
                continue
            # Drop related elements from totally unrelated pages AND sections
            hit_page = hit.get("page_number")
            hit_section = hit.get("section_heading", "")
            if hit_page not in primary_pages and hit_section not in primary_sections:
                continue
            hit["is_primary"] = False
            hit["score"] = 0.0
            hit["reranked_score"] = 0.0
            context_elements.append(hit)

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

    return balanced

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


def build_citations(context_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    seen = set()
    print("\n========== BUILDING CITATIONS ==========")

    for elem in context_elements:
        if not elem.get("is_primary", False):
            continue
        elem_type = elem.get("type", "unknown")

        # Headings are semantic anchors, not citable evidence
        if elem_type == "heading":
            continue

        # Skip forced-in elements that scored too low to be genuinely relevant
        if elem.get("reranked_score", 1.0) < 0.25:
            continue

        page = elem.get("page_number", "?")
        section = elem.get("section_heading", "")
        source = elem.get("source_document", "")

        key = f"{source}|{page}|{elem_type}|{elem.get('element_id', '')}"
        if key in seen:
            continue
        seen.add(key)

        print(
            f"[citation] type={elem_type} | "
            f"page={page} | "
            f"score={round(elem.get('reranked_score', 0.0), 3)} | "
            f"section={section}"
        )

        citations.append({
            "source_document": source,
            "page_number": page,
            "element_type": elem_type,
            "section_name": section,
        })

    print(f"[citation] total citations: {len(citations)}")
    print("========================================\n")

    return citations


def _humanize_relation(type_a: str, type_b: str) -> Optional[str]:
    pair = {type_a, type_b}
    if pair == {"paragraph", "table"}:
        return "Paragraph explains Table"
    if pair == {"paragraph", "image"}:
        return "Paragraph references Image"
    if pair == {"table", "image"}:
        return "Image visualizes Table data"
    if pair == {"table"} and type_a == type_b == "table":
        return "Related tables connected by topic"
    return None


def build_explainability(
    context_elements: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Judge-friendly, non-technical summary of how the answer was formed.
    No scores, IDs, or vector metadata — only type/page/section level info.
    """
    _seen_evidence_keys = set()
    evidence = []
    for e in context_elements:
        if e.get("type") == "heading":
            continue
        if not ((e.get("content") or "").strip() or (e.get("vision_summary") or "").strip()):
            continue
        # Dedup by element_id when available, falling back to a content-based
        # key when element_id is missing/empty so duplicate payloads from
        # upstream retrieval don't get double-counted in the evidence panel.
        eid = e.get("element_id", "")
        dedup_key = eid if eid else f"{e.get('source_document','')}|{e.get('page_number','')}|{e.get('type','')}|{(e.get('content') or e.get('vision_summary') or '')[:120]}"
        if dedup_key in _seen_evidence_keys:
            continue
        _seen_evidence_keys.add(dedup_key)
        evidence.append(e)

    evidence_used = [
        {
            "type": e.get("type", "unknown").capitalize(),
            "page_number": e.get("page_number", "?"),
            "section_heading": e.get("section_heading", ""),
        }
        for e in evidence
    ]

    # ── Relationships used (human-readable, derived from shared section) ──────
    relationships_used: List[str] = []
    seen_rel = set()
    for i, a in enumerate(evidence):
        for b in evidence[i + 1:]:
            if a.get("section_heading") and a.get("section_heading") == b.get("section_heading"):
                label = _humanize_relation(a.get("type", ""), b.get("type", ""))
                if label and label not in seen_rel:
                    seen_rel.add(label)
                    relationships_used.append(label)

            for para, img in ((a, b), (b, a)):
                if para.get("type") == "paragraph" and img.get("type") == "image":
                    content = (para.get("content") or "").lower()
                    if len(content.split()) <= 15 and any(
                        k in content for k in ["figure", "fig.", "chart", "table"]
                    ):
                        label = "Caption supports Image"
                        if label not in seen_rel:
                            seen_rel.add(label)
                            relationships_used.append(label)

    relationships_used = relationships_used[:6]

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

    if n >= 4 and modality_count >= 2 and citations and avg_score >= 0.45:
        confidence = "High"
    elif n >= 2 and avg_score >= 0.30:
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