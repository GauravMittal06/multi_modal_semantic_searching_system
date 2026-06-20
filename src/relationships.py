# src/relationships.py
"""
Relationship Modeling — Module 3
Builds a lightweight NetworkX graph linking document elements by:
- Proximity (same page, adjacent position)
- Heading ownership (heading → paragraph/table/image on same page)
- Caption-to-image linking
- Cross-element semantic bridges
"""

from typing import List, Dict, Any
import networkx as nx
import numpy as np
from src.rag_core import embed_passages

SIM_THRESHOLD = 0.60
SAME_SECTION_BONUS = 0.10
SAME_PAGE_BONUS = 0.05

def _cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def build_relationship_graph(elements: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Takes the flat list of elements and returns a directed graph
    where nodes are element IDs and edges are labeled relationships.
    """
    G = nx.DiGraph()
    
    embeds = embed_passages([e.get("content","") or e.get("vision_summary","") for e in elements])
    emb_map = {e["id"]: v for e, v in zip(elements, embeds)}

    def _hybrid_score(e1, e2):
        sim = _cos(emb_map[e1["id"]], emb_map[e2["id"]])
        bonus = 0.0
        if e1.get("section_heading") == e2.get("section_heading") and e1.get("section_heading"):
            bonus += SAME_SECTION_BONUS
        if e1.get("page_number") == e2.get("page_number"):
            bonus += SAME_PAGE_BONUS
        return 0.7 * sim + 0.3 * (bonus / 0.15)  # normalize bonus to 0-1 range before weighting


    # Add all nodes
    for elem in elements:
        G.add_node(elem["id"], **{
            "type": elem["type"],
            "page": elem["page_number"],
            "section": elem["section_heading"],
            "content_preview": (elem.get("content") or "")[:80],
        })

    # Index elements by page for proximity linking
    by_page: Dict[int, List[Dict]] = {}
    for elem in elements:
        p = elem["page_number"]
        by_page.setdefault(p, []).append(elem)

    for page_num, page_elems in by_page.items():
        headings = [e for e in page_elems if e["type"] == "heading"]
        tables = [e for e in page_elems if e["type"] == "table"]
        images = [e for e in page_elems if e["type"] == "image"]
        paragraphs = [e for e in page_elems if e["type"] == "paragraph"]

        # Heading → owns → everything on same page with same section heading
        for heading in headings:
            for elem in page_elems:
                if elem["id"] == heading["id"]:
                    continue
                heading_text = heading["content"].strip().lower()
                section_text = elem.get("section_heading", "").strip().lower()
                
                if (
                    heading_text
                    and section_text
                    and (
                        heading_text in section_text
                        or section_text in heading_text
                    )
                ):
                    G.add_edge(heading["id"], elem["id"], relation="owns")
                    if heading["id"] not in elem["related_elements"]:
                        elem["related_elements"].append(heading["id"])
                    if elem["id"] not in heading["related_elements"]:
                        heading["related_elements"].append(elem["id"])

        # Paragraph → explains → Table (same page, same section)
        for para in paragraphs:
            for table in tables:
                if _hybrid_score(para, table) > SIM_THRESHOLD:
                    G.add_edge(para["id"], table["id"], relation="explains")
                    if table["id"] not in para["related_elements"]:
                        para["related_elements"].append(table["id"])
                    if para["id"] not in table["related_elements"]:
                        table["related_elements"].append(para["id"])

        # Paragraph → references → Image (same page, same section)
        for para in paragraphs:
            for image in images:
                if _hybrid_score(para, image) > SIM_THRESHOLD:
                    G.add_edge(para["id"], image["id"], relation="references")
                    if image["id"] not in para["related_elements"]:
                        para["related_elements"].append(image["id"])
                    if para["id"] not in image["related_elements"]:
                        image["related_elements"].append(para["id"])

        # Caption → belongs_to → Image
        # Heuristic: a short paragraph (<= 15 words) near an image is likely a caption
        for para in paragraphs:
            word_count = len(para["content"].split())
            if word_count <= 15 and ("figure" in para["content"].lower() or
                                      "fig." in para["content"].lower() or
                                      "chart" in para["content"].lower() or
                                      "table" in para["content"].lower()):
                for image in images:
                    G.add_edge(para["id"], image["id"], relation="caption_of")
                    if image["id"] not in para["related_elements"]:
                        para["related_elements"].append(image["id"])

    # Cross-page: link same-section tables and images across adjacent pages
    all_sections = set(e.get("section_heading", "") for e in elements)
    for section in all_sections:
        section_elems = [e for e in elements if e.get("section_heading") == section]
        tables_in_section = [e for e in section_elems if e["type"] == "table"]
        images_in_section = [e for e in section_elems if e["type"] == "image"]

        # Link tables in a section to each other — only if section is small,
        # to avoid blanket-connecting every table under a broad heading
        MAX_SECTION_FANOUT = 4
        if len(tables_in_section) <= MAX_SECTION_FANOUT:
            for i, t1 in enumerate(tables_in_section):
                for t2 in tables_in_section[i+1:]:
                    G.add_edge(t1["id"], t2["id"], relation="related_table")
                    if t2["id"] not in t1["related_elements"]:
                        t1["related_elements"].append(t2["id"])

        # Link tables ↔ images in same section — only if section is small
        if len(tables_in_section) <= MAX_SECTION_FANOUT and len(images_in_section) <= MAX_SECTION_FANOUT:
            for table in tables_in_section:
                for image in images_in_section:
                    G.add_edge(table["id"], image["id"], relation="visualizes")
                    if image["id"] not in table["related_elements"]:
                        table["related_elements"].append(image["id"])

        # removed blanket paragraph -> every table/image in section linking.
        # Same-page linking (done earlier in this function, per-page) already
        # covers genuine proximity-based paragraph<->table/image relationships.
        # Cross-page section-wide linking was producing excessive, low-precision
        # related_elements lists once section_heading stopped being a useless
        # constant (see extractor.py fix).

    # ── Typed/directed edge list per element ───────────────────────────────
    # related_elements (above) is a bare list of IDs with no relation type or
    # direction, so it can't reconstruct edge labels later (e.g. for the UI
    # graph view or for relationships_used in explainability). We mirror the
    # same information here as a structured list, attached to each element so
    # it survives into the Qdrant payload without needing the graph object
    # itself to be persisted.
    elem_by_id = {e["id"]: e for e in elements}
    for elem in elements:
        elem["related_edges"] = []

    for src, dst, data in G.edges(data=True):
        relation = data.get("relation", "unknown")
        if src in elem_by_id:
            elem_by_id[src]["related_edges"].append({
                "to": dst,
                "relation": relation,
                "direction": "out",
            })
        if dst in elem_by_id:
            elem_by_id[dst]["related_edges"].append({
                "to": src,
                "relation": relation,
                "direction": "in",
            })

    return G


def get_related_elements(
    element_id: str,
    elements: List[Dict[str, Any]],
    graph: nx.DiGraph,
    depth: int = 1,
) -> List[Dict[str, Any]]:
    """
    Given an element ID, returns all directly related elements (neighbors in graph).
    depth=1 means immediate neighbors only.
    """
    elem_by_id = {e["id"]: e for e in elements}
    related_ids = set()

    if depth == 1:
        related_ids.update(graph.successors(element_id))
        related_ids.update(graph.predecessors(element_id))
    else:
        for node in nx.ego_graph(graph, element_id, radius=depth).nodes:
            if node != element_id:
                related_ids.add(node)

    return [elem_by_id[rid] for rid in related_ids if rid in elem_by_id]


def summarize_graph(graph: nx.DiGraph) -> Dict[str, Any]:
    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "relation_types": list(set(
            d.get("relation", "unknown")
            for _, _, d in graph.edges(data=True)
        )),
    }


def rebuild_graph_from_elements(elements: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Reconstructs a NetworkX graph from each element's persisted `related_edges`
    field (Qdrant payload), without needing the original in-memory graph object
    from ingestion. Used post-ingestion, e.g. by the UI's graph view and by
    explainability, where only Qdrant-fetched elements are available.

    Elements missing `related_edges` (e.g. ingested before this field existed)
    contribute no edges but still appear as isolated nodes.
    """
    G = nx.DiGraph()
    elem_by_id = {e.get("element_id") or e.get("id"): e for e in elements}

    for elem in elements:
        eid = elem.get("element_id") or elem.get("id")
        if not eid:
            continue
        G.add_node(eid, **{
            "type": elem.get("type", "unknown"),
            "page": elem.get("page_number"),
            "section": elem.get("section_heading", ""),
            "content_preview": (elem.get("content") or elem.get("vision_summary") or "")[:80],
        })

    for elem in elements:
        eid = elem.get("element_id") or elem.get("id")
        for edge in elem.get("related_edges", []) or []:
            other = edge.get("to")
            relation = edge.get("relation", "related")
            direction = edge.get("direction", "out")
            if other not in elem_by_id:
                # neighbor wasn't fetched/in this element set — skip rather
                # than draw an edge to a node we can't render
                continue
            if direction == "out":
                src, dst = eid, other
            else:
                src, dst = other, eid
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, relation=relation)

    return G