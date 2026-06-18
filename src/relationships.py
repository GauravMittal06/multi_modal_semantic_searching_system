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


def build_relationship_graph(elements: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Takes the flat list of elements and returns a directed graph
    where nodes are element IDs and edges are labeled relationships.
    """
    G = nx.DiGraph()

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
                if elem.get("section_heading") == heading["content"]:
                    G.add_edge(heading["id"], elem["id"], relation="owns")
                    if heading["id"] not in elem["related_elements"]:
                        elem["related_elements"].append(heading["id"])
                    if elem["id"] not in heading["related_elements"]:
                        heading["related_elements"].append(elem["id"])

        # Paragraph → explains → Table (same page, same section)
        for para in paragraphs:
            for table in tables:
                if table.get("section_heading") == para.get("section_heading"):
                    G.add_edge(para["id"], table["id"], relation="explains")
                    if table["id"] not in para["related_elements"]:
                        para["related_elements"].append(table["id"])
                    if para["id"] not in table["related_elements"]:
                        table["related_elements"].append(para["id"])

        # Paragraph → references → Image (same page, same section)
        for para in paragraphs:
            for image in images:
                if image.get("section_heading") == para.get("section_heading"):
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

        # Link all tables in a section to each other
        for i, t1 in enumerate(tables_in_section):
            for t2 in tables_in_section[i+1:]:
                G.add_edge(t1["id"], t2["id"], relation="related_table")
                if t2["id"] not in t1["related_elements"]:
                    t1["related_elements"].append(t2["id"])

        # Link tables ↔ images in same section
        for table in tables_in_section:
            for image in images_in_section:
                G.add_edge(table["id"], image["id"], relation="visualizes")
                if image["id"] not in table["related_elements"]:
                    table["related_elements"].append(image["id"])

        # Link paragraphs in section to images in same section (cross-page)
        paragraphs_in_section = [e for e in section_elems if e["type"] == "paragraph"]
        for para in paragraphs_in_section:
            for table in tables_in_section:
                if table["id"] not in para["related_elements"]:
                    para["related_elements"].append(table["id"])
                if para["id"] not in table["related_elements"]:
                    table["related_elements"].append(para["id"])
            for image in images_in_section:
                if image["id"] not in para["related_elements"]:
                    para["related_elements"].append(image["id"])
                if para["id"] not in image["related_elements"]:
                    image["related_elements"].append(para["id"])

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