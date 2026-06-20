# src/graph_view.py
"""
Relationship Graph Visualization — supports Priority 1 (relationship
awareness showcase) and feeds real relationships_used into explainability.

Two entry points:
  - answer_subgraph_elements(): the small, focused graph for a single answer
    (evidence used + their direct neighbors). This is what powers both the
    inline "Relationship Graph" panel under an answer and the
    relationships_used list in explainability.
  - render_pyvis_html(): turns either subgraph into embeddable HTML for
    Streamlit (st.components.v1.html).

Full-document exploration uses the same render_pyvis_html() on a graph built
from rag_core.fetch_all_elements_for_document() + relationships.rebuild_graph_from_elements().
"""

from typing import List, Dict, Any, Tuple
import os
import networkx as nx

try:
    from .relationships import rebuild_graph_from_elements, get_related_elements
    from .rag_core import fetch_elements_by_ids, fetch_all_elements_for_document
except ImportError:
    from relationships import rebuild_graph_from_elements, get_related_elements
    from rag_core import fetch_elements_by_ids, fetch_all_elements_for_document

# Visual encoding — kept simple and consistent between both graph views
TYPE_COLOR = {
    "heading": "#F4A261",
    "paragraph": "#2A9D8F",
    "table": "#E76F51",
    "image": "#264653",
}
TYPE_SHAPE = {
    "heading": "diamond",
    "paragraph": "dot",
    "table": "square",
    "image": "triangle",
}

RELATION_LABEL = {
    "owns": "owns",
    "explains": "explains",
    "references": "references",
    "caption_of": "caption of",
    "related_table": "related to",
    "visualizes": "visualizes",
}


def _node_label(elem: Dict[str, Any]) -> str:
    t = elem.get("type", "unknown").capitalize()
    page = elem.get("page_number", "?")
    preview = (elem.get("content") or elem.get("vision_summary") or "").strip()
    preview = preview.replace("\n", " ")[:60]
    if len(preview) == 60:
        preview += "…"
    return f"{t} · p{page}\n{preview}" if preview else f"{t} · p{page}"


def answer_subgraph_elements(
    context_elements: List[Dict[str, Any]],
) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
    """
    Builds the per-answer relationship graph: the elements actually used in
    the answer (context_elements) plus their direct graph neighbors, using
    real edges persisted in related_edges (not heuristics).

    Returns (graph, elem_by_id) where elem_by_id includes both the evidence
    elements and any fetched neighbor elements, so callers can render full
    node details.
    """
    elem_by_id: Dict[str, Dict[str, Any]] = {}
    for e in context_elements:
        eid = e.get("element_id") or e.get("id")
        if eid:
            elem_by_id[eid] = e

    # Collect neighbor IDs referenced by related_edges that aren't already
    # in our element set, then fetch them so the graph isn't missing nodes.
    missing_neighbor_ids = set()
    for e in context_elements:
        for edge in e.get("related_edges", []) or []:
            other = edge.get("to")
            if other and other not in elem_by_id:
                missing_neighbor_ids.add(other)

    if missing_neighbor_ids:
        neighbors = fetch_elements_by_ids(list(missing_neighbor_ids))
        for n in neighbors:
            nid = n.get("element_id")
            if nid:
                elem_by_id[nid] = n

    graph = rebuild_graph_from_elements(list(elem_by_id.values()))
    return graph, elem_by_id


def relationships_used_from_subgraph(
    context_elements: List[Dict[str, Any]],
) -> List[str]:
    """
    Human-readable relationship labels for explainability, derived from the
    real graph edges connecting the evidence elements used in an answer —
    not the same-section heuristic this replaces.
    """
    graph, elem_by_id = answer_subgraph_elements(context_elements)
    evidence_ids = {
        e.get("element_id") or e.get("id")
        for e in context_elements
        if e.get("is_primary") and e.get("type") != "heading"
    }

    labels: List[str] = []
    seen = set()

    for src, dst, data in graph.edges(data=True):
        # Only surface edges that actually connect pieces of evidence used
        # in this answer — neighbor-of-neighbor edges add noise, not insight.
        if src not in evidence_ids and dst not in evidence_ids:
            continue
        if src not in elem_by_id or dst not in elem_by_id:
            continue

        relation = data.get("relation", "related")
        verb = RELATION_LABEL.get(relation, relation)
        src_elem = elem_by_id[src]
        dst_elem = elem_by_id[dst]
        src_type = src_elem.get("type", "?").capitalize()
        dst_type = dst_elem.get("type", "?").capitalize()
        dst_page = dst_elem.get("page_number", "?")

        label = f"{src_type} {verb} {dst_type} (p.{dst_page})"
        if label not in seen:
            seen.add(label)
            labels.append(label)

    return labels[:8]


def full_document_graph(source_document: str) -> Tuple[nx.DiGraph, Dict[str, Dict[str, Any]]]:
    """
    Builds the complete relationship graph for a document, for the explorer
    tab. Pulls every element belonging to source_document from Qdrant and
    reconstructs the graph from their persisted related_edges.
    """
    elements = fetch_all_elements_for_document(source_document)
    elem_by_id = {e.get("element_id"): e for e in elements if e.get("element_id")}
    graph = rebuild_graph_from_elements(elements)
    return graph, elem_by_id


def render_pyvis_html(
    graph: nx.DiGraph,
    elem_by_id: Dict[str, Dict[str, Any]],
    highlight_ids: List[str] = None,
    height: str = "500px",
    physics: bool = True,
    max_nodes: int = 300,
) -> Tuple[str, Dict[str, int]]:
    """
    Renders a NetworkX graph as a self-contained pyvis HTML string, suitable
    for st.components.v1.html(). highlight_ids (e.g. the primary evidence
    elements) are drawn larger with a border so they stand out from neighbor
    context nodes in the per-answer view.

    Large graphs (full-document view on 500–2000+ element documents) are
    capped at max_nodes: force-directed physics layout becomes unreadable
    and slow well before 2000 nodes actually render in a browser iframe, so
    above the cap we (a) keep only the highest-degree nodes — the most
    connected, most demo-relevant elements — and (b) disable physics in
    favor of a server-computed static layout, since live force simulation
    is the actual bottleneck, not node count alone.

    Returns (html, render_info) where render_info has:
      - shown_nodes / total_nodes
      - shown_edges / total_edges
      - capped: bool, whether truncation occurred
      - physics_used: bool, whether physics simulation was enabled
    """
    from pyvis.network import Network

    highlight_ids = set(highlight_ids or [])
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    capped = total_nodes > max_nodes

    render_graph = graph
    if capped:
        # Keep highlighted (evidence) nodes unconditionally, then fill the
        # remaining budget with the highest-degree nodes — these are the
        # most-connected elements and the most informative ones to show in
        # a truncated view, rather than an arbitrary slice.
        degrees = dict(graph.degree())
        always_keep = [n for n in graph.nodes if n in highlight_ids]
        remaining_budget = max(max_nodes - len(always_keep), 0)
        candidates = sorted(
            (n for n in graph.nodes if n not in highlight_ids),
            key=lambda n: degrees.get(n, 0),
            reverse=True,
        )
        keep = set(always_keep) | set(candidates[:remaining_budget])
        render_graph = graph.subgraph(keep).copy()

    shown_nodes = render_graph.number_of_nodes()
    shown_edges = render_graph.number_of_edges()

    # Physics is the real cost, not node count — but a capped graph is also
    # the signal that this document is large, so use a static layout for it
    # regardless of the requested `physics` flag, since simulating physics
    # on a degree-filtered hairball is still slow and adds no readability.
    use_physics = physics and not capped

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1a1a1a",
    )

    precomputed_pos = None
    if not use_physics:
        # Server-side static layout — computed once with networkx instead of
        # simulated frame-by-frame in the browser. spring_layout is fine up
        # to a few hundred nodes (our post-cap ceiling).
        try:
            precomputed_pos = nx.spring_layout(render_graph, seed=42, k=None)
        except Exception:
            precomputed_pos = None
        net.toggle_physics(False)
    else:
        net.barnes_hut(spring_length=160, spring_strength=0.02, damping=0.85)

    SCALE = 600  # spring_layout coords are roughly [-1, 1]; scale up for pixel space

    for node_id in render_graph.nodes:
        elem = elem_by_id.get(node_id, {})
        etype = elem.get("type", "unknown")
        is_hl = node_id in highlight_ids
        node_kwargs = dict(
            label=_node_label(elem) if elem else node_id[:8],
            title=(elem.get("content") or elem.get("vision_summary") or "")[:400],
            color=TYPE_COLOR.get(etype, "#999999"),
            shape=TYPE_SHAPE.get(etype, "dot"),
            size=28 if is_hl else 16,
            borderWidth=3 if is_hl else 1,
            borderWidthSelected=4,
        )
        if precomputed_pos is not None and node_id in precomputed_pos:
            x, y = precomputed_pos[node_id]
            node_kwargs["x"] = float(x) * SCALE
            node_kwargs["y"] = float(y) * SCALE
            node_kwargs["physics"] = False
        net.add_node(node_id, **node_kwargs)

    for src, dst, data in render_graph.edges(data=True):
        relation = data.get("relation", "related")
        net.add_edge(src, dst, label=RELATION_LABEL.get(relation, relation), arrows="to")

    net.set_options(("""
    {
      "edges": {"font": {"size": 10, "align": "middle"}, "color": {"color": "#bbbbbb"}},
      "nodes": {"font": {"size": 12}},
      "interaction": {"hover": true, "tooltipDelay": 100}"""
      + (", \"physics\": {\"enabled\": false}" if not use_physics else "")
      + """
    }
    """))

    # generate_html()'s signature/behavior has changed across pyvis versions
    # (some write a file to disk by default). Try the in-memory call first,
    # fall back to a temp-file round trip if that's not supported.
    try:
        html = net.generate_html(notebook=False)
    except TypeError:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="r", suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name
        net.write_html(tmp_path, notebook=False)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    render_info = {
        "shown_nodes": shown_nodes,
        "total_nodes": total_nodes,
        "shown_edges": shown_edges,
        "total_edges": total_edges,
        "capped": capped,
        "physics_used": use_physics,
    }
    return html, render_info