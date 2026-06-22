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
import re
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

# Business-friendly relationship phrasing — single source of truth used by
# both the graph edges (Task 3) and the Evidence Chain panel in app.py
# (Task 4/5). Each value is a short verb phrase that reads naturally as
# "{Source} {phrase} {Target}" without requiring graph literacy.
RELATION_LABEL = {
    "owns": "provides context for",
    "explains": "explains",
    "references": "references",
    "caption_of": "captions",
    "related_table": "relates to",
    "visualizes": "visualizes",
    "cross_document": "connects across documents to",
}

RELATION_SENTENCE = {
    "owns": "Section heading provides context for this {dst_type}",
    "explains": "{src_type} explains the {dst_type}",
    "references": "{src_type} references the {dst_type}",
    "caption_of": "{src_type} captions the {dst_type}",
    "related_table": "{src_type} and {dst_type} relate to the same topic",
    "visualizes": "{src_type} and {dst_type} describe the same business metric",
    "cross_document": "{src_type} from one document connects to {dst_type} from another",
}


def _truncate_label(text: str, max_chars: int = 42) -> str:
    """Elegant truncation: cut at the last word boundary before max_chars,
    not mid-word, so labels never end on a chopped fragment."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return (cut or text[:max_chars]).rstrip(",.;:—-") + "…"


def _first_meaningful_words(content: str, n_words: int = 7) -> str:
    """Strips table-syntax artifacts and leading punctuation, then takes the
    first n_words words — used as the fallback content preview for
    paragraphs/tables/images that have no heading or caption to borrow."""
    if not content:
        return ""
    text = content.replace("\n", " ").strip()
    text = re.sub(r"\[TABLE COLUMNS:.*?\]", "", text)
    text = re.sub(r"\[KEY VALUES:.*?\]", "", text)
    text = re.sub(r"\s*\|\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" |-—:")
    words = text.split(" ")
    return " ".join(w for w in words[:n_words] if w)


def _find_caption_title(elem: Dict[str, Any], elem_by_id: Dict[str, Dict[str, Any]]) -> str:
    """For tables/images: looks for a paragraph linked via a caption_of edge
    and uses its content as the title source (e.g. 'FY2025 Guidance'),
    since tables/images have no dedicated title field in the schema. Only
    searches elem_by_id (the elements already loaded for this graph) —
    no extra fetch, so this stays cheap to call per-node."""
    for edge in elem.get("related_edges", []) or []:
        if edge.get("relation") != "caption_of":
            continue
        other_id = edge.get("to")
        other = elem_by_id.get(other_id)
        if other and other.get("type") == "paragraph":
            caption_text = (other.get("content") or "").strip()
            if caption_text:
                return caption_text
    return ""


def _node_label(elem: Dict[str, Any], elem_by_id: Dict[str, Dict[str, Any]] = None) -> str:
    """
    Builds a human-readable node label, in priority order:
      1. Heading text (for heading elements, and as a borrowed title for
         their owned children when content alone wouldn't be meaningful)
      2. Table/image title via a linked caption (caption_of edge)
      3. First meaningful 5-8 words of the element's own content
      4. Fallback to the original "{Type} · p{page}" identifier

    elem_by_id is optional (defaults to no cross-element lookup) so this
    still works standalone, but callers with a loaded graph should pass it
    to enable the caption-based title for tables/images.
    """
    elem_by_id = elem_by_id or {}
    etype = elem.get("type", "unknown")
    page = elem.get("page_number", "?")
    fallback = f"{etype.capitalize()} · p{page}"

    # Priority 1: heading text, directly for heading nodes
    if etype == "heading":
        heading_text = (elem.get("content") or "").strip()
        if heading_text:
            return _truncate_label(heading_text)
        return fallback

    # Priority 2: table/image title via linked caption
    if etype in ("table", "image"):
        caption_title = _find_caption_title(elem, elem_by_id)
        if caption_title:
            return _truncate_label(_first_meaningful_words(caption_title, n_words=8))

    # Priority 3: first meaningful words of own content (or vision_summary
    # for images, which carry no raw "content" text)
    source_text = elem.get("content") or elem.get("vision_summary") or ""
    preview = _first_meaningful_words(source_text, n_words=7)
    if preview:
        return _truncate_label(preview)

    # Priority 4: fallback to the original identifier
    return fallback


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


def _relation_sentence(relation: str, src_elem: Dict[str, Any], dst_elem: Dict[str, Any]) -> str:
    """Renders a business-friendly sentence for one relationship edge,
    using RELATION_SENTENCE templates (Task 5). Falls back to the shorter
    RELATION_LABEL phrase wrapped in a generic sentence if the relation
    type has no dedicated template."""
    src_type = src_elem.get("type", "element")
    dst_type = dst_elem.get("type", "element")
    template = RELATION_SENTENCE.get(relation)
    if template:
        sentence = template.format(src_type=src_type, dst_type=dst_type)
    else:
        phrase = RELATION_LABEL.get(relation, relation)
        sentence = f"{src_type} {phrase} {dst_type}"
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def evidence_chain_from_subgraph(
    context_elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Structured evidence-chain steps for the Evidence Chain panel (Task 4):
    each step is one relationship edge connecting two EVIDENCE elements
    used in the answer (both endpoints must be evidence — not a neighbor).
    This is intentionally stricter than relationships_used_from_subgraph(),
    which also surfaces edges touching one neighbor (e.g. heading context)
    for the explainability list. The Evidence Chain panel is meant to read
    as a single coherent answer path ("Table -> explains -> Paragraph ->
    supports -> Answer"), so pulling in a tangential neighbor edge (e.g. a
    caption paragraph that isn't itself part of the answer) would break
    that readability — so it's excluded here even though it's still a real
    edge in the graph.

    Steps are ordered to form a connected walk through the evidence (each
    step's destination feeds into the next step's source where possible),
    rather than dumped in arbitrary graph edge order, so the chain reads
    top-to-bottom as one path rather than disconnected fragments.

    Returns a list of dicts:
      {
        "src_label": str, "src_type": str, "src_page": int|str,
        "dst_label": str, "dst_type": str, "dst_page": int|str,
        "relation": str,            # raw relation type, e.g. "explains"
        "sentence": str,            # business-friendly full sentence
      }
    """
    graph, elem_by_id = answer_subgraph_elements(context_elements)
    evidence_ids = {
        e.get("element_id") or e.get("id")
        for e in context_elements
        if e.get("is_primary") and e.get("type") != "heading"
    }

    # Inject synthetic cross-document edges between evidence elements
    # from different source documents. These don't exist in the persisted
    # graph (which is per-document) but are valid for explainability since
    # both elements were retrieved and used together to answer the query.
    _evidence_elems = [elem_by_id[eid] for eid in evidence_ids if eid in elem_by_id]
    _docs_present = set(e.get("source_document", "") for e in _evidence_elems)
    if len(_docs_present) > 1:
        _by_doc: Dict[str, List[str]] = {}
        for e in _evidence_elems:
            doc = e.get("source_document", "")
            eid = e.get("element_id") or e.get("id")
            if doc and eid:
                _by_doc.setdefault(doc, []).append(eid)
        _doc_list = list(_by_doc.keys())
        for i in range(len(_doc_list) - 1):
            src_eid = _by_doc[_doc_list[i]][0]
            dst_eid = _by_doc[_doc_list[i + 1]][0]
            if not graph.has_edge(src_eid, dst_eid):
                graph.add_edge(src_eid, dst_eid, relation="cross_document")

    raw_steps: List[Dict[str, Any]] = []
    labels: List[str] = []
    seen = set()

    for src, dst, data in graph.edges(data=True):
        # Strict: both endpoints must be evidence — this is what keeps the
        # chain to a single readable answer path instead of branching into
        # neighbor context (headings, captions) that belongs in the graph
        # view / explainability list, not this panel.
        if src not in evidence_ids or dst not in evidence_ids:
            continue
        if src not in elem_by_id or dst not in elem_by_id:
            continue

        relation = data.get("relation", "related")
        src_elem = elem_by_id[src]
        dst_elem = elem_by_id[dst]

        dedup_key = (src, dst, relation)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        raw_steps.append({
            "src_id": src, "dst_id": dst,
            "src_label": _node_label(src_elem, elem_by_id),
            "src_type": src_elem.get("type", "?").capitalize(),
            "src_page": src_elem.get("page_number", "?"),
            "dst_label": _node_label(dst_elem, elem_by_id),
            "dst_type": dst_elem.get("type", "?").capitalize(),
            "dst_page": dst_elem.get("page_number", "?"),
            "relation": relation,
            "sentence": _relation_sentence(relation, src_elem, dst_elem),
        })

    if not raw_steps:
        return []

    # Order as a connected walk: start from a step whose source is never
    # someone else's destination (a natural starting point), then greedily
    # chain forward wherever the next step's source matches the current
    # destination. Any steps that don't chain in get appended at the end
    # rather than dropped, so we never silently lose a relationship.
    dst_ids = {s["dst_id"] for s in raw_steps}
    start_candidates = [s for s in raw_steps if s["src_id"] not in dst_ids]
    remaining = list(raw_steps)
    ordered: List[Dict[str, Any]] = []

    current = (start_candidates or remaining)[0]
    remaining.remove(current)
    ordered.append(current)

    while remaining:
        next_step = next((s for s in remaining if s["src_id"] == ordered[-1]["dst_id"]), None)
        if next_step is None:
            next_step = remaining[0]
        remaining.remove(next_step)
        ordered.append(next_step)

    return ordered[:8]


def relationships_used_from_subgraph(
    context_elements: List[Dict[str, Any]],
) -> List[str]:
    """
    Human-readable relationship sentences for explainability (Task 5),
    derived from the real graph edges connecting the evidence elements
    used in an answer — not the same-section heuristic this replaces.

    Returns flat business-friendly sentences (e.g. "Section heading
    provides context for this paragraph") rather than the previous
    "{Type} {verb} {Type} (p.N)" technical label format. Underlying
    relationship data/edges are unchanged — only the presentation text.

    Scope is intentionally broader than evidence_chain_from_subgraph():
    this includes any edge touching at least one evidence element (e.g.
    heading -> paragraph context, a caption on an evidence table), since
    explainability is meant to show all relationship context that
    informed the answer. evidence_chain_from_subgraph() is stricter
    (both endpoints must be evidence) because it renders as a single
    linear path and a tangential neighbor edge would break that
    readability — see its docstring.
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
        if src not in evidence_ids and dst not in evidence_ids:
            continue
        if src not in elem_by_id or dst not in elem_by_id:
            continue

        relation = data.get("relation", "related")
        src_elem = elem_by_id[src]
        dst_elem = elem_by_id[dst]
        dst_page = dst_elem.get("page_number", "?")

        sentence = _relation_sentence(relation, src_elem, dst_elem)
        label = f"{sentence} (p.{dst_page})"
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


def modality_contribution_summary(
    context_elements: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Builds the "How Each Source Contributed" summary (Task 6): one sentence
    per modality (table/paragraph/image) present among the evidence used in
    the answer, describing what that modality contributed. Uses only the
    evidence elements (is_primary), not their graph neighbors, since the
    contribution summary is about what was actually used, not what's nearby.

    Returns a dict keyed by capitalized type ("Table", "Paragraph", "Image")
    present in the evidence, each mapped to a short contribution sentence
    built from that element's own content/vision_summary. If multiple
    elements share a type, only the first (highest-ranked) one is used per
    type, to keep the summary to one line per modality as specified.
    """
    evidence = [e for e in context_elements if e.get("is_primary")]

    contributions: Dict[str, str] = {}
    for elem in evidence:
        etype = elem.get("type")
        if etype not in ("table", "paragraph", "image") or etype in contributions:
            continue

        if etype == "table":
            preview = _first_meaningful_words(elem.get("content", ""), n_words=12)
            contributions["Table"] = (
                f"Provided the data values: {preview}." if preview
                else "Provided supporting data values."
            )
        elif etype == "paragraph":
            preview = _first_meaningful_words(elem.get("content", ""), n_words=14)
            contributions["Paragraph"] = (
                f"Explained the business context: {preview}." if preview
                else "Explained the business context behind the data."
            )
        elif etype == "image":
            preview = _first_meaningful_words(elem.get("vision_summary", ""), n_words=14)
            contributions["Image"] = (
                f"Provided supporting visualization: {preview}." if preview
                else "Provided a supporting visualization of the trend."
            )

    return contributions


# Dark-theme palette for the premium graph presentation. Node/edge type
# colors (TYPE_COLOR) stay the same hues so the legend in app.py still
# matches — only the canvas, fonts, and chrome around them go dark.
DARK_BG = "#0F1117"
DARK_FONT = "#E6E6E6"
DARK_EDGE_DIM = "#3A3F4B"
ACCENT_GOLD = "#E8B339"


def _short_label(text: str, max_chars: int = 18) -> str:
    """Tighter on-canvas truncation than _truncate_label — node labels need
    to stay short enough to not overlap neighboring nodes at default zoom.
    Full text is preserved separately in the hover tooltip, so nothing is
    actually lost, just deferred to hover."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return (cut or text[:max_chars]).rstrip(",.;:—-") + "…"


def _tooltip_html(elem: Dict[str, Any], full_label: str) -> str:
    """Builds the hover tooltip shown on a node. vis-network renders the
    `title` field as plain escaped text by default (not HTML), so this
    returns a simple multi-line string — the full, untruncated label plus
    type and page — instead of HTML markup that would otherwise show up
    as literal tags in the tooltip."""
    etype = (elem.get("type") or "unknown").capitalize()
    page = elem.get("page_number", "?")
    body = (elem.get("content") or elem.get("vision_summary") or "").strip().replace("\n", " ")
    if len(body) > 220:
        body = body[:220].rsplit(" ", 1)[0] + "…"
    lines = [full_label, f"{etype} · Page {page}"]
    if body:
        lines.append(body)
    return "\n".join(lines)


def render_pyvis_html(
    graph: nx.DiGraph,
    elem_by_id: Dict[str, Dict[str, Any]],
    highlight_ids: List[str] = None,
    height: str = "560px",
    physics: bool = True,
    max_nodes: int = 300,
) -> Tuple[str, Dict[str, int]]:
    """
    Renders a NetworkX graph as a self-contained, dark-theme pyvis HTML
    string, suitable for st.components.v1.html(). highlight_ids (e.g. the
    primary evidence elements) are drawn larger with a gold border so they
    stand out from neighbor context nodes in the per-answer view.

    Visual design goals (premium product feel, not an engineering tool):
      - Dark canvas matching a dark app theme, not a white debug-tool box.
      - Generous node spacing and a stronger repulsion layout to minimize
        label overlap at default zoom.
      - Short on-canvas labels with full text deferred to a styled HTML
        hover tooltip, so density never sacrifices readability.
      - Evidence-path nodes/edges rendered in a bright gold accent so the
        "answer path" is visually obvious within seconds, with everything
        else dimmed to a supporting role.

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
        bgcolor=DARK_BG,
        font_color=DARK_FONT,
    )

    precomputed_pos = None
    if not use_physics:
        # Server-side static layout — computed once with networkx instead of
        # simulated frame-by-frame in the browser. spring_layout is fine up
        # to a few hundred nodes (our post-cap ceiling). A larger k spreads
        # nodes further apart to reduce label overlap in the static case.
        try:
            precomputed_pos = nx.spring_layout(render_graph, seed=42, k=1.4)
        except Exception:
            precomputed_pos = None
        net.toggle_physics(False)
    else:
        # Wider spring_length + stronger repulsion than the previous
        # defaults: this is the single biggest lever for reducing node and
        # label overlap in the force-directed (small, per-answer) graphs.
        net.barnes_hut(
            spring_length=260,
            spring_strength=0.012,
            damping=0.8,
            gravity=-3200,
            central_gravity=0.25,
            overlap=0.1,
        )

    SCALE = 700  # spring_layout coords are roughly [-1, 1]; scale up for pixel space

    for node_id in render_graph.nodes:
        elem = elem_by_id.get(node_id, {})
        etype = elem.get("type", "unknown")
        is_hl = node_id in highlight_ids
        base_color = TYPE_COLOR.get(etype, "#999999")
        full_label = _node_label(elem, elem_by_id) if elem else node_id[:8]
        node_kwargs = dict(
            label=_short_label(full_label),
            title=_tooltip_html(elem, full_label) if elem else full_label,
            shape=TYPE_SHAPE.get(etype, "dot"),
        )
        if is_hl:
            # Evidence nodes: larger, thicker gold border — the "answer
            # path" should be unmistakable at a glance, regardless of the
            # node's underlying type color.
            node_kwargs.update(
                color={
                    "background": base_color,
                    "border": ACCENT_GOLD,
                    "highlight": {"background": base_color, "border": ACCENT_GOLD},
                },
                size=38,
                borderWidth=4,
                borderWidthSelected=5,
                font={"size": 15, "bold": True, "color": "#FFFFFF", "strokeWidth": 3, "strokeColor": DARK_BG},
                opacity=1.0,
            )
        else:
            # Neighbor context nodes: smaller, thinner border, lower
            # opacity — visually recede behind the evidence path so the
            # gold path reads first.
            node_kwargs.update(
                color={"background": base_color, "border": "#2A2D36"},
                size=16,
                borderWidth=1,
                borderWidthSelected=3,
                opacity=0.65,
                font={"size": 11, "color": "#B8BCC4", "strokeWidth": 2, "strokeColor": DARK_BG},
            )
        if precomputed_pos is not None and node_id in precomputed_pos:
            x, y = precomputed_pos[node_id]
            node_kwargs["x"] = float(x) * SCALE
            node_kwargs["y"] = float(y) * SCALE
            node_kwargs["physics"] = False
        net.add_node(node_id, **node_kwargs)

    for src, dst, data in render_graph.edges(data=True):
        relation = data.get("relation", "related")
        # Edges where both endpoints are evidence are part of the answer
        # path — draw them heavier, solid gold. Edges touching only a
        # neighbor recede into a dim, low-opacity dark-theme gray.
        both_evidence = src in highlight_ids and dst in highlight_ids
        edge_kwargs = dict(
            label=RELATION_LABEL.get(relation, relation),
            arrows="to",
            font={
                "size": 12 if both_evidence else 9,
                "align": "top",
                "color": ACCENT_GOLD if both_evidence else "#7A7F8C",
                "strokeWidth": 3,
                "strokeColor": DARK_BG,
            },
            smooth={"type": "continuous", "roundness": 0.2},
        )
        if both_evidence:
            edge_kwargs.update(color={"color": ACCENT_GOLD, "opacity": 1.0}, width=3)
        else:
            edge_kwargs.update(color={"color": DARK_EDGE_DIM, "opacity": 0.6}, width=1)
        net.add_edge(src, dst, **edge_kwargs)

    net.set_options(("""
    {
      "edges": {
        "font": {"size": 10, "align": "top"},
        "smooth": {"type": "continuous", "roundness": 0.2},
        "selectionWidth": 2
      },
      "nodes": {
        "font": {"face": "Inter, -apple-system, sans-serif"},
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.4)", "size": 8, "x": 0, "y": 2}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true,
        "keyboard": {"enabled": false}
      },
      "physics": {
        "barnesHut": {"avoidOverlap": 0.6}
      },
      "layout": {"improvedLayout": true}"""
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

    # pyvis's generated HTML wraps the network in a plain white-bordered
    # <div id="mynetwork">...</div> regardless of bgcolor — patch in a dark
    # card wrapper (rounded corners, subtle border, no white edge bleed) so
    # the canvas reads as an intentional product surface, not a raw embed.
    html = html.replace(
        "<body>",
        f"<body style=\"margin:0;background:{DARK_BG};\">"
        f"<div style=\"border-radius:14px;overflow:hidden;border:1px solid #262A33;"
        f"box-shadow:0 4px 24px rgba(0,0,0,0.35);\">",
        1,
    ).replace("</body>", "</div></body>", 1)

    render_info = {
        "shown_nodes": shown_nodes,
        "total_nodes": total_nodes,
        "shown_edges": shown_edges,
        "total_edges": total_edges,
        "capped": capped,
        "physics_used": use_physics,
    }
    return html, render_info