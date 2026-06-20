# app.py
"""
Streamlit Frontend — Multi-Modal Document Intelligence Platform
"""

import os
import tempfile
import streamlit as st
import shutil
import tempfile
from pathlib import Path

from src.ingest import ingest_document
from src.answer import generate_answer
from src.graph_view import (
    answer_subgraph_elements,
    full_document_graph,
    render_pyvis_html,
    evidence_chain_from_subgraph,
    modality_contribution_summary,
)
import streamlit.components.v1 as components

def clear_temp_vision_artifacts():
    """
    Removes:
    - extracted_images folders
    - vision_cache folders

    from temp directories used during ingestion.
    """

    temp_root = Path(tempfile.gettempdir())

    removed_cache = 0
    removed_images = 0

    for cache_dir in temp_root.rglob("vision_cache"):
        try:
            shutil.rmtree(cache_dir)
            removed_cache += 1
        except Exception:
            pass

    for img_dir in temp_root.rglob("extracted_images"):
        try:
            shutil.rmtree(img_dir)
            removed_images += 1
        except Exception:
            pass

    print(
        f"[cleanup] removed "
        f"{removed_cache} vision_cache folders and "
        f"{removed_images} extracted_images folders"
    )

def render_explainability(explainability: dict):
    """Renders the judge-facing 'How the Answer Was Generated' panel."""
    if not explainability:
        return
    with st.expander("🔍 How the Answer Was Generated", expanded=False):
        st.markdown("**Evidence Used**")
        evidence = explainability.get("evidence_used", [])
        if evidence:
            for ev in evidence:
                st.markdown(f"- **{ev['type']}** — Page {ev['page_number']} · _{ev['section_heading']}_")
        else:
            st.caption("No supporting evidence found.")

        rels = explainability.get("relationships_used", [])
        if rels:
            st.markdown("**Relationships Used**")
            for r in rels:
                st.markdown(f"- {r}")

        st.markdown("**Modalities Used**")
        mods = explainability.get("modalities_used", {})
        mod_line = []
        for label, key in [("Text", "text"), ("Table", "table"), ("Image", "image")]:
            mod_line.append(f"{'✅' if mods.get(key) else '⬜'} {label}")
        st.markdown("&nbsp;&nbsp;".join(mod_line))

        summary = explainability.get("retrieval_summary", {})
        st.markdown("**Retrieval Summary**")
        type_counts = summary.get("type_counts", {})
        breakdown = ", ".join(f"{v} {k}{'s' if v != 1 else ''}" for k, v in type_counts.items())
        st.caption(
            f"Retrieved {summary.get('total_elements', 0)} evidence elements "
            f"across {summary.get('page_count', 0)} document page(s)"
            + (f" — {breakdown}" if breakdown else "")
        )

        st.markdown("**Confidence**")
        confidence = explainability.get("confidence", "Low")
        badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "🔴")
        st.markdown(f"{badge} **{confidence}**")


def render_evidence_chain(context_elements: list):
    """
    Renders the Evidence Chain panel (Task 4): a plain textual walkthrough
    of how evidence elements connect to each other and ultimately to the
    answer, built from the same graph data as relationships_used but shown
    as a labeled chain rather than a graph or a flat bullet list. Designed
    to be readable by a non-technical judge with zero graph-reading effort.
    """
    if not context_elements:
        return
    try:
        steps = evidence_chain_from_subgraph(context_elements)
    except Exception:
        return
    if not steps:
        return

    st.markdown("**🔗 Evidence Chain**")
    for step in steps:
        st.markdown(
            f"`{step['src_label']}` _(Page {step['src_page']})_  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;↓ {step['sentence'][0].lower() + step['sentence'][1:]}  \n"
            f"`{step['dst_label']}` _(Page {step['dst_page']})_"
        )
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;↓ supports  \n**Answer**")


def render_modality_contribution(context_elements: list):
    """
    Renders the "How Each Source Contributed" panel (Task 6): one sentence
    per modality present in the evidence used for the answer, explaining
    what that modality specifically contributed (data, context, or visual
    confirmation). Directly demonstrates multi-modal synthesis to judges.
    """
    if not context_elements:
        return
    try:
        contributions = modality_contribution_summary(context_elements)
    except Exception:
        return
    if not contributions:
        return

    st.markdown("**🧩 How Each Source Contributed**")
    icon = {"Table": "📊", "Paragraph": "📝", "Image": "🖼️"}
    for modality, sentence in contributions.items():
        st.markdown(f"{icon.get(modality, '•')} **{modality}**  \n{sentence}")


def render_relationship_graph(context_elements: list, key: str):
    """
    Renders the per-answer relationship graph: evidence elements used in the
    answer plus their direct graph neighbors, drawn with pyvis. This is the
    judge-facing view of relationship awareness — scoped to one answer so it
    stays small and readable rather than showing the whole document graph.
    """
    if not context_elements:
        return
    with st.expander("🕸️ Relationship Graph (this answer)", expanded=False):
        try:
            graph, elem_by_id = answer_subgraph_elements(context_elements)
        except Exception as e:
            st.caption(f"Graph unavailable: {e}")
            return

        if graph.number_of_nodes() == 0:
            st.caption("No relationship data available for this answer.")
            return

        evidence_ids = [
            e.get("element_id") or e.get("id")
            for e in context_elements
            if e.get("is_primary")
        ]

        st.caption(
            f"{graph.number_of_nodes()} elements · {graph.number_of_edges()} relationships shown. "
            "Larger, bordered nodes are the evidence used in the answer; smaller nodes are their "
            "direct neighbors in the document graph."
        )
        legend_cols = st.columns(4)
        for col, (t, color) in zip(legend_cols, [
            ("Heading", "#F4A261"), ("Paragraph", "#2A9D8F"),
            ("Table", "#E76F51"), ("Image", "#264653"),
        ]):
            col.markdown(
                f"<span style='color:{color}'>●</span> {t}",
                unsafe_allow_html=True,
            )

        html, info = render_pyvis_html(graph, elem_by_id, highlight_ids=evidence_ids, height="420px")
        if info["capped"]:
            st.caption(
                f"⚠️ Showing {info['shown_nodes']} of {info['total_nodes']} elements "
                "(most-connected nodes kept; evidence always shown)."
            )
        components.html(html, height=440, scrolling=False)

st.set_page_config(
    page_title="Multi-Modal Document Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ─── Session State ────────────────────────────────────────────────────────────
if "ingested_source" not in st.session_state:
    st.session_state.ingested_source = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingest_stats" not in st.session_state:
    st.session_state.ingest_stats = None
if "full_graph_html" not in st.session_state:
    st.session_state.full_graph_html = None
if "full_graph_info" not in st.session_state:
    st.session_state.full_graph_info = None

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Multi Modal Semantic Search")
    st.caption("Multi Modal Semantic Integration for Intelligent Unstructured Document Understanding")
    st.divider()

    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        help="Upload a document to analyze",
    )

    if uploaded_file:
        if st.button("🚀 Ingest Document", use_container_width=True, type="primary"):
            with st.spinner("Ingesting document... this may take a minute."):
                try:
                    # clear_temp_vision_artifacts()
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    stats = ingest_document(tmp_path, original_filename=uploaded_file.name)
                    os.unlink(tmp_path)

                    st.session_state.ingested_source = uploaded_file.name
                    st.session_state.ingest_stats = stats
                    st.session_state.chat_history = []
                    st.session_state.full_graph_html = None
                    st.session_state.full_graph_info = None
                    st.success(f"✅ Ingested: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

    if st.session_state.ingest_stats:
        st.divider()
        st.subheader("📊 Document Stats")
        stats = st.session_state.ingest_stats
        st.metric("Total Elements", stats["total_elements"])
        cols = st.columns(2)
        for i, (k, v) in enumerate(stats["type_counts"].items()):
            cols[i % 2].metric(k.capitalize(), v)
        st.divider()
        st.caption("Relationship Graph")
        graph_cols = st.columns(2)
        graph_cols[0].metric("Nodes", stats.get("graph_nodes", "—"))
        graph_cols[1].metric("Edges", stats.get("graph_edges", "—"))
        rel_types = stats.get("relation_types", [])
        if rel_types:
            st.caption("Edge types: " + ", ".join(sorted(rel_types)))

    if st.session_state.ingested_source:
        st.divider()
        st.success(f"Active: **{st.session_state.ingested_source}**")
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.ingested_source = None
            st.session_state.chat_history = []
            st.session_state.ingest_stats = None
            st.session_state.full_graph_html = None
            st.session_state.full_graph_info = None
            st.rerun()

    st.divider()

    if st.button("🧹 Clear Vision Cache", use_container_width=True):
        clear_temp_vision_artifacts()
        st.success("Vision cache cleared.")

# ─── Main Area ────────────────────────────────────────────────────────────────
st.title("🧠 Multi-Modal Document Intelligence")
st.caption("Understands paragraphs, tables, and images together — not just text chunks.")

if not st.session_state.ingested_source:
    st.info("👈 Upload and ingest a document from the sidebar to get started.")

    st.subheader("What this system does differently")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📝 Text + Tables**\nRetrieves paragraphs and their related tables together")
    with col2:
        st.markdown("**🖼️ Image Understanding**\nUses Gemini Vision to understand charts and figures semantically")
    with col3:
        st.markdown("**🔗 Relationship-Aware**\nLinks headings → sections → tables → images before retrieval")
else:
    tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Document Graph Explorer"])

    with tab_chat:
        # ─── Chat Interface ───────────────────────────────────────────────────
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📎 Sources", expanded=False):
                        for c in msg["citations"]:
                            st.markdown(
                                f"- **{c['element_type'].capitalize()}** · "
                                f"Page {c['page_number']} · "
                                f"Section: _{c['section_name']}_ · "
                                f"`{c['source_document']}`"
                            )
                if msg["role"] == "assistant":
                    render_explainability(msg.get("explainability"))
                    render_evidence_chain(msg.get("context_elements", []))
                    render_modality_contribution(msg.get("context_elements", []))
                    render_relationship_graph(
                        msg.get("context_elements", []),
                        key=f"hist_{i}",
                    )

        question = st.chat_input("Ask a question about the document...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context across paragraphs, tables, and images..."):
                    result = generate_answer(
                        question=question,
                        source_document=st.session_state.ingested_source,
                        top_k=8,
                    )

                st.markdown(result["answer"])

                if result["citations"]:
                    with st.expander("📎 Sources", expanded=True):
                        for c in result["citations"]:
                            st.markdown(
                                f"- **{c['element_type'].capitalize()}** · "
                                f"Page {c['page_number']} · "
                                f"Section: _{c['section_name']}_ · "
                                f"`{c['source_document']}`"
                            )

                if result.get("error"):
                    st.warning(f"Warning: {result['error']}")

                render_explainability(result.get("explainability"))
                render_evidence_chain(result.get("context_used", []))
                render_modality_contribution(result.get("context_used", []))
                render_relationship_graph(result.get("context_used", []), key="live")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
                "explainability": result.get("explainability"),
                "context_elements": result.get("context_used", []),
            })

        # ─── Debug Panel ──────────────────────────────────────────────────────
        with st.expander("🔍 Debug: Retrieved Context", expanded=False):
            if st.session_state.chat_history:
                last_q = next(
                    (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "user"),
                    None,
                )
                if last_q and st.button("Show context for last question"):
                    from src.retrieval import retrieve_context, format_context_for_llm
                    ctx = retrieve_context(last_q, source_document=st.session_state.ingested_source)
                    st.text(format_context_for_llm(ctx))
            else:
                st.caption("Ask a question first.")

    with tab_graph:
        st.subheader("🕸️ Full Document Relationship Graph")
        st.caption(
            "Every element and relationship extracted from this document — headings, "
            "paragraphs, tables, and images, linked the way they actually relate in the source."
        )
        legend_cols = st.columns(4)
        for col, (t, color) in zip(legend_cols, [
            ("Heading", "#F4A261"), ("Paragraph", "#2A9D8F"),
            ("Table", "#E76F51"), ("Image", "#264653"),
        ]):
            col.markdown(f"<span style='color:{color}'>●</span> {t}", unsafe_allow_html=True)

        if st.button("🔄 Load / Refresh Full Graph", use_container_width=True):
            with st.spinner("Fetching all document elements and rebuilding graph..."):
                try:
                    graph, elem_by_id = full_document_graph(st.session_state.ingested_source)
                    html, info = render_pyvis_html(
                        graph, elem_by_id, height="650px", physics=True, max_nodes=300,
                    )
                    st.session_state.full_graph_html = html
                    st.session_state.full_graph_info = info
                except Exception as e:
                    st.error(f"Failed to load full document graph: {e}")
                    st.session_state.full_graph_html = None
                    st.session_state.full_graph_info = None

        if st.session_state.get("full_graph_html"):
            info = st.session_state.get("full_graph_info") or {}
            st.caption(
                f"{info.get('shown_nodes', '—')} of {info.get('total_nodes', '—')} elements shown · "
                f"{info.get('shown_edges', '—')} of {info.get('total_edges', '—')} relationships shown"
            )
            if info.get("capped"):
                st.warning(
                    f"This document has {info.get('total_nodes')} elements — too many to render "
                    f"all at once and stay readable. Showing the {info.get('shown_nodes')} "
                    "most-connected elements with a static layout (physics disabled for "
                    "performance). For a full per-answer view of any specific evidence, ask a "
                    "question in the Chat tab instead."
                )
            components.html(st.session_state.full_graph_html, height=670, scrolling=False)
        else:
            st.info("Click **Load / Refresh Full Graph** to render the complete document relationship graph. This pulls every element from Qdrant, so it may take a few seconds on large documents.")