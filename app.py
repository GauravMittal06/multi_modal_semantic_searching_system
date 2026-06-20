# app.py
"""
Streamlit Frontend — Multi-Modal Document Intelligence Platform

Frontend/UX only. No retrieval, ingestion, graph-building, reranking,
explainability, citation, or answer-generation logic is modified here —
all of that lives in src/ and is only ever called, never altered.
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


# ─── Cleanup Helper (unchanged backend-adjacent utility) ──────────────────────
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


# ─── Page Config + Premium Dark Theme ──────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS: dark product chrome, sticky chat input anchored to the bottom
# of the viewport (ChatGPT/Claude/Perplexity-style), card styling, and a
# fixed-height scrollable chat history container so the input never drifts
# below long answers.
st.markdown("""
<style>
    .stApp { background-color: #0B0D12; }

    /* Make the chat input visually anchored at the bottom of the page,
       not just the bottom of the script output. Streamlit already pins
       st.chat_input to the bottom of the viewport on tall pages; this
       reinforces that with a solid dark backdrop so it doesn't look like
       it's floating disconnected over content as the page scrolls. */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0B0D12;
        border-top: 1px solid #232733;
        padding: 14px 48px 18px 48px;
        z-index: 999;
        box-shadow: 0 -8px 24px rgba(0,0,0,0.45);
    }
    [data-testid="stChatInput"] > div {
        max-width: 100%;
    }
    .main .block-container {
        padding-bottom: 120px;
    }

    /* stSidebar and the main view div are siblings of the app root.
       :has() lets the chat input react to the sidebar's expanded/
       collapsed state purely in CSS, with no JS/polling required. */
    div[data-testid="stAppViewContainer"]:has(section[data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stChatInput"] {
        left: 21rem;
    }
    div[data-testid="stAppViewContainer"]:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stChatInput"] {
        left: 0rem;
    }
</style>
<style>

    .trust-summary {
        display: flex;
        gap: 14px;
        align-items: center;
        padding: 10px 14px;
        margin: 10px 0 8px 0;
        background: #14171F;
        border: 1px solid #232733;
        border-radius: 10px;
        font-size: 14px;
    }
    .action-row {
        display: flex;
        gap: 8px;
        margin: 4px 0 14px 0;
    }
    .action-btn button, .action-btn-active button {
        background: #14171F !important;
        border: 1.5px solid #E8B339 !important;
        color: #E8B339 !important;
        font-weight: 500 !important;
    }
    .action-btn button p, .action-btn-active button p {
        color: #E8B339 !important;
    }
    .action-btn-active button {
        border-color: #E8B339 !important;
        color: #E8B339 !important;
        background: #2A2410 !important;
    }
    .action-btn-active button p {
        color: #E8B339 !important;
    }
    .modality-card {
        text-align: center;
        padding: 16px 10px;
        border-radius: 12px;
        border: 1px solid #232733;
        transition: border-color 0.15s ease;
    }
    .modality-card-on {
        background: linear-gradient(160deg, #14241C 0%, #0F1A14 100%);
        border-color: #2E7D5B;
    }
    .modality-card-off {
        background: #14171F;
        opacity: 0.45;
    }
    .modality-icon { font-size: 26px; }
    .modality-title { font-weight: 600; font-size: 13px; margin-top: 6px; color: #F0F0F0; }
    .modality-sub { font-size: 11px; color: #9CA3AF; margin-top: 2px; }

    .reasoning-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        margin: 4px 0;
        background: #14171F;
        border: 1px solid #232733;
        border-left: 3px solid #E8B339;
        border-radius: 8px;
        font-size: 13px;
    }
    .reasoning-arrow {
        text-align: center;
        color: #E8B339;
        font-size: 14px;
        margin: -2px 0;
    }
    .reasoning-final {
        padding: 10px 14px;
        margin: 4px 0;
        background: linear-gradient(135deg, #2A2410 0%, #1B1810 100%);
        border: 1px solid #E8B339;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #F0D898;
    }

    .sidebar-success-card {
        background: linear-gradient(160deg, #14241C 0%, #0F1A14 100%);
        border: 1px solid #2E7D5B;
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 8px;
    }
    .sidebar-check-line { font-size: 13px; margin: 4px 0; color: #D1D5DB; }
</style>
""", unsafe_allow_html=True)


# ─── Judge-Friendly Render Helpers ─────────────────────────────────────────────

def render_trust_summary(explainability: dict, citations: list):
    """
    Always-visible one-line trust summary directly under the answer:
    confidence + source count + modalities used. Surfaces the core judging
    criteria (explainability, multi-modal evidence) in a single glance,
    with zero clicks required.
    """
    if not explainability and not citations:
        return
    confidence = (explainability or {}).get("confidence", "Low")
    badge = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(confidence, "🔴")
    n_sources = len(citations or [])
    mods = (explainability or {}).get("modalities_used", {})
    mod_labels = [label for label, key in [("Text", "text"), ("Table", "table"), ("Image", "image")] if mods.get(key)]
    mod_str = " + ".join(mod_labels) if mod_labels else "—"
    st.markdown(
        f"<div class='trust-summary'>"
        f"<span>{badge} <b>{confidence} confidence</b></span>"
        f"<span style='color:#3A3F4B'>|</span>"
        f"<span>📎 {n_sources} source{'s' if n_sources != 1 else ''}</span>"
        f"<span style='color:#3A3F4B'>|</span>"
        f"<span>🧩 {mod_str}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_modality_cards(explainability: dict, context_elements: list):
    """
    Replaces the old text checklist (Text / Table / Image with ✓/⬜) with
    three premium product-feature cards. Active modalities light up with a
    green accent; inactive ones recede — communicates multi-modal ingestion
    as a capability, not debugging metadata.
    """
    mods = (explainability or {}).get("modalities_used", {})
    cards = [
        ("📝", "Text Understanding", "Context extracted from paragraphs", "text"),
        ("📊", "Table Understanding", "Structured data analyzed", "table"),
        ("🖼️", "Visual Understanding", "Charts and images interpreted", "image"),
    ]
    cols = st.columns(3)
    for col, (icon, title, sub, key) in zip(cols, cards):
        present = mods.get(key)
        css_class = "modality-card-on" if present else "modality-card-off"
        col.markdown(
            f"<div class='modality-card {css_class}'>"
            f"<div class='modality-icon'>{icon}</div>"
            f"<div class='modality-title'>{title}</div>"
            f"<div class='modality-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    contributions = {}
    try:
        contributions = modality_contribution_summary(context_elements) or {}
    except Exception:
        pass
    if contributions:
        st.write("")
        icon_map = {"Table": "📊", "Paragraph": "📝", "Image": "🖼️"}
        for modality, sentence in contributions.items():
            st.caption(f"{icon_map.get(modality, '•')} **{modality}** — {sentence}")


def render_reasoning_path(context_elements: list):
    """
    Replaces the prose "The system connected: A → B → C" line with a
    vertical visual reasoning path — one card per evidence element, joined
    by arrows, ending in a highlighted 'Answer Generated' card. This is the
    judge-friendly visual form of the same evidence-chain data; no
    underlying logic changes, only presentation.
    """
    if not context_elements:
        st.caption("No relationship chain available for this answer.")
        return
    try:
        steps = evidence_chain_from_subgraph(context_elements)
    except Exception:
        steps = []
    if not steps:
        st.caption("No relationship chain available for this answer.")
        return

    type_icon = {"Heading": "🏷️", "Paragraph": "📝", "Table": "📊", "Image": "🖼️"}

    first = steps[0]
    icon = type_icon.get(first["src_type"], "•")
    st.markdown(
        f"<div class='reasoning-step'>{icon} <b>{first['src_label']}</b> "
        f"<span style='color:#9CA3AF'>· {first['src_type']}, Page {first['src_page']}</span></div>",
        unsafe_allow_html=True,
    )
    for step in steps:
        verb = step["sentence"][0].lower() + step["sentence"][1:]
        dst_icon = type_icon.get(step["dst_type"], "•")
        st.markdown(f"<div class='reasoning-arrow'>↓ {verb}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='reasoning-step'>{dst_icon} <b>{step['dst_label']}</b> "
            f"<span style='color:#9CA3AF'>· {step['dst_type']}, Page {step['dst_page']}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='reasoning-arrow'>↓ supports</div>", unsafe_allow_html=True)
    st.markdown("<div class='reasoning-final'>✅ Answer Generated</div>", unsafe_allow_html=True)


def render_sources(citations: list, explainability: dict):
    """Flat, single-click sources list — no nested expander."""
    evidence = (explainability or {}).get("evidence_used", [])
    if citations:
        for c in citations:
            st.markdown(
                f"- **{c['element_type'].capitalize()}** · Page {c['page_number']} · "
                f"_{c['section_name']}_ · `{c['source_document']}`"
            )
    elif evidence:
        for ev in evidence:
            st.markdown(f"- **{ev['type']}** — Page {ev['page_number']} · _{ev['section_heading']}_")
    else:
        st.caption("No supporting sources found.")


def render_relationship_graph(context_elements: list):
    """
    Renders the per-answer relationship graph directly (no expander wrapper
    — the caller already gated this behind a single button click). Evidence
    elements used in the answer plus their direct graph neighbors, drawn
    with the dark-theme pyvis renderer.
    """
    if not context_elements:
        st.caption("No relationship data available for this answer.")
        return
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

    legend_cols = st.columns(4)
    for col, (t, color) in zip(legend_cols, [
        ("Heading", "#F4A261"), ("Paragraph", "#2A9D8F"),
        ("Table", "#E76F51"), ("Image", "#264653"),
    ]):
        col.markdown(f"<span style='color:{color}'>●</span> {t}", unsafe_allow_html=True)

    html, info = render_pyvis_html(
        graph, elem_by_id, highlight_ids=evidence_ids, height="520px", max_nodes=18,
    )
    components.html(html, height=560, scrolling=False)
    st.caption(
        f"{graph.number_of_nodes()} elements · {graph.number_of_edges()} relationships shown · "
        "gold path = evidence used in this answer."
    )


def render_answer_block(result: dict, key: str):
    """
    Single entry point for everything shown under one answer, in the
    judge-prioritized order: Answer → Trust Summary → (one-click) Graph,
    Reasoning, Sources. Graph is intentionally surfaced before the textual
    reasoning path since a visual explanation lands faster than a written
    one, per the demo-flow requirement.
    """
    explainability = result.get("explainability")
    citations = result.get("citations") or []
    context_elements = result.get("context_elements") or result.get("context_used") or []

    st.markdown(result.get("answer", ""))

    if result.get("error"):
        st.warning(f"Warning: {result['error']}")

    if not explainability and not citations:
        return

    render_trust_summary(explainability, citations)

    # ── One-click action row: each view is a single toggle, no nesting ──
    state_key = f"view_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    btn_cols = st.columns(3)
    labels = [
        ("🕸️ View Relationship Graph", "graph"),
        ("🔗 View Reasoning", "reasoning"),
        ("📎 View Sources", "sources"),
    ]
    for col, (label, view) in zip(btn_cols, labels):
        active = st.session_state[state_key] == view
        col.markdown(f"<div class='{'action-btn-active' if active else 'action-btn'}'>", unsafe_allow_html=True)
        if col.button(label, key=f"{key}_{view}", use_container_width=True):
            st.session_state[state_key] = None if active else view
        col.markdown("</div>", unsafe_allow_html=True)

    active_view = st.session_state[state_key]
    if active_view == "graph":
        with st.container(border=True):
            st.markdown("**🕸️ Explore How the Document Connects**")
            render_relationship_graph(context_elements)
    elif active_view == "reasoning":
        with st.container(border=True):
            st.markdown("**🔗 Relationship Reasoning**")
            render_reasoning_path(context_elements)
    elif active_view == "sources":
        with st.container(border=True):
            st.markdown("**📎 Sources Used**")
            render_sources(citations, explainability)
            st.write("")
            st.markdown("**🧩 Multi-Modal Evidence**")
            render_modality_cards(explainability, context_elements)


# ─── Session State ──────────────────────────────────────────────────────────────
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
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Document Intelligence")
    st.caption("Understands paragraphs, tables, and images together.")
    st.divider()

    st.markdown("**Step 1 — Upload**")
    uploaded_file = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx"],
        help="Upload a document to analyze",
        label_visibility="collapsed",
    )

    st.markdown("**Step 2 — Process**")
    if uploaded_file:
        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            with st.spinner("Reading text, tables, and images..."):
                try:
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
                    st.success(f"✅ Ready: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
    else:
        st.caption("Upload a document above to enable processing.")

    st.markdown("**Step 3 — Ask**")
    if st.session_state.ingested_source:
        stats = st.session_state.ingest_stats or {}
        n_elements = stats.get("total_elements", "—")
        n_edges = stats.get("graph_edges", "—")
        type_counts = stats.get("type_counts", {})
        n_modalities = sum(1 for k in ("text", "paragraph", "table", "image") if type_counts.get(k))

        st.markdown(
            f"<div class='sidebar-success-card'>"
            f"<div style='font-weight:600;margin-bottom:6px;'>📄 {st.session_state.ingested_source}</div>"
            f"<div class='sidebar-check-line'>✓ {n_elements} Elements Indexed</div>"
            f"<div class='sidebar-check-line'>✓ {n_edges} Relationships Mapped</div>"
            f"<div class='sidebar-check-line'>✓ {n_modalities} Modalities Processed</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption("Ask anything about this document in the Chat tab.")

        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.ingested_source = None
            st.session_state.chat_history = []
            st.session_state.ingest_stats = None
            st.session_state.full_graph_html = None
            st.session_state.full_graph_info = None
            st.rerun()
    else:
        st.caption("Process a document to start asking questions.")

    st.divider()
    st.session_state.dev_mode = st.toggle("🛠️ Developer Mode", value=st.session_state.dev_mode)

    if st.session_state.dev_mode:
        st.caption("Internal diagnostics — not shown to end users.")
        if st.session_state.ingest_stats:
            stats = st.session_state.ingest_stats
            st.metric("Total Elements", stats["total_elements"])
            cols = st.columns(2)
            for i, (k, v) in enumerate(stats["type_counts"].items()):
                cols[i % 2].metric(k.capitalize(), v)
            st.caption("Relationship Graph")
            graph_cols = st.columns(2)
            graph_cols[0].metric("Nodes", stats.get("graph_nodes", "—"))
            graph_cols[1].metric("Edges", stats.get("graph_edges", "—"))
            rel_types = stats.get("relation_types", [])
            if rel_types:
                st.caption("Edge types: " + ", ".join(sorted(rel_types)))
        if st.button("🧹 Clear Vision Cache", use_container_width=True):
            clear_temp_vision_artifacts()
            st.success("Vision cache cleared.")


# ─── Main Area ────────────────────────────────────────────────────────────────
st.title("🧠 Multi-Modal Document Intelligence")
st.caption("Understands paragraphs, tables, and images together — not just text chunks.")

if not st.session_state.ingested_source:
    st.info("👈 Get started in 3 steps from the sidebar.")

    step_cols = st.columns(3)
    with step_cols[0]:
        st.markdown("### 1️⃣ Upload")
        st.caption("PDF or DOCX — text, tables, and images included.")
    with step_cols[1]:
        st.markdown("### 2️⃣ Process")
        st.caption("The system reads and links every element in the document.")
    with step_cols[2]:
        st.markdown("### 3️⃣ Ask")
        st.caption("Ask questions and get explainable, multi-modal answers.")

    st.divider()
    st.subheader("What this system does differently")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📝 Text + Tables**\nRetrieves paragraphs and their related tables together")
    with col2:
        st.markdown("**🖼️ Image Understanding**\nUses Gemini Vision to understand charts and figures semantically")
    with col3:
        st.markdown("**🔗 Relationship-Aware**\nLinks headings → sections → tables → images before retrieval")

else:
    tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Explore Document"])

    with tab_chat:
        # ─── Scrollable chat history, input anchored below via sticky CSS ──
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.chat_history):
                with st.chat_message(msg["role"]):
                    if msg["role"] == "user":
                        st.markdown(msg["content"])
                    else:
                        render_answer_block(
                            {
                                "answer": msg["content"],
                                "citations": msg.get("citations"),
                                "explainability": msg.get("explainability"),
                                "context_elements": msg.get("context_elements", []),
                            },
                            key=f"hist_{i}",
                        )

        question = st.chat_input("Ask a question about the document...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Reading across paragraphs, tables, and images..."):
                    result = generate_answer(
                        question=question,
                        source_document=st.session_state.ingested_source,
                        top_k=8,
                    )
                render_answer_block(
                    {
                        "answer": result["answer"],
                        "citations": result.get("citations"),
                        "explainability": result.get("explainability"),
                        "context_elements": result.get("context_used", []),
                        "error": result.get("error"),
                    },
                    key="live",
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
                "explainability": result.get("explainability"),
                "context_elements": result.get("context_used", []),
            })

        # ─── Debug Panel (Developer Mode only) ─────────────────────────────────
        if st.session_state.dev_mode:
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
        st.subheader("How This Document Is Understood")
        st.markdown(
            "This graph represents how the system understands relationships between "
            "sections, tables, images, and paragraphs — the same connections it uses "
            "to answer your questions."
        )
        st.caption(
            "Headings own sections, paragraphs explain tables, captions belong to images — "
            "all linked the way they actually relate in the source document."
        )

        legend_cols = st.columns(4)
        for col, (t, color) in zip(legend_cols, [
            ("Heading", "#F4A261"), ("Paragraph", "#2A9D8F"),
            ("Table", "#E76F51"), ("Image", "#264653"),
        ]):
            col.markdown(f"<span style='color:{color}'>●</span> {t}", unsafe_allow_html=True)

        if st.button("🔄 Load Document Graph", use_container_width=True):
            with st.spinner("Mapping how this document's elements connect..."):
                try:
                    graph, elem_by_id = full_document_graph(st.session_state.ingested_source)
                    html, info = render_pyvis_html(
                        graph, elem_by_id, height="650px", physics=True, max_nodes=150,
                    )
                    st.session_state.full_graph_html = html
                    st.session_state.full_graph_info = info
                except Exception as e:
                    st.error(f"Failed to load document graph: {e}")
                    st.session_state.full_graph_html = None
                    st.session_state.full_graph_info = None

        if st.session_state.get("full_graph_html"):
            info = st.session_state.get("full_graph_info") or {}
            if st.session_state.dev_mode:
                st.caption(
                    f"{info.get('shown_nodes', '—')} of {info.get('total_nodes', '—')} elements shown · "
                    f"{info.get('shown_edges', '—')} of {info.get('total_edges', '—')} relationships shown"
                )
            if info.get("capped"):
                st.caption(
                    "Showing the most-connected elements for readability. "
                    "For a focused view tied to one answer, ask a question in the Chat tab instead."
                )
            components.html(st.session_state.full_graph_html, height=670, scrolling=False)
        else:
            st.info("Click **Load Document Graph** to see how this document's elements connect.")