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
                    clear_temp_vision_artifacts()
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    stats = ingest_document(tmp_path, original_filename=uploaded_file.name)
                    os.unlink(tmp_path)

                    st.session_state.ingested_source = uploaded_file.name
                    st.session_state.ingest_stats = stats
                    st.session_state.chat_history = []
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
        st.metric("Relationship Edges", stats["graph_edges"])

    if st.session_state.ingested_source:
        st.divider()
        st.success(f"Active: **{st.session_state.ingested_source}**")
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.ingested_source = None
            st.session_state.chat_history = []
            st.session_state.ingest_stats = None
            st.rerun()

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
    # ─── Chat Interface ───────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
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

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "explainability": result.get("explainability"),
        })

    # ─── Debug Panel ──────────────────────────────────────────────────────────
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