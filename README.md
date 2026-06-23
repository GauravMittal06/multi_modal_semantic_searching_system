# 🧠 MultiModal Semantic Integration for Intelligent Document Understanding

- **Team:** Delluminati
- **Hackathon:** Dell FutureMinds AI Hackathon 2026 — Problem Statement 2

---

## 📌 Problem Statement

Organizations are drowning in unstructured documents — reports, research papers, financial filings — where meaning is distributed across text, tables, images, and captions. Current systems treat documents as flat bags of text, severing the relational structure that gives content its meaning.

**Our solution:** A multimodal RAG system that understands documents the way a human expert would — by reasoning across all element types and their relationships.

---

## 🚀 What We Built

A full end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system capable of:

- Parsing PDFs into paragraphs, headings, tables, and images
- Generating semantic summaries of images and figures using Gemini Vision
- Building a relationship graph linking document elements (heading → paragraph → table → image)
- Answering natural language questions with grounded, cited responses
- Supporting **multi-document QA** across up to 3 documents simultaneously (stretch goal ✅)
- Visualizing the document element relationship graph interactively (stretch goal ✅)
- Explaining exactly which elements contributed to each answer (stretch goal ✅)

---

## 🏗️ System Architecture

```
Document(s)
    │
    ▼
extractor.py       →  Paragraphs, Headings, Tables, Images
    │
    ▼
vision.py          →  Gemini Vision summaries for images/figures
    │
    ▼
relationships.py   →  NetworkX graph (proximity, heading ownership, caption linking)
    │
    ▼
ingest.py          →  Embed → Qdrant upsert → BM25 index
    │
    ▼
hybrid_search.py   →  Dense ANN + BM25 + RRF fusion + document-level relevance boost
    │
    ▼
retrieval.py       →  Cross-encoder reranking → query-scope detection → context expansion
    │
    ▼
answer.py          →  Gemini answer generation → citation building → explainability
    │
    ▼
app.py             →  Streamlit UI with graph view, reasoning chain, source attribution
```

---

## 🛠️ Tech Stack & Models

| Component | Tool / Model |
|---|---|
| **LLM & Vision** | Google Gemini (gemini-2.0-flash) |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector Store** | Qdrant (local) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Graph Modeling** | NetworkX |
| **Frontend** | Streamlit |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Table Extraction** | pdfplumber |

---

## ✅ Core Features Implemented

### 1. Multimodal Ingestion
- Extracts paragraphs, headings, tables, and images from PDFs
- Gemini Vision generates semantic descriptions of all figures and charts
- Each element is embedded and stored in Qdrant with full metadata

### 2. Relationship Modeling
- NetworkX directed graph built at ingest time
- Edges: `owns`, `explains`, `references`, `caption_of`, `related_table`, `visualizes`
- Relationship data persisted in Qdrant payload as `related_edges` for post-ingest reconstruction

### 3. Hybrid Retrieval
- Dense ANN search (Qdrant) + BM25 sparse search fused via Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for precision
- Context expansion: retrieves related elements (graph neighbors) alongside primary hits
- Query-scope detection: suppresses documents not relevant to the current query

### 4. Grounded Answer Generation
- Gemini generates answers strictly from retrieved context
- Strict rules against hallucination — explicit refusal when context is insufficient
- Citations generated pointing to exact source elements (page, type, section)

### 5. Cross-Document QA *(Stretch Goal)*
- Up to 3 documents ingested simultaneously in one Qdrant collection
- Document-level relevance boost in hybrid search
- Scaled retrieval caps per document count
- Prompt conditionally instructs cross-document attribution

### 6. Graph Visualization *(Stretch Goal)*
- Interactive relationship graph rendered in the Streamlit UI
- Nodes colored by element type (paragraph, table, image, heading)
- Edges labeled by relationship type

### 7. Explainability Mode *(Stretch Goal)*
- "View Reasoning" panel shows step-by-step evidence chain
- "View Sources" panel lists all cited elements with page numbers
- Modality coverage stats (text / table / image usage per answer)

---

## 📊 Performance

| Metric | Result |
|---|---|
| Single-document benchmark | 147/160 (91.87%) |
| Modalities handled | Text ✅ Tables ✅ Images ✅ |
| Cross-element QA | ✅ Demonstrated |
| Multi-document QA | ✅ Up to 3 documents |
| Demo stability | ✅ No crashes on test documents |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Qdrant running locally (Docker recommended)
- Google Gemini API key

### 1. Clone the repository
```bash
git clone <repo-url>
cd <project-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Create a `.env` file at the project root with your API keys. You will need your Gemini key and the credentials for the remote Qdrant database:

```env
GEMINI_API_KEY="your-gemini-api-key"
QDRANT_URL="your-qdrant-cloud-url"
QDRANT_API_KEY="your-qdrant-api-key"
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Usage
1. Upload one or more PDF/ DOCX documents using the sidebar
2. Wait for ingestion to complete (progress shown per document)
3. Type a question in the chat input
4. View the answer, citations, reasoning chain, and relationship graph

---

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend
├── config.py               # API keys and config
├── src/
│   ├── extractor.py        # PDF and DOCX parsing (text, tables, images)
│   ├── vision.py           # Gemini Vision image summarization
│   ├── relationships.py    # NetworkX relationship graph builder
│   ├── ingest.py           # Embedding + Qdrant + BM25 indexing
│   ├── hybrid_search.py    # Dense + BM25 + RRF fusion
│   ├── retrieval.py        # Reranking, caps, context expansion
│   ├── answer.py           # Gemini answer generation + citations
│   ├── rag_core.py         # Qdrant client + embedding utilities
│   ├── bm25_index.py       # BM25 index management
│   └── graph_view.py       # Graph rendering + reasoning chain
├── bm25_cache/             # Per-document BM25 index files
└── requirements.txt
```

---

## 📦 Deliverables

- ✅ Source code (zipped)
- ✅ Demo video (5–7 min with voice-over)
- ✅ PPT presentation
- ✅ Setup instructions (this README)
- ✅ System architecture documentation (this README)
- ✅ Tools and models used (this README)

---

## 📚 References

- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Edge et al. (2024) — From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- Borgeaud et al. (2022) — Improving Language Models by Retrieving from Trillions of Tokens (RETRO)

---

*Built for Dell FutureMinds AI Hackathon 2026 by Team Delluminati*
