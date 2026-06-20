# src/ingest.py
"""
Full Ingestion Pipeline
Orchestrates: Extract → Vision → Relationships → Embed → Upsert
"""

import os
import json
import time
from typing import List, Dict, Any

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.extractor import extract_document
from src.vision import enrich_image_elements
from src.relationships import build_relationship_graph, summarize_graph
from src.rag_core import embed_passages, upsert_elements, delete_by_source
from src.bm25_index import build_bm25_index, delete_bm25_index

def ingest_document(filepath: str, original_filename: str = None) -> Dict[str, Any]:
    source_name = original_filename if original_filename else os.path.basename(filepath)
    print(f"\n[ingest] Starting ingestion: {source_name}")

    print("[1/5] Extracting document structure...")
    elements = extract_document(filepath, source_name=source_name)
    print(f"       → {len(elements)} elements extracted")
    type_counts = {}
    for e in elements:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    print(f"       → breakdown: {type_counts}")

    print("[2/5] Running Gemini Vision on images...")
    elements = enrich_image_elements(elements)

    print("[3/5] Building relationship graph...")
    graph = build_relationship_graph(elements)
    graph_summary = summarize_graph(graph)
    print(f"       → {graph_summary['total_nodes']} nodes, {graph_summary['total_edges']} edges")

    print("[4/5] Embedding all elements...")
    texts_to_embed = []
    for elem in elements:
        if elem["type"] == "image":
            text = elem.get("vision_summary") or elem.get("content") or ""
        else:
            text = elem.get("content") or ""
        texts_to_embed.append(text if text.strip() else f"[{elem['type']} on page {elem['page_number']}]")

    embeddings = embed_passages(texts_to_embed)
    print(f"       → {len(embeddings)} embeddings generated")

    print("[5/5] Upserting to Qdrant and building BM25 index...")
    delete_by_source(source_name)
    delete_bm25_index(source_name)
    upsert_elements(elements, embeddings)
    build_bm25_index(elements, source_name)

    print(f"\n[ingest] ✅ Done: {source_name}")
    return {
        "source": source_name,
        "total_elements": len(elements),
        "type_counts": type_counts,
        "graph_nodes": graph_summary["total_nodes"],
        "graph_edges": graph_summary["total_edges"],
        "relation_types": graph_summary["relation_types"],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest <path_to_pdf_or_docx>")
        sys.exit(1)
    result = ingest_document(sys.argv[1])
    print(json.dumps(result, indent=2))