# stage_b_probe.py
"""
STAGE B PROBE — run this in your actual environment (where config.py / Qdrant
credentials are available) to verify whether the Priority 2 element survived
ingestion into Qdrant.

This script makes NO changes to your data. It only reads.

Usage:
    python stage_b_probe.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

SOURCE_DOCUMENT = "STRESS TEST DOC NexaCorp_Annual_Intelligence_Report_2024.pdf"
TARGETS = ["Priority 2", "400M", "Mumbai", "Hyderabad", "60%"]

if QDRANT_API_KEY:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    client = QdrantClient(url=QDRANT_URL)

print(f"[STAGE_B_PROBE] Connecting to collection: {QDRANT_COLLECTION}")

try:
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"[STAGE_B_PROBE] Collection exists. Points count: {info.points_count}")
except Exception as e:
    print(f"[STAGE_B_PROBE] FAILED to get collection: {e}")
    sys.exit(1)

# ── Step 1: scroll ALL points for this exact source_document string ──────────
print(f"\n[STAGE_B_PROBE] Scrolling all points where source_document == {SOURCE_DOCUMENT!r}")
flt = qmodels.Filter(
    must=[qmodels.FieldCondition(
        key="source_document",
        match=qmodels.MatchValue(value=SOURCE_DOCUMENT)
    )]
)
results, _ = client.scroll(
    collection_name=QDRANT_COLLECTION,
    scroll_filter=flt,
    limit=5000,
    with_payload=True,
)
print(f"[STAGE_B_PROBE] Total points found for this source_document: {len(results)}")

if len(results) == 0:
    print("[STAGE_B_PROBE] ZERO points found under this exact source_document string.")
    print("[STAGE_B_PROBE] Checking what source_document values DO exist in the collection...")
    all_results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=5000,
        with_payload=True,
    )
    distinct_sources = sorted(set(
        (r.payload or {}).get("source_document", "<MISSING>") for r in all_results
    ))
    print(f"[STAGE_B_PROBE] Distinct source_document values in collection ({len(distinct_sources)}):")
    for s in distinct_sources:
        print(f"  - {s!r}")
    print("[STAGE_B_PROBE] ROOT CAUSE CANDIDATE: source_document string mismatch between")
    print("[STAGE_B_PROBE] ingestion time and retrieval-time filter. Compare exactly against")
    print("[STAGE_B_PROBE] the source_document passed into retrieve_context().")
    sys.exit(0)

# ── Step 2: page=11 breakdown ─────────────────────────────────────────────────
print(f"\n[STAGE_B_PROBE] All points on page_number=11 for this source:")
page11 = [r for r in results if (r.payload or {}).get("page_number") == 11]
print(f"[STAGE_B_PROBE] Count: {len(page11)}")
for r in page11:
    p = r.payload or {}
    content = p.get("content", "")
    hits = [t for t in TARGETS if t in content]
    print(f"  element_id={p.get('element_id')} type={p.get('type')} section_heading={p.get('section_heading')!r}")
    print(f"    content_preview={content[:200]!r}")
    print(f"    contains_targets={hits}")

# ── Step 3: explicit search for Priority 2 by content substring ──────────────
print(f"\n[STAGE_B_PROBE] Searching all {len(results)} points' content for 'Priority 2'...")
priority2_points = [r for r in results if "Priority 2" in (r.payload or {}).get("content", "")]
print(f"[STAGE_B_PROBE] Points containing 'Priority 2': {len(priority2_points)}")

if priority2_points:
    print("[STAGE_B_PROBE] FOUND — Priority 2 element IS present in Qdrant.")
    for r in priority2_points:
        p = r.payload or {}
        print(f"  element_id={p.get('element_id')}")
        print(f"  full_content={p.get('content')!r}")
        print(f"  section_heading={p.get('section_heading')!r}")
    print("\n[STAGE_B_PROBE] CONCLUSION: Stage B is CLEAR. Priority 2 survives ingestion.")
    print("[STAGE_B_PROBE] Proceed to Stage C using the retrieval.py instrumentation already provided.")
else:
    print("[STAGE_B_PROBE] NOT FOUND — Priority 2 element is MISSING from Qdrant despite")
    print("[STAGE_B_PROBE] other elements from the same source_document being present.")
    print("[STAGE_B_PROBE] CONCLUSION: ROOT CAUSE IS STAGE B (ingestion/upsert).")
    print("[STAGE_B_PROBE] Next: check upsert_elements() call site in ingest.py for batch")
    print("[STAGE_B_PROBE] size limits, exceptions swallowed mid-loop, or point_id collisions")
    print("[STAGE_B_PROBE] (uuid5 of element['id'] — check for duplicate IDs across elements).")