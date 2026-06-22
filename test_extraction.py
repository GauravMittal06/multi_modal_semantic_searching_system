import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.rag_core import search_elements, embed_query
except ImportError:
    from rag_core import search_elements, embed_query

q_vec = embed_query("risk factors environmental regulations government incentives")

hits = search_elements(
    query_vector=q_vec,
    k=500,
    source_document="tsla-20241231-gen.pdf",
)

target_pages = {138, 139, 140, 141, 142, 143, 144, 145}

print(f"\nTotal hits returned: {len(hits)}")
print("\n===== ALL CHUNKS ON PAGES 138-145 =====")
for h in hits:
    if h.get("page_number") in target_pages:
        content = h.get("content") or h.get("vision_summary") or ""
        print(f"\npage={h.get('page_number')} | type={h.get('type')} | element_id={h.get('element_id')}")
        print(f"section={h.get('section_heading')!r}")
        print(f"content_length={len(content)}")
        print(f"content={content[:300]!r}")
print("===== END =====")