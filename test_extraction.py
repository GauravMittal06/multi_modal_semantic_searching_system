from src.extractor import extract_document

pdf_path = "tsla-20241231-gen.pdf"

elements = extract_document(pdf_path, source_name="tsla-20241231-gen")

for e in elements:
    content = e.get("content", "")

    if "Principal Accounting Officer" in content:
        print("\nFOUND:")
        print("PAGE:", e["page_number"])
        print(content)