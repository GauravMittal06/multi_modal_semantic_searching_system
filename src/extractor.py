# src/extractor.py
"""
Document Structure Extractor — Module 1
Extracts headings, paragraphs, tables, and images from PDF and DOCX files.
Outputs every element as a structured dict matching the BRD JSON schema.
"""

import os
import uuid
import json
from typing import List, Dict, Any

import fitz  # PyMuPDF


# ─── JSON Schema helpers ─────────────────────────────────────────────────────

def _make_element(
    elem_type: str,
    content: str,
    page_number: int,
    section_heading: str,
    source_document: str,
    extra: Dict = None,
) -> Dict[str, Any]:
    base = {
        "id": str(uuid.uuid4()),
        "type": elem_type,
        "content": content,
        "page_number": page_number,
        "section_heading": section_heading,
        "source_document": source_document,
        "related_elements": [],
        "metadata": extra or {},
    }
    return base


# ─── PDF Extractor ────────────────────────────────────────────────────────────

def extract_from_pdf(filepath: str, source_name: str) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    current_heading = "Introduction"

    # --- Text + Headings via PyMuPDF ---
    doc = fitz.open(filepath)
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # 0 = text
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    font_size = span.get("size", 12)
                    font_flags = span.get("flags", 0)
                    is_bold = bool(font_flags & 2**4)

                    # Heading heuristic: large font or bold + short text
                    words = text.split()
                    digit_ratio = sum(1 for w in words if w.replace("$","").replace("%","").replace(",","").replace(".","").replace("+","").replace("-","").strip().isdigit()) / max(len(words), 1)
                    is_heading = (
                        (font_size >= 16 or (is_bold and font_size >= 13))
                        and len(text) < 80
                        and len(words) >= 3
                        and digit_ratio < 0.4
                        and not any(c in text for c in ["|", "\t"])
                    )
                    if is_heading:
                        current_heading = text
                        elements.append(_make_element(
                            "heading", text, page_num, current_heading,
                            source_name, {"font_size": font_size}
                        ))
                    elif len(text) > 30:
                        elements.append(_make_element(
                            "paragraph", text, page_num, current_heading,
                            source_name, {}
                        ))

    # --- Tables via pdfplumber ---
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    # Convert table rows to markdown-style string
                    rows = []
                    for row in table:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_text = "\n".join(rows)
                    if table_text.strip():
                        # Prepend column headers as a hint so the LLM knows what the table contains
                        header_hint = " | ".join([str(c).strip() for c in table[0]]) if table else ""
                        table_text_clean = table_text.replace("$", "USD ").replace("%", " pct")
                        header_hint_clean = header_hint.replace("$", "USD ").replace("%", " pct")
                        all_cells = []
                        for row in table[1:]:
                            for cell in row:
                                if cell and len(str(cell).strip()) > 2:
                                    all_cells.append(str(cell).strip().replace("$", "USD ").replace("%", " pct"))
                        semantic_tags = " | ".join(dict.fromkeys(all_cells[:12]))
                        table_text_with_hint = f"[TABLE COLUMNS: {header_hint_clean}]\n[KEY VALUES: {semantic_tags}]\n{table_text_clean}"
                        elem = _make_element(
                            "table", table_text_with_hint, page_num,
                            _get_heading_for_page(elements, page_num),
                            source_name,
                            {
                                "raw_rows": len(table),
                                "raw_cols": len(table[0]) if table else 0,
                                "table_header": header_hint_clean,
                            }
                        )
                        elements.append(elem)
    except ImportError:
        print("[extractor] pdfplumber not installed — skipping table extraction")
    except Exception as e:
        print(f"[extractor] table extraction error: {e}")

    # --- Images via PyMuPDF (pixmap per-page for vector charts) ---
    for page_num, page in enumerate(doc, start=1):
        images_dir = os.path.join(os.path.dirname(filepath), "extracted_images")
        os.makedirs(images_dir, exist_ok=True)

        # First try embedded raster images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{source_name}_p{page_num}_img{img_index}.{image_ext}"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                elem = {
                    "id": str(uuid.uuid4()),
                    "type": "image",
                    "content": "",
                    "page_number": page_num,
                    "section_heading": _get_heading_for_page(elements, page_num),
                    "source_document": source_name,
                    "related_elements": [],
                    "metadata": {"image_path": image_path, "image_filename": image_filename, "extension": image_ext},
                    "vision_summary": "",
                    "keywords": [],
                }
                elements.append(elem)
            except Exception as e:
                print(f"[extractor] raster image error page {page_num} img {img_index}: {e}")

        # Render full page as pixmap to capture vector charts/figures
        try:
            drawings = page.get_drawings()

            # Log drawing stats for tuning visibility
            substantial_drawings = [
                d for d in drawings
                if d.get("rect") and
                fitz.Rect(d["rect"]).width > 50 and
                fitz.Rect(d["rect"]).height > 50
            ]
            print(f"[extractor] Page {page_num}: {len(drawings)} total drawings, {len(substantial_drawings)} substantial")

            # Only render if enough substantial drawings exist (filters out table borders and decorators)
            has_vector_graphics = len(substantial_drawings) > 8

            # Additional guard: skip if no single drawing is tall enough to be a chart
            if has_vector_graphics and substantial_drawings:
                max_drawing_height = max(
                    fitz.Rect(d["rect"]).height
                    for d in substantial_drawings
                    if d.get("rect")
                )
                if max_drawing_height < 80:
                    has_vector_graphics = False
                    print(f"[extractor] Page {page_num}: skipping render — no tall drawings (max height {max_drawing_height:.1f}pt)")

            if has_vector_graphics and not image_list:
                print(f"[extractor] Page {page_num}: rendering as PAGE SNAPSHOT")
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                page_image_filename = f"{source_name}_p{page_num}_fullpage.png"
                page_image_path = os.path.join(images_dir, page_image_filename)
                pix.save(page_image_path)
                elem = {
                    "id": str(uuid.uuid4()),
                    "type": "image",
                    "content": "",
                    "page_number": page_num,
                    "section_heading": _get_heading_for_page(elements, page_num),
                    "source_document": source_name,
                    "related_elements": [],
                    "metadata": {
                        "image_path": page_image_path,
                        "image_filename": page_image_filename,
                        "extension": "png",
                        "is_page_render": True,
                        "image_type": "page_snapshot",
                    },
                    "vision_summary": "",
                    "keywords": [],
                }
                elements.append(elem)
            elif not image_list:
                print(f"[extractor] Page {page_num}: skipping render — threshold not met")

        except Exception as e:
            print(f"[extractor] vector/page render error page {page_num}: {e}")

    doc.close()
    return elements


# ─── DOCX Extractor ───────────────────────────────────────────────────────────

def extract_from_docx(filepath: str, source_name: str) -> List[Dict[str, Any]]:
    from docx import Document
    from docx.oxml.ns import qn

    elements: List[Dict[str, Any]] = []
    doc = Document(filepath)
    current_heading = "Introduction"
    page_num = 1  # DOCX doesn't expose real page numbers easily

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name or ""
        if style.startswith("Heading"):
            current_heading = text
            elements.append(_make_element(
                "heading", text, page_num, current_heading,
                source_name, {"style": style}
            ))
        elif len(text) > 20:
            elements.append(_make_element(
                "paragraph", text, page_num, current_heading,
                source_name, {}
            ))

    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        table_text = "\n".join(rows)
        if table_text.strip():
            elements.append(_make_element(
                "table", table_text, page_num, current_heading,
                source_name, {}
            ))

    return elements


# ─── Universal Entry Point ────────────────────────────────────────────────────

def extract_document(filepath: str, source_name: str = None) -> List[Dict[str, Any]]:
    if not source_name:
        source_name = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf(filepath, source_name)
    elif ext in (".docx", ".doc"):
        return extract_from_docx(filepath, source_name)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx")
    
# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_heading_for_page(elements: List[Dict], page_num: int) -> str:
    """Returns the most recent heading seen up to this page."""
    heading = "Introduction"
    for e in elements:
        if e["type"] == "heading" and e["page_number"] <= page_num:
            heading = e["content"]
    return heading