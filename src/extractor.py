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
def _line_text(line):
    spans = line.get("spans", [])
    return " ".join(
        s.get("text", "").strip()
        for s in spans
        if s.get("text", "").strip()
    ).strip()

def extract_from_pdf(filepath: str, source_name: str) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    current_heading = "Introduction"

    # --- Text + Headings via PyMuPDF ---
    doc = fitz.open(filepath)
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        all_sizes = [span.get("size", 12) for block in blocks if block.get("type") == 0
                     for line in block.get("lines", []) for span in line.get("spans", [])
                     if span.get("text", "").strip()]
        page_median_font = sorted(all_sizes)[len(all_sizes)//2] if all_sizes else 12
        for block in blocks:
            if block.get("type") != 0:
                continue
            
            block_lines = []
            max_font_size = 0
            has_bold = False
        
            for line in block.get("lines", []):
                text = _line_text(line)
        
                if not text:
                    continue
                
                block_lines.append(text)
        
                spans = line.get("spans", [])
        
                line_font = max(
                    (s.get("size", 12) for s in spans),
                    default=12
                )
        
                max_font_size = max(max_font_size, line_font)
        
                if any(bool(s.get("flags", 0) & (2**4)) for s in spans):
                    has_bold = True
        
            block_text = " ".join(block_lines).strip()
        
            if len(block_text) < 5:
                continue
            
            words = block_text.split()
        
            digit_ratio = (
                sum(
                    1
                    for w in words
                    if w.replace("$", "")
                    .replace("%", "")
                    .replace(",", "")
                    .replace(".", "")
                    .replace("+", "")
                    .replace("-", "")
                    .isdigit()
                )
                / max(len(words), 1)
            )
        
            is_heading = (
                (max_font_size > page_median_font * 1.15
                 or (has_bold and max_font_size > page_median_font * 1.05))
                and len(block_text) < 120
                and 2 <= len(words) <= 20
                and digit_ratio < 0.15
                and not block_text.endswith((".", ":", ";", ",", "?"))
            )
        
            if (
                block_text.lower() in {"n", "o"}
                or block_text in {"}", "{", ")", "(", "]", "[", "|"}
            ):
                is_heading = False
        
            if is_heading:
                current_heading = block_text
        
                elements.append(
                    _make_element(
                        "heading",
                        block_text,
                        page_num,
                        current_heading,
                        source_name,
                        {"font_size": max_font_size},
                    )
                )
        
            else:
                if len(block_text) > 20:
                    elements.append(
                        _make_element(
                            "paragraph",
                            block_text,
                            page_num,
                            current_heading,
                            source_name,
                            {},
                        )
                    )
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
            page_rect = page.rect
            total_drawings = len(drawings)

            ZONE_COUNT = 6
            zone_height = page_rect.height / ZONE_COUNT
            zone_drawing_counts = [0] * ZONE_COUNT

            for d in drawings:
                r = d.get("rect")
                if not r:
                    continue
                r = fitz.Rect(r)
                # Exclude full-width elements — page borders, horizontal rules, table edges
                if r.width > page_rect.width * 0.80:
                    continue
                center_y = (r.y0 + r.y1) / 2
                zone_idx = min(int(center_y / zone_height), ZONE_COUNT - 1)
                zone_drawing_counts[zone_idx] += 1

            max_zone_density = max(zone_drawing_counts) if zone_drawing_counts else 0

            # ── Hybrid trigger: local clustering AND overall complexity ──────
            ZONE_DENSITY_THRESHOLD = 7
            TOTAL_DRAWINGS_THRESHOLD = 15

            has_vector_graphics = (
                max_zone_density >= ZONE_DENSITY_THRESHOLD
                and total_drawings >= TOTAL_DRAWINGS_THRESHOLD
            )

            print(
                f"[extractor] Page {page_num}: total_drawings={total_drawings} | "
                f"zones={zone_drawing_counts} | max_zone_density={max_zone_density} | "
                f"trigger={'RENDER' if has_vector_graphics else 'skip'}"
            )

            # Removed 'and not image_list' so charts are always captured
            if has_vector_graphics:
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
                        "max_zone_density": max_zone_density,
                        "total_drawings": total_drawings,
                    },
                    "vision_summary": "",
                    "keywords": [],
                }
                elements.append(elem)

        except Exception as e:
            print(f"[extractor] vector/page render error page {page_num}: {e}")

    print("\n=== HEADING DEBUG ===")

    for e in elements:
        if e["type"] == "heading":
            print(
                f"page={e['page_number']} | "
                f"heading={e['content']}"
            )

    print("=====================\n")

    doc.close()
    return elements


# ─── DOCX Extractor ───────────────────────────────────────────────────────────

def extract_from_docx(filepath: str, source_name: str) -> List[Dict[str, Any]]:
    from docx import Document
    from docx.oxml.ns import qn

    elements: List[Dict[str, Any]] = []
    doc = Document(filepath)
    current_heading = "Introduction"
    page_num = 1

    all_font_sizes = [r.font.size.pt for p in doc.paragraphs for r in p.runs if r.font.size]
    doc_median_font = sorted(all_font_sizes)[len(all_font_sizes)//2] if all_font_sizes else 11

    # crude page estimator — docx has no real page numbers, so we
    # accumulate character count and roll over every ~CHARS_PER_PAGE
    CHARS_PER_PAGE = 3000
    char_count = 0

    def _has_page_break(paragraph) -> bool:
        brs = paragraph._p.xpath('.//w:br[@w:type="page"]')
        return len(brs) > 0

    def _is_heading_like(paragraph) -> bool:
        style = paragraph.style.name or ""
        if style.startswith("Heading") or style.startswith("Title"):
            return True
        text = paragraph.text.strip()
        if not text or len(text) > 90:
            return False
        words = text.split()
        if len(text.strip()) < 8:
            return False
        if len(words) < 2:
            return False
        # fallback heuristic: bold/large-font short paragraphs act as headings
        # even when not tagged with a Word "Heading" style
        runs = paragraph.runs
        if not runs:
            return False
        bold_run_chars = sum(len(r.text) for r in runs if r.bold)
        total_chars = sum(len(r.text) for r in runs) or 1
        mostly_bold = (bold_run_chars / total_chars) > 0.6
        large_font = any((r.font.size and r.font.size.pt > doc_median_font * 1.15) for r in runs)
        is_upper_or_titled = text.isupper() or text.istitle()
        return mostly_bold and (large_font or is_upper_or_titled)

    # build a table -> preceding-paragraph-index map so tables can be
    # inserted at the correct position relative to surrounding text
    body_children = list(doc.element.body)
    table_xml_elements = {id(t._tbl): t for t in doc.tables}

    para_iter = iter(doc.paragraphs)

    for child in body_children:
        tag = child.tag.split('}')[-1]

        if tag == "p":
            para = next(para_iter, None)
            if para is None:
                continue
            text = para.text.strip()

            if _has_page_break(para):
                page_num += 1
                char_count = 0

            if not text:
                continue

            char_count += len(text)
            if char_count > CHARS_PER_PAGE:
                page_num += 1
                char_count = 0

            if _is_heading_like(para):
                current_heading = text
                elements.append(_make_element(
                    "heading", text, page_num, current_heading,
                    source_name, {"style": para.style.name or ""}
                ))
            elif len(text) > 20:
                elements.append(_make_element(
                    "paragraph", text, page_num, current_heading,
                    source_name, {}
                ))

        elif tag == "tbl":
            tbl_obj = table_xml_elements.get(id(child))
            if tbl_obj is None:
                continue
            rows = []
            for row in tbl_obj.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            table_text = "\n".join(rows)
            if table_text.strip():
                table_text_clean = table_text.replace("$", "USD ").replace("%", " pct")
                header_hint = rows[0] if rows else ""
                all_cells = []
                for row in tbl_obj.rows[1:]:
                    for cell in row.cells:
                        ct = cell.text.strip().replace("$", "USD ").replace("%", " pct")
                        if ct and len(ct) > 2:
                            all_cells.append(ct)
                semantic_tags = " | ".join(dict.fromkeys(all_cells[:12]))
                table_text_with_hint = f"[TABLE COLUMNS: {header_hint}]\n[KEY VALUES: {semantic_tags}]\n{table_text_clean}"
                elements.append(_make_element(
                    "table", table_text_with_hint, page_num, current_heading,
                    source_name, {
                        "raw_rows": len(tbl_obj.rows),
                        "raw_cols": len(tbl_obj.columns),
                        "table_header": header_hint,
                    }
                ))
                char_count += len(table_text)

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
    """
    Prefer a heading from the same page.
    If none exists, fall back to the most recent previous heading.
    """

    page_headings = [
        e["content"]
        for e in elements
        if e["type"] == "heading"
        and e["page_number"] == page_num
    ]

    if page_headings:
        return page_headings[-1]

    heading = "Introduction"

    for e in elements:
        if e["type"] == "heading" and e["page_number"] < page_num:
            heading = e["content"]

    return heading