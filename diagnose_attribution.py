import fitz

doc = fitz.open("IPCC_AR6_SYR_LongerReport.pdf")
page = doc[38]  # page 39, 0-indexed

current_heading = "b) Annual mean total column soil moisture change"  # carried in from page 36 end

blocks = page.get_text("dict")["blocks"]

for idx, block in enumerate(blocks):
    if block.get("type") != 0:
        continue
    lines = block.get("lines", [])
    text = " ".join(
        s.get("text", "").strip()
        for l in lines for s in l.get("spans", [])
        if s.get("text", "").strip()
    ).strip()
    if not text:
        continue

    bbox = block.get("bbox", [0,0,0,0])
    y0 = bbox[1]

    max_font = max((s.get("size",12) for l in lines for s in l.get("spans",[])), default=12)
    is_heading_like = text.strip() in [
        "a) Risk of", "species losses",
        "b) Heat-humidity", "risks to human health",
        "c) Food production",
    ]

    if is_heading_like:
        current_heading = text

    print(f"idx={idx} | y0={y0:.1f} | heading={is_heading_like} | current_heading_at_time={current_heading!r} | text={text[:60]!r}")