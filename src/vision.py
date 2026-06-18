# src/vision.py
"""
Semantic Image Understanding — Module 2
Uses Gemini Vision to generate summaries and keywords for extracted images.
NOT OCR — understands semantic meaning of charts, diagrams, figures.
"""

import os
import base64
import hashlib
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GEMINI_KEY, GEMINI_VISION_MODEL


def _get_image_cache_path(image_path: str) -> str:
    """Returns path to cache file for a given image, based on file hash."""
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    cache_dir = os.path.join(os.path.dirname(image_path), "vision_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{file_hash}.json")


def _load_cache(image_path: str) -> Dict[str, Any] | None:
    """Returns cached vision result if available, else None."""
    try:
        cache_path = _get_image_cache_path(image_path)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _save_cache(image_path: str, result: Dict[str, Any]):
    """Saves vision result to cache."""
    try:
        cache_path = _get_image_cache_path(image_path)
        with open(cache_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Sends an image to Gemini Vision and returns:
    - vision_summary: semantic description of what the image shows
    - keywords: list of key concepts/topics
    Results are cached by image file hash to avoid redundant API calls.
    """
    if not os.path.exists(image_path):
        return {"vision_summary": "Image file not found.", "keywords": []}

    # Check cache first
    cached = _load_cache(image_path)
    if cached:
        print(f"[vision] Cache hit: {os.path.basename(image_path)}")
        return cached

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_KEY)

        image_data = _encode_image_base64(image_path)
        mime_type = _get_mime_type(image_path)

        prompt = """Analyze this image from a business/technical document.

Provide:
1. A concise semantic summary (2-4 sentences) describing what this image shows, what data it contains, and what insight it conveys. Focus on meaning, not appearance.
2. A comma-separated list of 5-8 keywords representing the key topics/concepts in this image.

Format your response EXACTLY as:
SUMMARY: <your summary here>
KEYWORDS: <keyword1>, <keyword2>, <keyword3>, ...

Do NOT describe colors, layout, or aesthetics. Focus on data, trends, relationships, and meaning."""

        response = client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=base64.b64decode(image_data),
                    mime_type=mime_type,
                ),
                prompt,
            ],
        )

        response_text = response.text or ""
        summary = ""
        keywords = []

        for line in response_text.splitlines():
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("KEYWORDS:"):
                kw_raw = line.replace("KEYWORDS:", "").strip()
                keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]

        if not summary:
            summary = response_text.strip()

        result = {"vision_summary": summary, "keywords": keywords}
        _save_cache(image_path, result)
        return result

    except Exception as e:
        print(f"[vision] Gemini Vision error for {image_path}: {e}")
        return {
            "vision_summary": f"Image analysis unavailable: {str(e)}",
            "keywords": [],
        }
    
def enrich_image_elements(elements: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Fills in vision_summary and keywords for all image elements.
    Uses parallel Gemini Vision calls and per-image caching.
    Modifies elements in-place and returns the updated list.
    """
    image_elements = [e for e in elements if e["type"] == "image"]
    image_count = len(image_elements)

    if image_count == 0:
        print("[vision] No images to process.")
        return elements

    print(f"[vision] Analyzing {image_count} images with up to {max_workers} parallel workers...")

    def _process(elem):
        image_path = elem.get("metadata", {}).get("image_path", "")
        if not image_path:
            return elem
        result = analyze_image(image_path)
        elem["vision_summary"] = result["vision_summary"]
        elem["keywords"] = result["keywords"]
        elem["content"] = result["vision_summary"]
        return elem

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, elem): elem for elem in image_elements}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            elem = future.result()
            filename = os.path.basename(elem.get("metadata", {}).get("image_path", "unknown"))
            print(f"[vision] Completed {completed}/{image_count}: {filename}")

    print(f"[vision] All {image_count} images processed.")
    return elements