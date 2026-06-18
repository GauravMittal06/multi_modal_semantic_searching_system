import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL") or ""
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or ""
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "multimodal_rag").strip()
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")
ENV = os.getenv("ENV", "development").strip().lower()
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "").strip()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")


def validate_env():
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL missing in .env")
    if not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_API_KEY missing in .env")
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY missing in .env")
    return True