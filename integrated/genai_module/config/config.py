from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
IMAGE_DIR = STORAGE_DIR / "images"
DB_PATH = STORAGE_DIR / "app.db"

GEMINI_MODEL = "gemini-2.5-flash"
HF_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_API_URL = "https://router.huggingface.co/v1/images/generations"
