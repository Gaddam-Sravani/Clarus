import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVED_PAGES_DIR = os.path.join(DATA_DIR, "saved_pages")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "campus_index.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")

HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", 8080)))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Retrieval Settings: Increased for complete rule & fee coverage
TOP_K_CHUNKS = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
