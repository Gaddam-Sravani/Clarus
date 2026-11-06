import os
import math
import requests
import re
from chunker import process_all_documents

# Read API key from environment (works locally via .env and in the cloud via Render)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"

session = requests.Session()
session.trust_env = False

def get_embedding(text: str):
    """Fetches a 768-dimensional dense semantic vector from Google Cloud (0 MB local download)."""
    if not API_KEY:
        return None
    try:
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text[:2000]}]}
        }
        res = session.post(EMBED_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if res.status_code == 200:
            return res.json()["embedding"]["values"]
    except Exception as e:
        print(f"[Embedding Error]: {e}")
    return None

def cosine_similarity(vec_a, vec_b):
    """Calculates cosine similarity between two dense vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

class SemanticCloudRetriever:
    """True semantic retrieval powered by Google's text-embedding-004."""
    
    ACRONYMS = {
        r"\bvc\b": "Vice Chancellor",
        r"\bcoe\b": "Controller of Examinations",
        r"\bao\b": "Administrative Officer",
        r"\bhod\b": "Head of Department",
        r"\bpuc\b": "Pre-University Course",
        r"\bese\b": "End Semester Examination",
        r"\bcse\b": "Computer Science and Engineering",
        r"\bece\b": "Electronics and Communication Engineering",
        r"\bme\b": "Mechanical Engineering",
        r"\bce\b": "Civil Engineering",
    }

    def __init__(self, chunks):
        self.chunks = chunks
        self.embedded_chunks = []
        self._build_index()

    def normalize_query(self, query: str) -> str:
        expanded = query
        for pattern, replacement in self.ACRONYMS.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded

    def _build_index(self):
        print(f"[Semantic Index] Generating cloud embeddings for {len(self.chunks)} chunks...")
        for i, ch in enumerate(self.chunks, 1):
            vec = get_embedding(ch["text"])
            if vec:
                ch_copy = dict(ch)
                ch_copy["vector"] = vec
                self.embedded_chunks.append(ch_copy)
            if i % 20 == 0:
                print(f"  -> Embedded {i}/{len(self.chunks)} chunks...")
        print(f"[Semantic Index] Successfully indexed {len(self.embedded_chunks)} chunks with dense vectors.\n")

    def search(self, query: str, top_k: int = 4):
        normalized_q = self.normalize_query(query)
        q_vector = get_embedding(normalized_q)

        # Fallback to basic match if API key fails
        if not q_vector or not self.embedded_chunks:
            print("[Warning] Cloud embedding failed, falling back to basic match.")
            return self.chunks[:top_k]

        scores = []
        for ch in self.embedded_chunks:
            sim = cosine_similarity(q_vector, ch["vector"])
            scores.append((sim, ch))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, ch in scores[:top_k]:
            item = dict(ch)
            item["similarity_score"] = round(score, 4)
            del item["vector"]  # Don't send heavy raw vectors forward
            results.append(item)

        return results

# Alias so server.py connects seamlessly
PurePythonRetriever = SemanticCloudRetriever
