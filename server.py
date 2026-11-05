import os
import sys
import time
import json
import mimetypes
import requests
import re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from chunker import process_all_documents
from retriever import PurePythonRetriever

PORT = int(os.environ.get("PORT", 8080))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

def load_env():
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_env()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

# Persistent HTTP session
session = requests.Session()
session.trust_env = False

# 1. Initialize RAG pipeline in memory on server start
print("Loading and indexing campus documents...")
chunks = process_all_documents()
retriever = PurePythonRetriever(chunks)
print(f"Clarus RAG Engine ready with {len(chunks)} chunks!\n")

GREETINGS = {"hi", "hii", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening"}

def extract_key_sentences(chunk_text: str, query: str, max_sentences: int = 3) -> str:
    """Finds informative sentences inside the chunk matching the query."""
    query_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", query.lower()))
    
    # Split into lines
    raw_lines = [l.strip() for l in chunk_text.splitlines() if l.strip() and not l.startswith("=")]
    scored_lines = []

    for line in raw_lines:
        # Skip repetitive titles or short headers
        if len(line) < 20 or line.lower().startswith("source url"):
            continue
        line_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", line.lower()))
        overlap = len(query_words.intersection(line_words))
        if overlap > 0:
            scored_lines.append((overlap, line))

    scored_lines.sort(key=lambda x: x[0], reverse=True)

    if scored_lines:
        selected = [line for score, line in scored_lines[:max_sentences]]
        return "\n".join([f"• {s}" for s in selected])
    
    # If no specific sentence scored higher, return the clean paragraph directly
    clean_chunk = "\n".join([l for l in raw_lines if len(l) > 20][:3])
    return f"• {clean_chunk}" if clean_chunk else f"• {chunk_text[:180]}..."

def ask_gemini(question: str):
    clean_q = question.strip().lower()
    
    # 1. Instant friendly reply for basic greetings (0 API calls, 0 delay)
    if clean_q in GREETINGS:
        return (
            "Namaste! I am Clarus, your RGUKT Basar Campus AI Assistant. "
            "How can I help you today? You can ask me about academic regulations, "
            "attendance rules, campus administration, or university history.",
            []
        )

    # 2. Retrieve top-4 relevant chunks
    matches = retriever.search(question, top_k=4)
    if not matches:
        return "I could not find any relevant RGUKT Basar records matching your question. Please verify on https://hub.rgukt.ac.in", []

    context_blocks = []
    sources = []
    seen_urls = set()

    for m in matches:
        context_blocks.append(f"--- Document: {m['source_file']} ({m['source_url']}) ---\n{m['text']}")
        if m['source_url'] not in seen_urls:
            seen_urls.add(m['source_url'])
            sources.append({"title": m['source_file'].replace(".txt", "").replace("_", " ").title(), "url": m['source_url']})

    combined_context = "\n\n".join(context_blocks)

    prompt = f"""You are Clarus, the official RGUKT Basar Campus AI Assistant.
Answer the student's question accurately, concisely, and helpfully using ONLY the provided official college context.
If asked about a current post, state the person currently holding office and details.
If asked about historical leadership (e.g. first Vice Chancellor), check the succession records provided.
Use bullet points and bold formatting for clarity.

COLLEGE CONTEXT:
{combined_context}

QUESTION:
{question}

ANSWER:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 3000}
    }

    # 3. Retry loop with fast fail-over (8s timeout per attempt so it never hangs for 90s)
    MAX_RETRIES = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 8-second timeout: fast response when online, quick fallback when offline
            res = session.post(GEMINI_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
            
            if res.status_code == 200:
                data = res.json()
                ans = data["candidates"][0]["content"]["parts"][0]["text"]
                return ans, sources
            
            # If Google returns 429 (Quota full) or 503 (Server busy), serve local facts
            elif res.status_code in (429, 503):
                print(f"[Gemini {res.status_code}] Serving verified local context directly.")
                clean_facts = []
                for m in matches:
                    key_sentence = extract_key_sentences(m['text'], question)
                    source_name = m['source_file'].replace('.txt', '').replace('_', ' ').title()
                    clean_facts.append(key_sentence)
                
                fallback_text ="\n\n".join(clean_facts)
                return fallback_text, sources
            
            else:
                return f"Gemini API returned error {res.status_code}: {res.text}", sources

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print(f"[Network Slow/Offline] Attempt {attempt}/{MAX_RETRIES} timed out.")
            if attempt < MAX_RETRIES:
                time.sleep(1.5)
                continue
            else:
                # Immediately serve offline context if network fails!
                print("[Offline Fallback Triggered] Serving verified local records without waiting.")
                clean_facts = []
                for m in matches:
                    key_sentence = extract_key_sentences(m['text'], question)
                    source_name = m['source_file'].replace('.txt', '').replace('_', ' ').title()
                    clean_facts.append(key_sentence)
                
                fallback_text = "📍 **[Offline Mode — Verified Local Records]**\n\n" + "\n\n".join(clean_facts)
                return fallback_text, sources

        except Exception as e:
            return f"Error contacting AI service: {str(e)}", sources

    return "Google's AI service is momentarily overloaded (503). Please wait a few seconds and try again.", sources

class ClarusHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        # 1. Root and HTML
        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        
        # 2. Serve CSS directly
        elif path in ("/style.css", "/static/style.css"):
            css_path = os.path.join(STATIC_DIR, "style.css")
            if not os.path.exists(css_path):
                css_path = os.path.join(BASE_DIR, "style.css")
            return self._serve_file(css_path, "text/css")

        # 3. Serve JS directly
        elif path in ("/app.js", "/static/app.js"):
            js_path = os.path.join(STATIC_DIR, "app.js")
            if not os.path.exists(js_path):
                js_path = os.path.join(BASE_DIR, "app.js")
            return self._serve_file(js_path, "application/javascript")

        # 4. Any other static asset
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            file_path = os.path.join(STATIC_DIR, rel)
            mime_type, _ = mimetypes.guess_type(file_path)
            return self._serve_file(file_path, mime_type or "application/octet-stream")
            
        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        if urlparse(self.path).path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            query = body.get("query", "").strip()

            answer, sources = ask_gemini(query)
            
            resp_bytes = json.dumps({"answer": answer, "sources": sources}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        else:
            self.send_error(404, "Endpoint Not Found")

    def _serve_file(self, filepath, mime_type):
        if not os.path.exists(filepath):
            self.send_error(404, "File Not Found")
            return
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ClarusHandler)
    print("=" * 60)
    print(f"  CLARUS SERVER RUNNING at http://localhost:{PORT}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Clarus...")
        server.server_close()
