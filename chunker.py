import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_PAGES_DIR = os.path.join(BASE_DIR, "data", "saved_pages")

CHUNK_SIZE = 700      # Target character length per chunk
CHUNK_OVERLAP = 150   # Overlap between consecutive chunks

NOISY_PATTERNS = {
    "skip to main content", "staff login", "webmaster@rgukt.ac.in",
    "a-", "a", "a+", "hc", "sr", "screen reader", "main navigation"
}

def extract_source_url(text: str) -> str:
    match = re.search(r"Source URL:\s*([^\r\n]+)", text)
    if match:
        return match.group(1).strip()
    return "https://www.rgukt.ac.in"

def clean_noise(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        line_clean = line.strip()
        if line_clean.lower() in NOISY_PATTERNS:
            continue
        if line_clean.startswith("Source URL:") or line_clean.startswith("Official Ref:") or line_clean.startswith("="):
            continue
        if line_clean:
            cleaned_lines.append(line_clean)
    return "\n".join(cleaned_lines)

def split_text_into_chunks(text: str, filename: str, source_url: str):
    cleaned_body = clean_noise(text)
    base_name = filename.replace(".txt", "")

    # Split text into sentences / natural lines
    lines = [l.strip() for l in cleaned_body.split("\n") if l.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for line in lines:
        # If adding this line stays within CHUNK_SIZE
        if len(current_chunk) + len(line) + 1 <= CHUNK_SIZE:
            current_chunk = f"{current_chunk}\n{line}".strip()
        else:
            # We hit the size limit! Save the current chunk
            if current_chunk:
                chunk_id = f"{base_name}_c{chunk_index:03d}"
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk,
                    "source_file": filename,
                    "source_url": source_url,
                    "length": len(current_chunk)
                })
                chunk_index += 1
                
                # Keep the last CHUNK_OVERLAP characters for smooth context continuity
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = f"{overlap_text}\n{line}".strip()
            else:
                current_chunk = line

    # Save the final remaining chunk
    if current_chunk:
        chunk_id = f"{base_name}_c{chunk_index:03d}"
        chunks.append({
            "id": chunk_id,
            "text": current_chunk,
            "source_file": filename,
            "source_url": source_url,
            "length": len(current_chunk)
        })

    return chunks

def process_all_documents():
    all_chunks = []
    if not os.path.exists(SAVED_PAGES_DIR):
        print(f"Directory not found: {SAVED_PAGES_DIR}")
        return all_chunks

    txt_files = [f for f in sorted(os.listdir(SAVED_PAGES_DIR)) if f.endswith(".txt")]
    print(f"Reading {len(txt_files)} text files from data/saved_pages/ ...")

    for filename in txt_files:
        filepath = os.path.join(SAVED_PAGES_DIR, filename)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        source_url = extract_source_url(raw_text)
        doc_chunks = split_text_into_chunks(raw_text, filename, source_url)
        all_chunks.extend(doc_chunks)
        print(f"  -> {filename:<22} : {len(doc_chunks)} chunks ({len(raw_text)} chars)")

    print(f"\nTotal Chunks Generated: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = process_all_documents()
    
    # Preview chunk count per file
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS FROM vc.txt:")
    print("=" * 60)
    vc_chunks = [c for c in chunks if c['source_file'] == 'vc.txt']
    for i, c in enumerate(vc_chunks[:3]):
        print(f"\n--- Chunk #{i+1} [ID: {c['id']}] (Length: {c['length']} chars) ---")
        print(c['text'][:180] + "...")
