import math
import re
from chunker import process_all_documents

class PurePythonRetriever:
    """Zero-download, pure-Python TF-IDF & Cosine Similarity search engine.
    Runs 100% locally on CPU using built-in math and re (0 MB downloads).
    """
    def __init__(self, chunks):
        self.chunks = chunks
        self.vocab = {}
        self.idf = {}
        self.tfidf_vectors = []
        self._build_index()

    def _tokenize(self, text: str):
        # Splits text into clean lowercase words (minimum 2 characters)
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def _build_index(self):
        num_docs = len(self.chunks)
        if num_docs == 0:
            return

        # 1. Calculate Document Frequency (DF) for each word
        df = {}
        for ch in self.chunks:
            words = set(self._tokenize(ch["text"]))
            for w in words:
                df[w] = df.get(w, 0) + 1

        # 2. Build vocabulary mapping and Inverse Document Frequency (IDF)
        self.vocab = {w: idx for idx, w in enumerate(sorted(df.keys()))}
        self.idf = {w: math.log((num_docs + 1) / (df[w] + 1)) + 1.0 for w in df}

        # 3. Pre-compute TF-IDF vector for every chunk
        for ch in self.chunks:
            vec = self._vectorize(ch["text"])
            self.tfidf_vectors.append(vec)

    def _vectorize(self, text: str):
        words = self._tokenize(text)
        if not words:
            return [0.0] * len(self.vocab)

        # Term Frequency (TF)
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1

        # Combine TF and IDF
        vec = [0.0] * len(self.vocab)
        for w, count in tf.items():
            if w in self.vocab:
                idx = self.vocab[w]
                vec[idx] = (count / len(words)) * self.idf.get(w, 1.0)

        # Normalize vector to unit length (Euclidean norm)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    ACRONYMS = {
        r"\bvc\b": "Vice Chancellor",
        r"\bcoe\b": "Controller of Examinations",
        r"\bao\b": "Administrative Officer",
        r"\bhod\b": "Head of Department",
        r"\bpuc\b": "Pre-University Course",
        r"\bdsw\b": "Dean of Student Welfare",
        r"\bese\b": "End Semester Examination",
        r"\bsgpa\b": "Semester Grade Point Average",
        r"\bcgpa\b": "Cumulative Grade Point Average",
        r"\btnp\b": "Training and Placement Cell",
        r"\bt&p\b": "Training and Placement Cell",
        r"\bsac\b": "Student Activity Center",
        r"\bcse\b": "Computer Science and Engineering",
        r"\bece\b": "Electronics and Communication Engineering",
    }
    
    def normalize_query(self, query: str) -> str:
        """Expands common campus acronyms into formal university terms."""
        expanded = query
        for pattern, replacement in self.ACRONYMS.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded
    
    def search(self, query: str, top_k: int = 3):
        # 1. Expand campus acronyms first!
        normalized_q = self.normalize_query(query)
        if normalized_q != query:
            print(f"[Acronym Expansion] '{query}' -> '{normalized_q}'")
    
        if not self.chunks or not self.vocab:
            return []
    
        # 2. Vectorize the expanded query
        query_vec = self._vectorize(normalized_q)
    
        # 3. Calculate Cosine Similarity as usual
        scores = []
        for idx, doc_vec in enumerate(self.tfidf_vectors):
            dot_product = sum(query_vec[i] * doc_vec[i] for i in range(len(query_vec)))
            scores.append((dot_product, idx))
    
        scores.sort(key=lambda x: x[0], reverse=True)
    
        results = []
        for score, idx in scores[:top_k]:
            if score > 0.01:
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["similarity_score"] = round(score, 4)
                results.append(chunk_copy)
    
        return results

if __name__ == "__main__":
    print("Loading chunks from chunker...")
    chunks = process_all_documents()
    
    print("\nBuilding search index...")
    retriever = PurePythonRetriever(chunks)
    print(f"Indexed {len(chunks)} chunks into vocabulary of {len(retriever.vocab)} unique words.")

    # Test Queries
    test_queries = [
        "Who is the Vice Chancellor?",
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: '{q}'")
        print("=" * 60)
        matches = retriever.search(q, top_k=2)
        for m in matches:
            print(f"-> [Score: {m['similarity_score']}] from {m['source_file']}")
            print(f"   URL: {m['source_url']}")
            print(f"   Preview: {m['text'][:150]}...\n")

