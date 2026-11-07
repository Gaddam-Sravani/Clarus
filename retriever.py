import os
import re
import math
from chunker import process_all_documents

# Common English filler words that must never influence search results
STOP_WORDS = {
    "what", "is", "the", "are", "in", "of", "and", "to", "for", "a", "an", "your",
    "our", "does", "do", "you", "tell", "me", "about", "who", "whom", "how", "can",
    "i", "we", "my", "this", "that", "there", "any", "from", "with", "at", "by", "on",
    "please", "give", "details", "information", "regarding", "college", "rgukt", "basar"
}

# Campus Acronyms & Synonyms
ACRONYMS = {
    r"\btc\b": "transfer certificate",
    r"\bvc\b": "vice chancellor",
    r"\bcoe\b": "controller of examinations",
    r"\bao\b": "administrative officer",
    r"\bhod\b": "head of department",
    r"\bpuc\b": "pre university course",
    r"\bdsw\b": "dean student welfare",
    r"\bcse\b": "computer science engineering",
    r"\bece\b": "electronics communication engineering",
    r"\beee\b": "electrical electronics engineering",
    r"\bme\b": "mechanical engineering",
    r"\bce\b": "civil engineering",
    r"\bmme\b": "metallurgical materials engineering",
    r"\bche\b": "chemical engineering",
    r"\btnp\b": "training placement",
    r"\bt&p\b": "training placement",
    r"\bsac\b": "student activity center",
}

class BM25Retriever:
    """Production-grade BM25 ranking engine with title boosting.
    Runs locally on CPU in <2ms with 0 MB downloads and zero API calls.
    """
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.tokenized_chunks = []
        self._build_index()

    def _tokenize(self, text: str):
        words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def _build_index(self):
        total_len = 0
        for ch in self.chunks:
            full_text = f"{ch.get('source_file', '')} {ch.get('title', '')} {ch['text']}"
            tokens = self._tokenize(full_text)
            self.tokenized_chunks.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            for word in set(tokens):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_doc_length = total_len / max(1, self.corpus_size)

    def _idf(self, word: str):
        df = self.doc_freqs.get(word, 0)
        return math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def normalize_query(self, query: str) -> str:
        expanded = query.lower()
        for pattern, replacement in ACRONYMS.items():
            expanded = re.sub(pattern, replacement, expanded)
        return expanded

    def search(self, query: str, top_k: int = 4):
        expanded_q = self.normalize_query(query)
        q_tokens = self._tokenize(expanded_q)
        
        if not q_tokens:
            return []

        scores = []
        for idx, tokens in enumerate(self.tokenized_chunks):
            doc_len = self.doc_lengths[idx]
            ch = self.chunks[idx]
            file_title = f"{ch.get('source_file', '')} {ch.get('title', '')}".lower()
            
            score = 0.0
            tf_dict = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            for q in q_tokens:
                if q not in tf_dict:
                    continue
                tf = tf_dict[q]
                idf = self._idf(q)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1, self.avg_doc_length)))
                bm25_score = idf * (numerator / denominator)
                
                # 10x multiplier if the keyword matches the filename or title
                title_boost = 10.0 if q in file_title else 1.0
                score += bm25_score * title_boost

            if score > 0.1:
                item = dict(ch)
                item["similarity_score"] = round(score, 2)
                scores.append((score, item))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]

# Alias so server.py connects seamlessly
PurePythonRetriever = BM25Retriever

