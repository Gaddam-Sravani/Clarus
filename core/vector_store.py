import os
import math
import re
from typing import List, Dict, Any
from config import CHROMA_DIR, TOP_K_CHUNKS

class FallbackVectorStore:
    """Smart BM25/TF-IDF retriever with stop-word filtering, stemming, and title/filename boosting."""

    STOP_WORDS = {
        "what", "is", "are", "the", "how", "to", "for", "in", "on", "at", "by",
        "from", "of", "and", "or", "do", "does", "did", "can", "could", "should",
        "would", "an", "a", "this", "that", "these", "those", "my", "your", "we",
        "you", "me", "it", "its", "they", "them", "their", "which", "who", "whom",
        "when", "where", "why", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "just", "don", "now"
    }

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[List[float]] = []

    def _stem(self, word: str) -> str:
        word = word.lower()
        for suffix in ["ing", "ies", "es", "s", "ed", "tion", "tions", "ment", "ments", "al"]:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        return [self._stem(w) for w in words if w not in self.STOP_WORDS]

    def index(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        num_docs = len(chunks)
        if num_docs == 0:
            return

        df: Dict[str, int] = {}
        for ch in chunks:
            # 5x Boost for Title and Filename
            boosted_text = f"{(ch.get('title', '') + ' ') * 5} {(ch.get('filename', '').replace('_', ' ').replace('.txt', '') + ' ') * 5} {ch['text']}"
            tokens = set(self._tokenize(boosted_text))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        self.vocab = {t: idx for idx, t in enumerate(sorted(df.keys()))}
        self.idf = {t: math.log((num_docs + 1) / (df[t] + 1)) + 1.0 for t in df}

        self.tfidf_vectors = []
        for ch in chunks:
            boosted_text = f"{(ch.get('title', '') + ' ') * 5} {(ch.get('filename', '').replace('_', ' ').replace('.txt', '') + ' ') * 5} {ch['text']}"
            vec = self._vectorize(boosted_text)
            self.tfidf_vectors.append(vec)

    def _vectorize(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * len(self.vocab)

        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        vec = [0.0] * len(self.vocab)
        for t, count in tf.items():
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] = (count / len(tokens)) * self.idf.get(t, 1.0)

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        if not self.chunks or not self.vocab:
            return []

        q_vec = self._vectorize(query)
        scores = []
        for idx, d_vec in enumerate(self.tfidf_vectors):
            dot = sum(q_vec[i] * d_vec[i] for i in range(len(q_vec)))
            scores.append((dot, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0.005:
                item = dict(self.chunks[idx])
                item["score"] = round(score, 4)
                results.append(item)

        return results

class CampusVectorStore:
    def __init__(self, chroma_dir: str = CHROMA_DIR):
        self.fallback_store = FallbackVectorStore()

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        self.fallback_store.index(chunks)

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        return self.fallback_store.search(query, top_k=top_k)
