import re
import json
import requests
from typing import Dict, Any, List
from config import GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL, TOP_K_CHUNKS
from core.vector_store import CampusVectorStore
from core.knowledge_base import KnowledgeBase

class CampusRAGEngine:
    ACRONYMS = {
        r"\btc\b": "transfer certificate tc",
        r"\btransfer certificate\b": "transfer certificate tc fee 50",
        r"\bbranches\b": "branch allocation engineering branches",
        r"\bbranch allocation\b": "branch allocation rules cgpa cutoff",
        r"\bcollege timings\b": "college timetable class periods schedule tmtable",
        r"\bclass timings\b": "college timetable class periods schedule tmtable",
        r"\btimetable\b": "timetable class periods schedule tmtable",
        r"\btimings\b": "timings schedule working hours timetable",
        r"\bvc\b": "Vice Chancellor Govardhan",
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
        r"\beee\b": "Electrical and Electronics Engineering",
        r"\bme\b": "Mechanical Engineering",
        r"\bce\b": "Civil Engineering",
    }

    SYSTEM_PROMPT = """You are the official "RGUKT Basar Campus AI Assistant", an autonomous, grounded intelligence platform for Rajiv Gandhi University of Knowledge Technologies (RGUKT) Basar, Telangana.

CRITICAL OPERATIONAL RULES:
1. Strict Grounding: Base your answer STRICTLY and FACTUALLY on the provided RGUKT Context below. Do not guess, speculate, or extrapolate beyond what is documented.
2. Accuracy on Specifics: State exact numbers, percentages (e.g. 75% attendance, 10% medical condonation), fees (e.g. Rs. 500/- condonation fee, Rs. 50/- TC fee, Rs. 100/- remedial fee), and period schedules (8:30 AM to 4:10 PM) verbatim.
3. Formatted & Actionable: Use concise paragraphs, bullet points, and bold headers.
4. Source Citing: Always reference the official circular or regulation title.
5. If the context does not contain enough steps for a process (e.g. TC issuance steps beyond fee), state the available facts (e.g., Rs. 50 fee in Examination Section) and note that specific administrative applications are processed at the Academic / Examination section counter.
"""

    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.kb = KnowledgeBase()
        self.vector_store = CampusVectorStore()
        self.is_initialized = False

    def initialize(self):
        chunks = self.kb.load_and_chunk_all()
        self.vector_store.index_chunks(chunks)
        self.is_initialized = True

    def normalize_query(self, query: str) -> str:
        normalized = query
        for pattern, full_term in self.ACRONYMS.items():
            normalized = re.sub(pattern, full_term, normalized, flags=re.IGNORECASE)
        return normalized

    def query(self, user_query: str) -> Dict[str, Any]:
        if not self.is_initialized:
            self.initialize()

        cleaned_query = user_query.strip()
        if not cleaned_query:
            return {
                "answer": "Please ask a question regarding RGUKT Basar regulations, circulars, or campus administration.",
                "sources": [],
                "model": GEMINI_MODEL,
                "status": "empty_query"
            }

        expanded_query = self.normalize_query(cleaned_query)
        retrieved_chunks = self.vector_store.search(expanded_query, top_k=TOP_K_CHUNKS)

        if not retrieved_chunks:
            return {
                "answer": "No relevant campus records found matching your query. Please check https://hub.rgukt.ac.in for the latest circulars.",
                "sources": [],
                "model": GEMINI_MODEL,
                "status": "no_context"
            }

        context_blocks = []
        unique_sources = []
        seen_urls = set()

        for idx, ch in enumerate(retrieved_chunks, 1):
            context_blocks.append(
                f"[Source {idx}: {ch['title']} | File: {ch.get('filename', '')} | Ref: {ch.get('official_ref', 'RGUKT-B')}]\n{ch['text']}"
            )
            src_url = ch.get("source_url", "https://www.rgukt.ac.in")
            if src_url not in seen_urls:
                seen_urls.add(src_url)
                unique_sources.append({
                    "title": ch.get("title", "Official RGUKT Regulation"),
                    "url": src_url,
                    "ref": ch.get("official_ref", "RGUKT-B/Official"),
                    "filename": ch.get("filename", "")
                })

        full_context = "\n\n---\n\n".join(context_blocks)
        prompt = f"""{self.SYSTEM_PROMPT}

=== OFFICIAL RGUKT BASAR CONTEXT ===
{full_context}
===================================

STUDENT QUERY: {cleaned_query}

ANSWER:"""

        answer = self._call_gemini(prompt)

        return {
            "query": cleaned_query,
            "answer": answer,
            "sources": unique_sources,
            "model": GEMINI_MODEL,
            "chunks_retrieved": len(retrieved_chunks),
            "status": "success"
        }

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key:
            return "Error: Gemini API key is not configured. Please set GEMINI_API_KEY."

        url = f"{GEMINI_API_URL}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "maxOutputTokens": 1024,
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "No text generated.")
                return "Received empty response from Gemini."
            else:
                return f"Gemini API returned status code {response.status_code}: {response.text}"
        except Exception as e:
            return f"Error connecting to Gemini API: {str(e)}"
