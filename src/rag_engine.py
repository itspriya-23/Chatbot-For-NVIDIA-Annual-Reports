"""
Core RAG Engine for NVIDIA Financial Documents.
Orchestrates semantic retrieval, strict citation grounding,
numerical/unit precision, and safe abstention when evidence is lacking.
"""

import os
import sys
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Dict, Any, List, Optional
from src.gemini_client import GeminiClient
from src.vector_store import SimpleVectorStore

STOPWORDS = {
    'what', 'when', 'where', 'which', 'who', 'whom', 'this', 'that', 'with', 'from',
    'about', 'according', 'does', 'did', 'have', 'been', 'were', 'tell', 'show',
    'report', 'reports', 'statement', 'annual', 'fiscal', 'year', 'the', 'and', 'for'
}

ACRONYMS = {
    'ceo': ['chief', 'executive', 'officer', 'president'],
    'cfo': ['chief', 'financial', 'officer'],
    'cto': ['chief', 'technology', 'officer'],
    'ipo': ['initial', 'public', 'offering'],
    'gaap': ['gaap', 'generally', 'accepted', 'accounting', 'principles'],
    'm&a': ['acquisition', 'acquire', 'merger', 'mellanox']
}

def sanitize_text(text: str) -> str:
    """Normalize typography, quotes, and dashes to standard clean characters."""
    if not text:
        return ""
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201c\u201d]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\ufffd', '')
    return text.strip()

class RAGEngine:
    def __init__(self, vector_store_dir: str = "data/vector_store", similarity_threshold: float = 0.55):
        self.client = GeminiClient()
        self.vector_store = SimpleVectorStore(storage_dir=vector_store_dir)
        self.is_loaded = self.vector_store.load()
        self.similarity_threshold = similarity_threshold

    def query(self, user_question: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Process a user question through the complete RAG pipeline with Hybrid Search.
        Combines 3072-dim dense cosine similarity with lexical keyword matching.
        """
        if not self.is_loaded:
            self.is_loaded = self.vector_store.load()
            if not self.is_loaded:
                return {
                    "question": user_question,
                    "answer": "The vector database is not loaded yet. Please run ingestion first using `python src/ingest.py`.",
                    "sources": [],
                    "abstained": True
                }

        # 1. Embed user question
        query_embedding = self.client.get_embedding(user_question)
        if not query_embedding or all(v == 0.0 for v in query_embedding):
            return {
                "question": user_question,
                "answer": "Unable to generate embedding for the question. Please verify your GEMINI_API_KEY.",
                "sources": [],
                "abstained": True
            }

        # 2. Dense semantic retrieval: fetch top 35 candidates
        dense_results = self.vector_store.similarity_search(query_embedding, top_k=35)

        if not dense_results:
            return {
                "question": user_question,
                "answer": "No relevant documents found in the database.",
                "sources": [],
                "abstained": True
            }

        top_dense_score = dense_results[0][1]

        # Fast abstention if query is entirely out of domain (very low semantic alignment)
        if top_dense_score < self.similarity_threshold:
            return {
                "question": user_question,
                "answer": "I cannot answer this question based on the provided NVIDIA reports because there is insufficient evidence in the ingested documents.",
                "sources": [],
                "top_score": round(top_dense_score, 4),
                "abstained": True
            }

        # 3. Hybrid Lexical Scoring & Temporal Re-ranking
        tokens = re.findall(r'[a-zA-Z0-9]+', user_question.lower())
        years_in_q = [int(y) for y in re.findall(r'\b(19\d\d|20\d\d)\b', user_question)]
        query_terms = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
        for acr, expansions in ACRONYMS.items():
            if acr in tokens or acr in user_question.lower():
                query_terms.extend(expansions)
        query_term_set = set(query_terms)

        hybrid_ranked = []
        for chunk, dense_score in dense_results:
            text_lower = chunk.get("text", "").lower()
            matches = sum(1 for term in query_term_set if term in text_lower)
            lexical_boost = (matches / max(len(query_term_set), 1)) * 0.08
            
            # Temporal boost: prioritize chunks from the exact fiscal year requested
            year_boost = 0.10 if (years_in_q and chunk.get("year") in years_in_q) else 0.0
            
            combined_score = dense_score + lexical_boost + year_boost
            hybrid_ranked.append((chunk, combined_score, dense_score))

        hybrid_ranked.sort(key=lambda x: x[1], reverse=True)
        top_candidates = hybrid_ranked[:top_k]

        # Prepare structured source list and evidence context
        sources = []
        context_blocks = []
        for rank, (chunk, combined_score, dense_score) in enumerate(top_candidates, start=1):
            sources.append({
                "rank": rank,
                "title": chunk.get("title", "NVIDIA Filing"),
                "year": chunk.get("year", "N/A"),
                "doc_type": chunk.get("doc_type", "Filing"),
                "page_number": chunk.get("page_number", "N/A"),
                "citation": chunk.get("citation", f"{chunk.get('title')}, Page {chunk.get('page_number')}"),
                "score": round(dense_score, 4),
                "hybrid_score": round(combined_score, 4),
                "text_snippet": sanitize_text(chunk.get("content_only", "")[:300]) + "..."
            })
            clean_content = sanitize_text(chunk.get("content_only", ""))
            context_blocks.append(
                f"--- EVIDENCE EXCERPT {rank} ---\n"
                f"Source Document: {chunk.get('title')}\n"
                f"Fiscal Year: FY{chunk.get('year')}\n"
                f"PDF Page Number: {chunk.get('page_number')}\n"
                f"Content:\n{clean_content}\n"
            )

        full_context = "\n\n".join(context_blocks)

        # 4. Construct System Prompt & Grounding Instructions
        system_instruction = (
            "You are an expert Senior Financial Analyst and AI Assistant specializing in NVIDIA Corporation "
            "Annual Reports (Form 10-K) and Proxy Statements (DEF 14A) covering 1998 through 2026.\n\n"
            "STRICT RULES YOU MUST FOLLOW:\n"
            "1. GROUNDING: Answer the user's question using ONLY the provided Evidence Excerpts. Do NOT use external or unmentioned knowledge.\n"
            "2. CITATIONS: You MUST cite the source document and exact page number for every factual statement, calculation, and metric. "
            "Format citations clearly, for example: [NVIDIA 2024 Annual Report (Form 10-K), Page 45].\n"
            "3. NUMERICAL & TEMPORAL ACCURACY: Be extremely careful with numbers, metric names (e.g., Data Center revenue vs Total revenue), "
            "units (e.g., millions '$M', billions '$B', percentages '%'), and fiscal periods (e.g., Fiscal Year 2024 ended January 28, 2024).\n"
            "4. APPROPRIATE ABSTENTION: If the provided evidence excerpts DO NOT contain sufficient information to answer the question accurately, "
            "you MUST state: 'I cannot answer this question based on the provided NVIDIA reports because there is insufficient evidence in the ingested documents.' "
            "Do NOT speculate, guess, or invent numbers.\n"
            "5. TONE: Professional, concise, objective, and audit-ready."
        )

        user_prompt = (
            f"USER QUESTION: {user_question}\n\n"
            f"RETRIEVED EVIDENCE EXCERPTS:\n"
            f"{full_context}\n\n"
            "INSTRUCTIONS:\n"
            "Synthesize an accurate, grounded answer using the excerpts above. "
            "Include inline citations with [Document Title, Page Number]. "
            "If the excerpts do not contain the answer, explicitly state that evidence is insufficient."
        )

        # 5. Generate answer via Gemini
        raw_answer = self.client.generate(user_prompt, system_instruction=system_instruction)
        answer = sanitize_text(raw_answer)

        # Detect abstention
        abstained = any(phrase in answer.lower() for phrase in [
            "insufficient evidence",
            "cannot answer",
            "do not contain",
            "not mentioned",
            "not provided in the excerpts"
        ])

        return {
            "question": user_question,
            "answer": answer,
            "sources": sources,
            "top_score": round(top_dense_score, 4),
            "abstained": abstained
        }

if __name__ == "__main__":
    engine = RAGEngine()
    test_q = "What was NVIDIA's total revenue in fiscal year 2024?"
    print(f"Testing RAG Engine with query: '{test_q}'\n")
    res = engine.query(test_q)
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"][:3]:
        print(f"- {s['citation']} (Score: {s['score']})")
