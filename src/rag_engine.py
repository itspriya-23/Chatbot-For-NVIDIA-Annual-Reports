"""
Core RAG Engine for NVIDIA Financial Documents.
Orchestrates semantic retrieval, strict citation grounding,
numerical/unit precision, and safe abstention when evidence is lacking.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Dict, Any, List, Optional
from src.gemini_client import GeminiClient
from src.vector_store import SimpleVectorStore

class RAGEngine:
    def __init__(self, vector_store_dir: str = "data/vector_store", similarity_threshold: float = 0.38):
        self.client = GeminiClient()
        self.vector_store = SimpleVectorStore(storage_dir=vector_store_dir)
        self.is_loaded = self.vector_store.load()
        self.similarity_threshold = similarity_threshold

    def query(self, user_question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user question through the complete RAG pipeline.
        Returns:
            Dict containing:
                - question: user question
                - answer: grounded LLM response with citations
                - sources: list of retrieved chunk metadata & scores
                - abstained: boolean indicating whether system abstained due to missing evidence
        """
        if not self.is_loaded:
            # Try reloading in case ingestion just finished
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
        if not query_embedding:
            return {
                "question": user_question,
                "answer": "Unable to generate embedding for the question. Please verify your GEMINI_API_KEY.",
                "sources": [],
                "abstained": True
            }

        # 2. Retrieve top-k relevant chunks
        search_results = self.vector_store.similarity_search(query_embedding, top_k=top_k)

        if not search_results:
            return {
                "question": user_question,
                "answer": "No relevant documents found in the database.",
                "sources": [],
                "abstained": True
            }

        # Check top similarity score against threshold
        top_score = search_results[0][1]
        
        # Prepare structured source list
        sources = []
        context_blocks = []
        for rank, (chunk, score) in enumerate(search_results, start=1):
            sources.append({
                "rank": rank,
                "title": chunk.get("title", "NVIDIA Filing"),
                "year": chunk.get("year", "N/A"),
                "doc_type": chunk.get("doc_type", "Filing"),
                "page_number": chunk.get("page_number", "N/A"),
                "citation": chunk.get("citation", f"{chunk.get('title')}, Page {chunk.get('page_number')}"),
                "score": round(score, 4),
                "text_snippet": chunk.get("content_only", "")[:300] + "..."
            })
            context_blocks.append(
                f"--- EVIDENCE EXCERPT {rank} ---\n"
                f"Source Document: {chunk.get('title')}\n"
                f"Fiscal Year: FY{chunk.get('year')}\n"
                f"PDF Page Number: {chunk.get('page_number')}\n"
                f"Content:\n{chunk.get('content_only')}\n"
            )

        full_context = "\n\n".join(context_blocks)

        # 3. Construct System Prompt & User Prompt
        system_instruction = (
            "You are an expert Senior Financial Analyst and AI Assistant specializing in NVIDIA Corporation "
            "Annual Reports (Form 10-K) and Proxy Statements (DEF 14A) covering 1998 through 2026.\n\n"
            "STRICT RULES YOU MUST FOLLOW:\n"
            "1. GROUNDING: Answer the user's question using ONLY the provided Evidence Excerpts. Do NOT use external or unmentioned knowledge.\n"
            "2. CITATIONS: You MUST cite the source document and exact page number for every factual statement, calculation, and metric. "
            "Format citations clearly, for example: [NVIDIA 2024 Annual Report, Page 45].\n"
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

        # 4. Generate answer via Gemini
        answer = self.client.generate(user_prompt, system_instruction=system_instruction)

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
            "top_score": top_score,
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
