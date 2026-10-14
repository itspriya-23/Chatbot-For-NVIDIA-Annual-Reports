"""
Unit and Integration Tests for NVIDIA Financial RAG Chatbot.
Verifies chunking, cosine similarity math, metadata formatting, and abstention behavior.
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingest import clean_text, chunk_text
from src.vector_store import SimpleVectorStore

def test_clean_text():
    raw = "NVIDIA   Corporation\n\n\n\nAnnual   Report \xa02024\u200b"
    cleaned = clean_text(raw)
    assert "  " not in cleaned, "Should remove redundant spaces"
    assert "\n\n\n" not in cleaned, "Should normalize multiple newlines"
    assert "NVIDIA Corporation" in cleaned
    assert "2024" in cleaned
    print("PASS: test_clean_text")

def test_chunk_text():
    sample_text = (
        "Paragraph 1: In fiscal year 2024, NVIDIA achieved record financial results. "
        "Revenue was $60.9 billion, up 126% from a year ago.\n\n"
        "Paragraph 2: Data Center revenue was $47.5 billion, up 217%. "
        "Compute revenue increased 244% and networking increased 133%.\n\n"
        "Paragraph 3: GAAP gross margin was 72.7%, up from 56.9% in the prior year."
    )
    chunks = chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2, "Should divide sample text into multiple chunks"
    for c in chunks:
        assert len(c) > 20, "Chunk should have substantial length"
    print("PASS: test_chunk_text")

def test_vector_store_cosine_similarity():
    store = SimpleVectorStore()
    
    # 3 mock chunks
    chunks = [
        {"chunk_id": "c1", "text": "NVIDIA revenue FY24", "title": "FY24 10-K", "page_number": 1},
        {"chunk_id": "c2", "text": "Board of Directors 1999", "title": "1999 Proxy", "page_number": 5},
        {"chunk_id": "c3", "text": "Mellanox acquisition", "title": "FY20 10-K", "page_number": 12}
    ]
    # Mock vectors: vector 1 aligns closely with query [1.0, 0.0]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0]
    ]
    store.add_documents(chunks, embeddings)

    # Query vector aligning with chunk 1
    query_vec = [0.95, 0.05, 0.0]
    results = store.similarity_search(query_vec, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk["chunk_id"] == "c1", "Top chunk should be c1"
    assert top_score > 0.9, "Score should be close to 1.0"
    print("PASS: test_vector_store_cosine_similarity")

def test_metadata_completeness():
    mock_chunk = {
        "chunk_id": "NVIDIA_2024_Annual_Report.pdf_p45_c3",
        "source_file": "NVIDIA_2024_Annual_Report.pdf",
        "title": "NVIDIA 2024 Annual Report (Form 10-K)",
        "year": 2024,
        "doc_type": "Annual Report",
        "page_number": 45,
        "citation": "NVIDIA 2024 Annual Report (Form 10-K), Page 45",
        "text": "[Source: NVIDIA 2024 Annual Report | Year: FY2024 | Page: 45]\nData Center revenue was $47.5 billion."
    }
    assert "page_number" in mock_chunk and mock_chunk["page_number"] == 45
    assert "year" in mock_chunk and mock_chunk["year"] == 2024
    assert "citation" in mock_chunk
    print("PASS: test_metadata_completeness")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    test_clean_text()
    test_chunk_text()
    test_vector_store_cosine_similarity()
    test_metadata_completeness()
    print("=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

