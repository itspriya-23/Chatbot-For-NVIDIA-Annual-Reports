"""
Lightweight, reproducible, zero-dependency Vector Store using NumPy.
Stores text chunks with metadata (source file, fiscal year, page number) and embedding vectors.
Computes cosine similarity ranking directly.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple

class SimpleVectorStore:
    def __init__(self, storage_dir: str = "data/vector_store"):
        self.storage_dir = storage_dir
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add chunks and their corresponding embedding vectors to the store."""
        if not chunks or not embeddings:
            return

        new_embeddings = np.array(embeddings, dtype=np.float32)
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
            self.chunks = list(chunks)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.chunks.extend(chunks)

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank chunks by Cosine Similarity to the query vector.
        Returns: List of tuples (chunk_dict, similarity_score)
        """
        if self.embeddings.size == 0 or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        # Matrix-vector dot product
        dot_products = np.dot(self.embeddings, q_vec)
        # Norms of stored chunk vectors
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Avoid division by zero
        cosine_scores = dot_products / (doc_norms * q_norm + 1e-10)

        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(cosine_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(cosine_scores[idx])))
        return results

    def save(self):
        """Persist embeddings to .npy and chunks to .json."""
        os.makedirs(self.storage_dir, exist_ok=True)
        chunks_path = os.path.join(self.storage_dir, "chunks.json")
        embeddings_path = os.path.join(self.storage_dir, "embeddings.npy")

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        np.save(embeddings_path, self.embeddings)
        print(f"[VectorStore] Successfully saved {len(self.chunks)} chunks to {self.storage_dir}")

    def load(self) -> bool:
        """Load embeddings and chunks from disk."""
        chunks_path = os.path.join(self.storage_dir, "chunks.json")
        embeddings_path = os.path.join(self.storage_dir, "embeddings.npy")

        if not os.path.exists(chunks_path) or not os.path.exists(embeddings_path):
            return False

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(embeddings_path)
        print(f"[VectorStore] Loaded {len(self.chunks)} chunks (embeddings shape: {self.embeddings.shape})")
        return True

