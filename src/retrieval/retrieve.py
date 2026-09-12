"""
retrieve.py — top-k similarity retrieval from the FAISS index.
Returns retrieved examples with similarity scores for use by the
intent classifier and reply generator.

Usage (as a module):
    from src.retrieval.retrieve import Retriever
    r = Retriever()
    results = r.retrieve("my spotify keeps crashing", k=5)
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
INDEX_PATH    = PROCESSED_DIR / "faiss.index"
METADATA_PATH = PROCESSED_DIR / "index_metadata.pkl"
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K     = 5
MIN_SIMILARITY = 0.55


class Retriever:
    def __init__(self):
        print("[retriever] Loading FAISS index ...")
        self.index = faiss.read_index(str(INDEX_PATH))

        with open(METADATA_PATH, "rb") as f:
            self.records = pickle.load(f)

        self.model = SentenceTransformer(MODEL_NAME)
        print(f"[retriever] Ready — {self.index.ntotal:,} vectors in index")

    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        """
        Retrieve top-k most similar historical threads for a query message.

        Returns a list of dicts, each with:
            customer_message, context, brand_response, intent,
            thread_id, similarity_score, meets_threshold
        """
        # embed query
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # search
        scores, indices = self.index.search(embedding, k)
        scores  = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            record = self.records[idx].copy()
            record["similarity_score"] = float(score)
            record["meets_threshold"]  = float(score) >= MIN_SIMILARITY
            results.append(record)

        return results

    def retrieval_quality(self, results: list[dict]) -> dict:
        """
        Summarise retrieval quality signals used by the escalation policy.
        """
        if not results:
            return {
                "top_score":         0.0,
                "avg_score":         0.0,
                "n_above_threshold": 0,
                "weak_retrieval":    True,
            }

        scores = [r["similarity_score"] for r in results]
        n_above = sum(1 for r in results if r["meets_threshold"])

        return {
            "top_score":         max(scores),
            "avg_score":         sum(scores) / len(scores),
            "n_above_threshold": n_above,
            "weak_retrieval":    n_above == 0,
        }


if __name__ == "__main__":
    # quick smoke test
    r = Retriever()
    query = "my spotify keeps crashing after the latest update"
    print(f"\nQuery: {query}\n")
    results = r.retrieve(query, k=3)
    for i, res in enumerate(results, 1):
        print(f"Result {i}  (similarity: {res['similarity_score']:.3f}  threshold_met: {res['meets_threshold']})")
        print(f"  Customer: {res['customer_message'][:100]}")
        print(f"  Spotify:  {res['brand_response'][:100]}")
        print()
    print("Quality:", r.retrieval_quality(results))