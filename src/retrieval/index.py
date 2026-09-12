"""
index.py — builds sentence-transformer embeddings + FAISS index over
historical SpotifyCares threads. Golden set threads are explicitly excluded
to prevent leakage.

Usage:
    python -m src.retrieval.index
"""

import json
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
THREADS_FILE  = PROCESSED_DIR / "threads.jsonl"
GOLDEN_FILE   = Path("evaluation/golden_set.csv")
INDEX_PATH    = PROCESSED_DIR / "faiss.index"
METADATA_PATH = PROCESSED_DIR / "index_metadata.pkl"
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
SEED          = 42


def load_threads(path: Path) -> list:
    print(f"[index] Loading threads from {path.name} ...")
    threads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    print(f"  Threads loaded: {len(threads):,}")
    return threads


def load_golden_ids(path: Path) -> set:
    """Load thread IDs from the golden set to exclude from the index."""
    if not path.exists():
        print("[index] WARNING: golden_set.csv not found — no leakage exclusion applied")
        return set()
    df = pd.read_csv(path, dtype=str)
    # golden set stores the customer_message; we match by thread content
    # the thread_id is stored in conversation_context prefix — use message matching
    # simpler: we stored thread_id in the pool but not in golden CSV
    # so we exclude by matching customer_message text against thread customer_opening
    messages = set(df["customer_message"].str.strip().str.lower().unique())
    print(f"[index] Golden set messages to exclude: {len(messages):,}")
    return messages


def build_records(threads: list, excluded_messages: set) -> list:
    """
    Build one retrievable record per thread.
    Each record has: customer_message, context, brand_response, intent, thread_id.
    """
    records = []
    skipped = 0

    for t in threads:
        msg = t["customer_opening"].strip()
        if not msg:
            skipped += 1
            continue

        # leakage exclusion
        if msg.strip().lower() in excluded_messages:
            skipped += 1
            continue

        brand_reply = t["first_brand_reply"].strip()
        if not brand_reply:
            skipped += 1
            continue

        # build context string from thread messages
        context_parts = []
        for m in t.get("messages", [])[:4]:
            role = "Customer" if m["role"] == "customer" else "Spotify"
            context_parts.append(f"{role}: {m['text_clean'][:120]}")

        records.append({
            "thread_id":        t["thread_id"],
            "customer_message": msg,
            "context":          " | ".join(context_parts),
            "brand_response":   brand_reply,
            "thread_length":    t["length"],
            # intent will be filled by the classifier in Phase 6
            # for now we store empty string as placeholder
            "intent":           "",
        })

    print(f"[index] Records built: {len(records):,}  (skipped: {skipped:,})")
    return records


def embed_records(records: list, model_name: str) -> np.ndarray:
    print(f"[index] Embedding {len(records):,} records ...")
    model = SentenceTransformer(model_name)
    texts = [r["customer_message"] for r in records]
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    # IndexFlatIP = exact inner product search (cosine since embeddings are normalised)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[index] FAISS index built: {index.ntotal:,} vectors  dim={dim}")
    return index


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    threads          = load_threads(THREADS_FILE)
    excluded         = load_golden_ids(GOLDEN_FILE)
    records          = build_records(threads, excluded)
    embeddings       = embed_records(records, MODEL_NAME)
    index            = build_faiss_index(embeddings)

    # save index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[index] FAISS index saved to {INDEX_PATH}")

    # save metadata (records list — needed to look up text after retrieval)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"[index] Metadata saved to {METADATA_PATH}")

    print(f"\n[index] Done. Index ready for retrieval.")


if __name__ == "__main__":
    main()