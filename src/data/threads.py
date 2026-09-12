"""
threads.py — reconstructs conversation threads from in_response_to_tweet_id.
Each thread is saved as a JSON record with the full message chain.

Usage:
    python -m src.data.threads
"""

import json
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INPUT_FILE    = PROCESSED_DIR / "tweets_clean.csv"
OUTPUT_FILE   = PROCESSED_DIR / "threads.jsonl"
BRAND_HANDLE  = "SpotifyCares"


def load(path: Path) -> pd.DataFrame:
    print(f"[threads] Loading {path.name} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"  Rows: {len(df):,}")
    return df


def build_reply_map(df: pd.DataFrame) -> dict:
    """Map each tweet_id → list of tweet_ids that reply to it."""
    reply_map = {}
    for _, row in df.iterrows():
        parent = row.get("in_response_to_tweet_id")
        if pd.notna(parent) and parent:
            reply_map.setdefault(parent, []).append(row["tweet_id"])
    return reply_map


def find_roots(df: pd.DataFrame) -> list:
    """
    A root tweet is a customer tweet that has no parent in our dataset,
    OR whose parent is not in our dataset.
    """
    all_ids     = set(df["tweet_id"].unique())
    has_parent  = df["in_response_to_tweet_id"].notna()
    parent_in   = df["in_response_to_tweet_id"].isin(all_ids)

    # roots: customer tweets with no parent in our data
    roots = df[
        (df["role"] == "customer") &
        (~has_parent | ~parent_in)
    ]["tweet_id"].tolist()

    print(f"[threads] Root (conversation-starting) tweets: {len(roots):,}")
    return roots


def build_thread(root_id: str, tweet_map: dict, reply_map: dict, max_depth: int = 10) -> list:
    """BFS from root, collecting the conversation chain."""
    thread  = []
    queue   = [root_id]
    visited = set()
    depth   = 0

    while queue and depth < max_depth:
        next_queue = []
        for tid in queue:
            if tid in visited or tid not in tweet_map:
                continue
            visited.add(tid)
            row = tweet_map[tid]
            thread.append({
                "tweet_id":   tid,
                "author_id":  row["author_id"],
                "role":       row["role"],
                "text":       row["text"],
                "text_clean": row["text_clean"],
                "created_at": row.get("created_at", ""),
            })
            next_queue.extend(reply_map.get(tid, []))
        queue = next_queue
        depth += 1

    return thread


def main():
    df = load(INPUT_FILE)

    tweet_map = df.set_index("tweet_id").to_dict(orient="index")
    reply_map = build_reply_map(df)
    roots     = find_roots(df)

    threads = []
    skipped = 0

    for root_id in roots:
        thread = build_thread(root_id, tweet_map, reply_map)

        # keep only threads that have at least one brand reply
        roles = [t["role"] for t in thread]
        if "brand" not in roles:
            skipped += 1
            continue

        # extract the customer opening message and the first brand reply
        customer_msgs = [t for t in thread if t["role"] == "customer"]
        brand_msgs    = [t for t in thread if t["role"] == "brand"]

        threads.append({
            "thread_id":        root_id,
            "length":           len(thread),
            "customer_opening": customer_msgs[0]["text_clean"] if customer_msgs else "",
            "first_brand_reply": brand_msgs[0]["text_clean"] if brand_msgs else "",
            "messages":         thread,
        })

    print(f"[threads] Threads built:  {len(threads):,}")
    print(f"[threads] Skipped (no brand reply): {skipped:,}")

    # save as JSONL — one thread per line
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[threads] Saved to {OUTPUT_FILE}")

    # quick stats
    lengths = [t["length"] for t in threads]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n[threads] Thread length stats:")
    print(f"  Min: {min(lengths)}  Max: {max(lengths)}  Avg: {avg_len:.2f}")


if __name__ == "__main__":
    main()