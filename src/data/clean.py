"""
clean.py — dedup, malformed-record removal, language filtering,
customer-vs-brand tagging, URL/mention normalisation.

Usage:
    python -m src.data.clean
"""

import re
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INPUT_FILE    = PROCESSED_DIR / "tweets_normalised.csv"
OUTPUT_FILE   = PROCESSED_DIR / "tweets_clean.csv"
BRAND_HANDLE  = "SpotifyCares"


def load(path: Path) -> pd.DataFrame:
    print(f"[clean] Loading {path.name} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"  Rows loaded: {len(df):,}")
    return df


def filter_brand(df: pd.DataFrame, handle: str) -> pd.DataFrame:
    """Keep only tweets involving SpotifyCares — either sent by them or sent to them."""
    brand_replied   = df[df["author_id"] == handle]["in_response_to_tweet_id"].dropna().unique()
    brand_tweet_ids = set(df[df["author_id"] == handle]["tweet_id"].unique())
    customer_ids    = set(brand_replied)

    mask = (
        (df["author_id"] == handle) |
        (df["tweet_id"].isin(customer_ids))
    )
    filtered = df[mask].copy()
    print(f"[clean] Rows after brand filter ({handle}): {len(filtered):,}")
    return filtered


def tag_role(df: pd.DataFrame, handle: str) -> pd.DataFrame:
    """Add a 'role' column: 'brand' or 'customer'."""
    df["role"] = df["author_id"].apply(
        lambda x: "brand" if x == handle else "customer"
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["tweet_id"])
    print(f"[clean] Duplicates removed: {before - len(df):,}  →  {len(df):,} rows remain")
    return df


def remove_malformed(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # drop rows with missing tweet_id, author_id, or text
    df = df.dropna(subset=["tweet_id", "author_id", "text"])
    # drop rows where text is empty/whitespace
    df = df[df["text"].str.strip().str.len() > 0]
    print(f"[clean] Malformed removed: {before - len(df):,}  →  {len(df):,} rows remain")
    return df


def normalise_text(text: str) -> str:
    """Replace URLs and @mentions with tokens; collapse whitespace."""
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    df["text_clean"] = df["text"].apply(normalise_text)
    return df


def filter_very_short(df: pd.DataFrame, min_chars: int = 10) -> pd.DataFrame:
    """Drop tweets whose cleaned text is too short to carry meaning."""
    before = len(df)
    df = df[df["text_clean"].str.len() >= min_chars]
    print(f"[clean] Very short tweets removed: {before - len(df):,}  →  {len(df):,} rows remain")
    return df


def main():
    df = load(INPUT_FILE)

    df = filter_brand(df, BRAND_HANDLE)
    df = remove_duplicates(df)
    df = remove_malformed(df)
    df = tag_role(df, BRAND_HANDLE)
    df = apply_normalisation(df)
    df = filter_very_short(df)

    # final stats
    brand_rows    = (df["role"] == "brand").sum()
    customer_rows = (df["role"] == "customer").sum()
    print(f"\n[clean] Final dataset: {len(df):,} rows")
    print(f"  Brand tweets:    {brand_rows:,}")
    print(f"  Customer tweets: {customer_rows:,}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[clean] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()