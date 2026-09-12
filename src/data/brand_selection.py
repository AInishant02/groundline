"""
brand_selection.py — inspects the raw dataset, builds the brand comparison
table, and documents the selected brand with measurable criteria.

Usage:
    python -m src.data.brand_selection
"""

import pandas as pd
import yaml
from pathlib import Path

RAW_DIR   = Path("data/raw")
OUT_DIR   = Path("data/processed")
CONFIG    = Path("config.yaml")

MIN_CONVERSATIONS   = 300
MIN_CUSTOMER_TWEETS = 200
MIN_BRAND_REPLIES   = 200


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/ — run `python -m src.data.download` first."
        )
    path = csv_files[0]
    print(f"[brand_selection] Loading {path.name} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"  Rows: {len(df):,}   Columns: {list(df.columns)}")
    return df


def inspect_schema(df: pd.DataFrame):
    print("\n--- Schema ---")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:35s}  non-null: {non_null:,} / {len(df):,}")


def build_comparison_table(df: pd.DataFrame):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower()
    customer_mask = df["inbound"] == "true"
    brand_mask    = df["inbound"] == "false"

    brand_tweets    = df[brand_mask]
    customer_tweets = df[customer_mask]

    brand_counts     = brand_tweets["author_id"].value_counts()
    candidate_brands = brand_counts[brand_counts >= MIN_BRAND_REPLIES].index.tolist()

    rows = []
    for brand in candidate_brands:
        b_replies  = brand_tweets[brand_tweets["author_id"] == brand]
        replied_to = set(b_replies["in_response_to_tweet_id"].dropna().unique())
        c_tweets   = customer_tweets[customer_tweets["tweet_id"].isin(replied_to)]

        n_conversations = len(replied_to)
        n_customer      = len(c_tweets)
        n_brand         = len(b_replies)
        avg_len         = round(n_customer / max(n_conversations, 1), 2)

        rows.append({
            "Brand":                   brand,
            "Conversations":           n_conversations,
            "Customer Tweets":         n_customer,
            "Brand Replies":           n_brand,
            "Avg Conversation Length": avg_len,
        })

    table = (
        pd.DataFrame(rows)
        .sort_values("Conversations", ascending=False)
        .reset_index(drop=True)
    )
    return table, df


def select_brand(table: pd.DataFrame) -> str:
    qualified = table[
        (table["Conversations"]   >= MIN_CONVERSATIONS) &
        (table["Customer Tweets"] >= MIN_CUSTOMER_TWEETS) &
        (table["Brand Replies"]   >= MIN_BRAND_REPLIES)
    ].copy()

    if qualified.empty:
        print("[brand_selection] WARNING: No brand meets all thresholds — using top 10.")
        qualified = table.head(10)

    qualified["in_sweet_spot"] = qualified["Avg Conversation Length"].between(1.5, 4.0)
    qualified = qualified.sort_values(
        ["in_sweet_spot", "Brand Replies"], ascending=[False, False]
    )

    return qualified.iloc[0]["Brand"]


def write_selection(brand: str, row: pd.Series):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["brand"]["name"]   = brand
    cfg["brand"]["handle"] = brand

    with open(CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  SELECTED BRAND: {brand}")
    print(f"{'='*60}")
    print(f"  Conversations:            {row['Conversations']:,}")
    print(f"  Customer Tweets:          {row['Customer Tweets']:,}")
    print(f"  Brand Replies:            {row['Brand Replies']:,}")
    print(f"  Avg Conversation Length:  {row['Avg Conversation Length']}")
    print(f"\n  config.yaml updated with brand.name = '{brand}'")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    inspect_schema(df)

    print("\n[brand_selection] Building comparison table ...")
    table, df_clean = build_comparison_table(df)

    print("\n--- Brand Comparison Table (top 20) ---")
    print(table.head(20).to_string(index=False))

    table_path = OUT_DIR / "brand_comparison.csv"
    table.to_csv(table_path, index=False)
    print(f"\n[brand_selection] Full table saved to {table_path}")

    selected = select_brand(table)
    row = table[table["Brand"] == selected].iloc[0]
    write_selection(selected, row)

    df_clean.to_csv(OUT_DIR / "tweets_normalised.csv", index=False)
    print(f"[brand_selection] Normalised tweets saved to data/processed/tweets_normalised.csv")


if __name__ == "__main__":
    main()