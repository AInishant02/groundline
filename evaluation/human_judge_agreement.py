"""
human_judge_agreement.py — compares human scores vs LLM judge scores.
Reports agreement rate, correlation, Cohen's kappa, mean absolute difference.

Usage:
    # Step 1: generate the human scoring sheet
    python -m evaluation.human_judge_agreement --generate

    # Step 2: after filling in human scores, run the analysis
    python -m evaluation.human_judge_agreement --analyze
"""

import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

REPLY_EVAL_FILE   = Path("data/processed/evaluation/reply_eval.csv")
HUMAN_SHEET_FILE  = Path("evaluation/human_scoring_sheet.csv")
ANALYSIS_FILE     = Path("data/processed/evaluation/human_agreement.json")
JUDGE_MODEL       = "gemini-3.5-flash-lite"
SAMPLE_SIZE       = 30   # use the 30 we already evaluated — no extra API calls needed
SEED              = 42

DIMENSIONS = [
    "groundedness",
    "correctness",
    "helpfulness",
    "safety",
    "brand_consistency",
    "conciseness",
    "overall",
]

SCORING_GUIDE = """
SCORING GUIDE (1-5 scale):
1 = unacceptable
2 = major problems
3 = acceptable but imperfect
4 = good
5 = excellent

DIMENSIONS:
- groundedness:       Is the reply supported by retrieved historical evidence?
- correctness:        Does it appropriately address the customer's actual issue?
- helpfulness:        Would this move the customer toward resolution?
- safety:             Does it avoid unsupported claims, promises, or invented policies?
- brand_consistency:  Does it match SpotifyCares' historical tone and style?
- conciseness:        Is it appropriately brief for Twitter/X support?
- overall:            Your overall impression of the reply quality
"""


def generate_scoring_sheet():
    """Load the LLM-evaluated replies and create a sheet for human scoring."""
    if not REPLY_EVAL_FILE.exists():
        print(f"[human] ERROR: {REPLY_EVAL_FILE} not found.")
        print("  Run `python -m evaluation.evaluate_replies` first.")
        return

    df = pd.read_csv(REPLY_EVAL_FILE)
    print(f"[human] Loaded {len(df)} LLM-evaluated replies")

    # build human scoring sheet
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "id":               row["id"],
            "intent_label":     row["intent_label"],
            "customer_message": row["customer_message"],
            "draft_reply":      row["draft_reply"],
            # LLM scores (for reference AFTER human scoring)
            "llm_groundedness":      row.get("score_groundedness", ""),
            "llm_correctness":       row.get("score_correctness", ""),
            "llm_helpfulness":       row.get("score_helpfulness", ""),
            "llm_safety":            row.get("score_safety", ""),
            "llm_brand_consistency": row.get("score_brand_consistency", ""),
            "llm_conciseness":       row.get("score_conciseness", ""),
            "llm_overall":           row.get("score_overall", ""),
            # human scores — YOU fill these in
            "human_groundedness":      "",
            "human_correctness":       "",
            "human_helpfulness":       "",
            "human_safety":            "",
            "human_brand_consistency": "",
            "human_conciseness":       "",
            "human_overall":           "",
            "human_notes":             "",
        })

    sheet = pd.DataFrame(rows)
    sheet.to_csv(HUMAN_SHEET_FILE, index=False)

    print(f"\n[human] Scoring sheet saved to {HUMAN_SHEET_FILE}")
    print(f"\n{SCORING_GUIDE}")
    print("="*60)
    print("INSTRUCTIONS:")
    print("1. Open evaluation/human_scoring_sheet.csv in Excel or VS Code")
    print("2. For each row, read the customer_message and draft_reply")
    print("3. Fill in human_groundedness through human_overall (1-5)")
    print("4. DO NOT look at the llm_* columns until you're done scoring")
    print("5. Add any notes in human_notes column")
    print("6. Save the file")
    print("7. Run: python -m evaluation.human_judge_agreement --analyze")
    print("="*60)


def analyze():
    """Compare human scores vs LLM judge scores."""
    if not HUMAN_SHEET_FILE.exists():
        print(f"[human] ERROR: {HUMAN_SHEET_FILE} not found.")
        print("  Run --generate first, fill in scores, then run --analyze.")
        return

    df = pd.read_csv(HUMAN_SHEET_FILE)

    # check human scores are filled in
    human_cols = [f"human_{d}" for d in DIMENSIONS]
    llm_cols   = [f"llm_{d}"   for d in DIMENSIONS]

    missing = df[human_cols].isnull().any().any() or (df[human_cols] == "").any().any()
    if missing:
        filled = df[human_cols].replace("", np.nan).notna().all(axis=1).sum()
        print(f"[human] WARNING: Only {filled}/{len(df)} rows fully scored.")
        df = df[df[human_cols].replace("", np.nan).notna().all(axis=1)].copy()
        print(f"[human] Analyzing {len(df)} complete rows.")

    for col in human_cols + llm_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=human_cols + llm_cols)
    print(f"\n[human] Analyzing {len(df)} fully scored examples...")

    results = {}
    print("\n" + "="*60)
    print("HUMAN vs LLM JUDGE AGREEMENT")
    print("="*60)
    print(f"{'Dimension':<22} {'MAD':>6} {'Pearson':>8} {'Spearman':>9} {'Kappa':>7} {'Agree%':>7}")
    print("-"*60)

    for dim in DIMENSIONS:
        human = df[f"human_{dim}"].values
        llm   = df[f"llm_{dim}"].values

        mad       = float(np.mean(np.abs(human - llm)))
        pearson_r = pearsonr(human, llm)[0] if len(set(human)) > 1 else 0.0
        spearman_r = spearmanr(human, llm)[0] if len(set(human)) > 1 else 0.0

        # Cohen's kappa — round to integers for ordinal agreement
        try:
            kappa = cohen_kappa_score(
                human.astype(int),
                llm.astype(int),
                labels=[1, 2, 3, 4, 5]
            )
        except Exception:
            kappa = 0.0

        agree_rate = float(np.mean(human == llm))

        results[dim] = {
            "mad":        round(mad, 3),
            "pearson_r":  round(pearson_r, 3),
            "spearman_r": round(spearman_r, 3),
            "kappa":      round(kappa, 3),
            "agree_rate": round(agree_rate, 3),
        }

        print(
            f"{dim:<22} {mad:>6.3f} {pearson_r:>8.3f} "
            f"{spearman_r:>9.3f} {kappa:>7.3f} {agree_rate:>6.1%}"
        )

    # overall summary
    avg_mad       = np.mean([r["mad"]        for r in results.values()])
    avg_pearson   = np.mean([r["pearson_r"]  for r in results.values()])
    avg_kappa     = np.mean([r["kappa"]      for r in results.values()])
    avg_agree     = np.mean([r["agree_rate"] for r in results.values()])

    print("-"*60)
    print(f"{'AVERAGE':<22} {avg_mad:>6.3f} {avg_pearson:>8.3f} {'':>9} {avg_kappa:>7.3f} {avg_agree:>6.1%}")

    print(f"\n[human] Kappa interpretation:")
    print(f"  < 0.20  = slight agreement")
    print(f"  0.20-0.40 = fair agreement")
    print(f"  0.40-0.60 = moderate agreement")
    print(f"  0.60-0.80 = substantial agreement")
    print(f"  > 0.80  = almost perfect agreement")
    print(f"\n  Average kappa: {avg_kappa:.3f}")

    # disagreement analysis
    print(f"\n--- Cases where human and LLM disagree by >= 2 points ---")
    for dim in DIMENSIONS:
        diff = np.abs(df[f"human_{dim}"] - df[f"llm_{dim}"])
        big_disagreements = df[diff >= 2]
        if len(big_disagreements) > 0:
            print(f"\n  {dim} ({len(big_disagreements)} large disagreements):")
            for _, row in big_disagreements.head(2).iterrows():
                print(f"    Message: {str(row['customer_message'])[:70]}...")
                print(f"    Reply:   {str(row['draft_reply'])[:70]}...")
                print(f"    Human: {int(row[f'human_{dim}'])}  LLM: {int(row[f'llm_{dim}'])}")

    # save results
    output = {
        "n_examples":   len(df),
        "avg_mad":      round(float(avg_mad), 3),
        "avg_pearson":  round(float(avg_pearson), 3),
        "avg_kappa":    round(float(avg_kappa), 3),
        "avg_agree":    round(float(avg_agree), 3),
        "per_dimension": results,
        "conclusion":   (
            "LLM judge is trustworthy" if avg_kappa >= 0.4
            else "LLM judge has moderate disagreement with human — treat scores with caution"
        ),
    }

    with open(ANALYSIS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[human] Results saved to {ANALYSIS_FILE}")
    print(f"\n[human] CONCLUSION: {output['conclusion']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate the human scoring sheet")
    parser.add_argument("--analyze",  action="store_true",
                        help="Analyze human vs LLM scores")
    args = parser.parse_args()

    if args.generate:
        generate_scoring_sheet()
    elif args.analyze:
        analyze()
    else:
        print("Usage:")
        print("  python -m evaluation.human_judge_agreement --generate")
        print("  python -m evaluation.human_judge_agreement --analyze")


if __name__ == "__main__":
    main()