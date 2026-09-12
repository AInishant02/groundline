"""
evaluate_escalation.py — evaluates the escalation policy on the golden set.
Reports precision, recall, false-auto rate, and confusion matrix.

Usage:
    python -m evaluation.evaluate_escalation
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.agent import GroundlineAgent
import src.intent.classifier 

GOLDEN_FILE = Path("evaluation/golden_set.csv")
OUTPUT_DIR  = Path("data/processed/evaluation")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[evaluate_escalation] Loading golden set ...")
    golden = pd.read_csv(GOLDEN_FILE, dtype=str)
    golden["should_escalate"] = golden["should_escalate"].str.lower() == "true"
    print(f"  Examples: {len(golden)}")
    print(f"  True escalations: {golden['should_escalate'].sum()}")
    print(f"  True auto-handle: {(~golden['should_escalate']).sum()}")

    agent = GroundlineAgent()

    print("\n[evaluate_escalation] Running pipeline on golden set ...")
    print("  (This makes one Gemini call per example — may take a few minutes)")

    predictions = []
    for i, row in golden.iterrows():
        print(f"  [{i+1}/{len(golden)}] {row['customer_message'][:55]}...")
        try:
            result = agent.run(
                row["customer_message"],
                row.get("conversation_context", ""),
            )
            pred_escalate = result["decision"] == "ESCALATE"
        except Exception as e:
            print(f"    ERROR: {e} — defaulting to ESCALATE")
            pred_escalate = True

        predictions.append({
            "id":               row["id"],
            "true_escalate":    row["should_escalate"],
            "pred_escalate":    pred_escalate,
            "intent_label":     row["intent_label"],
            "customer_message": row["customer_message"],
            "decision":         "ESCALATE" if pred_escalate else "AUTO_HANDLE",
        })

    df = pd.DataFrame(predictions)

    true_labels = df["true_escalate"].astype(int)
    pred_labels = df["pred_escalate"].astype(int)

    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall    = recall_score(true_labels, pred_labels, zero_division=0)
    f1        = f1_score(true_labels, pred_labels, zero_division=0)
    cm        = confusion_matrix(true_labels, pred_labels)

    # confusion matrix layout:
    # TN = auto-handle correct, FP = over-escalation
    # FN = FALSE AUTO (most dangerous!), TP = correct escalation
    tn, fp, fn, tp = cm.ravel()
    false_auto_rate   = fn / (fn + tn) if (fn + tn) > 0 else 0
    auto_handle_rate  = (tn + fp) / len(df)

    print("\n" + "="*60)
    print("ESCALATION EVALUATION")
    print("="*60)
    print(f"  Escalation Precision:  {precision:.4f}  (of escalated, how many needed it)")
    print(f"  Escalation Recall:     {recall:.4f}  (of those needing it, how many caught)")
    print(f"  Escalation F1:         {f1:.4f}")
    print(f"  False-Auto Rate:       {false_auto_rate:.4f}  *** most important ***")
    print(f"  Auto-Handling Rate:    {auto_handle_rate:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':20s} {'Pred: AUTO':>12} {'Pred: ESCALATE':>15}")
    print(f"  {'True: AUTO':20s} {tn:>12} {fp:>15}  (over-escalation: {fp})")
    print(f"  {'True: ESCALATE':20s} {fn:>12} {tp:>15}  (FALSE AUTO: {fn} ← critical)")

    results = {
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "false_auto_rate":  round(false_auto_rate, 4),
        "auto_handle_rate": round(auto_handle_rate, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                              "fn": int(fn), "tp": int(tp)},
    }

    out_json = OUTPUT_DIR / "escalation_eval.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_csv = OUTPUT_DIR / "escalation_predictions.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n[evaluate_escalation] Results saved to {out_json}")
    print(f"[evaluate_escalation] Predictions saved to {out_csv}")


if __name__ == "__main__":
    main()