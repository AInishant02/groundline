"""
evaluate_intent.py — evaluates intent classification on the golden set.
Reports accuracy, macro F1, weighted F1, per-intent P/R/F1, confusion matrix.

Usage:
    python -m evaluation.evaluate_intent
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from src.intent.classifier import TFIDFClassifier, EmbeddingClassifier, load_taxonomy

GOLDEN_FILE   = Path("evaluation/golden_set.csv")
MODELS_DIR    = Path("data/processed/models")
TAXONOMY_PATH = Path("src/taxonomy/taxonomy.yaml")
OUTPUT_DIR    = Path("data/processed/evaluation")


def evaluate(clf, name: str, golden: pd.DataFrame) -> dict:
    print(f"\n[evaluate_intent] Evaluating: {name}")
    preds = [clf.predict(msg)["intent"] for msg in golden["customer_message"]]
    true  = golden["intent_label"].tolist()

    acc        = accuracy_score(true, preds)
    report     = classification_report(true, preds, zero_division=0, output_dict=True)
    cm         = confusion_matrix(true, preds, labels=sorted(set(true)))
    labels     = sorted(set(true))

    macro_f1    = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\n  Per-intent report:")
    print(classification_report(true, preds, zero_division=0))

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    return {
        "classifier":   name,
        "accuracy":     round(acc, 4),
        "macro_f1":     round(macro_f1, 4),
        "weighted_f1":  round(weighted_f1, 4),
        "per_intent":   {
            k: v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": cm.tolist(),
        "labels":           labels,
        "predictions":      preds,
        "true_labels":      true,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[evaluate_intent] Loading golden set ...")
    golden = pd.read_csv(GOLDEN_FILE, dtype=str)
    print(f"  Examples: {len(golden)}")

    # load classifiers
    clf_tfidf = TFIDFClassifier.load(MODELS_DIR)
    taxonomy  = load_taxonomy(TAXONOMY_PATH)
    clf_emb   = EmbeddingClassifier(taxonomy)

    # evaluate both
    results_tfidf = evaluate(clf_tfidf, "tfidf_logreg", golden)
    results_emb   = evaluate(clf_emb,   "embedding_similarity", golden)

    # save results
    for name, res in [("tfidf", results_tfidf), ("embedding", results_emb)]:
        out = OUTPUT_DIR / f"intent_eval_{name}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[evaluate_intent] Results saved to {out}")

    # summary scorecard
    print("\n" + "="*60)
    print("INTENT CLASSIFICATION SCORECARD")
    print("="*60)
    print(f"{'Metric':<20} {'TF-IDF+LR':>12} {'Embedding':>12}")
    print("-"*46)
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        print(
            f"{metric:<20} "
            f"{results_tfidf[metric]:>12.4f} "
            f"{results_emb[metric]:>12.4f}"
        )


if __name__ == "__main__":
    main()