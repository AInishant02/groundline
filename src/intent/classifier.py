"""
classifier.py — intent classification for SpotifyCares support messages.

Two classifiers:
  1. TF-IDF + Logistic Regression (baseline)
  2. Embedding similarity against labelled intent examples (main)

Confidence is derived from real signals:
  - LogReg: max class probability
  - Embedding: similarity score + margin between top-2 candidates

Usage:
    python -m src.intent.classifier
"""

import yaml
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

TAXONOMY_PATH  = Path("src/taxonomy/taxonomy.yaml")
THREADS_FILE   = Path("data/processed/threads.jsonl")
GOLDEN_FILE    = Path("evaluation/golden_set.csv")
MODELS_DIR     = Path("data/processed/models")
MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
SEED           = 42
TRAIN_SAMPLE   = 8000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_threads(path: Path, n: int = TRAIN_SAMPLE) -> list:
    import json
    threads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    random.seed(SEED)
    random.shuffle(threads)
    return threads[:n]


def load_golden(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


# ---------------------------------------------------------------------------
# Keyword labeller
# ---------------------------------------------------------------------------

INTENT_SIGNALS = {
    "playback_issue": [
        "won't play", "not playing", "can't play", "skipping", "buffering",
        "crashes", "crashing", "stops", "freezes", "silent", "no sound",
        "playback", "keeps stopping", "wont play"
    ],
    "offline_download_issue": [
        "offline", "download", "downloaded", "downloading", "cant download",
        "won't download", "downloads gone", "offline mode"
    ],
    "subscription_billing": [
        "charged", "charge", "billing", "payment", "invoice", "price",
        "cost", "fee", "premium", "paid", "paying", "bill", "refund",
        "overcharged", "double charged", "student discount", "discount"
    ],
    "cancellation_request": [
        "cancel", "cancellation", "unsubscribe", "stop subscription",
        "end subscription", "cancel premium", "cancel my account"
    ],
    "account_access": [
        "log in", "login", "can't log", "password", "forgot password",
        "reset password", "locked out", "sign in", "cant sign", "access"
    ],
    "account_security": [
        "hacked", "hack", "someone else", "unauthorized", "compromised",
        "strange device", "not me", "account stolen", "security"
    ],
    "account_closure": [
        "delete account", "close account", "remove account",
        "delete my data", "gdpr", "permanently delete",
        "remove my account", "deactivate", "erase my account",
    ],
    "content_not_found": [
        "can't find", "not available", "missing", "removed", "album",
        "artist", "song not", "podcast", "disappeared", "no longer",
        "why isn't", "where is"
    ],
    "app_feature_question": [
        "how do i", "how to", "how can i", "crossfade", "equalizer",
        "collaborate", "share playlist", "connect", "feature", "settings",
        "enable", "disable", "where is the"
    ],
    "technical_issue": [
        "widget", "sync", "connect", "chromecast", "alexa", "speaker",
        "glitch", "bug", "ui", "interface", "update", "iphone", "android",
        "not syncing", "not working", "broken"
    ],
    "complaint_feedback": [
        "terrible", "awful", "worst", "hate", "disgusting", "unacceptable",
        "disappointed", "frustrated", "feedback", "suggestion", "please add",
        "bring back", "you removed", "new ui", "new update ruined"
    ],
    "other_unknown": [
        "???", "help me", "hi ", "hey ", "hello", "just wondering",
        "quick question", "anyone", "wtf", "lol", "seriously",
        "what is going on", "what happened", "why is", "why are",
        "not sure", "i dont know", "random", "weird", "strange"
    ],
}


def keyword_label(text: str) -> str:
    text_lower = text.lower()
    scores = {
        intent: sum(1 for s in signals if s in text_lower)
        for intent, signals in INTENT_SIGNALS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other_unknown"


# ---------------------------------------------------------------------------
# Baseline: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

class TFIDFClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=SEED,
            C=1.0,
            class_weight="balanced",
        )
        self.encoder = LabelEncoder()
        self.fitted  = False

    def fit(self, texts: list, labels: list):
        print("[tfidf] Fitting TF-IDF + LogReg baseline ...")
        y = self.encoder.fit_transform(labels)
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)
        self.fitted = True
        print(f"  Classes: {list(self.encoder.classes_)}")

    def predict(self, text: str) -> dict:
        X      = self.vectorizer.transform([text])
        proba  = self.model.predict_proba(X)[0]
        top_idx = np.argmax(proba)
        top2    = np.argsort(proba)[-2:][::-1]
        margin  = proba[top2[0]] - proba[top2[1]]
        intent  = self.encoder.classes_[top_idx]

        return {
            "intent":     intent,
            "confidence": round(float(proba[top_idx]), 4),
            "margin":     round(float(margin), 4),
            "all_scores": {
                cls: round(float(p), 4)
                for cls, p in zip(self.encoder.classes_, proba)
            },
            "classifier": "tfidf_logreg",
        }

    def save(self, path: Path):
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path / "tfidf_vectorizer.joblib")
        joblib.dump(self.model,      path / "tfidf_model.joblib")
        joblib.dump(self.encoder,    path / "tfidf_encoder.joblib")
        print(f"[tfidf] Model saved to {path}/tfidf_*.joblib")

    @classmethod
    def load(cls, path: Path) -> "TFIDFClassifier":
        import joblib
        obj = cls()
        obj.vectorizer = joblib.load(path / "tfidf_vectorizer.joblib")
        obj.model      = joblib.load(path / "tfidf_model.joblib")
        obj.encoder    = joblib.load(path / "tfidf_encoder.joblib")
        obj.fitted     = True
        return obj


# ---------------------------------------------------------------------------
# Main: Embedding similarity classifier
# ---------------------------------------------------------------------------

class EmbeddingClassifier:
    def __init__(self, taxonomy: dict):
        self.model    = SentenceTransformer(MODEL_NAME)
        self.taxonomy = taxonomy
        self.intent_embeddings = {}
        self._build_prototypes()

    def _build_prototypes(self):
        print("[embedding] Building intent prototypes from taxonomy examples ...")
        for intent in self.taxonomy["intents"]:
            name     = intent["name"]
            examples = intent.get("positive_examples", [])
            if not examples:
                continue
            embeddings = self.model.encode(
                examples,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.intent_embeddings[name] = embeddings.mean(axis=0)
        print(f"  Prototypes built for {len(self.intent_embeddings)} intents")

    def predict(self, text: str) -> dict:
        query_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = {
            name: float(np.dot(query_emb, proto))
            for name, proto in self.intent_embeddings.items()
        }

        sorted_intents = sorted(scores.items(), key=lambda x: -x[1])
        top_intent, top_score = sorted_intents[0]
        second_score = sorted_intents[1][1] if len(sorted_intents) > 1 else 0.0
        margin       = top_score - second_score
        confidence   = round((top_score * 0.7 + margin * 0.3), 4)

        return {
            "intent":     top_intent,
            "confidence": confidence,
            "top_score":  round(top_score, 4),
            "margin":     round(margin, 4),
            "all_scores": {k: round(v, 4) for k, v in sorted_intents},
            "classifier": "embedding_similarity",
        }


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_baseline(threads: list) -> TFIDFClassifier:
    texts  = [t["customer_opening"] for t in threads if t["customer_opening"].strip()]
    labels = [keyword_label(t) for t in texts]
    clf    = TFIDFClassifier()
    clf.fit(texts, labels)
    clf.save(MODELS_DIR)
    return clf


def evaluate_on_golden(clf_tfidf: TFIDFClassifier,
                        clf_emb: EmbeddingClassifier,
                        golden: pd.DataFrame):
    print("\n--- Quick evaluation on golden set ---")
    results = []
    for _, row in golden.iterrows():
        msg   = row["customer_message"]
        label = row["intent_label"]
        pred_tfidf = clf_tfidf.predict(msg)
        pred_emb   = clf_emb.predict(msg)
        results.append({
            "true":       label,
            "pred_tfidf": pred_tfidf["intent"],
            "pred_emb":   pred_emb["intent"],
            "conf_tfidf": pred_tfidf["confidence"],
            "conf_emb":   pred_emb["confidence"],
        })

    df = pd.DataFrame(results)
    acc_tfidf = (df["true"] == df["pred_tfidf"]).mean()
    acc_emb   = (df["true"] == df["pred_emb"]).mean()

    print(f"\n  TF-IDF + LogReg accuracy:      {acc_tfidf:.3f}")
    print(f"  Embedding similarity accuracy: {acc_emb:.3f}")
    print("\n--- TF-IDF Classification Report ---")
    print(classification_report(df["true"], df["pred_tfidf"], zero_division=0))
    print("\n--- Embedding Classification Report ---")
    print(classification_report(df["true"], df["pred_emb"], zero_division=0))
    return df


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("[classifier] Loading taxonomy ...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)

    print("[classifier] Loading training threads ...")
    threads = load_threads(THREADS_FILE)

    print("[classifier] Loading golden set ...")
    golden = load_golden(GOLDEN_FILE)

    clf_tfidf = train_baseline(threads)

    print("\n[classifier] Building embedding classifier ...")
    clf_emb = EmbeddingClassifier(taxonomy)

    results_df = evaluate_on_golden(clf_tfidf, clf_emb, golden)

    results_path = Path("data/processed/classifier_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[classifier] Results saved to {results_path}")

    with open(MODELS_DIR / "embedding_classifier.pkl", "wb") as f:
        pickle.dump(clf_emb, f)
    print(f"[classifier] Embedding classifier saved to {MODELS_DIR}/embedding_classifier.pkl")


if __name__ == "__main__":
    main()