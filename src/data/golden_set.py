"""
golden_set.py — builds a stratified golden evaluation set of ~200 examples
from SpotifyCares threads. Kept strictly separate from the retrieval index.

Usage:
    python -m src.data.golden_set
"""

import json
import random
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
GOLDEN_DIR    = Path("data/golden")
THREADS_FILE  = PROCESSED_DIR / "threads.jsonl"
OUTPUT_FILE   = Path("evaluation/golden_set.csv")
SEED          = 42
TARGET        = 200

STRATA = {
    "playback_issue":         18,
    "subscription_billing":   18,
    "account_access":         18,
    "content_not_found":      18,
    "app_feature_question":   18,
    "technical_issue":        18,
    "cancellation_request":   16,
    "complaint_feedback":     16,
    "offline_download_issue": 16,
    "account_security":       14,
    "account_closure":        14,
    "other_unknown":          20,
}

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
        "get rid of my account", "remove all my data"
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

ALWAYS_ESCALATE_SIGNALS = [
    "hacked", "hack", "someone else", "unauthorized", "compromised",
    "account stolen", "legal", "lawyer", "sue", "lawsuit", "gdpr",
    "delete my data", "harassment", "abuse", "double charged",
    "overcharged", "refund", "charged twice", "charged again",
    "still charged", "cancel", "cancelling", "cancelled",
    "won't let me cancel", "can't cancel", "delete account",
    "close account", "remove account", "security", "stolen",
    "not authorised", "not authorized"
]

DIFFICULTY_SIGNALS = {
    "hard": [
        "???", "help", "hi ", "hey ", "hello", "just", "wtf",
        "idk", "not sure", "maybe", "kind of", "sort of"
    ]
}


def load_threads(path: Path) -> list:
    threads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    return threads


def get_context(thread: dict) -> str:
    msgs = thread.get("messages", [])
    context_parts = []
    for m in msgs[:4]:
        role = "Customer" if m["role"] == "customer" else "Spotify"
        context_parts.append(f"{role}: {m['text_clean'][:120]}")
    return " | ".join(context_parts)


def classify_intent(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for intent, signals in INTENT_SIGNALS.items():
        scores[intent] = sum(1 for s in signals if s in text_lower)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other_unknown"
    return best


def should_escalate(text: str, intent: str) -> bool:
    text_lower = text.lower()
    if intent in ("account_security", "other_unknown"):
        return True
    if any(s in text_lower for s in ALWAYS_ESCALATE_SIGNALS):
        return True
    return False


def get_difficulty(text: str, intent: str) -> str:
    text_lower = text.lower()
    if len(text.strip()) < 40:
        return "hard"
    if any(s in text_lower for s in DIFFICULTY_SIGNALS["hard"]):
        return "medium"
    if intent == "other_unknown":
        return "hard"
    return "easy"


def get_expected_reason(intent: str, escalate: bool) -> str:
    if not escalate:
        return f"Standard {intent.replace('_', ' ')} — auto-handle with evidence"
    reasons = {
        "account_security":     "Account compromise — always requires human review",
        "subscription_billing": "Financial dispute or unexpected charge — human review needed",
        "cancellation_request": "Post-cancellation charge dispute — human review needed",
        "account_closure":      "Data deletion / GDPR request — DPO involvement required",
        "other_unknown":        "Intent unclear — human triage required",
    }
    return reasons.get(intent, "Sensitive issue or low confidence — escalate for human review")


def get_reference_resolution(intent: str) -> str:
    resolutions = {
        "playback_issue":         "Ask for device/OS/version; suggest restart, reinstall, cache clear",
        "offline_download_issue": "Check storage; re-enable offline mode; re-download playlist",
        "subscription_billing":   "Direct to account billing page; explain charge or pricing change",
        "cancellation_request":   "Direct to cancellation flow in account settings",
        "account_access":         "Direct to password reset; verify email",
        "account_security":       "ESCALATE — security team review required",
        "account_closure":        "Direct to account deletion page; confirm data implications",
        "content_not_found":      "Explain licensing/regional availability",
        "app_feature_question":   "Provide how-to guidance or help article link",
        "technical_issue":        "Gather device details; standard troubleshooting steps",
        "complaint_feedback":     "Acknowledge; direct to feedback channel; no promises",
        "other_unknown":          "ESCALATE — ask clarifying question or human triage",
    }
    return resolutions.get(intent, "Human review required")


def main():
    random.seed(SEED)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("[golden_set] Loading threads ...")
    threads = load_threads(THREADS_FILE)
    print(f"  Threads: {len(threads):,}")

    pool = []
    for t in threads:
        msg = t["customer_opening"]
        if not msg.strip():
            continue
        intent     = classify_intent(msg)
        escalate   = should_escalate(msg, intent)
        difficulty = get_difficulty(msg, intent)
        pool.append({
            "thread_id":   t["thread_id"],
            "message":     msg,
            "context":     get_context(t),
            "intent":      intent,
            "escalate":    escalate,
            "difficulty":  difficulty,
            "brand_reply": t["first_brand_reply"],
        })

    df_pool = pd.DataFrame(pool)
    print(f"[golden_set] Pool size: {len(df_pool):,}")

    rows = []
    example_id = 1

    for intent, n in STRATA.items():
        subset = df_pool[df_pool["intent"] == intent]
        if len(subset) == 0:
            print(f"  WARNING: no examples for intent '{intent}'")
            continue

        easy   = subset[subset["difficulty"] == "easy"]
        medium = subset[subset["difficulty"] == "medium"]
        hard   = subset[subset["difficulty"] == "hard"]

        n_hard   = max(2, n // 5)
        n_medium = max(2, n // 5)
        n_easy   = n - n_hard - n_medium

        sampled = pd.concat([
            easy.sample(min(n_easy, len(easy)),       random_state=SEED),
            medium.sample(min(n_medium, len(medium)), random_state=SEED),
            hard.sample(min(n_hard, len(hard)),       random_state=SEED),
        ]).head(n)

        for _, row in sampled.iterrows():
            rows.append({
                "id":                   f"GS{example_id:04d}",
                "customer_message":     row["message"],
                "conversation_context": row["context"],
                "intent_label":         row["intent"],
                "should_escalate":      row["escalate"],
                "expected_reason":      get_expected_reason(row["intent"], row["escalate"]),
                "reference_resolution": get_reference_resolution(row["intent"]),
                "difficulty":           row["difficulty"],
            })
            example_id += 1

    golden = pd.DataFrame(rows)

    print(f"\n[golden_set] Examples sampled: {len(golden)}")
    print(f"\n--- Intent distribution ---")
    print(golden["intent_label"].value_counts().to_string())
    print(f"\n--- Difficulty distribution ---")
    print(golden["difficulty"].value_counts().to_string())
    print(f"\n--- Escalation distribution ---")
    print(golden["should_escalate"].value_counts().to_string())

    golden.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[golden_set] Saved to {OUTPUT_FILE}")
    print("\nIMPORTANT: Review evaluation/golden_set.csv manually.")
    print("Fix any obviously wrong intent labels before Phase 6.")


if __name__ == "__main__":
    main()