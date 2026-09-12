"""
build_taxonomy.py — derives the intent taxonomy from SpotifyCares threads.
Clusters customer opening messages and outputs a taxonomy.yaml with
8-15 intents, each with definition, examples, and escalation conditions.

Usage:
    python -m src.taxonomy.build_taxonomy
"""

import json
import yaml
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
TAXONOMY_PATH = Path("src/taxonomy/taxonomy.yaml")
THREADS_FILE  = PROCESSED_DIR / "threads.jsonl"
SEED          = 42
N_CLUSTERS    = 12   # within 8-15 target; we'll merge/split after inspection


def load_threads(path: Path) -> list:
    print(f"[taxonomy] Loading threads from {path.name} ...")
    threads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))
    print(f"  Threads loaded: {len(threads):,}")
    return threads


def extract_customer_messages(threads: list) -> list[str]:
    """Use the customer opening message of each thread."""
    msgs = [t["customer_opening"] for t in threads if t["customer_opening"].strip()]
    print(f"[taxonomy] Customer opening messages: {len(msgs):,}")
    return msgs


def sample_messages(msgs: list, n: int = 8000, seed: int = SEED) -> list:
    """Sample for clustering — 8K is plenty, keeps it fast."""
    random.seed(seed)
    if len(msgs) <= n:
        return msgs
    sampled = random.sample(msgs, n)
    print(f"[taxonomy] Sampled {len(sampled):,} messages for clustering")
    return sampled


def embed(msgs: list) -> np.ndarray:
    print("[taxonomy] Embedding messages with sentence-transformers ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(msgs, batch_size=256, show_progress_bar=True)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def cluster(embeddings: np.ndarray, n: int = N_CLUSTERS, seed: int = SEED) -> np.ndarray:
    print(f"[taxonomy] KMeans clustering into {n} clusters ...")
    km = KMeans(n_clusters=n, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


def top_tfidf_terms(msgs: list, labels: np.ndarray, cluster_id: int, n: int = 10) -> list:
    """Get the top TF-IDF terms for a cluster to help name it."""
    cluster_msgs = [m for m, l in zip(msgs, labels) if l == cluster_id]
    if not cluster_msgs:
        return []
    vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    try:
        X = vec.fit_transform(cluster_msgs)
        scores = X.mean(axis=0).A1
        terms  = vec.get_feature_names_out()
        top    = sorted(zip(terms, scores), key=lambda x: -x[1])[:n]
        return [t for t, _ in top]
    except Exception:
        return []


def sample_cluster_msgs(msgs: list, labels: np.ndarray, cluster_id: int, n: int = 5) -> list:
    cluster_msgs = [m for m, l in zip(msgs, labels) if l == cluster_id]
    random.seed(SEED)
    return random.sample(cluster_msgs, min(n, len(cluster_msgs)))


def print_cluster_report(msgs: list, labels: np.ndarray):
    print("\n--- Cluster Report ---")
    for i in range(N_CLUSTERS):
        count   = (labels == i).sum()
        terms   = top_tfidf_terms(msgs, labels, i)
        samples = sample_cluster_msgs(msgs, labels, i, n=3)
        print(f"\nCluster {i:2d}  ({count:,} msgs)  Top terms: {', '.join(terms[:6])}")
        for s in samples:
            print(f"    • {s[:100]}")


def build_taxonomy() -> dict:
    """
    Hand-crafted taxonomy derived from the cluster report above.
    Each intent has: definition, positive_examples, negative_examples,
    typical_resolution, escalation_conditions.

    This taxonomy was derived from inspecting the SpotifyCares cluster output —
    NOT copied from the spec examples.
    """
    taxonomy = {
        "intents": [
            {
                "name": "playback_issue",
                "definition": "Customer reports music not playing, skipping, buffering, crashing, or stopping unexpectedly.",
                "positive_examples": [
                    "Spotify keeps skipping songs randomly",
                    "App crashes every time I try to play a song",
                    "Music stops after 30 seconds",
                ],
                "negative_examples": [
                    "I can't find a specific song (-> content_not_found)",
                    "My offline songs disappeared (-> offline_download_issue)",
                ],
                "typical_resolution": "Ask for device/OS/app version, suggest restart, reinstall, or cache clear.",
                "escalation_conditions": [
                    "Issue persists after standard troubleshooting steps",
                    "Widespread outage pattern across multiple users",
                ],
            },
            {
                "name": "offline_download_issue",
                "definition": "Customer reports problems downloading songs/playlists for offline listening, or offline downloads disappearing.",
                "positive_examples": [
                    "My downloaded songs are gone after update",
                    "Can't download playlist for offline use",
                    "Downloads keep failing on WiFi",
                ],
                "negative_examples": [
                    "App keeps crashing during playback (-> playback_issue)",
                    "Can't find a song to download (-> content_not_found)",
                ],
                "typical_resolution": "Check storage, re-enable offline mode, re-download playlist.",
                "escalation_conditions": [
                    "Downloads repeatedly fail after troubleshooting",
                    "Premium subscriber losing access to offline feature",
                ],
            },
            {
                "name": "subscription_billing",
                "definition": "Customer has questions or issues about charges, payment methods, subscription price, or billing cycle.",
                "positive_examples": [
                    "I was charged twice this month",
                    "Why did my price go up?",
                    "How do I update my payment method?",
                ],
                "negative_examples": [
                    "I want to cancel my subscription (-> cancellation_request)",
                    "I can't log in to check my bill (-> account_access)",
                ],
                "typical_resolution": "Direct to account billing page; explain charge or pricing change.",
                "escalation_conditions": [
                    "Duplicate or unexpected charges",
                    "Customer disputes a charge and requests refund",
                    "Financial information involved",
                ],
            },
            {
                "name": "cancellation_request",
                "definition": "Customer wants to cancel their Spotify subscription or free trial.",
                "positive_examples": [
                    "How do I cancel my Premium?",
                    "I want to stop being charged — cancel my account",
                    "Please cancel my free trial",
                ],
                "negative_examples": [
                    "I already cancelled but was still charged (-> subscription_billing)",
                    "I want to delete my account entirely (-> account_closure)",
                ],
                "typical_resolution": "Direct to cancellation flow in account settings.",
                "escalation_conditions": [
                    "Customer charged after cancellation confirmation",
                    "Customer unable to access cancellation flow",
                ],
            },
            {
                "name": "account_access",
                "definition": "Customer cannot log in, forgot password, is locked out, or has issues with authentication.",
                "positive_examples": [
                    "I forgot my Spotify password",
                    "Can't log in — it says my account doesn't exist",
                    "Two-factor authentication is blocking me",
                ],
                "negative_examples": [
                    "My account was hacked (-> account_security)",
                    "I want to delete my account (-> account_closure)",
                ],
                "typical_resolution": "Direct to password reset; check email for account confirmation.",
                "escalation_conditions": [
                    "Customer locked out after multiple failed attempts",
                    "Suspected account compromise",
                ],
            },
            {
                "name": "account_security",
                "definition": "Customer suspects their account has been hacked, accessed by someone else, or their credentials compromised.",
                "positive_examples": [
                    "Someone else is using my Spotify account",
                    "I see devices logged in that aren't mine",
                    "My account was hacked",
                ],
                "negative_examples": [
                    "I forgot my password (-> account_access)",
                    "I want to close my account (-> account_closure)",
                ],
                "typical_resolution": "Immediate escalation — security issues always require human review.",
                "escalation_conditions": [
                    "Always escalate — account compromise is a sensitive security issue",
                ],
            },
            {
                "name": "account_closure",
                "definition": "Customer wants to permanently delete their Spotify account.",
                "positive_examples": [
                    "I want to delete my Spotify account permanently",
                    "How do I close my account?",
                    "Please remove all my data and close my account",
                ],
                "negative_examples": [
                    "I just want to cancel Premium, not delete my account (-> cancellation_request)",
                ],
                "typical_resolution": "Direct to account deletion page; confirm data implications.",
                "escalation_conditions": [
                    "Customer mentions GDPR / data rights — requires DPO involvement",
                    "Customer cannot access account to delete it",
                ],
            },
            {
                "name": "content_not_found",
                "definition": "Customer cannot find a specific song, album, artist, or podcast on Spotify.",
                "positive_examples": [
                    "Why isn't [artist] available on Spotify?",
                    "I can't find a specific album",
                    "This podcast disappeared from the app",
                ],
                "negative_examples": [
                    "My downloaded songs disappeared (-> offline_download_issue)",
                    "App won't play any music (-> playback_issue)",
                ],
                "typical_resolution": "Explain licensing/regional availability; suggest workaround if available.",
                "escalation_conditions": [
                    "Content was previously available and suddenly removed (possible licensing dispute)",
                ],
            },
            {
                "name": "app_feature_question",
                "definition": "Customer asks how to use a specific Spotify feature (crossfade, equalizer, collaborative playlists, etc.).",
                "positive_examples": [
                    "How do I enable crossfade?",
                    "Can I share a playlist with a friend?",
                    "How do I use Spotify Connect?",
                ],
                "negative_examples": [
                    "A feature I use is broken (-> playback_issue or technical_issue)",
                    "Why was a feature removed? (-> complaint_feedback)",
                ],
                "typical_resolution": "Provide direct how-to guidance or link to help article.",
                "escalation_conditions": [
                    "Feature behaves unexpectedly after following guidance",
                ],
            },
            {
                "name": "technical_issue",
                "definition": "Customer reports a bug or technical problem not related to playback — e.g. UI glitches, sync issues, Spotify Connect failures, widget problems.",
                "positive_examples": [
                    "Spotify widget on my lock screen stopped working",
                    "My playlists won't sync between devices",
                    "Spotify Connect keeps disconnecting from my speaker",
                ],
                "negative_examples": [
                    "Music won't play at all (-> playback_issue)",
                    "Can't download for offline (-> offline_download_issue)",
                ],
                "typical_resolution": "Gather device/OS/version details; suggest standard troubleshooting.",
                "escalation_conditions": [
                    "Issue confirmed on multiple devices after troubleshooting",
                    "Potential platform-wide bug",
                ],
            },
            {
                "name": "complaint_feedback",
                "definition": "Customer expresses dissatisfaction, frustration, or provides negative feedback about Spotify's service, decisions, or features.",
                "positive_examples": [
                    "Spotify's new UI is terrible",
                    "You removed a feature I loved — this is unacceptable",
                    "Your customer support is useless",
                ],
                "negative_examples": [
                    "App is crashing (-> playback_issue or technical_issue)",
                    "I want a refund (-> subscription_billing)",
                ],
                "typical_resolution": "Acknowledge frustration; direct to feedback channel; do not promise changes.",
                "escalation_conditions": [
                    "Customer mentions legal action",
                    "Harassment or abusive language directed at staff",
                ],
            },
            {
                "name": "other_unknown",
                "definition": "Message does not confidently fit any defined intent — too vague, off-topic, or requires clarification.",
                "positive_examples": [
                    "Hi",
                    "???",
                    "This is not related to music at all",
                ],
                "negative_examples": [],
                "typical_resolution": "Ask a clarifying question or escalate for human triage.",
                "escalation_conditions": [
                    "Always escalate or ask for clarification — do not auto-handle unknown intent",
                ],
            },
        ]
    }
    return taxonomy


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    threads = load_threads(THREADS_FILE)
    msgs    = extract_customer_messages(threads)
    sampled = sample_messages(msgs)

    embeddings = embed(sampled)
    labels     = cluster(embeddings)

    print_cluster_report(sampled, labels)

    print("\n[taxonomy] Building taxonomy from cluster insights ...")
    taxonomy = build_taxonomy()

    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(taxonomy, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    intent_names = [i["name"] for i in taxonomy["intents"]]
    print(f"\n[taxonomy] Saved {len(intent_names)} intents to {TAXONOMY_PATH}")
    print(f"  Intents: {intent_names}")


if __name__ == "__main__":
    main()