"""
agent.py — wires the full Groundline pipeline:
  intent classification → retrieval → reply generation → escalation decision

Usage:
    python -m src.agent --message "my spotify keeps crashing"
"""

import json
import argparse
import pickle
from pathlib import Path

from src.retrieval.retrieve import Retriever
from src.intent.classifier import TFIDFClassifier, EmbeddingClassifier, load_taxonomy
from src.generation.reply_generator import ReplyGenerator
from src.escalation.policy import EscalationPolicy

MODELS_DIR    = Path("data/processed/models")
TAXONOMY_PATH = Path("src/taxonomy/taxonomy.yaml")


class GroundlineAgent:
    def __init__(self):
        print("[agent] Loading components ...")

        self.retriever = Retriever()
        self.generator = ReplyGenerator()
        self.policy    = EscalationPolicy()

        self.clf_tfidf = TFIDFClassifier.load(MODELS_DIR)
        taxonomy       = load_taxonomy(TAXONOMY_PATH)
        self.clf_emb   = EmbeddingClassifier(taxonomy)

        print("[agent] All components loaded. Ready.")

    def run(self, message: str, context: str = "", k: int = 5) -> dict:
        """
        Run the full pipeline for one customer message.

        Returns the structured JSON output:
        {
          intent, intent_confidence, intent_margin,
          retrieved_examples, retrieval_quality,
          draft_reply, supported_by_evidence, unsupported_claims,
          decision, escalation_reason, triggered_rules, risk_score,
          evidence
        }
        """

        # --- Step 1: Intent classification ---
        intent_tfidf = self.clf_tfidf.predict(message)
        intent_emb   = self.clf_emb.predict(message)

        # Blend confidence: agreement boosts, disagreement penalises
        if intent_tfidf["intent"] == intent_emb["intent"]:
            blended_confidence = min(1.0, intent_tfidf["confidence"] * 1.1)
        else:
            blended_confidence = intent_tfidf["confidence"] * 0.85

        intent_result = {
            "intent":       intent_tfidf["intent"],
            "confidence":   round(blended_confidence, 4),
            "margin":       intent_tfidf["margin"],
            "tfidf_intent": intent_tfidf["intent"],
            "emb_intent":   intent_emb["intent"],
            "agreement":    intent_tfidf["intent"] == intent_emb["intent"],
        }

        # --- Step 2: Retrieval ---
        retrieved        = self.retriever.retrieve(message, k=k)
        retrieval_quality = self.retriever.retrieval_quality(retrieved)

        # --- Step 3: Reply generation ---
        generation_result = self.generator.generate(message, context, retrieved)

        # --- Step 4: Escalation decision ---
        decision = self.policy.decide(
            intent_result,
            retrieval_quality,
            generation_result,
            message,
        )

        # --- Assemble final output ---
        return {
            "intent":                intent_result["intent"],
            "intent_confidence":     intent_result["confidence"],
            "intent_margin":         intent_result["margin"],
            "classifier_agreement":  intent_result["agreement"],
            "retrieved_examples":    [
                {
                    "customer_message": r["customer_message"][:150],
                    "brand_response":   r["brand_response"][:150],
                    "similarity_score": r["similarity_score"],
                    "meets_threshold":  r["meets_threshold"],
                }
                for r in retrieved
            ],
            "retrieval_quality":     retrieval_quality,
            "draft_reply":           generation_result["draft_reply"],
            "supported_by_evidence": generation_result["supported_by_evidence"],
            "unsupported_claims":    generation_result["unsupported_claims"],
            "decision":              decision["decision"],
            "escalation_reason":     decision["escalation_reason"],
            "triggered_rules":       decision["triggered_rules"],
            "risk_score":            decision["risk_score"],
            "evidence":              generation_result["evidence_used"],
        }


def main():
    parser = argparse.ArgumentParser(description="Groundline support agent")
    parser.add_argument("--message", type=str, required=True,
                        help="Customer message to process")
    parser.add_argument("--context", type=str, default="",
                        help="Optional conversation context")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of examples to retrieve (default: 5)")
    args = parser.parse_args()

    agent  = GroundlineAgent()
    result = agent.run(args.message, args.context, args.k)

    print("\n" + "="*60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()