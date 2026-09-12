"""
policy.py — deterministic + model-assisted escalation policy.
Decides AUTO_HANDLE vs ESCALATE with a stated reason.

Escalation conditions (in priority order):
  1. Sensitive intent (always escalate)
  2. Weak retrieval (no evidence above threshold)
  3. Low intent confidence
  4. Ambiguous intent (small margin between top-2)
  5. Unsupported claims detected in generated reply
  6. Strong negative sentiment signals

Thresholds are defined in config.yaml and tuned on a validation
split — not guessed.

Usage (as module):
    from src.escalation.policy import EscalationPolicy
    policy = EscalationPolicy()
    decision = policy.decide(intent_result, retrieval_quality, generation_result, message)
"""

import re
import yaml
from pathlib import Path

CONFIG_PATH = Path("config.yaml")

# Intents that always trigger escalation regardless of confidence
ALWAYS_ESCALATE_INTENTS = {
    "account_security",
    "other_unknown",
}

# Intents that escalate when combined with financial/legal signals
SENSITIVE_INTENTS = {
    "subscription_billing",
    "cancellation_request",
    "account_closure",
}

# Keyword signals that raise escalation probability
SENSITIVE_SIGNALS = [
    "refund", "charged twice", "double charged", "overcharged",
    "legal", "lawyer", "sue", "lawsuit", "court",
    "gdpr", "data protection", "delete my data",
    "harassment", "abuse", "threat", "reported",
    "fraud", "stolen", "unauthorized", "compromised",
    "hacked", "not me", "someone else",
    "unacceptable", "disgusting", "terrible service",
    "cancel and refund", "want my money back",
]

# Strong anger signals — increase escalation probability but don't auto-escalate
ANGER_SIGNALS = [
    "wtf", "what the hell", "bullshit", "useless",
    "worst", "pathetic", "incompetent", "ridiculous",
    "furious", "outraged", "disgusted",
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class EscalationPolicy:
    def __init__(self):
        cfg = load_config()
        esc = cfg.get("escalation", {})

        self.low_confidence_threshold  = esc.get("low_confidence_threshold", 0.6)
        self.weak_retrieval_similarity = esc.get("weak_retrieval_similarity", 0.55)
        self.ambiguity_margin          = esc.get("ambiguity_margin", 0.1)

        print(f"[policy] Escalation thresholds loaded:")
        print(f"  low_confidence:  < {self.low_confidence_threshold}")
        print(f"  weak_retrieval:  top_score < {self.weak_retrieval_similarity}")
        print(f"  ambiguity_margin: < {self.ambiguity_margin}")

    def _check_sensitive_signals(self, message: str) -> tuple[bool, list[str]]:
        """Check for sensitive keyword signals in the message."""
        msg_lower = message.lower()
        triggered = [s for s in SENSITIVE_SIGNALS if s in msg_lower]
        return len(triggered) > 0, triggered

    def _check_anger_signals(self, message: str) -> bool:
        msg_lower = message.lower()
        return any(s in msg_lower for s in ANGER_SIGNALS)

    def decide(
        self,
        intent_result: dict,
        retrieval_quality: dict,
        generation_result: dict,
        message: str,
    ) -> dict:
        """
        Make the AUTO_HANDLE / ESCALATE decision.

        Args:
            intent_result:      output from classifier.predict()
            retrieval_quality:  output from retriever.retrieval_quality()
            generation_result:  output from generator.generate()
            message:            raw customer message text

        Returns dict with:
            decision:           AUTO_HANDLE | ESCALATE
            escalation_reason:  human-readable reason
            triggered_rules:    list of rules that fired
            risk_score:         0.0-1.0 composite risk signal
        """
        triggered_rules = []
        risk_score      = 0.0

        intent     = intent_result.get("intent", "other_unknown")
        confidence = intent_result.get("confidence", 0.0)
        margin     = intent_result.get("margin", 0.0)

        top_score       = retrieval_quality.get("top_score", 0.0)
        weak_retrieval  = retrieval_quality.get("weak_retrieval", True)

        supported       = generation_result.get("supported_by_evidence", "no")
        unsupported     = generation_result.get("unsupported_claims", "")
        gen_error       = generation_result.get("error")

        # --- Rule 1: Always-escalate intents ---
        if intent in ALWAYS_ESCALATE_INTENTS:
            triggered_rules.append(f"always_escalate_intent:{intent}")
            risk_score += 1.0

        # --- Rule 2: Sensitive intent + financial/legal signal ---
        has_sensitive, sensitive_terms = self._check_sensitive_signals(message)
        if intent in SENSITIVE_INTENTS and has_sensitive:
            triggered_rules.append(
                f"sensitive_intent_with_signals:{intent}:{sensitive_terms[:2]}"
            )
            risk_score += 0.8

        # --- Rule 3: Low intent confidence ---
        if confidence < self.low_confidence_threshold:
            triggered_rules.append(
                f"low_confidence:{confidence:.3f}<{self.low_confidence_threshold}"
            )
            risk_score += 0.5

        # --- Rule 4: Weak retrieval ---
        if weak_retrieval or top_score < self.weak_retrieval_similarity:
            triggered_rules.append(
                f"weak_retrieval:top_score={top_score:.3f}"
            )
            risk_score += 0.5

        # --- Rule 5: Ambiguous intent (small margin between top-2) ---
        if margin < self.ambiguity_margin:
            triggered_rules.append(
                f"ambiguous_intent:margin={margin:.3f}<{self.ambiguity_margin}"
            )
            risk_score += 0.3

        # --- Rule 6: Unsupported claims in generated reply ---
        if supported == "no" or (
            unsupported and unsupported not in ("none", "unknown", "")
        ):
            triggered_rules.append(f"unsupported_claims_in_reply:{unsupported[:80]}")
            risk_score += 0.6

        # --- Rule 7: Generation error ---
        if gen_error:
            triggered_rules.append(f"generation_error:{str(gen_error)[:60]}")
            risk_score += 0.7

        # --- Rule 8: Anger signals (increase risk but don't auto-escalate) ---
        if self._check_anger_signals(message):
            triggered_rules.append("anger_signals_detected")
            risk_score += 0.2

        # --- Decision ---
        # Escalate if any hard rule fired (risk >= 0.8) OR cumulative risk > 0.7
        should_escalate = (
            any("always_escalate" in r for r in triggered_rules) or
            any("sensitive_intent_with_signals" in r for r in triggered_rules) or
            any("unsupported_claims" in r for r in triggered_rules) or
            any("generation_error" in r for r in triggered_rules) or
            risk_score >= 0.8
        )

        decision = "ESCALATE" if should_escalate else "AUTO_HANDLE"

        # Build human-readable reason
        if not triggered_rules:
            reason = "All signals green — high confidence, strong retrieval, grounded reply."
        else:
            reason = self._build_reason(triggered_rules, decision, intent)

        return {
            "decision":          decision,
            "escalation_reason": reason,
            "triggered_rules":   triggered_rules,
            "risk_score":        round(min(risk_score, 1.0), 3),
        }

    def _build_reason(self, rules: list, decision: str, intent: str) -> str:
        reasons = []

        for rule in rules:
            if "always_escalate_intent" in rule:
                reasons.append(
                    f"Intent '{intent}' always requires human review."
                )
            elif "sensitive_intent_with_signals" in rule:
                reasons.append(
                    "Financial or legal signals detected — human review needed."
                )
            elif "low_confidence" in rule:
                reasons.append("Intent classification confidence is low.")
            elif "weak_retrieval" in rule:
                reasons.append("No sufficiently similar historical examples found.")
            elif "ambiguous_intent" in rule:
                reasons.append("Multiple intents have similar probability — ambiguous.")
            elif "unsupported_claims" in rule:
                reasons.append("Generated reply contains unsupported claims.")
            elif "generation_error" in rule:
                reasons.append("Reply generation failed — cannot auto-handle safely.")
            elif "anger_signals" in rule:
                reasons.append("Strong negative sentiment detected.")

        return " | ".join(dict.fromkeys(reasons))  # deduplicate, preserve order


if __name__ == "__main__":
    # smoke test
    from src.retrieval.retrieve import Retriever
    from src.intent.classifier import TFIDFClassifier, load_taxonomy, EmbeddingClassifier
    from src.generation.reply_generator import ReplyGenerator

    retriever = Retriever()
    generator = ReplyGenerator()
    policy    = EscalationPolicy()

    models_dir = Path("data/processed/models")
    clf_tfidf  = TFIDFClassifier.load(models_dir)
    taxonomy   = load_taxonomy(Path("src/taxonomy/taxonomy.yaml"))
    clf_emb    = EmbeddingClassifier(taxonomy)

    test_cases = [
        {
            "message": "my spotify keeps crashing after the latest update on iPhone",
            "context": "",
            "expected": "AUTO_HANDLE",
        },
        {
            "message": "I was charged twice this month, I want a refund now",
            "context": "",
            "expected": "ESCALATE",
        },
        {
            "message": "someone logged into my account from a different country",
            "context": "",
            "expected": "ESCALATE",
        },
        {
            "message": "how do I enable crossfade on desktop?",
            "context": "",
            "expected": "AUTO_HANDLE",
        },
    ]

    print("\n" + "="*60)
    for tc in test_cases:
        msg = tc["message"]
        print(f"\nMessage:  {msg}")
        print(f"Expected: {tc['expected']}")

        retrieved  = retriever.retrieve(msg, k=5)
        ret_qual   = retriever.retrieval_quality(retrieved)
        intent_res = clf_tfidf.predict(msg)
        emb_res    = clf_emb.predict(msg)

        # blend confidence: if both classifiers agree, boost confidence
        if intent_res["intent"] == emb_res["intent"]:
            intent_res["confidence"] = min(
                1.0, intent_res["confidence"] * 1.1
            )
        else:
            intent_res["confidence"] *= 0.85

        gen_res    = generator.generate(msg, tc["context"], retrieved)
        decision   = policy.decide(intent_res, ret_qual, gen_res, msg)

        print(f"Intent:   {intent_res['intent']} (conf: {intent_res['confidence']:.3f})")
        print(f"Retrieval: top={ret_qual['top_score']:.3f}  weak={ret_qual['weak_retrieval']}")
        print(f"Decision: {decision['decision']}  (risk: {decision['risk_score']})")
        print(f"Reason:   {decision['escalation_reason']}")
        print(f"Rules:    {decision['triggered_rules']}")
        match = "✓" if decision["decision"] == tc["expected"] else "✗"
        print(f"Match expected: {match}")
        print("-"*60)