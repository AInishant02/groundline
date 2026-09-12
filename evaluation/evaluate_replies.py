"""
evaluate_replies.py — evaluates generated replies on a sample of the golden set.
Uses the Gemini LLM-as-judge (different model than generator).

Usage:
    python -m evaluation.evaluate_replies
"""

import os
import re
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.agent import GroundlineAgent
import src.intent.classifier 

load_dotenv()

GOLDEN_FILE  = Path("evaluation/golden_set.csv")
OUTPUT_DIR   = Path("data/processed/evaluation")
JUDGE_MODEL  = "gemini-3.5-flash-lite"   # different from generator
EVAL_SAMPLE  = 30    # evaluate a sample to manage API costs
SEED         = 42

JUDGE_SYSTEM = """You are an independent evaluator of AI-generated customer support replies
for Spotify (@SpotifyCares). You are NOT the model that generated these replies.

Score each reply on these dimensions (1-5 scale):
  1 = unacceptable
  2 = major problems
  3 = acceptable but imperfect
  4 = good
  5 = excellent

Dimensions:
- groundedness:       Is the reply supported by the retrieved historical evidence?
- correctness:        Does it appropriately address the customer's actual issue?
- helpfulness:        Would this move the customer toward resolution?
- safety:             Does it avoid unsupported claims, promises, or invented policies?
- brand_consistency:  Does it match SpotifyCares' historical tone and style?
- conciseness:        Is it appropriately brief for Twitter/X support?

Respond ONLY with valid JSON — no markdown, no preamble:
{
  "groundedness": <1-5>,
  "correctness": <1-5>,
  "helpfulness": <1-5>,
  "safety": <1-5>,
  "brand_consistency": <1-5>,
  "conciseness": <1-5>,
  "overall": <1-5>,
  "explanation": "<one sentence>",
  "unsupported_claims_detected": "<none or list>",
  "pass_fail": "pass" or "fail"
}
"""


def judge_reply(client, message: str, context: str,
                evidence: list, reply: str) -> dict:
    evidence_str = "\n".join(
        f"- Customer: {e.get('customer_message','')[:100]} | "
        f"Spotify: {e.get('brand_response','')[:100]}"
        for e in evidence[:3]
    )

    prompt = f"""CUSTOMER MESSAGE:
{message}

CONVERSATION CONTEXT:
{context or 'None'}

RETRIEVED EVIDENCE:
{evidence_str or 'None'}

GENERATED REPLY TO EVALUATE:
{reply}

Score this reply on all dimensions. Be critical and independent."""

    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM,
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        raw    = response.text.strip()
        clean  = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        return {
            "groundedness": 0, "correctness": 0, "helpfulness": 0,
            "safety": 0, "brand_consistency": 0, "conciseness": 0,
            "overall": 0, "explanation": f"Judge error: {e}",
            "unsupported_claims_detected": "error",
            "pass_fail": "error"
        }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GEMINI_API_KEY")
    client  = genai.Client(api_key=api_key)

    print("[evaluate_replies] Loading golden set ...")
    golden = pd.read_csv(GOLDEN_FILE, dtype=str)
    sample = golden.sample(n=min(EVAL_SAMPLE, len(golden)),
                           random_state=SEED).reset_index(drop=True)
    print(f"  Evaluating {len(sample)} examples ...")

    agent = GroundlineAgent()

    rows = []
    for i, row in sample.iterrows():
        print(f"  [{i+1}/{len(sample)}] {row['customer_message'][:60]}...")

        result = agent.run(
            row["customer_message"],
            row.get("conversation_context", ""),
        )

        if not result["draft_reply"]:
            err = result.get("error", "unknown")
            print(f"    Skipping — no reply generated. Error: {err}")
            continue

        scores = judge_reply(
            client,
            row["customer_message"],
            row.get("conversation_context", ""),
            result["retrieved_examples"],
            result["draft_reply"],
        )

        rows.append({
            "id":               row["id"],
            "intent_label":     row["intent_label"],
            "customer_message": row["customer_message"],
            "draft_reply":      result["draft_reply"],
            "decision":         result["decision"],
            "supported":        result["supported_by_evidence"],
            **{f"score_{k}": v for k, v in scores.items()
               if k not in ("explanation", "unsupported_claims_detected", "pass_fail")},
            "explanation":      scores.get("explanation", ""),
            "pass_fail":        scores.get("pass_fail", "error"),
        })

        time.sleep(5)  # rate limit buffer

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "reply_eval.csv"
    df.to_csv(out, index=False)

    # summary
    score_cols = [c for c in df.columns if c.startswith("score_")]
    print("\n" + "="*60)
    print("REPLY EVALUATION SCORECARD (LLM Judge)")
    print("="*60)
    for col in score_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col:<30} avg: {numeric.mean():.2f}  min: {numeric.min():.0f}  max: {numeric.max():.0f}")

    pass_rate = (df["pass_fail"] == "pass").mean()
    print(f"\n  Pass rate: {pass_rate:.2%}")
    print(f"\n[evaluate_replies] Results saved to {out}")


if __name__ == "__main__":
    main()