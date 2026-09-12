"""
reply_generator.py — retrieval-grounded reply generation via Gemini.
Never invents policies, refunds, timelines, or account actions not
supported by retrieved evidence.

Usage (as module):
    from src.generation.reply_generator import ReplyGenerator
    rg = ReplyGenerator()
    result = rg.generate(message, context, retrieved_examples)
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

BRAND        = "SpotifyCares"
MODEL_NAME   = "gemini-3.6-flash"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3

SYSTEM_PROMPT = """You are a customer support agent for Spotify (handle: @SpotifyCares).
You are given:
1. A customer's tweet
2. Conversation context (if any)
3. Retrieved historical support examples from Spotify's actual past responses

Your job is to draft a short, helpful reply in Spotify's support style.

STRICT RULES - you must follow all of these:
- NEVER invent policies, refund amounts, deadlines, or eligibility rules not present in the evidence
- NEVER promise account actions (refunds, credits, cancellations) - direct to the right channel instead
- NEVER copy a historical response verbatim - adapt it naturally
- If evidence is insufficient, say so and ask for more information or suggest escalation
- Keep replies under 280 characters where possible (Twitter/X limit)
- Match Spotify's tone: friendly, concise, uses the customer's name if available, ends with initials like /AB

For each reply, you must also output:
SUPPORTED_BY_EVIDENCE: <yes/partial/no>
UNSUPPORTED_CLAIMS: <list any claims not backed by evidence, or 'none'>

Format your response as JSON:
{
  "draft_reply": "...",
  "supported_by_evidence": "yes|partial|no",
  "unsupported_claims": "none" or "list of claims",
  "evidence_used": ["brief description of which examples informed the reply"]
}
"""


def build_evidence_block(retrieved: list) -> str:
    if not retrieved:
        return "No historical examples retrieved."
    lines = []
    for i, r in enumerate(retrieved[:5], 1):
        sim   = r.get("similarity_score", 0)
        meets = r.get("meets_threshold", False)
        lines.append(f"Example {i} (similarity: {sim:.3f}, threshold_met: {meets}):")
        lines.append(f"  Customer: {r['customer_message'][:200]}")
        lines.append(f"  Spotify:  {r['brand_response'][:200]}")
        lines.append("")
    return "\n".join(lines)


def build_prompt(message: str, context: str, retrieved: list) -> str:
    evidence_block = build_evidence_block(retrieved)
    prompt = f"""CUSTOMER TWEET:
{message}

CONVERSATION CONTEXT:
{context if context else 'No prior context.'}

RETRIEVED HISTORICAL EVIDENCE:
{evidence_block}

Based only on the evidence above, draft a Spotify support reply.
Remember: if the evidence does not support a specific action or policy,
say so honestly rather than inventing an answer.

Respond with valid JSON only - no markdown, no preamble."""
    return prompt


class ReplyGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        print(f"[generator] Gemini model loaded: {MODEL_NAME}")

    def generate(self, message: str, context: str, retrieved: list) -> dict:
        prompt = build_prompt(message, context, retrieved)

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                ),
            )
            raw    = response.text.strip()
            clean  = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(clean)

            return {
                "draft_reply":           parsed.get("draft_reply", ""),
                "supported_by_evidence": parsed.get("supported_by_evidence", "no"),
                "unsupported_claims":    parsed.get("unsupported_claims", "unknown"),
                "evidence_used":         parsed.get("evidence_used", []),
                "raw_response":          raw,
                "error":                 None,
            }

        except json.JSONDecodeError as e:
            return {
                "draft_reply":           "",
                "supported_by_evidence": "no",
                "unsupported_claims":    "parse error",
                "evidence_used":         [],
                "raw_response":          raw if "raw" in dir() else "",
                "error":                 f"JSON parse error: {e}",
            }
        except Exception as e:
            return {
                "draft_reply":           "",
                "supported_by_evidence": "no",
                "unsupported_claims":    "generation error",
                "evidence_used":         [],
                "raw_response":          "",
                "error":                 str(e),
            }


if __name__ == "__main__":
    from src.retrieval.retrieve import Retriever

    retriever = Retriever()
    generator = ReplyGenerator()

    test_cases = [
        {
            "message": "my spotify keeps crashing after the latest update on iPhone",
            "context": "",
        },
        {
            "message": "I was charged twice this month, this is unacceptable",
            "context": "",
        },
        {
            "message": "someone logged into my account from a different country",
            "context": "",
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Message: {tc['message']}")
        retrieved = retriever.retrieve(tc["message"], k=3)
        result    = generator.generate(tc["message"], tc["context"], retrieved)

        print(f"Draft reply:   {result['draft_reply']}")
        print(f"Evidence:      {result['supported_by_evidence']}")
        print(f"Unsupported:   {result['unsupported_claims']}")
        print(f"Evidence used: {result['evidence_used']}")
        if result["error"]:
            print(f"ERROR: {result['error']}")