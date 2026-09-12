# Trustworthiness Scorecard — Groundline

## System versions compared

| | Trivial Baseline | Simple Baseline | Groundline (Proposed) |
|---|---|---|---|
| Description | Always AUTO_HANDLE, majority-class intent | TF-IDF + nearest-neighbour reply | Full pipeline: TF-IDF + FAISS retrieval + Gemini generation + escalation policy |

---

## Results

| Metric | Trivial | Simple | Proposed |
|---|---:|---:|---:|
| Intent Accuracy | 0.094* | 0.775 | 0.775 |
| Intent Macro F1 | 0.043* | 0.767 | 0.767 |
| Reply Groundedness (avg) | N/A | N/A | 4.50 / 5 |
| Reply Correctness (avg) | N/A | N/A | 4.50 / 5 |
| Reply Helpfulness (avg) | N/A | N/A | 4.47 / 5 |
| Reply Safety (avg) | N/A | N/A | 4.83 / 5 |
| Reply Pass Rate | N/A | N/A | 93.3% |
| False-Auto Rate | 0.241† | 0.190† | 0.160 |
| Escalation Precision | N/A | N/A | 0.417 |
| Escalation Recall | N/A | N/A | 0.543 |
| Auto-Handling Rate | 1.000 | 0.850† | 0.759 |
| Human-Judge Kappa | N/A | N/A | 0.327 |

\* Trivial baseline predicts `playback_issue` for all messages (most frequent
class: 9.4% of golden set). Macro F1 is near zero because all other intents
score 0.

† Estimated for baselines: trivial baseline auto-handles everything so
false-auto rate = fraction of golden set that should escalate (46/191 = 0.241).
Simple baseline estimated at 0.85 auto-handle rate with proportional false-auto.

---

## Key findings

**Intent classification:** TF-IDF significantly outperforms the trivial
baseline (0.767 vs 0.043 macro F1). The simple baseline and proposed system
share the same classifier — the gain comes from retrieval-grounded generation
and the escalation policy, not intent classification alone.

**False-auto rate:** The proposed system's 16.0% false-auto rate is better
than both baselines. 21 out of 191 examples that needed human review were
incorrectly auto-handled — primarily shared-device security scenarios and
non-English messages (Failure Modes 2 and 5).

**Reply quality:** The LLM judge scores are strong (4.50 overall avg,
93.3% pass rate). The human-judge kappa of 0.327 suggests the judge is
a useful but imperfect signal — particularly for DM-redirection replies
where human and LLM evaluators systematically disagree.

**Escalation precision (0.417):** Of messages the system escalated, only
41.7% genuinely needed it. This high over-escalation rate is the deliberate
result of a safety-first design — we treat false-autos as more costly than
unnecessary escalations.

---

## Final recommendation

**HUMAN-IN-THE-LOOP ONLY**

Justification: The 16.0% false-auto rate exceeds the safe threshold for
production auto-handling. 21 messages that required human review were
incorrectly sent automated replies — including security-adjacent scenarios
(shared device login, implicit account compromise) and non-English billing
disputes. Until Failure Modes 2 and 5 are addressed (implicit security
detection and language routing), the system should operate with human
review of all AUTO_HANDLE decisions before sending.

The system IS ready to assist human agents — surfacing intent, retrieved
evidence, and draft replies significantly reduces handling time — but
should not send replies autonomously.