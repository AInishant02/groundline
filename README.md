# Groundline

A trustworthy AI customer-support agent for one brand from the Kaggle
**Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`).

Given an incoming customer message, Groundline:
1. Classifies intent (with a confidence score derived from real signals, not a guess)
2. Retrieves similar historical support conversations as evidence
3. Drafts a reply grounded in that evidence — never inventing policy, refunds, or timelines
4. Decides `AUTO_HANDLE` vs `ESCALATE`, with a stated reason

Groundline is built to prioritize **trustworthiness over automation rate**.
It should demonstrate not just that it can answer customers, but that it
knows when it should *not*.

> This README is filled in phase by phase. Sections marked `[TODO]` land
> as we complete the corresponding phase of the build.

## Installation

```bash
git clone <your-repo-url> groundline
cd groundline
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY` | Grounded reply generation |
| `GEMINI_API_KEY` | Independent LLM-as-judge (deliberately a different provider than generation) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Only needed if you don't already have `~/.kaggle/kaggle.json` |

## Dataset setup

You already have a Kaggle API token configured, so `src/data/download.py`
(Phase 1) will pull the dataset straight into `data/raw/` — no manual
download needed.

## Repository structure

```text
groundline/
│
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── decision_log.md
│
├── data/
│   ├── raw/            # untouched original dataset
│   ├── processed/      # cleaned + threaded conversations, FAISS index
│   └── golden/          # kept separate from anything the system is built/tuned on
│
├── src/
│   ├── data/            # download, clean, thread reconstruction
│   ├── taxonomy/        # intent taxonomy derivation
│   ├── retrieval/        # embeddings + FAISS index/retrieve
│   ├── intent/           # intent classifier
│   ├── generation/       # grounded reply generator
│   ├── escalation/       # escalation policy
│   ├── agent.py          # wires the full pipeline together
│   └── utils.py
│
├── evaluation/
│   ├── golden_set.csv
│   ├── evaluate_intent.py
│   ├── evaluate_replies.py
│   ├── evaluate_escalation.py
│   ├── baselines.py
│   ├── llm_judge.py
│   └── human_judge_agreement.py
│
├── notebooks/
└── report/
```

## Running the pipeline
## Running the pipeline

```bash
# 1. Download dataset
python -m src.data.download

# 2. Select brand and clean data
python -m src.data.brand_selection
python -m src.data.clean
python -m src.data.threads

# 3. Build taxonomy
python -m src.taxonomy.build_taxonomy

# 4. Build golden set
python -m src.data.golden_set

# 5. Build retrieval index
python -m src.retrieval.index

# 6. Train classifiers
python -m src.intent.classifier

# 7. Run the agent on a message
python -m src.agent --message "my spotify keeps crashing on iPhone"
```
## Running evaluation
## Running evaluation

```bash
python -m evaluation.evaluate_intent
python -m evaluation.evaluate_replies
python -m evaluation.evaluate_escalation
python -m evaluation.human_judge_agreement --generate
# fill in evaluation/human_scoring_sheet.csv
python -m evaluation.human_judge_agreement --analyze
```
## Reproducing headline results (<15 minutes)
## Reproducing headline results (<15 minutes)

```bash
# assumes dataset already downloaded and venv active
python -m src.data.brand_selection
python -m src.data.clean
python -m src.data.threads
python -m src.retrieval.index
python -m src.intent.classifier
python -m evaluation.evaluate_intent
# headline: TF-IDF macro F1 = 0.767
```
## Example input/output
## Example input/output

```bash
python -m src.agent --message "my payment was charged twice, can you help?"
```

```json
{
  "intent": "subscription_billing",
  "intent_confidence": 1.0,
  "decision": "ESCALATE",
  "escalation_reason": "Financial or legal signals detected — human review needed.",
  "draft_reply": "Hey there! We can certainly help look into this. Could you send us a DM with your account email or username? We'll check things out backstage /AB",
  "evidence": ["Example 1 (DM with email)", "Example 2 (DM with email or username)"]
}
```

## Metrics
## Metrics

| Metric | Value |
|---|---|
| Intent Accuracy (TF-IDF) | 0.775 |
| Intent Macro F1 (TF-IDF) | 0.767 |
| Reply Pass Rate (LLM Judge) | 93.3% |
| Reply Safety Avg | 4.83 / 5 |
| False-Auto Rate | 16.0% |
| Escalation Recall | 54.3% |
| Auto-Handling Rate | 75.9% |
| Human-Judge Kappa | 0.327 |
## Limitations
## Limitations

- English only — non-English messages are classified but should be escalated
- False-auto rate of 16% means 1 in 6 messages needing human review are auto-handled
- Reply evaluation covers 30 examples only — statistically limited
- LLM judge shares model family with generator — potential self-evaluation bias
- Offline evaluation only — production performance will differ
- Historical SpotifyCares responses used as evidence are themselves imperfect
- Free-tier Gemini API rate limits require 5s delays between calls in evaluation
