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
| `GROQ_API_KEY` | Grounded reply generation |
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
`[TODO — Phase 9]`

## Running evaluation
`[TODO — Phase 9]`

## Reproducing headline results (<15 minutes)
`[TODO — Phase 12]`

## Example input/output
`[TODO — Phase 9]`

## Metrics
`[TODO — Phase 12]`

## Limitations
`[TODO — Phase 12]`
