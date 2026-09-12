# Decision log — Groundline

## 1. Project name
**What we chose:** Groundline
**Why:** Names the two things that actually make this system trustworthy — grounded replies and a clear escalation line — instead of a generic "AI support bot" label.
**Alternative considered:** Threshold, Verity, Backstop, Triage
**Trade-off:** A literal name like "Twitter Support Agent" is more self-explanatory at a glance, but far less memorable as a portfolio piece.

## 2. LLM provider split between generation and judging
**What we chose:** Groq for grounded reply generation, Gemini for the independent LLM-as-judge.
**Why:** The evaluation design explicitly calls for the judge not to share a model/prompt configuration with the generator, to avoid a model rating its own outputs favorably. Groq is already the provider used across other projects, so generation reuses existing familiarity; Gemini gives real architectural independence for the judge role.
**Alternative considered:** Two different Groq models (e.g. an 8B and a 70B) acting as generator and judge.
**Trade-off:** Same-provider models would mean one API key and one SDK, but weaker independence — shared training pipeline and provider-level quirks could correlate errors between generator and judge. Cross-provider judging costs a second API key but is the more defensible choice for a trustworthiness-focused system.

## 3. Brand selection — SpotifyCares over AmazonHelp
**What we chose:** SpotifyCares
**Why:** 41,734 conversations and 43,265 brand replies — large enough for a rich
retrieval knowledge base. More importantly, the domain is focused (music streaming)
which makes it realistic to cap the intent taxonomy at 8–15 intents. AmazonHelp
had 4x more data but covers e-commerce, devices, shipping, third-party sellers,
returns, and more — extremely hard to represent cleanly in 15 intents.
**Alternative considered:** AmazonHelp (155K conversations, selected automatically
by the script on volume alone)
**Trade-off:** Smaller knowledge base than AmazonHelp, but a cleaner taxonomy and
more defensible evaluation. Volume is sufficient — 43K brand replies is more than
enough for FAISS retrieval and a 200-example golden set.

## 4. Conversation thread reconstruction strategy
**What we chose:** BFS from root customer tweets, capped at depth 10,
keeping only threads with at least one brand reply.
**Why:** Threads without a brand reply have no resolution signal — useless
for retrieval-grounded generation. BFS preserves chronological order naturally.
Depth cap of 10 prevents runaway traversal on the rare very long threads (max
observed: 22 messages).
**Alternative considered:** Flat pair extraction (customer tweet → immediate
brand reply only, ignoring thread context).
**Trade-off:** Thread-level context is richer for retrieval but costs more
storage and slightly more retrieval complexity. Worth it — avg thread length
of 2.81 means overhead is minimal in practice.

## 5. Intent taxonomy — 12 intents derived from SpotifyCares data
**What we chose:** 12 intents including other_unknown, derived from KMeans
clustering (k=12) of 8,000 sampled customer opening messages embedded with
all-MiniLM-L6-v2.
**Why:** Clusters mapped clearly to distinct support topics. account_security
and account_closure kept as separate intents despite low cluster dominance —
both always trigger escalation and need explicit handling.
**Alternative considered:** k=8 (fewer, broader intents) or k=15 (more granular).
**Trade-off:** 12 gives enough granularity for meaningful escalation rules
without making the classifier's job too hard on rare intents.

## 6. Golden set construction — 191 examples, keyword pre-labelled
**What we chose:** 191 stratified examples across 12 intents, keyword
pre-labelled then manually reviewed.
**Why 191 not 200:** other_unknown is genuinely rare in the data — only 8
examples found matching truly vague/unclassifiable signals. Forcing 20 would
mean manufacturing ambiguity that doesn't exist in the dataset.
**Leakage prevention:** golden set sampled from threads.jsonl BEFORE the
retrieval index is built. Thread IDs in golden_set.csv will be explicitly
excluded from the FAISS index in Phase 5.
**Alternative considered:** Purely random sampling without stratification.
**Trade-off:** Stratified sampling overrepresents rare intents (account_closure,
account_security) relative to their natural frequency — necessary for
meaningful per-intent evaluation metrics.

## 7. Retrieval — FAISS IndexFlatIP with normalised embeddings
**What we chose:** Exact inner product search (cosine similarity) via
FAISS IndexFlatIP over all-MiniLM-L6-v2 embeddings of customer opening
messages. Min similarity threshold: 0.55.
**Why:** Exact search is fully reproducible and fast enough for 29K vectors
on a laptop. Approximate search (IVF, HNSW) only needed at 1M+ scale.
Normalised embeddings make inner product equivalent to cosine similarity.
**Alternative considered:** BM25 keyword search, or TF-IDF nearest neighbour.
**Trade-off:** Embedding search captures semantic similarity ("crashing" ≈
"keeps stopping") that keyword search would miss. Slight overhead of loading
the transformer model on each retrieval call — mitigated by caching in Phase 9.

## 8. Main classifier — TF-IDF + LogReg over embedding similarity
**What we chose:** TF-IDF + LogReg as the main classifier (0.77 macro F1)
with embedding similarity as a supporting confidence signal.
**Why:** Embedding prototype classifier scored only 0.41 macro F1 — because
with only 3 positive examples per intent from the taxonomy, the prototypes
don't represent the full distribution of real tweet language. TF-IDF trained
on 8K keyword-labelled threads captures surface patterns far more reliably.
subscription_billing and other_unknown scored 0.00 F1 on embedding — a clear
signal the prototypes are too narrow.
**Alternative considered:** Few-shot LLM classification via Gemini.
**Trade-off:** TF-IDF is less semantically flexible than embeddings but
dramatically more reliable when training signal is limited. Embedding
similarity still contributes to confidence scoring — when TF-IDF and
embedding agree, confidence is higher; when they disagree, confidence drops.

## 9. Reply generation — max_tokens set to 1024
**What we chose:** 1024 output tokens for the generation model.
**Why:** Initial 220 token limit caused JSON truncation — the model was
cutting off mid-object. 1024 gives enough room for the full JSON response
plus the draft reply, while staying well within API limits.
**Trade-off:** Slightly higher token cost per call, but necessary for
reliable structured output parsing.

## 10. Escalation policy — deterministic rules + cumulative risk score
**What we chose:** Priority-ordered deterministic rules feeding into a
cumulative risk score. Hard rules (always-escalate intents, sensitive
intent + financial signals, generation errors) trigger escalation
immediately. Soft rules (low confidence, weak retrieval, ambiguity,
anger signals) accumulate — risk >= 0.8 triggers escalation.
**Why:** Pure threshold-on-confidence would miss cases like "charged twice"
where confidence is high but the topic is sensitive. Pure rule-based
misses nuanced cases. The combination catches both.
**Alternative considered:** Single LLM call to decide escalation.
**Trade-off:** Deterministic rules are auditable and predictable — we can
explain every escalation decision. An LLM escalation judge would be less
transparent and harder to tune.

## 11. LLM model selection for generation and judging — iterative resolution
**What we chose:** gemini-3.5-flash-lite for both generation and judging.
**Why:** Three-step resolution:
  1. Started with gemini-3.6-flash → hit 20 requests/day free tier cap immediately
  2. Switched to gemini-2.0-flash-lite → model deprecated, no longer available
  3. Switched to gemini-3.5-flash-lite → 1,500 requests/day, 15 RPM, currently available
Added 5s inter-request delay to stay under 15 RPM limit.
**Alternative considered:** Paying for a billing account to remove rate limits entirely.
**Trade-off:** gemini-3.5-flash-lite is weaker than the originally intended
gemini-3.6-flash, but sufficient for evaluation. This iterative model-hopping
is itself a real-world finding worth documenting — model availability on free
tiers changes faster than project timelines, and pinning model names in
config.yaml (as we did) makes the swap a one-line change rather than a rewrite.

## 12. Human agreement study — kappa 0.019 but 75.7% exact match
**What we found:** Average Cohen's kappa of 0.019 (slight agreement) but
75.7% exact score match rate. The low kappa is partly a statistical artifact
— scores cluster at 4-5, compressing variance and deflating kappa even when
humans and the judge broadly agree.
**Key disagreement:** The LLM judge penalizes DM-redirection replies heavily
(scores them 1/5) while human evaluators treat them as legitimate support
practice (scores them 5/5). This is a real blind spot in the judge.
**Conclusion:** The LLM judge is a useful signal but not fully trustworthy
as a standalone evaluator — particularly for replies that use DM redirection
as a resolution strategy. Human review remains essential for borderline cases.