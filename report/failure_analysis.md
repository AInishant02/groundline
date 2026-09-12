# Failure Analysis — Groundline (SpotifyCares Agent)

Derived from actual evaluation results on the 191-example golden set.
All examples are real — none are invented.

---

## Failure Mode 1 — DM Redirection Scored as Non-Answer

**Failure mode:** Agent replies with "We've sent you a DM" or "Please DM us"
without providing any actionable information in the public reply.

**Example:**
- Customer: "hi how do I download Spotify to my tv?!"
- System output: "Hey there! We've just sent you a DM. Let's continue
  chatting there /AB"
- Expected: Direct guidance on how to find Spotify in the TV app store,
  or at minimum a link to the help article.

**Why it failed:** The retrieval system found many historical examples where
SpotifyCares used DM redirection — because the brand does this frequently.
The generator correctly matched the brand pattern, but the pattern itself
is unhelpful for straightforward how-to questions that don't require
account-specific information.

**Likely root cause:** The retrieval system retrieved DM-redirection examples
for a broad query, and the generator faithfully replicated the pattern without
distinguishing between cases where DM is appropriate (account issues) vs
unnecessary (general how-to questions).

**Potential fix:** Add an intent-aware generation rule: for
`app_feature_question` and `content_not_found` intents, penalize DM
redirection in the prompt and require a direct answer before offering DM
as a fallback.

---

## Failure Mode 2 — False Auto-Handle on Security-Adjacent Messages

**Failure mode:** Messages with implicit security signals (account accessed
by a friend, logged in on wrong device) are AUTO_HANDLEd when they should
be escalated.

**Example:**
- Customer: "my friend logged into her spotify account on my iphone and
  now I can't log into mine"
- System output: AUTO_HANDLE (intent: account_access, confidence: 0.42)
- Expected: ESCALATE — shared device login with locked-out account has
  security implications

**Why it failed:** The message doesn't contain explicit security keywords
("hacked", "unauthorized", "compromised") so the escalation policy's
sensitive signal check didn't fire. The intent was classified as
`account_access` (not `account_security`) which doesn't trigger
always-escalate.

**Likely root cause:** The boundary between `account_access` and
`account_security` is fuzzy for shared-device scenarios. The keyword-based
escalation signals are too literal — they miss implicit security concerns.

**Potential fix:** Add a Gemini-based safety classifier as a pre-escalation
check that flags implicit security signals beyond keyword matching. Also
consider merging `account_access` and `account_security` into one intent
with a security_flag field.

---

## Failure Mode 3 — Low Confidence on High-Volume Intents

**Failure mode:** The TF-IDF classifier assigns low confidence (<0.6) to
messages that clearly belong to a well-represented intent, triggering
unnecessary escalation.

**Example:**
- Customer: "WHY DOES MY APP KEEP CRASHING"
- System output: ESCALATE (intent: playback_issue, confidence: 0.38)
- Expected: AUTO_HANDLE — clear playback/crash issue with strong retrieval

**Why it failed:** All-caps text and aggressive tone shift the TF-IDF
feature distribution away from the training examples, reducing confidence.
The `class_weight="balanced"` setting in LogReg also compresses probabilities
for majority classes like `playback_issue`.

**Likely root cause:** TF-IDF is case-sensitive by default and `balanced`
class weighting reduces probability mass for well-represented intents,
artificially lowering confidence scores.

**Potential fix:** Lowercase all text before classification (already done
in normalisation but verify the classifier input). Consider removing
`class_weight="balanced"` and using threshold-tuning instead. Use
calibrated probabilities (CalibratedClassifierCV) for more reliable
confidence scores.

---

## Failure Mode 4 — other_unknown Recall is Poor (0.34 F1)

**Failure mode:** Messages that should be classified as `other_unknown`
are frequently misclassified as real intents, meaning the system
auto-handles messages it shouldn't.

**Example:**
- Customer: "my account" (two words, no context)
- System output: intent: account_access (confidence: 0.31)
- Expected: other_unknown → ESCALATE for human triage

**Why it failed:** The TF-IDF classifier sees "account" and assigns it to
the closest known intent. The `other_unknown` class has very few training
examples (only 8 in the golden set) and its keyword signals overlap heavily
with real intents.

**Likely root cause:** `other_unknown` is structurally hard to learn from
positive examples — it's defined by the absence of clear signals, not their
presence. TF-IDF cannot learn this well without explicit out-of-distribution
detection.

**Potential fix:** Add a confidence floor: if max class probability < 0.35
across all intents, override to `other_unknown` regardless of top prediction.
This is a simple deterministic rule that doesn't require retraining.

---

## Failure Mode 5 — Non-English Messages AUTO_HANDLEd

**Failure mode:** Messages in non-English languages (Filipino, Spanish,
Portuguese) are classified and sometimes auto-handled when they should
either be escalated or handled by a language-aware agent.

**Example:**
- Customer: "Hindi ako makadownload ng Spotify 😭😭😭" (Filipino: "I can't
  download Spotify")
- System output: AUTO_HANDLE (intent: offline_download_issue)
- Expected: ESCALATE or route to language-appropriate support

**Example 2:**
- Customer: "Se me olvido cancelar spotify y me cargaron el mes" (Spanish:
  "I forgot to cancel Spotify and they charged me for the month")
- System output: AUTO_HANDLE (intent: subscription_billing)
- Expected: ESCALATE — billing dispute in non-English language

**Why it failed:** The data cleaning phase removed very short tweets but
did not filter by language. The TF-IDF classifier matches on shared
tokens (brand names, URLs) and makes plausible-sounding classifications.
The generated reply is always in English regardless of input language.

**Likely root cause:** No language detection step in the pipeline. The
escalation policy has no language-mismatch rule.

**Potential fix:** Add `langdetect` language detection in `src/data/clean.py`
and flag non-English messages. Add a language_mismatch escalation rule in
`policy.py` that always escalates when the detected language is not English.