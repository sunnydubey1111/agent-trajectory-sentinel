# Notice for the Gemini-generated corpora

Applies to every `gemini-2.5-flash` episode in this repository — **330 in
total**:

- `traces/real/` — 18 episodes
- `traces/real_gemini_long/` — 125 episodes
- `traces/` itself — **187 episodes**, listed in the top-level
  `traces/manifest.json` rather than in a corpus subdirectory. These are easy to
  miss: the data card, the claims ledger and the Hugging Face export all
  enumerate corpora by globbing `traces/*/manifest.json`, which matches
  subdirectories only, so this set appears in none of their totals. It is
  committed here all the same, and this notice covers it.

## What these are

Agent trajectories in which the acting model was `gemini-2.5-flash`, called
through the Gemini API on the **unpaid (free) tier**. Every other corpus in
this repository was produced by locally served open-weight models and is not
covered by this notice.

Two consequences of the unpaid tier apply to the episodes themselves:

- Google states that "to help with quality and improve our products, human
  reviewers may read, annotate, and process your API input and output." The
  prompts here are synthetic booking and research tasks with no personal,
  confidential or sensitive content, which is what the unpaid tier requires.
- The tier gates `response_logprobs`, which is why these corpora carry the
  `e+m` telemetry channels only.

## Google's restriction, quoted

From the Gemini API Additional Terms of Service, under *Use Restrictions*:

> You may not use the Services to develop models that compete with the
> Services (e.g., Gemini API or Google AI Studio).

## How they were used here

These traces were used to fit and evaluate a **one-class anomaly detector over
step telemetry** — an echo-state-network ensemble scoring numeric signals such
as entropy, tool-call structure and embedding drift. It is not a language
model, it does not generate text, and it does not compete with the Gemini API
or Google AI Studio.

## Condition on reuse

Anyone reusing **any of these 330 episodes** — all three sets listed above, not
only the 143 that sit in corpus subdirectories — must not use them to train
language models, or to develop any model or service competing with the Gemini
API or Google AI Studio. This condition is in addition to the repository's MIT
licence, which covers the code and the trace format but cannot grant rights
over the model output these episodes contain.

## Which counts appear where

The three numbers in circulation are all correct and all mean different things.
Stating them together, because a reader who meets them one at a time will read
one as contradicting another:

| count | what it is |
|---|---|
| **330** | every Gemini episode in this repository — the scope of this notice |
| **143** | the Gemini episodes in corpus subdirectories (`real/` 18 + `real_gemini_long/` 125), so the only ones `DATA_CARD.md`, the claims ledger and the Hugging Face export can see |
| **187** | the Gemini episodes in the top-level `traces/manifest.json`, invisible to those three tools but committed and covered here |

`DATA_CARD.md` totals 2,823 episodes; 143 of those are Gemini, leaving 2,680
unaffected. The 187 are not in that 2,823 at all. The `qwen2.5` corpora are
Apache-2.0 and unconditioned, and the `llama3.1:8b` corpora carry the Llama 3.1
Community License terms recorded in `DATA_CARD.md`.
