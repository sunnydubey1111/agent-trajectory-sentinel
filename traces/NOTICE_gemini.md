# Notice for the Gemini-generated corpora

Applies to `traces/real/` (18 episodes) and `traces/real_gemini_long/`
(125 episodes) — 143 episodes in total.

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

Anyone reusing these 143 episodes must not use them to train language models,
or to develop any model or service competing with the Gemini API or Google AI
Studio. This condition is in addition to the repository's MIT licence, which
covers the code and the trace format but cannot grant rights over the model
output these episodes contain.

The remaining 2,680 episodes are unaffected: the `qwen2.5` corpora are
Apache-2.0 and unconditioned, and the `llama3.1:8b` corpora carry the Llama 3.1
Community License terms recorded in `DATA_CARD.md`.
