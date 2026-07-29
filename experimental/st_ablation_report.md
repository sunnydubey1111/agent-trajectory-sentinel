# Final Results: Semantic Failure Detection on real_research7b

**Dataset:** 142 traces (100 healthy, 42 injected across 7 failure classes, 6 each)  
**All 3 channels live:** e+u+m (logprobs present in 142/142 traces)  
**Split:** 60 train / 20 val / 20 test-healthy + 42 injected  
**FA budget:** 5% (realized: 5% in all runs -- 1/20 healthy triggered)

---

## The result that matters

| Target Class | Baseline | MiniLM | MiniLM+x | x alone |
|--------------|:--------:|:------:|:--------:|:-------:|
| **wrong_document** | 1/6 (17%) | 1/6 (17%) | 1/6 (17%) | **3/6 (50%)** |
| **malformed_json** | 0/6 (0%) | 0/6 (0%) | **6/6 (100%)** | **6/6 (100%)** |
| **context_corruption** | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) |

---

## What worked

### malformed_json: 0% -> 100%

The `IDX_RESULT_JSON_BROKEN` feature (dim 52) does exactly what it was designed for: it flags tool results that look like JSON but fail to parse. **Perfect detection, zero false alarms.** This is the cleanest win in the study -- a domain-specific binary feature solving a specific problem completely.

### wrong_document: 17% -> 50% (x-channel alone)

The `IDX_RESULT_TASK_SIM` feature (dim 51) -- cosine similarity between the tool result embedding and the task anchor -- catches off-topic documents. The x-channel alone detected 3/6, up from 1/6 baseline. **However**, the combined `[e,u,m,x]` monitor only gets 1/6 -- the signal is diluted when fused via channel-max with the e/u/m channels.

---

## What didn't work

### MiniLM embeddings alone: zero impact

| Metric | Baseline (hash) | MiniLM |
|--------|:---:|:---:|
| Same theta? | -- | **Yes** (28.087) |
| Same confusion matrix? | -- | **Yes** (17 TA, 25 miss) |
| Any per-class change? | -- | **None** |

MiniLM projected to 32 dims through a random projection produces the same ESN prediction-error signal as the hash embedding. **The embedding upgrade is not the missing ingredient.**

### context_corruption: 0% everywhere

Completely undetected in all configurations. This class corrupts tool result *content* (shuffled words, injected spurious prices) without structural markers. Neither the embedding drift nor the result-similarity features catch it.

---

## Full class comparison

| Class | Baseline | MiniLM | MiniLM+x | x alone |
|-------|:--------:|:------:|:--------:|:-------:|
| looping | **100%** | **100%** | **100%** | 50% |
| rate_limit | **67%** | **67%** | **67%** | 17% |
| tool_cascade | 50% | 50% | **67%** | **83%** |
| timeout | 50% | 50% | 50% | 50% |
| wrong_document | 17% | 17% | 17% | **50%** |
| malformed_json | 0% | 0% | **100%** | **100%** |
| context_corruption | 0% | 0% | 0% | 0% |

---

## Overall metrics

| Config | AUC | Detection | FA | Lead | Delay |
|--------|:---:|:---------:|:--:|:----:|:-----:|
| Baseline [e,u,m] | 0.742 | 40.5% | 5% | 0.55 | 2.0 |
| MiniLM [e,u,m] | 0.775 | 40.5% | 5% | 0.55 | 2.0 |
| MiniLM [e,u,m,x] | **0.870** | **57.1%** | 5% | **1.00** | **1.0** |
| x alone | **0.907** | 50.0% | 5% | 0.52 | 3.0 |

---

## Honest verdict

| Question | Answer |
|----------|--------|
| Did MiniLM improve semantic failure detection? | **No.** Zero change in any class. |
| Did the x-channel features help? | **Yes, for 2 of 3 target classes.** |
| Which feature mattered most? | `IDX_RESULT_JSON_BROKEN` (malformed_json: 0->100%) |
| Which feature showed partial signal? | `IDX_RESULT_TASK_SIM` (wrong_document: 17->50% in x-alone) |
| Is context_corruption solved? | **No.** 0% in all configurations. |
| Is the x-channel signal diluted in channel-max fusion? | **Yes.** wrong_document is 50% with x-alone but 17% with [e,u,m,x]. |

---

## Implications

1. **The tool-result features are the real contribution**, not the embedding upgrade.
2. **Channel-max fusion dilutes the x-channel signal** for subtle semantic failures.
3. **context_corruption needs a different approach** (perplexity scoring, schema checks, n-gram coherence).
4. **MiniLM can be dropped from the claims.** The hash embedding is sufficient for the ESN pathway.

## Next steps (revised)

| Priority | Action |
|:--------:|--------|
| 1 | Investigate **fusion strategy** -- can we recover wrong_document detection? |
| 2 | Investigate **context_corruption** -- what does the corruption look like? |
| 3 | Consider **dropping MiniLM from the paper claims** |
| 4 | Cross-model validation only AFTER fusion and context_corruption are addressed |
