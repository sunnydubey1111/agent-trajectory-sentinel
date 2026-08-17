# The two layers, measured on one population

Regenerate: `py -m derail.experiments.run_layer_alignment`
Tables: `results/tables/layer_alignment_{summary,by_dataset}.csv`

The behavioural layer is reported from `run_hybrid_study` (1,002 injected
episodes, 8 datasets, 400 of them simulator) and the grounding layer from
`run_grounding_study` (874 episodes, 10 real corpora, no simulator). Those are
different populations, so a sentence pairing a figure from one with a figure
from the other is not a comparison between layers — part of any difference is
the difference in which corpora were counted.

## 1. The populations, exactly

| | episodes | corpora | simulator |
|---|---|---|---|
| behavioural study | 1,002 | 8 | 400 |
| grounding study | 874 | 10 | 0 |
| **both** | **602** | **7** | 0 |

The behavioural study holds the simulator and nothing else the grounding study
lacks. The grounding study holds three real corpora the behavioural study never
scored: `ollama_llama8b`, `real_gemini_long`, `real_research7b_long_ext`.

The intersection is a genuine matched population, not merely the same size:
merging on `(dataset, episode_id)` gives 602 rows with **zero disagreements**
on failure class, on ESN detection or on Mahalanobis detection. Where the two
studies overlap they say the same thing, so the only thing that differs is
inclusion. `load_aligned` raises if that ever stops being true.

## 2. What changes when the population is matched

Content detection, ESN alone against the content gate:

| population | n content | ESN | gate | **gain** |
|---|---|---|---|---|
| grounding study's own (10 corpora) | 313 | 0.281 | 0.578 | **+0.297** |
| **matched (7 corpora, both studies)** | 211 | 0.232 | 0.403 | **+0.171** |
| only the grounding study (3 corpora) | 102 | 0.382 | 0.941 | **+0.559** |

**The published +0.297 is a weighted average of two regimes**, +0.171 on the
population the behavioural layer is argued on and +0.559 on the corpora it is
not. The 0.126 difference between the pooled and matched figures is carried by
which corpora each study counts.

The behavioural half moves the other way, and the conclusion there survives
unchanged. Detection on non-content classes under the gate is **+0.072** on the
matched population against +0.054 pooled: the "grounding does not degrade
behavioural detection" claim is slightly *stronger* where the two layers are
comparable, not weaker.

## 3. Is the advantage caused by composition?

Partly, and the part that is has a measured mechanism rather than being an
accident of the corpus list. Two conditions have to hold for the gate to add
anything, and both are properties of the corpus:

1. **The grounding stream has to be able to see the corruption.** Across the 10
   corpora, content gain tracks that stream's own detection rate at
   **r = +0.712**. The four corpora where it detects nothing — `ollama_llama8b`,
   `gemini`, `autogen7b`, `ollama7b` — have a mean gain of **+0.008**.
2. **The behavioural monitor has to have left something to catch.** Gain tracks
   headroom (`1 - ESN detection`) at **r = +0.496**. On `ollama7b` the ESN
   already detects 1.00, so no gain is possible there whatever the gate does.

Per corpus, gain runs from **−0.029** to **+0.949**:

| corpus | in behavioural study | n | ESN | gate | gain |
|---|---|---|---|---|---|
| `real_research7b_long_ext` | no | 39 | 0.051 | 1.000 | +0.949 |
| `real_research7b_long` | yes | 18 | 0.000 | 0.667 | +0.667 |
| `real_gemini_long` | no | 31 | 0.323 | 0.935 | +0.613 |
| `real_research7b` | yes | 66 | 0.076 | 0.439 | +0.364 |
| `real_research3b` | yes | 17 | 0.294 | 0.353 | +0.059 |
| `ollama_llama8b` | no | 32 | 0.844 | 0.875 | +0.031 |
| `gemini` | yes | 21 | 0.333 | 0.333 | 0.000 |
| `autogen7b` | yes | 33 | 0.121 | 0.121 | 0.000 |
| `ollama7b` | yes | 22 | 1.000 | 1.000 | 0.000 |
| `langgraph7b` | yes | 34 | 0.176 | 0.147 | −0.029 |

Two of the three corpora the grounding study adds are high-gain research
corpora, which is why adding them raises the pooled figure. That is a real
effect of the layer on those corpora, not an error — but it is not evidence
about the population the behavioural layer was argued on.

## 4. What to quote

- **The layer comparison** — behavioural against grounding — takes the matched
  figures: **+0.171** content gain and **+0.072** behavioural change over 602
  episodes on 7 corpora. This is the only pairing where the two layers see the
  same episodes.
- **The grounding layer on its own** keeps its full-population figure of
  **+0.297** over 874 episodes, which is the right number for "what does this
  layer do across every corpus we have". It is not comparable to a behavioural
  figure computed on 1,002.
- **Neither pooled figure should be quoted without its population**, because
  the same quantity is +0.17 or +0.56 depending on which corpora are counted.

The ledger carries all three arms with their denominators
(`layers.content_gain_{shared,own,outside}`), and
`test_a_cross_layer_content_claim_names_its_population` fails if the pooled
figure stops sitting between the matched and unmatched ones — the signature of
this composition effect changing.
