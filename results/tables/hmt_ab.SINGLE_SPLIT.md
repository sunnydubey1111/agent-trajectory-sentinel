# Reading `hmt_ab_real.csv` and `hmt_ab_sim.csv`

**These two tables are one split and one reservoir draw each.** They are valid
records of the prespecified kill-switch run and nothing else. Do not read a
per-cell difference in them as a property of the architecture.

## What they are for

`derail/experiments/run_hmt_ab.py` decides one prespecified question: does
`hmt_full` beat the `esn_cusum_max` baseline by ≥ 0.02 episode AUC with an
interval excluding zero? The answer on both arms is **no**. That verdict is
what these files support, and re-measuring it over pooled replicates
reproduces it.

## What they are not for

Every other column is an exploratory snapshot. The healthy episodes that land
in train/val/test, and the random reservoir that gets drawn, move the per-cell
numbers by more than the architectures differ from each other. Three
conclusions were drawn from columns in these files and none of the three
survived pooling:

| read from a single split | pooled over 120 replicates |
|---|---|
| `hmt_mt` detection 0.257 vs `hmt_single` 0.368 — multi-timescale is much worse | +0.013, and gone under Holm correction |
| `hmt_mt` `wrong_document` detection −0.14 vs baseline — the content gain "reverses" | **+0.027**, small and positive |
| `hmt_mt` sim AUC 0.8736 vs `hmt_single` 0.8679 — multi-timescale is ahead on the sim arm | **−0.010, significantly behind** (p = 0.002) |

The first two are in `hmt_ab_real.csv`, the third in `hmt_ab_sim.csv`. All
three reversed sign or vanished once the split and the reservoir draw were
allowed to vary.

To scale the problem: across 120 replicates, `hmt_mt`'s detection rate minus
`hmt_single`'s ranges from **−0.251 to +0.292** on split and reservoir draw
alone, around a pooled mean of +0.015. Any single number drawn from that
distribution — including the one in `hmt_ab_real.csv` — says more about which
split was drawn than about the two architectures.

## What to use instead

```
py -m derail.experiments.run_hmt_ab --replicates 120          # real arm
py -m derail.experiments.run_hmt_ab --sim --replicates 20     # sim arm
```

writes `hmt_pooled_{real,sim}.csv` — paired differences over replicates that
vary the data split *and* the reservoir draw, with a bootstrap CI of the mean
and Holm-corrected permutation p-values — plus the per-replicate frame
(`hmt_pooled_{real,sim}_replicates.csv`) those statistics were computed from.
**Any claim about multi-timescale banks, depth, or reservoir size must cite
the pooled table.** Evidence and reasoning:
`_internal/docs/RC_FLAVOUR_DECISION_MEMO.md` §6.1b and §6.4.

## Provenance

| | |
|---|---|
| produced by | `py -m derail.experiments.run_hmt_ab [--sim]` |
| real corpus | `traces/real_research7b` (120 healthy / 171 injected) |
| splits | 1 (`rng_for(0, "real-split")`) |
| reservoir draws | 1 (every cell `seed=0`) |
