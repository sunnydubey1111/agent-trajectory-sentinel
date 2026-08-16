# QUARANTINED: `hmte_vs_baseline.csv`

**Status: not evidence. Do not cite, plot, or repeat the AUC 1.000 figure.**

`hmte_vs_baseline.csv` reports `HMTE-ESN-M` at episode AUC **1.000** against
`ChannelMax` at **0.750**. Both numbers are artifacts of the evaluation
protocol, not measurements of monitor quality. The file is retained as a
historical record of a run that happened; it is quarantined because it was
produced outside the kill-switch protocol every other monitor comparison in
this repository is held to.

## Why the number is not evidence

**1. The test set is five episodes.** `derail/experiments/evaluate_hmte.py`
runs on `traces/real/`, the 18-episode Gemini stub corpus: 17 healthy and
**1 injected** after the `T >= 4` filter. The 60/20/20 healthy split leaves
10 train / 3 val / **4 test-healthy**, and the test set is those 4 healthy
episodes plus the single injected one. An episode AUC over 4 negatives and 1
positive is decided by 4 pairwise comparisons. AUC 1.000 means the one
injected episode outscored all four healthy ones; ChannelMax's 0.750 means it
outscored three of four. The entire reported gap is **one episode moving past
one other episode.** No confidence interval on 4 comparisons excludes chance.

**2. The Mahalanobis fit is partly in-sample.** `HMTE_ESN_M_Monitor.fit`
(`derail/monitor/esn.py`) estimates its 9-D feature mean and covariance by
streaming over **all** healthy episodes passed to `fit()` — including the ones
its sub-monitors' ridge readouts were just fit on. The healthy feature
distribution is therefore measured on data the model has already seen, which
biases the healthy Mahalanobis distance downward and the detection rate upward.

**3. No threshold, no budget, no seeds, no CI.** The script computes `val` and
never uses it: there is no false-alarm-budget threshold, so detection rate and
false-alarm rate are never reported. It runs one seed. It reports no interval.
Every other monitor comparison in this repository picks a threshold on `val` at
a 5% FA budget, reports the realized FA rate, and carries a bootstrap CI.

## What would lift the quarantine

Re-run HMTE-ESN-M under `derail/experiments/run_hmt_ab.py`'s protocol:

* the `traces/real_research7b` corpus (120 healthy / 171 injected), not the
  18-episode stub;
* threshold picked on `val` at the 5% FA budget, realized FA rate reported;
* a held-out Mahalanobis fit — estimate mean/covariance on the calibration
  episodes the sub-monitors' readouts did **not** train on;
* a **pooled** interval against the `esn_cusum_max` baseline — paired
  differences over replicates that vary the data split *and* the reservoir
  draw (`run_hmt_ab.py --replicates N`), Holm-corrected across the metrics
  reported. A bootstrap over episode resamples at a single split is **not**
  sufficient: on this corpus the split-to-split spread is several times the
  difference between architectures, and three separate conclusions drawn that
  way (multi-timescale, depth, reservoir size) did not survive pooling.

Until then the honest statement about HMTE-ESN-M is: **untested.**

## Provenance

| | |
|---|---|
| produced by | `py -m derail.experiments.evaluate_hmte` |
| corpus | `traces/real/` (18 usable episodes: 17 healthy, 1 injected) |
| test set | 4 healthy + 1 injected |
| seeds | 1 |
