# Tables deliberately excluded from regeneration

`seq_baselines.py` was corrected to the ESN calibration contract — the shared
fit/held split seed (`_MONITOR_SPLIT_SEED`), the `DEGENERATE_EPS` guard on
`sigma_err`, and the guarded `_robust_loc_scale`. Every result table carrying
`linear_ar` / `gru` / `lstm` / `tcn` rows was re-run against the corrected
code, **except the three below**. Each exclusion is deliberate and the reason
is recorded here rather than left for a reader to discover from a timestamp.

Two of the three are **frozen historical records**: retained on purpose, marked
unreproducible, and **excluded from every claim and from the reproducibility
guarantee**. They are not cited by `CLAIMS.md`, and nothing in the codebase
reads them.

---

## 1. `runtime.csv` — intentionally NOT regenerated

**Status: current and correct. Do not re-run to "refresh" it.**

It carries `gru`, `lstm` and `tcn` rows, so a text search for sequence-baseline
names finds it. That match is misleading: the file contains **only wall-clock
timings and memory footprints** — `fit_seconds`,
`step_latency_us_median`, `step_latency_us_p95`, `footprint_mb`,
`n_steps_timed`.

Two independent reasons it is excluded:

**It cannot have been affected.** The correction changed which healthy
episodes a model calibrates on and how a degenerate scale is handled. It did
not change model architecture, parameter count, reservoir size, or the number
of training steps. Fit cost and per-step latency are therefore unchanged by
construction — a GRU still trains the same 21k parameters over the same
schedule.

**Re-running it would corrupt four published claims.** These four entries in
`CLAIMS.md` read this file, and all are explicitly flagged machine-specific:

| claim | value |
|---|---|
| Primary monitor step latency is 100-999 us | `100-999 us` |
| Primary monitor state footprint (MB) | `3.95` |
| Primary monitor step latency, median us (machine-specific) | `219` |
| delta-Mahalanobis step latency, median us | `4` |

They were measured on the reference machine. Re-running on any other machine
substitutes that machine's hardware for the published measurement while
changing nothing about correctness. Regenerate this file **only** on the
reference machine, and only when the monitor's cost profile actually changes.

---

## 2 & 3. Frozen historical records — no producing script exists

These two have **no producer in the codebase**. They date from the squashed
base commit and were never refreshed when the studies around them were re-run,
so they cannot be reproduced from the code as it stands.

| file | columns | why it is orphaned |
|---|---|---|
| `real_leave_one_out_baselines.csv` | `Held_Out_Class, Monitor, AUC, FA_Rate, Delay, Precision, Recall` | `run_leave_one_out.py` now writes only `real_per_class_baselines.csv`, whose schema differs. The script that emitted this shape no longer exists. |
| `real_traces_baseline_backup.csv` | same schema as `real_traces_*.csv` | a manual backup copy, not a study output. No script names it. |

**Their numbers are stale in two separate ways:**

1. Their **sequence-baseline rows** (`linear_ar`, `gru`, `lstm`) were produced
   before the calibration correction, so they are not comparable with any other
   table in this directory.
2. Their **ESN and memoryless rows** are stale for an older, unrelated reason:
   they predate the v1.0.0 reconciliation that refreshed the other corpus
   tables.

**Standing policy for these two files:**

* **Retained** as a historical record of runs that happened. Not deleted.
* **Excluded from claims.** Neither appears in `CLAIMS.md`, and neither may be
  cited, plotted, or quoted in any document, paper or README.
* **Excluded from the reproducibility guarantee.** `BASELINE_MANIFEST.json`
  hashes them so their bytes are pinned, but a hash only proves the file has
  not changed — it does not mean the numbers can be regenerated or trusted.
* Verified consumers: **none**. `grep -rn` across `.py`, `.md`, `.tex` and
  `.json` finds no script, document, claim, figure or table that reads either.

If they are ever needed as live evidence rather than history, the work is to
restore the per-class leave-one-out reporting to `run_leave_one_out.py` and
re-run it — not to trust these files.
