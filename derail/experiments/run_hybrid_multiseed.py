"""Hybrid-monitor multiseed stability (exp/hybrid-multiseed).

Re-runs the full hybrid study at four extra monitor-side seeds (7, 101,
202, 303 — the study's replication convention) and aggregates them with
the published seed-0 tables into `hybrid_multiseed.csv` (mean +- std per
dataset x monitor, plus per-monitor grand means).

What varies with the seed: the ESN reservoir initialization (all K
members), the logistic cross-fit fold assignment, and the simulator's
master seed (fresh generated data, matching run_multiseed's convention).
What stays frozen: the real trace datasets, their train/val/test splits
(rng_for(0, "real-split")), thresholds protocol, and metrics — per the
evaluation policy.

Run:  py -m derail.experiments.run_hybrid_multiseed        (~3 h CPU)
Keeps hybrid_seed<N>_benchmark.csv per seed; the other per-seed tables
are removed after aggregation to avoid clutter.
"""

from __future__ import annotations

import pandas as pd

from derail.experiments.run_hybrid_study import TABLES_DIR, main as study_main

EXTRA_SEEDS = (7, 101, 202, 303)
METRICS = ("auroc", "auprc", "detection_rate", "healthy_fa_rate",
           "mean_lead_all")


def main() -> None:
    # Every seed - including seed 0 - is RE-RUN here rather than read from the
    # already-published hybrid_benchmark.csv. Reusing that file assumed it came
    # from the same code and config as the later seeds, with no check; a stale
    # seed-0 table could silently enter the mean/std. Running all
    # seeds through one code path removes the provenance gap.
    frames = []
    for seed in (0, *EXTRA_SEEDS):
        prefix = f"hybrid_seed{seed}"
        print(f"\n===== seed {seed} =====")
        study_main(["--seed", str(seed), "--out-prefix", prefix])
        frames.append(pd.read_csv(TABLES_DIR / f"{prefix}_benchmark.csv")
                      .assign(seed=seed))
        for suffix in ("per_class", "stats", "diagnosis", "explain"):
            (TABLES_DIR / f"{prefix}_{suffix}.csv").unlink(missing_ok=True)

    allruns = pd.concat(frames, ignore_index=True)
    g = allruns.groupby(["dataset", "monitor"], sort=False)[list(METRICS)]
    summary = g.agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.round(4).reset_index()
    summary.to_csv(TABLES_DIR / "hybrid_multiseed.csv", index=False)

    grand = (allruns.groupby(["monitor", "seed"])["auroc"].mean()
             .groupby("monitor").agg(["mean", "std"]).round(4)
             .sort_values("mean", ascending=False))
    print("\n[multiseed] grand-mean AUROC per monitor over "
          f"{1 + len(EXTRA_SEEDS)} seeds (mean +- std across seeds):")
    print(grand.to_string())
    print(f"\n[multiseed] wrote {TABLES_DIR / 'hybrid_multiseed.csv'}")


if __name__ == "__main__":
    main()
