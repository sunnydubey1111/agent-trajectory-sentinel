"""Grounded-monitor multiseed stability.

Re-runs the full grounding study at four extra monitor-side seeds and
aggregates with the published seed-0 tables:

  - `grounding_multiseed.csv` — mean ± std per dataset x monitor.
  - `grounding_multiseed_criterion.csv` — THE deployment criterion per
    seed: pooled content / behavioral detection of each grounded fusion
    vs the ungrounded hybrid_weighted50, from the per-episode diagnosis
    records. The claim under test: content improves and behavioral does
    not degrade AT EVERY SEED, not just on average.

Seed varies ESN reservoirs and cross-fit fold assignment; data splits are
frozen (rng_for(0, "real-split")) per protocol. seed 0 = published.

Run:  py -m derail.experiments.run_grounding_multiseed   (~1.5 h CPU)
"""

from __future__ import annotations

import pandas as pd

from derail.experiments.run_hybrid_study import TABLES_DIR
from derail.experiments.run_grounding_study import main as study_main

EXTRA_SEEDS = (7, 101, 202, 303)
METRICS = ("auroc", "auprc", "detection_rate", "healthy_fa_rate",
           "mean_lead_all")
GROUNDED = ("hybrid_weighted_g", "hybrid_content_gate", "hybrid_adaptive",
            "hybrid_logistic_g", "dual_budget")
REF = "hybrid_weighted50"


def _criterion_rows(diag: pd.DataFrame, seed: int) -> list[dict]:
    rows = []
    for mon in GROUNDED:
        col = f"det_{mon}"
        if col not in diag.columns:
            continue
        for label, sub in (("content", diag[diag.is_content]),
                           ("behavioral", diag[~diag.is_content])):
            rows.append({
                "seed": seed, "monitor": mon, "group": label,
                "n": len(sub),
                "det_ref": round(float(sub[f"det_{REF}"].mean()), 4),
                "det_grounded": round(float(sub[col].mean()), 4),
                "delta": round(float(sub[col].mean()
                                     - sub[f"det_{REF}"].mean()), 4)})
    return rows


def main() -> None:
    # Re-run every seed, including seed 0, rather than reading the published
    # grounding_benchmark.csv - which was assumed, unchecked, to match the
    # later seeds' code and config.
    bench, crit = [], []
    for seed in (0, *EXTRA_SEEDS):
        prefix = f"grounding_seed{seed}"
        print(f"\n===== seed {seed} =====")
        study_main(["--seed", str(seed), "--out-prefix", prefix])
        bench.append(pd.read_csv(TABLES_DIR / f"{prefix}_benchmark.csv")
                     .assign(seed=seed))
        crit += _criterion_rows(
            pd.read_csv(TABLES_DIR / f"{prefix}_diagnosis.csv"), seed)
        for suffix in ("per_class", "stats", "ablation", "diagnosis"):
            (TABLES_DIR / f"{prefix}_{suffix}.csv").unlink(missing_ok=True)

    allruns = pd.concat(bench, ignore_index=True)
    g = allruns.groupby(["dataset", "monitor"], sort=False)[list(METRICS)]
    summary = g.agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.round(4).reset_index().to_csv(
        TABLES_DIR / "grounding_multiseed.csv", index=False)

    cdf = pd.DataFrame(crit)
    cdf.to_csv(TABLES_DIR / "grounding_multiseed_criterion.csv", index=False)

    print("\n[multiseed] criterion across seeds (delta = grounded - "
          f"{REF} pooled detection):")
    pv = cdf.pivot_table(index=["monitor", "group"], columns="seed",
                         values="delta")
    print(pv.round(3).to_string())
    print("\n[multiseed] criterion verdict per grounded monitor:")
    for mon in GROUNDED:
        sub = cdf[cdf.monitor == mon]
        c = sub[sub.group == "content"]["delta"]
        b = sub[sub.group == "behavioral"]["delta"]
        if not len(c):
            continue
        ok = bool((c > 0).all() and (b >= 0).all())
        print(f"  {mon:>20s}: content min {c.min():+.3f}, behavioral min "
              f"{b.min():+.3f} -> {'PASS at every seed' if ok else 'FAILS'}")
    print(f"\n[multiseed] wrote grounding_multiseed(_criterion).csv")


if __name__ == "__main__":
    main()
