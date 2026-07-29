"""Review point #3 - how cheap is per-deployment recalibration, really?

L4 established the deployment RULE: monitors calibrated on one agent model are
at chance on another (qwen7b -> llama8b, AUROC 0.527) while the same target
recalibrated on itself reaches 0.885. Recalibration is mandatory. It is also
label-free, so the only real question a deployer has is **how many healthy
episodes must I collect before the monitor is usable?**

The paper answered "~30 unlabeled healthy episodes". That number had no
backing artifact anywhere in the repo. This measures it.

Deployment-realistic protocol. A budget of n healthy episodes is ALL a new
deployment has, so n must cover both jobs: 75% fit the monitor, 25% set the
alarm threshold at the 5% FA budget. (Holding out a separate full-size
threshold set would answer a question no deployer can ask.) The test split is
fixed across every n, so curves are comparable. Each n is repeated over
several seeded subsamples, because at n=5 the draw matters more than the
method.

Reported per corpus: AUROC and detection against n, plus the smallest n
reaching 95% of the full-budget AUROC - the number a deployer actually needs.

Run:  py -m experimental.recalibration_cost      (free, no API, ~minutes)
Writes results/tables/recalibration_cost.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.experiments.run_hybrid_study import (
    REAL_DATASETS,
    TABLES_DIR,
    load_real,
)
from derail.monitor.hybrid import make_hybrids

FA_BUDGET = 0.05
BUDGETS = (5, 10, 15, 20, 30, 40, 50)
REPEATS = 5
CORPORA = ("ollama_llama8b", "ollama7b", "real_research7b", "real_gemini_long")


def _one(pool, test, channels, n: int, rep: int) -> list[dict]:
    """Fit+threshold on a budget of n healthy episodes; score the fixed test."""
    rng = rng_for(rep, "recal", n)
    idx = rng.permutation(len(pool))[:n]
    sample = [pool[i] for i in idx]
    n_fit = max(2, int(round(0.75 * n)))
    fit, thr = sample[:n_fit], sample[n_fit:]
    if not thr:                     # n too small to spare a threshold episode
        thr = fit
    std = Standardizer().fit(fit)
    esn, maha, _ = make_hybrids(std, channels=channels, seed=1300)
    rows = []
    for mon in (esn, maha):
        mon.fit(fit)
        theta = float(pick_threshold([mon.score_episode(ep) for ep in thr],
                                     fa_budget=FA_BUDGET))
        scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
        summ = summarize(evaluate_alarms(test, scores, theta))
        rows.append({"monitor": mon.name, "n_healthy": n, "rep": rep,
                     "auroc": float(episode_auc(test, scores)),
                     "detection_rate": float(summ["detection_rate"]),
                     "healthy_fa_rate": float(summ["healthy_fa_rate"])})
    return rows


def main() -> int:
    out = []
    for name in CORPORA:
        data, channels = load_real(REAL_DATASETS[name])
        # The calibration pool is everything a deployer could collect without
        # labels: the healthy train and val splits. Test stays fixed.
        pool = data["train"] + data["val"]
        test = data["test"]
        budgets = [n for n in BUDGETS if n <= len(pool)] or [len(pool)]
        if len(pool) not in budgets:
            budgets.append(len(pool))       # the full-budget reference
        print(f"[recal] {name}: pool={len(pool)} healthy, test={len(test)}, "
              f"budgets={budgets}", flush=True)
        for n in budgets:
            for rep in range(REPEATS):
                for row in _one(pool, test, channels, n, rep):
                    out.append({"dataset": name, "pool": len(pool), **row})

    df = pd.DataFrame(out)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    agg = (df.groupby(["dataset", "monitor", "n_healthy"])
             .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                  detection_rate=("detection_rate", "mean"),
                  healthy_fa_rate=("healthy_fa_rate", "mean"))
             .reset_index().round(4))
    path = TABLES_DIR / "recalibration_cost.csv"
    agg.to_csv(path, index=False)

    print("\n[recal] ESN: what does the calibration budget actually buy?")
    print("  (AUROC = ranking quality; FA = realized healthy false-alarm rate "
          f"against a {FA_BUDGET:.0%} budget)")
    fa_ns, auroc_ns = [], []
    for name in agg["dataset"].unique():
        sub = agg[(agg["dataset"] == name)
                  & (agg["monitor"].str.contains("esn"))].sort_values("n_healthy")
        if sub.empty:
            continue
        full = sub.iloc[-1]
        ok_auroc = sub[sub["auroc"] >= 0.95 * full["auroc"]]
        # The operating point is the expensive part: realized FA within 2x the
        # budget is the weakest defensible bar for "deployable".
        ok_fa = sub[sub["healthy_fa_rate"] <= 2 * FA_BUDGET]
        n_auroc = int(ok_auroc.iloc[0]["n_healthy"]) if not ok_auroc.empty else None
        n_fa = int(ok_fa.iloc[0]["n_healthy"]) if not ok_fa.empty else None
        print(f"\n  {name}  (pool {int(full['n_healthy'])})")
        print("    " + "  ".join(f"n={int(r.n_healthy)}:{r.auroc:.2f}/{r.healthy_fa_rate:.2f}"
                                 for r in sub.itertuples()))
        print(f"    95% of full AUROC at n={n_auroc};  "
              f"FA <= {2 * FA_BUDGET:.0%} at n={n_fa}")
        if n_auroc:
            auroc_ns.append(n_auroc)
        if n_fa:
            fa_ns.append(n_fa)
    if fa_ns:
        print(f"\n[recal] ranking is cheap (95% AUROC at n="
              f"{min(auroc_ns)}-{max(auroc_ns)}); the OPERATING POINT is the "
              f"cost (FA within 2x budget at n={min(fa_ns)}-{max(fa_ns)})")
    print(f"[recal] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
