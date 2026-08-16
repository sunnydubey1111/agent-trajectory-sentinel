"""Cross-model transfer: monitors calibrated on one agent model, deployed on
another. Same framework, same tasks, same tools — only the agent MODEL swaps.

Two conditions on the TARGET test split, same protocol:
  transfer   fit + threshold on the SOURCE dataset (deployed calibration),
             score target episodes with NO refit;
  in-domain  fit + threshold on the target's own healthy splits.

The cross-framework matrix already showed monitors do not transfer across
FRAMEWORKS without refit; this answers the finer question.

Two arms are run:
  within-family  qwen2.5:7b -> qwen2.5:3b   (the original study)
  cross-FAMILY   qwen2.5:7b -> llama3.1:8b  (external validity)

Cross-family is the stronger test: a different tokenizer, chat template and
tool-calling style, not just a smaller sibling.

Run:  py -m derail.experiments.run_model_transfer
      py -m derail.experiments.run_model_transfer --source ollama7b \\
          --target ollama_llama8b --label "qwen7b->llama8b" \\
          --out model_transfer_family
Writes results/tables/<out>.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from derail.common import Standardizer
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.experiments.run_grounding_study import GRD_DIM_NAMES, _view51
from derail.experiments.run_hybrid_study import (
    REAL_DATASETS,
    TABLES_DIR,
    load_real,
)
from derail.monitor.grounding import GroundingMonitor, HybridContentGate
from derail.monitor.hybrid import HybridWeighted, make_hybrids

FA_BUDGET = 0.05


def _fit_stack(train, channels):
    """Return [(monitor, grounded?)]. Behavioural monitors are fit on the
    published 51-dim view; the grounded gate masks its behavioural submodels
    to the same 51 dims (behav_slice), so its pre-fit submodels must
    also be fit on the 51-dim view (fitting on the full 60 both double-counts
    grounding and breaks the 51-vs-60 scoring shape)."""
    train51 = _view51(train)
    std51 = Standardizer().fit(train51)
    esn, maha, hybrids2 = make_hybrids(std51, channels=channels)
    esn.fit(train51)
    maha.fit(train51)
    weighted = next(h for h in hybrids2 if isinstance(h, HybridWeighted))
    weighted.fit(train51)
    std56 = Standardizer().fit(train51)
    esn56, maha56, _ = make_hybrids(std56, channels=channels)
    esn56.fit(train51)
    maha56.fit(train51)
    grd_cont = GroundingMonitor(dims=GRD_DIM_NAMES[:-1],
                                name="grounding_cont")
    grd_cont.fit(train)
    gate = HybridContentGate(esn56, maha56, grd_cont, std56, subs_prefit=True)
    gate.fit(train)
    return [(esn, False), (maha, False), (weighted, False), (gate, True)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_model_transfer",
        description="Cross-model transfer: calibrate on one agent model, "
                    "deploy on another.")
    ap.add_argument("--source", default="real_research7b",
                    help="dataset the monitors are CALIBRATED on")
    ap.add_argument("--target", default="real_research3b",
                    help="dataset they are DEPLOYED on (test split)")
    ap.add_argument("--label", default=None,
                    help="short name for the transfer arm, e.g. '7b->8b'")
    ap.add_argument("--out", default="model_transfer",
                    help="table filename stem (default: model_transfer)")
    args = ap.parse_args(argv)

    for name in (args.source, args.target):
        if name not in REAL_DATASETS:
            raise SystemExit(f"unknown dataset {name!r}; known: "
                             f"{sorted(REAL_DATASETS)}")
    data3, ch3 = load_real(REAL_DATASETS[args.target], grounding=True)
    data7, ch7 = load_real(REAL_DATASETS[args.source], grounding=True)
    label = args.label or f"{args.source}->{args.target}"
    test = data3["test"]
    test51 = _view51(test)
    rows = []
    for cond, (train, val, channels) in {
        f"in-domain({args.target})": (data3["train"], data3["val"], ch3),
        f"transfer({label})": (data7["train"], data7["val"], ch7),
    }.items():
        val51 = _view51(val)
        for mon, grounded in _fit_stack(train, channels):
            # plain monitors score the 51-dim view; grounded gate the full 60.
            v_eps, t_eps = (val, test) if grounded else (val51, test51)
            theta = float(pick_threshold(
                [mon.score_episode(ep) for ep in v_eps], fa_budget=FA_BUDGET))
            scores = {ep.episode_id: mon.score_episode(ep) for ep in t_eps}
            summ = summarize(evaluate_alarms(t_eps, scores, theta))
            y = np.array([0 if ep.is_healthy else 1 for ep in t_eps])
            mx = np.array([float(np.max(scores[ep.episode_id]))
                           for ep in t_eps])
            rows.append({
                "condition": cond, "monitor": mon.name,
                "auroc": round(float(episode_auc(t_eps, scores)), 4),
                "auprc": round(float(average_precision_score(y, mx)), 4),
                "detection_rate": round(summ["detection_rate"], 4),
                "healthy_fa_rate": round(summ["healthy_fa_rate"], 4),
                "mean_lead_all": round(summ["mean_lead_all"], 4)})
            r = rows[-1]
            print(f"  {cond:>17s} {r['monitor']:>18s}: "
                  f"auroc={r['auroc']:.3f} det={r['detection_rate']:.2f} "
                  f"fa={r['healthy_fa_rate']:.2f}")
    out = TABLES_DIR / f"{args.out}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[transfer] wrote {out}")


if __name__ == "__main__":
    main()
