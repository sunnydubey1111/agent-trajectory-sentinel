"""Kill-switch A/B: HMT-ESN vs the current channel-max baseline.

Decides whether HMT-ESN (monitor/hmt_esn.py) earns its place as the paper's
contribution, by measurement on held-out data — per the agreed criteria:

  C1  AUC improvement >= +0.02 over the baseline
  C2  mean detection lead improvement >= +1 step
  C3  same-or-better AUC with lower held-out false-alarm rate
  C4  better detection on slow goal drift (sim) / content classes (real)

Two arms:
  real  (default)  traces/real_research7b — 100 healthy / 42 injected real-
                   tool episodes, identical 60/20/20 split and 5% FA budget
                   as run_real_traces, channels (e,u,m,x).
  --sim            the controlled simulator arm (T=25-60 episodes incl. SLOW
                   goal drift — the multi-timescale hypothesis's home turf).

Writes results/tables/hmt_ab_{real,sim}.csv. Additive: no existing table or
module is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import DatasetConfig, Episode, SimConfig, Standardizer, rng_for
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.monitor.hmt_esn import HMTESNMonitor
from derail.telemetry.adapter import load_trace_jsonl

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
RESULTS = Path(__file__).resolve().parents[2] / "results" / "tables"
FA_BUDGET = 0.05


def _evaluate(monitors, train, val, test, fa_budget=FA_BUDGET):
    rows, all_scores = [], {}
    for mon in monitors:
        mon.fit(train)
        val_scores = [mon.score_episode(ep) for ep in val]
        theta = float(pick_threshold(val_scores, fa_budget=fa_budget))
        scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
        all_scores[mon.name] = scores
        summ = summarize(evaluate_alarms(test, scores, theta))
        row = {"monitor": mon.name,
               "detection_rate": summ["detection_rate"],
               "healthy_fa_rate": summ["healthy_fa_rate"],
               "mean_lead_all": summ["mean_lead_all"],
               "median_delay": summ["median_delay"],
               "episode_auc": float(episode_auc(test, scores))}
        row.update({f"det[{fc}]": v["detection_rate"]
                    for fc, v in summ["per_class"].items()})
        rows.append(row)
        print(f"  {mon.name:>28s}: det={row['detection_rate']:.2f} "
              f"fa={row['healthy_fa_rate']:.2f} "
              f"lead={row['mean_lead_all']:.1f} auc={row['episode_auc']:.3f}")
    return pd.DataFrame(rows), all_scores


#: The kill switch is decided on ONE prespecified architecture and ONE
#: prespecified primary criterion. The old switch passed if ANY of
#: four architectures met ANY of four criteria - 16 chances, with C4 firing on
#: any positive content-class delta > 1e-9 - which is optimistic test-set
#: search. hmt_full is the paper's actual contribution (multi-timescale +
#: hierarchical); the other cells are exploratory ablations, reported but not
#: gating.
PRIMARY_ARCH = "hmt_full"


def _bootstrap_dauc_ci(test, scores_a, scores_b, seed=0, n_boot=2000):
    """Percentile CI of episode-AUC(a) - AUC(b) over episode resamples."""
    from sklearn.metrics import roc_auc_score
    ids = [ep.episode_id for ep in test]
    y = np.array([0 if ep.is_healthy else 1 for ep in test])
    ma = np.array([float(np.max(scores_a[i])) for i in ids])
    mb = np.array([float(np.max(scores_b[i])) for i in ids])
    rng = rng_for(seed, "hmt-dauc")
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(test), len(test))
        if np.unique(y[idx]).size < 2:
            continue
        deltas.append(roc_auc_score(y[idx], ma[idx])
                      - roc_auc_score(y[idx], mb[idx]))
    d = np.asarray(deltas)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _verdict(df: pd.DataFrame, all_scores: dict, test, baseline_name: str,
             content_classes: tuple[str, ...]) -> dict:
    base = df[df.monitor == baseline_name].iloc[0]
    print(f"\n=== KILL-SWITCH VERDICT (baseline: {baseline_name}, "
          f"auc={base.episode_auc:.3f}, fa={base.healthy_fa_rate:.2f}, "
          f"lead={base.mean_lead_all:.1f}) ===")

    # -- PRIMARY (prespecified) decision: hmt_full vs baseline on AUC ---------
    if PRIMARY_ARCH not in set(df.monitor) or PRIMARY_ARCH not in all_scores:
        print(f"  primary architecture {PRIMARY_ARCH!r} absent; no verdict")
        return {"passed": False, "reason": "primary architecture absent"}
    d_auc, lo, hi = _bootstrap_dauc_ci(test, all_scores[PRIMARY_ARCH],
                                       all_scores[baseline_name])
    # Pass iff the prespecified architecture improves AUC by the prespecified
    # margin AND the improvement is significant (paired dAUC CI excludes 0).
    passed = bool(d_auc >= 0.02 and lo > 0.0)
    print(f"  PRIMARY {PRIMARY_ARCH}: dAUC={d_auc:+.3f} "
          f"(95% CI [{lo:+.3f}, {hi:+.3f}]) -> "
          f"{'PASS' if passed else 'FAIL — kill-switch'} "
          f"(needs margin >=0.02 AND CI excluding 0)")

    # -- EXPLORATORY ablations: reported, never gating -----------------------
    print("  exploratory (non-gating) ablations:")
    for _, row in df.iterrows():
        if row.monitor in (baseline_name, PRIMARY_ARCH):
            continue
        e_auc = row.episode_auc - base.episode_auc
        e_lead = row.mean_lead_all - base.mean_lead_all
        cdeltas = [f"{fc}:{row[f'det[{fc}]'] - base[f'det[{fc}]']:+.2f}"
                   for fc in content_classes if f"det[{fc}]" in df.columns
                   and not pd.isna(row.get(f"det[{fc}]"))]
        print(f"    {row.monitor:>26s}: dAUC={e_auc:+.3f} dlead={e_lead:+.1f} "
              f"{' '.join(cdeltas)}")
    return {"passed": passed, "primary": PRIMARY_ARCH, "dauc": d_auc,
            "dauc_ci_lo": lo, "dauc_ci_hi": hi}


def _hmt_cells(std, channels):
    return [
        HMTESNMonitor(std, channels, leak_rates=(0.3,), n_layers=1, K=8,
                      seed=0, name="hmt_single"),
        HMTESNMonitor(std, channels, leak_rates=(0.7, 0.3, 0.1), n_layers=1,
                      K=4, seed=0, name="hmt_mt"),
        HMTESNMonitor(std, channels, leak_rates=(0.3,), n_layers=2, K=4,
                      seed=0, name="hmt_h"),
        HMTESNMonitor(std, channels, leak_rates=(0.7, 0.3, 0.1), n_layers=2,
                      K=4, seed=0, name="hmt_full"),
    ]


def run_real(traces_dir: Path) -> None:
    from derail.experiments.run_real_traces import ChannelMax

    manifest = json.loads((traces_dir / "manifest.json").read_text("utf-8"))
    episodes: list[Episode] = []
    for e in manifest:
        if e["T"] < 4:
            continue
        episodes.append(load_trace_jsonl(
            traces_dir / e["file"], episode_id=e["episode_id"], tau=e["tau"],
            failure_class=e["failure_class"],
            severity=None if e["tau"] is None else 0.5,
            use_sentence_transformers=False, extended=True))
    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_tr, n_va = int(round(0.6 * len(healthy))), int(round(0.2 * len(healthy)))
    train = [healthy[i] for i in perm[:n_tr]]
    val = [healthy[i] for i in perm[n_tr:n_tr + n_va]]
    test = [healthy[i] for i in perm[n_tr + n_va:]] + injected
    print(f"[ab-real] {len(train)} train / {len(val)} val / "
          f"{len(test) - len(injected)} test-healthy / {len(injected)} injected")

    channels = ("e", "u", "m", "x")
    std = Standardizer().fit(train)
    monitors = [ChannelMax(std, channels)] + _hmt_cells(std, channels)
    df, all_scores = _evaluate(monitors, train, val, test)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "hmt_ab_real.csv", index=False)
    print(f"[ab-real] wrote {RESULTS / 'hmt_ab_real.csv'}")
    _verdict(df, all_scores, test, monitors[0].name,
             ("wrong_document", "malformed_json", "context_corruption"))


def run_sim() -> None:
    from derail.monitor.esn import ChannelMaxESNMonitor
    from derail.telemetry.generator import make_dataset

    cfg = DatasetConfig(n_train_healthy=240, n_val_healthy=120,
                        n_cal_healthy=0, n_cal_injected_per_class=0,
                        n_test_healthy=120, n_test_injected_per_class=40)
    data = make_dataset(cfg, SimConfig())
    train, val = data["train"], data["val"]
    test = data["test"]
    print(f"[ab-sim] {len(train)} train / {len(val)} val / {len(test)} test")

    channels = ("e", "u", "m")
    std = Standardizer().fit(train)
    monitors = ([ChannelMaxESNMonitor(std, K=8, cusum=True, seed=0)]
                + _hmt_cells(std, channels))
    df, all_scores = _evaluate(monitors, train, val, test)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "hmt_ab_sim.csv", index=False)
    print(f"[ab-sim] wrote {RESULTS / 'hmt_ab_sim.csv'}")
    _verdict(df, all_scores, test, monitors[0].name, ("goal_drift",))


def main() -> None:
    parser = argparse.ArgumentParser(prog="py -m derail.experiments.run_hmt_ab")
    parser.add_argument("--dir", default=str(TRACES_DIR / "real_research7b"))
    parser.add_argument("--sim", action="store_true",
                        help="run the simulator arm instead of the real arm")
    args = parser.parse_args()
    if args.sim:
        run_sim()
    else:
        run_real(Path(args.dir))


if __name__ == "__main__":
    main()
