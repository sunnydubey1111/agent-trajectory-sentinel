"""Kill-switch A/B: HMT-ESN vs the current channel-max baseline.

Decides whether HMT-ESN (monitor/hmt_esn.py) earns its place as the paper's
contribution, by measurement on held-out data — per the agreed criteria:

  C1  AUC improvement >= +0.02 over the baseline
  C2  mean detection lead improvement >= +1 step
  C3  same-or-better AUC with lower held-out false-alarm rate
  C4  better detection on slow goal drift (sim) / content classes (real)

Two arms:
  real  (default)  traces/real_research7b — 120 healthy / 171 injected real-
                   tool episodes, identical 60/20/20 split and 5% FA budget
                   as run_real_traces, channels (e,u,m,x).
  --sim            the controlled simulator arm (T=25-60 episodes incl. SLOW
                   goal drift — the multi-timescale hypothesis's home turf).

Alongside the HMT ablation cells the table carries NG-RC/NVAR CONTROLS
(monitor/ngrc.py): same contract, same channels, same CUSUM and channel-max
fusion, but no random reservoir at all. Every HMT cell contains a reservoir,
so the ablation grid alone cannot say whether the random recurrence is
earning its cost; the controls can.

Two protocols, and only one of them may settle an architecture question:

  default            ONE split, ONE reservoir draw. Writes
                     results/tables/hmt_ab_{real,sim}.csv and decides the
                     prespecified kill switch. Its per-cell deltas are a
                     snapshot; the split-to-split spread on this corpus is
                     several times the difference between architectures, so a
                     single-split interval around one of those deltas is not
                     evidence about the architecture.
  --replicates N     Pools N replicates, each with a fresh split AND a fresh
                     reservoir draw, reporting paired differences with a
                     bootstrap CI of the mean and Holm-corrected permutation
                     p-values. Writes results/tables/hmt_pooled_{real,sim}.csv
                     and the per-replicate frame it was computed from,
                     hmt_pooled_{real,sim}_replicates.csv. Every architecture
                     claim rests on this table.

Additive: no existing table or module is touched, and the default path is
unchanged — `hmt_ab_real.csv` re-runs byte-identical.
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
from derail.evaluation.protocol import holm_bonferroni
from derail.evaluation.stats import paired_permutation_test
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.monitor.hmt_esn import HMTESNMonitor
from derail.monitor.conceptor import ConceptorMonitor
from derail.monitor.ngrc import NGRCMonitor
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
#: prespecified primary criterion. Passing if ANY of four architectures met
#: ANY of four criteria would be 16 chances, with a content-class criterion
#: firing on any positive delta > 1e-9 - optimistic test-set search rather
#: than a decision. hmt_full is the paper's actual contribution (multi-timescale +
#: hierarchical); the other cells are exploratory ablations, reported but not
#: gating.
PRIMARY_ARCH = "hmt_full"


def _bootstrap_dauc_ci(test, scores_a, scores_b, seed=0, n_boot=2000):
    """Percentile CI of episode-AUC(a) - AUC(b) over episode resamples.

    SINGLE-SPLIT. This resamples EPISODES while holding fixed the two nuisance
    terms that actually dominate here — which healthy episodes landed in
    train/val/test, and which random reservoir was drawn. Measured on this
    corpus the split-to-split spread of episode AUC is 5x the mean absolute
    paired difference between architectures, so an interval that ignores it is
    far too narrow and will call noise significant (and, as often, hide a real
    difference). It decides the PRESPECIFIED kill switch and nothing else; no
    per-cell number from this interval may be quoted as evidence.

    Use `--replicates N` (run_pooled) for any architecture claim: it varies the
    split AND the reservoir draw per replicate and pools paired differences.
    """
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
    # Single split, single reservoir draw. The deltas below are a snapshot, not
    # evidence: over 120 replicates `hmt_mt`'s detection delta against
    # `hmt_single` ranges from -0.25 to +0.29 on split and reservoir draw
    # alone, around a pooled mean of +0.015 that does not survive correction.
    print("  exploratory (non-gating) ablations "
          "[SINGLE SPLIT — snapshot only, use --replicates to compare]:")
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


def _control_cells(std, channels):
    """NG-RC/NVAR controls: same contract, no random reservoir.

    These answer a question the HMT ablation grid cannot, because every cell
    in it contains a reservoir: is the RANDOM RECURRENCE earning its cost, or
    would a linear readout over an explicit delay embedding do as well?
    `ngrc_linear` is the strictest control (a VAR one-step predictor with no
    nonlinearity at all); the quadratic cells add NG-RC's polynomial features.
    """
    return [
        NGRCMonitor(std, channels, k=2, order=1, name="ngrc_linear"),
        NGRCMonitor(std, channels, k=1, order=2, name="ngrc_quad_k1"),
        NGRCMonitor(std, channels, k=2, order=2, name="ngrc_quad_k2"),
    ]


def _mechanism_cells(std, channels):
    """Conceptor arm: the only cell that scores something OTHER than
    prediction error.

    Every other monitor in this table — ESN, HMT, NG-RC — asks "how wrong was
    the one-step prediction". A conceptor asks "has the state left the
    subspace healthy runs occupy", which is a different failure mode and the
    proposed mechanism for slow goal drift. Reservoir hyperparameters are
    inherited from the ESN, so a delta here is the mechanism and not the
    calibration.
    """
    return [ConceptorMonitor(std, channels, seed=0, name="conceptor")]


# ---------------------------------------------------------------------------
# Pooled protocol — the only construction an architecture claim may rest on
# ---------------------------------------------------------------------------
#: Each replicate draws a fresh healthy split AND a fresh reservoir, shared by
#: every cell in that replicate so contrasts stay paired. Anything less holds
#: fixed the terms that dominate the variance: three separate conclusions in
#: this file's history (multi-timescale, depth, reservoir size) were single-
#: split artifacts that reversed sign or vanished once these axes moved.
#:
#: 30 replicates is a floor, not a target. A multi-timescale detection gain
#: significant at p=0.004 over replicates 0-29 came out at p=0.948 over
#: replicates 500-529 — a nominally significant result that did not survive an
#: independent family of the same size.
POOLED_METRICS = ("episode_auc", "detection_rate", "healthy_fa_rate",
                  "mean_lead_all")


def _pooled_cells(std, channels, seed):
    """The published ablation grid at one reservoir draw, plus the control the
    grid lacks.

    `single_fast` is bank-matched to `hmt_mt` (12 banks per channel) but spends
    them on ONE timescale, the fast one. Without it, `hmt_mt` vs `hmt_single`
    confounds three leak rates with 50% more capacity, and cannot tell "several
    timescales help" apart from "leak 0.3 is the wrong single timescale".
    """
    return [
        ChannelMaxESNMonitor(std, K=8, cusum=True, seed=seed, channels=channels),
        HMTESNMonitor(std, channels, leak_rates=(0.3,), n_layers=1, K=8,
                      seed=seed, name="hmt_single"),
        HMTESNMonitor(std, channels, leak_rates=(0.7, 0.3, 0.1), n_layers=1,
                      K=4, seed=seed, name="hmt_mt"),
        HMTESNMonitor(std, channels, leak_rates=(0.3,), n_layers=2, K=4,
                      seed=seed, name="hmt_h"),
        HMTESNMonitor(std, channels, leak_rates=(0.7, 0.3, 0.1), n_layers=2,
                      K=4, seed=seed, name="hmt_full"),
        HMTESNMonitor(std, channels, leak_rates=(0.7,), n_layers=1, K=12,
                      seed=seed, name="single_fast"),
    ]


def _paired(a: np.ndarray, b: np.ndarray, seed=0, n_boot=10000) -> dict:
    """Paired mean difference over replicates, with a bootstrap CI of that mean
    and a sign-flip permutation p-value. Pairing is by replicate, so the split
    and reservoir difficulty that dominates the raw spread cancels."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = rng_for(seed, "hmt-pooled-boot")
    boot = np.array([delta[rng.integers(0, delta.size, delta.size)].mean()
                     for _ in range(n_boot)])
    return {"paired_mean": float(delta.mean()),
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "p_perm": float(paired_permutation_test(a, b, seed=seed)["p_value"]),
            "wins": int((delta > 0).sum()), "n": int(delta.size)}


def _aggregate(frame: pd.DataFrame,
               contrasts: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Paired contrasts over the per-replicate frame.

    Each contrast is tested on all four metrics, so the family is Holm-
    corrected — an uncorrected 0.05 over four metrics is the same optimistic
    search the prespecified kill switch exists to prevent. `significant` is the
    corrected decision; the raw p is kept beside it.
    """
    out_rows = []
    for a, b in contrasts:
        stats = {m: _paired(frame[f"{m}.{a}"].to_numpy(),
                            frame[f"{m}.{b}"].to_numpy())
                 for m in POOLED_METRICS}
        holm = holm_bonferroni({m: s["p_perm"] for m, s in stats.items()})
        for m in POOLED_METRICS:
            out_rows.append({
                "contrast": f"{a} - {b}", "metric": m,
                "a_mean": float(frame[f"{m}.{a}"].mean()),
                "a_sd": float(frame[f"{m}.{a}"].std(ddof=1)),
                "b_mean": float(frame[f"{m}.{b}"].mean()),
                "b_sd": float(frame[f"{m}.{b}"].std(ddof=1)),
                **stats[m], "p_holm": holm[m]["p_holm"],
                "significant": holm[m]["reject"]})
    return pd.DataFrame(out_rows)


def run_pooled(build_split, channels: tuple[str, ...], n_replicates: int,
               out_name: str, contrasts: tuple[tuple[str, str], ...]) -> None:
    """Run the ablation grid over `n_replicates` split x reservoir draws.

    `build_split(k)` returns (train, val, test) for replicate k.
    """
    rows = []
    for k in range(n_replicates):
        train, val, test = build_split(k)
        std = Standardizer().fit(train)
        rec: dict[str, float] = {"replicate": k}
        for mon in _pooled_cells(std, channels, seed=k):
            mon.fit(train)
            theta = float(pick_threshold([mon.score_episode(ep) for ep in val],
                                         fa_budget=FA_BUDGET,
                                         warn_infeasible=False))
            scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
            summ = summarize(evaluate_alarms(test, scores, theta))
            rec[f"episode_auc.{mon.name}"] = float(episode_auc(test, scores))
            for m in POOLED_METRICS[1:]:
                rec[f"{m}.{mon.name}"] = float(summ[m])
            rec[f"det[goal_drift].{mon.name}"] = float(
                summ["per_class"].get("goal_drift", {}).get("detection_rate",
                                                            float("nan")))
        rows.append(rec)
        print(f"  replicate {k + 1}/{n_replicates}", flush=True)

    # The per-replicate frame is written alongside the contrast table: the
    # pooled means are only trustworthy if the reader can see the replicate
    # spread they were drawn from, and re-block them (a contrast can be
    # significant over one block of 30 and absent over the next).
    frame = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / out_name.replace(".csv", "_replicates.csv"),
                 index=False)
    out = _aggregate(frame, contrasts)
    out.to_csv(RESULTS / out_name, index=False)
    print(f"\n[pooled] {n_replicates} replicates -> {RESULTS / out_name}")
    for contrast in out.contrast.unique():
        print(f"\n  {contrast}")
        for _, r in out[out.contrast == contrast].iterrows():
            print(f"    {r.metric:>16s}: {r.b_mean:.3f} -> {r.a_mean:.3f}  "
                  f"paired {r.paired_mean:+.4f} "
                  f"CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] "
                  f"p={r.p_perm:.4f} holm={r.p_holm:.3f}"
                  f"{'  SIGNIFICANT' if r.significant else ''}")


#: Every contrast the pooled table reports. The first two are the architecture
#: axes (H-2 multi-timescale, H-3 depth); `single_fast` is the control that
#: separates timescale from capacity; the last two ask whether either axis beats
#: the SHIPPED monitor, which the within-HMT contrasts cannot answer.
POOLED_CONTRASTS = (("hmt_mt", "hmt_single"),
                    ("hmt_full", "hmt_mt"),
                    ("hmt_h", "hmt_single"),
                    ("hmt_mt", "single_fast"),
                    ("hmt_full", "esn_cusum_max"),
                    ("hmt_mt", "esn_cusum_max"))


def run_real(traces_dir: Path, n_replicates: int = 0) -> None:
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
    n_tr, n_va = int(round(0.6 * len(healthy))), int(round(0.2 * len(healthy)))

    def split(k: int):
        p = rng_for(k, "real-split").permutation(len(healthy))
        return ([healthy[i] for i in p[:n_tr]],
                [healthy[i] for i in p[n_tr:n_tr + n_va]],
                [healthy[i] for i in p[n_tr + n_va:]] + injected)

    if n_replicates:
        return run_pooled(split, ("e", "u", "m", "x"), n_replicates,
                          "hmt_pooled_real.csv", POOLED_CONTRASTS)

    train, val, test = split(0)
    print(f"[ab-real] {len(train)} train / {len(val)} val / "
          f"{len(test) - len(injected)} test-healthy / {len(injected)} injected")

    channels = ("e", "u", "m", "x")
    std = Standardizer().fit(train)
    monitors = ([ChannelMax(std, channels)] + _hmt_cells(std, channels)
                + _control_cells(std, channels)
                + _mechanism_cells(std, channels))
    df, all_scores = _evaluate(monitors, train, val, test)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "hmt_ab_real.csv", index=False)
    print(f"[ab-real] wrote {RESULTS / 'hmt_ab_real.csv'}")
    _verdict(df, all_scores, test, monitors[0].name,
             ("wrong_document", "malformed_json", "context_corruption"))


def run_sim(n_replicates: int = 0) -> None:
    from derail.telemetry.generator import make_dataset

    def dataset(k: int):
        # A fresh master seed per replicate: on the simulator the analogue of
        # re-splitting a fixed corpus is generating new episodes.
        cfg = DatasetConfig(n_train_healthy=240, n_val_healthy=120,
                            n_cal_healthy=0, n_cal_injected_per_class=0,
                            n_test_healthy=120, n_test_injected_per_class=40,
                            master_seed=DatasetConfig().master_seed + k)
        d = make_dataset(cfg, SimConfig())
        return d["train"], d["val"], d["test"]

    if n_replicates:
        return run_pooled(dataset, ("e", "u", "m"), n_replicates,
                          "hmt_pooled_sim.csv", POOLED_CONTRASTS)

    train, val, test = dataset(0)
    print(f"[ab-sim] {len(train)} train / {len(val)} val / {len(test)} test")

    channels = ("e", "u", "m")
    std = Standardizer().fit(train)
    monitors = ([ChannelMaxESNMonitor(std, K=8, cusum=True, seed=0)]
                + _hmt_cells(std, channels)
                + _control_cells(std, channels)
                + _mechanism_cells(std, channels))
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
    parser.add_argument("--replicates", type=int, default=0, metavar="N",
                        help="pool N split x reservoir-draw replicates instead "
                             "of running one split; required for any "
                             "architecture claim (30 is a floor, not a target)")
    args = parser.parse_args()
    if args.sim:
        run_sim(args.replicates)
    else:
        run_real(Path(args.dir), args.replicates)


if __name__ == "__main__":
    main()
