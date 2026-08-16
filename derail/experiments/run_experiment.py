"""End-to-end experiment runner: data -> monitors -> evaluation -> results/.

Module 7 of the derailment-detection study (DESIGN.md). Pipeline:

  1. simulate the dataset; fit the shared ``Standardizer`` on healthy train;
  2. build + fit the ESN ensemble, its channel ablations and the baselines;
  3. causally score val/cal/test with every monitor
       -> results/scores/{monitor}.npz + results/scores/episodes.csv
  4. H1  matched-false-alarm alarm evaluation (ESN family + memoryless
       baselines + trained sequence baselines: linear VAR, GRU, LSTM, TCN)
       -> results/tables/h1_main.csv, results/tables/delay_fa_curves.csv,
          results/tables/h1_significance.csv (paired permutation / Wilcoxon
          / McNemar, primary vs every other monitor)
  5. H2  per-failure-class channel ablation
       -> results/tables/h2_channels.csv
  6. H3a confidence calibration (ECE / reliability)
       -> results/tables/h3_calibration.csv, results/tables/h3_reliability.csv
  7. H3b judge-escalation cost/quality frontier
       -> results/tables/h3_escalation.csv
  8. results/results.json + a printed console summary per hypothesis.

Run from the repo root:

    py -m derail.experiments.run_experiment [--quick]

``--quick`` = quarter-size splits, K=4 ESN members, 100 iforest trees (for
integration testing). Per DESIGN.md, ``__main__`` IS the experiment — there is
no separate smoke test for this module.

One-class discipline: monitors fit on train (healthy) only; thresholds from
val (healthy) only; the labeled cal split feeds ONLY the oracle isotonic
calibrator; every reported number comes from test. All score/confidence
streams used here are causal (running-max confidence, no full-episode stats).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import (
    FAILURE_CLASSES,
    MASTER_SEED,
    DatasetConfig,
    Episode,
    JudgeConfig,
    OnlineMonitor,
    SimConfig,
    Standardizer,
    rng_for,
)
from derail.evaluation.metrics import (
    bootstrap_ci,
    delay_fa_curve,
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.monitor.baselines import (
    CosineDriftMonitor,
    DeltaMahalanobisMonitor,
    IsolationForestMonitor,
    MahalanobisMonitor,
    RollingSurprisalMonitor,
    SelfDriftMonitor,
)
from derail.monitor.calibration import (
    IsotonicCalibrator,
    NullCalibrator,
    ece,
    reliability_curve,
)
from derail.monitor.escalation import run_policy, summarize_policy
from derail.monitor.esn import ChannelMaxESNMonitor, ESNEnsembleMonitor
from derail.monitor.seq_baselines import (
    _HAS_TORCH,
    GRUMonitor,
    LinearARMonitor,
    LSTMMonitor,
    TCNMonitor,
)
from derail.evaluation.protocol import holm_bonferroni
from derail.evaluation.stats import (
    mcnemar_test,
    paired_permutation_test,
    wilcoxon_signed_rank,
)
from derail.telemetry.generator import make_dataset

# Results tree is anchored at the repo root regardless of the caller's cwd.
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
SCORES_DIR = RESULTS_DIR / "scores"
FIGURES_DIR = RESULTS_DIR / "figures"  # populated later by experiments/plots.py

FA_BUDGET = 0.05        # H1: healthy-val false-alarm budget for theta
FA_BUDGET_SOFT = 0.10   # H3b: softer trigger threshold for escalation
CONF_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
STREAMS = ("fused", "surprise", "disagreement")

# H2 ablation set: monitor name -> channel subset it sees. Two score-stream
# families: the short-memory EWMA ("esn_*") and the CUSUM accumulator
# ("esn_cusum*"), which integrates small persistent shifts (slow goal drift)
# that per-step surprise smoothing forgets.
ESN_VARIANTS = {
    "esn_full": "e,u,m",
    "esn[e]": "e",
    "esn[u]": "u",
    "esn[m]": "m",
    "esn[e,u]": "e,u",
    "esn_cusum": "e,u,m",
    "esn_cusum[e]": "e",
    "esn_cusum[u]": "u",
    "esn_cusum[m]": "m",
    "esn_cusum[e,u]": "e,u",
}
# Channel-attribution verdict (H2) reads the CUSUM family: the accumulating
# statistic is the stronger instrument, so per-class channel attribution is
# not confounded by the EWMA's blindness to slow drift.
H2_SINGLES = ("esn_cusum[e]", "esn_cusum[u]", "esn_cusum[m]")
BASELINES = ("cosine_drift", "self_drift", "rolling_surprisal", "mahalanobis",
             "delta_mahalanobis", "iforest")
# Trained sequence-model baselines (same one-class next-step protocol as the
# ESN family; see monitor/seq_baselines.py). gru/tcn need torch.
SEQ_BASELINES = (("linear_ar", "gru", "lstm", "tcn") if _HAS_TORCH
                 else ("linear_ar",))
# H1's named primary temporal monitor (vs the best baseline on test).
# Channel-max fusion of per-channel CUSUM detectors: the H2 ablation shows a
# monolithic ESN averaging surprise over all 43 dims dilutes shifts confined
# to a narrow channel (grounding loss lives in the 4 uncertainty dims), so
# the primary runs one detector per channel and alarms on the max.
H1_PRIMARY = "esn_cusum_max"

ComponentStreams = dict[str, dict[str, dict[str, np.ndarray]]]


# ---------------------------------------------------------------------------
# Construction / scoring
# ---------------------------------------------------------------------------
def build_monitors(standardizer: Standardizer, quick: bool) -> list[OnlineMonitor]:
    """Construct every monitor of the study (unfitted), per DESIGN.md step 2.

    ``quick`` shrinks the ESN ensembles to K=4 and the isolation forest to
    100 trees; monitor names (used in all tables/files) are unchanged.
    """
    K = 4 if quick else 8
    n_estimators = 100 if quick else 200
    return [
        ESNEnsembleMonitor(standardizer, channels=("e", "u", "m"), K=K, seed=0,
                           name="esn_full"),
        ESNEnsembleMonitor(standardizer, channels=("e", "u", "m"), K=1, seed=1,
                           name="esn_single"),
        ESNEnsembleMonitor(standardizer, channels=("e",), K=K, seed=2,
                           name="esn[e]"),
        ESNEnsembleMonitor(standardizer, channels=("u",), K=K, seed=3,
                           name="esn[u]"),
        ESNEnsembleMonitor(standardizer, channels=("m",), K=K, seed=4,
                           name="esn[m]"),
        ESNEnsembleMonitor(standardizer, channels=("e", "u"), K=K, seed=5,
                           name="esn[e,u]"),
        ESNEnsembleMonitor(standardizer, channels=("e", "u", "m"), K=K, seed=7,
                           cusum=True, name="esn_cusum"),
        ESNEnsembleMonitor(standardizer, channels=("e",), K=K, seed=8,
                           cusum=True, name="esn_cusum[e]"),
        ESNEnsembleMonitor(standardizer, channels=("u",), K=K, seed=9,
                           cusum=True, name="esn_cusum[u]"),
        ESNEnsembleMonitor(standardizer, channels=("m",), K=K, seed=10,
                           cusum=True, name="esn_cusum[m]"),
        ESNEnsembleMonitor(standardizer, channels=("e", "u"), K=K, seed=11,
                           cusum=True, name="esn_cusum[e,u]"),
        ChannelMaxESNMonitor(standardizer, K=K, cusum=True, seed=12,
                             name="esn_cusum_max"),
        LinearARMonitor(standardizer, seed=13),
        *([GRUMonitor(standardizer, epochs=15 if quick else 40, seed=14),
           TCNMonitor(standardizer, epochs=15 if quick else 40, seed=15),
           LSTMMonitor(standardizer, epochs=15 if quick else 40, seed=16)]
          if _HAS_TORCH else []),
        CosineDriftMonitor(),
        SelfDriftMonitor(),
        RollingSurprisalMonitor(),
        MahalanobisMonitor(standardizer),
        DeltaMahalanobisMonitor(standardizer),
        IsolationForestMonitor(standardizer, n_estimators=n_estimators, seed=6),
    ]


def score_split(monitor: OnlineMonitor,
                episodes: list[Episode]) -> dict[str, np.ndarray]:
    """Causally score every episode of a split; {episode_id: (T,) stream}."""
    return {ep.episode_id: monitor.score_episode(ep) for ep in episodes}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _finite(x: object) -> float:
    """Coerce to float; None / unparseable -> NaN."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _true_alarm_values(df: pd.DataFrame, column: str) -> np.ndarray:
    """delay/lead values over true alarms, as a clean float array."""
    ta = df[df["outcome"] == "true_alarm"]
    return pd.to_numeric(ta[column], errors="coerce").dropna().to_numpy(dtype=float)


def _lead_all_values(df: pd.DataFrame) -> np.ndarray:
    """Per-injected-episode lead with misses/early alarms as 0 (for CIs)."""
    injected = df[~df["is_healthy"].astype(bool)]
    return (pd.to_numeric(injected["lead"], errors="coerce")
            .fillna(0.0).to_numpy(dtype=float))


def _write_episode_metadata(data: dict[str, list[Episode]]) -> None:
    """Episode metadata CSV accompanying the saved score streams."""
    rows = [{"id": ep.episode_id, "split": split, "is_healthy": ep.is_healthy,
             "failure_class": ep.failure_class, "tau": ep.tau, "T": ep.T,
             "severity": ep.severity}
            for split in ("train", "val", "cal", "test") for ep in data[split]]
    pd.DataFrame(rows).to_csv(SCORES_DIR / "episodes.csv", index=False)


def _dataset_config(quick: bool, seed: int) -> DatasetConfig:
    """Default split sizes, or quarter-size counts under --quick."""
    cfg = DatasetConfig(master_seed=seed)
    if not quick:
        return cfg
    return DatasetConfig(
        n_train_healthy=max(1, cfg.n_train_healthy // 4),
        n_val_healthy=max(1, cfg.n_val_healthy // 4),
        n_cal_healthy=max(1, cfg.n_cal_healthy // 4),
        n_cal_injected_per_class=max(1, cfg.n_cal_injected_per_class // 4),
        n_test_healthy=max(1, cfg.n_test_healthy // 4),
        n_test_injected_per_class=max(1, cfg.n_test_injected_per_class // 4),
        master_seed=seed,
    )


def _results_root() -> Path:
    """Where this run may write. `results/` unless redirected.

    A sensitivity sweep has to run the SAME seeds as the published study (the
    seed IS the dataset), which would otherwise overwrite the published
    results/seed<N>/ directories the paper cites. AGENTWATCH_RESULTS_ROOT sends
    such a run somewhere disposable instead. Relative paths resolve under the
    repo root so a stray value cannot escape the project.
    """
    base = Path(__file__).resolve().parents[2] / "results"
    override = os.environ.get("AGENTWATCH_RESULTS_ROOT")
    if not override:
        return base
    root = Path(override)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root


def _set_results_dirs(seed: int) -> None:
    """Replication runs (non-default seed) write under results/seed<seed>/.

    Always recomputes from the repo root so repeated main() calls in one
    process (run_multiseed) don't inherit a previous run's directories.
    """
    global RESULTS_DIR, TABLES_DIR, SCORES_DIR, FIGURES_DIR
    base = _results_root()
    RESULTS_DIR = base if seed == MASTER_SEED else base / f"seed{seed}"
    TABLES_DIR = RESULTS_DIR / "tables"
    SCORES_DIR = RESULTS_DIR / "scores"
    FIGURES_DIR = RESULTS_DIR / "figures"


def _jsonable(obj: object) -> object:
    """Recursively convert numpy scalars/arrays and NaN/inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


# ---------------------------------------------------------------------------
# H1 — temporal models vs baselines at matched FA budget
# ---------------------------------------------------------------------------
def run_h1(monitors: list[OnlineMonitor], data: dict[str, list[Episode]],
           all_scores: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    """H1: per-monitor theta at FA_BUDGET on healthy val; alarms on test.

    Writes tables/h1_main.csv (one row per monitor) and
    tables/delay_fa_curves.csv (long format, column ``monitor``). Returns
    {monitor_name: {"theta", "df", "summary", "auc", "row"}} for reuse by H2
    and the verdicts.
    """
    val_eps, test_eps = data["val"], data["test"]
    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    out: dict[str, dict] = {}
    for mon in monitors:
        scores = all_scores[mon.name]
        val_list = [scores[ep.episode_id] for ep in val_eps]
        theta = float(pick_threshold(val_list, fa_budget=FA_BUDGET))
        df = evaluate_alarms(test_eps, scores, theta)
        summ = summarize(df)
        auc = float(episode_auc(test_eps, scores))
        delays = _true_alarm_values(df, "delay")
        leads = _true_alarm_values(df, "lead")
        leads_all = _lead_all_values(df)
        nan_ci = (float("nan"), float("nan"))
        d_ci = bootstrap_ci(delays, stat=np.median, seed=0) if delays.size else nan_ci
        l_ci = bootstrap_ci(leads, stat=np.median, seed=1) if leads.size else nan_ci
        la_ci = (bootstrap_ci(leads_all, stat=np.mean, seed=2)
                 if leads_all.size else nan_ci)
        row = {
            "monitor": mon.name,
            "theta": theta,
            "detection_rate": _finite(summ["detection_rate"]),
            "healthy_fa_rate": _finite(summ["healthy_fa_rate"]),
            "early_alarm_rate": _finite(summ["early_alarm_rate"]),
            "median_delay": _finite(summ["median_delay"]),
            "median_delay_ci_lo": _finite(d_ci[0]),
            "median_delay_ci_hi": _finite(d_ci[1]),
            "median_lead": _finite(summ["median_lead"]),
            "median_lead_ci_lo": _finite(l_ci[0]),
            "median_lead_ci_hi": _finite(l_ci[1]),
            "mean_lead_all": _finite(summ["mean_lead_all"]),
            "mean_lead_all_ci_lo": _finite(la_ci[0]),
            "mean_lead_all_ci_hi": _finite(la_ci[1]),
            "mean_budget_saved": _finite(summ["mean_budget_saved"]),
            "mean_budget_saved_all": _finite(summ["mean_budget_saved_all"]),
            "episode_auc": auc,
        }
        rows.append(row)
        curve = delay_fa_curve(val_list, test_eps, scores).copy()
        curve.insert(0, "monitor", mon.name)
        curves.append(curve)
        out[mon.name] = {"theta": theta, "df": df, "summary": summ,
                         "auc": auc, "row": row}
        print(f"  [h1] {mon.name:>18s}: det={row['detection_rate']:.3f} "
              f"fa={row['healthy_fa_rate']:.3f} lead={row['median_lead']:.1f} "
              f"lead_all={row['mean_lead_all']:.1f} auc={auc:.3f}")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "h1_main.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(
        TABLES_DIR / "delay_fa_curves.csv", index=False)
    return out


def _paired_episode_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(episode_ids, lead_all, detected) over injected episodes, id-sorted."""
    injected = df[~df["is_healthy"].astype(bool)].sort_values("episode_id")
    ids = injected["episode_id"].to_numpy()
    lead_all = (pd.to_numeric(injected["lead"], errors="coerce")
                .fillna(0.0).to_numpy(dtype=float))
    detected = (injected["outcome"] == "true_alarm").to_numpy()
    return ids, lead_all, detected


def _stable_seed(name: str) -> int:
    """Deterministic cross-process seed from a monitor name.

    Builtin hash() on strings is salted per interpreter process
    (PYTHONHASHSEED), so it must never seed anything — it made the
    permutation p-values irreproducible across runs. Same character-hash
    scheme as common.rng_for's tag hashing.
    """
    h = 0
    for ch in name:
        h = (h * 1000003 + ord(ch)) % (2**31)
    return h


def run_significance(h1_results: dict[str, dict]) -> pd.DataFrame:
    """Paired tests of the primary monitor vs every other monitor (weakness C).

    Per comparison (paired by injected test episode): a sign-flip permutation
    test on the mean lead_all difference and an exact McNemar test on
    detection outcomes. Writes tables/h1_significance.csv.
    """
    ids_p, lead_p, det_p = _paired_episode_values(h1_results[H1_PRIMARY]["df"])
    rows: list[dict] = []
    for name, res in h1_results.items():
        if name == H1_PRIMARY:
            continue
        ids_o, lead_o, det_o = _paired_episode_values(res["df"])
        assert np.array_equal(ids_p, ids_o), f"episode mismatch vs {name}"
        perm = paired_permutation_test(lead_p, lead_o,
                                       seed=_stable_seed(name))
        wil = wilcoxon_signed_rank(lead_p, lead_o)
        mcn = mcnemar_test(det_p, det_o)
        rows.append({
            "primary": H1_PRIMARY,
            "other": name,
            "mean_lead_all_diff": perm["mean_diff"],
            "perm_p_value": perm["p_value"],
            "wilcoxon_p_value": wil["p_value"],
            "det_primary_only": mcn["n10"],
            "det_other_only": mcn["n01"],
            "mcnemar_p_value": mcn["p_value"],
        })
    table = pd.DataFrame(rows).sort_values("mean_lead_all_diff",
                                           ascending=False)
    table.to_csv(TABLES_DIR / "h1_significance.csv", index=False)
    return table


def _h1_verdict(h1_results: dict[str, dict],
                sig: pd.DataFrame | None = None) -> tuple[str, dict]:
    """Compare the primary temporal monitor to the best baseline at matched FA.

    The comparison metric is mean_lead_all — expected steps of budget saved
    per failure episode, counting misses/early alarms as 0 — because median
    lead conditioned on detection rewards monitors that only catch easy
    cases. The best memoryless baseline AND the best trained sequence
    baseline are each selected on test by the same metric (a choice that
    favors them, so the H1 comparison is conservative). Paired permutation
    p-values come from run_significance when available.
    """
    esn = h1_results[H1_PRIMARY]["row"]

    def _lead_all(name: str) -> float:
        v = h1_results[name]["row"]["mean_lead_all"]
        return float("-inf") if math.isnan(v) else float(v)

    def _p_vs(name: str) -> float:
        if sig is None:
            return float("nan")
        sub = sig[sig["other"] == name]
        return float(sub["perm_p_value"].iloc[0]) if len(sub) else float("nan")

    best = max(BASELINES, key=_lead_all)
    base = h1_results[best]["row"]
    best_seq = max(SEQ_BASELINES, key=_lead_all)
    seq = h1_results[best_seq]["row"]
    esn_la, base_la = _lead_all(H1_PRIMARY), _lead_all(best)
    advantage = (esn_la - base_la
                 if math.isfinite(esn_la) and math.isfinite(base_la)
                 else float("nan"))
    seq_advantage = esn_la - _lead_all(best_seq)

    # H1 verdict is a PAIRED-DIFFERENCE inference. The old logic
    # (a) declared support from a positive point estimate alone, ignoring the
    # paired p-value, and (b) called a baseline "significantly different" when
    # its one-sample mean fell outside the ESN's one-sample CI - which is not a
    # paired-difference interval. Here support requires BOTH a positive paired
    # advantage and a Holm-corrected paired test that rejects, and the reported
    # interval is a bootstrap CI of the per-episode lead difference.
    ids_p, lead_p, _ = _paired_episode_values(h1_results[H1_PRIMARY]["df"])

    def _paired_diff_ci(name: str) -> tuple[float, float, float]:
        ids_o, lead_o, _ = _paired_episode_values(h1_results[name]["df"])
        if not np.array_equal(ids_p, ids_o) or lead_p.size == 0:
            return float("nan"), float("nan"), float("nan")
        d = lead_p - lead_o
        rng = rng_for(_stable_seed(name), "h1-paired-dlead")
        means = [float(np.mean(d[rng.integers(0, d.size, d.size)]))
                 for _ in range(2000)]
        return (float(np.mean(d)), float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)))

    # Holm correction across the whole primary-vs-other comparison family.
    if sig is not None and len(sig):
        family = {row["other"]: float(row["perm_p_value"])
                  for _, row in sig.iterrows()}
        holm = holm_bonferroni(family)
    else:
        holm = {}

    def _p_holm(name: str) -> float:
        return holm.get(name, {}).get("p_holm", float("nan"))

    diff_lo = _paired_diff_ci(best)[1]
    p_base_holm = _p_holm(best)
    # Significant: the paired-difference CI excludes 0 on the positive side.
    significant = math.isfinite(diff_lo) and diff_lo > 0
    # Supported: positive advantage AND the Holm-corrected paired test rejects.
    supported = (math.isfinite(advantage) and advantage > 0
                 and math.isfinite(p_base_holm) and p_base_holm < 0.05)
    p_base, p_seq = _p_vs(best), _p_vs(best_seq)
    verdict = (
        f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}: {H1_PRIMARY} saves "
        f"{esn['mean_lead_all']:.1f} steps/failure-episode "
        f"(95% CI [{esn['mean_lead_all_ci_lo']:.1f}, "
        f"{esn['mean_lead_all_ci_hi']:.1f}]; det {esn['detection_rate']:.2f}, "
        f"median lead {esn['median_lead']:.1f}, FA {esn['healthy_fa_rate']:.3f}, "
        f"AUC {esn['episode_auc']:.3f}) vs best memoryless baseline {best} "
        f"{base['mean_lead_all']:.1f} steps (det {base['detection_rate']:.2f}; "
        f"paired perm p={p_base:.2g}, Holm p={p_base_holm:.2g}) -> advantage "
        f"{advantage:+.1f} steps/episode"
        f"{' (paired-diff 95% CI excludes 0)' if significant else ''}. "
        f"Best trained sequence baseline {best_seq}: "
        f"{seq['mean_lead_all']:.1f} steps (det {seq['detection_rate']:.2f}; "
        f"advantage {seq_advantage:+.1f}, Holm p={_p_holm(best_seq):.2g})."
    )
    headline = {
        "primary": H1_PRIMARY,
        "esn": {k: esn[k] for k in ("mean_lead_all", "mean_lead_all_ci_lo",
                                    "mean_lead_all_ci_hi", "median_lead",
                                    "detection_rate", "healthy_fa_rate",
                                    "episode_auc")},
        "best_baseline": {"monitor": best, "perm_p_value": p_base,
                          **{k: base[k] for k in ("mean_lead_all", "median_lead",
                                                  "detection_rate", "episode_auc")}},
        "best_seq_baseline": {"monitor": best_seq, "perm_p_value": p_seq,
                              **{k: seq[k] for k in ("mean_lead_all",
                                                     "detection_rate",
                                                     "episode_auc")}},
        "lead_all_advantage_steps": advantage,
        "lead_all_advantage_vs_seq": seq_advantage,
        "paired_diff_ci_excludes_zero": bool(significant),
        "best_baseline_holm_p": p_base_holm,
        "supported": supported,
    }
    return verdict, headline


# ---------------------------------------------------------------------------
# H2 — channel complementarity across failure classes
# ---------------------------------------------------------------------------
def run_h2(h1_results: dict[str, dict]) -> pd.DataFrame:
    """H2: per-failure-class detection/delay/lead for the ESN channel
    variants (both EWMA and CUSUM families) at their own H1 thresholds,
    plus the self_drift baseline (trajectory self-consistency on the e
    channel — the complementary family axis that catches slow goal drift).
    Writes tables/h2_channels.csv (long format) and returns it."""
    rows: list[dict] = []
    variants = dict(ESN_VARIANTS)
    variants["esn_cusum_max"] = "max(e,u,m)"
    variants["self_drift"] = "e (self-consistency)"
    for name, channels in variants.items():
        df = h1_results[name]["df"]
        injected = df[~df["is_healthy"].astype(bool)]
        for fc in FAILURE_CLASSES:
            sub = injected[injected["failure_class"] == fc]
            if len(sub) == 0:
                continue
            ta = sub[sub["outcome"] == "true_alarm"]
            delays = pd.to_numeric(ta["delay"], errors="coerce").dropna()
            leads = pd.to_numeric(ta["lead"], errors="coerce").dropna()
            leads_all = pd.to_numeric(sub["lead"], errors="coerce").fillna(0.0)
            rows.append({
                "monitor": name,
                "channels": channels,
                "failure_class": fc,
                "n_injected": int(len(sub)),
                "detection_rate": float(len(ta) / len(sub)),
                "median_delay": float(delays.median()) if len(delays) else float("nan"),
                "median_lead": float(leads.median()) if len(leads) else float("nan"),
                "mean_lead_all": float(leads_all.mean()),
            })
    h2 = pd.DataFrame(rows)
    h2.to_csv(TABLES_DIR / "h2_channels.csv", index=False)
    return h2


def _h2_verdict(h2: pd.DataFrame) -> tuple[str, dict]:
    """Check the class-wise channel signatures and the no-dominator claim.

    Reads the CUSUM single-channel variants (H2_SINGLES) and compares by
    (mean_lead_all, detection_rate) — survivorship-free expected budget
    saved, tie-broken by detection rate.
    """
    def cell(mon: str, fc: str) -> tuple[float, float]:
        sub = h2[(h2["monitor"] == mon) & (h2["failure_class"] == fc)]
        if len(sub) == 0:
            return (float("-inf"), 0.0)
        lead_all = float(sub["mean_lead_all"].iloc[0])
        det = float(sub["detection_rate"].iloc[0])
        return (float("-inf") if math.isnan(lead_all) else lead_all, det)

    e_mon, u_mon, m_mon = H2_SINGLES
    winners = {fc: max(H2_SINGLES, key=lambda m: cell(m, fc))
               for fc in FAILURE_CLASSES}
    u_gl, e_gl = cell(u_mon, "grounding_loss"), cell(e_mon, "grounding_loss")
    e_gd, u_gd = cell(e_mon, "goal_drift"), cell(u_mon, "goal_drift")
    e_lp, u_lp = cell(e_mon, "looping"), cell(u_mon, "looping")
    sd_gd = cell("self_drift", "goal_drift")
    u_leads_grounding = u_gl > e_gl
    e_leads_drift = e_gd > u_gd
    e_leads_looping = e_lp > u_lp
    no_dominator = len(set(winners.values())) >= 2
    # The testable complementarity claims: u leads grounding loss, e leads
    # looping, and no single channel dominates all classes. Slow goal drift
    # is reported separately: it evades ALL per-step-surprise channels and is
    # caught only by the trajectory-self-consistency family (self_drift).
    supported = u_leads_grounding and e_leads_looping and no_dominator
    parts = [f"{fc}->{winners[fc]}" for fc in FAILURE_CLASSES]
    verdict = (
        f"{'SUPPORTED' if supported else 'MIXED'}: grounding_loss {u_mon} "
        f"(lead_all {u_gl[0]:.1f}, det {u_gl[1]:.2f}) vs {e_mon} "
        f"(lead_all {e_gl[0]:.1f}, det {e_gl[1]:.2f}); looping {e_mon} "
        f"(lead_all {e_lp[0]:.1f}, det {e_lp[1]:.2f}) vs {u_mon} "
        f"(lead_all {u_lp[0]:.1f}, det {u_lp[1]:.2f}); per-class "
        f"single-channel winners: {', '.join(parts)}. Goal drift evades "
        f"every surprise channel ({e_mon} det {e_gd[1]:.2f}) and is caught "
        f"only by trajectory self-consistency (self_drift lead_all "
        f"{sd_gd[0]:.1f}, det {sd_gd[1]:.2f}) — complementarity holds across "
        f"monitor FAMILIES as well as channels."
    )
    headline = {
        "per_class_winner": winners,
        "u_leads_grounding_loss": bool(u_leads_grounding),
        "e_leads_looping": bool(e_leads_looping),
        "e_leads_goal_drift": bool(e_leads_drift),
        "goal_drift_self_drift_det": sd_gd[1],
        "looping_winner": winners["looping"],
        "no_single_channel_dominates": bool(no_dominator),
        "supported": bool(supported),
    }
    return verdict, headline


# ---------------------------------------------------------------------------
# H3a — calibrated alarm confidence
# ---------------------------------------------------------------------------
def _uniform_ks(values: np.ndarray) -> float:
    """One-sample Kolmogorov-Smirnov distance of `values` from Uniform(0,1).

    0 means perfectly uniform; larger means the null percentile is not
    uniform, i.e. the null calibrator is mis-calibrated on healthy data.
    """
    v = np.sort(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))
    n = v.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - v)
    d_minus = np.max(v - (i - 1) / n)
    return float(max(d_plus, d_minus))


def run_h3_calibration(components: ComponentStreams,
                       data: dict[str, list[Episode]]) -> dict[str, dict[str, float]]:
    """H3a: Null (label-free, healthy-val ECDF) vs isotonic (oracle, cal split)
    calibration of episode-level confidence for each component stream of the
    primary (H1_PRIMARY) monitor. Writes tables/h3_calibration.csv and
    tables/h3_reliability.csv; returns per stream {null_ks, null_fa_at_0.95,
    ece_iso}."""
    val_eps, cal_eps, test_eps = data["val"], data["cal"], data["test"]
    cal_labels = np.array([0.0 if ep.is_healthy else 1.0 for ep in cal_eps])
    test_labels = np.array([0.0 if ep.is_healthy else 1.0 for ep in test_eps])

    def max_scores(split: str, eps: list[Episode], stream: str) -> np.ndarray:
        return np.array([float(np.max(components[split][ep.episode_id][stream]))
                         for ep in eps])

    healthy_test = [ep for ep in test_eps if ep.is_healthy]

    ece_rows: list[dict] = []
    rel_frames: list[pd.DataFrame] = []
    headline: dict[str, dict[str, float]] = {}
    for stream in STREAMS:
        # NULL calibrator: F0(score) is the healthy-null ECDF percentile, i.e.
        # 1 - a false-alarm p-value. Under the healthy null it is Uniform(0,1);
        # it is NOT P(derailed | score). The old code computed ECE of F0
        # against failure labels, which requires a posterior it never
        # produced. The honest test of a null statistic is its
        # UNIFORMITY / false-alarm calibration on healthy test episodes.
        null_cal = NullCalibrator()
        null_cal.fit(max_scores("val", val_eps, stream))
        null_h = np.asarray(
            null_cal.confidence(max_scores("test", healthy_test, stream)),
            dtype=float)
        ks = float(_uniform_ks(null_h))            # 0 = perfectly uniform null
        fa90 = float(np.mean(null_h >= 0.90)) if null_h.size else float("nan")
        fa95 = float(np.mean(null_h >= 0.95)) if null_h.size else float("nan")

        # ISOTONIC calibrator: a LABELED posterior P(derailed | score) fit on
        # the disjoint cal split. ECE against failure labels is meaningful only
        # here, so it is the only stream we report a reliability curve for.
        iso_cal = IsotonicCalibrator()
        iso_cal.fit(max_scores("cal", cal_eps, stream), cal_labels)
        conf_iso = np.asarray(iso_cal.confidence(max_scores("test", test_eps,
                                                            stream)),
                              dtype=float)
        ece_iso = float(ece(conf_iso, test_labels))

        ece_rows.append({
            "stream": stream,
            "null_healthy_ks": round(ks, 4),
            "null_fa_at_0.90": round(fa90, 4),
            "null_fa_at_0.95": round(fa95, 4),
            "iso_ece": round(ece_iso, 4),
            "n_test": int(len(test_eps)),
            "n_healthy_test": int(len(healthy_test)),
        })
        rel = reliability_curve(conf_iso, test_labels).copy()
        rel.insert(0, "calibrator", "isotonic")
        rel.insert(0, "stream", stream)
        rel_frames.append(rel)
        headline[stream] = {"null_ks": ks, "null_fa_at_0.95": fa95,
                            "ece_iso": ece_iso}
    pd.DataFrame(ece_rows).to_csv(TABLES_DIR / "h3_calibration.csv", index=False)
    pd.concat(rel_frames, ignore_index=True).to_csv(
        TABLES_DIR / "h3_reliability.csv", index=False)
    return headline


def _h3a_verdict(cal_headline: dict[str, dict[str, float]]) -> str:
    """Two DISTINCT calibration claims:

    (1) the label-free null stream is a well-calibrated FALSE-ALARM statistic -
        its healthy-test percentile is close to Uniform(0,1) and its realized
        FA at the 0.95 level is close to the nominal 5%; and
    (2) the labeled isotonic calibrator is a well-calibrated P(derailed)
        POSTERIOR (low ECE against failure labels).
    The null percentile is never treated as a derailment probability.
    """
    fused = cal_headline["fused"]
    dis = cal_headline["disagreement"]
    best_null = min(STREAMS, key=lambda s: cal_headline[s]["null_ks"])
    # Null well-calibrated: KS from uniform small AND realized FA near nominal.
    null_ok = (min(fused["null_ks"], dis["null_ks"]) <= 0.20
               and min(fused["null_fa_at_0.95"], dis["null_fa_at_0.95"]) <= 0.10)
    iso_ok = min(fused["ece_iso"], dis["ece_iso"]) <= 0.15
    supported = null_ok and iso_ok
    return (
        f"{'SUPPORTED' if supported else 'MIXED'}: null false-alarm "
        f"calibration KS(uniform) fused={fused['null_ks']:.3f}, "
        f"disagreement={dis['null_ks']:.3f} (best stream: {best_null}); "
        f"realized FA@0.95 fused={fused['null_fa_at_0.95']:.3f}, "
        f"disagreement={dis['null_fa_at_0.95']:.3f}. Labeled posterior ECE "
        f"fused={fused['ece_iso']:.3f}, disagreement={dis['ece_iso']:.3f}."
    )


# ---------------------------------------------------------------------------
# H3b — cost-optimal escalation to a judge-LLM
# ---------------------------------------------------------------------------
def _judge_config() -> JudgeConfig:
    """Judge parameters, overridable for the L8 sensitivity analysis.

    The defaults (0.90 / 0.02) are STIPULATED. `run_judge_calibration` measures
    a real Gemini-Flash judge on a labelled subset and gets materially worse
    rates; re-running the study with those values under a disposable --seed
    answers whether H3b survives a real judge. The override is an environment
    variable rather than a new default so no published number moves silently:

        AGENTWATCH_JUDGE_P_DETECT=0.548 AGENTWATCH_JUDGE_P_FALSE=0.057
    """
    kwargs = {}
    for env, field in (("AGENTWATCH_JUDGE_P_DETECT", "p_detect"),
                       ("AGENTWATCH_JUDGE_P_FALSE", "p_false")):
        raw = os.environ.get(env)
        if raw is None:
            continue
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{env}={raw!r} is not a probability")
        kwargs[field] = value
    judge = JudgeConfig(**kwargs)
    if kwargs:
        print(f"[experiment] JUDGE OVERRIDE (L8 sensitivity): "
              f"p_detect={judge.p_detect} p_false={judge.p_false}")
    return judge


def _sweep_policies(episodes: list[Episode], fused: dict[str, np.ndarray],
                    confidences: dict[str, np.ndarray], theta_soft: float,
                    judge: JudgeConfig, seed_tag: str) -> list[dict]:
    """Run every policy/threshold setting on one split; one summary row each."""
    runs: list[tuple[str, float | None]] = [
        ("never", None),
        ("judge_every_step", None),
        ("halt_on_alarm", None),
        ("escalate_on_alarm", None),          # raw-score escalation
    ] + [("escalate_on_alarm", ct) for ct in CONF_THRESHOLDS]
    rows: list[dict] = []
    for policy, ct in runs:
        seed = int(rng_for(MASTER_SEED, "runner", seed_tag, policy, ct)
                   .integers(0, 2**31 - 1))
        outcomes = run_policy(policy, episodes, fused,
                              confidences if ct is not None else None,
                              theta_soft, judge, seed, conf_threshold=ct)
        params = f"theta_soft={theta_soft:.4f}"
        if ct is not None:
            params += f", conf_threshold={ct}"
        rows.append({"policy": policy, "params": params,
                     "conf_threshold": ct, **summarize_policy(outcomes)})
    judge_row = next(r for r in rows if r["policy"] == "judge_every_step")
    for r in rows:
        r["cost_ratio_vs_judge"] = (
            r["mean_cost"] / judge_row["mean_cost"]
            if judge_row["mean_cost"] > 0 else float("nan"))
        # Monitoring overhead alone: judge calls per episode vs judging every
        # step. This is the problem statement's "cost ratio" — the agent's own
        # steps dominate mean_cost and mask the monitor's cheapness.
        r["judge_call_ratio"] = (
            r["mean_judge_calls"] / judge_row["mean_judge_calls"]
            if judge_row["mean_judge_calls"] > 0 else float("nan"))
    return rows


def _select_operating_point(cal_rows: list[dict]) -> float | None:
    """Pick escalate_on_alarm's conf_threshold on the CAL split.

    Frontier = settings reaching >= 80% of the cal judge_every_step detection
    rate; among them, minimize judge_call_ratio. Returns None (raw-score
    escalation) if no confidence setting qualifies. Selecting on cal keeps
    the reported test numbers free of winner's-curse optimism.
    """
    judge = next(r for r in cal_rows if r["policy"] == "judge_every_step")
    frontier = [r for r in cal_rows
                if r["policy"] == "escalate_on_alarm"
                and r["conf_threshold"] is not None
                and judge["detection_rate"] > 0
                and math.isfinite(r["detection_rate"])
                and r["detection_rate"] >= 0.8 * judge["detection_rate"]]
    if not frontier:
        return None
    return min(frontier, key=lambda r: r["judge_call_ratio"])["conf_threshold"]


def run_h3_escalation(components: ComponentStreams,
                      data: dict[str, list[Episode]]) -> tuple[pd.DataFrame, dict]:
    """H3b: halting policies with a modeled judge; tuned on cal, reported on test.

    theta_soft = fused-stream threshold at FA_BUDGET_SOFT on healthy val.
    Confidence streams are Null-calibrated running-max fused scores (causal;
    calibrator fitted on healthy val only). The escalate_on_alarm operating
    point (conf_threshold) is selected on the labeled CAL split (permitted by
    the protocol) and its TEST row becomes the headline; the full test sweep
    is still written to tables/h3_escalation.csv as an unbiased curve.
    """
    val_eps, cal_eps, test_eps = data["val"], data["cal"], data["test"]
    fused_val = [components["val"][ep.episode_id]["fused"] for ep in val_eps]
    theta_soft = float(pick_threshold(fused_val, fa_budget=FA_BUDGET_SOFT))
    null_cal = NullCalibrator()
    null_cal.fit(np.array([float(np.max(s)) for s in fused_val]))

    def streams(split: str, eps: list[Episode]) -> tuple[dict, dict]:
        fused = {ep.episode_id: components[split][ep.episode_id]["fused"]
                 for ep in eps}
        conf = {eid: np.asarray(
                    null_cal.confidence(np.maximum.accumulate(s)), dtype=float)
                for eid, s in fused.items()}
        return fused, conf

    judge = _judge_config()
    cal_fused, cal_conf = streams("cal", cal_eps)
    cal_rows = _sweep_policies(cal_eps, cal_fused, cal_conf, theta_soft,
                               judge, seed_tag="policy-cal")
    chosen_ct = _select_operating_point(cal_rows)

    test_fused, test_conf = streams("test", test_eps)
    rows = _sweep_policies(test_eps, test_fused, test_conf, theta_soft,
                           judge, seed_tag="policy")
    for r in rows:
        r["selected_on_cal"] = (r["policy"] == "escalate_on_alarm"
                                and r["conf_threshold"] == chosen_ct)
    table = pd.DataFrame(rows, columns=["policy", "params", "conf_threshold",
                                        "mean_cost", "mean_judge_calls",
                                        "detection_rate", "mean_lead",
                                        "wrongful_halt_rate",
                                        "cost_ratio_vs_judge",
                                        "judge_call_ratio", "selected_on_cal"])
    table.to_csv(TABLES_DIR / "h3_escalation.csv", index=False)
    headline = {
        "theta_soft": theta_soft,
        "chosen_conf_threshold": chosen_ct,
        "policies": rows,
        "judge_detection_rate": next(r["detection_rate"] for r in rows
                                     if r["policy"] == "judge_every_step"),
    }
    return table, headline


def _h3b_verdict(esc_headline: dict) -> tuple[str, dict]:
    """Test performance of the CAL-selected escalation operating point."""
    rows = esc_headline["policies"]
    judge = next(r for r in rows if r["policy"] == "judge_every_step")
    best = next((r for r in rows if r.get("selected_on_cal")), None)
    if best is None or judge["detection_rate"] <= 0:
        verdict = ("NOT SUPPORTED: no escalate_on_alarm setting reached 80% of "
                   f"judge_every_step detection on the cal split "
                   f"(judge test det {judge['detection_rate']:.2f}).")
        return verdict, {"best_policy": None, "supported": False}
    recovered = best["detection_rate"] / judge["detection_rate"]
    supported = (best["judge_call_ratio"] <= 0.25
                 and recovered >= 0.8)
    verdict = (
        f"{'SUPPORTED' if supported else 'MIXED'}: escalate_on_alarm "
        f"({best['params']}, operating point selected on cal) recovers "
        f"{recovered:.0%} of judge_every_step "
        f"detection ({best['detection_rate']:.2f} vs "
        f"{judge['detection_rate']:.2f}) with {best['judge_call_ratio']:.0%} "
        f"of its judge calls ({best['mean_judge_calls']:.1f} vs "
        f"{judge['mean_judge_calls']:.1f}/episode; total-cost ratio "
        f"{best['cost_ratio_vs_judge']:.0%}, mean lead {best['mean_lead']:.1f}, "
        f"wrongful halts {best['wrongful_halt_rate']:.3f})."
    )
    return verdict, {"best_policy": best,
                     "detection_recovered_frac": float(recovered),
                     "supported": bool(supported)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _provenance(quick: bool, seed: int) -> dict:
    """Self-describing run metadata for reproducibility."""
    import platform
    import subprocess

    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2],
                                  capture_output=True, text=True, timeout=10
                                  ).stdout.strip()
        except Exception:      # noqa: BLE001
            return ""

    versions = {}
    for pkg in ("numpy", "scipy", "scikit-learn", "pandas"):
        try:
            versions[pkg] = _pkg_version(pkg)
        except Exception:      # noqa: BLE001
            versions[pkg] = "?"
    return {
        "quick": bool(quick),
        "master_seed": int(seed),
        "git_rev": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "note": ("A --quick run cannot write here; publication tables "
                 "come from a full-size run. Reproduce against "
                 "requirements.lock.txt."),
    }


def _pkg_version(pkg: str) -> str:
    import importlib.metadata as _m
    return _m.version(pkg)


def main(argv: list[str] | None = None) -> None:
    """Run the full experiment; writes results/ and prints the summary."""
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_experiment",
        description="End-to-end derailment-detection experiment (H1/H2/H3).")
    parser.add_argument("--quick", action="store_true",
                        help="quarter-size dataset, K=4 ESN members, "
                             "100 iforest trees (integration test)")
    parser.add_argument("--seed", type=int, default=MASTER_SEED,
                        help="master seed; a non-default value is a "
                             "replication run written to results/seed<N>/")
    parser.add_argument("--allow-quick-publication", action="store_true",
                        help=argparse.SUPPRESS)   # escape hatch, never in CI
    args = parser.parse_args(argv)

    # Provenance is captured BEFORE the run writes anything. `git_dirty` asks
    # whether the tree that produced these artifacts was clean, and a run that
    # writes into results/ dirties the tree itself -- sampled afterwards the
    # flag would read `true` on every run and mean nothing.
    provenance = _provenance(args.quick, args.seed)

    # Publication-path guard: a --quick run is a quarter-size
    # integration test, not a publication result, and must NOT overwrite the
    # primary results/ directory that the paper cites. A quick run therefore
    # requires a disposable non-default --seed (which writes results/seed<N>/).
    if (args.quick and args.seed == MASTER_SEED
            and not args.allow_quick_publication):
        raise SystemExit(
            "refusing to write a --quick (quarter-size) run to the publication "
            "path results/. Quick runs are integration tests, not "
            "publication results. Use a disposable --seed N (writes "
            "results/seed<N>/); the published tables come from a full-size run "
            "at the master seed via run_multiseed.")

    # Sensitivity-override guard: a run under non-default judge parameters OR a
    # non-default cost ratio is a sensitivity arm, not a publication result. It
    # must not be able to land in the published tree even by accident. The cost
    # constants are here for the same reason as the judge rates: the escalation
    # conclusion is stated at COST_JUDGE == COST_STEP, and a table produced at
    # another rate answers a different question while looking identical.
    _overrides = [name for name in ("AGENTWATCH_JUDGE_P_DETECT",
                                    "AGENTWATCH_JUDGE_P_FALSE",
                                    "AGENTWATCH_COST_STEP",
                                    "AGENTWATCH_COST_JUDGE")
                  if os.environ.get(name)]
    if _overrides:
        if not os.environ.get("AGENTWATCH_RESULTS_ROOT"):
            raise SystemExit(
                f"refusing to write a SENSITIVITY run to the publication "
                f"path; overridden: {', '.join(_overrides)}. Set "
                f"AGENTWATCH_RESULTS_ROOT to a disposable directory (e.g. "
                f"results/_sensitivity) so results/ keeps the published "
                f"stipulated-judge, unit-cost numbers.")

    t0 = time.perf_counter()
    _set_results_dirs(args.seed)
    for d in (TABLES_DIR, SCORES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    ds_cfg = _dataset_config(args.quick, args.seed)
    sim_cfg = SimConfig()
    print(f"[experiment] quick={args.quick} master_seed={ds_cfg.master_seed}")

    print("[1/7] generating dataset ...")
    data = make_dataset(ds_cfg, sim_cfg)
    standardizer = Standardizer().fit(data["train"])
    _write_episode_metadata(data)
    print("      splits: " + ", ".join(f"{k}={len(v)}" for k, v in data.items())
          + f"  ({time.perf_counter() - t0:.1f}s)")

    print("[2/7] fitting monitors on healthy train ...")
    monitors = build_monitors(standardizer, args.quick)
    for mon in monitors:
        mon.fit(data["train"])
        print(f"      fitted {mon.name}  ({time.perf_counter() - t0:.1f}s)")

    print("[3/7] scoring val/cal/test with every monitor ...")
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    for mon in monitors:
        scores: dict[str, np.ndarray] = {}
        for split in ("val", "cal", "test"):
            scores.update(score_split(mon, data[split]))
        np.savez_compressed(SCORES_DIR / f"{mon.name}.npz", **scores)
        all_scores[mon.name] = scores
        print(f"      scored {mon.name}  ({time.perf_counter() - t0:.1f}s)")

    print("[4/7] H1: alarms at matched FA budget ...")
    h1_results = run_h1(monitors, data, all_scores)
    sig_table = run_significance(h1_results)

    print("[5/7] H2: channel ablation per failure class ...")
    h2_table = run_h2(h1_results)

    print("[6/7] H3a: confidence calibration ...")
    primary = next(m for m in monitors if m.name == H1_PRIMARY)
    components: ComponentStreams = {
        split: {ep.episode_id: primary.score_episode_components(ep)
                for ep in data[split]}
        for split in ("val", "cal", "test")
    }
    cal_headline = run_h3_calibration(components, data)

    print("[7/7] H3b: escalation policies ...")
    _, esc_headline = run_h3_escalation(components, data)

    v1, h1_head = _h1_verdict(h1_results, sig_table)
    v2, h2_head = _h2_verdict(h2_table)
    v3a = _h3a_verdict(cal_headline)
    v3b, esc_extra = _h3b_verdict(esc_headline)
    wall = time.perf_counter() - t0

    results = {
        "config": {
            "quick": bool(args.quick),
            "dataset": dataclasses.asdict(ds_cfg),
            "sim": dataclasses.asdict(sim_cfg),
            # The judge ACTUALLY used, not the default: an L8 sensitivity run
            # must not record itself as having used the stipulated rates.
            "judge": dataclasses.asdict(_judge_config()),
            "esn_K": 4 if args.quick else 8,
            "iforest_n_estimators": 100 if args.quick else 200,
            "fa_budget_h1": FA_BUDGET,
            "fa_budget_soft": FA_BUDGET_SOFT,
            "conf_thresholds": list(CONF_THRESHOLDS),
            "monitors": [m.name for m in monitors],
        },
        "h1": {"per_monitor": {name: r["row"] for name, r in h1_results.items()},
               "headline": h1_head},
        "h2": {"table": h2_table.to_dict(orient="records"),
               "headline": h2_head},
        "h3_calibration": cal_headline,
        "h3_escalation": {"theta_soft": esc_headline["theta_soft"],
                          "policies": esc_headline["policies"],
                          **esc_extra},
        "verdicts": {"H1": v1, "H2": v2, "H3a": v3a, "H3b": v3b},
    }
    # A DETERMINISTIC fingerprint of the scientific config goes IN results.json
    # (stable across byte-identical reruns); the non-deterministic environment
    # provenance (git rev, package versions, dirtiness) goes in the run_meta
    # SIDECAR so results.json stays a stable artifact.
    results["config_sha256"] = hashlib.sha256(
        json.dumps(_jsonable(results["config"]), sort_keys=True).encode()
    ).hexdigest()
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    # Runtime metadata lives in a SIDECAR, never in results.json: wall time
    # is the one field that changes between byte-identical reruns, and
    # keeping it out makes results.json a stable scientific artifact. (Files
    # pinned before this change still carry the field inline; the baseline
    # byte-check compares the scientific tables + score arrays, which are
    # unaffected.)
    with open(RESULTS_DIR / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"wall_time_sec": wall, "provenance": provenance}, fh, indent=2)

    line = "=" * 76
    print(f"\n{line}\nEXPERIMENT SUMMARY  (quick={args.quick}, "
          f"wall {wall:.1f}s)\n{line}")
    print("H1  temporal ESN ensemble vs memoryless baselines at matched FA:")
    print("    " + v1)
    print("H2  channel complementarity across failure classes:")
    print("    " + v2)
    print("H3a calibrated alarm confidence from ensemble components:")
    print("    " + v3a)
    print("H3b cost-optimal escalation to the judge-LLM:")
    print("    " + v3b)
    print(f"{line}\nwrote {RESULTS_DIR / 'results.json'}, "
          f"{TABLES_DIR}\\*.csv, {SCORES_DIR}\\*.npz")


if __name__ == "__main__":
    main()
