"""Experiment: Telemetry Robustness Study — ALL monitors on real agent traces.

Loads real traces from traces/, introduces corruptions at evaluation time,
and measures ROC-AUC degradation for EVERY monitor (ESN, GRU, LSTM, Linear AR,
Delta Mahalanobis, Mahalanobis, Isolation Forest, Self Drift).

Run: py -m derail.experiments.run_robustness_study
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from derail.common import Episode, Standardizer, rng_for
from derail.telemetry.adapter import load_trace_jsonl
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.monitor.baselines import (
    DeltaMahalanobisMonitor,
    IsolationForestMonitor,
    MahalanobisMonitor,
    SelfDriftMonitor,
)
from derail.monitor.seq_baselines import LinearARMonitor, GRUMonitor, LSTMMonitor
from derail.evaluation.metrics import episode_auc

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
MIN_T = 4


def load_dataset() -> tuple[list[Episode], list[Episode]]:
    manifest_path = TRACES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes: list[Episode] = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        ep = load_trace_jsonl(
            TRACES_DIR / entry["file"],
            episode_id=entry["episode_id"],
            tau=entry["tau"],
            failure_class=entry["failure_class"],
            severity=None if entry["tau"] is None else 0.5,
            use_sentence_transformers=False,
            extended=False
        )
        episodes.append(ep)
    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    return healthy, injected


def apply_packet_drop(ep: Episode, rate: float, seed: int) -> Episode:
    if rate == 0.0 or ep.T <= 2:
        return ep
    rng = rng_for(seed, "drop", ep.episode_id)
    keep_indices = [0]
    for i in range(1, ep.T - 1):
        if rng.random() > rate:
            keep_indices.append(i)
    keep_indices.append(ep.T - 1)
    X_new = ep.X[keep_indices]
    tau_new = ep.tau
    if ep.tau is not None:
        new_tau = sum(1 for idx in keep_indices if idx < ep.tau)
        tau_new = min(new_tau, X_new.shape[0] - 1)
    return Episode(
        X=X_new, episode_id=ep.episode_id, is_healthy=ep.is_healthy,
        failure_class=ep.failure_class, tau=tau_new,
        t_fail=X_new.shape[0] - 1 if ep.tau is not None else None,
        severity=ep.severity
    )


# Continuous feature dims that Gaussian noise is meaningful for: the semantic
# embedding e[0:32], the two surprisal LEVEL dims u[32:34], and the two log
# metadata dims m[40:42]. The action one-hot (36:40), the error flag (42), the
# bounded [0,1] fractions (u[34:36], x success/ctx dims) and integer counts are
# NOT perturbed with continuous Gaussian noise - doing so violated their
# feature semantics. The semantic block is renormalised to unit norm
# afterwards, respecting its invariant.
_CONTINUOUS_DIMS = list(range(0, 34)) + [40, 41]


def apply_gaussian_noise(ep: Episode, std: float, seed: int) -> Episode:
    if std == 0.0:
        return ep
    rng = rng_for(seed, "noise", ep.episode_id)
    X = ep.X.copy()
    dims = [d for d in _CONTINUOUS_DIMS if d < X.shape[1]]
    X[:, dims] = X[:, dims] + rng.normal(0.0, std, size=(X.shape[0], len(dims)))
    # Keep the surprisal LEVEL dims >= 0 and the semantic block unit-norm - the
    # invariants the adapter guarantees.
    X[:, 32:34] = np.maximum(X[:, 32:34], 0.0)
    norms = np.linalg.norm(X[:, :32], axis=1, keepdims=True)
    X[:, :32] = np.where(norms > 0, X[:, :32] / norms, X[:, :32])
    return Episode(
        X=X, episode_id=ep.episode_id, is_healthy=ep.is_healthy,
        failure_class=ep.failure_class, tau=ep.tau, t_fail=ep.t_fail,
        severity=ep.severity
    )


def apply_missing_metadata(ep: Episode) -> Episode:
    X_new = ep.X.copy()
    X_new[:, 40:43] = 0.0
    return Episode(
        X=X_new, episode_id=ep.episode_id, is_healthy=ep.is_healthy,
        failure_class=ep.failure_class, tau=ep.tau, t_fail=ep.t_fail,
        severity=ep.severity
    )


def score_all(monitor, episodes: list[Episode]) -> dict:
    scores = {}
    for ep in episodes:
        monitor.start_episode()
        scores[ep.episode_id] = np.array([monitor.score_step(x) for x in ep.X])
    return scores


def run_robustness_study():
    healthy, injected = load_dataset()

    perm = rng_for(42, "robust-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    train = [healthy[i] for i in perm[:n_train]]
    test_h = [healthy[i] for i in perm[n_train:]]
    test_base = test_h + injected

    std = Standardizer().fit(train)

    # All monitors to evaluate
    monitors = [
        ChannelMaxESNMonitor(std, K=15, reservoir_size=100, seed=0, channels=("e", "u", "m")),
        LinearARMonitor(std, seed=0),
        GRUMonitor(std, seed=0),
        LSTMMonitor(std, seed=0),
        DeltaMahalanobisMonitor(std),
        MahalanobisMonitor(std),
        IsolationForestMonitor(std, seed=0),
        SelfDriftMonitor(),
    ]

    # Pre-fit all monitors on clean training data
    print(f"Fitting {len(monitors)} monitors on {len(train)} clean training episodes...")
    for mon in monitors:
        mon.fit(train)

    # Stochastic conditions are evaluated over REPEATED realisations:
    # each is applied under several seeds and the AUC reported as mean +/- std,
    # instead of a single realisation with no interval. Deterministic
    # conditions (clean, missing metadata) use one realisation.
    SEEDS = (101, 202, 303, 404, 505)
    conditions = [
        ("Clean Baseline", "det", lambda s, ep: ep, 0.0),
        ("10% Packet Drop", "stoch", apply_packet_drop, 0.10),
        ("20% Packet Drop", "stoch", apply_packet_drop, 0.20),
        ("40% Packet Drop", "stoch", apply_packet_drop, 0.40),
        ("Noise std=0.05", "stoch", apply_gaussian_noise, 0.05),
        ("Noise std=0.10", "stoch", apply_gaussian_noise, 0.10),
        ("Noise std=0.20", "stoch", apply_gaussian_noise, 0.20),
        ("Missing Metadata", "det", lambda s, ep: apply_missing_metadata(ep), 0.0),
    ]

    results = []
    header = f"{'Condition':20s} | " + " | ".join(f"{m.name:18s}" for m in monitors)
    print("\n" + header)
    print("-" * len(header))

    for cond_name, kind, fn, param in conditions:
        seeds = SEEDS if kind == "stoch" else (0,)
        # {monitor: [auc per realisation]}
        per_mon: dict[str, list[float]] = {m.name: [] for m in monitors}
        for seed in seeds:
            episodes = [fn(seed, ep) if kind == "det"
                        else fn(ep, param, seed) for ep in test_base]
            for mon in monitors:
                per_mon[mon.name].append(
                    episode_auc(episodes, score_all(mon, episodes)))
        row = {"Condition": cond_name}
        cells = []
        for mon in monitors:
            vals = np.array(per_mon[mon.name], dtype=float)
            row[mon.name] = round(float(vals.mean()), 3)
            row[f"{mon.name}_std"] = round(float(vals.std()), 3)
            cells.append(f"{vals.mean():.3f}"
                         + (f"±{vals.std():.3f}" if len(vals) > 1 else ""))
        results.append(row)
        print(f"{cond_name:20s} | " + " | ".join(f"{v:18s}" for v in cells))

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "real_robustness_all_monitors.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved full robustness table to {out}")


if __name__ == "__main__":
    run_robustness_study()
