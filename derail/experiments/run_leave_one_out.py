"""Experiment: PER-CLASS evaluation on real agent traces across ALL baselines.

Loads real traces and, for each failure class in turn, evaluates every monitor
on healthy + that class's injected episodes. Every monitor here is ONE-CLASS
(fit on healthy only), so no failure class ever participates in training -
there is nothing to "leave out". This is per-class evaluation, NOT
leave-one-failure-out (which would require a learner trained on the OTHER
failure classes and tested on the held-out one); the old name overstated it. The output table is named accordingly.

Run: py -m derail.experiments.run_leave_one_out
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
from derail.evaluation.metrics import episode_auc, pick_threshold, evaluate_alarms, summarize

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


def run_leave_one_out():
    healthy, injected = load_dataset()

    # Extract unique failure classes present in traces
    failure_classes = sorted(list(set(ep.failure_class for ep in injected if ep.failure_class is not None)))

    # Split healthy traces (60/20/20)
    perm = rng_for(42, "lofo-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    n_val = int(round(0.2 * len(healthy)))

    train = [healthy[i] for i in perm[:n_train]]
    val = [healthy[i] for i in perm[n_train:n_train + n_val]]
    test_h = [healthy[i] for i in perm[n_train + n_val:]]

    std = Standardizer().fit(train)

    print(f"Cohort: {len(train)} Train | {len(val)} Val | {len(test_h)} Test Healthy")
    print(f"Failure classes: {failure_classes}")

    results = []

    for held_out in failure_classes:
        target_injected = [ep for ep in injected if ep.failure_class == held_out]
        test = test_h + target_injected

        # Instantiate all monitors
        monitors = [
            ChannelMaxESNMonitor(std, K=15, reservoir_size=100, seed=0, channels=("e", "u", "m")),
            SelfDriftMonitor(),
            LinearARMonitor(std, seed=0),
            GRUMonitor(std, seed=0),
            LSTMMonitor(std, seed=0),
            DeltaMahalanobisMonitor(std),
            MahalanobisMonitor(std),
            IsolationForestMonitor(std, seed=0)
        ]

        print(f"\n--- Held-Out Failure Class: {held_out} ---")
        print(f"{'Monitor':20s} | {'AUC':5s} | {'FA':5s} | {'Delay':5s} | {'Prec':5s} | {'Rec':5s}")
        print("-" * 62)

        for monitor in monitors:
            monitor.fit(train)

            # Extract validation scores
            val_scores = []
            for ep in val:
                monitor.start_episode()
                val_scores.append(np.array([monitor.score_step(x) for x in ep.X]))

            theta = pick_threshold(val_scores, fa_budget=0.05)

            # Score test
            test_scores = {}
            for ep in test:
                monitor.start_episode()
                test_scores[ep.episode_id] = np.array([monitor.score_step(x) for x in ep.X])

            # AUC
            auc = episode_auc(test, test_scores)

            # Alarms
            alarm_df = evaluate_alarms(test, test_scores, theta)
            metrics = summarize(alarm_df)

            tp = int((alarm_df["outcome"] == "true_alarm").sum())
            fp = int((alarm_df["outcome"] == "false_alarm").sum() + (alarm_df["outcome"] == "early_alarm").sum())
            fn = int((alarm_df["outcome"] == "miss").sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            print(f"{monitor.name:20s} | {auc:.3f} | {metrics['healthy_fa_rate']:.3f} | {str(metrics['median_delay']):5s} | {precision:.3f} | {recall:.3f}")

            results.append({
                "failure_class": held_out,
                "Monitor": monitor.name,
                "AUC": auc,
                "FA_Rate": metrics["healthy_fa_rate"],
                "Delay": metrics["median_delay"],
                "Precision": precision,
                "Recall": recall
            })

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "real_per_class_baselines.csv", index=False)
    print(f"\nSaved per-class baseline table to {RESULTS_DIR / 'real_per_class_baselines.csv'}")


if __name__ == "__main__":
    run_leave_one_out()
