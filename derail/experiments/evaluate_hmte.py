"""Experiment: Evaluate HMTE-ESN-M vs ChannelMax on real Gemini agent traces.

QUARANTINED — its output is not evidence. See
``results/tables/hmte_vs_baseline.QUARANTINE.md``.

This script does NOT follow the kill-switch protocol every other monitor
comparison in this repository is held to. It runs on ``traces/real/``, the
18-episode Gemini stub corpus, which leaves a test set of 4 healthy + 1
injected episode — an episode AUC decided by four pairwise comparisons. It
computes ``val`` and never uses it, so no false-alarm-budget threshold is
picked and no detection or false-alarm rate is reported. One seed, no
confidence interval. Separately, ``HMTE_ESN_M_Monitor.fit`` estimates its
Mahalanobis mean/covariance over ALL healthy episodes, including the ones its
sub-monitors' readouts were fit on, so the healthy feature distribution is
partly in-sample.

Do not cite the AUC 1.000 figure. To lift the quarantine, re-run HMTE-ESN-M
inside ``derail/experiments/run_hmt_ab.py`` on ``traces/real_research7b``
with a val-picked threshold, a held-out Mahalanobis fit, bootstrap dAUC CIs
and multiple seeds.

Run: py -m derail.experiments.evaluate_hmte
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from derail.common import Episode, Standardizer, rng_for
from derail.telemetry.adapter import load_trace_jsonl
from derail.monitor.esn import ChannelMaxESNMonitor, HMTE_ESN_M_Monitor
from derail.evaluation.metrics import episode_auc

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces" / "real"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
MIN_T = 4


QUARANTINE_BANNER = """
!!! QUARANTINED RESULT — NOT EVIDENCE !!!
This script runs outside the kill-switch protocol: 18-episode stub corpus
(test set = 4 healthy + 1 injected), no val-picked threshold, one seed, no CI,
and a partly in-sample Mahalanobis fit. Its AUC is not comparable to any other
monitor number in this repository and must not be cited.
See results/tables/hmte_vs_baseline.QUARANTINE.md
"""


def run_hmte_evaluation():
    print(QUARANTINE_BANNER)
    manifest_path = TRACES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}. Please run collect_real_traces.py first.")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episodes: list[Episode] = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        ep = load_trace_jsonl(TRACES_DIR / entry["file"],
                              episode_id=entry["episode_id"],
                              tau=entry["tau"], failure_class=entry["failure_class"],
                              severity=None if entry["tau"] is None else 0.5,
                              use_sentence_transformers=False,
                              extended=False)
        episodes.append(ep)

    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]

    if len(healthy) < 4:
        print(f"Only {len(healthy)} usable healthy traces — need >= 4 to evaluate.")
        return

    # Split healthy into train/val/test
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    n_val = int(round(0.2 * len(healthy)))
    train = [healthy[i] for i in perm[:n_train]]
    val = [healthy[i] for i in perm[n_train:n_train + n_val]]
    test_h = [healthy[i] for i in perm[n_train + n_val:]]
    test = test_h + injected

    # Fit standardizer
    std = Standardizer().fit(train)

    # Instantiate monitors
    # Proposed baseline: ChannelMaxESNMonitor
    channel_max = ChannelMaxESNMonitor(std, K=15, reservoir_size=100, seed=0)
    # Proposed new architecture: HMTE-ESN-M
    hmte_esn_m = HMTE_ESN_M_Monitor(std, K=15, reservoir_size=100, seed=0)

    # Fit monitors
    channel_max.fit(train)
    hmte_esn_m.fit(train)

    # Evaluate test AUC
    def get_auc(monitor):
        scores_dict = {}
        for ep in test:
            monitor.start_episode()
            step_scores = [monitor.score_step(x) for x in ep.X]
            scores_dict[ep.episode_id] = np.array(step_scores)
        return episode_auc(test, scores_dict)

    max_auc = get_auc(channel_max)
    hmte_auc = get_auc(hmte_esn_m)

    n_h, n_inj = len(test_h), len(injected)
    print("\n=== HMTE-ESN-M vs ChannelMax ESN-CUSUM on Real Traces ===")
    print(f"ChannelMax (AUC) : {max_auc:.3f}")
    print(f"HMTE-ESN-M (AUC) : {hmte_auc:.3f}")
    # Never print an AUC without the denominator it was computed over: this
    # one rests on n_h * n_inj pairwise comparisons.
    print(f"computed over {n_h} healthy x {n_inj} injected = "
          f"{n_h * n_inj} pairwise comparisons — QUARANTINED, not evidence")
    print(QUARANTINE_BANNER)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # The denominators travel with the numbers, so the AUC can never be read
    # out of this file without the sample size it rests on.
    df = pd.DataFrame([{
        "Monitor": name, "Test_AUC": auc,
        "n_test_healthy": n_h, "n_test_injected": n_inj,
        "n_pairwise_comparisons": n_h * n_inj,
        "status": "QUARANTINED - not evidence, see hmte_vs_baseline.QUARANTINE.md",
    } for name, auc in (("ChannelMax", max_auc), ("HMTE-ESN-M", hmte_auc))])
    df.to_csv(RESULTS_DIR / "hmte_vs_baseline.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR / 'hmte_vs_baseline.csv'}")


if __name__ == "__main__":
    run_hmte_evaluation()
