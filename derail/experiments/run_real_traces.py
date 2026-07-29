"""Evaluate the monitors on REAL collected traces (weaknesses A/B, step 2).

Reads traces/ (written by collect_traces.py), converts them to Episodes via
the adapter, fits monitors one-class on a train split of the healthy traces,
thresholds on a val split at the 5% FA budget, and evaluates on the held-out
healthy traces + all injected traces.

Channel selection is data-driven: Gemini traces collected with
response_logprobs carry a real uncertainty channel, so the primary monitor
is the full e+u+m channel-max; traces without logprobs fall back to e+m
(the manifest records which). Small-sample caveats are printed with the
results.

Run:  py -m derail.experiments.run_real_traces
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import Episode, OnlineMonitor, Standardizer, rng_for
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.monitor.baselines import (
    DeltaMahalanobisMonitor,
    IsolationForestMonitor,
    MahalanobisMonitor,
    SelfDriftMonitor,
)
from derail.monitor.esn import ESNEnsembleMonitor
from derail.monitor.seq_baselines import _HAS_TORCH, LinearARMonitor

if _HAS_TORCH:
    from derail.monitor.seq_baselines import GRUMonitor, LSTMMonitor
from derail.telemetry.adapter import load_trace_jsonl

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
RESULTS = Path(__file__).resolve().parents[2] / "results"
FA_BUDGET = 0.05
MIN_T = 4   # ESN washout is 3 steps; below this nothing can ever be scored


class ChannelMax(OnlineMonitor):
    """Per-channel ESN-CUSUM fused by max, over the channels that carry data.

    Real Gemini traces include the u channel (response_logprobs); traces
    collected without logprobs get e+m only — the manifest decides.
    """

    def __init__(self, standardizer: Standardizer,
                 channels: tuple[str, ...], K: int = 8) -> None:
        self.name = f"esn_cusum_max[{','.join(channels)}]"
        self.subs = [
            ESNEnsembleMonitor(standardizer, channels=(c,), K=K, cusum=True,
                               seed=1200 + i)
            for i, c in enumerate(channels)
        ]

    def fit(self, healthy_episodes: list[Episode]) -> None:
        for sub in self.subs:
            sub.fit(healthy_episodes)

    def start_episode(self) -> None:
        for sub in self.subs:
            sub.start_episode()

    def score_step(self, x_t: np.ndarray) -> float:
        return max(sub.score_step(x_t) for sub in self.subs)

    def score_episode(self, ep: Episode) -> np.ndarray:
        return np.max([sub.score_episode(ep) for sub in self.subs], axis=0)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_real_traces")
    parser.add_argument("--dir", default=str(TRACES_DIR),
                        help="trace directory with manifest.json (e.g. "
                             "traces, traces/ollama, traces/langgraph, "
                             "traces/autogen)")
    parser.add_argument("--extended", action="store_true",
                        help="telemetry v3: load episodes with the derived "
                             "x channel and compare the primary channel-max "
                             "with and without it (writes real_traces_ext_* "
                             "— default tables are untouched)")
    args = parser.parse_args(argv)
    traces_dir = Path(args.dir)
    label = traces_dir.name if traces_dir != TRACES_DIR else "gemini"

    manifest_path = traces_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Collect first:\n"
              "  py -m derail.experiments.collect_traces --mock-llm   (dry run)\n"
              "  py -m derail.experiments.collect_traces --yes        (gemini)\n"
              "  py -m derail.experiments.collect_traces --backend ollama\n"
              "  py -m derail.experiments.collect_framework_traces "
              "--framework langgraph|autogen")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episodes: list[Episode] = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        tau = entry["tau"]
        ep = load_trace_jsonl(traces_dir / entry["file"],
                              episode_id=entry["episode_id"],
                              tau=tau, failure_class=entry["failure_class"],
                              severity=None if tau is None else 0.5,
                              use_sentence_transformers=False,
                              extended=args.extended)
        episodes.append(ep)
    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    if len(healthy) < 4:
        print(f"Only {len(healthy)} usable healthy traces — need >= 4 "
              "(fit + val + test). Collect more.")
        return

    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    n_val = int(round(0.2 * len(healthy)))
    train = [healthy[i] for i in perm[:n_train]]
    val = [healthy[i] for i in perm[n_train:n_train + n_val]]
    test_h = [healthy[i] for i in perm[n_train + n_val:]]
    test = test_h + injected
    print(f"[real] healthy: {n_train} train / {len(val)} val / "
          f"{len(test_h)} test; injected: {len(injected)} "
          f"({sorted(set(ep.failure_class for ep in injected))})")

    has_lp = sum(bool(e.get("has_logprobs")) for e in manifest)
    channels = ("e", "u", "m") if has_lp >= 0.9 * len(manifest) else ("e", "m")
    print(f"[real] logprobs present in {has_lp}/{len(manifest)} traces -> "
          f"monitor channels {channels}")

    std = Standardizer().fit(train)
    if args.extended:
        # Focused v3 comparison: the primary with vs without the derived x
        # channel (same split, same seeds), the x channel alone, and one
        # non-ESN baseline over the widened vector.
        monitors = [ChannelMax(std, channels),
                    ChannelMax(std, channels + ("x",)),
                    ESNEnsembleMonitor(std, channels=("x",), K=8, cusum=True,
                                       seed=1203, name="esn_cusum[x]"),
                    DeltaMahalanobisMonitor(std)]
    else:
        monitors = [ChannelMax(std, channels),
                    ESNEnsembleMonitor(std, channels=("e",), K=8, cusum=True,
                                       seed=1210, name="esn_cusum[e]"),
                    SelfDriftMonitor(),
                    LinearARMonitor(std, seed=1211),
                    *([GRUMonitor(std, seed=1212), LSTMMonitor(std, seed=1213)]
                      if _HAS_TORCH else []),
                    DeltaMahalanobisMonitor(std),
                    MahalanobisMonitor(std),
                    IsolationForestMonitor(std, seed=1214)]
    # TCN is omitted on real traces: its receptive field (16) exceeds the
    # typical real episode length (~5 steps).
    rows = []
    for mon in monitors:
        mon.fit(train)
        val_scores = [mon.score_episode(ep) for ep in val]
        theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
        scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
        summ = summarize(evaluate_alarms(test, scores, theta))
        rows.append({
            "monitor": mon.name,
            "detection_rate": summ["detection_rate"],
            "healthy_fa_rate": summ["healthy_fa_rate"],
            "mean_lead_all": summ["mean_lead_all"],
            "median_delay": summ["median_delay"],
            "episode_auc": float(episode_auc(test, scores)),
            **{f"det[{fc}]": v["detection_rate"]
               for fc, v in summ["per_class"].items()},
        })
        r = rows[-1]
        print(f"  {r['monitor']:>18s}: det={r['detection_rate']:.2f} "
              f"fa={r['healthy_fa_rate']:.2f} lead_all={r['mean_lead_all']:.1f} "
              f"auc={r['episode_auc']:.3f}")

    table = pd.DataFrame(rows)
    stem = "real_traces_ext" if args.extended else "real_traces"
    out = (RESULTS / "tables" / f"{stem}.csv" if label == "gemini"
           else RESULTS / "tables" / f"{stem}_{label}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"[real] wrote {out}")
    print("[real] caveats: small-sample (interpret with CIs in mind); "
          f"thresholds from only {len(val)} val episodes. Mock-LLM traces "
          "validate the pipeline only — their dynamics are not real model "
          "behavior.")


if __name__ == "__main__":
    main()
