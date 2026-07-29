"""Sentence-transformers ablation study (MiniLM embedding upgrade).

Three controlled runs over the SAME traces and SAME train/val/test split:

  Baseline  — hash-embed, 43 dims (reproduced from the existing pipeline)
  Run A     — MiniLM embeddings, 43 dims (same channels, better embeddings)
  Run B     — MiniLM + tool-result similarity + JSON validity
              + result consistency (54-dim extended)

Every run reports six metrics per monitor:
  1. Overall AUC (episode-level ROC-AUC)
  2. Per-class detection rate at 5 % FA
  3. False-alarm rate (realized on healthy test)
  4. Detection lead (mean_lead_all, survivorship-free)
  5. Runtime (fit + score wall-clock seconds)
  6. Memory (tracemalloc peak MB during fit + score)

Run:  py -m derail.experiments.run_st_ablation
"""


from __future__ import annotations

raise SystemExit(  # archived, stale, kept for provenance
    "This archived experimental CLI is STALE: it references code that has since been removed or renamed and does not run against the current tree. It is kept for provenance of the results it produced (see the report alongside it) and is intentionally not repaired - it fails here rather than silently producing numbers from a pipeline that no longer exists.")

import json
import time
import tracemalloc
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
from derail.telemetry.pmi import AdjacentPMI

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
RESULTS = Path(__file__).resolve().parents[2] / "results"
FA_BUDGET = 0.05
MIN_T = 4


# ── Channel-max monitor (reused from run_real_traces.py) ─────────────────
class ChannelMax(OnlineMonitor):
    """Per-channel ESN-CUSUM fused by max."""

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


# ── Load episodes ────────────────────────────────────────────────────────
def _load_episodes(traces_dir: Path, manifest: list[dict],
                   use_st: bool | None, extended: bool,
                   pmi_model: AdjacentPMI | None = None
                   ) -> list[Episode]:
    episodes: list[Episode] = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        tau = entry["tau"]
        ep = load_trace_jsonl(traces_dir / entry["file"],
                              episode_id=entry["episode_id"],
                              tau=tau, failure_class=entry["failure_class"],
                              severity=None if tau is None else 0.5,
                              use_sentence_transformers=use_st,
                              extended=extended, pmi_model=pmi_model)
        episodes.append(ep)
    return episodes

def _get_train_files(manifest: list[dict]) -> list[str]:
    healthy = [e for e in manifest if e["T"] >= MIN_T and not e["tau"]]
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    train_entries = [healthy[i] for i in perm[:n_train]]
    return [e["file"] for e in train_entries]


def _make_split(episodes: list[Episode]):
    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    n_val = int(round(0.2 * len(healthy)))
    train = [healthy[i] for i in perm[:n_train]]
    val = [healthy[i] for i in perm[n_train:n_train + n_val]]
    test_h = [healthy[i] for i in perm[n_train + n_val:]]
    test = test_h + injected
    return train, val, test, healthy, injected


def _build_monitors(std: Standardizer, channels: tuple[str, ...],
                    extended: bool) -> list[OnlineMonitor]:
    if extended:
        return [ChannelMax(std, channels),
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
        return monitors


# ── Evaluate one run ─────────────────────────────────────────────────────
def _evaluate_run(run_name: str, episodes: list[Episode],
                  channels: tuple[str, ...], extended: bool
                  ) -> list[dict]:
    train, val, test, healthy, injected = _make_split(episodes)
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"  healthy: {len(train)} train / {len(val)} val / "
          f"{len(test) - len(injected)} test; injected: {len(injected)}")
    print(f"  dims: {episodes[0].X.shape[1]}  channels: {channels}")
    print(f"{'='*60}")

    std = Standardizer().fit(train)
    monitors = _build_monitors(std, channels, extended)

    rows = []
    for mon in monitors:
        # ── Measure runtime + memory for fit ──
        tracemalloc.start()
        t0 = time.perf_counter()
        mon.fit(train)
        fit_sec = time.perf_counter() - t0

        # ── Measure runtime + memory for scoring ──
        t1 = time.perf_counter()
        val_scores = [mon.score_episode(ep) for ep in val]
        theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
        scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
        score_sec = time.perf_counter() - t1

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 * 1024)

        summ = summarize(evaluate_alarms(test, scores, theta))
        auc = float(episode_auc(test, scores))

        row = {
            "run": run_name,
            "monitor": mon.name,
            "overall_auc": auc,
            "det_rate": summ["detection_rate"],
            "fa_rate": summ["healthy_fa_rate"],
            "mean_lead_all": summ["mean_lead_all"],
            "median_delay": summ["median_delay"],
            **{f"det[{fc}]": v["detection_rate"]
               for fc, v in summ["per_class"].items()},
            "fit_seconds": round(fit_sec, 3),
            "score_seconds": round(score_sec, 3),
            "peak_memory_mb": round(peak_mb, 2),
        }
        rows.append(row)
        print(f"  {row['monitor']:>30s}: AUC={auc:.3f}  det={row['det_rate']:.2f}  "
              f"fa={row['fa_rate']:.2f}  lead={row['mean_lead_all']:.1f}  "
              f"delay={row['median_delay']}  "
              f"fit={fit_sec:.2f}s  mem={peak_mb:.1f}MB")

    return rows


# ── Main ─────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_st_ablation")
    parser.add_argument("--dir", default=str(TRACES_DIR),
                        help="trace directory with manifest.json")
    args = parser.parse_args(argv)
    traces_dir = Path(args.dir)

    manifest_path = traces_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Collect traces first.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    has_lp = sum(bool(e.get("has_logprobs")) for e in manifest)
    channels = ("e", "u", "m") if has_lp >= 0.9 * len(manifest) else ("e", "m")
    print(f"[ablation] {len(manifest)} traces, logprobs in {has_lp}/{len(manifest)}"
          f" -> channels {channels}")

    # ── Pre-train PMI on training traces ─────────────────────────────────
    print("\n[ablation] Pre-training AdjacentPMI model on train split...")
    train_files = _get_train_files(manifest)
    pmi = AdjacentPMI()
    train_texts = []
    for f in train_files:
        for line in open(traces_dir / f, "r", encoding="utf-8"):
            text = json.loads(line).get("text", "")
            if text.strip():
                train_texts.append(text)
    pmi.fit(train_texts)

    # ── Baseline: hash-embed, 43 dims ────────────────────────────────────
    print("\n[ablation] Loading episodes: Baseline (hash-embed, 43d)...")
    eps_baseline = _load_episodes(traces_dir, manifest,
                                  use_st=False, extended=False)
    rows_baseline = _evaluate_run("Baseline (hash-embed, 43d)",
                                  eps_baseline, channels, extended=False)

    # ── Run A: MiniLM, 43 dims ──────────────────────────────────────────
    print("\n[ablation] Loading episodes: Run A (MiniLM, 43d)...")
    eps_a = _load_episodes(traces_dir, manifest,
                           use_st=True, extended=False)
    rows_a = _evaluate_run("Run A: MiniLM (43d)", eps_a, channels,
                           extended=False)

    # ── Run B: Hash + extended (55d) ────────────────────────────────────
    print("\n[ablation] Loading episodes: Run B (Hash + extended + PMI, 55d)...")
    eps_b = _load_episodes(traces_dir, manifest,
                           use_st=False, extended=True, pmi_model=pmi)
    rows_b = _evaluate_run(
        "Run B: Hash + result_task_sim + json_validity + result_self_sim + PMI (55d)",
        eps_b, channels, extended=True)

    # ── Write comparison table ───────────────────────────────────────────
    all_rows = rows_baseline + rows_a + rows_b
    table = pd.DataFrame(all_rows)
    out = RESULTS / "tables" / "st_ablation_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n[ablation] wrote {out}")

    # ── Print summary comparison (primary monitors only) ─────────────────
    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY (primary channel-max monitors)")
    print("=" * 72)
    primary = table[table["monitor"].str.startswith("esn_cusum_max")]
    cols = ["run", "monitor", "overall_auc", "det_rate", "fa_rate",
            "mean_lead_all", "median_delay", "fit_seconds", "peak_memory_mb"]
    avail = [c for c in cols if c in primary.columns]
    print(primary[avail].to_string(index=False))


if __name__ == "__main__":
    main()
