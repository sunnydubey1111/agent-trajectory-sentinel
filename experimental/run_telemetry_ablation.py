"""Semantic-telemetry ablation — does each feature fix the 0.00 content classes?

Architecture is FIXED (ESN + ChannelMax, channels e,u,m,x); only the telemetry
substrate changes. Every added feature gets its own row, isolated:

  Run0  hash embeddings, result features off      (the measured baseline)
  A     MiniLM embeddings, result features off    (embeddings alone)
  B1    A + result<->task similarity              (off-topic retrieval)
  B2    B1 + JSON-validity flag                    (malformed structured output)
  B3    B2 + result self-consistency               (scrambled / inconsistent)

Identical 60/20/20 healthy split and 5% FA budget as run_real_traces, same
seeds, so rows are comparable. For EVERY run we report (not only AUC):
  overall AUC · per-class detection · false-alarm rate · mean detection lead
  · telemetry-extraction runtime · monitor fit+score runtime · memory.

Writes results/tables/telemetry_ablation_{label}.csv. Additive; no existing
table or module is modified.
"""


from __future__ import annotations

raise SystemExit(  # archived, stale, kept for provenance
    "This archived experimental CLI is STALE: it references code that has since been removed or renamed and does not run against the current tree. It is kept for provenance of the results it produced (see the report alongside it) and is intentionally not repaired - it fails here rather than silently producing numbers from a pipeline that no longer exists.")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import (
    D_TOTAL_EXT,
    IDX_RESULT_JSON_BROKEN,
    IDX_RESULT_SELF_SIM,
    IDX_RESULT_TASK_SIM,
    Episode,
    Standardizer,
    rng_for,
)
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.telemetry.adapter import load_trace_jsonl

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
RESULTS = Path(__file__).resolve().parents[2] / "results" / "tables"
FA_BUDGET = 0.05

# Result-feature index -> neutral value (used to switch a feature OFF).
_RESULT_NEUTRAL = {IDX_RESULT_TASK_SIM: 1.0, IDX_RESULT_JSON_BROKEN: 0.0,
                   IDX_RESULT_SELF_SIM: 1.0}


def _load(traces_dir: Path, use_st: bool):
    """Load all extended episodes with the chosen embedding backend; also
    return the wall-clock telemetry-extraction time."""
    manifest = json.loads((traces_dir / "manifest.json").read_text("utf-8"))
    t0 = time.perf_counter()
    eps: list[Episode] = []
    for e in manifest:
        if e["T"] < 4:
            continue
        eps.append(load_trace_jsonl(
            traces_dir / e["file"], episode_id=e["episode_id"], tau=e["tau"],
            failure_class=e["failure_class"],
            severity=None if e["tau"] is None else 0.5,
            use_sentence_transformers=use_st, extended=True))
    return eps, time.perf_counter() - t0


def _neutralize(eps: list[Episode], keep: set[int]) -> list[Episode]:
    """Copy episodes, switching OFF every result feature not in `keep`."""
    out = []
    for ep in eps:
        X = ep.X.copy()
        for idx, neutral in _RESULT_NEUTRAL.items():
            if idx not in keep:
                X[:, idx] = neutral
        out.append(Episode(X=X, episode_id=ep.episode_id,
                           is_healthy=ep.is_healthy,
                           failure_class=ep.failure_class, tau=ep.tau,
                           t_fail=ep.t_fail, severity=ep.severity))
    return out


def _split(eps: list[Episode]):
    healthy = [ep for ep in eps if ep.is_healthy]
    injected = [ep for ep in eps if not ep.is_healthy]
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_tr, n_va = int(round(0.6 * len(healthy))), int(round(0.2 * len(healthy)))
    train = [healthy[i] for i in perm[:n_tr]]
    val = [healthy[i] for i in perm[n_tr:n_tr + n_va]]
    test = [healthy[i] for i in perm[n_tr + n_va:]] + injected
    return train, val, test


def _monitor_bytes(mon) -> int:
    total = 0
    for sub in mon.subs:
        for attr in ("_W", "_Win", "_Wout", "_sigma_err"):
            a = getattr(sub, attr, None)
            if isinstance(a, np.ndarray):
                total += a.nbytes
    return total


def _run(label: str, eps, keep: set[int], embed_s: float, use_st: bool) -> dict:
    from derail.experiments.run_real_traces import ChannelMax

    train, val, test = _split(_neutralize(eps, keep))
    std = Standardizer().fit(train)
    mon = ChannelMax(std, ("e", "u", "m", "x"))
    t0 = time.perf_counter()
    mon.fit(train)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    val_scores = [mon.score_episode(ep) for ep in val]
    scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
    n_steps = sum(ep.T for ep in val) + sum(ep.T for ep in test)
    score_us = (time.perf_counter() - t1) / max(n_steps, 1) * 1e6

    theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
    summ = summarize(evaluate_alarms(test, scores, theta))
    row = {"run": label, "embed": "MiniLM" if use_st else "hash",
           "auc": round(float(episode_auc(test, scores)), 3),
           "detection": round(summ["detection_rate"], 3),
           "false_alarm": round(summ["healthy_fa_rate"], 3),
           "lead": round(summ["mean_lead_all"], 2),
           "telemetry_s": round(embed_s, 1),
           "fit_s": round(fit_s, 2),
           "score_us_step": round(score_us, 1),
           "monitor_kb": round(_monitor_bytes(mon) / 1024, 1)}
    row.update({f"det[{fc}]": round(v["detection_rate"], 2)
                for fc, v in summ["per_class"].items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_telemetry_ablation")
    parser.add_argument("--dir", default=str(TRACES_DIR / "real_research7b"))
    args = parser.parse_args()
    traces_dir = Path(args.dir)
    label = traces_dir.name

    import truststore
    truststore.inject_into_ssl()   # MiniLM first-download TLS on this machine

    print("[tel-abl] loading (hash) ...")
    eps_hash, t_hash = _load(traces_dir, use_st=False)
    print(f"[tel-abl] loading (MiniLM) ... {len(eps_hash)} episodes")
    eps_st, t_st = _load(traces_dir, use_st=True)
    assert eps_hash[0].X.shape[1] == D_TOTAL_EXT

    T, T1, T12 = ({IDX_RESULT_TASK_SIM},
                  {IDX_RESULT_TASK_SIM, IDX_RESULT_JSON_BROKEN},
                  {IDX_RESULT_TASK_SIM, IDX_RESULT_JSON_BROKEN,
                   IDX_RESULT_SELF_SIM})
    runs = [
        _run("Run0 hash",           eps_hash, set(), t_hash, False),
        _run("A  MiniLM",           eps_st,   set(), t_st,   True),
        _run("B1 +result-task-sim", eps_st,   T,      t_st,   True),
        _run("B2 +json-validity",   eps_st,   T1,     t_st,   True),
        _run("B3 +result-consist",  eps_st,   T12,    t_st,   True),
    ]
    df = pd.DataFrame(runs)

    core = ["run", "embed", "auc", "detection", "false_alarm", "lead",
            "telemetry_s", "fit_s", "score_us_step", "monitor_kb"]
    print("\n=== TELEMETRY ABLATION (ESN + ChannelMax fixed; 5% FA) ===")
    print(df[core].to_string(index=False))
    det_cols = [c for c in df.columns if c.startswith("det[")]
    print("\nper-class detection:")
    print(df[["run"] + sorted(det_cols)].to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"telemetry_ablation_{label}.csv"
    df.to_csv(out, index=False)
    print(f"\n[tel-abl] wrote {out}")

    base = df.iloc[0]
    best = df.iloc[-1]
    print(f"\n[tel-abl] content classes  Run0 -> B3 (full):")
    for fc in ("wrong_document", "malformed_json", "context_corruption"):
        col = f"det[{fc}]"
        if col in df.columns:
            print(f"  {fc:20s} {base[col]:.2f} -> {best[col]:.2f}")


if __name__ == "__main__":
    main()
