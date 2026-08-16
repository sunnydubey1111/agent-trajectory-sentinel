"""What does a deployment LOSE without token logprobs?

The system already degrades gracefully: `load_real` inspects each corpus's
`has_logprobs` and drops the token-surprisal (`u`) channel when the provider
does not supply it, so a monitor runs on `(e, m, x)` instead of `(e, u, m, x)`
with no code change. What was missing is the *price* of that degradation.

The cross-provider comparison measured it - a Gemini tier that rejects
`response_logprobs` scores AUROC 0.794 against a paired qwen corpus's 0.790,
but detection at a matched FA budget of 0.38 against 0.57. That comparison
CANNOT establish the cause, because provider and channel availability move
together. This script removes the confound: it takes corpora that DO have
logprobs and scores each one twice, with and without `u`, everything else
identical - same episodes, same splits, same seeds, same thresholds.

If the within-corpus ablation reproduces the cross-provider gap, the honest
reading of L5 is "one telemetry channel missing", not "a worse model". If it
does not, the L5 gap is provider-specific and the paper must say so.

Deployment consequence either way: coverage is reported CONDITIONAL on the
telemetry a deployment can actually emit.

Run:  py -m experimental.telemetry_dependence      (free, no API, ~minutes)
Writes results/tables/telemetry_dependence.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from derail.common import Standardizer
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.experiments.run_hybrid_study import (
    REAL_DATASETS,
    TABLES_DIR,
    load_real,
)
from derail.monitor.hybrid import make_hybrids

FA_BUDGET = 0.05
#: Corpora that actually carry logprobs (the ablation is meaningless without).
CORPORA = ("real_research7b_long", "ollama7b", "ollama_llama8b",
           "real_research7b")


def _score(data, channels, seed: int = 0) -> list[dict]:
    """Fit on healthy train, threshold on healthy val, score test."""
    std = Standardizer().fit(data["train"])
    esn, maha, _ = make_hybrids(std, channels=channels, seed=1300 + seed)
    rows = []
    for mon in (esn, maha):
        mon.fit(data["train"])
        theta = float(pick_threshold(
            [mon.score_episode(ep) for ep in data["val"]],
            fa_budget=FA_BUDGET))
        scores = {ep.episode_id: mon.score_episode(ep) for ep in data["test"]}
        summ = summarize(evaluate_alarms(data["test"], scores, theta))
        rows.append({
            "monitor": mon.name,
            "auroc": round(float(episode_auc(data["test"], scores)), 4),
            "detection_rate": round(summ["detection_rate"], 4),
            "healthy_fa_rate": round(summ["healthy_fa_rate"], 4),
            "mean_lead_all": round(summ["mean_lead_all"], 4),
        })
    return rows


#: The L5 pair: same long task, same classes, same onset, different provider.
PAIR = ("real_gemini_long", "real_research7b_long")
HORIZON_MIN = 4


def _horizon(ep) -> float:
    """Post-onset horizon T-1-tau; +inf for healthy episodes."""
    if ep.is_healthy or ep.tau is None:
        return float("inf")
    return float(len(ep.X) - 1 - ep.tau)


def horizon_matched() -> list[dict]:
    """Second candidate explanation: the positives differ in horizon.

    The ESN needs post-onset steps to integrate evidence (H1). If one corpus's
    positives are systematically shorter after onset, its detection is lower for
    that reason alone. Restricting BOTH corpora to positives with horizon >=
    HORIZON_MIN removes the difference; healthy episodes are untouched, so the
    false-alarm budget still means the same thing.
    """
    rows = []
    for name in PAIR:
        data, channels = load_real(REAL_DATASETS[name])
        inj = [ep for ep in data["test"] if not ep.is_healthy]
        short = sum(1 for ep in inj if _horizon(ep) <= 3)
        full = _score(data, channels)
        kept = [ep for ep in data["test"]
                if ep.is_healthy or _horizon(ep) >= HORIZON_MIN]
        matched = _score({**data, "test": kept}, channels)
        n_kept = sum(1 for ep in kept if not ep.is_healthy)
        print(f"[horizon] {name}: {len(inj)} injected "
              f"({short} at horizon<=3) -> {n_kept} kept at horizon>={HORIZON_MIN}")
        for arm, got, n in (("all_positives", full, len(inj)),
                            (f"horizon>={HORIZON_MIN}", matched, n_kept)):
            for r in got:
                rows.append({"dataset": name, "arm": arm, "n_injected": n,
                             "channels": "+".join(channels), **r})
    return rows


def main() -> int:
    out = []
    for name in CORPORA:
        data, channels = load_real(REAL_DATASETS[name])
        if "u" not in channels:
            print(f"[telemetry] {name}: no logprobs, nothing to ablate — skipped")
            continue
        without = tuple(c for c in channels if c != "u")
        n_inj = sum(1 for ep in data["test"] if not ep.is_healthy)
        print(f"[telemetry] {name}: {channels} vs {without} "
              f"(test {len(data['test'])}, injected {n_inj})", flush=True)
        for arm, chans in (("with_logprobs", channels),
                           ("without_logprobs", without)):
            for row in _score(data, chans):
                out.append({"dataset": name, "arm": arm,
                            "channels": "+".join(chans),
                            "n_injected": n_inj, **row})

    if not out:
        print("[telemetry] no corpus with logprobs found")
        return 1
    df = pd.DataFrame(out)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "telemetry_dependence.csv"
    df.to_csv(path, index=False)

    print("\n[telemetry] cost of losing the token-surprisal channel")
    for name in df["dataset"].unique():
        sub = df[df["dataset"] == name]
        for mon in sub["monitor"].unique():
            a = sub[(sub["arm"] == "with_logprobs") & (sub["monitor"] == mon)]
            b = sub[(sub["arm"] == "without_logprobs") & (sub["monitor"] == mon)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            print(f"  {name:22s} {mon:18s} "
                  f"AUROC {a['auroc']:.3f} -> {b['auroc']:.3f} "
                  f"({b['auroc'] - a['auroc']:+.3f})   "
                  f"det {a['detection_rate']:.2f} -> {b['detection_rate']:.2f} "
                  f"({b['detection_rate'] - a['detection_rate']:+.2f})")
    deltas = []
    for name in df["dataset"].unique():
        sub = df[(df["dataset"] == name) & (df["monitor"].str.contains("esn"))]
        if len(sub) == 2:
            w = sub[sub["arm"] == "with_logprobs"].iloc[0]
            o = sub[sub["arm"] == "without_logprobs"].iloc[0]
            deltas.append((o["auroc"] - w["auroc"],
                           o["detection_rate"] - w["detection_rate"]))
    if deltas:
        da = float(np.mean([d[0] for d in deltas]))
        dd = float(np.mean([d[1] for d in deltas]))
        print(f"\n[telemetry] ESN mean effect of dropping u: "
              f"AUROC {da:+.3f}, detection {dd:+.3f} over {len(deltas)} corpora")
    print(f"[telemetry] wrote {path}")

    print("\n[horizon] second candidate explanation for the L5 provider gap")
    hrows = horizon_matched()
    hdf = pd.DataFrame(hrows)
    hpath = TABLES_DIR / "telemetry_horizon_matched.csv"
    hdf.to_csv(hpath, index=False)
    print()
    for name in hdf["dataset"].unique():
        sub = hdf[(hdf["dataset"] == name)
                  & (hdf["monitor"].str.contains("esn"))]
        for _, r in sub.iterrows():
            print(f"  {name:22s} {r['arm']:14s} n={int(r['n_injected']):3d} "
                  f"AUROC {r['auroc']:.3f}  det {r['detection_rate']:.2f} "
                  f"@ FA {r['healthy_fa_rate']:.2f}")
    print(f"[horizon] wrote {hpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
