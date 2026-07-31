"""Score the organic episodes per the pre-registered protocol.

Null is TEMPERATURE-MATCHED: built from the healthy-labelled subset of the
same temperature-0.9 runs, cross-fit 5-fold so no episode is scored by a
monitor that saw it. Labels come from the objective labeller and are fixed
before scoring. No threshold tuning here — the served 10% FA budget.

  py -m verification.score_organic_halluc
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import pick_threshold
from derail.evaluation.protocol import fold_of
from derail.experiments.demo import NUISANCE_DIMS, StreamingContentGate
from derail.telemetry.adapter import load_trace_jsonl
from verification.organic_hallucination import OUT, label_all

ROOT = Path(__file__).resolve().parents[1]
FA_BUDGET = 0.10
# Threshold estimator; see derail.evaluation.metrics.pick_threshold. "empirical"
# lands closer to the served budget than the log-normal tail fit on these
# corpora; the tail fit is worth revisiting where the empirical rule's
# order-statistic floor binds (few calibration episodes).
THETA_METHOD = os.environ.get("AGENTWATCH_THETA_METHOD", "empirical")
# Output CSV is overridable so an ADDITIVE batch (L2) in a separate corpus dir
# writes to its own table instead of clobbering the frozen organic_hallucination.csv.
OUT_CSV = Path(os.environ.get(
    "AGENTWATCH_ORGANIC_OUT_CSV",
    str(ROOT / "results" / "tables" / "organic_hallucination.csv")))


def _load(rows):
    eps = []
    for r in rows:
        ep = load_trace_jsonl(OUT / r["file"], episode_id=r["episode_id"],
                              use_sentence_transformers=False,
                              extended=True, grounding=True)
        for d in NUISANCE_DIMS:                 # same machine-invariance
            ep.X[:, d] = 0.0
        eps.append(ep)
    return eps


def main() -> None:
    rows = label_all()
    df = pd.DataFrame(rows)
    print("labels:", df["label"].value_counts().to_dict(), "\n")

    healthy = [r for r in rows if r["label"] == "healthy"]
    if len(healthy) < 15:
        raise SystemExit(f"only {len(healthy)} healthy episodes — too few "
                         f"to build a null")

    h_eps = _load(healthy)
    non_healthy = [r for r in rows if r["label"] != "healthy"]

    # Score the ACTUAL served monitor's decision: StreamingContentGate.
    # score_step fuses behaviour, grounding AND the lexical override into a
    # display score whose alarm line is 1.0.
    #
    # K=5 fold, by episode id (label-independent). Fold k's gate is fit and
    # thresholded on healthy OUTSIDE fold k, then scores every episode assigned
    # to fold k - healthy and hallucinated alike - so the threshold cohort is
    # disjoint from the scored episode and both classes go through one rule.
    K = 5

    def _fold(eid: str) -> int:
        return fold_of(eid, K, salt="organic")

    def _alarmed(gate: StreamingContentGate, ep) -> tuple[bool, float]:
        gate.start_episode()
        peak = max(float(gate.score_step(x)) for x in ep.X)
        return peak >= 1.0, peak       # 1.0 is the served alarm line

    gates: dict[int, StreamingContentGate] = {}
    fit_ids: dict[int, set] = {}
    for k in range(K):
        fit_h = [ep for ep, r in zip(h_eps, healthy)
                 if _fold(r["episode_id"]) != k]
        gate = StreamingContentGate(Standardizer().fit(fit_h))
        gate.fit(fit_h)
        gates[k] = gate
        fit_ids[k] = {r["episode_id"] for r in healthy
                      if _fold(r["episode_id"]) != k}

    # ---- threshold calibration, NESTED and out-of-fold -------------------
    # theta must never be read off episodes the gate was FIT on: those score
    # optimistically low, so theta lands low and the realized false-alarm rate
    # overshoots the budget it claims. Pass 1 scores every healthy episode with
    # the gate that did not see it; pass 2 sets fold k's threshold from the
    # out-of-fold scores of the other folds, so no episode helps choose the
    # threshold it is later measured against.
    oof_b: dict[int, np.ndarray] = {}
    for idx, (r, ep) in enumerate(zip(healthy, h_eps)):
        oof_b[idx] = gates[_fold(r["episode_id"])].score_episode_streams(ep)[0]
    theta: dict[int, float] = {}
    for k in range(K):
        cohort = [s for idx, s in oof_b.items()
                  if _fold(healthy[idx]["episode_id"]) != k]
        theta[k] = float(pick_threshold(cohort, fa_budget=FA_BUDGET,
                                        method=THETA_METHOD))
        gates[k]._theta_b10 = gates[k]._theta_b5 = theta[k]
    print(f"temperature-matched null: {len(h_eps)} healthy episodes, "
          f"{K}-fold served-gate scoring (alarm line 1.0)")
    print(f"nested out-of-fold theta per fold: "
          f"{[round(theta[k], 1) for k in range(K)]}\n")

    out = []
    for r, ep in zip(healthy, h_eps):
        k = _fold(r["episode_id"])
        assert r["episode_id"] not in fit_ids[k]       # scored out-of-fold
        alarmed, peak = _alarmed(gates[k], ep)
        out.append({"episode_id": r["episode_id"], "label": r["label"],
                    "peak": round(peak, 3), "alarmed": bool(alarmed),
                    "evidence": r["evidence"]})
    for r in non_healthy:
        ep = _load([r])[0]
        # Scored by a fold gate too (fit on healthy only, so out-of-sample) -
        # the SAME model family the healthy negatives use.
        alarmed, peak = _alarmed(gates[_fold(r["episode_id"])], ep)
        out.append({"episode_id": r["episode_id"], "label": r["label"],
                    "peak": round(peak, 3), "alarmed": bool(alarmed),
                    "evidence": r["evidence"]})
    res = pd.DataFrame(out)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)

    print(f"{'label':<18}{'n':>4}{'alarmed':>9}{'rate':>8}")
    for lab in ("healthy", "hallucinated", "arithmetic_error", "other"):
        d = res[res.label == lab]
        if len(d):
            print(f"{lab:<18}{len(d):>4}{int(d.alarmed.sum()):>9}"
                  f"{100 * d.alarmed.mean():>7.0f}%")

    hall = res[res.label == "hallucinated"]
    heal = res[res.label == "healthy"]
    print()
    if len(hall) < 10:
        print(f"[VERDICT] UNDERPOWERED — only {len(hall)} hallucinated "
              f"episodes (pre-registered minimum 10). No claim made.")
        return
    tbl = [[int(hall.alarmed.sum()), len(hall) - int(hall.alarmed.sum())],
           [int(heal.alarmed.sum()), len(heal) - int(heal.alarmed.sum())]]
    p = sps.fisher_exact(tbl, alternative="greater")[1]
    ok = p < 0.05
    print(f"[VERDICT] detection {100*hall.alarmed.mean():.0f}% vs healthy FA "
          f"{100*heal.alarmed.mean():.0f}%, Fisher exact p = {p:.4f} -> "
          f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
