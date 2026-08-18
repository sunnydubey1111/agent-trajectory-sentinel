"""T5 — label support + frozen-monitor scoring of organic episodes.

Two modes:

  --dump   Write a human-readable review sheet (per episode: task, each
           step's text truncated, tool errors) to
           traces/organic7b/review_sheet.txt, plus a labels template.
           A human (or an auditable manual review) then fills
           traces/organic7b/organic_labels.csv with columns:
             episode_id,label,failure_mode,onset_step,evidence
           label in {healthy, failed}; failure_mode free text (e.g.
           looping, junk_tokens, leaked_tool_syntax, wrong_answer,
           gave_up); onset_step = first visibly bad step (approximate);
           evidence = short quote. The monitors NEVER see the labels.

  (default) Score every organic episode with monitors fitted on the
           FROZEN real_research7b healthy train split and thresholded on
           its val split at the deployed 5% budget — no refit, no
           threshold tuning on organic data. Reports alarm rates for
           labeled-failed vs labeled-healthy episodes, per failure mode,
           per monitor. Writes results/tables/organic_validation.csv.

The claim under test: monitors calibrated only on injected-failure
methodology still fire on failures nobody injected — or they don't, and
we say so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import Standardizer
from derail.evaluation.metrics import pick_threshold
from derail.experiments.run_grounding_study import (
    GRD_DIM_NAMES, _view51, load_real)
from derail.experiments.run_hybrid_study import REAL_DATASETS, TABLES_DIR
from derail.monitor.grounding import (
    GroundingMonitor,
    HybridContentGate,
    HybridWeightedG,
)
from derail.monitor.hybrid import HybridWeighted, make_hybrids
from derail.telemetry.adapter import load_trace_jsonl, parse_tool_bits

ORGANIC = Path(__file__).resolve().parents[2] / "traces" / "organic7b"
FA_BUDGET = 0.05


def dump_review_sheet() -> None:
    manifest = json.loads((ORGANIC / "manifest.json").read_text("utf-8"))
    lines, template = [], ["episode_id,label,failure_mode,onset_step,evidence"]
    for e in manifest:
        steps = [json.loads(x) for x in
                 (ORGANIC / e["file"]).read_text("utf-8").splitlines() if x]
        lines.append(f"=== {e['episode_id']} (T={len(steps)}) ===")
        for t, s in enumerate(steps):
            text = str(s.get("text", "")).replace("\n", " ")
            calls, _ = parse_tool_bits(text)
            errs = sum(1 for ev in calls if ev.is_error)
            flag = f"  [{errs} tool errors]" if errs else ""
            lines.append(f"  step {t}: {text[:300]}{flag}")
        lines.append("")
        template.append(f"{e['episode_id']},,,,")
    (ORGANIC / "review_sheet.txt").write_text("\n".join(lines), "utf-8")
    tpl = ORGANIC / "organic_labels.csv"
    if not tpl.exists():
        tpl.write_text("\n".join(template), "utf-8")
    print(f"[organic] review sheet + labels template -> {ORGANIC}")


def score() -> None:
    labels = pd.read_csv(ORGANIC / "organic_labels.csv")
    assert labels["label"].isin(["healthy", "failed"]).all(), \
        "fill organic_labels.csv first (label in {healthy, failed})"

    # frozen deployed configuration: research7b splits, no refit
    data, channels = load_real(REAL_DATASETS["real_research7b"],
                               grounding=True)
    train, val = data["train"], data["val"]

    # Behavioural monitors see the published 51-dim view; the grounded gate
    # masks its behavioural submodels to the same 51 dims via behav_slice,
    # so their pre-fit submodels and standardizers must be fit on
    # the 51-dim view too — fitting them on the full 60-dim telemetry both
    # double-counts grounding and breaks the 51-vs-60 scoring shape.
    train51, val51 = _view51(train), _view51(val)
    std51 = Standardizer().fit(train51)
    esn, maha, hybrids2 = make_hybrids(std51, channels=channels)
    esn.fit(train51)
    maha.fit(train51)
    weighted = next(h for h in hybrids2 if isinstance(h, HybridWeighted))
    weighted.fit(train51)

    std56 = Standardizer().fit(train51)
    esn56, maha56, _ = make_hybrids(std56, channels=channels)
    esn56.fit(train51)
    maha56.fit(train51)
    grd = GroundingMonitor()
    grd.fit(train)
    grd_cont = GroundingMonitor(dims=GRD_DIM_NAMES[:-1],
                                name="grounding_cont")
    grd_cont.fit(train)
    gate = HybridContentGate(esn56, maha56, grd_cont, std56, subs_prefit=True)
    gate.fit(train)
    wg = HybridWeightedG(esn56, maha56, grd_cont, std56, subs_prefit=True)
    wg.fit(train)

    manifest = json.loads((ORGANIC / "manifest.json").read_text("utf-8"))
    episodes = {e["episode_id"]: load_trace_jsonl(
        ORGANIC / e["file"], episode_id=e["episode_id"], tau=None,
        failure_class=None, use_sentence_transformers=False,
        extended=True, grounding=True) for e in manifest}
    # 51-dim views keyed by id, for the behavioural monitors.
    episodes51 = {eid: _view51([ep])[0] for eid, ep in episodes.items()}

    # (monitor, val episodes, per-id episode dict) — plain monitors run on the
    # 51-dim view, grounded monitors on the full 60-dim telemetry.
    monitors = [(esn, val51, episodes51), (maha, val51, episodes51),
                (weighted, val51, episodes51), (grd, val, episodes),
                (wg, val, episodes), (gate, val, episodes)]
    rows = []
    for mon, val_eps, ep_dict in monitors:
        theta = float(pick_threshold([mon.score_episode(ep) for ep in val_eps],
                                     fa_budget=FA_BUDGET))
        for _, lab in labels.iterrows():
            ep = ep_dict[lab["episode_id"]]
            s = mon.score_episode(ep)
            alarm = int(np.flatnonzero(s > theta)[0]) \
                if np.any(s > theta) else None
            rows.append({
                "dataset": ORGANIC.name,
                "monitor": mon.name, "episode_id": lab["episode_id"],
                "label": lab["label"],
                "failure_mode": lab.get("failure_mode", ""),
                "alarmed": alarm is not None, "alarm_step": alarm,
                "max_score": float(np.max(s)),
                "T": ep.T, "theta": round(theta, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "organic_validation.csv", index=False)

    print("[organic] alarm rate at deployed thresholds "
          "(frozen research7b calibration, no refit):")
    pv = df.pivot_table(index="monitor", columns="label", values="alarmed",
                        aggfunc="mean")
    print(pv.round(2).to_string())
    failed = df[df.label == "failed"]
    if len(failed):
        print("\nper failure mode (fraction alarmed):")
        print(failed.pivot_table(index="failure_mode", columns="monitor",
                                 values="alarmed", aggfunc="mean")
              .round(2).to_string())
    # threshold-free discrimination WITHIN the organic distribution:
    # can the scores rank failed above healthy at all, independent of the
    # (temperature-mismatched) transferred thresholds?
    from sklearn.metrics import roc_auc_score
    print("\nwithin-organic episode AUROC (failed vs healthy, "
          "threshold-free):")
    for mon_name, sub in df.groupby("monitor"):
        y = (sub["label"] == "failed").astype(int)
        print(f"  {mon_name:>20s}: "
              f"{roc_auc_score(y, sub['max_score']):.3f}")
    print(f"\n[organic] wrote {TABLES_DIR / 'organic_validation.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.score_organic")
    parser.add_argument("--dump", action="store_true",
                        help="write review sheet + labels template only")
    args = parser.parse_args(argv)
    if args.dump:
        dump_review_sheet()
    else:
        score()


if __name__ == "__main__":
    main()
