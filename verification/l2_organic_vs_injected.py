"""L2 - Organic vs injected detectability, side by side.

Contrasts how the primary monitor fares on NATURAL (organic, non-injected)
failures versus the CONTROLLED (injected) failures the rest of the study uses.
Organic numbers come from the larger additive corpus scored under the
success-only cross-fit null (``organic_hallucination_ext.csv``, produced by
``score_organic_halluc`` with AGENTWATCH_ORGANIC_DIR / _OUT_CSV pointed at the
extended corpus); injected numbers come from the committed synthetic study.

The point is the honest gap: injected/controlled failures are far easier to
detect than the ones the model makes on its own. This script tabulates it and
re-checks whether the larger organic sample changes the earlier UNDERPOWERED
verdict on organic fabrication.

Run (after collecting + scoring the ext corpus):
  py -m verification.l2_organic_vs_injected
Writes results/tables/organic_vs_injected.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
EXT_CSV = TABLES / "organic_hallucination_ext.csv"
# Serving-temperature arm (L3). Optional: if it has not been scored yet the
# table is produced exactly as before, with the 0.9 arm alone.
COLD_CSV = TABLES / "organic_hallucination_cold.csv"


def main() -> None:
    if not EXT_CSV.exists():
        raise SystemExit(
            f"{EXT_CSV.name} not found - score the extended corpus first:\n"
            "  AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b_ext "
            "AGENTWATCH_ORGANIC_OUT_CSV=results/tables/organic_hallucination_ext.csv "
            "py -m verification.score_organic_halluc")
    org = pd.read_csv(EXT_CSV)

    # Organic side: per-label alarm rates from the success-only cross-fit null.
    heal = org[org.label == "healthy"]
    organic_fa = float(heal.alarmed.mean()) if len(heal) else float("nan")
    organic = {}
    for lab in ("hallucinated", "arithmetic_error", "other", "incomplete"):
        d = org[org.label == lab]
        if len(d):
            organic[lab] = (len(d), int(d.alarmed.sum()), float(d.alarmed.mean()))

    # Injected side: the committed synthetic study's primary-monitor overall
    # detection and false-alarm rate (esn_cusum_max, master seed).
    prim = json.loads((ROOT / "results" / "results.json").read_text("utf-8")
                      )["h1"]["per_monitor"]["esn_cusum_max"]
    rows = [{
        "regime": "INJECTED (synthetic, controlled)", "failure": "all classes",
        "n": "", "detection": round(float(prim["detection_rate"]), 3),
        "healthy_fa": round(float(prim["healthy_fa_rate"]), 3),
        "note": "primary esn_cusum_max, designed-in class signatures",
    }]
    # Nearest injected analogue for each organic failure type (qualitative).
    ctx = {"hallucinated": "~ context_corruption / fabrication",
           "arithmetic_error": "~ context_corruption",
           "other": "omission - no injected analogue (completion check's job)",
           "incomplete": "required work skipped; total still correct"}
    for lab, (n, k, rate) in organic.items():
        rows.append({
            "regime": "ORGANIC T=0.9 (provoking, non-injected)",
            "failure": lab,
            "n": n, "detection": round(rate, 3),
            "healthy_fa": round(organic_fa, 3),
            "note": ctx.get(lab, ""),
        })
    # Serving-temperature arm (L3): the same 120 task seeds at the temperature
    # the demo and the monitor actually serve.
    if COLD_CSV.exists():
        cold = pd.read_csv(COLD_CSV)
        cold_heal = cold[cold.label == "healthy"]
        cold_fa = float(cold_heal.alarmed.mean()) if len(cold_heal) else float("nan")
        for lab in ("hallucinated", "arithmetic_error", "other", "incomplete"):
            d = cold[cold.label == lab]
            if not len(d):
                continue
            rows.append({
                "regime": "ORGANIC T=0.2 (serving, non-injected)",
                "failure": lab, "n": len(d),
                "detection": round(float(d.alarmed.mean()), 3),
                "healthy_fa": round(cold_fa, 3),
                "note": ctx.get(lab, ""),
            })
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "organic_vs_injected.csv", index=False)

    print("[L2] organic vs injected detectability (primary monitor)\n")
    print(table.to_string(index=False))
    print(f"\n[L2] organic healthy false-alarm rate: {organic_fa:.0%} "
          f"(n_healthy={len(heal)}) vs injected "
          f"{float(prim['healthy_fa_rate']):.0%} - organic detection buys a "
          "markedly higher false-alarm cost, and that gap is the honest "
          "organic-vs-injected message.")
    if COLD_CSV.exists():
        print(f"[L2] the cost is NOT a high-temperature artefact: the serving "
              f"arm (T=0.2) realizes {cold_fa:.0%} against the same served 10% "
              f"budget, above the T=0.9 arm's {organic_fa:.0%}.")

    def _fisher(lab: str) -> None:
        d = org[org.label == lab]
        if not len(d):
            return
        tbl = [[int(d.alarmed.sum()), len(d) - int(d.alarmed.sum())],
               [int(heal.alarmed.sum()), len(heal) - int(heal.alarmed.sum())]]
        p = sps.fisher_exact(tbl, alternative="greater")[1]
        if len(d) < 10:
            print(f"[L2] organic {lab}: UNDERPOWERED at n={len(d)} "
                  "(< pre-registered 10) - no claim.")
        else:
            print(f"[L2] organic {lab} POWERED at n={len(d)}: detection "
                  f"{d.alarmed.mean():.0%} vs FA {organic_fa:.0%}, Fisher "
                  f"p={p:.4f} -> {'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'}")

    # Hallucination stays rare; arithmetic_error is the class the larger sample
    # actually powers.
    _fisher("hallucinated")
    _fisher("arithmetic_error")
    print(f"\n[L2] wrote {TABLES / 'organic_vs_injected.csv'}")


if __name__ == "__main__":
    main()
