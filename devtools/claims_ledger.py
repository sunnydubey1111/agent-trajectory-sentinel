"""Recompute every headline claim from the committed artifacts.

A number in a paper drifts from the artifact it came from as soon as either is
edited, and nothing catches it. This tool closes that gap: each claim below
names the artifact it is read from and the value it must equal, the value is
recomputed from that artifact on every run, and a mismatch is a failure rather
than a discrepancy someone has to notice.

    py -m devtools.claims_ledger --check          # verify; non-zero on mismatch
    py -m devtools.claims_ledger --write          # regenerate CLAIMS.md

`CLAIMS.md` is the reader-facing ledger this emits: claim, value, source
artifact, and the command that regenerates that artifact. `tests/` runs
``--check`` so a stale number cannot reach a release.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results" / "tables"
TRACES = REPO_ROOT / "traces"
LEDGER_PATH = REPO_ROOT / "CLAIMS.md"

#: Values are compared at this absolute tolerance, which is looser than float
#: noise and tighter than the rounding used in prose (a claim of "45%" must come
#: from something in [0.445, 0.455], not from 0.42).
TOL = 5e-3


@dataclass
class Claim:
    """One published number, the artifact it is read from, and its value."""

    id: str
    claim: str
    expected: float | int | str
    source: str
    regenerate: str
    compute: Callable[[], float | int | str]
    section: str = "general"
    actual: float | int | str | None = field(default=None, init=False)

    def check(self) -> bool:
        self.actual = self.compute()
        if isinstance(self.expected, str):
            return str(self.actual) == self.expected
        return abs(float(self.actual) - float(self.expected)) <= TOL

    def render(self) -> str:
        if isinstance(self.expected, float):
            return f"{self.expected:.3f}".rstrip("0").rstrip(".")
        return str(self.expected)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def _checks(name: str) -> pd.DataFrame:
    """A verification table with the two check columns coerced to bool."""
    d = _table(name)
    tot = d["total_consistency"].astype(bool)
    return d.assign(caught=tot, anycheck=tot | d["required_coverage"].astype(bool))


def _fail_rate(name: str, column: str) -> float:
    d = _checks(name)
    f = d[d.label != "healthy"]
    return float(f[column].mean())


def _healthy_fp(name: str, column: str) -> int:
    d = _checks(name)
    return int(d[d.label == "healthy"][column].sum())


def _multiseed(monitor: str, column: str) -> float:
    d = _table("multiseed_summary.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _repair_rate(rung: str) -> float:
    d = _table("repair_policies.csv")
    wrong = d[~d.was_correct.astype(bool)]
    return float(wrong[wrong.rung == rung].groupby("rep").now_correct.mean().mean())


def _episode_total() -> int:
    return sum(len(json.loads(m.read_text("utf-8")))
               for m in sorted(TRACES.glob("*/manifest.json")))


def _corpus_count() -> int:
    return len(list(TRACES.glob("*/manifest.json")))


def _real_tool_episodes() -> int:
    return sum(len(json.loads(m.read_text("utf-8")))
               for m in sorted(TRACES.glob("real*/manifest.json")))


def _real_traces(monitor: str, column: str) -> float:
    d = _table("real_traces.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _runtime(monitor: str, column: str) -> float:
    d = _table("runtime.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _contract_within_one_step() -> int:
    d = _table("tool_contract_coverage.csv")
    return int((d.first_violation_step - d.tau <= 1).sum())


def build() -> list[Claim]:
    """Every claim the README, DESIGN.md and both papers make in headline form."""
    return [
        # ---------------------------------------------------------- corpus
        Claim("corpus.episodes", "Committed agent episodes", 2823,
              "traces/*/manifest.json", "py -m devtools.claims_ledger --check",
              _episode_total, "Corpus"),
        Claim("corpus.datasets", "Committed corpora", 25,
              "traces/*/manifest.json", "py -m devtools.claims_ledger --check",
              _corpus_count, "Corpus"),
        Claim("corpus.real_tools", "Episodes using real tools", 770,
              "traces/real*/manifest.json", "py -m devtools.claims_ledger --check",
              _real_tool_episodes, "Corpus"),

        # ----------------------------------------------- behavioural monitor
        Claim("h1.detection", "esn_cusum_max detection (5 seeds)", 0.7065,
              "results/tables/multiseed_summary.csv",
              "py -m derail.experiments.run_multiseed",
              lambda: _multiseed("esn_cusum_max", "detection_rate_mean"), "Monitor"),
        Claim("h1.auc", "esn_cusum_max episode AUC (5 seeds)", 0.87205,
              "results/tables/multiseed_summary.csv",
              "py -m derail.experiments.run_multiseed",
              lambda: _multiseed("esn_cusum_max", "episode_auc_mean"), "Monitor"),
        Claim("h1.lead", "esn_cusum_max mean budget saved (5 seeds)", 4.613,
              "results/tables/multiseed_summary.csv",
              "py -m derail.experiments.run_multiseed",
              lambda: _multiseed("esn_cusum_max", "mean_lead_all_mean"), "Monitor"),
        Claim("h1.baseline", "delta-Mahalanobis detection (5 seeds)", 0.3745,
              "results/tables/multiseed_summary.csv",
              "py -m derail.experiments.run_multiseed",
              lambda: _multiseed("delta_mahalanobis", "detection_rate_mean"), "Monitor"),
        Claim("runtime.latency_us", "Primary monitor median step latency (us)", 219.0,
              "results/tables/runtime.csv", "py -m derail.experiments.run_benchmark",
              lambda: _runtime("esn_cusum_max", "step_latency_us_median"), "Monitor"),
        Claim("runtime.footprint_mb", "Primary monitor state footprint (MB)", 3.95,
              "results/tables/runtime.csv", "py -m derail.experiments.run_benchmark",
              lambda: _runtime("esn_cusum_max", "footprint_mb"), "Monitor"),
        Claim("real.auc", "Channel-max AUC on 187 live Gemini episodes", 0.840084,
              "results/tables/real_traces.csv", "py -m derail.experiments.run_real_traces",
              lambda: _real_traces("esn_cusum_max[e,m]", "episode_auc"), "Monitor"),
        Claim("real.fa", "Channel-max realized false-alarm rate (real traces)", 0.20,
              "results/tables/real_traces.csv", "py -m derail.experiments.run_real_traces",
              lambda: _real_traces("esn_cusum_max[e,m]", "healthy_fa_rate"), "Monitor"),
        Claim("real.context_corruption",
              "Channel-max detection on context corruption (real traces)", 0.285714,
              "results/tables/real_traces.csv", "py -m derail.experiments.run_real_traces",
              lambda: _real_traces("esn_cusum_max[e,m]", "det[context_corruption]"),
              "Monitor"),

        # ------------------------------------------------------ verification
        Claim("holdout.totals", "Held-out failures caught by totals check", 0.5357,
              "results/tables/verification_holdout.csv",
              "py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout",
              lambda: _fail_rate("verification_holdout.csv", "caught"), "Verification"),
        Claim("holdout.coverage", "Held-out failures caught with coverage", 0.9286,
              "results/tables/verification_holdout.csv",
              "py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout",
              lambda: _fail_rate("verification_holdout.csv", "anycheck"), "Verification"),
        Claim("holdout.fp", "Held-out false positives", 0,
              "results/tables/verification_holdout.csv",
              "py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout",
              lambda: _healthy_fp("verification_holdout.csv", "anycheck"), "Verification"),
        Claim("llama.caught", "llama3.1:8b failures caught (all checks)", 1.0,
              "results/tables/verification_organic_llama8b_cold.csv",
              "py -m derail.verify.run_verification_study",
              lambda: _fail_rate("verification_organic_llama8b_cold.csv", "anycheck"),
              "Verification"),
        Claim("llama.fp", "llama3.1:8b false positives", 0,
              "results/tables/verification_organic_llama8b_cold.csv",
              "py -m derail.verify.run_verification_study",
              lambda: _healthy_fp("verification_organic_llama8b_cold.csv", "anycheck"),
              "Verification"),
        Claim("provoked.fabrications", "Provoked fabrications caught", 26,
              "results/tables/verification_provoked.csv",
              "py -m verification.score_provoked_fabrication",
              lambda: int(_checks("verification_provoked.csv")
                          .query("label == 'hallucinated'").anycheck.sum()),
              "Verification"),
        Claim("contract.flagged", "Episodes flagged by tool_contract", 218,
              "results/tables/tool_contract_coverage.csv",
              "py -m derail.verify.run_verification_study --contract-coverage",
              lambda: len(_table("tool_contract_coverage.csv")), "Verification"),
        Claim("contract.immediate",
              "Flagged episodes caught within one step of onset", 215,
              "results/tables/tool_contract_coverage.csv",
              "py -m derail.verify.run_verification_study --contract-coverage",
              _contract_within_one_step, "Verification"),

        # ------------------------------------------------------------ repair
        Claim("repair.located", "`located` recovery rate", 0.4545,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("located"), "Repair"),
        Claim("repair.generic", "`generic` recovery rate", 0.3576,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("generic"), "Repair"),
        Claim("repair.specific", "`specific` recovery rate", 0.3636,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("specific"), "Repair"),
        Claim("repair.recompute", "`recompute` recovery rate (not significant)", 0.2788,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("recompute"), "Repair"),
        Claim("repair.adaptive", "`adaptive` recovery rate (not significant)", 0.2121,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("adaptive"), "Repair"),
        Claim("repair.resample", "`resample` control recovery rate", 0.1636,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("resample"), "Repair"),
        Claim("repair.broken", "Correct runs broken by any repair policy", 0,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: int((~_table("repair_policies.csv")
                           .query("was_correct").now_correct).sum()), "Repair"),
        Claim("repair.flagged_episodes", "Genuinely-wrong episodes in the repair study", 55,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: int(_table("repair_policies.csv")
                          .query("not was_correct").episode_id.nunique()), "Repair"),
        Claim("repair.alarms", "Behavioural alarms followed by a repair attempt", 18,
              "results/tables/alarm_repair.csv",
              "py -m derail.experiments.demo --alarm-repair-matrix",
              lambda: int(_table("alarm_repair.csv").alarm_step.notna().sum()), "Repair"),
    ]


def _render(claims: list[Claim], ok: bool) -> str:
    lines = [
        "# Claim-to-evidence ledger",
        "",
        "Every headline number in the README, `DESIGN.md` and both papers, with",
        "the artifact it is read from and the command that regenerates that",
        "artifact. This file is generated -- edit the study, not the ledger:",
        "",
        "```",
        "py -m devtools.claims_ledger --check    # recompute and verify",
        "py -m devtools.claims_ledger --write    # regenerate this file",
        "```",
        "",
        f"Status at generation: **{'all claims verified' if ok else 'MISMATCHES PRESENT'}**",
        f" ({len(claims)} claims checked).",
        "",
    ]
    for section in ("Corpus", "Monitor", "Verification", "Repair"):
        group = [c for c in claims if c.section == section]
        if not group:
            continue
        lines += [f"## {section}", "",
                  "| claim | value | source artifact | regenerate with |",
                  "|---|---|---|---|"]
        for c in group:
            lines.append(f"| {c.claim} | `{c.render()}` | `{c.source}` | "
                         f"`{c.regenerate}` |")
        lines.append("")
    lines += [
        "## What this ledger does not cover",
        "",
        "Numbers that are properties of a *statistical test* rather than of a",
        "stored table -- p-values, bootstrap intervals, and the per-seed",
        "hypothesis verdicts -- are regenerated by the study runners and checked",
        "by `tests/test_evaluation_validity.py`, not here. The same is true of",
        "the live-demo rehearsal figures, which are measured per run and are",
        "reported as ranges rather than as fixed values.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m devtools.claims_ledger")
    parser.add_argument("--check", action="store_true",
                        help="recompute every claim and fail on a mismatch")
    parser.add_argument("--write", action="store_true",
                        help="regenerate CLAIMS.md")
    args = parser.parse_args(argv)
    if not (args.check or args.write):
        parser.error("pass --check or --write")

    claims = build()
    bad = []
    for c in claims:
        if not c.check():
            bad.append(c)

    for c in bad:
        print(f"MISMATCH {c.id}: expected {c.expected!r}, artifact gives "
              f"{c.actual!r}  ({c.source})", file=sys.stderr)

    if args.write:
        LEDGER_PATH.write_text(_render(claims, not bad), encoding="utf-8",
                               newline="\n")
        print(f"wrote {LEDGER_PATH.name}: {len(claims)} claims, "
              f"{len(bad)} mismatched")
    if args.check and not bad:
        print(f"all {len(claims)} claims match their artifacts")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
