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
import re
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
    #: How many episodes the rate was computed over, and what that count must
    #: be. A rate with no denominator cannot be sanity-checked against
    #: anything: "AUC 0.840 on 187 episodes" passed for months while the
    #: number came from a held-out split of 94, because the table carries no
    #: n column and the ledger had nothing to compare. Supplying a callable
    #: makes the denominator drift-checked like every other number here.
    denominator: Callable[[], int] | None = None
    expected_denominator: int | None = None
    #: What the denominator counts. Not every rate here averages over
    #: episodes -- some are grand means over datasets, some over seeds -- and
    #: a bare "8" next to an AUROC reads as eight episodes, which would be a
    #: fresh version of the error this field exists to stop.
    denominator_unit: str = "episodes"
    actual: float | int | str | None = field(default=None, init=False)
    actual_denominator: int | None = field(default=None, init=False)

    def check(self) -> bool:
        self.actual = self.compute()
        if self.denominator is not None:
            self.actual_denominator = int(self.denominator())
            if self.actual_denominator != self.expected_denominator:
                return False
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


def _our_manifests() -> list[pathlib.Path]:
    """Manifests of corpora this project collected.

    A leading underscore marks a directory that is not ours - scratch output,
    or a corpus imported from another project (traces/_aftraj). Counting those
    would restate someone else's episodes as ours.
    """
    return [m for m in sorted(TRACES.glob("*/manifest.json"))
            if not m.parent.name.startswith("_")]


def _episode_total() -> int:
    return sum(len(json.loads(m.read_text("utf-8"))) for m in _our_manifests())


def _corpus_count() -> int:
    return len(_our_manifests())


def _real_tool_episodes() -> int:
    return sum(len(json.loads(m.read_text("utf-8")))
               for m in sorted(TRACES.glob("real*/manifest.json")))


def _real_traces(monitor: str, column: str) -> float:
    d = _table("real_traces.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _runtime(monitor: str, column: str) -> float:
    d = _table("runtime.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _telemetry_runtime(column: str) -> float:
    return float(_table("telemetry_runtime.csv")[column].iloc[0])


def _aftraj_latency(monitor: str) -> float:
    d = _table("aftraj_benchmark.csv")
    return float(d.loc[d.monitor == monitor, "step_latency_us"].iloc[0])


# ---------------------------------------------------------------------------
# Table-shaped claims.
#
# Everything below backs a TABLE that appears in a document rather than a
# single number in a sentence. Those were the ledger's blind spot: a table is
# reconstructed by hand in prose, so it drifts cell by cell and no single
# claim covers it. Two did drift -- the grounding table sat at an n=602 run
# after the study grew to 874, and DESIGN.md's repair table disagreed with its
# own CSV on every rung -- and neither was caught by anything here.
# ---------------------------------------------------------------------------

def _grounding_group(content: bool) -> pd.DataFrame:
    d = _table("grounding_diagnosis.csv")
    return d[d.is_content.astype(bool) == content]


def _grounding_n(content: bool) -> int:
    return int(len(_grounding_group(content)))


def _grounding_rate(column: str, content: bool) -> float:
    return float(_grounding_group(content)[column].astype(bool).mean())


def _vs_monitor(arm: str, column: str, healthy: bool = False) -> float:
    """One cell of the checks-versus-monitor table.

    The table is quoted in four documents, so a drifted cell would have to be
    corrected in all four; recomputing it here is what makes that visible.
    """
    d = _table("verification_vs_monitor.csv")
    a = d[d.arm.str.startswith(arm)]
    rows = a[a.label == "healthy"] if healthy else a[a.label != "healthy"]
    return float(rows[column].sum() / rows["n"].sum())


def _vs_monitor_n(arm: str, healthy: bool = False) -> int:
    d = _table("verification_vs_monitor.csv")
    a = d[d.arm.str.startswith(arm)]
    rows = a[a.label == "healthy"] if healthy else a[a.label != "healthy"]
    return int(rows["n"].sum())


def _repair_recovered(rung: str) -> float:
    """Mean count of episodes a rung turns correct, over the three repeats."""
    d = _table("repair_policies.csv")
    wrong = d[~d.was_correct.astype(bool)]
    return float(wrong[wrong.rung == rung].groupby("rep")
                 .now_correct.apply(lambda s: s.astype(bool).sum()).mean())


def _repair_net(rung: str | None) -> float:
    """Net task success over the whole 120-episode arm, not just the flagged.

    The 65 unflagged episodes are untouched by any policy, so the net rate is
    the baseline correct count plus what the rung recovers. The baseline comes
    from the same corpus's verification table (63 healthy of 120), which is
    why this is recomputable rather than a number copied from a runner's
    stdout.
    """
    v = _table("verification_cold.csv")
    baseline = int((v.label == "healthy").sum())
    total = int(len(v))
    recovered = 0.0 if rung is None else _repair_recovered(rung)
    return (baseline + recovered) / total


def _hybrid_grand_mean(monitor: str) -> float:
    """Grand-mean AUROC across the eight benchmark datasets."""
    d = _table("hybrid_benchmark.csv")
    return float(d.loc[d.monitor == monitor, "auroc"].mean())


def _horizon_advantage(lo: int, hi: int) -> float:
    """ESN-minus-Mahalanobis detection gap inside one post-onset horizon band."""
    d = _table("hybrid_diagnosis.csv")
    band = d[(d.horizon >= lo) & (d.horizon <= hi)]
    return float(band.det_esn.mean() - band.det_maha.mean())


def _aftraj(monitor: str, column: str) -> float:
    """One monitor's value on the imported AFTraj-2K corpus (one row)."""
    d = _table("aftraj_benchmark.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _aftraj_horizon(lo: int, hi: int, column: str) -> float:
    d = _table("aftraj_diagnosis.csv")
    horizon = d["T"] - 1 - d["tau"]
    band = d[(horizon >= lo) & (horizon <= hi)]
    return float(band[column].mean()) if column in band else float(len(band))


#: Sources and regenerate commands shared by the table claims below, named
#: once so a table's provenance cannot drift between its own cells.
GROUNDING_SRC = "results/tables/grounding_diagnosis.csv"
GROUNDING_CMD = "py -m derail.experiments.run_grounding_study"
VS_SRC = "results/tables/verification_vs_monitor.csv"
VS_CMD = "py -m derail.verify.run_verification_study"
REPAIR_CMD = "py -m derail.intervene.evaluate_repair_policies --from-csv"

#: The episode-length floor every real-trace study applies before splitting.
#: Kept here so the denominators below cannot silently disagree with the
#: runners about which episodes were in scope.
_MIN_T = 4


def _real_split() -> dict[str, int]:
    """Split sizes of the root Gemini corpus, from the committed manifest.

    Recomputed rather than read: real_traces.csv has no n column, which is
    precisely how "AUC on 187 episodes" survived when the number came from a
    held-out split. The rule is the runner's - drop episodes under _MIN_T,
    then 60/20/20 over the healthy ones - so this stays honest as long as the
    two agree, and `tests/` asserts that they do.
    """
    manifest = json.loads((TRACES / "manifest.json").read_text("utf-8"))
    kept = [e for e in manifest if e["T"] >= _MIN_T]
    healthy = [e for e in kept if e["tau"] is None]
    n_train = round(0.6 * len(healthy))
    n_val = round(0.2 * len(healthy))
    return {"healthy_train": n_train, "healthy_val": n_val,
            "healthy_test": len(healthy) - n_train - n_val,
            "injected": sum(1 for e in kept if e["tau"] is not None)}


def _real_eval_n() -> int:
    """Episodes the real-trace AUC is computed over: held-out healthy + all
    injected. Not the 187 the corpus contains."""
    split = _real_split()
    return split["healthy_test"] + split["injected"]


def _corpus_eval_n(corpus: str) -> int:
    """Held-out healthy plus all injected for a corpus subdirectory.

    Same rule as `_real_split`, applied to the corpora that live under
    `traces/<name>/`.
    """
    manifest = json.loads(
        (TRACES / corpus / "manifest.json").read_text("utf-8"))
    kept = [e for e in manifest if e["T"] >= _MIN_T]
    healthy = [e for e in kept if e["tau"] is None]
    held_out = len(healthy) - round(0.6 * len(healthy)) - round(0.2 * len(healthy))
    return held_out + sum(1 for e in kept if e["tau"] is not None)


def _atbench_eval_n() -> int:
    """ATBench scores held-out safe against every unsafe trajectory."""
    d = _table("atbench_benchmark.csv").iloc[0]
    return int(d.n_safe_test) + int(d.n_unsafe)


def _atbench_mode_n(mode: str) -> int:
    d = _table("atbench_per_mode.csv")
    row = d[(d.monitor == "esn_cusum_max") & (d.failure_mode == mode)]
    return int(row.n.iloc[0])


def _judge_n(field_name: str) -> int:
    """Positives or negatives the measured judge was scored on."""
    payload = json.loads(
        (TABLES / "judge_calibration_summary.json").read_text("utf-8"))
    return int(payload[field_name])


def _multiseed_seed_count() -> int:
    """How many master seeds the multiseed means average over.

    Read from the runner's own SEEDS tuple rather than hard-coded, so adding
    a seed without regenerating the summary fails the ledger.
    """
    source = (REPO_ROOT / "derail" / "experiments"
              / "run_multiseed.py").read_text("utf-8")
    match = re.search(r"^SEEDS\s*=\s*\(([^)]*)\)", source, re.M)
    if not match:
        raise RuntimeError("run_multiseed.py no longer defines SEEDS")
    return len([p for p in match.group(1).split(",") if p.strip()])


def _sim_test_n() -> int:
    """Simulator test episodes per seed, from the config the runners use.

    Arithmetic on DatasetConfig rather than a generated dataset, so it costs
    nothing and still fails if a split size changes.
    """
    from derail.common import FAILURE_CLASSES, DatasetConfig

    cfg = DatasetConfig()
    return (cfg.n_test_healthy
            + cfg.n_test_injected_per_class * len(FAILURE_CLASSES))


def _repair_episodes() -> int:
    """Distinct genuinely-wrong episodes every repair rung is scored over."""
    d = _table("repair_policies.csv")
    return int(d.episode_id.nunique())


def _hybrid_datasets() -> int:
    """Datasets a hybrid grand mean is averaged over."""
    return int(_table("hybrid_benchmark.csv").dataset.nunique())


def _aftraj_injected() -> int:
    return int(len(_table("aftraj_diagnosis.csv")))


def _atbench(monitor: str, column: str) -> float:
    d = _table("atbench_benchmark.csv")
    return float(d.loc[d.monitor == monitor, column].iloc[0])


def _atbench_mode(monitor: str, mode: str) -> float:
    d = _table("atbench_per_mode.csv")
    row = d[(d.monitor == monitor) & (d.failure_mode == mode)]
    return float(row.detection_rate.iloc[0])


def _criterion(monitor: str, group: str, column: str) -> float:
    """Worst per-seed delta for one grounded fusion on one failure group."""
    d = _table("grounding_multiseed_criterion.csv")
    g = d[(d.monitor == monitor) & (d.group == group)]
    return float(g[column].min())


def _model_transfer_auroc(condition_contains: str) -> float:
    """Best AUROC any monitor reaches under a transfer condition."""
    d = _table("model_transfer.csv")
    return float(d[d.condition.str.contains(condition_contains)].auroc.max())


def _every_alarm_attempted() -> str:
    """Did every behavioural alarm get a repair attempt, by either trigger?"""
    d = _table("alarm_repair.csv")
    alarmed = d.alarm_step.notna()
    attempted = (d.alarm_repair_used.astype(bool)
                 | d.check_repair_used.astype(bool))
    missed = int((alarmed & ~attempted).sum())
    return ("all alarms attempted" if missed == 0
            else f"{missed} alarm(s) with no repair attempt")


def _judge(key: str) -> float:
    """A measured judge rate from the calibration summary sidecar."""
    path = TABLES / "judge_calibration_summary.json"
    return float(json.loads(path.read_text("utf-8"))[key])


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
              lambda: _multiseed("esn_cusum_max", "episode_auc_mean"), "Monitor",
              denominator=_sim_test_n, expected_denominator=560,
              denominator_unit="episodes/seed, 5 seeds"),
        Claim("h1.lead", "esn_cusum_max mean budget saved (5 seeds)", 4.613,
              "results/tables/multiseed_summary.csv",
              "py -m derail.experiments.run_multiseed",
              lambda: _multiseed("esn_cusum_max", "mean_lead_all_mean"), "Monitor"),
        Claim("h1.baseline", "delta-Mahalanobis detection (5 seeds)", 0.3745,
              "results/tables/multiseed_summary.csv",
              "py -m derail.experiments.run_multiseed",
              lambda: _multiseed("delta_mahalanobis", "detection_rate_mean"), "Monitor"),
        # Latency is the one published figure that is NOT bit-reproducible: it
        # measures the machine, and re-running the benchmark on the same box
        # moved it 219 -> 252 us. The committed runtime.csv is the source of
        # record for the quoted value; what the ledger can honestly assert is
        # the order of magnitude the "three orders below a judge call" claim
        # rests on, so that is what it checks.
        Claim("runtime.latency_order", "Primary monitor step latency is 100-999 us",
              "100-999 us", "results/tables/runtime.csv",
              "py -m derail.experiments.run_benchmark (timings are machine-specific)",
              lambda: ("100-999 us"
                       if 100.0 <= _runtime("esn_cusum_max", "step_latency_us_median") < 1000.0
                       else "OUT OF RANGE"), "Monitor"),
        Claim("runtime.footprint_mb", "Primary monitor state footprint (MB)", 3.95,
              "results/tables/runtime.csv", "py -m derail.experiments.run_benchmark",
              lambda: _runtime("esn_cusum_max", "footprint_mb"), "Monitor"),
        # The claims below pin the EXACT timings the papers quote, which the
        # range claim above deliberately does not. The two answer different
        # questions and both are worth asking. The range asks whether the
        # architectural claim ("three orders below a judge call") still holds,
        # and must survive a hardware change. These ask whether the number
        # printed in a paper still equals the committed CSV it was read from --
        # and that is the check that was missing: a paper carried 608 us/step
        # for the telemetry cost against a committed 673.7 for months, because
        # no claim named it and nothing recomputes prose. Re-running the
        # benchmark on different hardware SHOULD fail these, because the prose
        # is then stale and wants updating.
        Claim("runtime.latency_us",
              "Primary monitor step latency, median us (machine-specific)",
              219.0, "results/tables/runtime.csv",
              "py -m derail.experiments.run_benchmark (timings are machine-specific)",
              lambda: _runtime("esn_cusum_max", "step_latency_us_median"),
              "Monitor",
              denominator=lambda: int(_runtime("esn_cusum_max", "n_steps_timed")),
              expected_denominator=4316, denominator_unit="timed steps"),
        Claim("runtime.maha_latency_us",
              "delta-Mahalanobis step latency, median us -- the baseline the "
              "reservoir is ~50x more expensive than", 4.0,
              "results/tables/runtime.csv",
              "py -m derail.experiments.run_benchmark (timings are machine-specific)",
              lambda: _runtime("delta_mahalanobis", "step_latency_us_median"),
              "Monitor",
              denominator=lambda: int(_runtime("delta_mahalanobis", "n_steps_timed")),
              expected_denominator=4316, denominator_unit="timed steps"),
        Claim("telemetry.v4_step_us",
              "Full v4 telemetry construction cost at the adapter, median us",
              673.7, "results/tables/telemetry_runtime.csv",
              "py -m experimental.telemetry_runtime (timings are machine-specific)",
              lambda: _telemetry_runtime("median_us"), "Monitor",
              denominator=lambda: int(_telemetry_runtime("n_steps")),
              expected_denominator=491, denominator_unit="timed steps"),
        Claim("telemetry.v4_step_p95_us",
              "Full v4 telemetry construction cost, p95 us", 1045.0,
              "results/tables/telemetry_runtime.csv",
              "py -m experimental.telemetry_runtime (timings are machine-specific)",
              lambda: _telemetry_runtime("p95_us"), "Monitor"),
        Claim("aftraj.esn_latency_us",
              "Channel-max ESN step latency on AFTraj-2K, median us "
              "(NOT the hybrid's 172.6 -- the two were once conflated)",
              162.8, "results/tables/aftraj_benchmark.csv",
              "py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj",
              lambda: _aftraj_latency("esn_cusum_max"), "Monitor"),
        # Named for the evaluation set, not the corpus. The corpus is 187
        # episodes; the number is computed on a held-out split of 79 injected
        # and 15 healthy drawn from it, and calling it "on 187" reads as a
        # denominator it never had.
        Claim("real.auc",
              "Channel-max AUC, held-out split of the 187-episode Gemini "
              "corpus (79 injected + 15 healthy)", 0.840084,
              "results/tables/real_traces.csv", "py -m derail.experiments.run_real_traces",
              lambda: _real_traces("esn_cusum_max[e,m]", "episode_auc"), "Monitor",
              denominator=_real_eval_n, expected_denominator=94),
        # 0.20 is 3 of 15 healthy test episodes: one episode is worth 6.7
        # points, which is the whole reason the 5% budget is unreachable here.
        Claim("real.fa",
              "Channel-max realized false-alarm rate, 15 healthy test "
              "episodes (real traces)", 0.20,
              "results/tables/real_traces.csv", "py -m derail.experiments.run_real_traces",
              lambda: _real_traces("esn_cusum_max[e,m]", "healthy_fa_rate"), "Monitor",
              denominator=lambda: _real_split()["healthy_test"],
              expected_denominator=15),
        Claim("real.context_corruption",
              "Channel-max detection on context corruption (real traces)", 0.285714,
              "results/tables/real_traces.csv", "py -m derail.experiments.run_real_traces",
              lambda: _real_traces("esn_cusum_max[e,m]", "det[context_corruption]"),
              "Monitor"),

        Claim("hybrid.weighted50", "hybrid_weighted50 grand-mean AUROC (label-free default)",
              0.8119, "results/tables/hybrid_benchmark.csv",
              "py -m derail.experiments.run_hybrid_study",
              lambda: _hybrid_grand_mean("hybrid_weighted50"), "Monitor",
              denominator=_hybrid_datasets, expected_denominator=8,
              denominator_unit="datasets"),
        Claim("hybrid.logistic", "hybrid_logistic grand-mean AUROC (with labels)",
              0.8262, "results/tables/hybrid_benchmark.csv",
              "py -m derail.experiments.run_hybrid_study",
              lambda: _hybrid_grand_mean("hybrid_logistic"), "Monitor",
              denominator=_hybrid_datasets, expected_denominator=8,
              denominator_unit="datasets"),
        Claim("hybrid.esn", "esn_cusum_max grand-mean AUROC on the same eight datasets",
              0.8020, "results/tables/hybrid_benchmark.csv",
              "py -m derail.experiments.run_hybrid_study",
              lambda: _hybrid_grand_mean("esn_cusum_max"), "Monitor",
              denominator=_hybrid_datasets, expected_denominator=8,
              denominator_unit="datasets"),
        # ------------------------------------------- external validation
        # AFTraj-2K is another project's corpus, imported by
        # derail.experiments.import_aftraj. The tables ARE committed; the
        # traces are not, so these regenerate only after the import step.
        Claim("aftraj.esn_auroc",
              "esn_cusum_max episode AUROC on AFTraj-2K (external)", 0.7452,
              "results/tables/aftraj_benchmark.csv",
              "py -m derail.experiments.import_aftraj && "
              "py -m derail.experiments.run_hybrid_study --datasets aftraj "
              "--out-prefix aftraj",
              lambda: _aftraj("esn_cusum_max", "auroc"), "Monitor",
              denominator=_aftraj_injected, expected_denominator=771),
        Claim("aftraj.esn_detection",
              "esn_cusum_max detection on AFTraj-2K at the 5% budget", 0.0480,
              "results/tables/aftraj_benchmark.csv",
              "py -m derail.experiments.run_hybrid_study --datasets aftraj "
              "--out-prefix aftraj",
              lambda: _aftraj("esn_cusum_max", "detection_rate"), "Monitor"),
        Claim("aftraj.deep_horizon_detection",
              "esn_cusum_max detection on AFTraj-2K failures with >= 9 steps "
              "of post-onset horizon", 0.5094,
              "results/tables/aftraj_diagnosis.csv",
              "py -m derail.experiments.run_hybrid_study --datasets aftraj "
              "--out-prefix aftraj",
              lambda: _aftraj_horizon(9, 10**6, "det_esn"), "Monitor"),
        Claim("aftraj.deep_horizon_share",
              "AFTraj-2K failures with >= 9 steps of post-onset horizon", 53,
              "results/tables/aftraj_diagnosis.csv",
              "py -m derail.experiments.run_hybrid_study --datasets aftraj "
              "--out-prefix aftraj",
              lambda: _aftraj_horizon(9, 10**6, "__count__"), "Monitor"),

        # ATBench: a second external corpus, trajectory-labelled only, so
        # AUROC and alarm-anywhere detection are the only defined quantities.
        Claim("atbench.esn_auroc",
              "esn_cusum_max episode AUROC on ATBench (external)", 0.7787,
              "results/tables/atbench_benchmark.csv",
              "py -m derail.experiments.run_atbench_study",
              lambda: _atbench("esn_cusum_max", "auroc"), "Monitor",
              denominator=_atbench_eval_n, expected_denominator=381),
        Claim("atbench.esn_detection",
              "esn_cusum_max detection on ATBench at the 5% budget", 0.3108,
              "results/tables/atbench_benchmark.csv",
              "py -m derail.experiments.run_atbench_study",
              lambda: _atbench("esn_cusum_max", "detection_rate"), "Monitor"),
        Claim("atbench.hybrid_auroc",
              "hybrid_weighted50 episode AUROC on ATBench (fusion collapses "
              "to chance when a parent does)", 0.4626,
              "results/tables/atbench_benchmark.csv",
              "py -m derail.experiments.run_atbench_study",
              lambda: _atbench("hybrid_weighted50", "auroc"), "Monitor",
              denominator=_atbench_eval_n, expected_denominator=381),
        Claim("atbench.action_failure_detection",
              "esn_cusum_max detection on unconfirmed/over-privileged actions "
              "(ATBench)", 0.5085,
              "results/tables/atbench_per_mode.csv",
              "py -m derail.experiments.run_atbench_study",
              lambda: _atbench_mode("esn_cusum_max",
                                    "unconfirmed_or_over_privileged_action"),
              "Monitor"),
        Claim("atbench.content_failure_detection",
              "esn_cusum_max detection on inaccurate/misleading information "
              "(ATBench, the known content blind spot)", 0.0385,
              "results/tables/atbench_per_mode.csv",
              "py -m derail.experiments.run_atbench_study",
              lambda: _atbench_mode(
                  "esn_cusum_max",
                  "provide_inaccurate_misleading_or_unverified_information"),
              "Monitor",
              denominator=lambda: _atbench_mode_n(
                  "provide_inaccurate_misleading_or_unverified_information"),
              expected_denominator=26),

        Claim("horizon.short", "ESN advantage at post-onset horizon <= 3 steps",
              0.0865, "results/tables/hybrid_diagnosis.csv",
              "py -m derail.experiments.run_hybrid_study",
              lambda: _horizon_advantage(0, 3), "Monitor"),
        Claim("horizon.mid", "ESN advantage at post-onset horizon 4-8 steps",
              0.1353, "results/tables/hybrid_diagnosis.csv",
              "py -m derail.experiments.run_hybrid_study",
              lambda: _horizon_advantage(4, 8), "Monitor"),
        Claim("horizon.long", "ESN advantage at post-onset horizon >= 9 steps",
              0.4042, "results/tables/hybrid_diagnosis.csv",
              "py -m derail.experiments.run_hybrid_study",
              lambda: _horizon_advantage(9, 10**6), "Monitor"),

        Claim("grounding.content_gain",
              "Content-gate detection gain on the content classes, worst seed",
              0.3067, "results/tables/grounding_multiseed_criterion.csv",
              "py -m derail.experiments.run_grounding_multiseed",
              lambda: _criterion("hybrid_content_gate", "content", "delta"), "Monitor"),
        Claim("grounding.no_degradation",
              "Content gate does not degrade behavioural detection, worst seed",
              0.0392, "results/tables/grounding_multiseed_criterion.csv",
              "py -m derail.experiments.run_grounding_multiseed",
              lambda: _criterion("hybrid_content_gate", "behavioral", "delta"), "Monitor"),
        Claim("transfer.within_family",
              "Best within-family transfer AUROC (qwen2.5:7b -> 3b), uncalibrated",
              0.4876, "results/tables/model_transfer.csv",
              "py -m derail.experiments.run_model_transfer",
              lambda: _model_transfer_auroc("transfer"), "Monitor",
              denominator=lambda: _corpus_eval_n("real_research3b"),
              expected_denominator=53),

        Claim("judge.p_detect", "Measured gemini-2.5-flash judge detection rate",
              0.5476, "results/tables/judge_calibration_summary.json",
              "py -m derail.experiments.run_judge_calibration --replay --n-per-stratum 120",
              lambda: _judge("p_detect_measured"), "Monitor",
              denominator=lambda: _judge_n("n_positive"),
              expected_denominator=84, denominator_unit="positives"),
        Claim("judge.p_false", "Measured gemini-2.5-flash judge false-alarm rate",
              0.0519, "results/tables/judge_calibration_summary.json",
              "py -m derail.experiments.run_judge_calibration --replay --n-per-stratum 120",
              lambda: _judge("p_false_measured"), "Monitor",
              denominator=lambda: _judge_n("n_negative"),
              expected_denominator=77, denominator_unit="negatives"),

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
              lambda: _repair_rate("located"), "Repair",
              denominator=_repair_episodes,
              expected_denominator=55),
        Claim("repair.generic", "`generic` recovery rate", 0.3576,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("generic"), "Repair",
              denominator=_repair_episodes,
              expected_denominator=55),
        Claim("repair.specific", "`specific` recovery rate", 0.3636,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("specific"), "Repair",
              denominator=_repair_episodes,
              expected_denominator=55),
        Claim("repair.recompute", "`recompute` recovery rate (not significant)", 0.2788,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("recompute"), "Repair",
              denominator=_repair_episodes,
              expected_denominator=55),
        Claim("repair.adaptive", "`adaptive` recovery rate (not significant)", 0.2121,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("adaptive"), "Repair",
              denominator=_repair_episodes,
              expected_denominator=55),
        Claim("repair.resample", "`resample` control recovery rate", 0.1636,
              "results/tables/repair_policies.csv",
              "py -m derail.intervene.evaluate_repair_policies --from-csv",
              lambda: _repair_rate("resample"), "Repair",
              denominator=_repair_episodes,
              expected_denominator=55),
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
        # No offline regenerator: this table is the record of 25 live episodes
        # (5 injection classes x 5 task seeds, halting off) driven through the
        # demo by hand. The committed CSV *is* the evidence; re-running it needs
        # a served model and produces a fresh sample, not this one.
        # A live study: re-running samples the same experiment afresh rather
        # than reproducing these bytes, so the ledger checks the invariant the
        # claim rests on -- every alarm gets a repair attempt -- not the count.
        Claim("repair.every_alarm_attempted",
              "Every behavioural alarm is followed by a repair attempt",
              "all alarms attempted", "results/tables/alarm_repair.csv",
              "py -m derail.experiments.demo --alarm-repair-matrix (live)",
              _every_alarm_attempted, "Repair"),

        # ---------------------------------------------------- table claims
        # Cells of tables that appear in the papers and DESIGN.md. Registered
        # after two of these tables were found stale: prose reconstructs a
        # table by hand, so it drifts a cell at a time and no sentence-level
        # claim covers it. The grounding table sat at an n=602 run long after
        # the study grew to 874, and DESIGN.md's repair table disagreed with
        # its own CSV on every rung.

        # -- grounding detection table (main.tex, workshop.tex, paper.md)
        Claim("grounding.pooled_n", "Pooled injected episodes in the grounding table",
              874, GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_n(True) + _grounding_n(False), "Monitor"),
        Claim("grounding.content_n", "Content-class episodes in the grounding table",
              313, GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_n(True), "Monitor"),
        Claim("grounding.behavioural_n", "Behavioural-class episodes in the grounding table",
              561, GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_n(False), "Monitor"),
        Claim("grounding.ref_content",
              "Ungrounded parent detection on the content classes", 0.2716,
              GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_rate("det_hybrid_weighted50", True), "Monitor",
              denominator=lambda: _grounding_n(True), expected_denominator=313),
        Claim("grounding.gate_content",
              "Content-gate detection on the content classes", 0.5783,
              GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_rate("det_hybrid_content_gate", True), "Monitor",
              denominator=lambda: _grounding_n(True), expected_denominator=313),
        Claim("grounding.ref_behavioural",
              "Ungrounded parent detection on the behavioural classes", 0.7380,
              GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_rate("det_hybrid_weighted50", False), "Monitor",
              denominator=lambda: _grounding_n(False), expected_denominator=561),
        Claim("grounding.gate_behavioural",
              "Content-gate detection on the behavioural classes -- the gate "
              "must not cost behavioural detection", 0.7861,
              GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_rate("det_hybrid_content_gate", False), "Monitor",
              denominator=lambda: _grounding_n(False), expected_denominator=561),
        Claim("grounding.joint_budget_content",
              "Joint-budget fusion detection on the content classes", 0.4537,
              GROUNDING_SRC, GROUNDING_CMD,
              lambda: _grounding_rate("det_joint_budget", True), "Monitor",
              denominator=lambda: _grounding_n(True), expected_denominator=313),

        # -- checks-versus-monitor table (quoted in four documents)
        Claim("verify.checks_served", "Checks: failures caught at T=0.2 (totals only)",
              0.5965, VS_SRC, VS_CMD,
              lambda: _vs_monitor("T=0.2", "total_consistency"), "Verification",
              denominator=lambda: _vs_monitor_n("T=0.2"),
              expected_denominator=57, denominator_unit="failures"),
        Claim("verify.checks_served_cov", "Checks: failures caught at T=0.2 with coverage",
              0.9649, VS_SRC, VS_CMD,
              lambda: _vs_monitor("T=0.2", "with_coverage"), "Verification",
              denominator=lambda: _vs_monitor_n("T=0.2"),
              expected_denominator=57, denominator_unit="failures"),
        Claim("verify.monitor_served", "Monitor: failures caught at T=0.2",
              0.5439, VS_SRC, VS_CMD,
              lambda: _vs_monitor("T=0.2", "monitor_alarmed"), "Verification",
              denominator=lambda: _vs_monitor_n("T=0.2"),
              expected_denominator=57, denominator_unit="failures"),
        Claim("verify.monitor_fp_served",
              "Monitor false-alarm rate at T=0.2, against the checks' 0", 0.1746,
              VS_SRC, VS_CMD,
              lambda: _vs_monitor("T=0.2", "monitor_alarmed", healthy=True),
              "Verification",
              denominator=lambda: _vs_monitor_n("T=0.2", healthy=True),
              expected_denominator=63, denominator_unit="healthy episodes"),
        Claim("verify.checks_provoking", "Checks: failures caught at T=0.9 (totals only)",
              0.6463, VS_SRC, VS_CMD,
              lambda: _vs_monitor("T=0.9", "total_consistency"), "Verification",
              denominator=lambda: _vs_monitor_n("T=0.9"),
              expected_denominator=82, denominator_unit="failures"),
        Claim("verify.monitor_provoking", "Monitor: failures caught at T=0.9",
              0.4024, VS_SRC, VS_CMD,
              lambda: _vs_monitor("T=0.9", "monitor_alarmed"), "Verification",
              denominator=lambda: _vs_monitor_n("T=0.9"),
              expected_denominator=82, denominator_unit="failures"),

        # -- net task-success table (DESIGN.md, and the 52->73 headline)
        Claim("repair.net_baseline", "Net task success with no intervention", 0.525,
              "results/tables/repair_policies.csv", REPAIR_CMD,
              lambda: _repair_net(None), "Repair",
              denominator=lambda: int(len(_table("verification_cold.csv"))),
              expected_denominator=120),
        Claim("repair.net_located", "Net task success under `located`", 0.7333,
              "results/tables/repair_policies.csv", REPAIR_CMD,
              lambda: _repair_net("located"), "Repair",
              denominator=lambda: int(len(_table("verification_cold.csv"))),
              expected_denominator=120),
        Claim("repair.recovered_located", "Failures `located` recovers, mean of 3 repeats",
              25.0, "results/tables/repair_policies.csv", REPAIR_CMD,
              lambda: _repair_recovered("located"), "Repair"),
    ]


def _render(claims: list[Claim], ok: bool) -> str:
    lines = [
        "# Claim-to-evidence ledger",
        "",
        "Every headline number in the README, `DESIGN.md` and both papers, with",
        "the artifact it is read from and the command that regenerates that",
        "artifact. This file is generated -- edit the study, not the ledger:",
        "",
        "`n` is the denominator the value was computed over, recomputed and",
        "drift-checked like the value itself, and labelled with what it counts:",
        "these are not all episodes. A rate shown without one is a rate nobody",
        "can sanity-check, which is how an AUC computed on a held-out split of",
        "94 was published as being on a corpus of 187.",
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
                  "| claim | value | n | source artifact | regenerate with |",
                  "|---|---|---|---|---|"]
        for c in group:
            # `n` is the denominator the value was computed over, recomputed
            # and checked like the value itself. A rate shown without one is a
            # rate nobody can sanity-check.
            n = ("—" if c.expected_denominator is None
                 else f"`{c.expected_denominator}` {c.denominator_unit}")
            lines.append(f"| {c.claim} | `{c.render()}` | {n} | `{c.source}` | "
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
        "One row above has no offline regenerator, and says so in its command",
        "column: `alarm_repair.csv` records 25 live episodes driven through the",
        "demo with halting off. Re-running it needs a served model and yields a",
        "fresh sample rather than that one, so the committed CSV is itself the",
        "evidence. Every other row regenerates from committed code and data.",
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
