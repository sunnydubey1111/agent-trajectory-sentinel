"""Score the LangGraph/AutoGen x real-tool corpora against the frozen
`esn_cusum_max[e,m]` monitor.

Reuses the project's existing evaluation primitives unchanged --
`evaluate_alarms`/`summarize`/`episode_auc` from `derail.evaluation.metrics`,
the same functions every other corpus in this project is scored with. No
recalibration: `theta_b5` comes from `results/framework_monitor_freeze.json`,
fit once before any episode of this study was scored (see
`framework_monitor_freeze.py`).

Every episode in `traces/{langgraph,autogen}7b_real/manifest.json` is
COLLECTED (collection there is outcome-independent -- this study's frozen
protocol -- so this script never filters by task success). SCORING is
narrower: an injected episode whose injector never actually applied
(`inject.replay_against_trace` finds `applied_count == 0`, recovered from
the trace's own recorded tool calls) or applied with no following step is
excluded, the same two conditions `harness.collection.accept_episode`
already enforces everywhere else in this project -- this collector just
never routes through it, by design, so it is checked here instead. That
exclusion is fixed structurally (which tool calls actually happened, when,
against the trace's own content) and is computed before any monitor ever
sees the episode, so it cannot depend on this or any other monitor's score.

Writes:
  results/tables/framework_real_tool_alarms.csv  -- one row per SCORED episode
  results/framework_real_tool_report.md          -- per-framework, pooled
                                                     and per-class
                                                     detection/false-alarm/AUC,
                                                     plus a diagnosis section
                                                     comparing against the
                                                     same monitor fit
                                                     in-population
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from derail.evaluation.metrics import episode_auc, evaluate_alarms, summarize
from derail.experiments.framework_monitor_freeze import load_frozen_monitor
from derail.harness.inject import replay_against_trace
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_CSV = REPO_ROOT / "results" / "tables" / "framework_real_tool_alarms.csv"
OUT_REPORT = REPO_ROOT / "results" / "framework_real_tool_report.md"
FRAMEWORKS = ("langgraph7b_real", "autogen7b_real")


def _load_episodes(corpus_dir: Path) -> tuple[list, list[dict]]:
    """(episodes, excluded).

    The collector (`collect_framework_real_traces.py`) enforces admission
    at collection time via its own `check_admission`, which requires
    `applied_count > 0` and `0 < first_applied_t < T - 1` before an episode
    is ever promoted into `manifest.json` -- so every entry here is already
    admission-valid by construction. This function still recomputes that
    same check independently, as a defense-in-depth audit (no dependence on
    the collector's self-report, no dependence on any monitor score): it
    replays `replay_against_trace` with the episode's ORIGINAL REQUESTED
    onset (`entry["requested_tau"]`, what the injector was actually
    configured with during collection) and the same seed, to recover
    `applied_count`/`first_applied_t` from the trace's own recorded tool
    calls. Replaying with anything other than the original requested tau
    desyncs a ramped class's (`tool_cascade`) RNG draw sequence from what
    actually happened live and silently miscomputes `applied_count` --
    `entry["tau"]` (the collector's own recorded ACTUAL/landed onset) is
    NOT interchangeable with `entry["requested_tau"]` here.

    An episode is excluded from scoring when either:
      - the injector never actually fired (`applied_count == 0`) -- there is
        no fault for a monitor to have missed, so scoring it as a "miss"
        would be a false negative that never happened; or
      - it fired but with no following step (`Episode.__post_init__`'s
        `0 < tau < T`, the same condition `accept_episode` already refuses
        elsewhere as "mutation landed with no following step").
    This is an audit safeguard, not this script's primary admission
    mechanism -- the collector's own admission check is that. Episodes that
    DO score use the actual onset
    (`entry["tau"]`) as ground truth, matching `accept_episode`'s convention
    everywhere else in this project.
    """
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    episodes, excluded = [], []
    for entry in manifest:
        fc = entry["failure_class"]
        steps = [json.loads(l) for l in
                (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                if l.strip()]
        tau = None
        if fc is not None:
            requested_tau = entry["requested_tau"]
            seed = entry["provenance"]["episode_seed"]
            injector = replay_against_trace(steps, fc, requested_tau, seed)
            T = entry["T"]
            if injector.applied_count == 0:
                excluded.append({"episode_id": entry["episode_id"], "T": T,
                                 "requested_tau": requested_tau,
                                 "reason": "injection never applied "
                                          "(applied_count=0): no fault "
                                          "exists for the monitor to have "
                                          "missed"})
                continue
            actual_tau = injector.first_applied_t
            if not (0 < actual_tau < T - 1):
                excluded.append({"episode_id": entry["episode_id"], "T": T,
                                 "requested_tau": requested_tau,
                                 "actual_tau": actual_tau,
                                 "reason": f"mutation landed at step "
                                          f"{actual_tau} with no following "
                                          f"step (T={T})"})
                continue
            tau = actual_tau
        ep = episode_from_trace(steps, entry["episode_id"],
                                tau=tau, failure_class=fc,
                                use_sentence_transformers=False, extended=True)
        episodes.append(ep)
    return episodes, excluded


def _score(monitor, episodes: list) -> dict:
    scores = {}
    for ep in episodes:
        monitor.start_episode()
        scores[ep.episode_id] = [monitor.score_step(ep.X[t])
                                 for t in range(ep.X.shape[0])]
    return scores


def _fmt(m: dict) -> list[str]:
    lines = [
        f"- healthy_fa_rate: {m['healthy_fa_rate']:.3f}",
        f"- detection_rate: {m['detection_rate']:.3f}",
        f"- early_alarm_rate: {m['early_alarm_rate']:.3f}",
        f"- median_delay: {m['median_delay']}",
        f"- median_lead: {m['median_lead']}",
    ]
    if m["per_class"]:
        lines.append("- per_class:")
        for cls, v in m["per_class"].items():
            lines.append(f"  - {cls}: detection_rate={v['detection_rate']:.3f}")
    return lines


def _in_population_fa(dataset: str) -> float | None:
    """`esn_cusum_max`'s healthy false-alarm rate when fit and thresholded
    on its OWN population -- the same per-corpus refit-and-score procedure
    `run_hybrid_study.py` uses for every corpus, here read from its already-
    committed output. `None` if that table or row is unavailable (this
    diagnosis is a cross-reference, not something this script recomputes).
    """
    path = REPO_ROOT / "results" / "tables" / "hybrid_benchmark.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    row = d[(d.dataset == dataset) & (d.monitor == "esn_cusum_max")]
    return float(row["healthy_fa_rate"].iloc[0]) if len(row) else None


def _diagnosis_section() -> list[str]:
    """Root-cause note for the false-alarm rates above: BUG or GENUINE
    DISTRIBUTION SHIFT, per the frozen-monitor scoring-path audit this
    module's numbers were checked against. Not itself a live recomputation
    (the audit is a one-time investigation, not a per-run check), but the
    in-population contrast it cites IS read live from `hybrid_benchmark.csv`
    so it cannot silently drift out of date with that table.
    """
    lg_own = _in_population_fa("langgraph7b")
    ag_own = _in_population_fa("autogen7b")
    lg_own_s = f"{lg_own:.3f}" if lg_own is not None else "n/a"
    ag_own_s = f"{ag_own:.3f}" if ag_own is not None else "n/a"
    lines = ["## Diagnosis: why LangGraph false-alarms at 100%",
            "",
            "Both harnesses measure `latency_s` identically -- LLM-call time "
            "only, tool execution excluded -- structurally verified and "
            "covered by a live cross-harness contract test "
            "(`tests/test_telemetry_contracts.py`, against real Ollama).",
            "",
            "Verified against the frozen monitor's own per-channel CUSUM "
            "streams (`ChannelMaxESNMonitor.subs`, channels e and m scored "
            "separately): on every LangGraph and AutoGen healthy episode "
            "checked, the **meta channel (m)** -- action one-hot, "
            "log-latency, log-token-count, error flag -- crosses theta_b5 "
            "first and by a wide margin over the semantic channel (e). The "
            "dominant driver within m is log-latency: per-step LLM-call "
            "latency in this corpus averages **0.53s** (LangGraph) / "
            "**0.47s** (AutoGen) (excluding each episode's final synthesis "
            "step), against the frozen monitor's native-harness training "
            "mean of **4.22s** -- roughly an 8x gap, with per-step z-scores "
            "of -3 to -4 against the training distribution. The final "
            "synthesis step separately spikes log-token-count (z up to +4) "
            "on both frameworks, a real content difference (a longer final "
            "answer than the training distribution's typical step). Every "
            "other part of the scoring path (channel/feature indices, "
            "standardization against the frozen training statistics, "
            "per-episode CUSUM reset, threshold semantics, u-channel "
            "exclusion for the no-logprobs case) was checked against source "
            "and found correct.",
            "",
            "This is calibration/generalization failure, not architecture "
            "failure or a scoring bug: LangGraph/AutoGen's underlying Ollama "
            "client issues and consumes each call differently enough that "
            "its LLM-call latency runs on a different scale than this "
            "project's native `OllamaBackend` harness, even though both "
            "measure the identical quantity. Root-causing exactly why the "
            "call itself is faster is future/secondary work, out of scope "
            "here. Confirming evidence, not merely consistent with it: "
            "scored IN-POPULATION -- fit and thresholded on the corpus's "
            "own mock-tool LangGraph/AutoGen traces (`traces/langgraph7b`, "
            "`traces/autogen7b`), the same per-corpus refit procedure "
            "`run_hybrid_study.py` uses -- `esn_cusum_max`'s healthy "
            f"false-alarm rate is **{lg_own_s}** for LangGraph and "
            f"**{ag_own_s}** for AutoGen (`results/tables/hybrid_benchmark"
            ".csv`), against the zero-shot rates above. The architecture "
            "instruments both frameworks fine; a threshold frozen on this "
            "project's own native harness does not transfer to a different "
            "orchestrator's step/latency shape. Framework COMPATIBILITY "
            "(\"ATS can instrument LangGraph/AutoGen\") and calibration "
            "GENERALIZATION (\"one frozen threshold transfers across "
            "orchestrators\") are separate claims; only the first holds "
            "here. Per-framework recalibration is future/secondary work, "
            "not a substitute for this frozen result.",
            ""]
    return lines


def main() -> None:
    monitor, theta_b5, freeze_artifact = load_frozen_monitor()
    print(f"[framework-real-tool] frozen monitor loaded: theta_b5={theta_b5}")

    all_dfs = []
    per_framework = {}
    all_episodes = []
    all_excluded: list[dict] = []
    pooled_scores: dict = {}
    for fw_dir in FRAMEWORKS:
        corpus_dir = TRACES_ROOT / fw_dir
        if not (corpus_dir / "manifest.json").exists():
            raise SystemExit(f"[framework-real-tool] {corpus_dir}/manifest.json "
                             f"missing -- run the collector first")
        episodes, excluded = _load_episodes(corpus_dir)
        for e in excluded:
            e["dataset"] = fw_dir
        all_excluded.extend(excluded)
        if excluded:
            print(f"[framework-real-tool] {fw_dir}: {len(excluded)} episode(s) "
                 f"excluded from scoring: "
                 f"{[(e['episode_id'], e['reason']) for e in excluded]}")
        scores = _score(monitor, episodes)
        df = evaluate_alarms(episodes, scores, theta_b5)
        df.insert(0, "framework", fw_dir.replace("7b_real", ""))
        df.insert(1, "dataset", fw_dir)
        all_dfs.append(df)
        all_episodes.extend(episodes)
        pooled_scores.update(scores)
        summ = summarize(df)
        auc = episode_auc(episodes, scores)
        per_framework[fw_dir] = (summ, auc, len(episodes))
        print(f"[framework-real-tool] {fw_dir}: n={len(episodes)} "
             f"det={summ['detection_rate']:.3f} fa={summ['healthy_fa_rate']:.3f} "
             f"auc={auc:.3f}")

    combined = pd.concat(all_dfs, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"[framework-real-tool] wrote {OUT_CSV}")

    pooled_summ = summarize(combined)
    pooled_auc = episode_auc(all_episodes, pooled_scores)

    lines = ["# Framework x real-tool validation: LangGraph and AutoGen",
             "",
             f"Frozen monitor: `esn_cusum_max[e,m]`, theta_b5={theta_b5} "
             f"(`results/framework_monitor_freeze.json`). Tasks: "
             f"arxiv_paper_search, multi_city_weather. Model: qwen2.5:7b. "
             f"Seed base: 52026.",
             ""]
    if all_excluded:
        lines.append(f"**{len(all_excluded)} collected episode(s) excluded from "
                     f"monitor scoring** -- the same two conditions "
                     f"`accept_episode` already refuses elsewhere in this "
                     f"project (\"injection never applied (no-op positive)\" "
                     f"and \"mutation landed with no following step\"), "
                     f"recovered here from each trace's own recorded tool "
                     f"calls (`inject.replay_against_trace`) since this "
                     f"collector's outcome-independent admission never routed "
                     f"through `accept_episode` at collection time. These "
                     f"episodes were still collected and are counted in the "
                     f"corpus, just not scorable -- no fault ever existed for "
                     f"the no-op ones, so scoring them as a miss would count a "
                     f"false negative that never happened:")
        for e in all_excluded:
            detail = (f"actual onset {e['actual_tau']}" if "actual_tau" in e
                      else "never applied")
            lines.append(f"- `{e['dataset']}/{e['episode_id']}`: T={e['T']}, "
                         f"requested tau={e['requested_tau']} ({detail}) -- "
                         f"{e['reason']}")
        lines.append("")
    lines.append(
        "The two frameworks disagree sharply (per-framework `healthy_fa_rate` "
        "below): this monitor was fit once, only on episodes collected "
        "through this project's own custom `OllamaBackend` harness (see "
        "`framework_monitor_freeze.py`), and never retrained per framework, "
        "by design (no recalibration after this study's episodes exist). A "
        "monitor's detection/false-alarm rate does not automatically "
        "zero-shot-transfer to a different agent orchestrator's step/"
        "latency/tool-call shape -- that is the population difference this "
        "study measures, not a defect in either corpus. The pooled row "
        "below states this population explicitly; treat it as descriptive, "
        "not as a single operating point, since it averages two frameworks "
        "whose rates differ by construction.")
    lines.append("")
    lg_summ = per_framework["langgraph7b_real"][0]
    ag_summ = per_framework["autogen7b_real"][0]
    lines.append(
        f"**LangGraph's `detection_rate={lg_summ['detection_rate']:.3f}` is "
        f"not a meaningful detection result on its own** -- it sits beside "
        f"`healthy_fa_rate={lg_summ['healthy_fa_rate']:.3f}`: the monitor "
        f"alarms on every LangGraph episode collected, healthy or injected, "
        f"past its washout, so it does not distinguish derailment from this "
        f"framework's own step/latency shape at all. AutoGen's "
        f"`detection_rate={ag_summ['detection_rate']:.3f}` at "
        f"`healthy_fa_rate={ag_summ['healthy_fa_rate']:.3f}` false alarms is "
        f"the more informative of the two per-framework results here, though "
        f"a false-alarm rate that high is still not a usable operating "
        f"point.")
    lines.append("")
    for fw_dir in FRAMEWORKS:
        summ, auc, n = per_framework[fw_dir]
        lines.append(f"## {fw_dir} (n={n})")
        lines.append(f"- episode_auc (offline ranking): {auc:.3f}")
        lines += _fmt(summ)
        lines.append("")
    lines.append(f"## Pooled (both frameworks, n={len(all_episodes)})")
    lines.append(f"- episode_auc (offline ranking): {pooled_auc:.3f}")
    lines += _fmt(pooled_summ)
    lines.append("")
    lines += _diagnosis_section()
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(f"[framework-real-tool] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
