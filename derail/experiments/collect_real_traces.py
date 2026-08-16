"""Collect REAL agent traces using real tools and tasks under Gemini.

Usage:
  py -m derail.experiments.collect_real_traces --yes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from derail.common import rng_for, stable_hash
from derail.config import get_api_key
from derail.harness.collection import (CorpusInUse, guard_output_dir,accept_episode, make_provenance,
                                       reusable, write_episode,
                                       write_manifest)
from derail.harness.real_tools import build_registry, _ensure_tls
from derail.harness.agent_loop import run_real_episode
from derail.harness.inject import ToolInjector
from derail.harness.tasks import REAL_TASKS
from derail.harness.record_replay import Cassette, CostMeter
from derail.experiments.collect_traces import GeminiBackend

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces" / "real"
MODEL_DEFAULT = "gemini-2.5-flash"
MAX_STEPS = 12

#: Study-level grouping kept ALONGSIDE the original class, never
#: replacing it.
STUDY_CLASS_OF = {
    "looping": "looping",
    "rate_limit": "tool_cascade",
    "malformed_json": "tool_cascade",
    "timeout": "tool_cascade",
    "sql_timeout": "tool_cascade",
    "mcp_unavailable": "tool_cascade",
    "browser_fail": "tool_cascade",
    "tool_cascade": "tool_cascade",
    "wrong_document": "grounding_loss",
    "context_corruption": "context_corruption",
    "goal_drift": "goal_drift",
}


def main(argv: list[str] | None = None) -> None:
    global TRACES_DIR
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_real_traces")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--yes", action="store_true",
                        help="confirm real API spend")
    parser.add_argument("--budget", type=float, default=0.50,
                        help="hard USD cap; the meter raises before exceeding it")
    parser.add_argument("--out-dir", default=None,
                        help=f"corpus directory (default: {TRACES_DIR})")
    parser.add_argument("--allow-existing", action="store_true",
                        help="collect into a corpus that already holds episodes")
    args = parser.parse_args(argv)

    if args.out_dir:
        TRACES_DIR = Path(args.out_dir)
    try:
        guard_output_dir(TRACES_DIR, allow_existing=args.allow_existing)
    except CorpusInUse as exc:
        raise SystemExit(f"[collect_real] {exc}")

    if not args.yes:
        print("[collect_real] Refusing to run without --yes to confirm Gemini API spend.")
        return

    _ensure_tls()
    get_api_key("GEMINI_API_KEY", required=True)

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    manifest: list[dict] = []
    backend_tokens = [0, 0]
    # Hard cap agreed with the operator before the run; CostMeter raises
    # BEFORE a call that would exceed it.
    meter = CostMeter(budget_usd=args.budget)

    # For validation, we run the 10 real tasks:
    # 1. Healthy runs for all 10 tasks
    # 2. Injected runs for a subset of tasks/failures
    plan = []
    # Healthy runs
    for t_idx, task in enumerate(REAL_TASKS):
        plan.append(("healthy", None, t_idx))

    # Injected runs (e.g. inject looping, rate_limit, wrong_document, malformed_json)
    selected_failures = ["looping", "rate_limit", "wrong_document", "malformed_json"]
    for f_idx, fc in enumerate(selected_failures):
        # Run on tasks 3, 4, 7, 9
        for task_idx in [3, 4, 7, 9]:
            plan.append((fc, fc, task_idx))

    print(f"[collect_real] Starting collection: {len(plan)} episodes total.")

    previous = {}
    manifest_path = TRACES_DIR / "manifest.json"
    if manifest_path.exists():
        previous = {e["episode_id"]: e
                    for e in json.loads(manifest_path.read_text("utf-8"))}
    rejected: list[dict] = []
    backend_cassette = Cassette("traces/real/_cassettes/_backend", mode="auto")

    for kind, fc, task_idx in plan:
        task_def = REAL_TASKS[task_idx]
        episode_id = f"real-{task_def.name}-{kind}-{task_idx:02d}"
        # Seed depends on the TASK only, so a healthy episode and its injected
        # counterpart sample the model identically and differ only by the
        # fault.
        seed = args.seed * 1000 + stable_hash("task", task_idx) % 100000
        rng = rng_for(args.seed, "inject", episode_id)
        tau = None if fc is None else int(rng.integers(2, 4))

        # Per-task capability allowlist: the agent is offered exactly the tools
        # this task's prompt needs and nothing else.
        registry = build_registry(task_def.tools, fs_root=repo_root)
        injector = (None if fc is None
                    else ToolInjector(fc, tau=tau, seed=seed))
        provenance = make_provenance(
            collector="collect_real_traces", backend="gemini",
            model=args.model, temperature=None, episode_seed=seed,
            task_text=task_def.prompt, registry=registry,
            task_name=task_def.name, injector=injector)

        ok, why = reusable(TRACES_DIR, previous.get(episode_id), provenance)
        if ok:
            manifest.append(previous[episode_id])
            print(f"  [resume] {episode_id}: unchanged")
            continue
        if previous.get(episode_id):
            print(f"  [recollect] {episode_id}: {why}")
        print(f"Running episode: {episode_id} ...")

        backend = GeminiBackend(args.model, cost_meter=meter,
                                tool_specs=registry.specs(),
                                tool_schemas=registry.schemas(),
                                seed=seed, cassette=backend_cassette)
        cassette = Cassette(f"traces/real/_cassettes/{episode_id}", mode="auto")

        try:
            steps = run_real_episode(
                backend, registry, task_def.prompt,
                max_steps=MAX_STEPS, cassette=cassette, injector=injector
            )
        except Exception as exc:
            print(f"  [error] {episode_id}: {type(exc).__name__}: {exc}")
            continue

        backend_tokens[0] += backend.input_tokens
        backend_tokens[1] += backend.output_tokens

        # The task's own success criterion decides whether a non-injected run
        # may count as healthy.
        final_text = steps[-1].get("text", "") if steps else ""
        success = task_def.verify(final_text, steps)
        verdict = accept_episode(steps, injector=injector,
                                 success=None if fc else success)
        if not verdict.accepted:
            rejected.append({"episode_id": episode_id, "requested_class": fc,
                             "reason": verdict.reason, "facts": verdict.facts})
            print(f"  [reject] {episode_id}: {verdict.reason}")
            continue

        # Keep the ORIGINAL class and record the study-level grouping
        # separately, instead of collapsing rate_limit and malformed_json into
        # tool_cascade and losing the distinction.
        entry = write_episode(
            TRACES_DIR, episode_id, steps, provenance, verdict,
            extra={"task_name": task_def.name,
                   "success": success,
                   "original_class": fc,
                   "study_class": STUDY_CLASS_OF.get(fc) if fc else None})
        manifest.append(entry)
        write_manifest(TRACES_DIR, manifest)
        print(f"  [ok] {episode_id}: T={len(steps)} success={success}"
              + (f" onset={verdict.tau} ({fc})" if fc else ""))

    write_manifest(TRACES_DIR, manifest)
    if rejected:
        (TRACES_DIR / "rejected.json").write_text(
            json.dumps(rejected, indent=2), "utf-8")
        print(f"[collect_real] {len(rejected)} episode(s) rejected -> "
              f"traces/real/rejected.json")

    print(f"[collect_real] Finished! Wrote {len(manifest)} traces to {TRACES_DIR}")
    print(f"[collect_real] Total tokens: {backend_tokens[0]:,} in / {backend_tokens[1]:,} out")


if __name__ == "__main__":
    main()
