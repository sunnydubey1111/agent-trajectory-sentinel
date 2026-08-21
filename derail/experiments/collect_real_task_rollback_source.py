"""New live real-tool source corpus for the rollback/retry recovery study.

The five pre-existing real-tool research corpora (`real_research7b` and
siblings) cannot objectively score recovery: their task is an open-ended
research prompt with no `RealTask.success_fn`, so nothing can tell a
recovered continuation from a still-wrong one. Rather than invent a
subjective verifier or a recovery heuristic, this collector builds a small,
separate corpus directly from `derail.harness.tasks.REAL_TASKS`, where an
objective `success_fn` already exists and is used unchanged.

Restricted to the two REAL_TASKS whose tools are genuine external services
already proven live end-to-end for this project (arxiv_search/
wikipedia_search, get_weather/wikipedia_search) -- both already exercised by
the framework x real-tool collector. Four failure classes whose
`APPLICABLE_TOOLS` covers every tool either task uses (`None` or explicitly
includes arxiv_search/wikipedia_search): tool_cascade, looping,
context_corruption, wrong_document.

Deterministic seeds: a flat, ordered plan (healthy first, then each class in
a fixed order) assigns seed = SEED_BASE*1000 + running_index, used for BOTH
the Ollama sampling seed and the injector's own RNG -- the same
seed-sharing convention `derail.harness.collect_real._make_backend_factory`
already documents ("a healthy episode and its injected counterfactual
sample the model identically").

An episode is written only if `accept_episode` says the evidence supports
its label (WS4/collection.py's existing gate, same as every other real-tool
corpus in this project) -- an injection that never landed is refused, never
silently kept. This is required so a later rollback/retry pass has a
genuinely-faulted trace to check-point from, not a healthy run mislabeled as
failed.

n=16 injected (2 tasks x 4 classes x 2/class), fixed in this file before any
episode was collected -- one above the 10-15 target range; kept in full
since it was fixed pre-outcome, not selected after seeing results.

Not pooled with the existing booking-domain repair study
(`repair_policies.csv` / `alarm_repair.csv`): different task domain,
different verifier, different corpus directory.

Usage:
  py -m derail.experiments.collect_real_task_rollback_source
  py -m derail.experiments.collect_real_task_rollback_source \
      --n-healthy-per-task 2 --n-per-class 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from derail.common import rng_for
from derail.harness.collection import (ModelUnavailable, accept_episode,
                                       make_provenance, require_ollama_model,
                                       reusable, write_episode, write_manifest)
from derail.harness.agent_loop import run_real_episode
from derail.harness.inject import ToolInjector
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.record_replay import Cassette
from derail.harness.tasks import REAL_TASKS

TRACES_ROOT = Path(__file__).resolve().parents[2] / "traces"
DEFAULT_OUT_DIR = TRACES_ROOT / "real_task_rollback"
SEED_BASE = 52026
MODEL = "qwen2.5:7b"
MAX_STEPS = 12
TEMPERATURE = 0.2

#: Both tasks' tools are covered by every class below (either
#: APPLICABLE_TOOLS is None, or it explicitly lists arxiv_search/
#: wikipedia_search) -- no class here is a silent no-op on either task.
TASK_NAMES = ("arxiv_paper_search", "multi_city_weather")
TASKS = [t for t in REAL_TASKS if t.name in TASK_NAMES]
assert len(TASKS) == 2, "expected exactly 2 matching REAL_TASKS"

CLASSES = ("tool_cascade", "looping", "context_corruption", "wrong_document")


def _plan(n_healthy_per_task: int, n_per_class: int
         ) -> list[tuple[str, str | None, int]]:
    """[(task_name, failure_class_or_None, index)], healthy first."""
    plan: list[tuple[str, str | None, int]] = []
    for task in TASKS:
        for k in range(n_healthy_per_task):
            plan.append((task.name, None, k))
    for fc in CLASSES:
        for task in TASKS:
            for k in range(n_per_class):
                plan.append((task.name, fc, k))
    return plan


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_real_task_rollback_source")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-healthy-per-task", type=int, default=2)
    ap.add_argument("--n-per-class", type=int, default=2)
    args = ap.parse_args(argv)

    try:
        require_ollama_model(args.model)
    except ModelUnavailable as exc:
        raise SystemExit(f"[collect-rollback-source] {exc}")

    _ensure_tls()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = _plan(args.n_healthy_per_task, args.n_per_class)
    print(f"[collect-rollback-source] {len(plan)} episodes planned "
         f"({args.n_healthy_per_task}/task healthy + "
         f"{args.n_per_class}/task/class x {len(CLASSES)} classes)")

    manifest_path = out_dir / "manifest.json"
    previous = ({e["episode_id"]: e
                for e in json.loads(manifest_path.read_text("utf-8"))}
               if manifest_path.exists() else {})
    manifest: list[dict] = list(previous.values())
    rejected: list[dict] = (
        json.loads((out_dir / "rejected.json").read_text("utf-8"))
        if (out_dir / "rejected.json").exists() else [])

    backend_cassette = Cassette(out_dir / "_cassettes" / "_backend", mode="auto")
    task_by_name = {t.name: t for t in TASKS}

    for running_index, (task_name, fc, k) in enumerate(plan):
        task = task_by_name[task_name]
        kind = fc or "healthy"
        episode_id = f"rollback-{task_name}-{kind}-{k:02d}"
        seed = SEED_BASE * 1000 + running_index

        registry = build_registry(task.tools)
        injector = None
        if fc is not None:
            tau_rng = rng_for(SEED_BASE, "e1b-tau", episode_id)
            tau = 2 + int(tau_rng.integers(0, 2))
            injector = ToolInjector(fc, tau=tau, seed=seed)

        provenance = make_provenance(
            collector="collect_real_task_rollback_source", backend="ollama",
            model=args.model, temperature=TEMPERATURE, episode_seed=seed,
            task_text=task.prompt, registry=registry, task_name=task.name,
            injector=injector)

        ok, why = reusable(out_dir, previous.get(episode_id), provenance)
        if ok:
            print(f"  [resume] {episode_id}: unchanged")
            continue
        if previous.get(episode_id):
            print(f"  [recollect] {episode_id}: {why}")
        print(f"[collect-rollback-source] running {episode_id} "
             f"(task={task_name})...")

        from derail.experiments.collect_traces import OllamaBackend
        backend = OllamaBackend(args.model, tool_specs=registry.specs(),
                                tool_schemas=registry.schemas(), seed=seed,
                                temperature=TEMPERATURE, cassette=backend_cassette)
        cassette = Cassette(out_dir / "_cassettes" / episode_id, mode="record")

        try:
            steps = run_real_episode(backend, registry, task.prompt,
                                     max_steps=MAX_STEPS, cassette=cassette,
                                     injector=injector, episode_id=episode_id)
        except Exception as exc:                              # noqa: BLE001
            print(f"  [error] {episode_id}: {type(exc).__name__}: {exc}")
            continue

        final_text = steps[-1].get("text", "") if steps else ""
        success = None if fc else task.verify(final_text, steps)
        verdict = accept_episode(steps, injector=injector, success=success)
        if not verdict.accepted:
            rejected.append({"episode_id": episode_id, "requested_class": fc,
                             "reason": verdict.reason, "facts": verdict.facts})
            (out_dir / "rejected.json").write_text(
                json.dumps(rejected, indent=2), "utf-8")
            print(f"  [reject] {episode_id}: {verdict.reason}")
            continue

        entry = write_episode(out_dir, episode_id, steps, provenance, verdict,
                              extra={"task_name": task.name, "success": success})
        manifest = [e for e in manifest if e["episode_id"] != episode_id]
        manifest.append(entry)
        write_manifest(out_dir, manifest)
        print(f"  [ok] {episode_id}: T={len(steps)} success={success}"
             + (f" tau={verdict.tau} ({fc})" if fc else ""))

    n_labeled = sum(1 for e in manifest if e["failure_class"])
    print(f"[collect-rollback-source] {len(manifest)} episodes ({n_labeled} "
         f"labeled/accepted-injected), {len(rejected)} rejected -> {out_dir}")


if __name__ == "__main__":
    main()
