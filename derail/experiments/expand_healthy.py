"""Script to expand the healthy Gemini real trace cohort by running additional seeds.

Runs healthy episodes of the 10 real tasks across seeds [812, 813, 814]
to increase the healthy cohort size to ~39 traces.

Run: py -m derail.experiments.expand_healthy
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from derail.common import rng_for
from derail.config import get_api_key
from derail.common import stable_hash
from derail.harness.collection import (accept_episode, make_provenance,
                                       write_episode, write_manifest)
from derail.harness.real_tools import build_registry, _ensure_tls
from derail.harness.agent_loop import run_real_episode
from derail.harness.record_replay import Cassette, CostMeter
from derail.experiments.collect_real_traces import GeminiBackend

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces" / "real"
MAX_STEPS = 12


def expand_cohort():
    _ensure_tls()
    get_api_key("GEMINI_API_KEY", required=True)

    from derail.harness.tasks import REAL_TASKS

    repo_root = Path(__file__).resolve().parents[2]
    meter = CostMeter(budget_usd=0.50)  # hard cap agreed before the run

    # Load manifest
    manifest_path = TRACES_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []

    rejected: list[dict] = []
    seeds = [812, 813, 814]

    print(f"Expanding healthy cohort with seeds {seeds} across 10 tasks...")

    for seed in seeds:
        for t_idx, task_def in enumerate(REAL_TASKS):
            episode_id = f"real-{task_def.name}-healthy-seed{seed}-{t_idx:02d}"

            # Check if already exists in manifest
            if any(entry["episode_id"] == episode_id for entry in manifest):
                print(f"  [skip] {episode_id} already exists in manifest")
                continue

            path = TRACES_DIR / f"{episode_id}.jsonl"
            print(f"Running episode: {episode_id} ...")

            # Per-task capability allowlist.
            registry = build_registry(task_def.tools, fs_root=repo_root)
            # A "seed replication" that does not seed anything is not a
            # replication: the request seed is derived from the cohort seed and
            # the task, and it is recorded in the provenance.
            episode_seed = seed * 1000 + stable_hash("task", t_idx) % 100000
            backend = GeminiBackend("gemini-2.5-flash", cost_meter=meter,
                                    tool_specs=registry.specs(),
                                    tool_schemas=registry.schemas(),
                                    seed=episode_seed)
            cassette = Cassette(f"traces/real/_cassettes/{episode_id}", mode="auto")
            provenance = make_provenance(
                collector="expand_healthy", backend="gemini",
                model="gemini-2.5-flash", temperature=None,
                episode_seed=episode_seed, task_text=task_def.prompt,
                registry=registry, task_name=task_def.name)

            try:
                steps = run_real_episode(
                    backend, registry, task_def.prompt,
                    max_steps=MAX_STEPS, cassette=cassette, injector=None
                )
            except Exception as exc:
                print(f"  [error] {episode_id}: {type(exc).__name__}: {exc}")
                continue

            final_text = steps[-1].get("text", "") if steps else ""
            success = bool(task_def.verify(final_text, steps))
            verdict = accept_episode(steps, success=success)
            if not verdict.accepted:
                # An unsuccessful run is not an additional healthy seed; it is
                # a failure of an unknown kind.
                rejected.append({"episode_id": episode_id,
                                 "reason": verdict.reason,
                                 "facts": verdict.facts})
                print(f"  [reject] {episode_id}: {verdict.reason}")
                continue

            entry = write_episode(TRACES_DIR, episode_id, steps, provenance,
                                  verdict,
                                  extra={"task_name": task_def.name,
                                         "seed_cohort": seed})
            manifest.append(entry)
            write_manifest(TRACES_DIR, manifest)
            print(f"  [saved] {episode_id}: T={len(steps)} success={success}")

    if rejected:
        (TRACES_DIR / "rejected_expand.json").write_text(
            json.dumps(rejected, indent=2), encoding="utf-8")
        print(f"{len(rejected)} episode(s) rejected -> traces/real/rejected_expand.json")
    print("Healthy cohort expansion complete!")


if __name__ == "__main__":
    expand_cohort()
