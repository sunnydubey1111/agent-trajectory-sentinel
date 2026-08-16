"""Expand the healthy Gemini real-trace cohort by running additional seeds.

Runs healthy episodes of the 10 real tasks across seeds [812, 813, 814].

This is a LIVE, PAID collector: it calls the Gemini API and writes into a
committed corpus. Nothing spends without an explicit instruction to spend: it
refuses to run without `--yes` -- so no invocation, `--help` included, can
start a real collection by accident -- and refuses to write into a corpus that
already holds episodes without `--allow-existing`.

    py -m derail.experiments.expand_healthy --estimate
    py -m derail.experiments.expand_healthy --yes --out-dir traces/_scratch
    py -m derail.experiments.expand_healthy --yes --allow-existing   # the real thing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from derail.common import rng_for
from derail.config import get_api_key
from derail.common import stable_hash
from derail.harness.collection import (CorpusInUse, accept_episode,
                                       guard_output_dir, make_provenance,
                                       write_episode, write_manifest)
from derail.harness.real_tools import build_registry, _ensure_tls
from derail.harness.agent_loop import run_real_episode
from derail.harness.record_replay import Cassette, CostMeter
from derail.experiments.collect_real_traces import GeminiBackend

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces" / "real"
MAX_STEPS = 12


def expand_cohort(out_dir: Path | None = None, budget_usd: float = 0.50):
    global TRACES_DIR
    if out_dir is not None:
        TRACES_DIR = Path(out_dir)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_tls()
    get_api_key("GEMINI_API_KEY", required=True)

    from derail.harness.tasks import REAL_TASKS

    repo_root = Path(__file__).resolve().parents[2]
    meter = CostMeter(budget_usd=budget_usd)  # hard cap, set before the run

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
            # Not coerced with bool(): the verifier returns None when it could
            # not decide, and bool(None) would record that as a failed task.
            success = task_def.verify(final_text, steps)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="py -m derail.experiments.expand_healthy",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true",
                    help="confirm a live, paid Gemini collection")
    ap.add_argument("--estimate", action="store_true",
                    help="print what a run would cost and do nothing")
    ap.add_argument("--out-dir", default=None,
                    help=f"where to collect (default: {TRACES_DIR})")
    ap.add_argument("--allow-existing", action="store_true",
                    help="write into a corpus that already holds episodes")
    ap.add_argument("--budget-usd", type=float, default=0.50,
                    help="hard spend cap for the run (default: 0.50)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else TRACES_DIR
    if args.estimate:
        print(f"[expand] would collect 10 tasks x 3 seed cohorts into {out_dir}, "
              f"capped at ${args.budget_usd:.2f} of live Gemini calls")
        return 0
    if not args.yes:
        print("[expand] refusing to start a live, paid collection without --yes "
              "(try --estimate first)")
        return 1
    try:
        guard_output_dir(out_dir, allow_existing=args.allow_existing)
    except CorpusInUse as exc:
        raise SystemExit(f"[expand] {exc}")
    expand_cohort(out_dir=out_dir, budget_usd=args.budget_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
