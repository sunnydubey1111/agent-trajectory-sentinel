"""LangGraph/AutoGen x real-tool episode collector.

Frozen protocol, not a general-purpose tool: seed base 52026, tau in {2,3},
two tasks (arxiv_paper_search, multi_city_weather) split by a fixed
deterministic parity rule, four injected classes (goal_drift, tool_cascade,
looping, context_corruption), qwen2.5:7b on both frameworks with matched
generation config. 100% live collection via `mode="record"` (never reads/
replays, confirmed against the actual Cassette.call implementation) inside a
PER-EPISODE cassette directory (so a repeated identical call across episodes
cannot overwrite an earlier episode's own recording), with occurrence-suffixed
keys for a repeat within one episode (`record_provenance=True` end to end).

Admission (injected only): promotion requires `applied_count > 0` and
`0 < first_applied_t < T - 1` (same rule as `accept_episode` elsewhere).
Non-landing attempts retry with the next deterministic seed/tau
(`_seed_for`/`_tau_for`, capped at `_LANDING_ATTEMPTS_MAX`), never based on
a monitor score, and are logged to `landing_failures.json`, not promoted.
Healthy admission is unchanged (outcome-independent, no landing to check).

This module runs one episode at a time (`collect_one_episode`); the full
48-episode run is orchestrated by `main`, deliberately NOT invoked by any
test in this repo -- this study's own collection is a separate,
explicitly-approved step, not something a pytest run should ever trigger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from derail.common import rng_for
from derail.harness.collection import (ModelUnavailable, Provenance, Verdict,
                                       _sha256_text, registry_roster_sha256,
                                       require_ollama_model, write_episode, write_manifest)
from derail.harness.frameworks import run_autogen_episode, run_langgraph_episode
from derail.harness.inject import ToolInjector, replay_against_trace
from derail.harness.real_tools import build_registry
from derail.harness.record_replay import Cassette
from derail.harness.tasks import REAL_TASKS

TRACES_ROOT = Path(__file__).resolve().parents[2] / "traces"
SEED_BASE = 52026
MODEL = "qwen2.5:7b"
MAX_STEPS = 12
GEN_OPTIONS = {"temperature": 0.2, "num_predict": 512}      # frozen protocol
INJECTED_CLASSES = ("goal_drift", "tool_cascade", "looping", "context_corruption")
N_HEALTHY = 12
N_PER_CLASS = 3
#: Fixed before any episode of the corrected recollection was run: how many
#: deterministic landing attempts a single injected plan slot gets before
#: this collector gives up on it. Not tuned per episode or per observed
#: result -- the same cap applies to every plan slot.
_LANDING_ATTEMPTS_MAX = 8

TASK1 = next(t for t in REAL_TASKS if t.name == "arxiv_paper_search")
TASK2 = next(t for t in REAL_TASKS if t.name == "multi_city_weather")

ADAPTERS = {"langgraph": run_langgraph_episode, "autogen": run_autogen_episode}


def _task_for(kind_index: int) -> "RealTask":            # noqa: F821
    """Frozen deterministic parity rule: even index -> task1, odd -> task2."""
    return TASK1 if kind_index % 2 == 0 else TASK2


def _plan(framework: str, shuffle_seed: int | None = None
          ) -> list[tuple[str, str | None, int]]:
    """[(kind, failure_class_or_None, index)] -- 12 healthy + 4x3 injected.

    `shuffle_seed` permutes the ORDER episodes are collected in, leaving every
    episode's id and seed untouched. Collecting all healthy episodes before any
    injected one aligns collection time with the label, so any drift in host
    load lands in `latency_s` as a healthy-vs-injected difference that no agent
    produced. Interleaving spreads that drift across both labels instead.
    """
    plan = [("healthy", None, i) for i in range(N_HEALTHY)]
    for fc in INJECTED_CLASSES:
        plan += [(fc, fc, i) for i in range(N_PER_CLASS)]
    if shuffle_seed is not None:
        rng = rng_for(shuffle_seed, "collect-order", framework)
        plan = [plan[i] for i in rng.permutation(len(plan))]
    return plan


def _fresh_staging_dir(out_dir: Path, episode_id: str) -> Path:
    """A new, never-before-used attempt directory for `episode_id`. A
    previous failed attempt's directory (if any) is left exactly as it was
    -- never reused, never cleared -- so a retried infrastructure failure
    cannot silently mix content with the next attempt."""
    staging_root = out_dir / "_cassettes" / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while (staging_root / f"{episode_id}__attempt{attempt}").exists():
        attempt += 1
    d = staging_root / f"{episode_id}__attempt{attempt}"
    d.mkdir(parents=True)
    return d


def _seed_for(index: int, landing_attempt: int) -> int:
    """Injector seed, a pure function of (index, landing_attempt)."""
    return SEED_BASE * 1000 + index * 100 + landing_attempt


def _tau_for(framework: str, episode_id: str, landing_attempt: int) -> int:
    tau_rng = rng_for(SEED_BASE, "e1-tau", framework, episode_id, landing_attempt)
    return 2 + int(tau_rng.integers(0, 2))


def collect_one_episode(framework: str, kind: str, failure_class: str | None,
                        index: int, *, out_dir: Path, model: str = MODEL,
                        landing_attempt: int = 0) -> dict:
    """Collect exactly one episode; returns its manifest entry, unpromoted.

    `landing_attempt` picks this attempt's deterministic seed/tau. Does not
    decide admission or promote -- the caller does both, after checking
    landing validity. Writes to a fresh per-attempt staging dir; refuses to
    overwrite an already-accepted `_cassettes/{episode_id}/`.
    """
    if framework not in ADAPTERS:
        raise ValueError(f"unknown framework {framework!r}")
    task = _task_for(index)
    episode_id = f"{framework}-real-{kind}-{index:03d}"

    final_dir = out_dir / "_cassettes" / episode_id
    if final_dir.exists():
        raise FileExistsError(
            f"{final_dir} already exists -- refusing to overwrite an "
            f"accepted recording. Delete it deliberately first if this "
            f"episode is meant to be re-collected.")

    registry = build_registry(tuple(task.tools))
    seed = _seed_for(index, landing_attempt)
    tau = None
    injector = None
    if failure_class is not None:
        tau = _tau_for(framework, episode_id, landing_attempt)
        injector = ToolInjector(failure_class, tau=tau, seed=seed)

    staging_dir = _fresh_staging_dir(out_dir, episode_id)
    cassette = Cassette(staging_dir, mode="record")

    run_episode = ADAPTERS[framework]
    kwargs = dict(model=model, max_steps=MAX_STEPS, cassette=cassette,
                 record_provenance=True, injector=injector)
    if framework == "autogen":
        kwargs["options"] = GEN_OPTIONS
    # No try/except here: an exception leaves the staging directory exactly
    # as it is, unpromoted, at its own distinct attempt path -- the next
    # attempt gets a new one, never overwriting this failed one.
    steps = run_episode(registry, task.prompt, **kwargs)

    manifest_entry = {
        "episode_id": episode_id, "framework": framework, "task": task.name,
        "kind": kind, "failure_class": failure_class,
        "requested_tau": tau, "seed": seed, "landing_attempt": landing_attempt,
        "model": model, "T": len(steps),
    }
    return {"manifest_entry": manifest_entry, "steps": steps,
           "cassette_summary": cassette.summary(),
           "staging_dir": staging_dir, "final_dir": final_dir}


def _promote(result: dict) -> None:
    """Rename an accepted attempt's staging dir into its final location.
    Call only after admission passes; a rejected attempt stays unpromoted."""
    staging_dir, final_dir = result["staging_dir"], result["final_dir"]
    if final_dir.exists():          # created concurrently since the earlier check
        raise FileExistsError(
            f"{final_dir} was created while this attempt was running -- "
            f"refusing to overwrite it. This episode's recording is at "
            f"{staging_dir}, unpromoted.")
    staging_dir.rename(final_dir)


def check_admission(steps: list[dict], failure_class: str | None,
                    tau: int | None, seed: int) -> tuple[bool, str, dict]:
    """(ok, reason, facts). Same rule as `accept_episode`: healthy is
    outcome-independent; injected needs `applied_count > 0` and
    `0 < first_applied_t < T - 1`. Never reads a monitor score."""
    T = len(steps)
    if failure_class is None:
        return True, "healthy, outcome-independent admission", {"T": T}
    injector = replay_against_trace(steps, failure_class, tau, seed)
    facts = {"T": T, "requested_tau": tau,
             "applied_count": injector.applied_count,
             "first_applied_t": injector.first_applied_t,
             "applied_tools": sorted(set(injector.applied_tools))}
    if injector.applied_count == 0:
        return False, "injection never applied (no-op positive)", facts
    if not (0 < injector.first_applied_t < T - 1):
        return False, (f"mutation landed at step {injector.first_applied_t} "
                       f"with no following step (T={T})"), facts
    return True, "injection applied and observable", facts


def _provenance_for(framework: str, kind: str, failure_class: str | None,
                    index: int, model: str, landing_attempt: int,
                    tau: int | None, seed: int) -> Provenance:
    """Provenance for the ACTUAL attempt that was promoted -- `tau`/`seed`
    are the caller's own already-known values for that attempt, not
    reconstructed here, so this can never drift from what really ran."""
    task = _task_for(index)
    registry = build_registry(tuple(task.tools))
    options_sha = (_sha256_text(json.dumps(GEN_OPTIONS, sort_keys=True))
                  if framework == "autogen" else "")
    return Provenance(
        collector="collect_framework_real_traces", backend=framework,
        model=model, temperature=GEN_OPTIONS["temperature"],
        episode_seed=seed, task_name=task.name,
        task_sha256=_sha256_text(task.prompt),
        tools=tuple(sorted(registry.names())),
        tool_roster_sha256=registry_roster_sha256(registry),
        requested_class=failure_class, requested_tau=tau,
        injector_seed=None if failure_class is None else seed,
        system_instruction_sha256=options_sha)


def collect_episode_with_retries(framework: str, kind: str,
                                 failure_class: str | None, index: int, *,
                                 out_dir: Path, model: str,
                                 max_attempts: int,
                                 landing_attempt: int = 0) -> dict:
    """Retry ONLY a raised exception (dropped connection, transient outage).
    A completed run is never retried here for its content -- that's the
    caller's separate `landing_attempt` decision, on admission, not this."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return collect_one_episode(framework, kind, failure_class, index,
                                       out_dir=out_dir, model=model,
                                       landing_attempt=landing_attempt)
        except Exception as exc:                             # noqa: BLE001
            last_exc = exc
            print(f"  [infra-retry {attempt}/{max_attempts}] "
                 f"{framework}-real-{kind}-{index:03d} "
                 f"(landing_attempt={landing_attempt}): "
                 f"{type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"{framework}-real-{kind}-{index:03d} (landing_attempt="
        f"{landing_attempt}) failed after {max_attempts} attempts, last "
        f"error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=tuple(ADAPTERS), required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed-base", type=int, default=None,
                    help="override SEED_BASE, for a disjoint collection "
                         "into a different --out-dir (default: unchanged)")
    ap.add_argument("--limit", type=int, default=None,
                    help="collect at most this many NEW episodes this "
                         "invocation, then stop (already-collected plan "
                         "entries are still skipped on resume)")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="infrastructure-failure retry cap per episode")
    ap.add_argument("--n-healthy", type=int, default=None,
                    help="override N_HEALTHY (default: unchanged)")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="override N_PER_CLASS (default: unchanged)")
    ap.add_argument("--shuffle-order", type=int, default=None,
                    help="interleave healthy and injected collection under "
                         "this seed, so host-load drift cannot align with the "
                         "label (episode ids and seeds are unchanged)")
    args = ap.parse_args(argv)

    if args.seed_base is not None:
        global SEED_BASE
        SEED_BASE = args.seed_base
    if args.n_healthy is not None:
        global N_HEALTHY
        N_HEALTHY = args.n_healthy
    if args.n_per_class is not None:
        global N_PER_CLASS
        N_PER_CLASS = args.n_per_class

    try:
        require_ollama_model(args.model)
    except ModelUnavailable as exc:
        raise SystemExit(f"[collect-real:{args.framework}] {exc}")

    out_dir = (Path(args.out_dir) if args.out_dir
              else TRACES_ROOT / f"{args.framework}7b_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan(args.framework, shuffle_seed=args.shuffle_order)
    print(f"[collect-real:{args.framework}] {len(plan)} episodes planned "
         f"({N_HEALTHY} healthy + {len(INJECTED_CLASSES)}x{N_PER_CLASS} injected)")

    manifest_path = out_dir / "manifest.json"
    manifest: list[dict] = (json.loads(manifest_path.read_text("utf-8"))
                            if manifest_path.exists() else [])
    done_ids = {e["episode_id"] for e in manifest}
    infra_failures_path = out_dir / "infra_failures.json"
    infra_failures: list[dict] = (
        json.loads(infra_failures_path.read_text("utf-8"))
        if infra_failures_path.exists() else [])
    landing_failures_path = out_dir / "landing_failures.json"
    landing_failures: list[dict] = (
        json.loads(landing_failures_path.read_text("utf-8"))
        if landing_failures_path.exists() else [])

    n_collected_this_run = 0
    for kind, fc, index in plan:
        episode_id = f"{args.framework}-real-{kind}-{index:03d}"
        if episode_id in done_ids:
            continue
        if args.limit is not None and n_collected_this_run >= args.limit:
            print(f"[collect-real:{args.framework}] --limit {args.limit} "
                 f"reached, stopping (resume to continue).")
            break

        promoted = False
        for landing_attempt in range(_LANDING_ATTEMPTS_MAX):
            print(f"[collect-real:{args.framework}] running {episode_id} "
                 f"(task={_task_for(index).name}, "
                 f"landing_attempt={landing_attempt})...")
            try:
                result = collect_episode_with_retries(
                    args.framework, kind, fc, index, out_dir=out_dir,
                    model=args.model, max_attempts=args.max_attempts,
                    landing_attempt=landing_attempt)
            except RuntimeError as exc:
                infra_failures.append({"episode_id": episode_id, "kind": kind,
                                       "failure_class": fc,
                                       "landing_attempt": landing_attempt,
                                       "error": str(exc)})
                infra_failures_path.write_text(
                    json.dumps(infra_failures, indent=2), "utf-8")
                print(f"  [infra-failure] {episode_id} "
                     f"(landing_attempt={landing_attempt}): giving up on "
                     f"this attempt, logged to {infra_failures_path.name}")
                continue

            steps = result["steps"]
            me = result["manifest_entry"]
            ok, reason, facts = check_admission(
                steps, fc, me["requested_tau"], me["seed"])
            if not ok:
                landing_failures.append({
                    "episode_id": episode_id, "kind": kind, "failure_class": fc,
                    "landing_attempt": landing_attempt, "reason": reason,
                    **facts})
                landing_failures_path.write_text(
                    json.dumps(landing_failures, indent=2), "utf-8")
                print(f"  [not-landed] {episode_id} "
                     f"(landing_attempt={landing_attempt}): {reason} -- "
                     f"trying the next deterministic attempt")
                continue

            # Admitted: the ACTUAL onset (facts["first_applied_t"] for an
            # injected episode) is the recorded tau, matching
            # `accept_episode`'s convention everywhere else in this project
            # -- not the requested one, which this attempt's `_tau_for` only
            # used to configure the injector.
            actual_tau = facts.get("first_applied_t") if fc else None
            provenance = _provenance_for(args.framework, kind, fc, index,
                                         args.model, landing_attempt,
                                         me["requested_tau"], me["seed"])
            verdict = Verdict(accepted=True, reason=reason, label=fc,
                              tau=actual_tau, facts=facts)
            _promote(result)
            entry = write_episode(
                out_dir, episode_id, steps, provenance, verdict,
                extra={"framework": args.framework, "task": me["task"],
                       "landing_attempt": landing_attempt,
                       "requested_tau": me["requested_tau"],
                       "cassette_summary": result["cassette_summary"]})
            manifest.append(entry)
            done_ids.add(episode_id)
            write_manifest(out_dir, manifest)
            n_collected_this_run += 1
            promoted = True
            print(f"  [ok] {episode_id}: T={me['T']}"
                 + (f" tau={actual_tau} ({fc})" if fc else "")
                 + (f" (landing_attempt={landing_attempt})"
                    if landing_attempt else ""))
            break

        if not promoted:
            print(f"  [GIVE UP] {episode_id}: no landing after "
                 f"{_LANDING_ATTEMPTS_MAX} deterministic attempts -- not in "
                 f"manifest.json; see {landing_failures_path.name}")

    print(f"[collect-real:{args.framework}] {len(manifest)}/{len(plan)} "
         f"episodes in manifest, {len(infra_failures)} infra-failure(s), "
         f"{len(landing_failures)} landing-failure(s) -> {out_dir}")


if __name__ == "__main__":
    main()
