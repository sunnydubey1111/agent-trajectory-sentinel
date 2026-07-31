"""Episode acceptance and provenance for trace collection.

found three ways a corpus could lie about itself:

  * an episode was labelled with the class that was *requested*, even when the
    stochastic injector never fired;
  * an episode counted as healthy even when the task failed;
  * resuming a collection reused old trace bytes but rewrote the metadata from
    the *current* configuration, so a trace collected at tau=2 could end up
    labelled tau=5.

This module is the single gate every collector goes through:

  `accept_episode`  decides, from what actually happened, whether an episode
                    may be written and with which label;
  `Provenance`      pins the immutable identity of a run (collector, model,
                    seed, task, tool roster, injector, schema);
  `write_episode`   writes trace and manifest entry atomically with a
                    checksum;
  `reusable`        decides on resume whether stored bytes still match the
                    configuration - and refuses to relabel them if not.

`require_ollama_model` is the preflight for local collection: a collector that
clears its output directory before writing must know the model exists *first*,
or a missing model turns a re-collection into a deletion.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from derail.telemetry.events import SCHEMA_VERSION

#: Bumped when the acceptance rules change in a way that invalidates a corpus.
COLLECTOR_CONTRACT_VERSION = 2


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Provenance:
    """Immutable identity of one collected episode."""

    collector: str
    backend: str
    model: str
    temperature: float | None
    episode_seed: int | None
    task_name: str | None
    task_sha256: str
    tools: tuple[str, ...]
    tool_roster_sha256: str
    requested_class: str | None
    requested_tau: int | None
    injector_seed: int | None
    schema: int = SCHEMA_VERSION
    contract: int = COLLECTOR_CONTRACT_VERSION

    def fingerprint(self) -> str:
        """Hash of everything that must not change under a resume."""
        return _sha256_text(json.dumps(asdict(self), sort_keys=True,
                                       separators=(",", ":"), default=str))


class ModelUnavailable(RuntimeError):
    """A local model a collection needs is not pulled (or Ollama is down)."""


def require_ollama_model(model: str,
                         base_url: str = "http://localhost:11434",
                         timeout: float = 5.0) -> None:
    """Raise `ModelUnavailable` unless Ollama can serve `model` right now.

    Call this BEFORE any destructive step (clearing an output directory) and
    before spending hours on a run. This guard exists because `qwen2.5:3b` was
    once removed while it was still the default of three collectors: without
    the preflight, a re-collection deletes a frozen corpus and only then fails.
    """
    import httpx
    try:
        resp = httpx.post(f"{base_url}/api/show", json={"model": model},
                          timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - any transport failure is fatal
        raise ModelUnavailable(
            f"could not reach Ollama at {base_url} to check {model!r}: {exc}\n"
            "Start Ollama, or pass a model that is available.") from exc
    if resp.status_code != 200:
        raise ModelUnavailable(
            f"Ollama has no model {model!r} (HTTP {resp.status_code}).\n"
            f"Pull it first:  ollama pull {model}\n"
            "Or re-run with an explicit --model that is present "
            "(`ollama list`).")


class CorpusInUse(RuntimeError):
    """A collector was about to write into a corpus that already has episodes."""


def guard_output_dir(out_dir: Path, *, allow_existing: bool,
                     flag: str = "--allow-existing") -> None:
    """Refuse to collect into a populated corpus unless told to explicitly.

    Collected traces cost real money and real hours and are not regenerable:
    the same task run again is a different sample, not the same one. Yet the
    collectors default to the very directories the published corpora live in,
    so an exploratory invocation lands on top of them. That has happened twice
    -- a `--mock-llm` dry run overwrote 70 committed Gemini traces with
    scripted ones, and `expand_healthy` had no CLI at all, so even `--help`
    started a live collection into `traces/real/`.

    A collector must therefore say what it intends. Writing into an empty or
    new directory is always fine; writing into one that already holds a
    manifest with episodes needs `allow_existing` (surfaced as `flag`), which
    is what a genuine resume or re-collection passes.
    """
    manifest = Path(out_dir) / "manifest.json"
    if allow_existing or not manifest.exists():
        return
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:      # noqa: BLE001 - unreadable manifest is not a corpus
        return
    if not entries:
        return
    raise CorpusInUse(
        f"{out_dir} already holds {len(entries)} collected episode(s).\n"
        "Collected traces are not regenerable -- re-running produces a "
        "different sample, not the same one.\n"
        f"Pass --out-dir to collect somewhere new, or {flag} to write into "
        "this corpus on purpose.")


def registry_roster_sha256(registry) -> str:
    """Hash of the tool roster and its schemas (what the agent could call)."""
    from derail.harness.tools import tool_fingerprint
    payload = {name: {"schema": registry.schemas()[name],
                      "impl": tool_fingerprint(registry.get(name))}
               for name in sorted(registry.names())}
    return _sha256_text(json.dumps(payload, sort_keys=True,
                                   separators=(",", ":"), default=str))


def make_provenance(*, collector: str, backend: str, model: str,
                    temperature: float | None, episode_seed: int | None,
                    task_text: str, registry, task_name: str | None = None,
                    injector=None) -> Provenance:
    return Provenance(
        collector=collector, backend=backend, model=model,
        temperature=temperature, episode_seed=episode_seed,
        task_name=task_name, task_sha256=_sha256_text(task_text),
        tools=tuple(sorted(registry.names())),
        tool_roster_sha256=registry_roster_sha256(registry),
        requested_class=None if injector is None else injector.failure_class,
        requested_tau=None if injector is None else injector.tau,
        injector_seed=None if injector is None else injector.seed)


# --------------------------------------------------------------- acceptance
@dataclass
class Verdict:
    """Outcome of the acceptance gate for one episode."""

    accepted: bool
    reason: str
    label: str | None = None
    tau: int | None = None
    facts: dict = field(default_factory=dict)


def accept_episode(steps: list[dict], *, injector=None, min_steps: int = 4,
                   success: bool | None = None) -> Verdict:
    """Decide whether an episode may be written, and with which label.

    An injected episode is accepted only if the injector actually mutated a
    result and at least one step follows that mutation - otherwise there is
    nothing for a monitor to detect and the label would be a claim about
    intent, not about the trace.  A healthy episode is accepted only
    if its task verifier says it succeeded, when a verifier exists;
    an unsuccessful run is a failure of an unknown kind, not a negative.

    The reported onset is the step where the mutation FIRST landed, not the
    requested tau.
    """
    facts: dict = {"T": len(steps)}
    if len(steps) < min_steps:
        return Verdict(False, f"too short: T={len(steps)} < {min_steps}",
                       facts=facts)

    if injector is not None and injector.failure_class is not None:
        applied = int(getattr(injector, "applied_count", 0))
        first = getattr(injector, "first_applied_t", None)
        facts.update(applied_count=applied, first_applied_t=first,
                     requested_tau=injector.tau,
                     applied_tools=sorted(set(getattr(injector,
                                                      "applied_tools", []))))
        if applied == 0 or first is None:
            return Verdict(False, "injection never applied (no-op positive)",
                           facts=facts)
        if first >= len(steps) - 1:
            return Verdict(False,
                           f"mutation landed at step {first} with no following "
                           f"step (T={len(steps)})", facts=facts)
        return Verdict(True, "injection applied and observable",
                       label=injector.failure_class, tau=int(first),
                       facts=facts)

    facts["success"] = success
    if success is False:
        return Verdict(False, "task did not succeed: not a healthy episode",
                       facts=facts)
    return Verdict(True, "healthy run accepted", label=None, tau=None,
                   facts=facts)


# ------------------------------------------------------------------- writing
def trace_bytes(steps: list[dict]) -> bytes:
    return "\n".join(json.dumps(s, ensure_ascii=False) for s in steps).encode("utf-8")


def write_episode(corpus_dir: Path, episode_id: str, steps: list[dict],
                  provenance: Provenance, verdict: Verdict,
                  extra: dict | None = None) -> dict:
    """Write the trace atomically and return its manifest entry.

    Trace and metadata are written together and hashed, so a resumed run can
    tell whether stored bytes still belong to the current configuration
    instead of relabelling them.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / f"{episode_id}.jsonl"
    payload = trace_bytes(steps)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    entry = {
        "episode_id": episode_id,
        "file": path.name,
        "failure_class": verdict.label,
        "tau": verdict.tau,
        "T": len(steps),
        "has_logprobs": any(s.get("token_logprobs") for s in steps),
        "model": provenance.model,
        "success": verdict.facts.get("success"),
        "accepted_because": verdict.reason,
        "injection": {k: v for k, v in verdict.facts.items()
                      if k in ("applied_count", "first_applied_t",
                               "requested_tau", "applied_tools")},
        "provenance": asdict(provenance),
        "provenance_fingerprint": provenance.fingerprint(),
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    entry.update(extra or {})
    return entry


def write_manifest(corpus_dir: Path, manifest: list[dict]) -> None:
    path = corpus_dir / "manifest.json"
    tmp = path.with_name(f".manifest.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2), "utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def reusable(corpus_dir: Path, entry: dict | None,
             provenance: Provenance) -> tuple[bool, str]:
    """(may the stored episode be reused, why not).

    Reuse requires the file to exist, its checksum to match what was recorded,
    and the provenance fingerprint to equal the current configuration's.  A
    mismatch means re-collect: the one thing never allowed is keeping the bytes
    and rewriting the label.
    """
    if not entry:
        return False, "not in the manifest"
    path = corpus_dir / entry.get("file", "")
    if not path.exists():
        return False, "trace file is missing"
    if "trace_sha256" not in entry or "provenance_fingerprint" not in entry:
        return False, "recorded before provenance was tracked"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["trace_sha256"]:
        return False, "trace bytes changed since collection"
    if entry["provenance_fingerprint"] != provenance.fingerprint():
        return False, "configuration differs from the recorded provenance"
    return True, "unchanged"


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import tempfile

    from derail.harness.inject import ToolInjector, UnknownFailureClass
    from derail.harness.tools import SimpleTool, ToolRegistry, ToolResult

    reg = ToolRegistry([SimpleTool("search", "Search.", {"q": "query"},
                                   lambda q: f"results for {q}")])
    steps = [{"text": f"step {t}", "latency_s": 1.0} for t in range(6)]

    # --- healthy acceptance is gated on task success ---
    assert accept_episode(steps).accepted
    assert accept_episode(steps, success=True).accepted
    bad = accept_episode(steps, success=False)
    assert not bad.accepted and "did not succeed" in bad.reason
    assert not accept_episode(steps[:2]).accepted            # too short

    # --- an injection that never fired is refused ---
    inj = ToolInjector("rate_limit", tau=2, seed=1)
    v = accept_episode(steps, injector=inj)
    assert not v.accepted and "no-op" in v.reason, v

    # --- an injection that fired is accepted, and reports the REAL onset ---
    inj = ToolInjector("wrong_document", tau=2, seed=1)
    inj.t = 3
    inj.apply(ToolResult("web_search", {"q": "x"}, "clean result", False, 0.1))
    v = accept_episode(steps, injector=inj)
    assert v.accepted and v.tau == 3 and v.label == "wrong_document", v
    assert v.facts["requested_tau"] == 2, "requested tau must stay visible"

    # --- a class that cannot touch this tool does not count as applied ---
    inj = ToolInjector("sql_timeout", tau=0, seed=1)
    inj.t = 1
    unchanged = inj.apply(ToolResult("wikipedia_search", {}, "fine", False, 0.1))
    assert unchanged.content == "fine" and inj.applied_count == 0

    # --- unknown classes are refused outright, never silently no-op ---
    try:
        ToolInjector("not_a_class", tau=1)
        raise AssertionError("unknown class accepted")
    except UnknownFailureClass:
        pass

    # --- provenance + resume ---
    with tempfile.TemporaryDirectory() as d:
        corpus = Path(d)
        prov = make_provenance(collector="smoke", backend="none", model="m",
                               temperature=0.2, episode_seed=7,
                               task_text="do the thing", registry=reg)
        entry = write_episode(corpus, "ep-000", steps, prov,
                              accept_episode(steps, success=True))
        write_manifest(corpus, [entry])
        ok, why = reusable(corpus, entry, prov)
        assert ok, why

        other = make_provenance(collector="smoke", backend="none", model="m",
                                temperature=0.2, episode_seed=7,
                                task_text="a DIFFERENT task", registry=reg)
        ok, why = reusable(corpus, entry, other)
        assert not ok and "configuration differs" in why, why

        (corpus / "ep-000.jsonl").write_text("tampered", "utf-8")
        ok, why = reusable(corpus, entry, prov)
        assert not ok and "changed since collection" in why, why

    print("PASS collection.py smoke test | contract version",
          COLLECTOR_CONTRACT_VERSION)
