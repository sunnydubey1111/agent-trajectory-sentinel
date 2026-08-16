"""An episode is labelled by what happened, not by what was asked.

Covers the validity rules this module enforces.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from derail.harness.collection import (accept_episode, make_provenance,
                                       reusable, write_episode, write_manifest)
from derail.harness.inject import (APPLICABLE_TOOLS, FAILURE_CLASSES,
                                   ToolInjector, UnknownFailureClass)
from derail.harness.tools import SimpleTool, ToolRegistry, ToolResult


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry([SimpleTool("web_search", "Search.", {"q": "query"},
                                    lambda q: f"results for {q}")])


def _steps(n: int = 6) -> list[dict]:
    return [{"text": f"step {t}", "latency_s": 1.0} for t in range(n)]


# ---------------------------------------------------------------------
def test_a_no_op_injection_is_refused():
    injector = ToolInjector("rate_limit", tau=2, seed=1)   # never applied
    verdict = accept_episode(_steps(), injector=injector)
    assert not verdict.accepted and "no-op" in verdict.reason
    assert verdict.facts["applied_count"] == 0


def test_an_applied_injection_reports_the_real_onset():
    injector = ToolInjector("wrong_document", tau=2, seed=1)
    injector.t = 4                       # the ramp landed two steps late
    injector.apply(ToolResult("web_search", {"q": "x"}, "clean", False, 0.1))
    verdict = accept_episode(_steps(8), injector=injector)
    assert verdict.accepted
    assert verdict.tau == 4, "the requested tau was recorded instead of the real onset"
    assert verdict.facts["requested_tau"] == 2


def test_a_mutation_on_the_last_step_is_refused():
    injector = ToolInjector("looping", tau=2, seed=1)
    injector.t = 5
    injector.apply(ToolResult("web_search", {}, "clean", False, 0.1))
    verdict = accept_episode(_steps(6), injector=injector)
    assert not verdict.accepted and "no following" in verdict.reason


# ---------------------------------------------------------------------
def test_an_unsuccessful_run_is_not_a_healthy_episode():
    assert accept_episode(_steps(), success=True).accepted
    verdict = accept_episode(_steps(), success=False)
    assert not verdict.accepted and "did not succeed" in verdict.reason


def test_short_episodes_are_refused():
    assert not accept_episode(_steps(2)).accepted


# ---------------------------------------------------------------------
def test_unknown_failure_classes_are_rejected_not_silently_ignored():
    with pytest.raises(UnknownFailureClass):
        ToolInjector("not_a_class", tau=1)
    with pytest.raises(ValueError):
        ToolInjector("looping", tau=None)       # labelled without an onset
    with pytest.raises(ValueError):
        ToolInjector("looping", tau=-1)


def test_a_class_only_touches_tools_it_can_plausibly_break():
    injector = ToolInjector("sql_timeout", tau=0, seed=1)
    injector.t = 1
    untouched = injector.apply(ToolResult("wikipedia_search", {}, "fine",
                                          False, 0.1))
    assert untouched.content == "fine" and injector.applied_count == 0

    injector = ToolInjector("wrong_document", tau=0, seed=1)
    injector.t = 1
    untouched = injector.apply(ToolResult("python", {}, "42", False, 0.1))
    assert untouched.content == "42", "a decoy document replaced a REPL result"


def test_every_class_declares_its_applicable_tools():
    assert set(APPLICABLE_TOOLS) == set(FAILURE_CLASSES)


def test_a_transform_never_masks_a_genuine_error():
    injector = ToolInjector("context_corruption", tau=0, seed=1)
    injector.t = 1
    failed = ToolResult("web_search", {}, "Error: upstream exploded", True, 0.1)
    out = injector.apply(failed)
    assert out.is_error, "a real error was relabelled as a successful result"


# ---------------------------------------------------------------------
def test_resume_refuses_to_relabel_stale_bytes(tmp_path: Path, registry):
    steps = _steps()
    prov_a = make_provenance(collector="t", backend="none", model="m",
                             temperature=0.2, episode_seed=1,
                             task_text="task A", registry=registry)
    entry = write_episode(tmp_path, "ep-000", steps, prov_a,
                          accept_episode(steps, success=True))
    write_manifest(tmp_path, [entry])
    assert reusable(tmp_path, entry, prov_a)[0]

    prov_b = make_provenance(collector="t", backend="none", model="m",
                             temperature=0.2, episode_seed=1,
                             task_text="task B", registry=registry)
    ok, why = reusable(tmp_path, entry, prov_b)
    assert not ok and "configuration differs" in why

    (tmp_path / "ep-000.jsonl").write_text("tampered", "utf-8")
    ok, why = reusable(tmp_path, entry, prov_a)
    assert not ok and "changed since collection" in why


def test_provenance_covers_the_identity_of_a_run(registry):
    base = dict(collector="t", backend="none", model="m", temperature=0.2,
                episode_seed=1, task_text="task", registry=registry)
    prov = make_provenance(**base)
    assert prov.fingerprint() == make_provenance(**base).fingerprint()
    for changed in [dict(model="other"), dict(episode_seed=2),
                    dict(temperature=0.9), dict(task_text="different")]:
        assert make_provenance(**{**base, **changed}).fingerprint() != prov.fingerprint()

    injected = make_provenance(**base,
                               injector=ToolInjector("looping", tau=2, seed=5))
    assert injected.fingerprint() != prov.fingerprint()


def test_manifest_entry_carries_evidence_and_checksums(tmp_path: Path, registry):
    steps = _steps()
    prov = make_provenance(collector="t", backend="none", model="m",
                           temperature=0.2, episode_seed=1, task_text="task",
                           registry=registry)
    entry = write_episode(tmp_path, "ep-001", steps, prov,
                          accept_episode(steps, success=True))
    for key in ("provenance", "provenance_fingerprint", "trace_sha256",
                "accepted_because", "collected_at"):
        assert key in entry, key
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"


# ---------------------------------------------------------------------
def test_local_backend_is_seeded_and_recordable():
    from derail.experiments.collect_traces import OllamaBackend
    backend = OllamaBackend("qwen2.5:7b", seed=1234, temperature=0.3)
    assert backend.seed == 1234 and backend.temperature == 0.3
    assert hasattr(backend, "cassette")


def test_collect_dataset_pairs_healthy_and_injected_tasks():
    """Healthy k and injected k must run the same task."""
    import inspect

    from derail.harness import collect_real
    source = inspect.getsource(collect_real.collect_dataset)
    assert 'for k in range(n_healthy)' in source
    assert '_run_one(f"real-{fc}-{k:03d}", k, fc)' in source, (
        "injected episodes no longer reuse the healthy task index")


# -------------------------------------------------------- collection preflight
def test_collectors_do_not_default_to_a_removed_model():
    """qwen2.5:3b was `ollama rm`-ed; no collector may still default to it."""
    from derail.experiments import collect_framework_traces, collect_traces

    assert collect_framework_traces.MODEL_DEFAULT == "qwen2.5:7b"
    assert collect_traces.OLLAMA_MODEL_DEFAULT == "qwen2.5:7b"


def test_missing_local_model_raises_before_any_work(monkeypatch):
    """A 404 from /api/show is fatal and names the pull command."""
    import httpx

    from derail.harness import collection

    class _Resp:
        status_code = 404

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(collection.ModelUnavailable) as exc:
        collection.require_ollama_model("qwen2.5:3b")
    assert "ollama pull qwen2.5:3b" in str(exc.value)


def test_unreachable_ollama_is_reported_not_swallowed(monkeypatch):
    import httpx

    from derail.harness import collection

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(httpx, "post", _boom)
    with pytest.raises(collection.ModelUnavailable, match="could not reach"):
        collection.require_ollama_model("qwen2.5:7b")


def test_recollect_refuses_and_deletes_nothing_when_a_model_is_gone(monkeypatch,
                                                                   capsys):
    """The rmtree in recollect_frameworks must never run past a bad model."""
    from devtools import recollect_frameworks as rf

    def _unavailable(model, *a, **k):
        raise rf.ModelUnavailable(f"no model {model!r}")

    def _no_delete(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("rmtree ran despite an unavailable model")

    monkeypatch.setattr(rf, "require_ollama_model", _unavailable)
    monkeypatch.setattr(rf.shutil, "rmtree", _no_delete)
    monkeypatch.setattr(rf.subprocess, "run", _no_delete)

    assert rf.main(["--only", "langgraph"]) == 1
    assert "nothing was deleted" in capsys.readouterr().out


# ------------------------------------------------------- L8 judge calibration
def test_judge_samples_are_labelled_by_verified_onset():
    """A positive is a step at/after the onset the injection was OBSERVED at."""
    from derail.experiments.run_judge_calibration import build_samples

    samples = build_samples("ollama7b", 40, seed=811)
    assert samples, "no labelled samples built"
    for s in samples:
        if s.label == 1:
            assert s.onset is not None and s.t >= s.onset
            assert s.failure_class is not None
        else:
            assert s.onset is None or s.t < s.onset
    assert s.task and s.transcript, "judge prompt needs task and transcript"


def test_judge_sampling_is_seed_deterministic():
    from derail.experiments.run_judge_calibration import build_samples

    a = build_samples("ollama7b", 20, seed=811)
    b = build_samples("ollama7b", 20, seed=811)
    c = build_samples("ollama7b", 20, seed=812)
    assert [x.episode_id + str(x.t) for x in a] == [x.episode_id + str(x.t) for x in b]
    assert [x.episode_id + str(x.t) for x in a] != [x.episode_id + str(x.t) for x in c]


@pytest.mark.parametrize("text,expected", [
    ('{"derailed": true}', True),
    ('{"derailed": false}', False),
    ('```json\n{"derailed": true}\n```', True),
    ("I cannot answer that.", None),
])
def test_judge_verdict_parsing(text, expected):
    from derail.experiments.run_judge_calibration import parse_verdict

    assert parse_verdict(text) is expected


def test_wilson_interval_brackets_the_point_estimate():
    from derail.experiments.run_judge_calibration import wilson

    lo, hi = wilson(46, 84)          # the measured p_detect cell
    assert lo < 46 / 84 < hi
    assert wilson(0, 10)[0] == 0.0   # no negative lower bound at zero counts


# --------------------------------------------------- L5 paid-collection guards
def test_gemini_long_collector_refuses_to_spend_without_confirmation():
    from derail.experiments import collect_gemini_long as cgl

    assert cgl.main(["--healthy", "1"]) == 2      # no --yes -> refuses
    assert cgl.main(["--estimate"]) == 0          # estimate never calls out


def test_gemini_long_reuses_the_qwen_long_task_verbatim():
    """The provider must be the ONLY difference from real_research7b_long."""
    from derail.experiments import collect_gemini_long as cgl
    from derail.experiments import collect_research7b_long as crl

    assert cgl._long_task is crl._long_task
    assert cgl.CLASSES is crl.CLASSES
    assert cgl.MAX_STEPS == 24 and cgl.TAU == 2


def test_gemini_backend_thinking_budget_defaults_to_unchanged():
    """Existing corpora were collected without a thinking budget; keep it."""
    import inspect

    from derail.experiments.collect_traces import GeminiBackend

    sig = inspect.signature(GeminiBackend.__init__)
    assert sig.parameters["thinking_budget"].default is None


# ----------------------------------------- provocation must stay honest
def test_provocation_is_transient_not_a_permanent_withhold():
    """The retry MUST succeed: a permanent withhold makes the correct answer
    unreachable, which manufactures positives instead of provoking them."""
    import numpy as np

    from verification import organic_hallucination as oh

    monkey_rate = 1.0                      # always fail the first attempt
    old, oh.WITHHOLD_RATE = oh.WITHHOLD_RATE, monkey_rate
    try:
        seen: set = set()
        rng = np.random.default_rng(0)
        args = {"item": "item-1"}
        assert oh._transient_failure("search_catalog", args, seen, rng) is True
        # Same call again -> must be served, not failed again.
        assert oh._transient_failure("search_catalog", args, seen, rng) is False
    finally:
        oh.WITHHOLD_RATE = old


def test_provocation_is_off_by_default():
    """No committed corpus may be provoked unless explicitly asked for."""
    from verification import organic_hallucination as oh

    assert oh.WITHHOLD_RATE == 0.0 or "AGENTWATCH_ORGANIC_WITHHOLD" in __import__("os").environ
    assert oh.WITHHOLD_TOOLS == ("search_catalog", "lookup_flight",
                                 "lookup_hotel")


def test_verifier_scoring_separates_assertions_from_tool_results():
    """A tool result is not an assertion: scoring the raw step text would make
    every returned price look like a fabricated figure."""
    from verification.score_provoked_fabrication import verifier_flags

    steps = [
        # Model asserts $500 in prose; the tool only ever returned $299.
        {"text": 'The flight costs $500. [lookup_flight({"a": 1}) -> $299]'},
    ]
    flags = verifier_flags(steps)
    assert 500.0 in flags
    assert 299.0 not in flags, "a tool result was treated as an assertion"


# ------------------------------- L2b: the invalid llama organic arm stays out
def test_organic_llama_corpus_is_not_a_scoreable_dataset():
    """organic_llama8b is harness-confounded evidence, not organic evidence.
    Registering it in REAL_DATASETS would let a study score it by accident."""
    from derail.experiments.run_hybrid_study import REAL_DATASETS

    assert "organic_llama8b" not in REAL_DATASETS


def test_organic_model_override_defaults_to_the_demo_model():
    """No existing organic corpus may change because the override exists."""
    from derail.experiments.demo import MODEL
    from verification import organic_hallucination as oh

    import os
    if "AGENTWATCH_ORGANIC_MODEL" not in os.environ:
        assert oh.ORGANIC_MODEL == MODEL


def test_organic_temperature_override_defaults_to_the_organic_setting():
    """Every published organic table was collected at 0.9; the serving-arm
    override must not move it, and the collected temperature must reach the
    manifest so the two arms can never be confused after the fact."""
    import inspect
    import os

    from verification import organic_hallucination as oh

    if "AGENTWATCH_ORGANIC_TEMPERATURE" not in os.environ:
        assert oh.TEMPERATURE == 0.9
    src = inspect.getsource(oh._run_one)      # where the manifest entry is built
    assert '"temperature": TEMPERATURE' in src, \
        "the arm's temperature must be recorded per episode"
    assert '"seed": seed' in src, \
        "seed pairing across arms is only checkable if the seed is recorded"


def test_parallel_collection_is_off_by_default_and_always_recorded():
    """Concurrency inflates wall-clock latency, and two telemetry dims are
    built from it. A corpus must therefore never be parallel-collected by
    accident, and must always say which it was."""
    import inspect

    from verification import organic_hallucination as oh

    assert inspect.signature(oh.collect).parameters["n_parallel"].default == 1
    src = inspect.getsource(oh.collect)
    assert 'entry["n_parallel"] = n_parallel' in src, \
        "every episode must record the concurrency it was collected under"
    assert '"latency_dims_valid": n_parallel == 1' in src, \
        "the corpus must declare whether its latency dims mean anything"


def test_early_stopping_reads_labels_not_monitor_scores():
    """Stopping on a monitor score would tune collection to the result. The
    rule may only read the objective labeller, and must keep enough healthy
    episodes for a null to exist."""
    import inspect

    from verification import organic_hallucination as oh

    assert inspect.signature(oh.collect).parameters["min_failures"].default is None
    assert oh.MIN_HEALTHY_FOR_NULL == 15
    src = inspect.getsource(oh.collect)
    assert "label(steps, e[\"expected_total\"])" in src, \
        "the stop rule must use the objective labeller"
    for banned in ("score_step", "StreamingContentGate", "peak", "alarmed"):
        assert banned not in src, f"collection must not read {banned}"
    assert "healthy >= min_healthy" in src, \
        "stopping without enough healthy episodes leaves an unscoreable corpus"


def test_frozen_long_corpus_is_not_grown_in_place():
    """Published tables were computed from 72 episodes. Growth belongs in the
    _ext sibling; a corpus without v5 provenance cannot be verified on reuse,
    so extending it in place silently invalidates every table citing it."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "traces"
    frozen = json.loads((root / "real_research7b_long" / "manifest.json")
                        .read_text("utf-8"))
    ext = json.loads((root / "real_research7b_long_ext" / "manifest.json")
                     .read_text("utf-8"))
    assert len(frozen) == 72, "frozen long corpus changed size"
    assert len(ext) == 121
    # The frozen corpus predates v5; that is exactly why it must not be reused.
    assert "trace_sha256" not in frozen[0]
    assert "trace_sha256" in ext[0] and "provenance_fingerprint" in ext[0]


# ------------------------------------------------- L2b tool-call nudge (off)
def test_tool_nudge_is_off_by_default():
    """Enabling it changes agent-loop semantics for every future episode, so it
    must never turn on implicitly."""
    import os

    from verification import organic_hallucination as oh

    if "AGENTWATCH_TOOL_NUDGE" not in os.environ:
        assert oh.TOOL_NUDGE is False


def test_text_toolcall_detection():
    from verification.organic_hallucination import looks_like_text_toolcall

    assert looks_like_text_toolcall(
        'I will check. {"name": "hotel_price", "parameters": {"nights": 2}}'
    ) == "hotel_price"
    assert looks_like_text_toolcall("The total is $2,140.") is None


def test_nudge_names_only_real_tools():
    """The nudge must enumerate the real roster - telling a confabulating model
    the wrong tool names would make the problem worse."""
    from derail.experiments.demo import DEMO_TOOL_SPECS
    from verification.organic_hallucination import nudge_message

    msg = nudge_message("hotel_price", tuple(DEMO_TOOL_SPECS))
    for name in DEMO_TOOL_SPECS:
        assert name in msg
    assert "PER NIGHT" in msg


# ------------------------------------------------- L10: container repro path
def test_core_lock_covers_the_deterministic_gate():
    """`requirements-core.lock.txt` must install everything the gate reaches.

    That lockfile is the lean environment: enough to run the fast tests and the
    behaviour snapshot, and deliberately without the real-trace, framework and
    Gemini stack. Every MODULE-LEVEL import reachable from the gate must be in
    it. A collection dependency imported at module level instead of lazily
    would make the lean environment fail on import, and nobody would notice
    until someone installed from it.
    """
    import ast
    import re
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lock = (root / "requirements-core.lock.txt").read_text(encoding="utf-8")
    pinned = {re.split(r"[=<>!\[]", line.strip())[0].strip().lower()
              for line in lock.splitlines()
              if line.strip() and not line.startswith("#")}
    # Import name on the left, distribution name as the lockfile pins it.
    alias = {"sklearn": "scikit-learn", "yaml": "pyyaml", "google": "google-genai",
             "PIL": "pillow"}
    local = {"derail", "devtools", "tests", "experimental", "verification",
             "conftest"}

    offenders = {}
    for pkg in ("derail", "devtools", "tests"):
        for path in (root / pkg).rglob("*.py"):
            # utf-8-sig: at least one source file carries a BOM, and ast.parse
            # rejects U+FEFF - silently skipping those files would hollow out
            # this check.
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in tree.body:                      # module level only
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                        and node.module:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                for mod in names:
                    if mod in sys.stdlib_module_names or mod in local:
                        continue
                    if mod == "torch":       # supplied by REPRO_MODE=full
                        continue
                    if alias.get(mod, mod).lower() not in pinned:
                        offenders.setdefault(mod, path.name)
    assert not offenders, (
        "module-level imports missing from requirements-core.lock.txt "
        f"(the lean environment would fail on import): {offenders}")



# --------------------------------------------------- dry runs are not collections
def test_mock_dry_run_never_targets_a_committed_corpus():
    """`--mock-llm` writes scripted traces; they must not land on real data.

    The dry run shares the real collector's default output directory, so
    before the guard below existed a "safe offline dry run" silently
    overwrote 70 committed Gemini traces -- data that cost real money and is
    not regenerable. The redirect is asserted here rather than trusted.
    """
    import inspect

    from derail.experiments import collect_traces

    src = inspect.getsource(collect_traces.main)
    assert "args.mock_llm and not args.out_dir" in src, (
        "the --mock-llm output redirect is gone; a dry run can overwrite a "
        "committed corpus again")
    assert "_mock_dry_run" in src


def test_mock_dry_run_directory_is_gitignored():
    """Scratch traces must not be committable, or a dry run pollutes the corpus."""
    repo_root = Path(__file__).resolve().parents[1]
    ignore = (repo_root / ".gitignore").read_text("utf-8")
    assert "traces/_mock_dry_run/" in ignore


# ------------------------------------------- collectors never clobber a corpus
def test_guard_refuses_to_collect_over_an_existing_corpus(tmp_path: Path):
    from derail.harness.collection import CorpusInUse, guard_output_dir

    guard_output_dir(tmp_path, allow_existing=False)          # empty: fine
    (tmp_path / "manifest.json").write_text("[]", "utf-8")
    guard_output_dir(tmp_path, allow_existing=False)          # no episodes: fine

    (tmp_path / "manifest.json").write_text('[{"episode_id": "a"}]', "utf-8")
    with pytest.raises(CorpusInUse):
        guard_output_dir(tmp_path, allow_existing=False)
    guard_output_dir(tmp_path, allow_existing=True)           # explicit: fine


def test_guard_refuses_an_unreadable_manifest(tmp_path: Path):
    """An unreadable manifest is the case the guard exists for, not a pass.

    A directory holding a manifest.json demonstrably had a collection run into
    it. Failing to parse that file is the least safe moment to conclude there
    is nothing worth protecting — the guard's own docstring records that a dry
    run once overwrote 70 committed Gemini traces.
    """
    from derail.harness.collection import CorpusInUse, guard_output_dir

    (tmp_path / "manifest.json").write_text("{not json", "utf-8")
    with pytest.raises(CorpusInUse, match="cannot be read"):
        guard_output_dir(tmp_path, allow_existing=False)


def test_guard_still_yields_to_an_explicit_allow_existing(tmp_path: Path):
    """Refusing must stay overridable, or a resume cannot run at all."""
    from derail.harness.collection import guard_output_dir

    (tmp_path / "manifest.json").write_text("{not json", "utf-8")
    guard_output_dir(tmp_path, allow_existing=True)


@pytest.mark.parametrize("module", [
    "derail.experiments.collect_organic",
    "derail.experiments.collect_real_traces",
    "derail.experiments.expand_healthy",
])
def test_every_live_collector_wires_the_guard(module: str):
    """A collector that skips the guard can silently destroy paid-for data."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module).main)
    assert "guard_output_dir" in src, f"{module} does not guard its output dir"
    assert "--out-dir" in src, f"{module} offers no way to collect elsewhere"


def test_expand_healthy_cannot_collect_without_confirmation():
    """It had no CLI at all, so even --help started a paid live collection."""
    import inspect

    from derail.experiments import expand_healthy

    src = inspect.getsource(expand_healthy.main)
    assert "--yes" in src and "refusing" in src
    assert expand_healthy.main(["--estimate"]) == 0     # never collects
    assert expand_healthy.main([]) == 1                 # refuses without --yes
