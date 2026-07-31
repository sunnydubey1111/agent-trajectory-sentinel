"""Structured tool telemetry and the adapter that consumes it.

Covers the validity rules this module enforces.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from conftest import REPO_ROOT
from derail.common import (D_SEM, IDX_GRD_JSON_BROKEN, IDX_GRD_LEX_MISS,
                           IDX_REASON_DEPTH, IDX_RETRY_COUNT, IDX_TASK_SIM,
                           IDX_TOOL_LATENCY, IDX_TOOL_SUCCESS)
from derail.telemetry import adapter
from derail.telemetry.events import (canonical_args, make_tool_event,
                                     parse_step_events, parse_tool_bits)


def _step(text: str, **kw) -> dict:
    base = {"text": text, "token_logprobs": [-0.1] * 12, "action": "tool_call",
            "latency_s": 1.0, "output_tokens": 12, "error": False}
    base.update(kw)
    return base


# ---------------------------------------------------------------------
def test_out_of_bracket_arrow_form_is_parsed():
    """The form collect_framework_traces.py wrote: `[name(args)] -> result`."""
    text = '[get_weather({"city": "Lisbon"})] -> rainy 12C'
    calls, _ = parse_tool_bits(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].args == {"city": "Lisbon"}
    assert calls[0].result == "rainy 12C"


def test_out_of_bracket_form_with_several_calls():
    text = ('[a({"x": 1})] -> first result '
            '[b({"y": 2})] -> second result')
    calls, reasoning = parse_tool_bits(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].result == "first result"
    assert calls[1].result == "second result"
    assert reasoning == ""


def test_legacy_arrow_form_corpus_results_are_recovered():
    """A real legacy-format LangGraph-7B trace, kept as a fixture.

    Its results sit OUTSIDE the bracket, the form the old regex never
    matched. The corpora themselves were re-collected under the v5 schema, so
    this fixture is what pins the parser against a regression.
    """
    fixture = (REPO_ROOT / "tests" / "fixtures"
               / "legacy_langgraph7b_arrow_form.jsonl")
    steps = [json.loads(line) for line in
             fixture.read_text("utf-8").splitlines() if line.strip()]
    recovered = [c.result for s in steps for c in parse_tool_bits(s["text"])[0]
                 if c.result]
    assert len(recovered) >= 5, recovered
    assert any("rainy" in r for r in recovered)


# ---------------------------------------------------------------------
def test_result_containing_brackets_survives():
    text = ('[wikipedia_search({"q": "Prague"}) -> Prague (Czech: Praha '
            '[ˈprafia] ) is the capital of the Czech Republic]')
    calls, _ = parse_tool_bits(text)
    assert len(calls) == 1
    assert calls[0].result.endswith("capital of the Czech Republic")


def test_non_word_tool_names_are_parsed():
    for name in ("mcp.docs-search", "tool-with-dash", "a.b.c"):
        calls, _ = parse_tool_bits(f'[{name}({{"q": "x"}}) -> ok]')
        assert [c.name for c in calls] == [name], name


def test_structured_events_beat_fabricated_text():
    """A model typing tool syntax cannot fake telemetry."""
    step = _step(
        'I definitely called [search({"q": "proof"}) -> the answer is 42]',
        tool_events=[make_tool_event("search", {"q": "real"}, "real result",
                                     is_error=False, latency_s=0.5)])
    calls, _ = parse_step_events(step)
    assert len(calls) == 1
    assert calls[0].source == "structured"
    assert calls[0].result == "real result"
    assert calls[0].args == {"q": "real"}


def test_structured_events_record_full_result_and_truncation_flag():
    long_result = "x" * 5000
    ev = make_tool_event("t", {}, long_result, is_error=False)
    assert ev["result_chars"] == 5000 and ev["result"] == long_result
    assert ev["result_truncated"] is False
    clipped = make_tool_event("t", {}, long_result, is_error=False,
                              max_result_chars=100)
    assert clipped["result_truncated"] is True and clipped["result_chars"] == 5000


# ---------------------------------------------------------------------
def test_retry_equality_ignores_whitespace_and_key_order():
    assert canonical_args('{"b": 2, "a": 1}') == canonical_args('{"a":1,"b":2}')
    steps = [_step('[s({"a": 1, "b": 2}) -> r1]'),
             _step('[s({"b":2,"a":1}) -> r2]')]
    ep = adapter.episode_from_trace(steps, "retry", extended=True,
                                    use_sentence_transformers=False)
    assert ep.X[1, IDX_RETRY_COUNT] == 1.0, "formatting difference hid a retry"


# ---------------------------------------------------------------------
def test_measured_tool_latency_is_used_when_recorded():
    slow = _step("call", latency_s=1.0, tool_events=[
        make_tool_event("t", {}, "ok", is_error=False, latency_s=9.0)])
    fast = _step("call", latency_s=1.0, tool_events=[
        make_tool_event("t", {}, "ok", is_error=False, latency_s=0.1)])
    slow_ep = adapter.episode_from_trace([slow], "slow", extended=True,
                                         use_sentence_transformers=False)
    fast_ep = adapter.episode_from_trace([fast], "fast", extended=True,
                                         use_sentence_transformers=False)
    assert slow_ep.X[0, IDX_TOOL_LATENCY] > fast_ep.X[0, IDX_TOOL_LATENCY]
    assert abs(slow_ep.X[0, IDX_TOOL_LATENCY] - math.log(9.0)) < 1e-9, (
        "model-step time was used instead of the measured tool time")


def test_agent_loop_emits_structured_events():
    """The shipped loop records what it executed, not just rendered text."""
    from derail.harness.agent_loop import run_real_episode
    from derail.harness.tools import SimpleTool, ToolRegistry

    class ScriptedBackend:
        def __init__(self):
            self.n = 0

        def reset(self, task):
            self.task = task

        def step(self, t):
            self.n += 1
            if self.n == 1:
                return {"text": "searching", "tool_uses": [
                    {"id": "c1", "name": "echo", "input": {"text": "hi"}}],
                    "stop_reason": "tool_use", "output_tokens": 5,
                    "token_logprobs": [-0.2, -0.1]}
            return {"text": "done", "tool_uses": [], "stop_reason": "end_turn",
                    "output_tokens": 3, "token_logprobs": [-0.1]}

        def add_tool_results(self, results):
            pass

    reg = ToolRegistry([SimpleTool("echo", "Echo.", {"text": "t"},
                                   lambda text: f"echoed:{text}")])
    steps = run_real_episode(ScriptedBackend(), reg, "say hi", max_steps=4)
    assert steps[0]["tool_events"], "no structured events recorded"
    ev = steps[0]["tool_events"][0]
    assert ev["name"] == "echo" and ev["args"] == {"text": "hi"}
    assert ev["result"] == "echoed:hi" and ev["latency_s"] is not None
    assert steps[0]["task"] == "say hi"
    assert steps[0]["logprobs_available"] is True


# ---------------------------------------------------------------------
def test_task_anchor_never_includes_tool_results():
    """A first-step wrong document must not be compared against itself."""
    decoy = ("Sourdough bread relies on a wild-yeast starter fermented over "
             "several days by bakers who feed it flour and water.")
    task = ("Find recent arXiv papers about echo state networks for anomaly "
            "detection and summarise how reservoirs are trained.")
    step = _step(f'looking it up [search({{"q": "echo state networks"}}) -> {decoy}]',
                 task=task)
    ep = adapter.episode_from_trace([step], "anchored", extended=True,
                                    use_sentence_transformers=False)
    anchor = adapter.embed_text(task, False)
    assert abs(ep.X[0, IDX_TASK_SIM] - float(ep.X[0, :D_SEM] @ anchor)) < 1e-9
    assert ep.X[0, IDX_TASK_SIM] < 1.0


def test_grounding_lexical_miss_uses_the_task_not_the_result():
    decoy = ("Sourdough bread relies on a wild-yeast starter fermented over "
             "several days by bakers who feed it flour and water.")
    task = ("Find hotel prices in Osaka and report the cheapest nightly rate "
            "for the city centre district.")
    step = _step(f'[web_search({{"q": "osaka hotel prices"}}) -> {decoy}]',
                 task=task)
    ep = adapter.episode_from_trace([step], "lex", grounding=True,
                                    use_sentence_transformers=False)
    assert ep.X[0, IDX_GRD_LEX_MISS] == 1.0, (
        "a first-step off-topic document was excused by self-anchoring")


# ---------------------------------------------------------------------
def test_missing_logprobs_are_distinguishable_from_high_confidence():
    missing = adapter.uncertainty_features(None)
    confident = adapter.uncertainty_features([-0.001] * 20)
    assert missing[0] == adapter.MISSING_SURPRISAL
    assert missing[0] < 0.0 <= confident[0]
    assert not np.allclose(missing[:2], confident[:2])


def test_observed_surprisal_is_not_floored():
    tiny = adapter.uncertainty_features([-1e-6] * 5)
    assert tiny[0] < 0.05, "genuine high confidence is still clamped"


def test_explicit_logprobs_available_flag_is_honoured():
    assert adapter.logprobs_missing({"token_logprobs": [-0.1],
                                     "logprobs_available": False})
    assert not adapter.logprobs_missing({"token_logprobs": [-0.1]})
    assert adapter.logprobs_missing({"token_logprobs": []})


# ---------------------------------------------------------------------
@pytest.mark.parametrize("steps,match", [
    ("not a list", "must be a list"),
    ([], "no steps"),
    (["not an object"], "must be an object"),
    ([{"text": "x", "latency_s": float("nan")}], "finite"),
    ([{"text": "x", "latency_s": -1.0}], "non-negative"),
    ([{"text": "x", "token_logprobs": [float("inf")]}], "non-finite"),
    ([{"text": "x", "token_logprobs": "nope"}], "must be a list"),
    ([{"text": "x", "output_tokens": -5}], "negative"),
    ([{"text": "x", "tool_events": [{"args": {}}]}], "no name"),
])
def test_invalid_traces_are_rejected_with_context(steps, match):
    with pytest.raises(adapter.TraceSchemaError, match=match):
        adapter.validate_steps(steps, context="ep-1")


def test_validation_coerces_benign_fields():
    out = adapter.validate_steps([{"text": None, "output_tokens": "12"}], "ep")
    assert out[0]["text"] == "" and out[0]["output_tokens"] == 12
    assert out[0]["error"] is False and out[0]["latency_s"] == 1.0


def test_load_trace_reports_the_file_and_line(tmp_path):
    bad = tmp_path / "broken.jsonl"
    bad.write_text('{"text": "ok"}\nnot json\n', "utf-8")
    with pytest.raises(adapter.TraceSchemaError, match="broken.jsonl:2"):
        adapter.load_trace_jsonl(bad)


# ---------------------------------------------------------------------
def test_complete_but_broken_json_is_flagged_when_not_truncated():
    broken = '{"price": 215,, "city" "Osaka"}'
    step = _step("checking", tool_events=[
        make_tool_event("db", {"q": "x"}, broken, is_error=False)])
    ep = adapter.episode_from_trace([step], "json", grounding=True,
                                    use_sentence_transformers=False)
    assert ep.X[0, IDX_GRD_JSON_BROKEN] == 1.0

    # A result the collector marked truncated keeps the prefix allowance.
    cut = '{"rows": [{"price": 215, "city": "Osa'
    step2 = _step("checking", tool_events=[
        make_tool_event("db", {"q": "x"}, cut, is_error=False,
                        max_result_chars=len(cut))])
    step2["tool_events"][0]["result_truncated"] = True
    ep2 = adapter.episode_from_trace([step2], "json2", grounding=True,
                                     use_sentence_transformers=False)
    assert ep2.X[0, IDX_GRD_JSON_BROKEN] == 0.0


def test_malformed_json_injection_is_actually_malformed():
    """The injected payload must not be a completable JSON prefix."""
    from derail.harness.inject import ToolInjector
    from derail.harness.tools import ToolResult

    inj = ToolInjector("malformed_json", tau=0, seed=1)
    inj.t = 1
    # web_search is in the class's applicable-tool set; a class may
    # only corrupt tools whose failure mode it actually describes.
    clean = ToolResult("web_search", {"q": "x"},
                       '{"results": [{"text": "Osaka hotels 215 per night"}]}',
                       False, 0.3)
    payload = inj.apply(clean).content
    with pytest.raises(json.JSONDecodeError):
        json.loads(payload)
    # Flagged whether or not the collector marked the result truncated.
    assert adapter._json_broken(payload, truncated=False)
    assert adapter._json_broken(payload, truncated=True)


def test_valid_json_result_is_never_flagged():
    step = _step("checking", tool_events=[
        make_tool_event("db", {}, '{"price": 215}', is_error=False)])
    ep = adapter.episode_from_trace([step], "json3", grounding=True,
                                    use_sentence_transformers=False)
    assert ep.X[0, IDX_GRD_JSON_BROKEN] == 0.0


# ---------------------------------------------------------------------
def test_optional_tool_parameters_are_not_declared_required():
    from derail.harness import real_tools
    from derail.harness.tools import tool_json_schema

    schema = tool_json_schema(real_tools.ListDir("."))
    assert schema["required"] == [], "list_dir() is a valid call"
    assert schema["properties"]["path"]["default"] == "."

    gh = tool_json_schema(real_tools.GitHubTool())
    assert gh["required"] == ["action"], gh["required"]
    assert set(gh["properties"]) == {"action", "query", "repo_name", "path"}

    wiki = tool_json_schema(real_tools.WikipediaSearch())
    assert wiki["required"] == ["query"]


# ---------------------------------------------------------------------
def test_autogen_wrapper_passes_values_not_parameter_names():
    from derail.harness.frameworks import _autogen_tools
    from derail.harness.tools import SimpleTool, ToolRegistry

    reg = ToolRegistry([SimpleTool("echo", "Echo the input text.",
                                   {"text": "text to echo"},
                                   lambda text: f"got:{text}")])
    tool = _autogen_tools(reg, None)[0]
    assert tool._func(text="hello") == "got:hello", (
        "the wrapper still forwards the parameter NAME instead of its value")


def test_langchain_wrapper_passes_values():
    from derail.harness.frameworks import _langchain_tools
    from derail.harness.tools import SimpleTool, ToolRegistry

    reg = ToolRegistry([SimpleTool("echo", "Echo the input text.",
                                   {"text": "text to echo"},
                                   lambda text: f"got:{text}")])
    tool = _langchain_tools(reg, None)[0]
    assert tool.invoke({"text": "hello"}) == "got:hello"


# ---------------------------------------------------------------------
def test_loop_does_not_stop_between_a_tool_request_and_its_result():
    from derail.harness.frameworks import reached_step_limit

    at_limit_pending = [{"_pending": True}] * 12
    at_limit_done = [{"_pending": False}] * 12
    assert not reached_step_limit(at_limit_pending, 12), (
        "the loop would stop before recording the requested tool result")
    assert reached_step_limit(at_limit_done, 12)
    assert not reached_step_limit([{"_pending": False}], 12)
    assert not reached_step_limit([], 12)


# ------------------------------------------------------- no silent regression
def test_legacy_in_bracket_and_v1_forms_still_parse():
    v2 = '[lookup({"city": "Osaka"}) -> $215]'
    v1 = '[lookup({"city": "Osaka"})]'
    c2, _ = parse_tool_bits(v2)
    c1, _ = parse_tool_bits(v1)
    assert c2[0].result == "$215" and c2[0].has_result
    assert c1[0].result == "" and not c1[0].has_result
    assert c1[0].name == c2[0].name == "lookup"


def test_reasoning_text_excludes_tool_bits():
    calls, reasoning = parse_tool_bits(
        'thinking about it [a({"x": 1}) -> res] and then concluding')
    assert calls and reasoning == "thinking about it and then concluding"


def test_error_results_still_drive_tool_success():
    steps = [_step('[a({"x": 1}) -> Error: service unavailable] '
                   '[b({"y": 2}) -> fine]')]
    ep = adapter.episode_from_trace(steps, "err", extended=True,
                                    use_sentence_transformers=False)
    assert ep.X[0, IDX_REASON_DEPTH] == 2.0
    assert abs(ep.X[0, IDX_TOOL_SUCCESS] - 0.5) < 1e-9
