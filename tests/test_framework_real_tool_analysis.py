"""Regression: the analysis script's admission replay must use each
episode's ORIGINAL REQUESTED tau (`entry["requested_tau"]`), not its
recorded actual/landed tau (`entry["tau"]`) -- replaying a ramped class
(`tool_cascade`) with the wrong tau desyncs its RNG draws from what
actually happened live and can wrongly exclude a genuinely-landed episode
from scoring. Offline, no live calls.
"""
from __future__ import annotations

import json

from derail.experiments.run_framework_real_tool_analysis import _load_episodes
from derail.harness.inject import ToolInjector
from derail.harness.tools import ToolResult

SEED, REQUESTED_TAU = 1, 2  # deterministic ramp-miss: lands at t=3, not t=2


def _make_ramped_trace(failure_class, tau, seed, n_steps=6):
    """A real trace built by driving a live ToolInjector step-by-step --
    not hand-authored numbers."""
    injector = ToolInjector(failure_class, tau=tau, seed=seed)
    steps = []
    for t in range(n_steps):
        injector.t = t
        clean = ToolResult(name="wikipedia_search", args={"q": "x"},
                           content="clean result", is_error=False, latency_s=0.1)
        out = injector.apply(clean)
        steps.append({"text": "", "action": "tool_call",
                     "tool_events": [{"name": out.name, "args": out.args,
                                      "result": out.content,
                                      "is_error": out.is_error,
                                      "latency_s": out.latency_s}]})
    return steps, injector


def test_load_episodes_uses_requested_tau_not_actual_tau_for_ramped_classes(tmp_path):
    steps, injector = _make_ramped_trace("tool_cascade", REQUESTED_TAU, SEED)
    assert injector.applied_count > 0, "fixture must land, or this test proves nothing"
    actual_tau = injector.first_applied_t
    assert actual_tau != REQUESTED_TAU, (
        "fixture must have a ramp miss so actual != requested, "
        "otherwise this test can't distinguish the two fields")

    corpus_dir = tmp_path / "fw7b_real"
    corpus_dir.mkdir()
    ep_file = "ep-000.jsonl"
    (corpus_dir / ep_file).write_text(
        "\n".join(json.dumps(s) for s in steps), "utf-8")
    manifest = [{
        "episode_id": "ep-000", "file": ep_file, "failure_class": "tool_cascade",
        "tau": actual_tau, "requested_tau": REQUESTED_TAU, "T": len(steps),
        "provenance": {"episode_seed": SEED},
    }]
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")

    episodes, excluded = _load_episodes(corpus_dir)
    assert excluded == [], f"a genuinely-landed episode must not be excluded: {excluded}"
    assert len(episodes) == 1
    assert episodes[0].tau == actual_tau
