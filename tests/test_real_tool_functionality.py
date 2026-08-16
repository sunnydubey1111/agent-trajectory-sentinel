"""Real-tool functionality.

The tools reach live third-party endpoints, so these check the contracts that
hold regardless of what those endpoints return: output caps are applied, the
SQL tool stays read-only, and a tool's fingerprint changes when a setting that
would change its output changes.
"""
from __future__ import annotations

import inspect

import pytest


# -------------------------------------------------------------------
def test_vector_search_ranks_by_lexical_relevance():
    from derail.harness.real_tools import VectorSearchTool
    t = VectorSearchTool()
    # A CUSUM query must return the CUSUM document, not a random one; an ESN
    # query must return the ESN document. The old hashing embedding ranked
    # these at random.
    assert "CUSUM" in t.run("CUSUM change detection", limit=1)
    assert "Echo State" in t.run("echo state network reservoir", limit=1)
    # Shared words drive relevance; a query with no overlap returns nothing.
    assert "No documentation results" in t.run("banana quantum bicycle")


def test_vector_search_has_no_silent_init_failure_branch():
    from derail.harness.real_tools import VectorSearchTool
    src = inspect.getsource(VectorSearchTool)
    assert "Qdrant client not initialized" not in src
    assert "_embed" not in src            # the random hashing embedding is gone


# -------------------------------------------------------------------
def test_cost_meter_reserves_before_spending():
    from derail.harness.record_replay import CostMeter, BudgetExceeded
    m = CostMeter(budget_usd=0.001)
    with pytest.raises(BudgetExceeded):
        m.reserve("gemini-2.5-flash", 200_000, 1024)   # over cap, refused up front
    assert m.spent_usd == 0.0                            # reserve never commits


def test_cost_meter_reserve_surfaces_unknown_model():
    from derail.harness.record_replay import CostMeter
    with pytest.raises(KeyError):
        CostMeter(budget_usd=1.0).reserve("no-such-model", 10, 10)


def test_gemini_backend_reserves_before_the_request():
    from derail.experiments import collect_traces
    src = inspect.getsource(collect_traces.GeminiBackend.step)
    i_reserve = src.index(".reserve(")
    i_generate = src.index("self._generate(")
    assert i_reserve < i_generate, "reserve must run before the billed request"
    # the unknown-model KeyError is no longer swallowed at charge time
    assert "skip metering" not in src


# -------------------------------------------------------------------
def test_generator_rejects_unsupported_d_sem():
    from derail.common import SimConfig
    from derail.telemetry.generator import EpisodeGenerator
    with pytest.raises(ValueError, match="d_sem"):
        EpisodeGenerator(SimConfig(d_sem=64), seed=0)


# -------------------------------------------------------------------
def test_duckduckgo_limit_is_not_off_by_one():
    from derail.harness.real_tools import DuckDuckGoSearch
    import json

    payload = json.dumps({
        "AbstractText": "a0",
        "RelatedTopics": [{"Text": f"t{i}"} for i in range(10)],
    }).encode()
    tool = DuckDuckGoSearch(get=lambda url: payload, max_results=3)
    out = tool.run("x")
    # abstract + related, capped at max_results total parts (was one extra).
    assert len(out.split(" | ")) <= 3


# -------------------------------------------------------------------
def test_seg_shift_handles_empty_and_out_of_width_slices():
    import warnings

    import numpy as np

    from derail.common import D_TOTAL
    from derail.telemetry.generator import _seg_shift

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # tau=0 -> empty pre-segment; must return 0.0, not NaN with a warning.
        z = np.zeros((5, D_TOTAL))
        out = _seg_shift(z, 0)
        assert all(v == 0.0 for v in out.values())
        # A 43-D episode with extended channel slices beyond its width.
        out2 = _seg_shift(np.zeros((6, 43)), 3)
        assert all(np.isfinite(v) for v in out2.values())


# -------------------------------------------------------------------
def test_torch_baselines_give_an_actionable_error_without_torch():
    from derail.monitor import seq_baselines
    src = inspect.getsource(seq_baselines)
    # the ImportError names the install command, not just "requires torch"
    assert "pip install torch" in src


# -------------------------------------------------------------------
@pytest.mark.parametrize("mod", ["experimental.run_telemetry_ablation",
                                 "experimental.run_st_ablation"])
def test_archived_clis_fail_explicitly(mod):
    import importlib
    with pytest.raises(SystemExit, match="archived|STALE|provenance"):
        importlib.import_module(mod)


def test_pmi_score_does_not_mutate_vocabulary():
    from experimental.pmi import AdjacentPMI as PMIModel
    m = PMIModel()
    m.fit(["the cat sat on the mat", "a dog ran fast"])
    before = (len(m.unigrams), len(m.bigrams))
    m.score("completely unseen novel tokens here")   # must not grow the vocab
    assert (len(m.unigrams), len(m.bigrams)) == before
