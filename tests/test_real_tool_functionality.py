"""Real-tool functionality.

The tools reach live third-party endpoints, so these check the contracts that
hold regardless of what those endpoints return: output caps are applied, the
SQL tool stays read-only, and a tool's fingerprint changes when a setting that
would change its output changes.
"""
from __future__ import annotations

import inspect
import json

import pytest


# -------------------------------------------------------------------
def test_vector_search_ranks_by_lexical_relevance():
    from derail.harness.real_tools import VectorSearchTool
    t = VectorSearchTool()
    # A CUSUM query must return the CUSUM document, not a random one; an ESN
    # query must return the ESN document. A hashing embedding ranks these at
    # random.
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
def test_real_tool_endpoints_are_centralized():
    """URLs pinned so a future edit cannot drift an endpoint unnoticed."""
    from derail.harness import real_tools as rt

    assert rt._WIKIPEDIA_SEARCH_URL == "https://en.wikipedia.org/w/api.php"
    assert rt._ARXIV_SEARCH_URL == "https://export.arxiv.org/api/query"
    assert rt._DUCKDUCKGO_SEARCH_URL == "https://api.duckduckgo.com/"
    assert rt._TAVILY_SEARCH_URL == "https://api.tavily.com/search"
    assert (rt._OPEN_METEO_GEOCODING_URL ==
            "https://geocoding-api.open-meteo.com/v1/search")
    assert rt._OPEN_METEO_FORECAST_URL == "https://api.open-meteo.com/v1/forecast"


def test_real_tool_run_reads_the_live_endpoint_constant(monkeypatch):
    """Each tool must read its module-level URL constant at call time, not a
    value captured once elsewhere -- otherwise the "centralized" constant
    would be cosmetic and an override would not actually change behaviour."""
    from derail.harness import real_tools as rt

    seen = {}

    def record(label, payload):
        def get(u, *a, **kw):
            seen[label] = u
            return payload
        return get

    monkeypatch.setattr(rt, "_WIKIPEDIA_SEARCH_URL", "https://example.invalid/wiki")
    rt.WikipediaSearch(get=record("wiki", b'{"query": {"search": []}}')).run("x")
    assert seen["wiki"].startswith("https://example.invalid/wiki?")

    monkeypatch.setattr(rt, "_ARXIV_SEARCH_URL", "https://example.invalid/arxiv")
    rt.ArxivSearch(get=record(
        "arxiv", b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>')).run("x")
    assert seen["arxiv"].startswith("https://example.invalid/arxiv?")

    monkeypatch.setattr(rt, "_DUCKDUCKGO_SEARCH_URL", "https://example.invalid/ddg")
    rt.DuckDuckGoSearch(get=record("ddg", b"{}")).run("x")
    assert seen["ddg"].startswith("https://example.invalid/ddg?")

    def fake_post(u, payload, *a, **kw):
        seen["tavily"] = u
        return b'{"results": []}'

    monkeypatch.setattr(rt, "_TAVILY_SEARCH_URL", "https://example.invalid/tavily")
    monkeypatch.setattr("derail.config.get_api_key", lambda name, required=False: "k")
    monkeypatch.setattr(rt, "_http_post", fake_post)
    rt.TavilySearch().run("x")
    assert seen["tavily"] == "https://example.invalid/tavily"

    monkeypatch.setattr(rt, "_OPEN_METEO_GEOCODING_URL", "https://example.invalid/geo")
    monkeypatch.setattr(rt, "_OPEN_METEO_FORECAST_URL", "https://example.invalid/fc")
    urls = []

    def fake_get(u, *a, **kw):
        urls.append(u)
        if len(urls) == 1:
            return json.dumps({"results": [{"latitude": 1, "longitude": 2,
                                            "name": "X", "country": "Y"}]}).encode()
        return json.dumps({"current_weather": {"temperature": 1, "windspeed": 1,
                                                "weathercode": 0}}).encode()

    monkeypatch.setattr(rt, "_http_get", fake_get)
    rt.OpenMeteoWeather().run("X")
    assert urls[0].startswith("https://example.invalid/geo?")
    assert urls[1].startswith("https://example.invalid/fc?")


def test_committed_cassettes_replay_after_endpoint_centralization():
    """Moving the endpoint URLs into module constants changed
    `tool_fingerprint()`'s source digest for every affected tool class, and
    therefore the cassette key every one of their recordings lived under.
    127 already-recorded calls (arXiv, Wikipedia, DuckDuckGo) in the
    `demo_real_varied_ext` corpus were re-keyed (same content, new filename)
    so replay keeps working without a live call. This pins one committed
    example per re-keyed tool so a future regression (e.g. reverting the
    re-key, or breaking source-based fingerprinting) fails loudly here
    instead of silently forcing a live re-record."""
    from pathlib import Path

    from derail.harness.record_replay import Cassette
    from derail.harness.real_tools import ArxivSearch, DuckDuckGoSearch, WikipediaSearch
    from derail.harness.tools import ToolRegistry

    cassette_dir = (Path(__file__).resolve().parents[1] / "traces" /
                    "_cassettes" / "demo_real_varied_ext")
    cases = [
        (ArxivSearch, "arxiv_search", {"query": "ECG anomaly detection survey"}),
        (WikipediaSearch, "wikipedia_search", {"query": "ECG anomaly detection"}),
        (DuckDuckGoSearch, "web_search",
         {"query": "Gaussian processes convolutional networks ECG anomaly detection"}),
    ]
    for ctor, name, args in cases:
        reg = ToolRegistry([ctor()])
        cas = Cassette(str(cassette_dir), mode="replay")
        res = reg.call(name, args, cassette=cas)          # KeyError = regression
        assert not res.is_error and res.content
        assert cas.n_replayed == 1 and cas.n_recorded == 0


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
