"""Provenance-tagging infrastructure: cassette occurrence-keys, atomic
provenance, tool-call locality classification, and backward compatibility.

These are gate proofs required before any real-tool collection using them,
not experiment results -- no live network or Ollama calls here.
"""

from __future__ import annotations

import threading
from dataclasses import fields

from derail.harness.record_replay import Cassette
from derail.harness.tools import (ToolRegistry, ToolResult, SimpleTool,
                                  _call_locality)


def test_key_suffix_disambiguates_repeated_identical_calls(tmp_path):
    c = Cassette(tmp_path, mode="record")
    calls = iter(["first", "second", "third"])
    c.call("samekey", lambda: next(calls), key_suffix="occ0")
    c.call("samekey", lambda: next(calls), key_suffix="occ1")
    c.call("samekey", lambda: next(calls), key_suffix="occ2")
    # Reading back in replay mode must return each occurrence's own value,
    # not the last write clobbering the other two.
    r = Cassette(tmp_path, mode="replay")
    assert r.call("samekey", None, key_suffix="occ0") == "first"
    assert r.call("samekey", None, key_suffix="occ1") == "second"
    assert r.call("samekey", None, key_suffix="occ2") == "third"


def test_mode_record_never_reads_or_replays(tmp_path):
    c = Cassette(tmp_path, mode="record")
    for _ in range(5):
        c.call("k", lambda: "v")
    assert c.n_replayed == 0
    # A second Cassette instance pointed at the same (now-populated) dir,
    # still in mode="record", must still call live every time -- no lookup.
    c2 = Cassette(tmp_path, mode="record")
    tag_seen = []
    val, tag = c2.call("k", lambda: "fresh", return_provenance=True)
    tag_seen.append(tag)
    assert c2.n_replayed == 0
    assert tag_seen == ["live"]


def test_provenance_tag_matches_branch_not_shared_counters():
    """The tag returned by call() must reflect the branch THIS call took,
    verified under concurrent access -- not a counter read after the fact,
    which could be stale under a race."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cass = Cassette(d, mode="auto")
        # Prime one key so later calls to it are guaranteed replays.
        cass.call("primed", lambda: "cached")

        results: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            val, tag = cass.call("primed", lambda: "should-not-run",
                                 return_provenance=True)
            with lock:
                results.append(tag)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Every single call independently observed a hit -- correct
        # regardless of how n_replayed's increments interleaved across
        # threads, because the tag never reads that counter.
        assert results == ["replayed"] * 16


def test_tool_call_locality_classification():
    assert _call_locality("arxiv_search") == "external"
    assert _call_locality("wikipedia_search") == "external"
    assert _call_locality("get_weather") == "external"
    assert _call_locality("python") == "local"
    assert _call_locality("sql_query") == "local"
    assert _call_locality("read_file") == "local"
    assert _call_locality("list_dir") == "local"
    assert _call_locality("vector_search") == "local"


def test_registry_call_provenance_end_to_end(tmp_path):
    reg = ToolRegistry([SimpleTool("echo", "echo", {"text": "t"},
                                   lambda text: f"echo:{text}")])
    cass = Cassette(tmp_path, mode="auto")
    res1 = reg.call("echo", {"text": "hi"}, cassette=cass,
                    record_provenance=True)
    assert res1.source == "live_external"          # unclassified -> external
    res2 = reg.call("echo", {"text": "hi"}, cassette=cass,
                    record_provenance=True)
    assert res2.source == "cassette_replay"
    # No cassette at all: always live, locality decides external/local.
    reg2 = ToolRegistry([SimpleTool("python", "py", {"code": "c"},
                                    lambda code: "ran")])
    res3 = reg2.call("python", {"code": "1"}, cassette=None,
                     record_provenance=True)
    assert res3.source == "live_local"


def test_registry_call_default_is_backward_compatible(tmp_path):
    """No opt-in flags passed -> identical to the prior behavior: bare
    ToolResult, source stays None, no key_suffix applied."""
    reg = ToolRegistry([SimpleTool("echo", "echo", {"text": "t"},
                                   lambda text: f"echo:{text}")])
    cass = Cassette(tmp_path, mode="auto")
    res = reg.call("echo", {"text": "hi"}, cassette=cass)
    assert res.source is None
    assert isinstance(res, ToolResult)


def test_toolresult_positional_construction_unaffected():
    """Every pre-existing positional ToolResult(...) call site still works;
    source defaults to None without being passed."""
    r = ToolResult("t", {"a": 1}, "content", False, 0.01)
    assert r.source is None
    assert [f.name for f in fields(ToolResult)][:5] == [
        "name", "args", "content", "is_error", "latency_s"]


def test_cassette_call_default_return_shape_unaffected(tmp_path):
    """Every pre-existing caller of Cassette.call (no return_provenance)
    still gets a bare value back, not a tuple."""
    c = Cassette(tmp_path, mode="auto")
    v = c.call("k", lambda: "value")
    assert v == "value"
