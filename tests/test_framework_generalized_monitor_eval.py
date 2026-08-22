"""End-to-end offline smoke test for the post-fix disjoint-corpus
evaluation script, against synthetic manifests -- no live calls, no
dependency on the real collected corpora.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from derail.experiments import run_framework_generalized_monitor_eval as ev


def _write_corpus(corpus_dir, n_healthy=40, n_injected=8):
    corpus_dir.mkdir(parents=True)
    manifest = []
    rng = np.random.default_rng(0)

    def steps(n, err_from=None):
        out = []
        for t in range(n):
            out.append({
                "text": f"step {t} {rng.integers(0, 1000)}",
                "token_logprobs": [-0.1] * 10, "action": "tool_call",
                "latency_s": float(rng.uniform(0.3, 0.8)), "output_tokens": 10,
                "error": err_from is not None and t >= err_from})
        return out

    for i in range(n_healthy):
        eid = f"healthy-{i:03d}"
        s = steps(10)
        (corpus_dir / f"{eid}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in s), "utf-8")
        manifest.append({"episode_id": eid, "file": f"{eid}.jsonl",
                         "failure_class": None, "tau": None, "T": len(s)})
    for i in range(n_injected):
        eid = f"injected-{i:03d}"
        s = steps(10, err_from=4)
        (corpus_dir / f"{eid}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in s), "utf-8")
        manifest.append({"episode_id": eid, "file": f"{eid}.jsonl",
                         "failure_class": "looping", "tau": 4, "T": len(s)})
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest), "utf-8")


@pytest.mark.filterwarnings("ignore:calibrating on")
def test_main_runs_end_to_end_on_synthetic_disjoint_corpora(tmp_path, monkeypatch):
    traces_root = tmp_path / "traces"
    _write_corpus(traces_root / "langgraph7b_real2")
    _write_corpus(traces_root / "autogen7b_real2")
    baseline = tmp_path / "baseline.md"

    monkeypatch.setattr(ev, "TRACES_ROOT", traces_root)
    monkeypatch.setattr(ev, "BASELINE_REPORT", baseline)
    # Redirect EVERY output path, found by name rather than listed one by one.
    # Listing them let `OUT_SUMMARY` be added without a matching line here, and
    # this test then overwrote the real, published summary table with numbers
    # from its own synthetic corpus.
    redirected = {}
    for name in [n for n in dir(ev) if n.startswith("OUT_")]:
        target = tmp_path / getattr(ev, name).name
        monkeypatch.setattr(ev, name, target)
        redirected[name] = target
    assert {"OUT_CSV", "OUT_REPORT", "OUT_SUMMARY"} <= set(redirected)

    ev.main()

    for name, target in redirected.items():
        assert target.exists(), f"{name} was not written"
        assert tmp_path in target.parents, f"{name} escaped the tmp path"
    text = redirected["OUT_REPORT"].read_text("utf-8")
    assert "| langgraph |" in text and "| autogen |" in text
    assert "| frozen |" in text and "| calibrated |" in text


def test_every_output_path_is_under_results(tmp_path):
    """A new OUT_* must stay inside `results/`, so the redirect above keeps
    working and a stray write lands somewhere the manifest hashes."""
    for name in [n for n in dir(ev) if n.startswith("OUT_")]:
        path = getattr(ev, name)
        assert (ev.REPO_ROOT / "results") in path.parents, f"{name}: {path}"
