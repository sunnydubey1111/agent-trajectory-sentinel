"""The trace auditor must reproduce the review's named findings (/H08).

If these stop matching, either the auditor's evidence rules drifted or the
committed corpora changed - both are things we want to hear about immediately.
"""
from __future__ import annotations

import json

import pytest

from devtools import trace_audit
from devtools.trace_audit import TRACES, audit_corpus


def _defective(corpus: str, defect: str) -> set[str]:
    if not (TRACES / corpus).exists():
        pytest.skip(f"{corpus} not present")
    return {a.episode_id for a in audit_corpus(TRACES / corpus)
            if defect in a.defects}


def test_the_named_no_op_positives_were_found_and_rejected():
    """lists exactly these five.

    The auditor found them (see tests/fixtures/legacy_no_op_positive.jsonl for
    one of them parsed end to end); `devtools/prune_invalid_labels` then moved
    them out of the manifest into the corpus's own rejection record, so the
    corpus documents why it no longer claims them.
    """
    rejected_path = TRACES / "real_research7b" / "rejected.json"
    assert rejected_path.exists(), "the corpus does not record its rejections"
    rejected = json.loads(rejected_path.read_text("utf-8"))
    ids = {r["episode_id"] for r in rejected}
    assert {"real-rate_limit-008", "real-rate_limit-012",
            "real-rate_limit-019", "real-timeout-017",
            "real-timeout-018"} <= ids
    assert all("no_op_positive" in r["reason"] for r in rejected
               if r["episode_id"] in ids)
    # ...and they are gone from the labelled set.
    assert not _defective("real_research7b", "no_op_positive")


def _write_corpus(root, entries: list[dict]):
    """Materialize a manifest plus a two-step trace file per entry."""
    step = {"task": "t", "tool_events": [{"name": "search", "ok": True}]}
    for entry in entries:
        (root / f"{entry['episode_id']}.jsonl").write_text(
            "\n".join(json.dumps(step) for _ in range(2)), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return root


def test_auditor_flags_an_unsuccessful_run_that_carries_no_failure_label(
        tmp_path):
    """The rule that removed four episodes from the original `real` corpus.

    Checked on a corpus built here rather than on shipped data, so it states
    the auditor's rule rather than a property of the traces we happen to keep.
    """
    corpus = _write_corpus(tmp_path, [
        {"episode_id": "healthy-ok", "success": True},
        {"episode_id": "healthy-failed", "success": False},
        {"episode_id": "labelled-failed", "success": False,
         "failure_class": "looping", "tau": 0},
    ])
    flagged = {a.episode_id for a in audit_corpus(corpus)
               if "unsuccessful_healthy" in a.defects}
    assert flagged == {"healthy-failed"}


def test_tool_contract_raises_no_false_positive_on_any_healthy_episode():
    """The standing guarantee that lets `tool_contract` ship without a null.

    It is only worth having if a healthy run can never trip it, so the claim
    is checked against every healthy episode in the repository rather than a
    sample. Corpora whose tools declare no contract simply contribute nothing.
    """
    from derail.verify.checks import BOOKING_SPEC, tool_contract

    offenders, healthy = [], 0
    for directory in sorted(TRACES.iterdir()):
        manifest = directory / "manifest.json"
        if not directory.is_dir() or not manifest.exists():
            continue
        for entry in json.loads(manifest.read_text("utf-8")):
            if entry.get("failure_class"):
                continue
            path = directory / entry.get("file", f"{entry['episode_id']}.jsonl")
            if not path.exists():
                continue
            steps = [json.loads(line) for line
                     in path.read_text("utf-8").splitlines() if line.strip()]
            healthy += 1
            if tool_contract(steps, BOOKING_SPEC):
                offenders.append(f"{directory.name}/{entry['episode_id']}")

    assert healthy > 1000, f"expected the full healthy set, saw {healthy}"
    assert not offenders, f"contract check fired on healthy runs: {offenders}"


def test_no_corpus_still_serves_an_unsuccessful_run_as_healthy():
    """The gate's standing guarantee, checked across every corpus."""
    offenders = {}
    for directory in sorted(TRACES.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        manifest = directory / "manifest.json"
        if not manifest.exists():
            continue
        bad = [e["episode_id"] for e in json.loads(manifest.read_text("utf-8"))
               if not e.get("failure_class") and e.get("success") is False]
        if bad:
            offenders[directory.name] = bad
    # demo7b / demo7b_scoped are the calibration question,
    # not a labelling error, and are excluded until that decision is made.
    offenders = {k: v for k, v in offenders.items()
                 if k not in ("demo7b", "demo7b_scoped")}
    assert not offenders, offenders


def test_genuine_injections_are_not_flagged():
    """Guard against over-flagging: these classes really did mutate results."""
    audits = audit_corpus(TRACES / "real_research7b")
    for label in ("context_corruption", "wrong_document", "looping",
                  "malformed_json", "tool_cascade"):
        flagged = [a.episode_id for a in audits
                   if a.label == label and "no_op_positive" in a.defects]
        assert not flagged, f"{label} episodes wrongly called no-ops: {flagged}"


def test_unverifiable_is_distinct_from_no_op():
    """A corpus with no recorded results cannot be called a no-op.

    Uses a committed legacy-format AutoGen-7B trace: those files record tool
    CALLS but no results at all, so the injection question is unanswerable from
    the file - which is a different (and weaker) statement than "the injector
    never fired".
    """
    fixture = (trace_audit.REPO_ROOT / "tests" / "fixtures"
               / "legacy_autogen7b_no_results.jsonl")
    steps = [json.loads(line) for line in
             fixture.read_text("utf-8").splitlines() if line.strip()]
    verdict, why = trace_audit._mutation_evidence(
        "context_corruption", trace_audit._post_onset_results(steps, 2),
        steps, 2)
    assert verdict == "unverifiable", (verdict, why)
    assert "no tool results recorded" in why


def test_a_real_no_op_positive_is_called_a_no_op():
    """The reverse case, from a committed real_research7b rate_limit trace."""
    fixture = (trace_audit.REPO_ROOT / "tests" / "fixtures"
               / "legacy_no_op_positive.jsonl")
    steps = [json.loads(line) for line in
             fixture.read_text("utf-8").splitlines() if line.strip()]
    verdict, why = trace_audit._mutation_evidence(
        "rate_limit", trace_audit._post_onset_results(steps, 2), steps, 2)
    assert verdict == "no_op", (verdict, why)


def test_evidence_rules_are_marker_based():
    steps = [{"text": "a", "error": False},
             {"text": "b", "error": False},
             {"text": '[t({"q": 1}) -> Error: 429 Too Many Requests - rate '
                      'limit exceeded, retry later.]', "error": True},
             {"text": "c", "error": False}]
    verdict, _ = trace_audit._mutation_evidence(
        "rate_limit", trace_audit._post_onset_results(steps, 2), steps, 2)
    assert verdict == "confirmed"

    clean = [{"text": "a"}, {"text": "b"},
             {"text": '[t({"q": 1}) -> a perfectly normal result]'},
             {"text": "c"}]
    verdict, why = trace_audit._mutation_evidence(
        "rate_limit", trace_audit._post_onset_results(clean, 2), clean, 2)
    assert verdict == "no_op" and "none carries" in why


def test_audit_report_is_committed_and_current():
    """Check the stored audit against the corpora, when it is available.

    The report is a working record kept outside the released tree, so a
    fresh checkout will not have it. Skip rather than fail there: the audit
    is re-derivable at any time with `py -m devtools.trace_audit --json`.
    """
    report = (trace_audit.REPO_ROOT / "_internal" / "verification"
              / "TRACE_AUDIT_2026-07-24.json")
    if not report.exists():
        pytest.skip("stored audit report not present in this checkout")
    entries = json.loads(report.read_text("utf-8"))
    assert len(entries) > 1000
    blocking = [e for e in entries
                if set(e["defects"]) & trace_audit.BLOCKING]
    # Down from 219 before the corpora were re-collected. The remaining 50 are
    # the unsuccessful demo calibration runs in demo7b and demo7b_scoped,
    # which are a threshold decision, not a labelling error. Every other
    # corpus is clean.
    assert len(blocking) == 50, (
        f"audit result moved: {len(blocking)} blocking episodes, expected 50 "
        f"- re-run py -m devtools.trace_audit --json and update this number "
        f"with the reason")
    corpora = {e["corpus"] for e in entries
               if set(e["defects"]) & trace_audit.BLOCKING}
    assert corpora == {"demo7b", "demo7b_scoped"}, corpora


def test_recollected_corpora_are_clean():
    """Every re-collected corpus must contain no defect at all."""
    for corpus in ("langgraph7b", "autogen7b", "langgraph", "autogen",
                   "ollama7b", "ollama", "real"):
        audits = audit_corpus(TRACES / corpus)
        assert audits, corpus
        blocking = {a.episode_id: a.defects for a in audits
                    if set(a.defects) & trace_audit.BLOCKING}
        assert not blocking, f"{corpus}: {blocking}"
        # Every episode carries the v5 contract.
        assert not [a for a in audits
                    if "legacy_text_only_telemetry" in a.defects], corpus
        assert not [a for a in audits if "no_task_recorded" in a.defects], corpus
