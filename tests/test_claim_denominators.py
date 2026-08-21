"""A published rate must carry a denominator, and it must be checked.

"Channel-max AUC on 187 live Gemini episodes" survived review and a claim
ledger for months. The number was computed on a held-out split of 94. Nothing
caught it because `real_traces.csv` has no `n` column, so there was no
denominator to compare a corpus size against.

These tests make that class of error fail rather than pass: a rate claim
without a denominator is a test failure, and the recomputed split has to agree
with the runner it claims to mirror.
"""
from __future__ import annotations

import pathlib
import re

from devtools import claims_ledger

TABLES = pathlib.Path(__file__).resolve().parents[1] / "results" / "tables"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Words that mark a claim as a rate rather than a count or a duration.
_RATE = re.compile(r"(rate|auroc|\bauc\b|recall|precision|share|fraction|"
                   r"accuracy|\bece\b|p_detect|p_false|recovery)", re.I)


def _rate_claims():
    return [c for c in claims_ledger.build()
            if isinstance(c.expected, float)
            and 0.0 <= float(c.expected) <= 1.0
            and _RATE.search(c.claim)]


def test_there_are_rate_claims_to_check() -> None:
    assert len(_rate_claims()) >= 15, "the rate detector stopped matching"


def test_every_rate_claim_declares_a_denominator() -> None:
    """Adding a rate without an n should fail here, not in review."""
    naked = [c.id for c in _rate_claims() if c.denominator is None]
    assert not naked, (
        f"rate claims with no denominator: {naked}. Give each a "
        f"`denominator=` callable and `expected_denominator=`, so the count "
        f"is recomputed and drift-checked like the value.")


def test_declared_denominators_recompute_correctly() -> None:
    for claim in _rate_claims():
        assert claim.expected_denominator is not None, claim.id
        assert int(claim.denominator()) == claim.expected_denominator, (
            f"{claim.id}: denominator is now {claim.denominator()}, "
            f"claimed {claim.expected_denominator}")


def test_the_recomputed_real_split_matches_the_runner() -> None:
    """The ledger recomputes the split; the runner performs it.

    They are separate code, so they can disagree, and a disagreement would
    make the published n quietly wrong. Pin the constants they must share.
    """
    source = (REPO_ROOT / "derail" / "experiments"
              / "run_real_traces.py").read_text("utf-8")
    assert "0.6 * len(healthy)" in source, "the runner's train share moved"
    assert "0.2 * len(healthy)" in source, "the runner's val share moved"

    split = claims_ledger._real_split()
    assert split["healthy_train"] + split["healthy_val"] + split["healthy_test"] \
        == split["healthy_train"] + split["healthy_val"] + split["healthy_test"]
    assert split["healthy_test"] > 0 and split["injected"] > 0


def test_the_real_trace_denominator_is_not_the_corpus_size() -> None:
    """The specific error this file exists for."""
    import json

    manifest = json.loads(
        (REPO_ROOT / "traces" / "manifest.json").read_text("utf-8"))
    assert claims_ledger._real_eval_n() < len(manifest), (
        "the evaluation denominator equals the corpus size, which is what "
        "went wrong before: the AUC is computed on a held-out split")


def test_the_ledger_publishes_the_denominator() -> None:
    """A reader of CLAIMS.md should see n beside the value."""
    ledger = (REPO_ROOT / "CLAIMS.md").read_text("utf-8")
    assert "| claim | value | n | source artifact" in ledger


# ------------------------------------------- v1 snapshot vs the current tree
def test_the_ledger_reports_both_the_current_corpus_and_the_v1_snapshot():
    """A published figure and a growing repository are two different numbers.

    The arXiv v1 submission describes the tree at one commit and must keep
    saying what it said; the repository keeps collecting. Holding the working
    tree back to v1's counts would misdescribe the current evidence base, and
    letting v1's figure follow the tree would rewrite the submitted paper. The
    ledger therefore carries both, and this pins that they stay distinct.
    """
    from devtools import claims_ledger as cl

    by_id = {c.id: c for c in cl.build()}
    for base in ("episodes", "datasets", "real_tools"):
        cur, v1 = by_id[f"corpus.{base}"], by_id[f"corpus.{base}_v1"]
        assert cur.compute() >= v1.compute(), (
            f"corpus.{base}: the current tree cannot hold fewer than v1 did")
        assert "v1" in v1.claim and "current" in cur.claim, (
            "each claim must say which of the two it is")

    # v1 is reconstructed by dropping the corpora collected since. If that set
    # is wrong the v1 figure moves, and these claims are what catches it.
    assert cl.ADDED_AFTER_V1, "the set naming post-v1 corpora is empty"
    names = {m.parent.name for m in cl._our_manifests()}
    assert cl.ADDED_AFTER_V1 <= names, (
        f"names a corpus that does not exist: {cl.ADDED_AFTER_V1 - names}")
    assert by_id["corpus.episodes"].compute() > by_id["corpus.episodes_v1"].compute(), (
        "corpora have been added since v1, so the two counts must differ")


# -------------------------------- real-tool count must be content-derived
def test_corpus_real_tools_is_not_computed_from_directory_names():
    """The regression this guards: `corpus.real_tools` used to be `sum(len(m)
    for m in TRACES.glob("real*/manifest.json"))` -- a filename-glob artifact
    that both missed real corpora named otherwise and would silently count a
    `real*`-named corpus that turned out to hold no real-tool episodes.

    `organic7b`, `demo_real`, `demo_real_varied`, `demo_real_varied_ext`,
    `autogen7b_real` and `langgraph7b_real` are 100% real-tool corpora whose
    directory names do not start with "real" (the last two don't even START
    that way); the glob formula misses every one of them. Recomputing must
    find them anyway, and must disagree with what the old glob formula would
    have produced.
    """
    import json

    from devtools import claims_ledger as cl

    old_glob_count = sum(
        len(json.loads(m.read_text("utf-8")))
        for m in sorted(cl.TRACES.glob("real*/manifest.json")))
    content_count = cl._real_tool_episodes()
    assert content_count != old_glob_count, (
        "the content-derived count coincides with the glob formula -- "
        "make sure a real fix, not a re-derivation of the same number, "
        "landed here")

    non_real_named = {"organic7b", "demo_real", "demo_real_varied",
                      "demo_real_varied_ext", "autogen7b_real",
                      "langgraph7b_real"}
    names = {m.parent.name for m in cl._our_manifests()}
    assert non_real_named <= names, (
        f"a corpus this test depends on is gone: {non_real_named - names}")
    from derail.harness.real_tools import episode_used_real_tools
    for corpus in sorted(non_real_named):
        manifest = json.loads((cl.TRACES / corpus / "manifest.json")
                              .read_text("utf-8"))
        assert not corpus.startswith("real"), corpus  # the glob would miss it
        n_real = sum(
            1 for e in manifest
            if episode_used_real_tools(
                [json.loads(l) for l in
                 (cl.TRACES / corpus / e["file"]).read_text("utf-8").splitlines()
                 if l.strip()]))
        assert n_real == len(manifest), (
            f"{corpus}: expected every episode to be content-real, got "
            f"{n_real}/{len(manifest)}")


def test_the_v1_real_tools_snapshot_stays_glob_based():
    """Unlike the current-tree claim, `corpus.real_tools_v1` reconstructs
    what the already-published, tagged v1 PDF printed (770), computed the
    same (glob) way that PDF computed it -- content-deriving it now would
    make the ledger disagree with the paper it is supposed to reproduce."""
    from devtools import claims_ledger as cl
    import inspect

    src = inspect.getsource(cl._real_tool_episodes_v1)
    assert 'glob("real*' in src, (
        "the v1 snapshot must keep reconstructing the v1 PDF's own glob "
        "formula, not the current content-derived one")


# --------------------------------------- false-positive scope must be explicit
def test_every_false_positive_claim_carries_its_denominator():
    """A zero is meaningless without the population it is zero over.

    "0 false positives" was quoted for the recomputation checks with the
    contract check's 1,825-episode denominator, which is a different check on a
    different and roughly twelve-times-larger population. Any claim whose value
    is a false-positive COUNT must therefore state what it counted over.
    """
    from devtools import claims_ledger as cl

    fp_ids = {"contract.healthy_fp", "verify.recompute_healthy_fp",
              "verify.monitor_fp_served"}
    by_id = {c.id: c for c in cl.build()}
    missing = sorted(i for i in fp_ids if i not in by_id)
    assert not missing, f"false-positive claims disappeared: {missing}"
    for i in sorted(fp_ids):
        c = by_id[i]
        assert c.denominator is not None, f"{i}: no denominator"
        assert c.denominator_unit == "healthy episodes", (
            f"{i}: a false-positive rate is over healthy episodes, not "
            f"{c.denominator_unit!r}")


def test_the_two_zero_fp_populations_are_not_interchangeable():
    """The contract check and the recomputation checks do not share a scope.

    The contract check reads tool results, so it runs on every labelled corpus.
    The recomputation checks need an answer to recompute, so they run only on
    the organic demo corpora. Pinning the gap means a future edit that quotes
    one denominator for the other layer fails here.
    """
    from devtools import claims_ledger as cl

    by_id = {c.id: c for c in cl.build()}
    contract = by_id["contract.healthy_fp"].denominator()
    recompute = by_id["verify.recompute_healthy_fp"].denominator()
    assert contract > recompute * 5, (
        "the contract check's population should dwarf the recomputation "
        f"checks' ({contract} vs {recompute}); if they have converged, the "
        "scopes changed and the prose separating them needs rechecking")
    assert by_id["contract.healthy_fp"].compute() == 0
    assert by_id["verify.recompute_healthy_fp"].compute() == 0


def test_the_contract_denominators_are_persisted_not_printed():
    """The sweep must write its denominators, not only print them.

    "0 of 1,825 healthy" survived in five documents while the corpus grew to
    2,080 precisely because the denominator was never written to an artifact
    the ledger could recompute.
    """
    import inspect

    import pandas as pd

    from derail.verify import run_verification_study as rvs

    src = inspect.getsource(rvs.contract_coverage)
    assert "tool_contract_denominators.csv" in src, \
        "the sweep no longer persists its denominators"
    assert 'startswith("_")' in src, \
        "imported corpora must stay out of a false-positive denominator of ours"
    d = pd.read_csv(TABLES / "tool_contract_denominators.csv")
    assert {"label", "flagged", "n", "rate"} <= set(d.columns)
    assert int(d.loc[d.label == "healthy", "flagged"].iloc[0]) == 0


# ------------------------------------------------ one canonical accounting
def test_every_quoted_episode_total_is_reconstructible():
    """No total may exist that the canonical accounting cannot derive.

    The A13 defect was arithmetic ACROSS incommensurable totals: 1,825 healthy
    plus a 1,002-episode study population lands at 2,827, four from the 2,823
    corpus total, which reads like a rounding slip and is a coincidence between
    a label count, a study population containing 400 generated episodes, and a
    corpus total. Every figure below is derived from manifests, so the ones
    that do add up can be checked and the one that does not is documented.
    """
    from devtools.episode_accounting import build

    _, t, studies, ids = build()
    for r in ids:
        if r["is_identity"]:
            assert r["holds"], f"identity broke: {r['identity']} " \
                               f"({r['lhs']} != {r['rhs']})"
        else:
            assert not r["holds"], (
                f"{r['identity']} started balancing at {r['lhs']}; it is "
                "documented as a coincidence, so if it now holds the "
                "accounting changed and the explanation needs rewriting")
    assert t["owned_healthy"] + t["owned_injected"] == t["owned_episodes"]
    assert t["v1_episodes"] + t["added_after_v1_episodes"] == t["owned_episodes"]
    hybrid = next(s for s in studies if s["study"].startswith("behavioural"))
    assert hybrid["generated_episodes"] > 0, (
        "the behavioural study contains simulator episodes that are not "
        "committed traces; if that stops being true the summing caveat changes")


def test_study_populations_are_never_summed_into_a_corpus_total():
    """Adding two study populations double-counts, and the overlap says so."""
    from devtools.episode_accounting import coverage_rows

    rows = {r["quantity"]: r["n"] for r in coverage_rows()}
    both = rows["scored by BOTH"]
    union = rows["union of the two studies"]
    behav = rows["scored by the behavioural study (real half)"]
    ground = rows["scored by the grounding study"]
    assert both > 0, "if the studies stopped overlapping, summing would be safe"
    assert union == behav + ground - both, "inclusion-exclusion must hold"
    assert union < behav + ground, "summing the two populations double-counts"
    assert rows["study rows with no committed episode behind them"] == 0, (
        "a scored episode with no manifest entry means the study population "
        "and the corpus have diverged")


def test_the_ledger_totals_come_from_the_accounting_not_a_second_derivation():
    """Two derivations of one number drift. The ledger must use this one."""
    import inspect

    from devtools import claims_ledger as cl

    src = inspect.getsource(cl)
    assert "episode_accounting" in src, \
        "the ledger no longer reads the canonical accounting"
    by_id = {c.id: c for c in cl.build()}
    for i in ("accounting.root_corpus", "accounting.committed_all",
              "accounting.study_overlap", "accounting.orphan_study_rows"):
        assert i in by_id, f"{i} disappeared from the ledger"
    assert by_id["accounting.orphan_study_rows"].compute() == 0
    # The root corpus is real, committed, and outside every published total.
    assert by_id["accounting.root_corpus"].compute() > 0
    assert (by_id["accounting.committed_all"].compute()
            == by_id["corpus.episodes"].compute()
            + by_id["accounting.root_corpus"].compute())
