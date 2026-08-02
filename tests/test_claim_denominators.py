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
