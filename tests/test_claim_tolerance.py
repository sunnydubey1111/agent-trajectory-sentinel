"""The ledger's acceptance window has to mean the same thing at every scale.

A single absolute tolerance does not. At 5e-3 it granted 13% of a 0.0385 rate
and 0.0005% of a 1045-microsecond timing: too loose to catch drift in a small
rate, and tighter than a machine-dependent measurement can reproduce.
"""
from __future__ import annotations

import pytest

from devtools import claims_ledger as cl


def _numeric_claims():
    return [c for c in cl.build() if not isinstance(c.expected, str)]


def test_a_count_claim_is_exact():
    """2,823 episodes is not 2,824, and 0.5% of it is fourteen episodes."""
    assert cl.tolerance_for(2823) == 0.0
    assert cl.tolerance_for(0) == 0.0


def test_a_rate_claim_scales_with_its_own_magnitude():
    small = cl.tolerance_for(0.0385)
    large = cl.tolerance_for(0.8262)
    assert small < large, "a small rate must not get a larger window"
    assert large / 0.8262 == pytest.approx(cl.REL_TOL)


def test_a_near_zero_claim_keeps_a_usable_floor():
    """Without a floor a 0.001 claim would be held to a 5e-6 window, which is
    recomputation noise rather than drift."""
    assert cl.tolerance_for(1e-6) == cl.ABS_TOL_FLOOR


def test_no_claim_gets_a_window_wider_than_two_percent_of_itself():
    """The defect this file exists for: an absolute window is a different
    fraction of every claim, and on the smallest ones it was 13%."""
    wide = []
    for c in _numeric_claims():
        exp = float(c.expected)
        if exp == 0 or isinstance(c.expected, int):
            continue
        frac = cl.tolerance_for(c.expected) / abs(exp)
        if frac > 0.02:
            wide.append((c.id, exp, frac))
    assert not wide, (
        "claims whose acceptance window exceeds 2% of their own value:\n"
        + "\n".join(f"  {i}: {e:g} -> {f:.2%}" for i, e, f in wide))


def test_every_committed_claim_clears_the_tighter_window_comfortably():
    """Tightening must not put a passing claim on the edge of failing.

    A claim using most of its window is one regeneration away from a false
    alarm, which trains people to ignore the gate.
    """
    tight = []
    for c in _numeric_claims():
        window = cl.tolerance_for(c.expected)
        gap = abs(float(c.compute()) - float(c.expected))
        if window == 0.0:
            assert gap == 0.0, f"{c.id}: count claim off by {gap}"
            continue
        if gap > 0.5 * window:
            tight.append((c.id, gap, window))
    assert not tight, (
        "claims using over half their window:\n"
        + "\n".join(f"  {i}: gap {g:.6f} of {w:.6f}" for i, g, w in tight))
