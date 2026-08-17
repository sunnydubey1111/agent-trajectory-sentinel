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


#: The widest window any claim may carry, as one expression rather than a rule
#: plus exceptions: 2% of the claim, or the fixed absolute floor, whichever is
#: larger. Both halves are needed and neither is a carve-out. A pure relative
#: bound sends the window to zero as the claim does, so a near-zero claim would
#: be held to recomputation noise; a pure absolute bound is the original defect,
#: where 5e-3 was 13% of a 0.0385 rate. The floor is a fixed, small, declared
#: quantity, so a claim it covers has a TIGHT window in absolute terms even
#: when that window is a large fraction of a near-zero value.
def _widest_allowed(expected: float) -> float:
    return max(0.02 * abs(float(expected)), cl.ABS_TOL_FLOOR)


def test_no_claim_gets_a_window_wider_than_the_policy_allows():
    """No claim may carry a window wider than `_widest_allowed` of itself."""
    wide = []
    for c in _numeric_claims():
        if c.expected == 0 or isinstance(c.expected, int):
            continue
        window = cl.tolerance_for(c.expected)
        if window > _widest_allowed(c.expected):
            wide.append((c.id, float(c.expected), window))
    assert not wide, (
        "claims whose acceptance window exceeds max(2% of value, floor):\n"
        + "\n".join(f"  {i}: {e:g} -> window {w:g}" for i, e, w in wide))


def test_the_tolerance_function_has_the_shape_the_policy_describes():
    """Pin `tolerance_for` itself, since the claim-level bound cannot.

    With REL_TOL below 2% the bound above passes for every possible claim, so
    on its own it would guard nothing about the current constants -- it only
    catches REL_TOL being loosened past 2% later. The real protection is the
    shape of the function: relative above the knee, a fixed floor below it,
    never decreasing, and a knee where the two rules meet. The original defect
    was a function with no relative half at all.
    """
    knee = cl.ABS_TOL_FLOOR / cl.REL_TOL          # where floor and relative meet
    assert cl.tolerance_for(knee * 10) == pytest.approx(cl.REL_TOL * knee * 10)
    assert cl.tolerance_for(knee / 10) == cl.ABS_TOL_FLOOR
    assert cl.tolerance_for(knee) == pytest.approx(cl.ABS_TOL_FLOOR)

    # Monotone non-decreasing: a bigger claim never gets a tighter window.
    xs = [1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 100.0, 1045.0]
    tols = [cl.tolerance_for(x) for x in xs]
    assert tols == sorted(tols)

    # The relative half must actually exist and be the binding rule at scale,
    # which is what a single absolute tolerance got wrong.
    assert cl.tolerance_for(1045.0) > cl.ABS_TOL_FLOOR * 100
    assert cl.REL_TOL <= 0.02, (
        "REL_TOL above the 2% policy would let every claim carry a window "
        "wider than the ledger promises")


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
