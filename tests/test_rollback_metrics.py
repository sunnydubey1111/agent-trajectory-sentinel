from derail.evaluation.rollback_metrics import compute_metrics
from derail.intervene.real_tool_rollback import (
    HALTED, NOT_TRIGGERED, RECONSTRUCTION_FAILED, RECOVERED, STILL_WRONG)


def test_three_denominators_are_distinct_and_correctly_scoped():
    outcomes = ([RECOVERED] * 3 + [STILL_WRONG] * 2 + [HALTED] * 1
               + [NOT_TRIGGERED] * 3 + [RECONSTRUCTION_FAILED] * 1)
    m = compute_metrics(outcomes)
    n_selected = len(outcomes)              # 10
    n_triggered = n_selected - 3            # 7 (not_triggered excluded)
    n_attempted = n_triggered - 1           # 6 (reconstruction_failed excluded too)

    assert m.trigger_rate.k == n_triggered and m.trigger_rate.n == n_selected
    assert m.conditional_recovery.k == 3 and m.conditional_recovery.n == n_attempted
    assert m.end_to_end_recovery.k == 3 and m.end_to_end_recovery.n == n_selected
    # not_triggered contributes to end_to_end's denominator as not-recovered,
    # but never to conditional_recovery's denominator at all.
    assert m.end_to_end_recovery.rate < m.conditional_recovery.rate
    assert m.n_not_triggered == 3
    assert m.n_reconstruction_failed == 1


def test_not_triggered_stays_independently_visible():
    outcomes = [RECOVERED, NOT_TRIGGERED, NOT_TRIGGERED]
    m = compute_metrics(outcomes)
    assert m.n_not_triggered == 2
    assert m.end_to_end_recovery.k == 1 and m.end_to_end_recovery.n == 3


def test_ci_widens_at_small_n():
    m3 = compute_metrics([RECOVERED, STILL_WRONG, STILL_WRONG])
    m30 = compute_metrics([RECOVERED, STILL_WRONG, STILL_WRONG] * 10)
    width3 = m3.end_to_end_recovery.ci_high - m3.end_to_end_recovery.ci_low
    width30 = m30.end_to_end_recovery.ci_high - m30.end_to_end_recovery.ci_low
    assert width3 > width30
