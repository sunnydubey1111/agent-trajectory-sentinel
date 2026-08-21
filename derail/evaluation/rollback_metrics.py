"""Pre-registered rollback/retry metrics, frozen before collection.

Three denominators, never conflated:
  trigger_rate         = triggered / all selected
  conditional_recovery = recovered / (triggered AND reconstruction succeeded)
  end_to_end_recovery  = recovered / all selected

CI method frozen as Clopper-Pearson exact binomial (scipy's `binomtest`,
method="exact") -- appropriate at n as low as 3-12, unlike a normal
approximation. Reused, not reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import scipy.stats as sps

from derail.intervene.real_tool_rollback import (
    HALTED, NOT_TRIGGERED, RECONSTRUCTION_FAILED, RECOVERED, STILL_WRONG)

_TRIGGERED_OUTCOMES = {"checkpoint_at_start", "normal",
                      RECOVERED, STILL_WRONG, HALTED, RECONSTRUCTION_FAILED}


@dataclass
class RateEstimate:
    k: int
    n: int
    rate: float
    ci_low: float
    ci_high: float


def _rate(k: int, n: int) -> RateEstimate:
    if n == 0:
        return RateEstimate(0, 0, float("nan"), float("nan"), float("nan"))
    ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return RateEstimate(k, n, k / n, float(ci.low), float(ci.high))


@dataclass
class RollbackMetrics:
    trigger_rate: RateEstimate
    conditional_recovery: RateEstimate
    end_to_end_recovery: RateEstimate
    n_not_triggered: int
    n_reconstruction_failed: int
    n_recovered: int
    n_still_wrong: int
    n_halted: int


def compute_metrics(outcomes: list[str]) -> RollbackMetrics:
    """`outcomes` is one entry per selected episode: NOT_TRIGGERED,
    RECONSTRUCTION_FAILED, RECOVERED, STILL_WRONG, or HALTED (checkpoint_at_start
    is a sub-case of a normal trigger, already resolved into one of the last
    three before this call). Every input is retained -- nothing is filtered.
    """
    n_selected = len(outcomes)
    n_not_triggered = outcomes.count(NOT_TRIGGERED)
    n_reconstruction_failed = outcomes.count(RECONSTRUCTION_FAILED)
    n_recovered = outcomes.count(RECOVERED)
    n_still_wrong = outcomes.count(STILL_WRONG)
    n_halted = outcomes.count(HALTED)
    assert (n_not_triggered + n_reconstruction_failed + n_recovered
           + n_still_wrong + n_halted) == n_selected, "unaccounted outcome value"

    n_triggered = n_selected - n_not_triggered
    n_attempted = n_triggered - n_reconstruction_failed   # a retry actually ran

    return RollbackMetrics(
        trigger_rate=_rate(n_triggered, n_selected),
        conditional_recovery=_rate(n_recovered, n_attempted),
        end_to_end_recovery=_rate(n_recovered, n_selected),
        n_not_triggered=n_not_triggered,
        n_reconstruction_failed=n_reconstruction_failed,
        n_recovered=n_recovered,
        n_still_wrong=n_still_wrong,
        n_halted=n_halted,
    )
