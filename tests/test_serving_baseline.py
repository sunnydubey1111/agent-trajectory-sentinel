"""The rolling baseline a deployment calibrates against at serving time."""
from __future__ import annotations


from derail.monitor.baseline import (DRIFTING, RECALIBRATING, TRUSTED,
                                     RollingBaseline, ServingConfig)

CFG = ServingConfig(model="qwen2.5:7b", temperature=0.2)


def _fill(b: RollingBaseline, n: int, score) -> None:
    for i in range(n):
        b.observe(score(i) if callable(score) else score, checks_passed=True)


def test_a_stable_deployment_stays_trusted():
    b = RollingBaseline(config=CFG, fa_budget=0.10)
    _fill(b, 80, lambda i: 1.0 + (i % 7) * 0.01)
    assert b.state == TRUSTED


def test_drifting_is_reachable():
    """The state existed but nothing could enter it.

    `realized_fa` scores the window with a threshold fitted to that same
    window, and `pick_threshold` guarantees the in-sample rate is at most the
    budget -- so `> 2 * budget` was unsatisfiable and the branch was dead.
    """
    b = RollingBaseline(config=CFG, fa_budget=0.10)
    _fill(b, 40, lambda i: 1.0 + (i % 7) * 0.01)
    _fill(b, 40, lambda i: 50.0 + i)
    assert b.state == DRIFTING
    assert b.drift_fa() > 2 * b.fa_budget


def test_the_in_sample_rate_alone_can_never_report_drift():
    """Pins why the fix had to change the measurement, not the threshold."""
    b = RollingBaseline(config=CFG, fa_budget=0.10)
    _fill(b, 40, lambda i: 1.0 + (i % 7) * 0.01)
    _fill(b, 40, lambda i: 50.0 + i)
    assert b.realized_fa() <= b.fa_budget, (
        "in-sample rate exceeded its own budget, which pick_threshold "
        "should make impossible")


def test_drift_counts_the_runs_guarded_admission_refuses():
    """Those are the runs that most indicate drift, and they never enter the
    null -- so a rate computed from the null alone is blind to them."""
    b = RollingBaseline(config=CFG, fa_budget=0.10, admit_alarming=False)
    _fill(b, 40, lambda i: 1.0 + (i % 7) * 0.01)
    _fill(b, 40, lambda i: 50.0 + i)
    assert b.rejected > 0, "no run was refused, so the test proves nothing"
    assert b.drift_fa() > 2 * b.fa_budget


def test_drift_is_unknown_before_enough_runs_are_judged():
    b = RollingBaseline(config=CFG, fa_budget=0.10)
    assert b.drift_fa() is None
    _fill(b, 3, 1.0)
    assert b.drift_fa() is None


# --------------------------------------------------------- fingerprinting
def test_changing_the_monitor_retires_the_baseline():
    """A threshold belongs to the scorer as much as to the system scored.

    Without the monitor in the fingerprint, changing reservoir size or K left
    the stored theta in place, confidently wrong on a new score scale.
    """
    b = RollingBaseline(config=ServingConfig(model="m", temperature=0.2,
                                             monitor="esn:R=128,K=3"))
    _fill(b, 40, 1.0)
    assert b.state == TRUSTED
    retired = b.reconfigure(ServingConfig(model="m", temperature=0.2,
                                          monitor="esn:R=50,K=3"))
    assert retired and b.n == 0 and b.state == RECALIBRATING


def test_an_unchanged_configuration_keeps_the_baseline():
    cfg = ServingConfig(model="m", temperature=0.2, monitor="esn:R=128")
    b = RollingBaseline(config=cfg)
    _fill(b, 40, 1.0)
    assert not b.reconfigure(ServingConfig(model="m", temperature=0.2,
                                           monitor="esn:R=128"))
    assert b.n == 40


def test_reconfiguring_clears_the_drift_record_too():
    b = RollingBaseline(config=ServingConfig(model="m", temperature=0.2))
    _fill(b, 40, lambda i: 1.0 + (i % 7) * 0.01)
    _fill(b, 40, lambda i: 50.0 + i)
    assert b.state == DRIFTING
    b.reconfigure(ServingConfig(model="m2", temperature=0.2))
    assert b.drift_fa() is None, "drift carried across a retired baseline"


def test_snapshot_publishes_both_rates():
    b = RollingBaseline(config=CFG)
    _fill(b, 40, 1.0)
    snap = b.snapshot()
    assert "realized_fa" in snap and "drift_fa" in snap
