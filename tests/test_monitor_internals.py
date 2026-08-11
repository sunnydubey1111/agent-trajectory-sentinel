"""Monitor / calibration / escalation internals.

Covers, H25, H26, H27, M12, M13, M16, M17, M18/, M19, M20, M21,
M22, M23, L01, L02,,.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from derail.common import D_TOTAL, Episode, JudgeConfig, Standardizer, rng_for
from derail.monitor import escalation
from derail.monitor.calibration import NullCalibrator
from derail.monitor.grounding_verify import NumericGroundingMonitor, _nums


# -------------------------------------------------------------------
def test_constant_healthy_scale_is_unscaled_not_amplified():
    """A quantity that never varies in healthy data gives no basis for
    counting sigmas, so a deviation is reported raw rather than divided by a
    tiny floor, which would turn a no-information channel into the most
    sensitive one in the system. See DESIGN.md Amendment 6.
    """
    from derail.common import DEGENERATE_EPS, Standardizer, safe_scale
    from derail.monitor.esn import _robust_loc_scale
    from derail.monitor.hybrid import _robust_stats

    assert safe_scale(0.0) == 1.0
    assert safe_scale(2.5) == 2.5                      # real spread untouched

    med, scale = _robust_stats(np.full(50, 3.0))       # zero variance
    assert med == 3.0 and scale == 1.0
    assert (4.0 - med) / scale == 1.0                  # raw deviation, bounded

    loc, sc = _robust_loc_scale(np.full(50, 7.0))
    assert loc == 7.0 and sc == 1.0

    # Quantities with real spread are unaffected.
    _, scale2 = _robust_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.isfinite(scale2) and scale2 > 0.0

    # Standardizer: a constant dim is passed through unscaled; a varying dim
    # still gets the 1e-3 floor it always had.
    class _E:
        def __init__(self, X):
            self.X = X

    X = np.zeros((20, 3))
    X[:, 0] = np.arange(20.0)          # varies
    X[:, 1] = 5.0                      # constant -> unscaled
    X[:, 2] = 1e-6 * np.arange(20.0)   # tiny but real -> floored, as before
    std = Standardizer().fit([_E(X)])
    assert std.std_[1] == 1.0
    assert std.std_[2] == 1e-3
    assert std.std_[0] > 1e-3
    # A first-ever deviation on the constant dim stays visible but bounded.
    z = std.transform(np.array([[0.0, 6.0, 0.0]]))
    assert abs(z[0, 1] - 1.0) < 1e-12
    assert DEGENERATE_EPS < 1e-3


def test_pick_threshold_flags_an_unreachable_false_alarm_budget():
    """With n healthy episodes an empirical threshold cannot beat 1/(n+1): a
    fresh healthy episode exceeds the maximum of n with exactly that
    probability. An unreachable budget must be reported, not silently missed."""
    import warnings

    from derail.evaluation.metrics import (min_calibration_episodes,
                                           pick_threshold)

    assert min_calibration_episodes(0.05) == 19
    assert min_calibration_episodes(0.10) == 9

    streams = [np.array([float(i)]) for i in range(10)]   # n=10 < 19
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pick_threshold(streams, fa_budget=0.05)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    # Enough episodes -> no warning, and the default rule is unchanged.
    streams20 = [np.array([float(i)]) for i in range(20)]
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        th = pick_threshold(streams20, fa_budget=0.05)
    assert not caught2
    assert th == float(np.quantile(np.arange(20.0), 0.95, method="higher"))


def test_null_calibrator_does_not_escalate_a_degenerate_null():
    cal = NullCalibrator().fit(np.zeros(120))
    assert float(cal.confidence(0.0)) <= 0.05      # score AT the null: not anomalous
    assert float(cal.confidence(1.0)) >= 0.95      # strictly above: flagged
    assert float(cal.confidence(-1.0)) <= 0.05


def test_null_calibrator_rejects_non_finite():
    cal = NullCalibrator().fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        cal.confidence(np.nan)
    with pytest.raises(ValueError):
        cal.confidence(np.inf)


def test_null_calibrator_is_monotone_and_bounded():
    cal = NullCalibrator().fit(rng_for(0, "cal").normal(size=200))
    grid = cal.confidence(np.linspace(-3, 5, 40))
    assert np.all(np.diff(grid) >= 0)
    assert grid.min() >= 0.0 and grid.max() < 1.0


# ------------------------------------------------------------- / M17
def _ep(eid, healthy, tau=None, T=30):
    return Episode(X=rng_for(0, eid).normal(size=(T, D_TOTAL)), episode_id=eid,
                   is_healthy=healthy, failure_class=None if healthy else "looping",
                   tau=tau, t_fail=None if healthy else T - 1,
                   severity=None if healthy else 0.5)


def test_pre_onset_injected_halt_is_wrongful_and_penalised():
    T, tau = 30, 20
    inj = _ep("inj", False, tau=tau, T=T)
    # An alarm at step 5 (before tau=20) forces a pre-onset halt.
    scores = {"inj": np.where(np.arange(T) >= 5, 5.0, 0.1)}
    out = escalation.run_policy("halt_on_alarm", [inj], scores, None,
                                theta_soft=1.0, judge=JudgeConfig(), seed=1)[0]
    assert out.halted_at == 5 and not out.detected
    assert out.wrongful_halt, "pre-onset injected halt not flagged wrongful"
    from derail.common import COST_STEP
    assert out.cost == COST_STEP * T, "pre-onset halt did not pay the redo penalty"
    summ = escalation.summarize_policy([out])
    assert summ["wrongful_halt_rate"] == 1.0
    assert summ["early_injected_halt_rate"] == 1.0


def test_judge_verdict_is_order_independent():
    inj = _ep("inj2", False, tau=5, T=20)
    j = JudgeConfig()
    # The verdict at a given (episode, step) is keyed by (seed, id, t): calling
    # in a different order / subset must not change it.
    forward = [escalation.judge_verdict(inj, t, j, seed=7) for t in range(20)]
    backward = [escalation.judge_verdict(inj, t, j, seed=7)
                for t in reversed(range(20))][::-1]
    assert forward == backward


# -------------------------------------------------------------------
def test_demo_numeric_verifier_checks_text_before_observing_same_turn_results():
    from derail.experiments import demo
    src = inspect.getsource(demo)
    i_check = src.index("gcheck.check_step(out[\"text\"])")
    i_obs = src.index("gcheck.observe_tool_results(", i_check - 2000)
    # The check_step on this step's text must appear BEFORE the observe of this
    # step's results in the loop body (causal order).
    assert i_check < src.index("gcheck.observe_tool_results(\n", i_check), \
        "same-turn results are observed before the text is checked"


# ------------------------------------------------------------- / M13
def test_numeric_grounding_signs_preserved():
    assert _nums("charge of $200") == [200.0]
    assert _nums("refund of -$200") == [-200.0]
    assert _nums("shown $-200") == [-200.0]
    m = NumericGroundingMonitor()
    m.start_episode()
    m.observe_tool_results("[refund -> -$200]")
    assert 200.0 in m.check_step("You were charged $200.")


def test_numeric_grounding_provenance_is_monotone():
    m = NumericGroundingMonitor()
    m.start_episode()
    vals = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]
    m.observe_tool_results(" ".join(f"[t -> ${v:g}]" for v in vals))
    total = round(sum(vals), 2)
    assert m.check_step(f"Total ${total:g}.") == []
    m.observe_tool_results("[t -> $999]")            # 8th, unrelated value
    assert m.check_step(f"Total ${total:g}.") == [], \
        "a grounded total became ungrounded after a new value (M12)"


# -------------------------------------------------------------------
def test_esn_ridge_trains_the_first_scored_transition():
    from derail.monitor import esn
    src = inspect.getsource(esn.ESNEnsembleMonitor.fit)
    assert "Z[_WASHOUT - 1 : T - 1]" in src            # train from washout-1
    assert "Z[_WASHOUT : T - 1]" not in src            # not the old off-by-one


# -------------------------------------------------------------------
def test_esn_split_is_independent_of_the_model_seed():
    from derail.monitor import esn
    src = inspect.getsource(esn.ESNEnsembleMonitor.fit)
    assert "_MONITOR_SPLIT_SEED" in src
    assert 'rng_for(self.seed, "esn", "split")' not in src


# -------------------------------------------------------------------
def test_seq_baselines_share_the_esn_calibration_contract():
    """The trained sequence baselines must be calibrated identically to the
    ESN they are benchmarked against, or the published GRU/LSTM/TCN-vs-ESN
    comparison measures calibration luck instead of architecture.

    Three quantities decide this, and all three must be the SAME OBJECTS as
    the ESN's, not local copies that can drift: the fit/held split seed, the
    residual-std guard, and the robust location-scale estimator.
    """
    from derail.monitor import esn, seq_baselines

    assert seq_baselines._robust_loc_scale is esn._robust_loc_scale
    assert seq_baselines._MONITOR_SPLIT_SEED == esn._MONITOR_SPLIT_SEED
    assert seq_baselines._WASHOUT == esn._WASHOUT

    src = inspect.getsource(seq_baselines._NextStepMonitor.fit)
    assert "_MONITOR_SPLIT_SEED" in src, "per-model calibration split"
    assert 'rng_for(self.seed, self.name, "split")' not in src
    assert "DEGENERATE_EPS" in src, "missing Amendment 6 guard on sigma_err"


# -------------------------------------------------------------------
def test_seq_baseline_constant_healthy_dim_is_unscaled_not_amplified():
    """Behavioural half of the above: a dim with zero healthy variation is
    left unscaled (1.0), not divided by the 1e-3 floor."""
    from derail.monitor.seq_baselines import LinearARMonitor

    def mk(idx, T=40):
        rng = rng_for(0, "seq-degen", idx)
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
        X[:, 33] = 0.0          # zero-variation dim inside the u channel
        return Episode(X=X, episode_id=f"s{idx}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [mk(i) for i in range(24)]
    mon = LinearARMonitor(Standardizer().fit(train), seed=0)
    mon.fit(train)
    assert mon._sigma_err[33] == 1.0, (
        f"degenerate dim floored to {mon._sigma_err[33]!r} instead of left "
        f"unscaled — Amendment 6 guard missing")
    assert np.all(mon._sigma_err > 0.0)


# -------------------------------------------------------------------
def test_hmt_refit_does_not_accumulate_previous_dataset():
    from derail.monitor.hmt_esn import HMTESNMonitor
    src = inspect.getsource(HMTESNMonitor.fit)
    assert "reset_accumulators" in src


# -------------------------------------------------------------------
def test_hmt_uses_the_shared_fit_held_split():
    """HMT must calibrate on the SAME held-out episodes as the ESN baseline
    it is A/B'd against. A private 85/15 partition is worth +0.121 episode AUC
    on the real arm on its own — enough to manufacture a passing kill-switch
    verdict out of a monitor that is actually behind the baseline.
    """
    from derail.monitor.hmt_esn import HMTESNMonitor
    src = inspect.getsource(HMTESNMonitor.fit)
    assert "_MONITOR_SPLIT_SEED" in src
    assert 'rng_for(0, "hmt", "split")' not in src


# -------------------------------------------------------------------
def test_hmt_constant_healthy_dim_is_unscaled_not_amplified():
    """The Amendment 6 degenerate-scale contract, enforced in hmt_esn.py.

    A dim the banks predict exactly on healthy data has residual std 0.
    Dividing by the 1e-3 floor would amplify its first deviation ~1000x and
    make a no-information dim the most sensitive one in the monitor; it must
    be left unscaled (sigma_err == 1.0) instead.
    """
    from derail.monitor.hmt_esn import HMTESNMonitor

    def mk(idx, T=40):
        rng = rng_for(0, "hmt-degen", idx)
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
        X[:, 33] = 0.0          # a dim with zero healthy variation (u channel)
        return Episode(X=X, episode_id=f"d{idx}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [mk(i) for i in range(24)]
    mon = HMTESNMonitor(Standardizer().fit(train), channels=("u",),
                        leak_rates=(0.3,), n_layers=1, K=4, seed=0)
    mon.fit(train)
    banks = list(mon._all_banks())
    assert banks, "no banks constructed"
    for b in banks:
        # local col 1 of the u slice [32, 36) is global dim 33
        assert b.sigma_err[1] == 1.0, (
            f"degenerate dim was floored to {b.sigma_err[1]!r}, not left "
            f"unscaled -- Amendment 6 guard missing")
        assert np.all(b.sigma_err > 0.0)


# -------------------------------------------------------------------
def test_hmte_null_does_not_drift():
    from derail.monitor.esn import HMTE_ESN_M_Monitor

    def mk(idx, perturb=None, T=40):
        rng = rng_for(0, "h27t", idx)
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
            if perturb and t > perturb:
                X[t] = X[t] + 1.5 * rng.standard_normal(D_TOTAL)
        return Episode(X=X, episode_id=f"e{idx}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [mk(i) for i in range(30)]
    mon = HMTE_ESN_M_Monitor(Standardizer().fit(train), seed=0)
    mon.fit(train)
    mon.start_episode()
    healthy = [mon.score_step(x) for x in mk(500).X]
    mon.start_episode()
    perturbed = [mon.score_step(x) for x in mk(501, perturb=20).X]
    # The healthy CUSUM stays far below the perturbed one; the old uncentered
    # distance (mean ~2.9 into a k=0.5 CUSUM) drifted up on every healthy step,
    # so the null tracked the signal. Now it is a small fraction of it.
    assert healthy[-1] < 0.2 * perturbed[-1]


# -------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"K": 0}, {"ewma_alpha": 2.0},
                                {"reservoir_size": 0}, {"leak_rate": 0.0},
                                {"density": 1.5}])
def test_esn_constructor_validates_ranges(kw):
    from derail.monitor.esn import ESNEnsembleMonitor
    with pytest.raises(ValueError):
        ESNEnsembleMonitor(None, **kw)


# -------------------------------------------------------------------
def test_supervised_hybrid_resets_stale_coefficients():
    from derail.monitor.hybrid import HybridLogistic
    src = inspect.getsource(HybridLogistic.fit_supervised)
    # The reset must happen BEFORE the degenerate-label early return.
    reset_i = src.index("self.coef_ = np.array([0.5, 0.5])")
    ret_i = src.index("return  # degenerate labels")
    assert reset_i < ret_i, "coefficients not reset before the early return"


# ------------------------------------------------------------- M16
def test_grounding_z_is_capped_and_finite():
    from derail.monitor.grounding import GroundingMonitor
    src = inspect.getsource(GroundingMonitor.z_dims)
    assert "_Z_CAP" in src and "isfinite" in src


# -------------------------------------------------------------------
def test_beta_disagreement_is_ablated():
    from derail.experiments.run_ablation import SWEEPS
    assert "beta_disagreement" in SWEEPS
    assert 0.0 in SWEEPS["beta_disagreement"]


def test_judge_is_disclosed_as_stipulated():
    from derail.monitor import escalation as esc
    assert "STIPULATED" in (esc.__doc__ or "")
    # L8: the disclosure must also carry the measured rates, so a reader of the
    # module cannot take 0.90/0.02 for a measurement.
    assert "0.548" in (esc.__doc__ or "") and "0.057" in (esc.__doc__ or "")


def test_judge_defaults_are_unchanged_by_the_l8_measurement():
    """Measuring the judge must not silently move any published number."""
    from derail.common import JudgeConfig
    assert JudgeConfig().p_detect == 0.90
    assert JudgeConfig().p_false == 0.02


def test_judge_override_is_env_gated_and_validated(monkeypatch):
    from derail.experiments.run_experiment import _judge_config

    monkeypatch.delenv("AGENTWATCH_JUDGE_P_DETECT", raising=False)
    monkeypatch.delenv("AGENTWATCH_JUDGE_P_FALSE", raising=False)
    assert _judge_config().p_detect == 0.90        # default path untouched

    monkeypatch.setenv("AGENTWATCH_JUDGE_P_DETECT", "0.548")
    monkeypatch.setenv("AGENTWATCH_JUDGE_P_FALSE", "0.057")
    judge = _judge_config()
    assert (judge.p_detect, judge.p_false) == (0.548, 0.057)

    monkeypatch.setenv("AGENTWATCH_JUDGE_P_DETECT", "1.5")
    with pytest.raises(SystemExit):
        _judge_config()


def test_results_root_override_redirects_and_stays_in_repo(monkeypatch):
    """A sensitivity run must be redirectable, and cannot escape the repo."""
    from derail.experiments import run_experiment as rx

    monkeypatch.delenv("AGENTWATCH_RESULTS_ROOT", raising=False)
    assert rx._results_root().name == "results"

    monkeypatch.setenv("AGENTWATCH_RESULTS_ROOT", "results/_sensitivity")
    root = rx._results_root()
    assert root.name == "_sensitivity"
    repo = Path(rx.__file__).resolve().parents[2]
    assert repo in root.parents, "relative override escaped the repo root"


def test_judge_override_cannot_write_to_the_publication_path(monkeypatch):
    """Measured-judge runs are sensitivity arms; results/ keeps the published
    stipulated-judge numbers (the run must abort before any file is written)."""
    from derail.experiments import run_experiment as rx

    monkeypatch.setenv("AGENTWATCH_JUDGE_P_DETECT", "0.548")
    monkeypatch.delenv("AGENTWATCH_RESULTS_ROOT", raising=False)

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the run proceeded past the guard")

    monkeypatch.setattr(rx, "_set_results_dirs", _boom)
    with pytest.raises(SystemExit, match="publication path"):
        rx.main([])


# ------------------------------------------------- deterministic verification
def test_checks_never_read_the_hidden_world():
    """The checks must be deployable, so they may only use what the agent
    observed. Reading the task's generating world would make them an oracle,
    not a check — and would silently inflate every reported detection rate.

    Enforced on the parsed module (imports and names actually referenced), not
    on the source text, so prose describing the rule cannot trip or satisfy it.
    """
    import ast
    import inspect

    from derail.verify import checks

    tree = ast.parse(inspect.getsource(checks))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("demo" in m or "generator" in m or "harness" in m
                   for m in imported), imported

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("_make_world", "_demo_expected_total", "expected_total"):
        assert forbidden not in used, f"checks must not reference {forbidden}"


def test_total_consistency_catches_miscombination_without_ground_truth():
    """Every input grounded but the total wrong: a dropped line item, or a
    spurious operation in what the agent asked the calculator to compute."""
    from derail.verify.checks import BOOKING_SPEC, total_consistency

    base = [{"text": '[lookup_flight({"i": %d}) -> $%d]' % (i, 100 * i)}
            for i in range(1, 5)]                      # 100+200+300+400
    base += [{"text": '[lookup_hotel({"c": "%s"}) -> $10/night]' % c}
             for c in "xyz"]                           # 2*(10+10+10)
    good = base + [{"text": "Total: $1060 USD."}]
    assert not total_consistency(good, BOOKING_SPEC).findings

    spurious = base + [{"text": "Total: $3060 USD."}]   # flights x3
    assert total_consistency(spurious, BOOKING_SPEC).findings

    dropped = base[:3] + base[4:] + [{"text": "Total: $660 USD."}]
    r = total_consistency(dropped, BOOKING_SPEC)
    assert not r.findings, "consistent over what it saw -> coverage's job"


def test_halting_and_repairing_are_exclusive_responses_to_an_alarm():
    """The halt toggle chooses which response an alarm gets.

    Halting means the operator wants the run stopped for inspection, so
    nothing may repair underneath them; with halting off an alarm is recovered
    from instead, capped at one retry so the cost stays about one model call
    whatever the failure class was.
    """
    import inspect

    from derail.experiments import demo

    src = inspect.getsource(demo.run_demo_episode)
    assert "if alarmed and st.halt_on_alarm:" in src, \
        "halting must take precedence over repairing"
    assert 'st.status = "halted"' in src
    # The repair gate is reached only after the halt branch has returned, so
    # it must not re-test the toggle: that is what made them exclusive.
    gate = src.split("if alarmed and st.halt_on_alarm:", 1)[1]
    assert "repairable = (alarmed and st.repair_enabled" in gate
    assert "not st.alarm_repair_used" in gate, "the retry must stay capped"


def test_a_run_stuck_on_its_tools_is_told_to_stop_calling_them():
    """"Re-check your work" is the wrong repair for a stuck run.

    It invites another call to the tool that is feeding the loop, which is how
    an injected loop trap survived its own repair and kept scoring above the
    alarm line.
    """
    from derail.experiments import demo
    from derail.intervene.rollback import repair_message

    progressing = [
        {"action": "tool_call", "error": False,
         "text": '[lookup_flight({"i": 1}) -> $100]'},
        {"action": "tool_call", "error": False,
         "text": '[lookup_hotel({"c": "x"}) -> $10/night]'}]
    assert not demo._stuck_on_tools(progressing)

    erroring = [{"action": "tool_call", "error": True,
                 "text": '[lookup_flight({"i": 1}) -> Error: 503]'}] * 2
    assert demo._stuck_on_tools(erroring)

    repeating = [{"action": "tool_call", "error": False,
                  "text": '[lookup_flight({"i": 1}) -> please retry]'}] * 2
    assert demo._stuck_on_tools(repeating), "same call twice is a loop"

    assert not demo._stuck_on_tools(progressing[:1]), "one step is no evidence"

    hint = repair_message("unstick", [])
    assert "not call it again" in hint
    assert "re-check" not in hint.lower(), "must not invite another tool call"


def test_a_spent_repair_against_a_dead_tool_layer_escalates():
    """Repair first, escalate second.

    Once the retry is spent and the circuit breaker is still open, no further
    step can help: the agent keeps calling a tool that cannot answer and every
    refusal is another anomalous step. A measured loop trap burned 16 such
    steps and drove the score from 0.17 to 97 before its budget ran out.
    """
    import inspect

    from derail.experiments import demo

    src = inspect.getsource(demo.run_demo_episode)
    assert "escalated_tool_layer_down" in src
    # The escalation must come AFTER the repair has been offered, or the
    # alarm would never get its retry at all.
    spent = src.split("repairable = (alarmed and st.repair_enabled", 1)[1]
    assert "st.alarm_repair_used and st.breaker_open" in spent
    assert "not repairable" in spent, "must not pre-empt an available retry"


def test_every_exit_resolves_an_in_flight_repair():
    """A stopped run must never still read as "repairing".

    Only the halting path resolved it at first, so a repair that ran out of
    step budget left the UI showing a finished episode as still recovering.
    """
    import inspect

    from derail.experiments import demo

    src = inspect.getsource(demo.run_demo_episode)
    exits = ("budget_exhausted", "stopped_by_user", "agent_error",
             "halted_by_watchdog")
    for reason in exits:
        assert reason in src, f"{reason} exit missing"
    # One resolution per terminal path.
    assert src.count('st.repair_state = "repair_failed"') >= len(exits)


def test_no_unreachable_code_in_the_live_episode_loop():
    """Statements after a `continue`/`return`/`break` never run.

    The episode loop mixes control flow with state the rest of the run depends
    on, so a misplaced branch can silently strand a block: an early edit left
    the injection-arming code after a `continue`, and injections stopped
    arming while every test still passed.
    """
    import ast
    import inspect

    from derail.experiments import demo

    tree = ast.parse(inspect.getsource(demo.run_demo_episode))
    stranded = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, (ast.Continue, ast.Break, ast.Return)):
                    stranded.append(
                        f"line {block[i + 1].lineno} follows "
                        f"{type(stmt).__name__.lower()} on line {stmt.lineno}")
    assert not stranded, "unreachable code in run_demo_episode: " + \
        "; ".join(stranded)


def test_tool_contract_rejects_a_malformed_result_at_its_own_step():
    """Corruption that destroys a result's shape is caught where it arrives,
    without a null, a threshold, or an answer to check against."""
    from derail.verify.checks import BOOKING_SPEC, tool_contract

    clean = [{"text": '[lookup_flight({"i": 1}) -> $361]'},
             {"text": '[lookup_hotel({"c": "x"}) -> $120/night]'},
             {"text": '[get_weather({"c": "x"}) -> sunny, 21C]'},
             {"text": '[calculator({"expression": "1+1"}) -> 2]'}]
    assert not tool_contract(clean, BOOKING_SPEC), "healthy shapes are legal"

    garbled = list(clean)
    garbled[1] = {"text": '[lookup_hotel({"c": "x"}) -> Ã¢12Â§0/night $214]'}
    findings = tool_contract(garbled, BOOKING_SPEC)
    assert [f.step for f in findings] == [1], "reported at the corrupted step"
    assert findings[0].check == "tool_contract"
    assert "lookup_hotel" in findings[0].terse


def test_tool_contract_is_silent_on_corruption_that_keeps_a_legal_shape():
    """The honest boundary: a price changed to another price is well formed,
    and separating it from a real one needs a reference this layer lacks."""
    from derail.verify.checks import BOOKING_SPEC, tool_contract

    plausible = [{"text": '[lookup_flight({"i": 1}) -> $605]'}]
    assert not tool_contract(plausible, BOOKING_SPEC)


def test_tool_contract_does_not_fire_on_a_declared_error_or_free_text_tool():
    """An error is a legal outcome and `get_weather` returns prose; neither is
    a contract violation, or the check would alarm on healthy runs."""
    from derail.verify.checks import BOOKING_SPEC, tool_contract

    steps = [{"text": '[lookup_flight({"i": 1}) -> No route found between '
                      'those cities.]'},
             {"text": '[calculator({"expression": "x"}) -> Error: only basic '
                      'arithmetic is supported.]'},
             {"text": '[get_weather({"c": "x"}) -> Rain showers, 12C, wind '
                      '9 km/h]'}]
    assert not tool_contract(steps, BOOKING_SPEC)


def test_requeried_item_is_priced_once():
    """An agent re-querying the same item is not buying it twice."""
    from derail.verify.checks import BOOKING_SPEC, total_consistency

    steps = [{"text": '[lookup_hotel({"c": "x"}) -> $50/night]'},
             {"text": '[lookup_hotel({"c": "x"}) -> $50/night]'},
             {"text": "Total: $100 USD."}]
    assert total_consistency(steps, BOOKING_SPEC).recomputed_total == 100.0


def test_coverage_and_consistency_are_complementary():
    """Neither check subsumes the other: consistency catches a bad
    combination of what was looked up, coverage catches work never done."""
    from derail.verify.checks import BOOKING_SPEC, verify

    only_three_legs = [{"text": '[lookup_flight({"i": %d}) -> $100]' % i}
                       for i in range(3)]
    only_three_legs += [{"text": '[lookup_hotel({"c": "%s"}) -> $10/night]' % c}
                        for c in "xyz"]
    only_three_legs += [{"text": '[get_weather({"c": "x"}) -> sunny]'},
                        {"text": "Total: $360 USD."}]
    findings = verify(only_three_legs, BOOKING_SPEC).findings
    kinds = {f.check for f in findings}
    assert kinds == {"required_coverage"}, kinds


# ------------------------------------------------------ intervention loop
def test_repair_hint_cannot_carry_the_oracle_answer():
    """The `specific` rung is only sound if the hint derives from the check
    output alone. A string test for the true total cannot serve here, since the
    recomputed total legitimately equals it whenever the agent looked every
    figure up correctly and merely mis-added; the guarantee is enforced on the
    signature and the module's imports instead."""
    import ast
    import inspect

    from derail.intervene import rollback

    assert list(inspect.signature(rollback.repair_message).parameters) == \
        ["rung", "findings"]

    tree = ast.parse(inspect.getsource(rollback))
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "_demo_expected_total" not in used
    assert "expected_total" not in used, \
        "the rollback path must never see the oracle"


def test_rollback_resumes_after_the_last_fact_gathering_step():
    from derail.intervene.rollback import rollback_step

    trace = [
        {"text": '[lookup_flight({"i": 1}) -> $100]'},
        {"text": '[lookup_hotel({"c": "x"}) -> $10/night]'},
        {"text": '[calculator({"expression": "100+20"}) -> 120]'},
        {"text": "Total: $999 USD."},
    ]
    # Resume just after the hotel lookup: the calculator step and the answer
    # are what the agent must redo.
    assert rollback_step(trace) == 2


def test_rebuilt_history_replays_calls_and_their_results_in_order():
    from derail.intervene.rollback import rebuild_history

    class _B:
        def reset(self, task):
            self.history = [{"role": "user", "content": task}]

    trace = [{"text": 'ok [lookup_flight({"i": 1}) -> $100]'},
             {"text": '[lookup_hotel({"c": "x"}) -> $10/night]'}]
    b = _B()
    rebuild_history(b, "T", trace, 2)
    assert [m["role"] for m in b.history] == \
        ["user", "assistant", "tool", "assistant", "tool"]
    assert b.history[1]["tool_calls"][0]["function"]["arguments"] == {"i": 1}
    assert b.history[4]["content"] == "$10/night"


def test_demo_serves_the_deterministic_check_alongside_the_monitor():
    """The demo's own booking task produces organic failures the behavioural
    monitor does not separate, so the live UI must also report the
    deterministic verification, and must report it from observed tool results
    rather than the task's ground-truth total."""
    from pathlib import Path

    from derail.experiments import demo

    src = inspect.getsource(demo.run_demo_episode)
    # The checks parse tool calls out of a step's text, so they must read the
    # telemetry-shaped list. st.steps is UI-shaped: its `text` holds only the
    # agent's prose, with tool calls in a separate field, so verifying against
    # it silently sees zero tool calls.
    assert "verify(tele, BOOKING_SPEC)" in src
    assert "verify(st.steps" not in src, "checks must not read the UI list"
    assert "st.check_verdict" in src

    snap = inspect.getsource(demo.DemoState.snapshot)
    for field in ("check_verdict", "check_findings", "check_recomputed"):
        assert field in snap, f"{field} missing from the served snapshot"

    html = (Path(demo.__file__).parent / "demo.html").read_text("utf-8")
    assert "consistency-check" in html, "the UI never renders the check"
    assert "check_verdict" in html


def test_demo_real_excludes_vacuous_episodes_from_the_healthy_null():
    """An episode shorter than the washout scores 0.0 at every step because no
    step was ever scored. Counting it as "healthy, no false alarm" reports the
    washout back as evidence: 3 of 8 live healthy runs ended at T=2 and
    returned exactly 0.0. Such runs must be excluded and the exclusion
    reported, not silently averaged in.
    """
    from derail.harness import demo_real
    from derail.monitor.esn import _WASHOUT

    assert demo_real.MIN_SCOREABLE_T == _WASHOUT + 1
    assert not demo_real._is_scoreable(_WASHOUT)
    assert demo_real._is_scoreable(_WASHOUT + 1)

    src = inspect.getsource(demo_real.fit_monitor)
    assert "_is_scoreable" in src, "the null is not vacuity-filtered"
    assert "vacuous-episode policy" in src, "the exclusion must be reported"


# -------------------------------------------------------------------
def test_demo_real_injected_episodes_are_committed_and_detectable():
    """The demo's injected side must exist OFFLINE, not only as live runs.

    Without committed injected episodes a detection regression can only be
    caught by someone happening to run `--demo` against a live model, which is
    not a test. This scores the committed corpus end to end: fit the served
    monitor on the healthy null, take its threshold, and require that the
    behavioural failure classes clear it while the healthy episodes do not.

    Content classes (`wrong_document`, `malformed_json`) are deliberately NOT
    asserted on: they are the documented blind spot of the behavioural
    channel, caught by the content gate in `experiments/demo.py` instead.
    """
    import json

    import numpy as np

    from derail.harness import demo_real
    from derail.telemetry.adapter import load_trace_jsonl

    src = demo_real.TRACES
    man_path = src / "manifest.json"
    if not man_path.exists():
        pytest.skip(f"{src.name} not collected")
    manifest = json.loads(man_path.read_text("utf-8"))
    injected = [e for e in manifest if e["failure_class"] is not None
                and demo_real._is_scoreable(e["T"])]
    if not injected:
        pytest.skip(f"{src.name} has no committed injected episodes")

    mon, theta, _ = demo_real.fit_monitor()
    behavioural = {"looping", "rate_limit"}
    scored = {}
    for e in injected:
        ep = load_trace_jsonl(src / e["file"], episode_id=e["episode_id"],
                              tau=e["tau"], failure_class=e["failure_class"],
                              severity=0.5, use_sentence_transformers=False,
                              extended=True)
        demo_real._drop_machine_nuisance(ep.X)
        scored.setdefault(e["failure_class"], []).append(
            float(np.max(mon.score_episode(ep))))

    beh = [v for fc, vs in scored.items() if fc in behavioural for v in vs]
    assert beh, "no behavioural injected episodes committed"
    det = sum(v > theta for v in beh)
    assert det >= 0.5 * len(beh), (
        f"behavioural detection collapsed: {det}/{len(beh)} above "
        f"theta={theta:.2f}; scores={sorted(round(v, 2) for v in beh)}")


# -------------------------------------------------------------------
def test_demo_null_holds_only_runs_that_did_the_task_and_got_it_right():
    """A healthy null must hold runs that performed the task AND answered it
    correctly. Task-incomplete runs are strongly anomalous to the monitor, and
    wrong-total runs are outright failures; either inflates the null and lifts
    the threshold above where real failures live (DESIGN.md Amendment 7)."""
    from pathlib import Path

    from derail.experiments import demo

    src = inspect.getsource(demo.fit_monitor)
    assert "_is_task_complete" in src, "the null is not completeness-filtered"
    assert "_is_answer_correct" in src, "the null is not correctness-filtered"
    assert "healthy-null policy" in src, "the exclusions must be reported"

    # Correctness is recomputed, never read from the manifest's `success`
    # field, which is a substring test.
    assert "_stated_total" in inspect.getsource(demo._is_answer_correct)
    assert demo._demo_seed_of("demo-healthy-000") == 5000
    assert demo._demo_seed_of("demo-healthy-p-019") == 7019
    assert demo._demo_seed_of("not-a-demo-trace") is None

    d = Path(demo.__file__).resolve().parents[2] / "traces" / "demo7b_scoped"
    complete = d / "demo-healthy-000.jsonl"
    if complete.exists():
        # The policy must actually discriminate on this corpus.
        entries = [p for p in d.glob("demo-healthy-*.jsonl")]
        flags = [demo._is_task_complete(p) for p in entries]
        assert any(flags) and not all(flags), \
            "completeness filter excludes nothing or everything"


def test_research_task_null_is_completeness_filtered():
    """Both demo engines apply the same healthy-null policy. The research task
    has no computable ground-truth answer, so only completeness is checkable
    there; correctness filtering is a stated limitation, not an omission."""
    from derail.harness import demo_real
    from derail.verify.checks import RESEARCH_SPEC, required_coverage

    src = inspect.getsource(demo_real.fit_monitor)
    assert "required_coverage" in src and "RESEARCH_SPEC" in src
    assert "correctness is not" in src, "the limitation must be stated"

    complete = [{"text": '[arxiv_search({"q": "a"}) -> A]'},
                {"text": '[arxiv_search({"q": "b"}) -> B]'},
                {"text": '[wikipedia_search({"q": "a"}) -> A]'},
                {"text": '[wikipedia_search({"q": "b"}) -> B]'},
                {"text": '[web_search({"q": "c"}) -> C]'},
                {"text": '[python({"code": "print(1)"}) -> 1]'}]
    assert not required_coverage(complete, RESEARCH_SPEC)
    assert required_coverage(complete[1:], RESEARCH_SPEC)


def test_demo_repairs_a_failing_run_and_shows_it():
    """The demo closes the loop: when the checks reject an answer the agent is
    rewound to its last fact-gathering step and asked again with the finding,
    capped at one attempt, and the UI reports the outcome."""
    from pathlib import Path

    from derail.experiments import demo

    src = inspect.getsource(demo.run_demo_episode)
    assert "rollback_step(tele, BOOKING_SPEC)" in src
    # `located` names the failing check and no computed value: measured best
    # of the rungs and the only one of the two strongest that cannot hand the
    # agent its own answer back.
    assert 'repair_message("located"' in src
    assert 'repair_message("specific"' not in src, \
        "specific quotes the recomputed total, which is the answer"
    assert "rebuild_history(backend, task, tele, k)" in src
    # Capped per trigger, and the two budgets are separate: an alarm-triggered
    # retry must not consume the attempt the checks would have used, or the
    # weaker signal spends the budget the measured recovery rate belongs to.
    assert "not st.check_repair_used" in src, "checks repair must be capped"
    assert "not st.alarm_repair_used" in src, "alarm repair must be capped"
    assert "st.check_repair_used = True" in src
    assert "st.alarm_repair_used = True" in src
    # The retry needs its own step allowance; without it the rewind consumes
    # the original budget and the episode ends with no answer at all.
    assert "budget = t + 1 + (DEMO_MAX_STEPS - k)" in src
    assert "del tele[k:]" in src, "telemetry must rewind with the agent"

    snap = inspect.getsource(demo.DemoState.snapshot)
    for f in ("repair_state", "repair_from_step", "first_answer",
              "first_check_findings"):
        assert f in snap, f"{f} missing from the served snapshot"

    html = (Path(demo.__file__).parent / "demo.html").read_text("utf-8")
    assert "repair-panel" in html and "repair_state" in html
    assert "Repaired After Rollback" in html


def test_located_rung_states_the_fault_without_the_answer():
    """The check recomputes the total from the agent's own figures, so for a
    run that merely mis-added, that value is the correct answer. `specific`
    passes it on; `located` must name the fault and no computed value, so the
    study can separate localization from supplying the answer."""
    from derail.intervene.rollback import RUNGS, repair_message
    from derail.verify.checks import BOOKING_SPEC, verify

    assert "located" in RUNGS
    steps = [{"text": '[lookup_flight({"i": %d}) -> $%d]' % (i, 100 * i)}
             for i in range(1, 5)]
    steps += [{"text": '[lookup_hotel({"c": "%s"}) -> $10/night]' % c}
              for c in "xyz"]
    steps += [{"text": '[get_weather({"c": "x"}) -> sunny]'},
              {"text": "Total: $9999 USD."}]
    findings = verify(steps, BOOKING_SPEC).findings
    assert findings

    spec_hint = repair_message("specific", findings)
    loc_hint = repair_message("located", findings)
    recomputed = "1060"                      # 100+200+300+400 + 2*(10+10+10)
    assert recomputed in spec_hint, "specific should carry the recomputed value"
    assert recomputed not in loc_hint, "located must not carry any computed value"
    assert "does not match" in loc_hint


def test_repair_rewinds_monitor_and_feature_state_with_the_agent():
    """A rollback that rewinds only the conversation leaves the ESN CUSUM and
    the causal feature accumulators integrating steps the agent no longer has,
    so the post-repair score would describe a history that was discarded."""
    from derail.experiments import demo

    src = inspect.getsource(demo.run_demo_episode)
    i = src.index("del tele[k:]")
    after = src[i:i + 1200]
    for expected in ("st.monitor.start_episode()", "xstate = ExtFeatureState()",
                     "gstate = GrdFeatureState()",
                     "gcheck = NumericGroundingMonitor()"):
        assert expected in after, f"{expected} not reset on rollback"
    # State is not merely reset: the retained steps are replayed so it matches
    # the rewound conversation rather than starting blank mid-episode.
    assert "for prev in tele:" in after
    assert "_score_one_step(st.monitor, prev" in after


def test_qualified_subtotal_is_not_read_as_the_grand_total():
    """A repaired run that answers with a breakdown -- "Total flight cost:
    $2755, hotel cost: $1696, ..." -- states no grand total at all. Reading the
    first labelled figure would grade it on a line item."""
    from derail.experiments.demo import _stated_total as demo_total
    from derail.verify.checks import stated_total

    breakdown = ("Total flight cost: $2755, hotel cost: $1696, "
                 "weather in Lisbon: sunny")
    for fn in (stated_total, demo_total):
        assert fn(breakdown) != 2755.0, "a qualified subtotal is not the total"

    # Genuine totals still read correctly, however they are phrased.
    for text, want in (("The grand total for the trip is $2969 USD.", 2969.0),
                       ("The total cost for the grand tour is $3251.", 3251.0),
                       ("Total: $660 USD.", 660.0)):
        assert stated_total(text) == want
        assert demo_total(text) == want


def test_baseline_will_not_learn_from_a_failing_run():
    """A self-calibrating null must not absorb the runs it exists to catch.
    Admission is gated on the deterministic verdict, which needs no baseline
    and is therefore available from the very first run."""
    from derail.monitor.baseline import RollingBaseline, ServingConfig

    b = RollingBaseline(ServingConfig(model="m", temperature=0.2),
                        fa_budget=0.10)
    for i in range(b.n_required):
        assert b.observe(0.5, checks_passed=True)
    assert b.state == "trusted"

    assert not b.observe(0.5, checks_passed=False)   # failed its checks
    assert not b.observe(99.0, checks_passed=True)   # alarmed on itself
    assert b.rejected == 2


def test_baseline_reports_no_threshold_until_the_budget_is_reachable():
    """Below 1/(n+1) the requested false-alarm rate is arithmetically
    impossible, so the monitor must say it cannot judge rather than judge
    badly — and must not drive autonomous action meanwhile."""
    from derail.monitor.baseline import RollingBaseline, ServingConfig

    b = RollingBaseline(ServingConfig(model="m", temperature=0.2),
                        fa_budget=0.05)
    assert b.n_required == 19
    assert b.threshold() is None and not b.can_act()
    assert b.state == "warming_up"
    for _ in range(19):
        b.observe(0.5, checks_passed=True)
    assert b.threshold() is not None and b.can_act()


def test_changing_the_serving_config_retires_the_baseline():
    """A baseline belongs to one configuration. Model, temperature, prompt,
    tool roster and telemetry version all change what healthy looks like, so a
    changed fingerprint must clear the null rather than leave it confidently
    wrong."""
    from derail.monitor.baseline import RollingBaseline, ServingConfig

    cfg = ServingConfig(model="m", temperature=0.2, prompt="p",
                        tools=("a", "b"), telemetry_schema=4)
    b = RollingBaseline(cfg, fa_budget=0.10)
    for _ in range(b.n_required):
        b.observe(0.5, checks_passed=True)
    assert b.state == "trusted"

    for changed in (ServingConfig(model="other", temperature=0.2, prompt="p",
                                  tools=("a", "b"), telemetry_schema=4),
                    ServingConfig(model="m", temperature=0.9, prompt="p",
                                  tools=("a", "b"), telemetry_schema=4),
                    ServingConfig(model="m", temperature=0.2, prompt="q",
                                  tools=("a", "b"), telemetry_schema=4),
                    ServingConfig(model="m", temperature=0.2, prompt="p",
                                  tools=("a",), telemetry_schema=4),
                    ServingConfig(model="m", temperature=0.2, prompt="p",
                                  tools=("a", "b"), telemetry_schema=5)):
        b2 = RollingBaseline(cfg, fa_budget=0.10)
        b2.extend([0.5] * b2.n_required)
        assert b2.reconfigure(changed), "config change did not retire the null"
        assert b2.n == 0 and not b2.can_act()


def test_demo_installs_a_fingerprinted_self_calibrating_baseline():
    """The demo seeds a rolling baseline from its corpus and keeps learning.
    The fingerprint must cover every axis that changes what healthy looks
    like, or a stale null would survive a configuration change."""
    from derail.experiments import demo
    from derail.monitor.baseline import ServingConfig

    src = inspect.getsource(demo._install_baseline)
    for axis in ("model=MODEL", "temperature=DEMO_TEMPERATURE",
                 "prompt=DEMO_PROMPT_PREFIX", "tools=", "telemetry_schema="):
        assert axis in src, f"{axis} missing from the serving fingerprint"
    assert "extend(calib" in src, "the baseline must be seeded from the corpus"

    # A completed run is offered to it, gated on the deterministic verdict.
    loop = inspect.getsource(demo.run_demo_episode)
    assert "st.baseline.observe(" in loop
    assert "checks_passed=not verdict.failed" in loop

    snap = inspect.getsource(demo.DemoState.snapshot)
    assert "baseline" in snap, "state must be visible to a caller"

    # The prompt really is part of the identity.
    a = ServingConfig(model="m", temperature=0.2, prompt="one")
    b = ServingConfig(model="m", temperature=0.2, prompt="two")
    assert a.fingerprint() != b.fingerprint()
