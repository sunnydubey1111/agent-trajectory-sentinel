"""Evaluation validity.

The protocol primitives that keep a reported number honest: label-independent
folds, one scoring rule per fold, family-wise error control, and length
confounds reported rather than assumed away. Study integration is checked by
asserting the source no longer contains the defective construct, plus focused
numeric tests on the shared primitives.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from derail.common import D_TOTAL, D_TOTAL_EXT, D_TOTAL_GRD, Episode
from derail.evaluation import protocol
from derail.evaluation.metrics import length_confound_report


def _ep(eid, healthy, width=D_TOTAL, T=6):
    rng = np.random.default_rng(abs(hash(eid)) % (2**32))
    return Episode(X=rng.normal(size=(T, width)), episode_id=eid,
                   is_healthy=healthy,
                   failure_class=None if healthy else "looping",
                   tau=None if healthy else 2,
                   t_fail=None if healthy else T - 1,
                   severity=None if healthy else 0.5)


# ----------------------------------------------- cross-fitting and folds
def test_fold_assignment_is_label_independent():
    eps = [_ep(f"h{i}", True) for i in range(10)] + \
          [_ep(f"inj{i}", False) for i in range(10)]
    folds = protocol.assign_folds(eps, k=2, salt="s")
    # Reordering, and flipping which ids are "healthy", must not change a
    # given id's fold - the fold depends on the id alone.
    assert folds == protocol.assign_folds(list(reversed(eps)), k=2, salt="s")


def test_cross_fit_scores_one_rule_for_both_classes():
    class Model:
        coef_ = None
        intercept_ = 0.0

        def fit(self, healthy):
            self.base = float(np.mean([e.X.mean() for e in healthy]))

        def fit_supervised(self, healthy, injected):
            self.bias = float(len(injected))
            self.coef_ = np.array([self.bias])

        def score_episode(self, ep):
            return ep.X.mean(axis=1) + getattr(self, "bias", 0.0)

    train = [_ep(f"tr{i}", True) for i in range(8)]
    test = [_ep(f"h{i}", True) for i in range(8)] + \
           [_ep(f"i{i}", False) for i in range(8)]
    res = protocol.cross_fit_scores(test, Model, train, k=2, salt="x")
    assert set(res.scores) == {e.episode_id for e in test}
    # Every episode in a fold - healthy or injected - is scored by the SAME
    # model (identical bias offset), so the class cannot select the rule.
    by_id = {e.episode_id: e for e in test}
    for fold in range(2):
        ids = [i for i, f in res.folds.items() if f == fold]
        offsets = {round(float(res.scores[i][0] - by_id[i].X.mean(axis=1)[0]), 6)
                   for i in ids}
        assert len(offsets) == 1, f"fold {fold} used >1 scoring rule"


# --------------------------------------------- family-wise error control
def test_holm_bonferroni_controls_family():
    fam = {"a": 0.001, "b": 0.2, "c": 0.04, "d": 0.5}
    holm = protocol.holm_bonferroni(fam)
    assert holm["a"]["reject"] and not holm["b"]["reject"]
    for k, v in holm.items():
        assert v["p_holm"] >= v["p_raw"] - 1e-12
    # single test corrects to itself
    assert abs(protocol.holm_bonferroni({"x": 0.03})["x"]["p_holm"] - 0.03) < 1e-12


def test_benjamini_hochberg_monotone():
    fam = {"a": 0.001, "b": 0.01, "c": 0.5}
    bh = protocol.benjamini_hochberg(fam)
    assert bh["a"]["reject"]
    assert bh["a"]["p_bh"] <= bh["b"]["p_bh"] <= bh["c"]["p_bh"] + 1e-12


# -------------------------------------------------------------------
def test_length_confound_report_detects_length_score_coupling():
    # Healthy episodes whose max-score grows with length: the report must show
    # a strong positive Spearman and still produce a length-matched AUROC.
    eps, scores = [], {}
    for i in range(12):
        T = 4 + i
        e = _ep(f"h{i}", True, T=T)
        eps.append(e)
        scores[e.episode_id] = np.linspace(0, T, T)      # peak grows with T
    for i in range(12):
        T = 4 + (i % 6)
        e = _ep(f"i{i}", False, T=T)
        eps.append(e)
        scores[e.episode_id] = np.full(T, 100.0)         # clearly separable
    rep = length_confound_report(eps, scores)
    assert rep["healthy_len_score_spearman"] > 0.9
    assert not np.isnan(rep["length_matched_auroc"])


def test_episode_auc_is_documented_as_offline():
    from derail.evaluation.metrics import episode_auc
    assert "OFFLINE" in (episode_auc.__doc__ or "")


# ------------------------------------------------------------- (mask)
def test_grounded_hybrid_behavioural_submodel_is_masked_to_51():
    from derail.monitor.grounding import _GroundedBase
    src = inspect.getsource(_GroundedBase.__init__)
    assert "behav_slice=D_TOTAL_EXT" in src, "behavioural submodel not masked to 51"


def test_hybrid_base_slices_behaviour():
    from derail.monitor.hybrid import _HybridBase
    src = inspect.getsource(_HybridBase)
    assert "_behav_slice" in src and "_behav_x" in src


# ------------------------------------------------------------- (split)
def test_cross_framework_splits_before_fitting():
    from derail.experiments import run_cross_framework
    src = inspect.getsource(run_cross_framework)
    assert "diag_split" in src
    assert "healthy leakage" in src           # the guard assertion is present


# ------------------------------------------------------------- (select)
def test_hybrid_study_reports_every_comparison_not_the_best():
    from derail.experiments import run_hybrid_study
    src = inspect.getsource(run_hybrid_study.evaluate_dataset)
    assert 'max(hybrid_rows, key=lambda r: r["auroc"])' not in src
    assert "holm_bonferroni" in src


# ------------------------------------------------------------- (kill)
def test_hmt_kill_switch_is_single_prespecified_decision():
    from derail.experiments import run_hmt_ab
    src = inspect.getsource(run_hmt_ab._verdict)
    assert "PRIMARY_ARCH" in src
    assert "c1 or c2 or c3 or c4" not in src       # the old any-of-any gate
    assert "exploratory" in src.lower()


# ------------------------------------------------------------- (verdict)
def test_h1_verdict_uses_paired_difference_and_holm():
    from derail.experiments import run_experiment
    src = inspect.getsource(run_experiment._h1_verdict)
    assert "_paired_diff_ci" in src
    assert "p_base_holm" in src
    # the old one-sample-CI significance test is gone
    assert 'base["mean_lead_all"] < esn["mean_lead_all_ci_lo"]' not in src


# ------------------------------------------------------------- (calib)
def test_h3a_does_not_treat_null_percentile_as_a_probability():
    from derail.experiments import run_experiment
    src = inspect.getsource(run_experiment.run_h3_calibration)
    assert "_uniform_ks" in src                 # null tested for uniformity
    # no ECE COLUMN named for the null (the reliability/ECE is isotonic-only)
    assert '"ece_null"' not in src and "'ece_null'" not in src
    assert "null_healthy_ks" in src


# ------------------------------------------------------------- (organic)
def test_organic_verification_scores_the_shipped_gate():
    from verification import score_organic_halluc
    src = inspect.getsource(score_organic_halluc.main)
    assert "score_step" in src            # the served fusion, incl. lex
    assert "np.maximum(zb, zg)" not in src  # the old manual re-fusion is gone


def test_organic_theta_is_selected_out_of_fold_not_in_sample():
    """Choosing theta on the episodes the gate was fit on makes the operating
    point a fiction: in-sample scores run low, theta lands low, and every class
    over-alarms. Fold k's threshold must come from the out-of-fold scores of
    the other folds."""
    from verification import score_organic_halluc

    src = inspect.getsource(score_organic_halluc.main)
    # The out-of-fold pass must exist and must score with a gate that did not
    # see the episode.
    assert "oof_b" in src and 'gates[_fold(r["episode_id"])]' in src
    # And the per-fold cohort must EXCLUDE that fold.
    assert '_fold(healthy[idx]["episode_id"]) != k' in src, \
        "theta cohort must exclude the fold it will be applied to"
    # The in-sample calibration helper must not reappear.
    assert "def _calibrate" not in src
    assert "_calibrate(gate, fit_h)" not in src


def test_serving_temperature_arms_are_scored_against_their_own_nulls():
    """L3's whole point is that the 0.9 and 0.2 arms carry SEPARATE
    temperature-matched nulls. Reading one arm's healthy FA while scoring the
    other would reintroduce the confound the study exists to remove."""
    from verification import l3_serving_temperature as l3

    assert l3.MIN_N == 10, "pre-registered power floor changed"
    src = inspect.getsource(l3.main)
    # Each arm's healthy cohort is counted from that arm's own frame.
    assert '_counts(hot, "healthy")' in src
    assert '_counts(cold, "healthy")' in src
    # And it must refuse to run on one arm alone rather than compare stale data.
    assert "raise SystemExit" in src


def test_serving_temperature_reports_failure_mix_before_claiming_detection():
    """A detection drop is only interpretable if the failure MIX is shown:
    fewer aborted runs at 0.2 would lower detection without the monitor
    changing at all."""
    from verification import l3_serving_temperature as l3

    src = inspect.getsource(l3.main)
    mix_at = src.index("failure-mix shift")
    verdict_at = src.index("[VERDICT]")
    assert mix_at < verdict_at, "mix must be reported before any verdict"
    assert "chi2_contingency" in src


# ------------------------------------------------------------- (labels)
def test_hallucination_labeller_separates_arithmetic_from_fabrication():
    from verification import organic_hallucination as oh

    def _step(text, action="tool_call"):
        return {"text": text, "action": action, "token_logprobs": [],
                "latency_s": 1.0, "output_tokens": 5, "error": False}

    # Grounded inputs: flights 100, 200; hotel 300 (=> 600 for 2 nights).
    tool_steps = [
        _step('[lookup_flight({"x": 1}) -> $100]'),
        _step('[lookup_flight({"x": 2}) -> $200]'),
        _step('[lookup_hotel({"city": "A"}) -> $300]'),
    ]
    # Correct total 100+200+600 = 900.
    healthy = tool_steps + [_step("The total is $900.", action="synthesis")]
    assert oh.label(healthy, 900)[0] == "healthy"

    # Wrong total but derivable arithmetic (forgot a component) -> arithmetic.
    arith = tool_steps + [_step("The total is $600.", action="synthesis")]
    assert oh.label(arith, 900)[0] == "arithmetic_error"

    # A fabricated ITEM price not in any tool result -> hallucination, and a
    # coincidental subset sum must NOT excuse it.
    fab = tool_steps + [_step("Flights were $555 each; total $900.",
                             action="synthesis")]
    assert oh.label(fab, 900)[0] == "hallucinated"


def test_allowed_numbers_no_longer_enumerates_all_subset_sums():
    from verification import organic_hallucination as oh
    src = inspect.getsource(oh._grounded_values)
    assert "combinations" not in src      # item provenance excludes subset sums


# ------------------------------------------------------------- (name)
def test_leave_one_out_is_named_per_class():
    from derail.experiments import run_leave_one_out
    assert "per-class" in (run_leave_one_out.__doc__ or "").lower()
    assert "real_per_class_baselines.csv" in inspect.getsource(
        run_leave_one_out.run_leave_one_out)


# ------------------------------------------------------------- (seeds)
def test_multiseed_reruns_seed_zero():
    from derail.experiments import run_hybrid_multiseed
    src = inspect.getsource(run_hybrid_multiseed.main)
    assert "for seed in (0, *EXTRA_SEEDS)" in src
    assert 'pd.read_csv(TABLES_DIR / "hybrid_benchmark.csv")' not in src


# ------------------------------------------------------------- (noise)
def test_robustness_noise_is_feature_aware_and_repeated():
    from derail.experiments import run_robustness_study as rrs
    # One-hot action dims (36-39) and the bounded fraction dims are NOT in the
    # continuous-noise set.
    assert 36 not in rrs._CONTINUOUS_DIMS and 42 not in rrs._CONTINUOUS_DIMS
    src = inspect.getsource(rrs.run_robustness_study)
    assert "SEEDS" in src and "std" in src        # repeated realisations + spread


# ------------------------------------------------------------- (seed avg)
def test_seq_scaling_does_not_average_the_seed_column():
    from derail.experiments import seq_data_scaling
    src = inspect.getsource(seq_data_scaling.run_scaling_experiment)
    # The mean must be taken over the metric columns only, never the whole df.
    assert 'df.groupby("N_train")[metric_cols].mean()' in src
    assert 'df.groupby("N_train").mean().reset_index()' not in src


# ------------------------------------------- reported coefficient shares
def test_coefficient_share_is_bounded():
    import pandas as pd
    from derail.experiments.explain_hybrid import coefficients_table
    # Opposite-sign coefficients make a naive share exceed 1.0, which would
    # publish "this monitor contributes 130% of the fusion".
    ex = pd.DataFrame([{"dataset": "d", "coef_esn": -0.2, "coef_maha": 0.9,
                        "intercept": 0.0}])
    out = coefficients_table(ex)
    share = float(out["maha_magnitude_share"].iloc[0])
    assert 0.0 <= share <= 1.0


# ------------------------------------------------------------- (policy)
def test_demo_degenerate_filter_is_a_declared_policy():
    from derail.experiments import demo
    src = inspect.getsource(demo.fit_monitor)
    assert "degenerate-output policy" in src
    assert "n_degenerate" in src


# ------------------------- what a deployment loses without token logprobs
def test_channel_selection_degrades_on_missing_logprobs():
    """A corpus without logprobs must drop the u channel, not fake it."""
    from derail.experiments.run_hybrid_study import REAL_DATASETS, load_real

    _, gemini = load_real(REAL_DATASETS["real_gemini_long"])
    _, qwen = load_real(REAL_DATASETS["real_research7b_long"])
    assert "u" not in gemini, "u channel claimed on a corpus with no logprobs"
    assert "u" in qwen


def test_horizon_helper_treats_healthy_as_unbounded():
    """Healthy episodes have no onset, so a horizon filter must keep them."""
    import numpy as np

    from derail.common import D_TOTAL, Episode
    from experimental.telemetry_dependence import _horizon

    healthy = Episode(X=np.zeros((10, D_TOTAL)), episode_id="h",
                      is_healthy=True, failure_class=None, tau=None,
                      t_fail=None, severity=None)
    injected = Episode(X=np.zeros((10, D_TOTAL)), episode_id="i",
                       is_healthy=False, failure_class="looping", tau=2,
                       t_fail=9, severity=0.5)
    assert _horizon(healthy) == float("inf")
    assert _horizon(injected) == 7.0


# ---------------------------- cross-channel tamper check, and its limits
def test_tamper_check_flags_a_pinned_channel_and_not_a_varying_one():
    import numpy as np

    from derail.common import CHANNEL_SLICES, D_TOTAL, Episode, Standardizer
    from experimental.tamper_check import fit_thresholds, is_tampered

    def mk(seed, pin=None):
        # Seeded explicitly: str.hash() is salted per process, which would make
        # this test pass or fail depending on the interpreter run.
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(30, D_TOTAL))
        if pin:
            X[:, CHANNEL_SLICES[pin]] = 0.0     # constant => zero variability
        return Episode(X=X, episode_id=f"e{seed}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    healthy = [mk(i) for i in range(40)]
    std = Standardizer().fit(healthy)
    bounds = fit_thresholds(healthy, std)
    assert not is_tampered(mk(1000), std, bounds)
    assert is_tampered(mk(1001, pin="e"), std, bounds)


def test_replay_attack_preserves_healthy_variability():
    """The adaptive attack must stay inside the healthy variability range —
    that is exactly why adversarial robustness stays future work."""
    import numpy as np

    from derail.common import D_TOTAL, Episode, Standardizer
    from experimental.tamper_check import CHANNELS, flatness, replay_channels

    def mk(seed):
        rng = np.random.default_rng(seed)
        return Episode(X=rng.normal(size=(30, D_TOTAL)), episode_id=f"d{seed}",
                       is_healthy=True, failure_class=None, tau=None,
                       t_fail=None, severity=None)

    donors = [mk(i) for i in range(40)]
    std = Standardizer().fit(donors)
    attacked = replay_channels(mk(999), CHANNELS, donors, seed=3)
    got = flatness(attacked, std)
    # The attacked channels ARE a donor's trace, so their variability cannot
    # fall below the least-varying healthy episode - no flatness rule can
    # separate them without also flagging real healthy traffic.
    for ch in CHANNELS:
        floor = min(flatness(d, std)[ch] for d in donors)
        assert got[ch] >= floor, ch


# --------------------------------- recalibration budget must cover both
def test_recalibration_budget_covers_fit_and_threshold():
    """n healthy episodes is the WHOLE budget: a deployment cannot also have a
    separate full-size threshold set, so the split must come out of n."""
    import inspect

    from experimental import recalibration_cost as rc

    src = inspect.getsource(rc._one)
    assert "0.75 * n" in src, "fit/threshold split no longer taken out of n"
    assert "pick_threshold" in src and "thr" in src


def test_recalibration_reports_realized_false_alarms():
    """The finding is about the operating point; dropping FA from the table
    would hide the only column that showed '~30 episodes' was wrong."""
    import csv
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "results" / "tables" / "recalibration_cost.csv")
    with path.open(encoding="utf-8") as fh:
        cols = next(csv.reader(fh))
    for needed in ("n_healthy", "auroc", "healthy_fa_rate", "detection_rate"):
        assert needed in cols, needed


# --------------------------------------------- L7b: collection targets honest
def test_power_target_is_nan_when_no_feasible_n_helps():
    """A tie must report NaN, not a huge number: 'collect 50k episodes' would
    read as an action when the honest answer is 'there is nothing to detect'."""
    import numpy as np

    from experimental.power_analysis import _n_for_power

    rng = np.random.default_rng(0)
    # Symmetric, tiny discordance = the autogen7b situation.
    assert _n_for_power(0.0, 0.0, rng) != _n_for_power(0.0, 0.0, rng) or True
    got = _n_for_power(0.034, 0.034, rng)
    assert got != got, "a perfectly tied comparison reported a finite target"


def test_power_target_is_finite_for_a_real_effect():
    import numpy as np

    from experimental.power_analysis import _n_for_power

    # Strongly asymmetric discordance = a real, detectable effect.
    got = _n_for_power(0.30, 0.02, np.random.default_rng(1))
    assert got == got and got < 200


# ------------------------------------------ L1b: figures lead with real data
def test_paper_figures_are_all_real_data():
    """Every figure in the paper must come from measured telemetry.

    The simulator score-stream figure was removed: its one unique panel
    (grounding loss) is now covered by a real episode, and its goal-drift panel
    showed the simulator's *designed* slow rotation as "no alarm", which reads
    as a capability gap when real traces detect that class at 0.66-0.86.
    """
    import re
    from pathlib import Path

    main_tex = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
    if not main_tex.exists():
        pytest.skip("manuscripts are local-only; nothing to inspect here")
    tex = main_tex.read_text(encoding="utf-8")
    used = re.findall(r"includegraphics[^{]*\{([^}]+)\}", tex)
    assert used, "no figures found in the paper"
    assert "fig1_score_traces.png" not in used, (
        "the simulator score-stream figure is back in the paper")
    for name in used:
        assert name.endswith("_real.png") or name == "hybrid_explain.png", name


def test_simulator_figure_is_labelled_on_the_image():
    import inspect

    from derail.experiments import plots

    src = inspect.getsource(plots.fig_score_traces)
    assert "SIMULATOR (illustrative" in src, "sim figure title no longer self-labels"


def test_every_simulator_figure_self_labels():
    """Any figure built from simulated telemetry must say so on the image."""
    import inspect

    from derail.experiments import plots

    for fn in (plots.fig_score_traces, plots.fig_h1_lead,
               plots.fig_h2_heatmap, plots.fig_reliability):
        assert "SIMULATOR (illustrative" in inspect.getsource(fn), fn.__name__


def test_real_figures_exclude_the_simulator():
    import inspect

    from derail.experiments import plots

    for fn in (plots.fig_class_coverage_real, plots.fig_monitor_benchmark_real):
        assert '!= "sim"' in inspect.getsource(fn), fn.__name__


# ------------------------------------------ published scope stays pinned
def test_published_table_scope_is_pinned_not_implicit():
    """A regeneration must reproduce the PUBLISHED scope, not silently widen it.

    Corpora added to REAL_DATASETS after the published tables were generated
    are reported in their own sections with their own --out-prefix tables;
    folding them into results/tables/hybrid_*.csv would change every published
    table's scope without anyone asking, and would invalidate the verdict those
    tables carry. `aftraj` is additionally not ours and not committed, so it
    could not be regenerated from a fresh checkout in any case.
    """
    import inspect

    from derail.experiments.run_hybrid_study import (PUBLISHED_DATASETS,
                                                     REAL_DATASETS, main)

    assert PUBLISHED_DATASETS < set(REAL_DATASETS), "pin no longer a subset"
    assert set(REAL_DATASETS) - PUBLISHED_DATASETS == {
        "ollama_llama8b", "real_gemini_long", "real_research7b_long_ext",
        "aftraj"}
    src = inspect.getsource(main)
    assert "PUBLISHED_DATASETS" in src, "default run no longer honours the pin"
    assert "--all-datasets" in src, "no explicit way to sweep everything"


def test_the_grounding_study_pins_its_published_scope_too():
    """The same guard, for the same reason, on the other published study.

    Defaulting to ALL of REAL_DATASETS lets every corpus added later widen the
    scope of grounding_*.csv on the next regeneration, and
    `run_grounding_multiseed` inherits that default for all five seeds.
    """
    import inspect

    from derail.experiments.run_grounding_study import (
        GROUNDING_PUBLISHED_DATASETS, main)
    from derail.experiments.run_hybrid_study import REAL_DATASETS

    pinned = set(GROUNDING_PUBLISHED_DATASETS)
    assert pinned < set(REAL_DATASETS), "pin no longer a subset"
    assert set(REAL_DATASETS) - pinned == {"aftraj"}, (
        "a corpus entered REAL_DATASETS without a decision about whether the "
        "published grounding tables should cover it")
    assert "GROUNDING_PUBLISHED_DATASETS" in inspect.getsource(main), \
        "default run no longer honours the pin"


def test_paper_leads_with_real_evidence_not_the_simulator():
    """L1: real-ecosystem validation is the spine; the simulator is a labelled
    mechanism study that must not precede it."""
    from pathlib import Path

    main_tex = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
    if not main_tex.exists():
        pytest.skip("manuscripts are local-only; nothing to inspect here")
    tex = main_tex.read_text(encoding="utf-8")
    real = tex.index(r"\section{Real-Ecosystem Validation")
    sim = tex.index(r"\section{Controlled Study")
    assert real < sim, "the simulator section precedes the real-trace section"
    assert "no deployment claim rests on it" in tex


def test_reliability_figure_rebuilds_from_published_artifacts():
    """fig4 must be buildable from what results/tables actually contains.

    It previously read an `ece` column that h3_calibration.csv no longer has,
    so the figure could not be regenerated at all - a published figure with no
    reproducible path.
    """
    from derail.experiments import plots

    plots.fig_reliability()          # raises if the schema drifts again
    assert (plots.FIGURES / "fig4_reliability.png").exists()


def test_reliability_ece_is_computed_from_the_bins_it_draws():
    """The annotation must agree with the curve, and with the summary table."""
    import pandas as pd

    from derail.experiments import plots

    rel = pd.read_csv(plots.RESULTS / "tables" / "h3_reliability.csv",
                      keep_default_na=False)
    cal = pd.read_csv(plots.RESULTS / "tables" / "h3_calibration.csv")
    sub = rel[(rel["stream"] == "fused") & (rel["calibrator"] == "isotonic")]
    n = sub["count"].sum()
    ece = ((sub["count"] / n)
           * (sub["mean_confidence"] - sub["empirical_freq"]).abs()).sum()
    published = cal[cal["stream"] == "fused"]["iso_ece"].iloc[0]
    assert abs(ece - published) < 5e-4, (ece, published)


def test_real_figure_covers_every_demo_use_case():
    """The real score-stream figure must show all five injected classes plus a
    grounding-loss panel, so no use case is silently absent from it."""
    import inspect

    from derail.experiments import plots
    from derail.experiments.demo import BUTTON_CLASSES

    src = inspect.getsource(plots.fig_score_traces_real)
    assert "_real_grounding_loss_panel" in src, "grounding-loss panel dropped"
    # The other four classes come from the corpus itself; grounding loss cannot
    # be injected, so it is drawn from the organic corpus instead.
    assert "grounding_loss" in BUTTON_CLASSES
    helper = inspect.getsource(plots._real_grounding_loss_panel)
    assert "organic_demo7b_ext" in helper
    assert "hallucinated" in helper


# ----------------------------------------------- repair-policy harness
def test_intervention_grades_with_the_oracle_but_fires_on_the_checks():
    """The loop must be trigger-blind to the label and outcome-blind to the
    trigger: interventions fire on what `verify` flags, while success is
    graded against the manifest's expected_total, which the repair prompt
    never sees."""
    from derail.intervene import evaluate_repair_policies as ris

    src = inspect.getsource(ris.main)
    assert "verify(steps, BOOKING_SPEC).failed" in src, \
        "the intervention must fire on the checks, not on the label"
    assert 'r["label"]' not in src.split("flagged = []")[1].split("n_healthy")[0], \
        "episode selection must not read the label"
    grade = inspect.getsource(ris._correct)
    assert "expected" in grade, "outcome must be graded against the oracle"


def test_intervention_reports_the_retry_luck_control():
    """A stochastic agent improves on retry alone, so any rung above
    `resample` is only credited if it beats `resample` — the comparison must
    be present, and paired."""
    from derail.intervene import evaluate_repair_policies as ris
    from derail.intervene.rollback import SCORED_RUNGS

    assert SCORED_RUNGS[:2] == ("none", "resample")
    src = inspect.getsource(ris.main)
    assert "vs resample" in src
    # Clustered permutation over episodes, not a summary of per-repeat
    # p-values: the repeats re-run the same episodes, so a median of their
    # p-values has no null distribution and drifts toward 0.5 as repeats are
    # added regardless of the effect.
    assert "paired_permutation_test" in src
    assert "np.median(ps)" not in src


def test_intervention_net_effect_counts_broken_runs_against_recovered():
    """Reporting recoveries without the correct runs an intervention broke
    would overstate its benefit."""
    from derail.intervene import evaluate_repair_policies as ris

    src = inspect.getsource(ris.main)
    i_net = src.index("NET task success")
    tail = src[i_net:]
    assert "rec" in tail and "brk" in tail
    assert "n_correct_base + rec - brk" in tail


def test_intervention_can_regrade_without_rerunning_the_agent():
    """When the grader changes, stored outcomes go stale. Re-running the agent
    would answer a different question -- fresh samples, not the same ones -- so
    the study must be able to re-derive outcomes from the persisted traces."""
    from derail.intervene import evaluate_repair_policies as ris

    src = inspect.getsource(ris.main)
    assert "args.regrade" in src
    assert "_retry" in src, "regrade must read the persisted retry traces"
    assert 'f"{row.rung}-r{rep}"' in src, "traces are keyed by rung and repeat"


def test_intervention_repeats_give_a_variance_estimate():
    """A single measurement of a stochastic retry is not a result. Repeats are
    tagged so the recovery rate can be reported with its spread."""
    from derail.intervene import evaluate_repair_policies as ris

    sig = inspect.signature(ris._one)
    assert len(sig.parameters) == 1          # takes one packed job tuple
    src = inspect.getsource(ris.main)
    assert "for rep in range(1, args.repeats + 1)" in src
    assert "independent repeats" in src, "the spread must be reported"
