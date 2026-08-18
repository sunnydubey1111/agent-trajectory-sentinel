"""Evaluation metrics for online derailment detection (DESIGN.md Module 6).

Alarm extraction, threshold selection on healthy validation episodes,
episode-level outcome accounting (true/early/false alarms, misses, correct
silences), lead/delay statistics, delay-vs-false-alarm trade-off curves,
episode-level ROC-AUC, and bootstrap confidence intervals.

Conventions (fixed by DESIGN.md):
  - An alarm fires at the first step t with s_t > theta (strictly greater).
  - On an injected episode, an alarm strictly before tau is an EARLY alarm --
    it counts as a false alarm, never as a detection.
  - For a true alarm: delay = alarm_step - tau, lead = (T - 1) - alarm_step,
    budget_saved_frac = lead / (T - 1).
  - Empty subsets yield NaN statistics (never exceptions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from derail.common import Episode, FAILURE_CLASSES, rng_for

_ALARM_COLUMNS = [
    "episode_id", "is_healthy", "failure_class", "tau", "T",
    "alarm_step", "outcome", "delay", "lead", "budget_saved_frac",
]


def first_alarm(scores: np.ndarray, theta: float) -> int | None:
    """Return the first step index t with scores[t] > theta, or None.

    Strict inequality, matching the alarm rule tau_hat = min{t : s_t > theta}.

    A non-finite score is REFUSED rather than treated as quiet. `NaN > theta`
    is False, so a monitor that failed to score a step would otherwise be
    indistinguishable from one that scored it and saw nothing -- the failure
    would be counted as evidence of health. The monitors guard finiteness
    themselves, but those guards are asserts and `python -O` removes them, so
    the last gate before the alarm decision has to be a real check.
    """
    s = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(s)):
        bad = int(np.flatnonzero(~np.isfinite(s))[0])
        raise FloatingPointError(
            f"non-finite score at step {bad}; the monitor could not score "
            f"this episode, which is not the same as scoring it as quiet")
    idx = np.flatnonzero(s > theta)
    return int(idx[0]) if idx.size else None


def min_calibration_episodes(fa_budget: float) -> int:
    """Smallest healthy-episode count at which `fa_budget` is reachable.

    A threshold read off the maxima of n healthy episodes cannot deliver an
    expected false-alarm rate below 1/(n+1): a fresh healthy episode exceeds
    the maximum of n exchangeable ones with exactly that probability. So an
    empirical threshold needs n >= 1/fa_budget - 1, and below that the budget
    is unreachable no matter which quantile is taken.
    """
    if not 0.0 < fa_budget < 1.0:
        raise ValueError("fa_budget must be in (0, 1)")
    return int(np.ceil(1.0 / fa_budget - 1.0))


def pick_threshold(val_healthy_scores: list[np.ndarray],
                   fa_budget: float = 0.05,
                   method: str = "empirical",
                   warn_infeasible: bool = True) -> float:
    """Threshold theta from healthy validation score streams.

    ``method="empirical"`` (default, unchanged): the per-episode maximum of
    each stream, then the (1 - fa_budget) quantile of those maxima with
    np.quantile method="higher", so theta is an observed maximum and the
    realized healthy-val false-alarm rate (fraction of maxima strictly above
    theta) is <= fa_budget *on that sample*.

    That in-sample guarantee does not transfer. Measured on six real corpora
    at a 5% budget, the realized held-out rate is 8.2% (per-corpus 4.7-11.0%),
    for two separate reasons:

      - an ORDER-STATISTIC FLOOR of 1/(n+1) (see `min_calibration_episodes`);
        corpora calibrating on 12-15 episodes cannot reach 5% at all, and are
        measured sitting exactly on their floor;
      - HEAVY TAILS, which push some corpora well above even that floor
        (real_research7b: floor 4.0%, realized 11.0%).

    ``method="lognormal"`` fits a log-normal to the maxima and returns its
    analytic (1 - fa_budget) quantile. Because it extrapolates past the
    largest observed value it escapes the order-statistic floor legitimately,
    and it is less sensitive to a single extreme episode. Measured on the same
    six corpora: realized FA 6.7% (vs 8.2%) for 3.0 points of detection
    (50.8% -> 47.8%). It is available but is not the default anywhere, because
    on corpora where the empirical rule already lands near the budget the tail
    fit overshoots it instead.

    Neither method fabricates data: with few episodes a budget may simply be
    unreachable, and `warn_infeasible` says so rather than returning a
    threshold that quietly misses it.
    """
    if not val_healthy_scores:
        raise ValueError("val_healthy_scores must be non-empty")
    maxima = np.array([float(np.max(np.asarray(s, dtype=float)))
                       for s in val_healthy_scores])
    n_min = min_calibration_episodes(fa_budget)
    if warn_infeasible and len(maxima) < n_min and method == "empirical":
        import warnings
        warnings.warn(
            f"pick_threshold: {len(maxima)} healthy episodes cannot deliver a "
            f"{fa_budget:.0%} false-alarm budget by an empirical quantile "
            f"(order-statistic floor 1/(n+1) = {1/(len(maxima)+1):.1%}; "
            f"needs n >= {n_min}). Collect more healthy episodes, relax the "
            f'budget, or use method="lognormal" to extrapolate the tail.',
            RuntimeWarning, stacklevel=2)
    if method == "empirical":
        return float(np.quantile(maxima, 1.0 - fa_budget, method="higher"))
    if method == "lognormal":
        from scipy import stats as _sps
        pos = maxima[maxima > 0.0]
        if len(pos) < 3:            # too few to fit a tail; stay empirical
            return float(np.quantile(maxima, 1.0 - fa_budget, method="higher"))
        lg = np.log(pos)
        sd = float(lg.std(ddof=1))
        if not np.isfinite(sd) or sd == 0.0:
            return float(np.quantile(maxima, 1.0 - fa_budget, method="higher"))
        return float(np.exp(lg.mean() + sd * _sps.norm.ppf(1.0 - fa_budget)))
    raise ValueError(f"unknown method {method!r}; "
                     "expected 'empirical' or 'lognormal'")


def evaluate_alarms(episodes: list[Episode], scores: dict[str, np.ndarray],
                    theta: float) -> pd.DataFrame:
    """Per-episode alarm outcomes at threshold theta.

    Returns one row per episode with columns: episode_id, is_healthy,
    failure_class, tau, T, alarm_step, outcome, delay, lead,
    budget_saved_frac.

    Outcome taxonomy:
      injected: "true_alarm" (alarm at t >= tau), "early_alarm" (alarm at
                t < tau; a false alarm, not a detection), "miss" (no alarm);
      healthy:  "false_alarm" (any alarm), "correct_silence" (no alarm).

    delay, lead, budget_saved_frac are populated only for true alarms
    (NaN otherwise); tau and alarm_step are NaN where undefined.
    """
    rows = []
    for ep in episodes:
        s = np.asarray(scores[ep.episode_id], dtype=float)
        assert s.shape == (ep.T,), (
            f"scores for {ep.episode_id} have shape {s.shape}, expected ({ep.T},)")
        alarm = first_alarm(s, theta)
        if ep.is_healthy:
            outcome = "correct_silence" if alarm is None else "false_alarm"
        elif alarm is None:
            outcome = "miss"
        elif alarm < ep.tau:
            outcome = "early_alarm"
        else:
            outcome = "true_alarm"
        delay = lead = budget_saved = float("nan")
        if outcome == "true_alarm":
            delay = float(alarm - ep.tau)
            lead = float(ep.T - 1 - alarm)
            budget_saved = lead / (ep.T - 1)
        rows.append({
            "episode_id": ep.episode_id,
            "is_healthy": bool(ep.is_healthy),
            "failure_class": ep.failure_class,
            "tau": float("nan") if ep.tau is None else float(ep.tau),
            "T": int(ep.T),
            "alarm_step": float("nan") if alarm is None else float(alarm),
            "outcome": outcome,
            "delay": delay,
            "lead": lead,
            "budget_saved_frac": budget_saved,
        })
    return pd.DataFrame(rows, columns=_ALARM_COLUMNS)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else float("nan")


def _agg(values, fn) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(fn(arr)) if arr.size else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate an evaluate_alarms frame into headline metrics.

    Returns a dict with:
      healthy_fa_rate    false alarms / healthy episodes
      early_alarm_rate   early alarms / injected episodes
      detection_rate     true alarms / injected episodes
      median_delay       median delay over true alarms
      median_lead        median lead over true alarms
      mean_budget_saved  mean budget_saved_frac over true alarms
      mean_lead_all      mean lead over ALL injected episodes, counting
                         misses and early alarms as 0 (survivorship-free:
                         expected steps of budget saved per failure episode)
      mean_budget_saved_all  same zero-filled convention on budget_saved_frac
      per_class          {failure_class: {detection_rate, median_delay,
                          median_lead, mean_lead_all}} for classes in df
    Empty subsets give NaN values, never exceptions.

    ``median_lead``/``median_delay`` condition on detection and therefore
    reward a monitor that only catches easy cases; use the ``*_all`` metrics
    to compare monitors whose detection rates differ.
    """
    healthy_mask = df["is_healthy"].astype(bool)
    healthy = df[healthy_mask]
    injected = df[~healthy_mask]
    true_alarms = injected[injected["outcome"] == "true_alarm"]
    present = set(injected["failure_class"])
    ordered = ([c for c in FAILURE_CLASSES if c in present]
               + sorted(c for c in present if c not in FAILURE_CLASSES))
    per_class: dict[str, dict] = {}
    for cls in ordered:
        sub = injected[injected["failure_class"] == cls]
        ta = sub[sub["outcome"] == "true_alarm"]
        per_class[cls] = {
            "detection_rate": _rate(len(ta), len(sub)),
            "median_delay": _agg(ta["delay"], np.median),
            "median_lead": _agg(ta["lead"], np.median),
            "mean_lead_all": _agg(sub["lead"].fillna(0.0), np.mean),
        }
    return {
        "healthy_fa_rate": _rate(int((healthy["outcome"] == "false_alarm").sum()),
                                 len(healthy)),
        "early_alarm_rate": _rate(int((injected["outcome"] == "early_alarm").sum()),
                                  len(injected)),
        "detection_rate": _rate(len(true_alarms), len(injected)),
        "median_delay": _agg(true_alarms["delay"], np.median),
        "median_lead": _agg(true_alarms["lead"], np.median),
        "mean_budget_saved": _agg(true_alarms["budget_saved_frac"], np.mean),
        "mean_lead_all": _agg(injected["lead"].fillna(0.0), np.mean),
        "mean_budget_saved_all": _agg(injected["budget_saved_frac"].fillna(0.0),
                                      np.mean),
        "per_class": per_class,
    }


def episode_auc(episodes: list[Episode],
                scores: dict[str, np.ndarray]) -> float:
    """OFFLINE ranking AUROC of the per-episode max score, healthy (0) vs
    injected (1).

    This is an offline ranking statistic: the maximum is taken over the WHOLE
    episode, so it is sensitive to exposure length and is NOT a causal
    detection metric (a longer episode gives its maximum more chances to
    exceed a healthy one). It ranks episodes well but must not be presented
    beside the causal alarm metrics (detection rate, lead, false-alarm rate
    from `evaluate_alarms`) without that distinction, and length must be
    controlled when comparing corpora of different horizons.

    Returns NaN if either class is absent.
    """
    y = np.array([0 if ep.is_healthy else 1 for ep in episodes], dtype=int)
    if np.unique(y).size < 2:
        return float("nan")
    s = np.array([float(np.max(np.asarray(scores[ep.episode_id], dtype=float)))
                  for ep in episodes])
    return float(roc_auc_score(y, s))


def length_confound_report(episodes: list[Episode],
                           scores: dict[str, np.ndarray]) -> dict:
    """Quantify how much episode length confounds the offline AUROC.

    Returns:
      healthy_len_score_spearman  rank corr between healthy episode length T
                                  and healthy max-score (a large value means a
                                  monitor scores longer HEALTHY episodes higher,
                                  so length alone inflates the AUROC);
      raw_auroc                   `episode_auc` over all episodes;
      length_matched_auroc        AUROC restricted to healthy/injected episodes
                                  in overlapping length bins, resampled so both
                                  classes share the same length distribution -
                                  the length-controlled comparison.
    """
    from scipy import stats as _sps

    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]

    def _mx(ep):
        return float(np.max(np.asarray(scores[ep.episode_id], dtype=float)))

    hl = np.array([ep.T for ep in healthy], dtype=float)
    hs = np.array([_mx(ep) for ep in healthy], dtype=float)
    spearman = (float(_sps.spearmanr(hl, hs).statistic)
                if hl.size >= 3 and np.unique(hl).size > 1 else float("nan"))

    # Length-matched AUROC: bin by T, keep bins that contain BOTH classes, and
    # within each bin compare equal numbers of healthy and injected episodes.
    def _bin(t):
        return int(t) // 2
    by_bin: dict[int, dict] = {}
    for ep in episodes:
        b = by_bin.setdefault(_bin(ep.T), {"h": [], "i": []})
        (b["h"] if ep.is_healthy else b["i"]).append(ep)
    matched = []
    for b in by_bin.values():
        n = min(len(b["h"]), len(b["i"]))
        matched += b["h"][:n] + b["i"][:n]
    matched_auroc = (episode_auc(matched, scores)
                     if matched and any(e.is_healthy for e in matched)
                     and any(not e.is_healthy for e in matched)
                     else float("nan"))
    return {"healthy_len_score_spearman": spearman,
            "raw_auroc": episode_auc(episodes, scores),
            "length_matched_auroc": matched_auroc,
            "n_healthy": len(healthy), "n_injected": len(injected),
            "n_matched": len(matched)}


def delay_fa_curve(val_healthy_scores: list[np.ndarray],
                   test_episodes: list[Episode],
                   test_scores: dict[str, np.ndarray],
                   quantiles: np.ndarray = np.linspace(0.5, 0.999, 25),
                   ) -> pd.DataFrame:
    """Sweep theta over quantiles of healthy-val per-episode maxima.

    For each quantile q, theta = np.quantile(val maxima, q, method="higher");
    test episodes are re-evaluated at that theta. Returns a DataFrame with
    columns: quantile, theta, healthy_fa_rate (realized on healthy test
    episodes), detection_rate, median_delay, median_lead.
    """
    if not val_healthy_scores:
        raise ValueError("val_healthy_scores must be non-empty")
    maxima = np.array([float(np.max(np.asarray(s, dtype=float)))
                       for s in val_healthy_scores])
    rows = []
    for q in np.asarray(quantiles, dtype=float):
        theta = float(np.quantile(maxima, q, method="higher"))
        summ = summarize(evaluate_alarms(test_episodes, test_scores, theta))
        rows.append({
            "quantile": float(q),
            "theta": theta,
            "healthy_fa_rate": summ["healthy_fa_rate"],
            "detection_rate": summ["detection_rate"],
            "median_delay": summ["median_delay"],
            "median_lead": summ["median_lead"],
        })
    return pd.DataFrame(rows, columns=["quantile", "theta", "healthy_fa_rate",
                                       "detection_rate", "median_delay",
                                       "median_lead"])


def bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 1000,
                 seed: int = 0, ci: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for stat(values).

    Non-finite entries are dropped first; an empty (or all-NaN) input returns
    (nan, nan). Resampling is deterministic via rng_for(seed, "bootstrap_ci").
    stat may be a numpy reduction accepting axis= (vectorized path) or any
    callable on a 1-D array (per-resample fallback).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = rng_for(seed, "bootstrap_ci")
    samples = v[rng.integers(0, v.size, size=(n_boot, v.size))]
    try:
        boot = np.asarray(stat(samples, axis=1), dtype=float)
        if boot.shape != (n_boot,):
            raise TypeError("stat(axis=1) returned wrong shape")
    except TypeError:
        boot = np.array([float(stat(row)) for row in samples])
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha)))


if __name__ == "__main__":
    import math

    from derail.common import D_TOTAL

    rng = rng_for(0, "metrics", "smoke")

    def make_ep(eid: str, T: int, tau: int | None = None,
                failure_class: str | None = None) -> Episode:
        X = rng.normal(size=(T, D_TOTAL))
        if tau is None:
            return Episode(X, eid, True, None, None, None, None)
        return Episode(X, eid, False, failure_class, tau, T - 1, 0.5)

    # --- first_alarm: strict inequality, None when never crossed ----------
    assert first_alarm(np.array([0.0, 1.0, 2.0]), 1.0) == 2
    assert first_alarm(np.array([1.0, 1.0]), 1.0) is None
    # must raise, not skip past to the later 3.0 (see first_alarm's docstring):
    try:
        first_alarm(np.array([np.nan, 0.5, 3.0]), 1.0)
        raise AssertionError("non-finite score was not refused")
    except FloatingPointError:
        pass

    # --- pick_threshold: quantile method="higher", budget honored ---------
    val = [np.array([0.0, float(m)]) for m in range(1, 101)]  # maxima 1..100
    theta95 = pick_threshold(val, fa_budget=0.05)
    assert theta95 == float(np.quantile(np.arange(1.0, 101.0), 0.95,
                                        method="higher")) == 96.0
    realized = np.mean([np.max(s) > theta95 for s in val])
    assert realized <= 0.05 and realized == 0.04

    # --- hand-built episodes with known alarm outcomes at theta = 1.0 -----
    eps = [
        make_ep("h1", 10),
        make_ep("h2", 10),
        make_ep("i1", 12, tau=5, failure_class="goal_drift"),
        make_ep("i2", 12, tau=5, failure_class="looping"),
        make_ep("i3", 12, tau=6, failure_class="grounding_loss"),
        make_ep("i4", 12, tau=5, failure_class="goal_drift"),
    ]
    sc = {eid: np.full(T, 0.1) for eid, T in
          [("h1", 10), ("h2", 10), ("i1", 12), ("i2", 12), ("i3", 12), ("i4", 12)]}
    sc["h2"][4] = 2.0    # healthy spike -> false_alarm at 4
    sc["i1"][7] = 1.5    # alarm 2 steps after tau=5 -> true_alarm
    sc["i2"][3] = 1.5    # alarm before tau=5 -> early_alarm
    sc["i4"][5] = 1.5    # alarm exactly at tau=5 -> true_alarm, delay 0

    df = evaluate_alarms(eps, sc, theta=1.0)
    out = dict(zip(df["episode_id"], df["outcome"]))
    assert out == {"h1": "correct_silence", "h2": "false_alarm",
                   "i1": "true_alarm", "i2": "early_alarm",
                   "i3": "miss", "i4": "true_alarm"}
    r1 = df[df["episode_id"] == "i1"].iloc[0]
    assert r1["alarm_step"] == 7 and r1["delay"] == 2 and r1["lead"] == 4
    assert math.isclose(r1["budget_saved_frac"], 4 / 11)
    r4 = df[df["episode_id"] == "i4"].iloc[0]
    assert r4["delay"] == 0 and r4["lead"] == 6
    assert math.isnan(df[df["episode_id"] == "i3"].iloc[0]["delay"])

    summ = summarize(df)
    assert summ["healthy_fa_rate"] == 0.5
    assert summ["early_alarm_rate"] == 0.25
    assert summ["detection_rate"] == 0.5
    assert summ["median_delay"] == 1.0 and summ["median_lead"] == 5.0
    assert math.isclose(summ["mean_budget_saved"], 5 / 11)
    assert summ["mean_lead_all"] == 2.5              # (4 + 0 + 0 + 6) / 4
    assert math.isclose(summ["mean_budget_saved_all"], 10 / 44)
    assert summ["per_class"]["goal_drift"]["mean_lead_all"] == 5.0
    assert summ["per_class"]["looping"]["mean_lead_all"] == 0.0
    assert summ["per_class"]["goal_drift"]["detection_rate"] == 1.0
    assert summ["per_class"]["goal_drift"]["median_delay"] == 1.0
    assert summ["per_class"]["looping"]["detection_rate"] == 0.0
    assert math.isnan(summ["per_class"]["looping"]["median_delay"])
    assert summ["per_class"]["grounding_loss"]["detection_rate"] == 0.0

    # --- empty / one-sided subsets: NaNs, not crashes ----------------------
    empty = summarize(evaluate_alarms([], {}, 1.0))
    assert math.isnan(empty["detection_rate"]) and math.isnan(empty["median_delay"])
    assert math.isnan(empty["healthy_fa_rate"]) and empty["per_class"] == {}

    # --- episode_auc on a separable toy case -------------------------------
    auc_eps = ([make_ep(f"ah{i}", 8) for i in range(3)]
               + [make_ep(f"ai{i}", 8, tau=3, failure_class="looping")
                  for i in range(3)])
    auc_sc = {ep.episode_id: (np.full(8, 0.3) if ep.is_healthy
                              else np.full(8, 2.5)) for ep in auc_eps}
    assert episode_auc(auc_eps, auc_sc) == 1.0
    assert math.isnan(episode_auc(auc_eps[:3], auc_sc))  # single class

    # --- delay_fa_curve: theta sweep, FA rate non-increasing ---------------
    curve_val = [np.array([0.05, m]) for m in np.linspace(0.1, 2.0, 20)]
    curve = delay_fa_curve(curve_val, eps, sc,
                           quantiles=np.array([0.5, 0.9, 0.999]))
    assert list(curve.columns) == ["quantile", "theta", "healthy_fa_rate",
                                   "detection_rate", "median_delay",
                                   "median_lead"]
    assert len(curve) == 3 and np.all(np.diff(curve["theta"]) >= 0)
    assert np.all(np.diff(curve["healthy_fa_rate"]) <= 0)
    assert curve["detection_rate"].iloc[0] == 0.5  # theta=1.1 catches i1, i4

    # --- bootstrap_ci: brackets the statistic, deterministic ---------------
    vals = rng.normal(size=200) + 5.0
    lo, hi = bootstrap_ci(vals, stat=np.median, seed=7)
    assert lo < np.median(vals) < hi
    assert (lo, hi) == bootstrap_ci(vals, stat=np.median, seed=7)
    plo, phi = bootstrap_ci(vals, stat=lambda v: float(np.sort(v)[len(v) // 2]),
                            seed=7)  # non-axis callable -> fallback path
    assert plo < phi
    assert all(math.isnan(b) for b in bootstrap_ci(np.array([]), seed=3))

    print("PASS: metrics smoke test — first_alarm, pick_threshold, "
          "evaluate_alarms, summarize, episode_auc, delay_fa_curve, "
          "bootstrap_ci all verified on hand-built cases.")
