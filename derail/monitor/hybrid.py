"""Hybrid ESN + Mahalanobis monitors (exp/hybrid-fusion).

Motivation (measured, not assumed): the ESN-CUSUM ensemble dominates on
temporal/behavioral failures (looping, tool_cascade, rate_limit, timeout)
and on long simulator episodes, while DeltaMahalanobis — memoryless plus a
1-lag delta — wins on the short (T~5-6) real_research7b episodes where the
reservoir has almost no post-onset horizon to integrate evidence over.
These monitors fuse the two score streams so one detector covers both
regimes.

Score calibration: raw ESN-CUSUM values and Mahalanobis distances live on
incomparable scales, so fusion first maps each stream to a healthy-referenced
robust z-score: z = (s - median) / (1.4826 * MAD), with the reference
statistics pooled over the healthy TRAIN episodes' per-step scores (ESN
statistics exclude its washout steps, which emit exactly 0). This is
label-free, so the one-class discipline of DESIGN.md is preserved.

Fusion variants (all causal `OnlineMonitor`s):
  - HybridWeighted   s_t = w * z_esn + (1 - w) * z_maha   (default w = 0.5)
  - HybridMax        s_t = max(z_esn, z_maha)
  - HybridGated      s_t = g_t * z_maha + (1 - g_t) * z_esn, where the gate
                     g_t = sigmoid((d_t - d_med) / d_scale) is the calibrated
                     abruptness of the step (d_t = ||z(x_t) - z(x_{t-1})|| /
                     sqrt(D)): abrupt state jumps route weight to the
                     memoryless distance, smooth drift to the reservoir.
  - HybridLogistic   s_t = a * z_esn + b * z_maha + c with (a, b, c) from a
                     logistic regression on labeled steps. Training labels
                     require injected episodes, so `fit()` alone leaves it
                     equal to HybridWeighted(0.5); the runner must call
                     `fit_supervised()` on a calibration split that is
                     disjoint from the test episodes (cross-fit on the real
                     datasets, the `cal` split on the simulator).

The ESN and DeltaMahalanobis sub-monitors are passed in (optionally already
fitted, so several hybrids can share one fit); `fit()` fits them if needed
and then computes the calibration statistics.
"""

from __future__ import annotations

import numpy as np

from derail.common import Episode, OnlineMonitor, Standardizer, safe_scale
from derail.monitor.baselines import DeltaMahalanobisMonitor
from derail.monitor.esn import _WASHOUT, ChannelMaxESNMonitor

_EPS = 1e-9


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    """(median, robust scale). Scale = 1.4826*MAD, falling back to std.

    A channel with NO healthy variation is left UNSCALED (common.safe_scale)
    rather than divided by a tiny epsilon, which would make an uninformative
    channel maximally sensitive. Non-degenerate inputs are unaffected.
    See DESIGN.md Amendment 6.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if scale < _EPS:
        scale = float(v.std())
    return med, safe_scale(scale, eps=_EPS)


class _HybridBase(OnlineMonitor):
    """Shared machinery: sub-monitor fitting, robust score calibration."""

    def __init__(self, esn: ChannelMaxESNMonitor,
                 maha: DeltaMahalanobisMonitor,
                 standardizer: Standardizer,
                 subs_prefit: bool = False,
                 behav_slice: int | None = None) -> None:
        self.esn = esn
        self.maha = maha
        self.standardizer = standardizer
        self._subs_prefit = bool(subs_prefit)
        # When set, the behavioural sub-monitors see only x_t[:behav_slice].
        # Grounded hybrids pass 51 so the behavioural Mahalanobis/ESN are
        # computed WITHOUT the grounding dims, which the explicit grounding
        # stream already covers; without this mask the g dims were counted in
        # both the behavioural distance and the grounding stream.
        self._behav_slice = behav_slice
        self._esn_stats: tuple[float, float] | None = None
        self._maha_stats: tuple[float, float] | None = None
        self._t = 0

    def _behav_x(self, x_t: np.ndarray) -> np.ndarray:
        return x_t if self._behav_slice is None else x_t[:self._behav_slice]

    def _behav_ep(self, ep: Episode) -> Episode:
        if self._behav_slice is None or ep.X.shape[1] <= self._behav_slice:
            return ep
        return Episode(X=ep.X[:, :self._behav_slice].copy(),
                       episode_id=ep.episode_id, is_healthy=ep.is_healthy,
                       failure_class=ep.failure_class, tau=ep.tau,
                       t_fail=ep.t_fail, severity=ep.severity)

    # -- fitting -----------------------------------------------------------
    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Fit subs (unless prefit) and calibrate both score streams.

        NOTE: the robust score-scale calibration below is estimated on
        the SAME healthy episodes the sub-monitors were fit on (resubstitution),
        which makes the Mahalanobis/fusion scales optimistically narrow. A
        held-out or cross-fit calibration would remove that optimism; the study
        runners that need it already cross-fit the supervised fusion,
        and the label-free scales are disclosed here as resubstituted.
        """
        behav = [self._behav_ep(ep) for ep in healthy_episodes]
        if not self._subs_prefit:
            self.esn.fit(behav)
            self.maha.fit(behav)
        esn_pool, maha_pool = [], []
        for ep in behav:
            s_e = self.esn.score_episode(ep)
            s_m = self.maha.score_episode(ep)
            esn_pool.append(s_e[_WASHOUT:])   # washout steps emit exactly 0
            maha_pool.append(s_m)
        self._esn_stats = _robust_stats(np.concatenate(esn_pool))
        self._maha_stats = _robust_stats(np.concatenate(maha_pool))
        self._fit_extra(healthy_episodes)

    def _fit_extra(self, healthy_episodes: list[Episode]) -> None:
        """Hook for subclass calibration (e.g. the gate)."""

    # -- streaming ---------------------------------------------------------
    def start_episode(self) -> None:
        assert self._esn_stats is not None, "fit() first"
        self.esn.start_episode()
        self.maha.start_episode()
        self._t = 0

    def _z_scores(self, x_t: np.ndarray) -> tuple[float, float]:
        """Advance both subs one step; return calibrated (z_esn, z_maha).

        z_esn is 0 during the ESN washout (no evidence yet, matching the
        sub-monitor's own convention) rather than the misleading negative
        z of a raw 0 score.
        """
        bx = self._behav_x(x_t)
        s_e = self.esn.score_step(bx)
        s_m = self.maha.score_step(bx)
        me, se = self._esn_stats
        mm, sm = self._maha_stats
        z_e = 0.0 if self._t < _WASHOUT else (s_e - me) / se
        z_m = (s_m - mm) / sm
        self._t += 1
        return float(z_e), float(z_m)


class HybridWeighted(_HybridBase):
    """Fixed convex combination of the calibrated streams."""

    def __init__(self, esn, maha, standardizer, w: float = 0.5,
                 subs_prefit: bool = False, name: str | None = None) -> None:
        super().__init__(esn, maha, standardizer, subs_prefit)
        if not 0.0 <= w <= 1.0:
            raise ValueError("w must be in [0, 1]")
        self.w = float(w)
        self.name = name or f"hybrid_weighted{int(round(100 * w)):02d}"

    def score_step(self, x_t: np.ndarray) -> float:
        z_e, z_m = self._z_scores(x_t)
        return self.w * z_e + (1.0 - self.w) * z_m


class HybridMax(_HybridBase):
    """Max of the calibrated streams (union-of-alarms behavior)."""

    name = "hybrid_max"

    def score_step(self, x_t: np.ndarray) -> float:
        z_e, z_m = self._z_scores(x_t)
        return max(z_e, z_m)


class HybridGated(_HybridBase):
    """Abruptness-gated fusion.

    The gate observes d_t = ||z(x_t) - z(x_{t-1})||_2 / sqrt(D) on the
    standardized raw telemetry (d_0 = 0) and is calibrated on healthy train
    steps: g_t = sigmoid((d_t - median) / scale). Abrupt anomalies (large
    state jump -> large d_t) weight the memoryless Mahalanobis; smooth,
    temporally-building anomalies weight the ESN.
    """

    name = "hybrid_gated"

    def __init__(self, esn, maha, standardizer,
                 subs_prefit: bool = False) -> None:
        super().__init__(esn, maha, standardizer, subs_prefit)
        self._gate_stats: tuple[float, float] | None = None
        self._prev_z: np.ndarray | None = None

    @staticmethod
    def _deltas(Z: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(np.diff(Z, axis=0, prepend=Z[:1]), axis=1)
        return d / np.sqrt(Z.shape[1])

    def _fit_extra(self, healthy_episodes: list[Episode]) -> None:
        pool = [self._deltas(self.standardizer.transform(ep.X))
                for ep in healthy_episodes]
        self._gate_stats = _robust_stats(np.concatenate(pool))

    def start_episode(self) -> None:
        super().start_episode()
        self._prev_z = None

    def score_step(self, x_t: np.ndarray) -> float:
        assert self._gate_stats is not None, "fit() first"
        z = self.standardizer.transform(np.asarray(x_t, dtype=float))
        d = (0.0 if self._prev_z is None
             else float(np.linalg.norm(z - self._prev_z)) / np.sqrt(z.size))
        self._prev_z = z
        dm, ds = self._gate_stats
        g = 1.0 / (1.0 + np.exp(-(d - dm) / ds))
        z_e, z_m = self._z_scores(x_t)
        return float(g * z_m + (1.0 - g) * z_e)


class HybridLogistic(_HybridBase):
    """Learned linear fusion: logit of P(post-onset | z_esn, z_maha).

    `fit()` (one-class) initializes the weights to the 0.5/0.5 fallback;
    `fit_supervised(healthy, injected)` then trains a logistic regression on
    per-step labels (steps >= tau of injected episodes are 1, everything
    else 0, class-balanced). The runner is responsible for keeping the
    supervised episodes disjoint from the test episodes (cross-fit).
    score_step returns the raw logit (monotone in the probability).

    Features are clipped to +-`clip` robust-z units (default 50) both when
    training and when scoring: a heavy anomaly can push the raw calibrated
    z past 1e6 (the healthy MAD in the denominator is tiny on concentrated
    score streams), and at that scale sklearn's L2 penalty drives the
    learned weights to numerical zero (observed on real_research7b_long).
    Clipping bounds the feature scale without changing which steps look
    anomalous. The bound is deliberately loose: at +-5 so many episodes'
    max scores saturate at the ceiling that episode ranking collapses
    (autogen7b AUROC fell 0.854 -> 0.639); at +-50 conditioning is fixed
    and every benchmark dataset matches or beats the unclipped variant.
    The other hybrids keep unclipped z: their fixed fusion rules
    are scale-monotone, and clipping could only introduce max-score ties
    between saturated healthy and saturated injected episodes.
    """

    name = "hybrid_logistic"

    def __init__(self, esn, maha, standardizer,
                 subs_prefit: bool = False, clip: float = 50.0) -> None:
        super().__init__(esn, maha, standardizer, subs_prefit)
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.clip = float(clip)
        self.coef_ = np.array([0.5, 0.5])
        self.intercept_ = 0.0
        self.supervised_ = False

    def _clipped_z(self, x_t: np.ndarray) -> tuple[float, float]:
        z_e, z_m = self._z_scores(x_t)
        c = self.clip
        return float(np.clip(z_e, -c, c)), float(np.clip(z_m, -c, c))

    def step_features(self, ep: Episode) -> np.ndarray:
        """Clipped calibrated (T, 2) feature matrix [z_esn, z_maha]."""
        self.start_episode()
        return np.array([self._clipped_z(x) for x in ep.X], dtype=float)

    def fit_supervised(self, healthy_episodes: list[Episode],
                       injected_episodes: list[Episode]) -> None:
        """Train the fusion weights on labeled steps (see class docstring)."""
        from sklearn.linear_model import LogisticRegression

        # Reset supervised state before every fit: a degenerate-label
        # early return, or a refit on a different dataset, must NOT silently
        # keep the coefficients from a previous fit. On a degenerate fit this
        # falls back to the label-free 0.5/0.5 default, not a stale model.
        self.coef_ = np.array([0.5, 0.5])
        self.intercept_ = 0.0
        self.supervised_ = False

        feats, labels = [], []
        for ep in healthy_episodes:
            f = self.step_features(ep)
            feats.append(f)
            labels.append(np.zeros(ep.T, dtype=int))
        for ep in injected_episodes:
            f = self.step_features(ep)
            y = np.zeros(ep.T, dtype=int)
            y[ep.tau:] = 1
            feats.append(f)
            labels.append(y)
        X = np.concatenate(feats, axis=0)
        y = np.concatenate(labels)
        if len(np.unique(y)) < 2:
            return  # degenerate labels: keep the one-class fallback
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X, y)
        self.coef_ = clf.coef_[0].astype(float)
        self.intercept_ = float(clf.intercept_[0])
        self.supervised_ = True

    def score_step(self, x_t: np.ndarray) -> float:
        z_e, z_m = self._clipped_z(x_t)
        return float(self.coef_[0] * z_e + self.coef_[1] * z_m
                     + self.intercept_)


def recommended_monitor(standardizer: Standardizer,
                        healthy_episodes: list[Episode],
                        labeled_failures: list[Episode] | None = None,
                        channels: tuple[str, ...] = ("e", "u", "m"),
                        K: int = 8, seed: int = 1300,
                        min_labeled: int = 20) -> _HybridBase:
    """Build and fit the recommended production monitor (see the hybrid study).

    Default is HybridWeighted(w=0.5) — fully label-free, grand-mean AUROC
    0.812 across the eight benchmark datasets vs 0.802 (ESN) / 0.807
    (DeltaMahalanobis), and never far from the per-dataset winner. When at
    least `min_labeled` labeled failure episodes are available (~20 is
    enough; injection runs produce them cheaply), returns HybridLogistic
    fit-supervised on them instead — best grand mean (0.826) and
    statistically at-or-above the better standalone on every development
    dataset. See results/hybrid_report.md for the full evidence, and
    CLAIMS.md for these figures checked against the table they come from.
    """
    from derail.common import D_TOTAL_EXT, D_TOTAL_GRD

    labeled = [ep for ep in (labeled_failures or []) if not ep.is_healthy]
    grounded = bool(healthy_episodes
                    and healthy_episodes[0].X.shape[1] == D_TOTAL_GRD)

    # Branch on telemetry width FIRST: on grounded (v4) telemetry the
    # explicit grounding detector must never be dropped just because more
    # labels became available. The old order checked the label count first, so
    # reaching min_labeled switched a grounded gate for an UNgrounded logistic
    # and silently removed the content detector.
    if grounded:
        from derail.monitor.grounding import (GRD_DIM_NAMES,
                                              GroundingMonitor,
                                              HybridContentGate,
                                              HybridLogisticG)
        # Behavioural submodels on the 51-dim view; the grounding dims reach
        # the score only through `grd`, never the behavioural distance
        #. The grounded hybrids mask scoring to 51 via behav_slice.
        behav = [Episode(X=ep.X[:, :D_TOTAL_EXT].copy(),
                         episode_id=ep.episode_id, is_healthy=ep.is_healthy,
                         failure_class=ep.failure_class, tau=ep.tau,
                         t_fail=ep.t_fail, severity=ep.severity)
                 for ep in healthy_episodes]
        std51 = Standardizer().fit(behav)
        esn = ChannelMaxESNMonitor(std51, channels=channels, K=K,
                                   cusum=True, seed=seed)
        maha = DeltaMahalanobisMonitor(std51)
        esn.fit(behav)
        maha.fit(behav)
        # Continuous dims only: the binary lex flag reaches the gate/logistic
        # via its clean-null override path, never the trip calibration.
        grd = GroundingMonitor(dims=GRD_DIM_NAMES[:-1], name="grounding_cont")
        grd.fit(healthy_episodes)
        if len(labeled) >= min_labeled:
            # Supervised AND grounded: the grounding stream stays in the model.
            mon = HybridLogisticG(esn, maha, grd, standardizer,
                                  subs_prefit=True)
            mon.fit(healthy_episodes)
            mon.fit_supervised(healthy_episodes, labeled)
            return mon
        mon = HybridContentGate(esn, maha, grd, standardizer, subs_prefit=True)
        mon.fit(healthy_episodes)
        return mon

    # Ungrounded telemetry (43/51-dim): behavioural submodels see everything.
    esn = ChannelMaxESNMonitor(standardizer, channels=channels, K=K,
                               cusum=True, seed=seed)
    maha = DeltaMahalanobisMonitor(standardizer)
    esn.fit(healthy_episodes)
    maha.fit(healthy_episodes)
    if len(labeled) >= min_labeled:
        mon = HybridLogistic(esn, maha, standardizer, subs_prefit=True)
        mon.fit(healthy_episodes)
        mon.fit_supervised(healthy_episodes, labeled)
        return mon
    mon = HybridWeighted(esn, maha, standardizer, w=0.5, subs_prefit=True)
    mon.fit(healthy_episodes)
    return mon


def make_hybrids(standardizer: Standardizer,
                 channels: tuple[str, ...] = ("e", "u", "m"),
                 K: int = 8, seed: int = 1300,
                 ) -> tuple[ChannelMaxESNMonitor, DeltaMahalanobisMonitor,
                            list[_HybridBase]]:
    """Build the shared subs plus one instance of every hybrid variant.

    The subs are returned unfitted; fit them once, then fit each hybrid with
    subs_prefit semantics already wired (every hybrid here shares the two sub
    instances, so calls must stay sequential — no interleaved streaming).
    """
    esn = ChannelMaxESNMonitor(standardizer, channels=channels, K=K,
                               cusum=True, seed=seed)
    maha = DeltaMahalanobisMonitor(standardizer)
    hybrids = [
        HybridWeighted(esn, maha, standardizer, w=0.5, subs_prefit=True),
        HybridMax(esn, maha, standardizer, subs_prefit=True),
        HybridGated(esn, maha, standardizer, subs_prefit=True),
        HybridLogistic(esn, maha, standardizer, subs_prefit=True),
    ]
    return esn, maha, hybrids


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import time

    from derail.common import D_SEM, rng_for

    t0 = time.time()
    rng = rng_for(0, "hybrid", "smoke")

    def _make_X(T: int, shift_at: int | None = None,
                abrupt: bool = True) -> np.ndarray:
        base = rng.normal(size=D_SEM)
        base /= np.linalg.norm(base)
        E = base[None, :] + 0.15 * rng.normal(size=(T, D_SEM))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        mean_ent = np.clip(1.0 + 0.15 * rng.normal(size=T), 0.05, None)
        max_ent = mean_ent + np.abs(0.2 * rng.normal(size=T))
        slope = -0.05 + 0.1 * rng.normal(size=T)
        frac = np.clip(0.2 + 0.1 * rng.normal(size=T), 0.0, 1.0)
        onehot = np.zeros((T, 4))
        onehot[np.arange(T), rng.integers(0, 4, size=T)] = 1.0
        lat = 0.5 + 0.4 * rng.normal(size=T)
        outl = 4.0 + 0.5 * rng.normal(size=T)
        err = np.zeros(T)
        if shift_at is not None:
            s = slice(shift_at, None)
            n = T - shift_at
            if abrupt:  # hard state jump (Mahalanobis territory)
                E[s] = -base[None, :] + 0.6 * rng.normal(size=(n, D_SEM))
                E[s] /= np.linalg.norm(E[s], axis=1, keepdims=True)
                mean_ent[s] += 2.5
                lat[s] += 2.5
                err[s] = 1.0
            else:       # slow drift (ESN territory)
                drift = rng.normal(size=D_SEM)
                drift /= np.linalg.norm(drift)
                for i, t in enumerate(range(shift_at, T)):
                    a = min(0.08 * (i + 1), 0.9)
                    E[t] = (1 - a) * E[t] + a * drift
                    E[t] /= np.linalg.norm(E[t])
                mean_ent[s] += 0.12 * (np.arange(n) + 1)
        return np.column_stack(
            [E, mean_ent, max_ent, slope, frac, onehot, lat, outl, err]
        )

    train = [Episode(X=_make_X(int(rng.integers(30, 50))),
                     episode_id=f"train-{i:04d}", is_healthy=True,
                     failure_class=None, tau=None, t_fail=None, severity=None)
             for i in range(30)]
    std = Standardizer().fit(train)

    T, tau = 45, 20
    healthy_ep = Episode(X=_make_X(T), episode_id="h", is_healthy=True,
                         failure_class=None, tau=None, t_fail=None,
                         severity=None)
    abrupt_ep = Episode(X=_make_X(T, tau, abrupt=True), episode_id="fa",
                        is_healthy=False, failure_class="tool_cascade",
                        tau=tau, t_fail=T - 1, severity=1.0)
    drift_ep = Episode(X=_make_X(T, tau, abrupt=False), episode_id="fd",
                       is_healthy=False, failure_class="goal_drift",
                       tau=tau, t_fail=T - 1, severity=1.0)

    esn, maha, hybrids = make_hybrids(std, channels=("e", "u", "m"), seed=7)
    esn.fit(train)
    maha.fit(train)
    for h in hybrids:
        h.fit(train)

    for mon in hybrids:
        for ep in (abrupt_ep, drift_ep):
            s_h = mon.score_episode(healthy_ep)
            s_f = mon.score_episode(ep)
            assert s_f.shape == (T,) and np.all(np.isfinite(s_f)), mon.name
            assert s_f[tau:].mean() > s_h[tau:].mean(), (mon.name, ep.episode_id)
        # streaming == batch (causality sanity)
        mon.start_episode()
        streamed = np.array([mon.score_step(x) for x in abrupt_ep.X])
        assert np.allclose(streamed, mon.score_episode(abrupt_ep)), mon.name
        print(f"  {mon.name:<18} abrupt {mon.score_episode(abrupt_ep)[tau:].mean():7.2f}"
              f"  drift {mon.score_episode(drift_ep)[tau:].mean():7.2f}"
              f"  healthy {mon.score_episode(healthy_ep)[tau:].mean():7.2f}")

    # supervised logistic path: train on disjoint injected episodes
    log = hybrids[-1]
    assert isinstance(log, HybridLogistic) and not log.supervised_
    cal_inj = [Episode(X=_make_X(T, tau, abrupt=bool(i % 2)),
                       episode_id=f"cal-{i}", is_healthy=False,
                       failure_class="looping", tau=tau, t_fail=T - 1,
                       severity=1.0) for i in range(6)]
    log.fit_supervised(train[:10], cal_inj)
    assert log.supervised_ and np.all(np.isfinite(log.coef_))
    assert (log.score_episode(abrupt_ep)[tau:].mean()
            > log.score_episode(healthy_ep)[tau:].mean())
    # features are clipped, so the learned weights stay well-conditioned
    f = log.step_features(abrupt_ep)
    assert np.all(np.abs(f) <= log.clip) and np.abs(f).max() == log.clip
    assert np.abs(log.coef_).max() > 1e-3, "clipping should prevent " \
        "the near-zero-coefficient degeneracy"

    # recommended_monitor: label count decides the variant
    few = recommended_monitor(std, train, cal_inj[:3], channels=("e", "u", "m"))
    many = recommended_monitor(std, train, cal_inj * 4, channels=("e", "u", "m"))
    assert isinstance(few, HybridWeighted) and few.w == 0.5
    assert isinstance(many, HybridLogistic) and many.supervised_
    for mon in (few, many):
        assert (mon.score_episode(abrupt_ep)[tau:].mean()
                > mon.score_episode(healthy_ep)[tau:].mean()), mon.name

    # determinism
    esn2, maha2, hybrids2 = make_hybrids(std, channels=("e", "u", "m"), seed=7)
    esn2.fit(train)
    maha2.fit(train)
    hybrids2[1].fit(train)
    assert np.allclose(hybrids[1].score_episode(drift_ep),
                       hybrids2[1].score_episode(drift_ep)), "not deterministic"

    print(f"PASS hybrid smoke test ({time.time() - t0:.1f}s, "
          f"{len(hybrids)} hybrids; logistic coef={log.coef_.round(3)})")
