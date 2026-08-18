"""Baseline online monitors (DESIGN.md Module 3).

Memoryless / near-memoryless one-class baselines against which the ESN
ensemble (Module 2) is compared for H1:

  - CosineDriftMonitor      semantic drift from the episode's own early centroid
                            (a NEGATIVE CONTROL on this synthetic data, ~0.10
                            detection: a single-channel geometric drift with no
                            temporal model; kept to show the floor, not as a
                            competitive baseline)
  - SelfDriftMonitor        semantic drift from the episode's RUNNING centroid
                            (trajectory self-consistency; catches slow drift
                            that stays locally predictable)
  - RollingSurprisalMonitor EWMA of |z| of the mean token-uncertainty dim
  - MahalanobisMonitor      Ledoit-Wolf Mahalanobis distance of x_t
  - DeltaMahalanobisMonitor same on [x_t ; x_t - x_{t-1}] (1-lag). A STRONG
                            memoryless baseline that makes H1 falsifiable
                            (see class docstring).
  - IsolationForestMonitor  isolation forest on pooled healthy steps (also a
                            NEGATIVE CONTROL here, ~0.07 detection: no temporal
                            model, reported to bound the static-outlier floor)

All are causal `OnlineMonitor`s: `score_step(x_t)` uses only x_1..x_t of the
current episode plus statistics learned in `fit()` on HEALTHY train episodes.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

from derail.common import (
    D_SEM,
    IDX_MEAN_ENTROPY,
    Episode,
    OnlineMonitor,
    SEM_SLICE,
    Standardizer,
    rng_for,
)

_EPS = 1e-12


class CosineDriftMonitor(OnlineMonitor):
    """Cosine drift of e_t from the CURRENT episode's early semantic centroid.

    The centroid is the mean of the first `warmup` semantic vectors of the
    episode being scored, accumulated causally step by step; during warmup
    the score is 0. From step `warmup` on (0-indexed), the centroid is frozen
    and score_t = 1 - cos(e_t, centroid). No global fitting is needed.
    """

    name = "cosine_drift"

    def __init__(self, warmup: int = 5) -> None:
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        self.warmup = int(warmup)
        self._sum: np.ndarray | None = None
        self._count = 0
        self._centroid: np.ndarray | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """No-op: the reference centroid is per-episode, not learned."""

    def start_episode(self) -> None:
        """Reset the causal centroid accumulator for a new episode."""
        self._sum = np.zeros(D_SEM)
        self._count = 0
        self._centroid = None

    def score_step(self, x_t: np.ndarray) -> float:
        """Return 0 during warmup, else 1 - cos(e_t, warmup centroid)."""
        assert self._sum is not None, "call start_episode() first"
        e = np.asarray(x_t, dtype=float)[SEM_SLICE]
        if self._count < self.warmup:
            self._sum += e
            self._count += 1
            return 0.0
        if self._centroid is None:
            self._centroid = self._sum / self._count
        c = self._centroid
        cos = float(e @ c) / max(
            float(np.linalg.norm(e)) * float(np.linalg.norm(c)), _EPS
        )
        return float(1.0 - cos)


class SelfDriftMonitor(OnlineMonitor):
    """Semantic drift of e_t from the episode's RUNNING centroid (all past e).

    score_t = 1 - cos(e_t, mean(e_0..e_{t-1})) for t >= warmup, else 0. The
    running mean gives the statistic long memory over the episode's own
    trajectory, so it catches slow persistent rotation (goal drift) that
    remains locally predictable and therefore invisible to one-step-ahead
    surprise. O(1) state per step; causal; no global fitting.
    """

    name = "self_drift"

    def __init__(self, warmup: int = 3) -> None:
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        self.warmup = int(warmup)
        self._sum: np.ndarray | None = None
        self._count = 0

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """No-op: the reference centroid is per-episode, not learned."""

    def start_episode(self) -> None:
        """Reset the running-centroid accumulator for a new episode."""
        self._sum = np.zeros(D_SEM)
        self._count = 0

    def score_step(self, x_t: np.ndarray) -> float:
        """1 - cos(e_t, running mean of the episode's PAST e); 0 in warmup."""
        assert self._sum is not None, "call start_episode() first"
        e = np.asarray(x_t, dtype=float)[SEM_SLICE]
        if self._count < self.warmup:
            score = 0.0
        else:
            c = self._sum / self._count
            cos = float(e @ c) / max(
                float(np.linalg.norm(e)) * float(np.linalg.norm(c)), _EPS
            )
            score = float(1.0 - cos)
        self._sum += e
        self._count += 1
        return score


class RollingSurprisalMonitor(OnlineMonitor):
    """Causal EWMA of the two-sided z-score of the mean uncertainty dim.

    fit() learns the healthy mean/std of the mean-uncertainty dimension
    (IDX_MEAN_ENTROPY) over pooled train steps. score_t is an EWMA of its
    |z| — two-sided because DROPS (e.g. confident looping) are anomalous too.

    Named for what it measures on real traces: sampled-token surprisal, not
    predictive or semantic entropy. On simulator episodes the same
    dimension carries generated token entropy.
    """

    name = "rolling_surprisal"

    def __init__(self, ewma_alpha: float = 0.3) -> None:
        self.ewma_alpha = float(ewma_alpha)
        self._mu: float | None = None
        self._sd: float | None = None
        self._ewma: float | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Learn healthy mean/std of the mean-uncertainty dim over steps."""
        vals = np.concatenate(
            [ep.X[:, IDX_MEAN_ENTROPY] for ep in healthy_episodes]
        )
        self._mu = float(vals.mean())
        self._sd = float(max(vals.std(), 1e-3))

    def start_episode(self) -> None:
        """Reset the EWMA state for a new episode."""
        self._ewma = None

    def score_step(self, x_t: np.ndarray) -> float:
        """EWMA of |z(mean uncertainty)|; initialized at the first step's |z|."""
        assert self._mu is not None and self._sd is not None, "fit() first"
        z = abs((float(x_t[IDX_MEAN_ENTROPY]) - self._mu) / self._sd)
        if self._ewma is None:
            self._ewma = z
        else:
            self._ewma = self.ewma_alpha * z + (1.0 - self.ewma_alpha) * self._ewma
        return float(self._ewma)


class MahalanobisMonitor(OnlineMonitor):
    """Memoryless Mahalanobis distance under a Ledoit-Wolf healthy covariance.

    fit() pools standardized healthy train steps and estimates a shrunk
    (Ledoit-Wolf) covariance; score_t is the Mahalanobis distance of the
    standardized x_t via the cached precision matrix.
    """

    name = "mahalanobis"

    def __init__(self, standardizer: Standardizer) -> None:
        self.standardizer = standardizer
        self._loc: np.ndarray | None = None
        self._precision: np.ndarray | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Ledoit-Wolf covariance on pooled standardized healthy steps."""
        Z = self.standardizer.transform(
            np.concatenate([ep.X for ep in healthy_episodes], axis=0)
        )
        lw = LedoitWolf().fit(Z)
        self._loc = lw.location_
        self._precision = lw.precision_

    def start_episode(self) -> None:
        """Memoryless: nothing to reset."""

    def score_step(self, x_t: np.ndarray) -> float:
        """Mahalanobis distance of the standardized step signal."""
        assert self._precision is not None, "fit() first"
        z = self.standardizer.transform(np.asarray(x_t, dtype=float))
        d = z - self._loc
        return float(np.sqrt(max(float(d @ self._precision @ d), 0.0)))


class DeltaMahalanobisMonitor(OnlineMonitor):
    """Mahalanobis distance on [z_t ; z_t - z_{t-1}] (86-dim, Ledoit-Wolf).

    A strong, near-memoryless 1-lag baseline that makes H1 falsifiable:
    identical to MahalanobisMonitor but the feature vector appends the
    standardized step-to-step delta, with x_{-1} := x_0 (first delta is zero).
    Deltas are computed within episodes only, both in fit() and while
    streaming. It is roughly 51x cheaper to score than the primary ESN monitor
    and beats it on at least one real tool-cascade metric, so H1 is the neutral
    question of whether the ESN's added recurrent state buys detection/AUROC --
    not a contest against a deliberately weak opponent.
    """

    name = "delta_mahalanobis"

    def __init__(self, standardizer: Standardizer) -> None:
        self.standardizer = standardizer
        self._loc: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._prev_z: np.ndarray | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Ledoit-Wolf covariance on pooled [z_t ; dz_t] healthy features."""
        feats = []
        for ep in healthy_episodes:
            Z = self.standardizer.transform(ep.X)
            dZ = np.diff(Z, axis=0, prepend=Z[:1])  # first-row delta = 0
            feats.append(np.hstack([Z, dZ]))
        lw = LedoitWolf().fit(np.concatenate(feats, axis=0))
        self._loc = lw.location_
        self._precision = lw.precision_

    def start_episode(self) -> None:
        """Reset the previous-step buffer for a new episode."""
        self._prev_z = None

    def score_step(self, x_t: np.ndarray) -> float:
        """Mahalanobis distance of [z_t ; z_t - z_{t-1}] (delta 0 at t=0)."""
        assert self._precision is not None, "fit() first"
        z = self.standardizer.transform(np.asarray(x_t, dtype=float))
        dz = np.zeros_like(z) if self._prev_z is None else z - self._prev_z
        self._prev_z = z
        d = np.concatenate([z, dz]) - self._loc
        return float(np.sqrt(max(float(d @ self._precision @ d), 0.0)))


class IsolationForestMonitor(OnlineMonitor):
    """Isolation forest over pooled standardized healthy steps (memoryless).

    score_t = -decision_function(z_t): higher = more anomalous. Randomness is
    derived via rng_for(seed, "iforest") so reruns are deterministic.
    """

    name = "iforest"

    def __init__(self, standardizer: Standardizer, n_estimators: int = 200,
                 seed: int = 0) -> None:
        self.standardizer = standardizer
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self._forest: IsolationForest | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """Fit the forest on pooled standardized healthy train steps."""
        Z = self.standardizer.transform(
            np.concatenate([ep.X for ep in healthy_episodes], axis=0)
        )
        random_state = int(rng_for(self.seed, "iforest").integers(2**31 - 1))
        self._forest = IsolationForest(
            n_estimators=self.n_estimators, random_state=random_state
        ).fit(Z)

    def start_episode(self) -> None:
        """Memoryless: nothing to reset."""

    def score_step(self, x_t: np.ndarray) -> float:
        """-decision_function of the standardized step (single-row call)."""
        assert self._forest is not None, "fit() first"
        z = self.standardizer.transform(np.asarray(x_t, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn feature-name warnings
            return float(-self._forest.decision_function(z.reshape(1, -1))[0])

    def score_episode(self, episode: Episode) -> np.ndarray:
        """Batched scoring, numerically identical to per-step streaming.

        The scorer is memoryless, so one vectorized decision_function call
        over the whole episode returns exactly the per-step stream — this
        override only removes the per-call sklearn overhead (still causal).
        """
        assert self._forest is not None, "fit() first"
        self.start_episode()
        Z = self.standardizer.transform(episode.X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(-self._forest.decision_function(Z), dtype=float)


if __name__ == "__main__":
    import time

    t0 = time.time()
    rng = rng_for(0, "baselines", "smoke")

    def _make_X(T: int, shift_at: int | None = None) -> np.ndarray:
        """Synthetic healthy telemetry; optional blatant shift from shift_at."""
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
            E[s] = -base[None, :] + 0.6 * rng.normal(size=(n, D_SEM))
            E[s] /= np.linalg.norm(E[s], axis=1, keepdims=True)
            mean_ent[s] += 2.5
            max_ent[s] += 3.0
            slope[s] += 0.4
            frac[s] = 0.95
            lat[s] += 2.5
            err[s] = 1.0
        return np.column_stack(
            [E, mean_ent, max_ent, slope, frac, onehot, lat, outl, err]
        )

    train = [
        Episode(X=_make_X(int(rng.integers(30, 50))), episode_id=f"train-{i:04d}",
                is_healthy=True, failure_class=None, tau=None, t_fail=None,
                severity=None)
        for i in range(30)
    ]
    std = Standardizer().fit(train)

    T, tau = 45, 20
    healthy_ep = Episode(X=_make_X(T), episode_id="test-h", is_healthy=True,
                         failure_class=None, tau=None, t_fail=None, severity=None)
    shifted_ep = Episode(X=_make_X(T, shift_at=tau), episode_id="test-f",
                         is_healthy=False, failure_class="goal_drift", tau=tau,
                         t_fail=T - 1, severity=1.0)

    monitors: list[OnlineMonitor] = [
        CosineDriftMonitor(),
        SelfDriftMonitor(),
        RollingSurprisalMonitor(),
        MahalanobisMonitor(std),
        DeltaMahalanobisMonitor(std),
        IsolationForestMonitor(std, n_estimators=100, seed=0),
    ]
    for mon in monitors:
        mon.fit(train)
        s_h = mon.score_episode(healthy_ep)
        s_f = mon.score_episode(shifted_ep)
        for s in (s_h, s_f):
            assert s.shape == (T,) and np.all(np.isfinite(s)), mon.name
        post_h, post_f = s_h[tau:].mean(), s_f[tau:].mean()
        assert post_f > post_h, (mon.name, post_f, post_h)
        print(f"  {mon.name:<18} post-shift mean {post_f:7.3f}  "
              f"vs healthy {post_h:7.3f}")

    cd_scores = monitors[0].score_episode(healthy_ep)
    assert np.all(cd_scores[: monitors[0].warmup] == 0.0), "warmup must score 0"

    ifm = monitors[-1]
    ifm.start_episode()
    streamed = np.array([ifm.score_step(x) for x in shifted_ep.X])
    assert np.allclose(streamed, ifm.score_episode(shifted_ep)), \
        "iforest batch override must equal per-step streaming"

    ifm2 = IsolationForestMonitor(std, n_estimators=100, seed=0)
    ifm2.fit(train)
    assert np.allclose(ifm.score_episode(shifted_ep),
                       ifm2.score_episode(shifted_ep)), "iforest not deterministic"

    print(f"PASS baselines smoke test ({time.time() - t0:.1f}s, "
          f"{len(monitors)} monitors)")
