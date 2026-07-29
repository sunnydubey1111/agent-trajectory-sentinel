"""Echo-state-network ensemble monitor (DESIGN.md Module 2).

An ensemble of K leaky-tanh echo-state networks, each predicting the NEXT
standardized (channel-selected) step signal from a ridge readout over
[h_t; x_t; 1]. Streaming anomaly evidence per step:

  - surprise: how badly the ensemble's one-step-ahead prediction (made at the
    PREVIOUS step) matches the just-arrived input, normalized by per-dim
    healthy residual stds;
  - disagreement: spread of the ensemble members' predictions for the current
    input, in the same residual-normalized space.

Both raw streams are robustly z-normalized (median / IQR location-scale
estimated on held-out healthy episodes inside fit()), fused as
z(surprise) + beta * z(disagreement), and EWMA-smoothed. With ``cusum=True``
the emitted streams are instead one-sided CUSUM accumulators
c_t = max(0, c_{t-1} + raw_t - cusum_k) over the same fused/component
z-scores — the standard sequential change-point statistic, which integrates
small persistent shifts (e.g. slow goal drift) that a short-memory EWMA
forgets. Everything emitted by score_step at step t depends only on
x_0..x_t plus fit()-time quantities: strictly causal, no lookahead, no
full-episode statistics.

The single-ESN ablation is just ESNEnsembleMonitor(K=1, name="esn_single");
no extra class is needed.
"""

from __future__ import annotations

import numpy as np

from derail.common import (
    CHANNEL_SLICES,
    DEGENERATE_EPS,
    Episode,
    OnlineMonitor,
    Standardizer,
    rng_for,
)

_WASHOUT = 3          # steps 0..2 emit score 0 and are skipped in ridge fitting
_IQR_TO_STD = 1.349   # IQR of a standard normal; converts IQR to a std-like scale
_SCALE_FLOOR = 1e-6   # floor for robust scales (degenerate distributions)
#: Fixed seed for the fit/held calibration split, shared by every monitor so a
#: model comparison is not confounded by different calibration data.
_MONITOR_SPLIT_SEED = 20260713
_SIGMA_FLOOR = 1e-3   # floor for per-dim residual stds


def _robust_loc_scale(values: np.ndarray) -> tuple[float, float]:
    """Median / IQR-based location-scale of a 1-D sample.

    A constant sample has IQR 0; flooring the scale would make an
    uninformative stream maximally sensitive, so it is left unscaled
    (common.safe_scale). Samples with real spread are unchanged.
    """
    loc = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr_scale = float(q75 - q25) / _IQR_TO_STD
    if iqr_scale < DEGENERATE_EPS:
        return loc, 1.0
    return loc, max(iqr_scale, _SCALE_FLOOR)


class ESNEnsembleMonitor(OnlineMonitor):
    """Causal ESN-ensemble derailment monitor over selected telemetry channels.

    Parameters
    ----------
    standardizer : fitted (by the runner) per-dim z-scorer over the full x_t.
    channels : which channel groups of x_t the monitor sees ("e", "u", "m"),
        selected via common.CHANNEL_SLICES AFTER standardizer.transform and
        concatenated in the given order (H2 ablation knob).
    K : ensemble size (K=1 gives the single-ESN ablation).
    reservoir_size, spectral_radius, leak_rate, input_scale, density :
        reservoir hyperparameters; input weights are random signs * input_scale.
        NOTE: `density` only zeros a fraction of the recurrent weight
        ENTRIES - the reservoir is still STORED and OPERATED as a dense matrix
        (dense multiply, dense eigendecomposition), so a low density does not
        reduce memory or per-step cost. It is a connectivity/dynamics knob, not
        a performance control, and is not presented as one.
    ridge_lambda : L2 penalty of the next-input ridge readout.
    beta_disagreement : weight of z(disagreement) in the fused score.
    ewma_alpha : causal EWMA smoothing factor of the emitted streams.
    cusum : if True, emit one-sided CUSUM accumulators of the fused /
        component z-scores instead of their EWMAs (drift allowance cusum_k
        per step, in z-units). Catches slow persistent shifts.
    cusum_k : per-step drift allowance of the CUSUM recursion.
    seed : all randomness derives from rng_for(seed, "esn", ...).
    """

    def __init__(
        self,
        standardizer: Standardizer,
        channels: tuple[str, ...] = ("e", "u", "m"),
        K: int = 8,
        reservoir_size: int = 128,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scale: float = 0.5,
        density: float = 0.1,
        ridge_lambda: float = 1e-2,
        beta_disagreement: float = 0.5,
        ewma_alpha: float = 0.35,
        cusum: bool = False,
        cusum_k: float = 0.5,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        self.standardizer = standardizer
        self.channels = tuple(channels)
        self.K = int(K)
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.input_scale = float(input_scale)
        self.density = float(density)
        self.ridge_lambda = float(ridge_lambda)
        self.beta_disagreement = float(beta_disagreement)
        self.ewma_alpha = float(ewma_alpha)
        self.cusum = bool(cusum)
        self.cusum_k = float(cusum_k)
        self.seed = int(seed)
        # Constructor range validation: reject values that produce
        # oscillatory/degenerate behaviour or a shape error only later at
        # scoring time.
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.reservoir_size < 1:
            raise ValueError(f"reservoir_size must be >= 1, got {self.reservoir_size}")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError(f"density must be in [0, 1], got {self.density}")
        if self.leak_rate <= 0.0 or self.leak_rate > 1.0:
            raise ValueError(f"leak_rate must be in (0, 1], got {self.leak_rate}")
        if self.ridge_lambda < 0.0:
            raise ValueError(f"ridge_lambda must be >= 0, got {self.ridge_lambda}")
        if name is None:
            name = f"esn[{','.join(self.channels)}]K{self.K}"
            if self.cusum:
                name += "-cusum"
        self.name = name

        cols = [np.arange(CHANNEL_SLICES[c].start, CHANNEL_SLICES[c].stop) for c in self.channels]
        self._cols = np.concatenate(cols)
        self.D = int(self._cols.size)

        # Fitted state (set in fit()).
        self._W: np.ndarray | None = None        # (K, R, R) reservoirs
        self._Win: np.ndarray | None = None      # (K, R, D) input weights
        self._Wout: np.ndarray | None = None     # (K, F, D) ridge readouts
        self._sigma_err: np.ndarray | None = None  # (D,) healthy residual stds
        self._sup_loc = 0.0
        self._sup_scale = 1.0
        self._dis_loc = 0.0
        self._dis_scale = 1.0

        # Per-episode streaming state (set in start_episode()).
        self._H: np.ndarray | None = None        # (K, R) reservoir states
        self._prev_pred: np.ndarray | None = None  # (K, D) prediction for current step
        self._t = 0
        self._ewma = (0.0, 0.0, 0.0)             # (fused, surprise, disagreement)
        self._cusum_state = (0.0, 0.0, 0.0)      # CUSUM accumulators, same order

    # ------------------------------------------------------------------ fit

    def fit(self, healthy_episodes: list[Episode]) -> None:
        """One-class fit on healthy episodes.

        Deterministically splits episodes 85/15 (rng_for(seed, "esn",
        "split")). The ridge readouts are fit on the 85% via teacher-forced
        reservoir runs (skipping the washout = first 3 steps of each episode).
        The held-out 15% then provides (a) per-dim residual stds sigma_err
        (floored at 1e-3) and (b) the healthy distributions of raw surprise
        and raw disagreement, computed with exactly the streaming alignment
        of score_step, from which robust median/IQR normalizers are derived.
        Nothing in scoring uses statistics of the episode being scored.
        """
        if len(healthy_episodes) < 2:
            raise ValueError("ESNEnsembleMonitor.fit needs >= 2 healthy episodes")
        self._init_weights()

        # Shared fit/held split INDEPENDENT of the model seed: every
        # monitor - each ESN member, each channel, and the sequence baselines -
        # must calibrate and normalise on the SAME held-out episodes, else a
        # model comparison is confounded by different calibration data. The
        # split seed is fixed, not self.seed.
        perm = rng_for(_MONITOR_SPLIT_SEED, "monitor", "fit-held-split"
                       ).permutation(len(healthy_episodes))
        n_fit = min(max(int(round(0.85 * len(healthy_episodes))), 1), len(healthy_episodes) - 1)
        fit_eps = [healthy_episodes[i] for i in perm[:n_fit]]
        held_eps = [healthy_episodes[i] for i in perm[n_fit:]]

        # --- ridge readout on the 85% (teacher forced, washout skipped) ---
        F = self.reservoir_size + self.D + 1
        ZtZ = np.zeros((self.K, F, F))
        ZtY = np.zeros((self.K, F, self.D))
        n_rows = 0
        for ep in fit_eps:
            U = self._inputs(ep)
            T = U.shape[0]
            if T < _WASHOUT + 1:
                continue
            Z = self._features(self._states(U), U)   # (T, K, F)
            # Train from washout-1 so the readout learns the SAME first scored
            # transition that held-out normalization and streaming serve:
            # feature at t=2 -> target x_3. The old start (t=3 -> x_4) never
            # trained on the t=2->t=3 transition, yet that is the first
            # transition scored at serving time - the readout was extrapolating
            # by one step on every episode's first scored step.
            Zr = Z[_WASHOUT - 1 : T - 1]             # features at t = 2..T-2
            Yr = U[_WASHOUT : T]                     # targets x_{t+1} at 3..T-1
            n_rows += Zr.shape[0]
            for k in range(self.K):
                Zk = Zr[:, k, :]
                ZtZ[k] += Zk.T @ Zk
                ZtY[k] += Zk.T @ Yr
        if n_rows == 0:
            raise ValueError("no usable ridge-training steps (episodes too short)")
        A = ZtZ + self.ridge_lambda * np.eye(F)[None, :, :]
        self._Wout = np.linalg.solve(A, ZtY)         # (K, F, D)

        # --- held-out 15%: residual stds + surprise/disagreement normalizers ---
        preds_all: list[np.ndarray] = []
        resid_all: list[np.ndarray] = []
        for ep in held_eps:
            U = self._inputs(ep)
            T = U.shape[0]
            if T < _WASHOUT + 1:
                continue
            Z = self._features(self._states(U), U)
            preds = np.einsum("tkf,kfd->tkd", Z, self._Wout)
            # Streaming alignment: the score at step t (t >= washout) uses the
            # prediction made at t-1; so pair feature times 2..T-2 with
            # targets 3..T-1.
            P = preds[_WASHOUT - 1 : T - 1]          # (n, K, D)
            Y = U[_WASHOUT:T]                        # (n, D)
            preds_all.append(P)
            resid_all.append(P - Y[:, None, :])
        if not resid_all:
            raise ValueError("no usable held-out steps (episodes too short)")
        resid = np.concatenate(resid_all, axis=0)    # (N, K, D)
        # Per-dim residual std. A dim the reservoirs predict EXACTLY on healthy
        # data has residual std 0; it is left unscaled rather than divided by
        # _SIGMA_FLOOR, which would amplify its first deviation. common.safe_scale.
        _sd = resid.std(axis=(0, 1))
        self._sigma_err = np.where(_sd < DEGENERATE_EPS, 1.0,
                                   np.maximum(_sd, _SIGMA_FLOOR))

        resid_n = resid / self._sigma_err
        raw_surprise = np.mean(resid_n * resid_n, axis=(1, 2))          # (N,)
        preds_n = np.concatenate(preds_all, axis=0) / self._sigma_err
        raw_disagreement = np.mean(preds_n.std(axis=1, ddof=0), axis=1)  # (N,)
        self._sup_loc, self._sup_scale = _robust_loc_scale(raw_surprise)
        self._dis_loc, self._dis_scale = _robust_loc_scale(raw_disagreement)

    # ------------------------------------------------------- streaming API

    def start_episode(self) -> None:
        """Reset per-episode streaming state (reservoirs, prediction, EWMAs)."""
        if self._Wout is None:
            raise RuntimeError(f"{self.name}: fit() must be called before scoring")
        self._H = np.zeros((self.K, self.reservoir_size))
        self._prev_pred = None
        self._t = 0
        self._ewma = (0.0, 0.0, 0.0)
        self._cusum_state = (0.0, 0.0, 0.0)

    def score_step(self, x_t: np.ndarray) -> float:
        """Consume x_t, emit the causal fused derailment score s_t.

        At step t the surprise compares the prediction made at t-1 against the
        just-arrived x_t; then the reservoirs absorb x_t and predict x_{t+1}
        for the next call. Washout steps (0..2, including the first step)
        emit 0.0.
        """
        if self._H is None:
            self.start_episode()
        u = self.standardizer.transform(np.asarray(x_t, dtype=float))[self._cols]
        return self._advance(u)[0]

    def score_episode_components(self, episode: Episode) -> dict[str, np.ndarray]:
        """Score an episode, returning all three causal EWMA-smoothed streams.

        Returns {"fused": (T,), "surprise": (T,), "disagreement": (T,)}:
        the EWMA (or, with cusum=True, the one-sided CUSUM) of
        z(surprise) + beta * z(disagreement), of z(surprise), and of
        z(disagreement) respectively — each computable online (value at t
        depends only on x_0..x_t). "fused" is identical to the score_step
        stream.
        """
        self.start_episode()
        U = self.standardizer.transform(episode.X)[:, self._cols]
        out = np.zeros((episode.T, 3))
        for t in range(episode.T):
            out[t] = self._advance(U[t])
        return {"fused": out[:, 0], "surprise": out[:, 1], "disagreement": out[:, 2]}

    # ----------------------------------------------------------- internals

    def _init_weights(self) -> None:
        """Draw per-member reservoir + input weights via rng_for(seed, "esn", k)."""
        R, D = self.reservoir_size, self.D
        self._W = np.empty((self.K, R, R))
        self._Win = np.empty((self.K, R, D))
        for k in range(self.K):
            rng = rng_for(self.seed, "esn", k)
            Wk = rng.standard_normal((R, R)) * (rng.random((R, R)) < self.density)
            sr = float(np.max(np.abs(np.linalg.eigvals(Wk))))
            if sr > 0.0:
                Wk *= self.spectral_radius / sr
            self._W[k] = Wk
            self._Win[k] = np.where(rng.random((R, D)) < 0.5, -1.0, 1.0) * self.input_scale

    def _inputs(self, episode: Episode) -> np.ndarray:
        """Standardize the full x_t then column-select the chosen channels."""
        return self.standardizer.transform(episode.X)[:, self._cols]

    def _states(self, U: np.ndarray) -> np.ndarray:
        """Teacher-forced leaky-tanh reservoir states for all members.

        h_t = (1 - leak) * h_{t-1} + leak * tanh(W h_{t-1} + W_in x_t),
        h_{-1} = 0. Returns (T, K, R): the state AFTER absorbing U[t].
        """
        T = U.shape[0]
        proj = np.einsum("krd,td->tkr", self._Win, U)
        H = np.zeros((self.K, self.reservoir_size))
        out = np.empty((T, self.K, self.reservoir_size))
        lk = self.leak_rate
        for t in range(T):
            H = (1.0 - lk) * H + lk * np.tanh(np.einsum("krs,ks->kr", self._W, H) + proj[t])
            out[t] = H
        return out

    def _features(self, Hs: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Readout features [h_t; x_t; 1] per member: (T, K, R + D + 1)."""
        T = U.shape[0]
        Ub = np.broadcast_to(U[:, None, :], (T, self.K, self.D))
        return np.concatenate([Hs, Ub, np.ones((T, self.K, 1))], axis=2)

    def _advance(self, u: np.ndarray) -> tuple[float, float, float]:
        """One causal step on a standardized, channel-selected input u.

        Scores u against the prediction made last step (0 during washout),
        updates the EWMAs, then absorbs u and predicts the next input.
        Returns (fused, surprise, disagreement) EWMA values for this step.
        """
        if self._t >= _WASHOUT and self._prev_pred is not None:
            err = (self._prev_pred - u[None, :]) / self._sigma_err       # (K, D)
            raw_sup = float(np.mean(err * err))
            raw_dis = float(np.mean((self._prev_pred / self._sigma_err).std(axis=0, ddof=0)))
            z_sup = (raw_sup - self._sup_loc) / self._sup_scale
            z_dis = (raw_dis - self._dis_loc) / self._dis_scale
            fused_raw = z_sup + self.beta_disagreement * z_dis
            a = self.ewma_alpha
            self._ewma = (
                a * fused_raw + (1.0 - a) * self._ewma[0],
                a * z_sup + (1.0 - a) * self._ewma[1],
                a * z_dis + (1.0 - a) * self._ewma[2],
            )
            k = self.cusum_k
            self._cusum_state = (
                max(0.0, self._cusum_state[0] + fused_raw - k),
                max(0.0, self._cusum_state[1] + z_sup - k),
                max(0.0, self._cusum_state[2] + z_dis - k),
            )
            out = self._cusum_state if self.cusum else self._ewma
        else:
            out = (0.0, 0.0, 0.0)

        lk = self.leak_rate
        self._H = (1.0 - lk) * self._H + lk * np.tanh(
            np.einsum("krs,ks->kr", self._W, self._H)
            + np.einsum("krd,d->kr", self._Win, u)
        )
        Z = np.concatenate(
            [self._H, np.broadcast_to(u, (self.K, self.D)), np.ones((self.K, 1))], axis=1
        )
        self._prev_pred = np.einsum("kf,kfd->kd", Z, self._Wout)
        self._t += 1
        return out


class ChannelMaxESNMonitor(OnlineMonitor):
    """Per-channel ESN-CUSUM detectors fused by max (H2-motivated).

    A single ESN averaging surprise over all dims dilutes a shift confined to
    a narrow channel (grounding loss lives in 4 uncertainty dims out of 43).
    This monitor runs one ESNEnsembleMonitor per channel group ("e", "u",
    "m"), each z-normalized against its own healthy statistics, and emits the
    max of the three causal streams. Still one-class, causal, and cheap.
    """

    def __init__(self, standardizer: Standardizer, K: int = 8,
                 cusum: bool = True, seed: int = 0,
                 name: str = "esn_cusum_max", channels: tuple[str, ...] = ("e", "u", "m"), **esn_kwargs) -> None:
        self.name = name
        self.channels = tuple(channels)
        self.subs = [
            ESNEnsembleMonitor(standardizer, channels=(c,), K=K, cusum=cusum,
                               seed=seed * 100 + i, **esn_kwargs)
            for i, c in enumerate(self.channels)
        ]

    def fit(self, healthy_episodes: list[Episode]) -> None:
        for sub in self.subs:
            sub.fit(healthy_episodes)

    def start_episode(self) -> None:
        for sub in self.subs:
            sub.start_episode()

    def score_step(self, x_t: np.ndarray) -> float:
        return max(sub.score_step(x_t) for sub in self.subs)

    def score_episode_components(self, episode: Episode) -> dict[str, np.ndarray]:
        """Per-stream max over the three per-channel component streams."""
        parts = [sub.score_episode_components(episode) for sub in self.subs]
        return {key: np.max([p[key] for p in parts], axis=0)
                for key in ("fused", "surprise", "disagreement")}


class HMTE_ESN_M_Monitor(OnlineMonitor):
    """Hierarchical Multi-timescale Ensemble ESN with Mahalanobis Fusion (HMTE-ESN-M).

    1. Runs three separate ESNEnsembleMonitors (for "e", "u", and "m").
    2. Extracts ESN surprise scores at three timescales. NOTE: the
       base feature is the sub-monitor's OWN EWMA surprise (already smoothed at
       the sub's alpha), NOT the raw instantaneous surprise; the short and long
       features smooth that EWMA again. The three timescales are therefore
       fast-EWMA / medium / slow, not instantaneous / short / long. It is
       labelled honestly rather than claiming an instantaneous channel it does
       not expose.
       - fast-EWMA surprise (the sub-monitor's EWMA)
       - medium moving average (alpha=0.5 of the above)
       - slow moving average (alpha=0.1 of the above)
    3. Fuses the resulting 9-dimensional features using Mahalanobis distance.
    4. Applies a one-sided CUSUM detector on the CENTERED Mahalanobis distance
       (the distance is standardised on the healthy distribution before
       accumulation so the null does not drift).
    """

    def __init__(
        self,
        standardizer: Standardizer,
        K: int = 8,
        reservoir_size: int = 100,
        cusum_k: float = 0.5,
        seed: int = 0,
        name: str = "hmte_esn_m",
        **esn_kwargs
    ) -> None:
        self.name = name
        self.K = K
        self.reservoir_size = reservoir_size
        self.cusum_k = float(cusum_k)
        self.seed = int(seed)
        self.subs = [
            ESNEnsembleMonitor(standardizer, channels=(c,), K=K, reservoir_size=reservoir_size,
                               cusum=False, seed=seed * 100 + i, **esn_kwargs)
            for i, c in enumerate(("e", "u", "m"))
        ]

        # Fit states for Mahalanobis
        self._mu: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None

        # Streaming state
        self._short_ma = np.zeros(3)
        self._long_ma = np.zeros(3)
        self._cusum = 0.0
        self._t = 0

    def fit(self, healthy_episodes: list[Episode]) -> None:
        # 1. Fit the sub-monitors
        for sub in self.subs:
            sub.fit(healthy_episodes)

        # 2. Extract 9-D features over healthy episodes to fit Mahalanobis
        features_all = []
        for ep in healthy_episodes:
            self.start_episode()
            for t in range(ep.X.shape[0]):
                x_t = ep.X[t]
                sups = []
                for sub in self.subs:
                    sub.score_step(x_t)
                    sups.append(sub._ewma[1] if sub._t > _WASHOUT else 0.0)

                sups = np.array(sups)
                if self._t >= _WASHOUT:
                    self._short_ma = 0.5 * sups + 0.5 * self._short_ma
                    self._long_ma = 0.1 * sups + 0.9 * self._long_ma
                    feat = np.concatenate([sups, self._short_ma, self._long_ma])
                    features_all.append(feat)
                self._t += 1

        if not features_all:
            raise ValueError("No usable features extracted to fit Mahalanobis covariance")

        F = np.array(features_all)
        self._mu = F.mean(axis=0)
        cov = np.cov(F.T)
        self._inv_cov = np.linalg.pinv(cov + 1e-4 * np.eye(9))

        # Center/scale the Mahalanobis DISTANCE on the healthy distribution
        # before it drives the CUSUM. A 9-D Mahalanobis distance has
        # a healthy mean of ~sqrt(9)=2.9; feeding it raw into a CUSUM that
        # subtracts only k=0.5 makes the healthy null accumulate ~2.4 per step
        # and fire on everything. Here the healthy distance is standardised to
        # ~0 mean, so the CUSUM's k=0.5 drift keeps the null flat and only
        # genuine excursions accumulate.
        diff = F - self._mu
        dists = np.sqrt(np.clip(np.einsum("ni,ij,nj->n", diff, self._inv_cov,
                                          diff), 0.0, None))
        self._dist_loc, self._dist_scale = _robust_loc_scale(dists)

    def start_episode(self) -> None:
        for sub in self.subs:
            sub.start_episode()
        self._short_ma = np.zeros(3)
        self._long_ma = np.zeros(3)
        self._cusum = 0.0
        self._t = 0

    def score_step(self, x_t: np.ndarray) -> float:
        sups = []
        for sub in self.subs:
            sub.score_step(x_t)
            sups.append(sub._ewma[1] if sub._t > _WASHOUT else 0.0)

        sups = np.array(sups)

        if self._t >= _WASHOUT and self._mu is not None:
            self._short_ma = 0.5 * sups + 0.5 * self._short_ma
            self._long_ma = 0.1 * sups + 0.9 * self._long_ma
            feat = np.concatenate([sups, self._short_ma, self._long_ma])

            diff = feat - self._mu
            dist = float(np.sqrt(np.clip(diff.T @ self._inv_cov @ diff, 0.0, None)))
            # Standardise on the healthy-distance distribution before the CUSUM
            #, so the null does not drift upward.
            z = (dist - self._dist_loc) / self._dist_scale
            self._cusum = max(0.0, self._cusum + z - self.cusum_k)
            out = self._cusum
        else:
            out = 0.0

        self._t += 1
        return out


# --------------------------------------------------------------- smoke test


if __name__ == "__main__":
    import time

    from derail.common import D_TOTAL, MASTER_SEED

    t0 = time.time()

    def make_episode(idx: int, perturb_after: int | None = None) -> Episode:
        """AR(1) random-walk telemetry; optional iid noise after a given step."""
        rng = rng_for(MASTER_SEED, "esn-smoke", idx)
        T = int(rng.integers(35, 51))
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
            if perturb_after is not None and t > perturb_after:
                X[t] = X[t] + 1.2 * rng.standard_normal(D_TOTAL)
        return Episode(X=X, episode_id=f"smoke-{idx:03d}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [make_episode(i) for i in range(30)]
    std = Standardizer().fit(train)
    mon = ESNEnsembleMonitor(std, seed=0)
    assert mon.name == "esn[e,u,m]K8"
    mon.fit(train)

    clean = make_episode(900)
    s_clean = mon.score_episode(clean)
    assert s_clean.shape == (clean.T,), s_clean.shape
    assert np.all(np.isfinite(s_clean)), "non-finite scores"
    assert np.all(s_clean[:_WASHOUT] == 0.0), "washout steps must score 0"

    comp = mon.score_episode_components(clean)
    assert set(comp) == {"fused", "surprise", "disagreement"}
    for key, arr in comp.items():
        assert arr.shape == (clean.T,) and np.all(np.isfinite(arr)), key
    # (2-D vs 1-D numpy vectorization gives ~1e-15 fp noise between the paths)
    assert np.allclose(comp["fused"], s_clean, atol=1e-9), "fused stream != score_step stream"
    assert np.array_equal(mon.score_episode(clean), s_clean), "rescoring not deterministic"

    pert = make_episode(901, perturb_after=20)
    s_pert = mon.score_episode_components(pert)["fused"]
    pre = float(np.mean(s_pert[_WASHOUT:21]))
    post = float(np.mean(s_pert[21:]))
    assert post > pre, f"perturbed episode not scored higher post-onset ({post:.3f} <= {pre:.3f})"

    cus = ESNEnsembleMonitor(std, cusum=True, seed=0)
    assert cus.name == "esn[e,u,m]K8-cusum"
    cus.fit(train)
    sc = cus.score_episode_components(pert)["fused"]
    assert np.all(np.isfinite(sc)) and np.all(sc >= 0.0), "CUSUM must be >= 0"
    assert float(np.mean(sc[21:])) > float(np.mean(sc[_WASHOUT:21])), \
        "CUSUM not elevated post-perturbation"
    assert np.allclose(cus.score_episode(pert), sc, atol=1e-9)

    cmx = ChannelMaxESNMonitor(std, K=4, seed=0)
    cmx.fit(train)
    smx = cmx.score_episode(pert)
    assert smx.shape == (pert.T,) and np.all(np.isfinite(smx))
    cmp_mx = cmx.score_episode_components(pert)
    assert np.allclose(cmp_mx["fused"], smx, atol=1e-9)
    assert float(np.mean(smx[21:])) > float(np.mean(smx[_WASHOUT:21]))

    single = ESNEnsembleMonitor(std, K=1, name="esn_single", seed=0)
    single.fit(train)
    s1 = single.score_episode(pert)
    assert single.name == "esn_single"
    assert np.all(np.isfinite(s1)) and s1.shape == (pert.T,)
    d1 = single.score_episode_components(pert)["disagreement"]
    assert np.allclose(d1, 0.0), "K=1 disagreement stream should be ~0"

    print(
        f"PASS esn.py smoke test in {time.time() - t0:.2f}s | "
        f"clean mean={np.mean(s_clean[_WASHOUT:]):.3f} | "
        f"perturbed pre={pre:.3f} post={post:.3f} | K=1 ok"
    )
