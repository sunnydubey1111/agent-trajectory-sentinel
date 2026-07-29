"""Hierarchical Multi-Timescale ESN monitor (HMT-ESN) — candidate architecture.

The proposed research contribution, built to be KILLED OR KEPT by measurement
(experiments/run_hmt_ab.py), not by construction.

Architecture, per telemetry channel:

  Layer 1: one reservoir bank per TIMESCALE (leak rates fast -> slow, e.g.
           0.7 / 0.3 / 0.1). A small leak rate integrates history slowly, so
           a slow bank accumulates gradual shifts (the documented slow-drift
           blind spot); a fast bank keeps the abrupt-jump sensitivity.
  Layer 2 (hierarchical): reservoir banks whose INPUT is a fixed random
           projection of the layer-1 states (mean over ensemble members,
           concatenated across timescales). Deeper layers see the channel
           through layer-1's temporal features — DeepESN-style — while the
           self-supervised target stays the same next-input prediction.

Every (channel x layer x timescale) bank is its own detector: one-step
prediction error, robust-z normalized on held-out healthy, one-sided CUSUM.
The monitor emits the MAX over all detectors — the same channel-max wrapper
the fairness study identified as the transferable contribution, extended
along two new axes (timescale, depth).

Ablation cells are configuration, not separate code:
  single           n_layers=1, leak_rates=(0.3,)   (~ existing per-channel ESN)
  multi-timescale  n_layers=1, leak_rates=(0.7, 0.3, 0.1)
  hierarchical     n_layers=2, leak_rates=(0.3,)
  HMT (full)       n_layers=2, leak_rates=(0.7, 0.3, 0.1)

Contract unchanged: one-class (fit sees healthy only), strictly causal
(score at t depends on x_0..x_t plus fit()-time quantities), OnlineMonitor
lifecycle. Nothing in the existing study code is modified.
"""

from __future__ import annotations

import numpy as np

from derail.common import (
    CHANNEL_SLICES,
    Episode,
    OnlineMonitor,
    Standardizer,
    rng_for,
)
from derail.monitor.esn import _SIGMA_FLOOR, _WASHOUT, _robust_loc_scale


class _Bank:
    """One ensemble of K leaky-tanh reservoirs + ridge readout at one
    (layer, timescale). Predicts the channel's next input u_{t+1} from
    [state; own_input; 1]; scores one-step surprise + disagreement."""

    def __init__(self, d_in: int, d_out: int, R: int, K: int, leak: float,
                 spectral_radius: float, input_scale: float, density: float,
                 seed: int, tag: str) -> None:
        self.d_in, self.d_out, self.R, self.K = d_in, d_out, R, K
        self.leak = float(leak)
        self.W = np.empty((K, R, R))
        self.Win = np.empty((K, R, d_in))
        for k in range(K):
            rng = rng_for(seed, "hmt", tag, k)
            Wk = rng.standard_normal((R, R)) * (rng.random((R, R)) < density)
            sr = float(np.max(np.abs(np.linalg.eigvals(Wk))))
            if sr > 0.0:
                Wk *= spectral_radius / sr
            self.W[k] = Wk
            self.Win[k] = (np.where(rng.random((R, d_in)) < 0.5, -1.0, 1.0)
                           * input_scale)
        F = R + d_in + 1
        self._ZtZ = np.zeros((K, F, F))
        self._ZtY = np.zeros((K, F, d_out))
        self._rows = 0
        self.Wout: np.ndarray | None = None        # (K, F, d_out)
        self.sigma_err: np.ndarray | None = None   # (d_out,)
        self.sup_loc, self.sup_scale = 0.0, 1.0
        self.dis_loc, self.dis_scale = 0.0, 1.0
        # per-episode streaming state
        self.H: np.ndarray | None = None
        self.prev_pred: np.ndarray | None = None
        self.cusum = 0.0

    # ------------------------------------------------------ teacher-forced
    def states(self, U: np.ndarray) -> np.ndarray:
        """(T, d_in) -> (T, K, R): state AFTER absorbing U[t]."""
        T = U.shape[0]
        proj = np.einsum("krd,td->tkr", self.Win, U)
        H = np.zeros((self.K, self.R))
        out = np.empty((T, self.K, self.R))
        lk = self.leak
        for t in range(T):
            H = (1.0 - lk) * H + lk * np.tanh(
                np.einsum("krs,ks->kr", self.W, H) + proj[t])
            out[t] = H
        return out

    def _feats(self, Hs: np.ndarray, U: np.ndarray) -> np.ndarray:
        T = U.shape[0]
        Ub = np.broadcast_to(U[:, None, :], (T, self.K, self.d_in))
        return np.concatenate([Hs, Ub, np.ones((T, self.K, 1))], axis=2)

    def reset_accumulators(self) -> None:
        """Zero the ridge sufficient statistics before a fit.

        __init__ set these once; fit() never cleared them, so a refit on
        dataset B trained on A+B while calibrating on B. fit() now
        calls this on every bank first.
        """
        self._ZtZ[...] = 0.0
        self._ZtY[...] = 0.0
        self._rows = 0

    def accumulate(self, U_in: np.ndarray, Y: np.ndarray) -> None:
        """Add one episode's ridge rows: feature at t=2..T-2 -> target x_{t+1}
        at t=3..T-1, matching the first transition the held-out normalisation
        and streaming score (feature t=2 -> x_3). The old start (t=3 -> x_4)
        never trained the first scored transition."""
        T = U_in.shape[0]
        if T < _WASHOUT + 1:
            return
        Z = self._feats(self.states(U_in), U_in)
        Zr, Yr = Z[_WASHOUT - 1:T - 1], Y[_WASHOUT:T]
        self._rows += Zr.shape[0]
        for k in range(self.K):
            Zk = Zr[:, k, :]
            self._ZtZ[k] += Zk.T @ Zk
            self._ZtY[k] += Zk.T @ Yr

    def solve(self, ridge_lambda: float) -> None:
        if self._rows == 0:
            raise ValueError("HMT bank: no usable ridge rows (episodes too short)")
        F = self._ZtZ.shape[1]
        A = self._ZtZ + ridge_lambda * np.eye(F)[None, :, :]
        self.Wout = np.linalg.solve(A, self._ZtY)

    def heldout_pairs(self, U_in: np.ndarray, Y: np.ndarray):
        """Streaming-aligned (preds (n,K,d_out), targets (n,d_out)) or None."""
        T = U_in.shape[0]
        if T < _WASHOUT + 1:
            return None
        Z = self._feats(self.states(U_in), U_in)
        preds = np.einsum("tkf,kfd->tkd", Z, self.Wout)
        return preds[_WASHOUT - 1:T - 1], Y[_WASHOUT:T]

    # ---------------------------------------------------------- streaming
    def reset(self) -> None:
        self.H = np.zeros((self.K, self.R))
        self.prev_pred = None
        self.cusum = 0.0

    def step(self, x_in: np.ndarray, u_now: np.ndarray, t: int,
             beta: float, k_drift: float) -> float:
        """Score prev prediction against u_now, then absorb x_in and predict.

        Returns this bank's CUSUM after the update (0.0 during washout)."""
        if t >= _WASHOUT and self.prev_pred is not None:
            err = (self.prev_pred - u_now[None, :]) / self.sigma_err
            raw_sup = float(np.mean(err * err))
            raw_dis = float(np.mean(
                (self.prev_pred / self.sigma_err).std(axis=0, ddof=0)))
            z = ((raw_sup - self.sup_loc) / self.sup_scale
                 + beta * (raw_dis - self.dis_loc) / self.dis_scale)
            self.cusum = max(0.0, self.cusum + z - k_drift)
        lk = self.leak
        self.H = (1.0 - lk) * self.H + lk * np.tanh(
            np.einsum("krs,ks->kr", self.W, self.H)
            + np.einsum("krd,d->kr", self.Win, x_in))
        Z = np.concatenate(
            [self.H, np.broadcast_to(x_in, (self.K, self.d_in)),
             np.ones((self.K, 1))], axis=1)
        self.prev_pred = np.einsum("kf,kfd->kd", Z, self.Wout)
        return self.cusum


class HMTESNMonitor(OnlineMonitor):
    """Hierarchical multi-timescale ESN detectors, max-fused across
    (channel x layer x timescale). One-class, causal, cheap."""

    def __init__(
        self,
        standardizer: Standardizer,
        channels: tuple[str, ...] = ("e", "u", "m"),
        leak_rates: tuple[float, ...] = (0.7, 0.3, 0.1),
        n_layers: int = 2,
        K: int = 4,
        reservoir_size: int = 96,
        d_proj: int = 16,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        density: float = 0.1,
        ridge_lambda: float = 1e-2,
        beta_disagreement: float = 0.5,
        cusum_k: float = 0.5,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        assert n_layers in (1, 2), "n_layers must be 1 or 2"
        self.standardizer = standardizer
        self.channels = tuple(channels)
        self.leak_rates = tuple(leak_rates)
        self.n_layers = int(n_layers)
        self.beta = float(beta_disagreement)
        self.k_drift = float(cusum_k)
        self.ridge_lambda = float(ridge_lambda)
        self.name = (name if name is not None else
                     f"hmt[{','.join(channels)}]L{n_layers}"
                     f"T{len(self.leak_rates)}K{K}")

        self._cols: dict[str, np.ndarray] = {}
        self._l1: dict[str, list[_Bank]] = {}
        self._l2: dict[str, list[_Bank]] = {}
        self._proj: dict[str, np.ndarray] = {}
        n_ts, R = len(self.leak_rates), reservoir_size
        for ci, c in enumerate(self.channels):
            sl = CHANNEL_SLICES[c]
            cols = np.arange(sl.start, sl.stop)
            D = cols.size
            self._cols[c] = cols
            self._l1[c] = [
                _Bank(D, D, R, K, lr, spectral_radius, input_scale, density,
                      seed, f"{c}-l1-{ti}")
                for ti, lr in enumerate(self.leak_rates)]
            if self.n_layers == 2:
                rng = rng_for(seed, "hmt", "proj", c)
                self._proj[c] = (rng.standard_normal((n_ts * R, d_proj))
                                 / np.sqrt(n_ts * R))
                self._l2[c] = [
                    _Bank(d_proj, D, R, K, lr, spectral_radius, input_scale,
                          density, seed, f"{c}-l2-{ti}")
                    for ti, lr in enumerate(self.leak_rates)]
            else:
                self._l2[c] = []
        self._t = 0

    def _all_banks(self):
        for c in self.channels:
            for b in self._l1[c]:
                yield b
            for b in self._l2[c]:
                yield b

    def _summary_seq(self, c: str, states_per_bank: list[np.ndarray]
                     ) -> np.ndarray:
        """Layer-2 input sequence: concat over timescales of the mean-over-
        members layer-1 state at each t, through the fixed projection."""
        mean_states = [S.mean(axis=1) for S in states_per_bank]  # (T,R) each
        cat = np.concatenate(mean_states, axis=1)                # (T, n_ts*R)
        return cat @ self._proj[c]                               # (T, d_proj)

    # ------------------------------------------------------------------ fit
    def fit(self, healthy_episodes: list[Episode]) -> None:
        if len(healthy_episodes) < 2:
            raise ValueError("HMTESNMonitor.fit needs >= 2 healthy episodes")
        perm = rng_for(0, "hmt", "split").permutation(len(healthy_episodes))
        n_fit = min(max(int(round(0.85 * len(healthy_episodes))), 1),
                    len(healthy_episodes) - 1)
        fit_eps = [healthy_episodes[i] for i in perm[:n_fit]]
        held_eps = [healthy_episodes[i] for i in perm[n_fit:]]

        # Reset every bank's ridge accumulators so a refit does not train on
        # the union of this and the previous dataset.
        for bank in self._all_banks():
            bank.reset_accumulators()

        # --- ridge accumulation on the 85% ---
        for ep in fit_eps:
            Xs = self.standardizer.transform(ep.X)
            for c in self.channels:
                U = Xs[:, self._cols[c]]
                for bank in self._l1[c]:
                    bank.accumulate(U, U)
                if self._l2[c]:
                    S = self._summary_seq(
                        c, [b.states(U) for b in self._l1[c]])
                    for bank in self._l2[c]:
                        bank.accumulate(S, U)
        for bank in self._all_banks():
            bank.solve(self.ridge_lambda)

        # --- held-out 15%: residual stds + robust normalizers per bank ---
        pairs: dict[int, list] = {id(b): [] for b in self._all_banks()}
        for ep in held_eps:
            Xs = self.standardizer.transform(ep.X)
            for c in self.channels:
                U = Xs[:, self._cols[c]]
                for bank in self._l1[c]:
                    pr = bank.heldout_pairs(U, U)
                    if pr is not None:
                        pairs[id(bank)].append(pr)
                if self._l2[c]:
                    S = self._summary_seq(
                        c, [b.states(U) for b in self._l1[c]])
                    for bank in self._l2[c]:
                        pr = bank.heldout_pairs(S, U)
                        if pr is not None:
                            pairs[id(bank)].append(pr)
        for bank in self._all_banks():
            got = pairs[id(bank)]
            if not got:
                raise ValueError("HMT: no usable held-out steps")
            P = np.concatenate([p for p, _ in got], axis=0)   # (N, K, D)
            Y = np.concatenate([y for _, y in got], axis=0)   # (N, D)
            resid = P - Y[:, None, :]
            bank.sigma_err = np.maximum(resid.std(axis=(0, 1)), _SIGMA_FLOOR)
            rn = resid / bank.sigma_err
            raw_sup = np.mean(rn * rn, axis=(1, 2))
            pn = P / bank.sigma_err
            raw_dis = np.mean(pn.std(axis=1, ddof=0), axis=1)
            bank.sup_loc, bank.sup_scale = _robust_loc_scale(raw_sup)
            bank.dis_loc, bank.dis_scale = _robust_loc_scale(raw_dis)

    # ------------------------------------------------------- streaming API
    def start_episode(self) -> None:
        for bank in self._all_banks():
            if bank.Wout is None:
                raise RuntimeError(f"{self.name}: fit() before scoring")
            bank.reset()
        self._t = 0

    def score_step(self, x_t: np.ndarray) -> float:
        Xs = self.standardizer.transform(np.asarray(x_t, dtype=float))
        best = 0.0
        n_ts, R = len(self.leak_rates), self._l1[self.channels[0]][0].R
        for c in self.channels:
            u = Xs[self._cols[c]]
            for bank in self._l1[c]:
                best = max(best, bank.step(u, u, self._t,
                                           self.beta, self.k_drift))
            if self._l2[c]:
                # summary from the just-advanced layer-1 states (causal)
                cat = np.concatenate(
                    [b.H.mean(axis=0) for b in self._l1[c]])
                s = cat @ self._proj[c]
                for bank in self._l2[c]:
                    best = max(best, bank.step(s, u, self._t,
                                               self.beta, self.k_drift))
        self._t += 1
        return best


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import time

    from derail.common import D_TOTAL, MASTER_SEED

    t0 = time.time()

    def make_episode(idx: int, perturb_after: int | None = None,
                     slow_drift: bool = False) -> Episode:
        rng = rng_for(MASTER_SEED, "hmt-smoke", idx)
        T = int(rng.integers(35, 51))
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        drift = rng.standard_normal(D_TOTAL)
        drift /= np.linalg.norm(drift)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
            if perturb_after is not None and t > perturb_after:
                if slow_drift:   # small persistent shift that grows slowly
                    X[t] = X[t] + 0.12 * (t - perturb_after) * drift
                else:            # abrupt iid noise
                    X[t] = X[t] + 1.2 * rng.standard_normal(D_TOTAL)
        return Episode(X=X, episode_id=f"hmt-smoke-{idx:03d}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [make_episode(i) for i in range(30)]
    std = Standardizer().fit(train)

    # All four ablation cells construct, fit, and score.
    cells = {
        "single": dict(leak_rates=(0.3,), n_layers=1, K=8),
        "mt":     dict(leak_rates=(0.7, 0.3, 0.1), n_layers=1, K=4),
        "h":      dict(leak_rates=(0.3,), n_layers=2, K=4),
        "hmt":    dict(leak_rates=(0.7, 0.3, 0.1), n_layers=2, K=4),
    }
    clean = make_episode(900)
    abrupt = make_episode(901, perturb_after=20)
    slow = make_episode(902, perturb_after=15, slow_drift=True)

    for label, cfg in cells.items():
        mon = HMTESNMonitor(std, seed=0, **cfg)
        mon.fit(train)
        s_clean = mon.score_episode(clean)
        assert s_clean.shape == (clean.T,) and np.all(np.isfinite(s_clean))
        assert np.all(s_clean[:_WASHOUT] == 0.0), "washout must score 0"
        assert np.all(s_clean >= 0.0), "CUSUM must be >= 0"
        assert np.array_equal(mon.score_episode(clean), s_clean), \
            "rescoring not deterministic"
        s_ab = mon.score_episode(abrupt)
        pre = float(np.mean(s_ab[_WASHOUT:21]))
        post = float(np.mean(s_ab[21:]))
        assert post > pre, f"{label}: abrupt perturbation not elevated"
        s_sl = mon.score_episode(slow)
        pre_s = float(np.mean(s_sl[_WASHOUT:16]))
        post_s = float(np.mean(s_sl[16:]))
        assert post_s > pre_s, f"{label}: slow drift not elevated"
        print(f"  {label:6s} ok  clean_max={s_clean.max():7.2f}  "
              f"abrupt_post={post:9.2f}  slowdrift_post={post_s:9.2f}")

    print(f"PASS hmt_esn.py smoke test in {time.time() - t0:.1f}s "
          f"(4 ablation cells: causal, deterministic, responsive)")
