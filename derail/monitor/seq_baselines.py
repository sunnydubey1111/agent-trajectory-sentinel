"""Trained sequence-model baselines (review weakness D).

Answers "why ESN instead of LSTM/GRU/TCN?" empirically, under the SAME
protocol as the ESN family so the comparison is fair:

  - one-class: fit on healthy train episodes only, 85/15 episode split
    (ridge/backprop on the 85%, residual stds + robust surprise normalizer
    from the held-out 15% with the exact streaming alignment);
  - causal: the score at step t compares the prediction made at t-1 with the
    just-arrived x_t; washout steps score 0;
  - identical emission: one-sided CUSUM of z(surprise) with the same
    drift allowance as the primary monitor (no ensemble -> no disagreement
    term).

Monitors:
  LinearARMonitor  ("linear_ar")  ridge vector-autoregression, lag 3 — the
                                  linear control separating "temporal" from
                                  "nonlinear temporal".
  GRUMonitor       ("gru")        1-layer GRU(64) + linear head, Adam/BPTT.
  LSTMMonitor      ("lstm")       1-layer LSTM(64), identical protocol (a
                                  GRU subclass; reported as its own arm).
  TCNMonitor       ("tcn")        causal dilated conv stack (RF 16) + head.

Capacity note: GRU trains ~21k parameters, LSTM ~28k, TCN ~27k; the ESN trains
only its ridge readouts (~59k for K=8, reservoirs untrained). Training
budgets are minutes-scale for all; none was tuned beyond the defaults here,
matching the tuning effort spent on the ESN.

torch is an optional dependency: LinearARMonitor is pure numpy; GRUMonitor /
TCNMonitor raise ImportError at construction if torch is unavailable.
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

# Imported, not re-declared. These three quantities decide whether a
# comparison against the ESN measures the MODEL or measures a difference in
# how the two were calibrated. Local copies drifted apart once already: the
# baselines were splitting fit/held on a per-model seed and flooring
# degenerate scales where the ESN left them unscaled, which is worth enough
# episode AUC on its own to invert a model ranking.
from derail.monitor.esn import (
    _MONITOR_SPLIT_SEED,
    _SIGMA_FLOOR,
    _WASHOUT,
    _robust_loc_scale,
)

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


class _NextStepMonitor(OnlineMonitor):
    """Shared scaffold: predictor-agnostic fit/normalize/stream logic.

    Subclasses implement _train(seqs), _reset_predictor(), and
    _predict_next(u) -> prediction for the NEXT standardized input (called
    after absorbing u). Everything else (split, residual normalization,
    causal scoring, CUSUM emission) lives here and mirrors esn.py.
    """

    def __init__(self, standardizer: Standardizer, cusum_k: float = 0.5,
                 seed: int = 0, name: str = "seq",
                 channels: tuple[str, ...] | None = None) -> None:
        self.standardizer = standardizer
        self.cusum_k = float(cusum_k)
        self.seed = int(seed)
        self.name = name
        # Optional channel restriction (as in the ESN ablation): None = all.
        self._cols = (None if channels is None else
                      np.concatenate([np.arange(CHANNEL_SLICES[c].start,
                                                CHANNEL_SLICES[c].stop)
                                      for c in channels]))
        self._sigma_err: np.ndarray | None = None
        self._sup_loc, self._sup_scale = 0.0, 1.0
        self._prev_pred: np.ndarray | None = None
        self._cusum = 0.0
        self._t = 0

    # -- subclass API -------------------------------------------------
    def _train(self, seqs: list[np.ndarray]) -> None:
        raise NotImplementedError

    def _reset_predictor(self) -> None:
        raise NotImplementedError

    def _predict_next(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- shared logic --------------------------------------------------
    def _prep(self, X: np.ndarray) -> np.ndarray:
        """Standardize, then apply the optional channel restriction."""
        Z = self.standardizer.transform(X)
        return Z if self._cols is None else Z[..., self._cols]

    def fit(self, healthy_episodes: list[Episode]) -> None:
        if len(healthy_episodes) < 2:
            raise ValueError(f"{self.name}: need >= 2 healthy episodes")
        # The SHARED fit/held split seed, not a per-model one. Every monitor
        # in a comparison must calibrate and normalise on the SAME held-out
        # episodes; a per-model split gave GRU, LSTM and TCN each a different
        # 85/15 partition from each other AND from the ESN, so the reported
        # gap measured calibration luck as much as architecture.
        perm = rng_for(_MONITOR_SPLIT_SEED, "monitor", "fit-held-split"
                       ).permutation(len(healthy_episodes))
        n_fit = min(max(int(round(0.85 * len(healthy_episodes))), 1),
                    len(healthy_episodes) - 1)
        fit_eps = [healthy_episodes[i] for i in perm[:n_fit]]
        held_eps = [healthy_episodes[i] for i in perm[n_fit:]]
        self._train([self._prep(ep.X) for ep in fit_eps])

        # Held-out residuals with the exact streaming alignment.
        resid: list[np.ndarray] = []
        for ep in held_eps:
            U = self._prep(ep.X)
            if U.shape[0] < _WASHOUT + 1:
                continue
            self._reset_predictor()
            preds = [self._predict_next(u) for u in U[:-1]]  # pred for t+1
            P = np.asarray(preds[_WASHOUT - 1:])             # scores t>=WASHOUT
            Y = U[_WASHOUT:]
            resid.append(P - Y)
        if not resid:
            raise ValueError(f"{self.name}: no usable held-out steps")
        R = np.concatenate(resid, axis=0)
        # Degenerate-scale contract (DESIGN.md Amendment 6): a dim predicted
        # exactly on healthy data is left unscaled, not divided by the floor,
        # which would amplify its first deviation ~1000x. The ESN applies this
        # guard; the baselines it is compared against must apply the same one.
        _sd = R.std(axis=0)
        self._sigma_err = np.where(_sd < DEGENERATE_EPS, 1.0,
                                   np.maximum(_sd, _SIGMA_FLOOR))
        raw = np.mean((R / self._sigma_err) ** 2, axis=1)
        self._sup_loc, self._sup_scale = _robust_loc_scale(raw)

    def start_episode(self) -> None:
        if self._sigma_err is None:
            raise RuntimeError(f"{self.name}: fit() before scoring")
        self._reset_predictor()
        self._prev_pred = None
        self._cusum = 0.0
        self._t = 0

    def score_step(self, x_t: np.ndarray) -> float:
        u = self._prep(np.asarray(x_t, dtype=float))
        if self._t >= _WASHOUT and self._prev_pred is not None:
            err = (self._prev_pred - u) / self._sigma_err
            z = (float(np.mean(err * err)) - self._sup_loc) / self._sup_scale
            self._cusum = max(0.0, self._cusum + z - self.cusum_k)
            out = self._cusum
        else:
            out = 0.0
        self._prev_pred = self._predict_next(u)
        self._t += 1
        return out


class LinearARMonitor(_NextStepMonitor):
    """Ridge vector autoregression: [x_{t-2}; x_{t-1}; x_t; 1] -> x_{t+1}."""

    def __init__(self, standardizer: Standardizer, lag: int = 3,
                 ridge_lambda: float = 1.0, cusum_k: float = 0.5,
                 seed: int = 0, name: str = "linear_ar") -> None:
        super().__init__(standardizer, cusum_k, seed, name)
        self.lag = int(lag)
        self.ridge_lambda = float(ridge_lambda)
        self._W: np.ndarray | None = None
        self._buf: list[np.ndarray] = []

    def _train(self, seqs: list[np.ndarray]) -> None:
        D = seqs[0].shape[1]
        F = self.lag * D + 1
        ZtZ = np.zeros((F, F))
        ZtY = np.zeros((F, D))
        for U in seqs:
            T = U.shape[0]
            if T < self.lag + 1:
                continue
            Z = np.concatenate(
                [U[i:T - self.lag + i] for i in range(self.lag)]
                + [np.ones((T - self.lag, 1))], axis=1)
            Y = U[self.lag:]
            ZtZ += Z.T @ Z
            ZtY += Z.T @ Y
        self._W = np.linalg.solve(ZtZ + self.ridge_lambda * np.eye(F), ZtY)

    def _reset_predictor(self) -> None:
        self._buf = []

    def _predict_next(self, u: np.ndarray) -> np.ndarray:
        self._buf.append(u)
        if len(self._buf) < self.lag:
            pad = [self._buf[0]] * (self.lag - len(self._buf))
            window = pad + self._buf
        else:
            window = self._buf[-self.lag:]
        z = np.concatenate(window + [np.ones(1)])
        return z @ self._W


class GRUMonitor(_NextStepMonitor):
    """1-layer GRU next-step predictor, trained with Adam/BPTT."""

    def __init__(self, standardizer: Standardizer, hidden: int = 64,
                 epochs: int = 40, lr: float = 3e-3, cusum_k: float = 0.5,
                 seed: int = 0, name: str = "gru",
                 channels: tuple[str, ...] | None = None) -> None:
        if not _HAS_TORCH:
            raise ImportError(f"GRUMonitor requires torch - " + "install the optional torch dependency: pip install torch --index-url https://download.pytorch.org/whl/cpu (or use requirements.lock.txt). GRU/LSTM/TCN baselines need it; the ESN and statistical monitors do not.")
        super().__init__(standardizer, cusum_k, seed, name, channels)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.loss_log: list[float] = []   # mean training MSE per epoch
        self._gru: "nn.GRU | None" = None
        self._head: "nn.Linear | None" = None
        self._h: "torch.Tensor | None" = None

    def _make_rnn(self, D: int) -> "nn.Module":
        return nn.GRU(D, self.hidden, batch_first=True).double()

    def _train(self, seqs: list[np.ndarray]) -> None:
        D = seqs[0].shape[1]
        torch.manual_seed(int(rng_for(self.seed, self.name, "torch")
                              .integers(0, 2**31 - 1)))
        self._gru = self._make_rnn(D)
        self._head = nn.Linear(self.hidden, D).double()
        params = list(self._gru.parameters()) + list(self._head.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        tensors = [torch.from_numpy(U) for U in seqs if U.shape[0] > _WASHOUT + 1]
        self.loss_log = []
        for _ in range(self.epochs):
            epoch_losses = []
            for U in tensors:
                opt.zero_grad()
                out, _ = self._gru(U[None, :-1, :])
                pred = self._head(out[0])
                loss = torch.mean((pred[_WASHOUT - 1:] - U[_WASHOUT:]) ** 2)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach()))
            self.loss_log.append(float(np.mean(epoch_losses)))
        self._gru.eval()
        self._head.eval()

    def _reset_predictor(self) -> None:
        self._h = torch.zeros(1, 1, self.hidden, dtype=torch.float64)

    def _predict_next(self, u: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(u, dtype=float)).view(1, 1, -1)
            out, self._h = self._gru(x, self._h)
            return self._head(out[0, 0]).numpy()


class LSTMMonitor(GRUMonitor):
    """1-layer LSTM next-step predictor (identical protocol to GRUMonitor).

    Included as a named comparator in its own right; at this scale GRU and
    LSTM are near-equivalent (GRU has fewer parameters).
    """

    def __init__(self, standardizer: Standardizer, hidden: int = 64,
                 epochs: int = 40, lr: float = 3e-3, cusum_k: float = 0.5,
                 seed: int = 0, name: str = "lstm",
                 channels: tuple[str, ...] | None = None) -> None:
        super().__init__(standardizer, hidden=hidden, epochs=epochs, lr=lr,
                         cusum_k=cusum_k, seed=seed, name=name,
                         channels=channels)

    def _make_rnn(self, D: int) -> "nn.Module":
        return nn.LSTM(D, self.hidden, batch_first=True).double()

    def _reset_predictor(self) -> None:
        zeros = torch.zeros(1, 1, self.hidden, dtype=torch.float64)
        self._h = (zeros, zeros.clone())   # LSTM hidden state is (h, c)


class TCNMonitor(_NextStepMonitor):
    """Causal dilated-conv next-step predictor (receptive field 16)."""

    def __init__(self, standardizer: Standardizer, channels: int = 48,
                 epochs: int = 40, lr: float = 3e-3, cusum_k: float = 0.5,
                 seed: int = 0, name: str = "tcn") -> None:
        if not _HAS_TORCH:
            raise ImportError(f"TCNMonitor requires torch - " + "install the optional torch dependency: pip install torch --index-url https://download.pytorch.org/whl/cpu (or use requirements.lock.txt). GRU/LSTM/TCN baselines need it; the ESN and statistical monitors do not.")
        super().__init__(standardizer, cusum_k, seed, name)
        self.channels = int(channels)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.dilations = (1, 2, 4, 8)
        self.rf = 1 + sum(self.dilations)          # receptive field = 16
        self._net: "nn.Module | None" = None
        self._buf: list[np.ndarray] = []

    def _build(self, D: int) -> "nn.Module":
        layers: list[nn.Module] = []
        c_in = D
        for d in self.dilations:
            layers += [nn.ConstantPad1d((d, 0), 0.0),
                       nn.Conv1d(c_in, self.channels, kernel_size=2,
                                 dilation=d),
                       nn.ReLU()]
            c_in = self.channels
        layers.append(nn.Conv1d(c_in, D, kernel_size=1))
        return nn.Sequential(*layers).double()

    def _train(self, seqs: list[np.ndarray]) -> None:
        D = seqs[0].shape[1]
        torch.manual_seed(int(rng_for(self.seed, self.name, "torch")
                              .integers(0, 2**31 - 1)))
        self._net = self._build(D)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        tensors = [torch.from_numpy(U) for U in seqs if U.shape[0] > _WASHOUT + 1]
        for _ in range(self.epochs):
            for U in tensors:
                opt.zero_grad()
                pred = self._net(U[:-1].T[None])[0].T   # (T-1, D), causal
                loss = torch.mean((pred[_WASHOUT - 1:] - U[_WASHOUT:]) ** 2)
                loss.backward()
                opt.step()
        self._net.eval()

    def _reset_predictor(self) -> None:
        self._buf = []

    def _predict_next(self, u: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(u, dtype=float))
        window = np.asarray(self._buf[-self.rf:])
        with torch.no_grad():
            out = self._net(torch.from_numpy(window.T[None]))[0].T
            return out[-1].numpy()


if __name__ == "__main__":
    import time

    from derail.common import D_TOTAL, MASTER_SEED

    t0 = time.time()

    def make_episode(idx: int, perturb_after: int | None = None) -> Episode:
        rng = rng_for(MASTER_SEED, "seq-smoke", idx)
        T = int(rng.integers(35, 51))
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
            if perturb_after is not None and t > perturb_after:
                X[t] = X[t] + 1.2 * rng.standard_normal(D_TOTAL)
        return Episode(X=X, episode_id=f"seq-{idx:03d}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [make_episode(i) for i in range(30)]
    std = Standardizer().fit(train)
    clean, pert = make_episode(900), make_episode(901, perturb_after=20)

    monitors: list[_NextStepMonitor] = [LinearARMonitor(std, seed=0)]
    if _HAS_TORCH:
        monitors += [GRUMonitor(std, epochs=10, seed=0),
                     LSTMMonitor(std, epochs=10, seed=0),
                     TCNMonitor(std, epochs=10, seed=0)]
    for mon in monitors:
        mon.fit(train)
        s_c = mon.score_episode(clean)
        s_p = mon.score_episode(pert)
        assert s_c.shape == (clean.T,) and np.all(np.isfinite(s_c)), mon.name
        assert np.all(s_c >= 0.0) and np.all(s_c[:_WASHOUT] == 0.0), mon.name
        pre = float(np.mean(s_p[_WASHOUT:21]))
        post = float(np.mean(s_p[21:]))
        assert post > pre, (mon.name, pre, post)
        assert np.array_equal(mon.score_episode(pert), s_p), mon.name
        print(f"  {mon.name:<10} clean={np.mean(s_c[_WASHOUT:]):7.2f} "
              f"pert pre={pre:7.2f} post={post:8.2f}")
    print(f"PASS seq_baselines smoke test in {time.time() - t0:.1f}s "
          f"(torch={_HAS_TORCH})")
