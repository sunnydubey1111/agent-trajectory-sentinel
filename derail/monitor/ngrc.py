"""Next-generation reservoir computing (NG-RC / NVAR) monitor — the control
that asks whether the RANDOM RESERVOIR is doing any work at all.

Every reservoir monitor in this repository (`esn.py`, `hmt_esn.py`) spends its
capacity on a random recurrent matrix W: draw it, rescale it to a spectral
radius, run a leaky-tanh recursion, and read out linearly. NG-RC
(Gauthier et al., "Next generation reservoir computing", Nat. Commun. 12:5564,
2021) makes the observation that a linear readout over a random reservoir is,
for many tasks, an expensive approximation of a linear readout over an
EXPLICIT nonlinear feature map of a short input history. Replace the reservoir
with that feature map and there is no W, no spectral radius, no leak rate, no
ensemble and no random seed — the features are deterministic:

    o_t = [1;  x_t, x_{t-1}, ..., x_{t-k+1};  (unique quadratic products)]

and the same ridge readout predicts x_{t+1} from o_t.

This module is deliberately NOT a proposed improvement. It is a controlled
comparison: identical contract (one-class, strictly causal, OnlineMonitor
lifecycle), identical channel selection, identical held-out robust-z
normalisation, identical one-sided CUSUM, identical channel-max fusion. The
only thing that changes is what produces the features. If NG-RC matches the
ESN, the random reservoir is not earning its cost; if the ESN wins, the
recurrence is doing something a fixed delay embedding cannot.

Two structural facts make this a fair test on THIS corpus rather than a
strawman:

  * Episodes are 5-8 steps long. A delay embedding of k=2 already spans a
    third of a median episode, so NG-RC is not being starved of history
    relative to a reservoir whose effective memory is a few steps anyway.
  * There is no ensemble, so there is no disagreement stream. The fused score
    is z(surprise) alone (beta is structurally 0), which is reported rather
    than hidden — the ESN's disagreement term is a genuine extra signal that
    a deterministic model cannot provide, and the comparison should say so.

Cost: the quadratic block over a k-step embedding of a D-dim channel has
D*k*(D*k+1)/2 terms, which is why `order=2` is only sensible on the narrow
channels. `warn_quadratic_budget` caps it explicitly rather than silently
building a 100k-column design matrix.
"""

from __future__ import annotations

import numpy as np

from derail.common import (
    CHANNEL_SLICES,
    DEGENERATE_EPS,
    Episode,
    OnlineMonitor,
    Standardizer,
)
from derail.monitor.esn import (
    _MONITOR_SPLIT_SEED,
    _SIGMA_FLOOR,
    _WASHOUT,
    _robust_loc_scale,
)
from derail.common import rng_for

#: Refuse to build a quadratic block wider than this many columns.
_MAX_QUADRATIC_COLS = 4096


def _n_quadratic(n_lin: int) -> int:
    return n_lin * (n_lin + 1) // 2


class NVARFeatures:
    """Deterministic NG-RC feature map over a k-step delay embedding.

    Given a channel's standardized inputs, emits
    [1; x_t..x_{t-k+1}; upper-triangular products of that linear block].
    Steps before the embedding is full are left-padded with zeros, which is
    what the washout already discards.
    """

    def __init__(self, d_in: int, k: int = 2, order: int = 2) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        self.d_in, self.k, self.order = int(d_in), int(k), int(order)
        self.n_lin = self.d_in * self.k
        self.n_quad = _n_quadratic(self.n_lin) if self.order == 2 else 0
        if self.n_quad > _MAX_QUADRATIC_COLS:
            raise ValueError(
                f"quadratic block would be {self.n_quad} columns "
                f"(d_in={d_in}, k={k}); over the {_MAX_QUADRATIC_COLS} budget. "
                f"Use order=1 on this channel.")
        self.n_out = 1 + self.n_lin + self.n_quad
        self._iu = np.triu_indices(self.n_lin) if self.order == 2 else None

    def _from_linear(self, lin: np.ndarray) -> np.ndarray:
        """(..., n_lin) -> (..., n_out)."""
        parts = [np.ones(lin.shape[:-1] + (1,)), lin]
        if self.order == 2:
            outer = lin[..., :, None] * lin[..., None, :]
            parts.append(outer[..., self._iu[0], self._iu[1]])
        return np.concatenate(parts, axis=-1)

    def teacher_forced(self, U: np.ndarray) -> np.ndarray:
        """(T, d_in) -> (T, n_out); row t uses x_t..x_{t-k+1}."""
        T = U.shape[0]
        lin = np.zeros((T, self.n_lin))
        for j in range(self.k):
            lin[j:, j * self.d_in:(j + 1) * self.d_in] = U[:T - j] if j else U
        return self._from_linear(lin)


class _NVARChannel:
    """One channel's NG-RC next-step predictor + robust-z + CUSUM."""

    def __init__(self, d_in: int, k: int, order: int, ridge_lambda: float,
                 cusum_k: float) -> None:
        self.feat = NVARFeatures(d_in, k, order)
        self.d_in = d_in
        self.ridge_lambda = float(ridge_lambda)
        self.cusum_k = float(cusum_k)
        self.Wout: np.ndarray | None = None
        self.sigma_err: np.ndarray | None = None
        self.sup_loc, self.sup_scale = 0.0, 1.0
        self._hist: list[np.ndarray] = []
        self._prev_pred: np.ndarray | None = None
        self.cusum = 0.0

    # ------------------------------------------------------------- fitting
    def solve(self, fit_eps_inputs: list[np.ndarray]) -> None:
        F = self.feat.n_out
        ZtZ = np.zeros((F, F))
        ZtY = np.zeros((F, self.d_in))
        rows = 0
        for U in fit_eps_inputs:
            T = U.shape[0]
            if T < _WASHOUT + 1:
                continue
            Z = self.feat.teacher_forced(U)
            # Same alignment as esn.py: feature at t -> target x_{t+1}, from
            # t = washout-1 so the first SCORED transition is also trained.
            Zr, Yr = Z[max(_WASHOUT - 1, 0):T - 1], U[max(_WASHOUT, 1):T]
            rows += Zr.shape[0]
            ZtZ += Zr.T @ Zr
            ZtY += Zr.T @ Yr
        if rows == 0:
            raise ValueError("NG-RC: no usable ridge rows (episodes too short)")
        self.Wout = np.linalg.solve(ZtZ + self.ridge_lambda * np.eye(F), ZtY)

    def heldout_pairs(self, U: np.ndarray):
        T = U.shape[0]
        if T < _WASHOUT + 1:
            return None
        Z = self.feat.teacher_forced(U)
        preds = Z @ self.Wout
        return preds[max(_WASHOUT - 1, 0):T - 1], U[max(_WASHOUT, 1):T]

    def calibrate(self, pairs: list) -> None:
        P = np.concatenate([p for p, _ in pairs], axis=0)
        Y = np.concatenate([y for _, y in pairs], axis=0)
        resid = P - Y
        # Degenerate-scale contract (DESIGN.md Amendment 6): a dim predicted
        # exactly on healthy data is left unscaled, never divided by a floor.
        sd = resid.std(axis=0)
        self.sigma_err = np.where(sd < DEGENERATE_EPS, 1.0,
                                  np.maximum(sd, _SIGMA_FLOOR))
        rn = resid / self.sigma_err
        self.sup_loc, self.sup_scale = _robust_loc_scale(np.mean(rn * rn, axis=1))

    # ------------------------------------------------------------ streaming
    def reset(self) -> None:
        self._hist = []
        self._prev_pred = None
        self.cusum = 0.0

    def step(self, u: np.ndarray, t: int) -> float:
        if t >= _WASHOUT and self._prev_pred is not None:
            err = (self._prev_pred - u) / self.sigma_err
            z = (float(np.mean(err * err)) - self.sup_loc) / self.sup_scale
            self.cusum = max(0.0, self.cusum + z - self.cusum_k)
        self._hist.insert(0, np.asarray(u, dtype=float))
        del self._hist[self.feat.k:]
        lin = np.zeros(self.feat.n_lin)
        for j, past in enumerate(self._hist):
            lin[j * self.d_in:(j + 1) * self.d_in] = past
        self._prev_pred = self.feat._from_linear(lin) @ self.Wout
        return self.cusum


class NGRCMonitor(OnlineMonitor):
    """Channel-max NG-RC/NVAR monitor — the no-random-reservoir control.

    Parameters
    ----------
    channels : channel groups of x_t to monitor, fused by max (as ChannelMax).
    k : delay-embedding depth (how many past steps enter the feature map).
    order : 1 for a purely linear delay embedding (a VAR one-step predictor,
        the strictest control), 2 to add the quadratic products that give
        NG-RC its nonlinearity.
    order_by_channel : optional per-channel override of `order`, so a wide
        channel can drop to linear while narrow ones stay quadratic. A channel
        whose quadratic block exceeds the column budget falls back to order 1
        and is recorded in `self.effective_order`.
    """

    def __init__(
        self,
        standardizer: Standardizer,
        channels: tuple[str, ...] = ("e", "u", "m"),
        k: int = 2,
        order: int = 2,
        order_by_channel: dict[str, int] | None = None,
        ridge_lambda: float = 1e-2,
        cusum_k: float = 0.5,
        name: str | None = None,
    ) -> None:
        self.standardizer = standardizer
        self.channels = tuple(channels)
        self.k = int(k)
        self.name = name or f"ngrc[{','.join(channels)}]k{k}o{order}"
        self._cols: dict[str, np.ndarray] = {}
        self._chan: dict[str, _NVARChannel] = {}
        self.effective_order: dict[str, int] = {}
        for c in self.channels:
            sl = CHANNEL_SLICES[c]
            cols = np.arange(sl.start, sl.stop)
            self._cols[c] = cols
            want = (order_by_channel or {}).get(c, order)
            try:
                ch = _NVARChannel(cols.size, k, want, ridge_lambda, cusum_k)
            except ValueError:
                # Quadratic block over budget for this channel: fall back to a
                # linear embedding rather than silently building a huge design.
                ch = _NVARChannel(cols.size, k, 1, ridge_lambda, cusum_k)
                want = 1
            self._chan[c] = ch
            self.effective_order[c] = want
        self._t = 0

    @property
    def n_parameters(self) -> int:
        """Readout parameters across channels — the cost to compare with R^2."""
        return sum(ch.feat.n_out * ch.d_in for ch in self._chan.values())

    def fit(self, healthy_episodes: list[Episode]) -> None:
        if len(healthy_episodes) < 2:
            raise ValueError("NGRCMonitor.fit needs >= 2 healthy episodes")
        # SAME shared fit/held split as esn.py, so a comparison against the
        # reservoir monitors is not confounded by different calibration data.
        perm = rng_for(_MONITOR_SPLIT_SEED, "monitor", "fit-held-split"
                       ).permutation(len(healthy_episodes))
        n_fit = min(max(int(round(0.85 * len(healthy_episodes))), 1),
                    len(healthy_episodes) - 1)
        fit_eps = [healthy_episodes[i] for i in perm[:n_fit]]
        held_eps = [healthy_episodes[i] for i in perm[n_fit:]]

        for c in self.channels:
            ch = self._chan[c]
            ch.solve([self.standardizer.transform(ep.X)[:, self._cols[c]]
                      for ep in fit_eps])
            pairs = []
            for ep in held_eps:
                pr = ch.heldout_pairs(
                    self.standardizer.transform(ep.X)[:, self._cols[c]])
                if pr is not None:
                    pairs.append(pr)
            if not pairs:
                raise ValueError(f"NG-RC: no usable held-out steps for {c!r}")
            ch.calibrate(pairs)

    def start_episode(self) -> None:
        for ch in self._chan.values():
            if ch.Wout is None:
                raise RuntimeError(f"{self.name}: fit() before scoring")
            ch.reset()
        self._t = 0

    def score_step(self, x_t: np.ndarray) -> float:
        Xs = self.standardizer.transform(np.asarray(x_t, dtype=float))
        best = max(self._chan[c].step(Xs[self._cols[c]], self._t)
                   for c in self.channels)
        self._t += 1
        return best


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import time

    from derail.common import D_TOTAL, MASTER_SEED

    t0 = time.time()

    def make_episode(idx: int, perturb_after: int | None = None) -> Episode:
        rng = rng_for(MASTER_SEED, "ngrc-smoke", idx)
        T = int(rng.integers(35, 51))
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
            if perturb_after is not None and t > perturb_after:
                X[t] = X[t] + 1.2 * rng.standard_normal(D_TOTAL)
        return Episode(X=X, episode_id=f"ngrc-smoke-{idx:03d}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [make_episode(i) for i in range(30)]
    std = Standardizer().fit(train)
    clean, pert = make_episode(900), make_episode(901, perturb_after=20)

    # The e channel is 32-dim: its quadratic block (k=2 -> 64 linear -> 2080
    # quadratic) is inside budget, but the linear fallback is exercised too.
    for label, kw in (("linear  k=2", dict(order=1, k=2)),
                      ("quad    k=1", dict(order=2, k=1)),
                      ("quad    k=2", dict(order=2, k=2))):
        mon = NGRCMonitor(std, ("e", "u", "m"), **kw)
        mon.fit(train)
        s_clean = mon.score_episode(clean)
        assert s_clean.shape == (clean.T,) and np.all(np.isfinite(s_clean))
        assert np.all(s_clean[:_WASHOUT] == 0.0), "washout must score 0"
        assert np.all(s_clean >= 0.0), "CUSUM must be >= 0"
        assert np.array_equal(mon.score_episode(clean), s_clean), \
            "rescoring not deterministic"
        s_p = mon.score_episode(pert)
        pre = float(np.mean(s_p[_WASHOUT:21]))
        post = float(np.mean(s_p[21:]))
        assert post > pre, f"{label}: perturbation not elevated"
        print(f"  {label}  ok  params={mon.n_parameters:6d}  "
              f"orders={mon.effective_order}  "
              f"clean_max={s_clean.max():8.2f}  post={post:10.2f}")

    # Deterministic: no seed argument exists, so two instances must agree.
    a, b = (NGRCMonitor(std, ("e", "u", "m")) for _ in range(2))
    a.fit(train); b.fit(train)
    assert np.array_equal(a.score_episode(pert), b.score_episode(pert)), \
        "NG-RC must be seed-free and reproducible"

    # Budget guard fires rather than building a huge design matrix.
    try:
        NVARFeatures(d_in=64, k=2, order=2)
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("quadratic column budget not enforced")

    print(f"PASS ngrc.py smoke test in {time.time() - t0:.1f}s "
          f"(causal, deterministic, seed-free, budget-guarded)")
