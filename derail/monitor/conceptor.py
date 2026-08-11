"""Conceptor monitor — the one candidate whose failure mode is different.

Every other monitor in this repository scores PREDICTION ERROR: the reservoir
predicts the next input, and surprise is how badly it missed. That is exactly
the wrong instrument for slow goal drift. A gradual rotation of the agent's
trajectory changes WHICH SUBSPACE the reservoir's dynamics occupy while barely
changing how predictable the next step is, so a one-step-ahead error stays
flat while the geometry moves underneath it. Measured detection for slow
goal_drift is 0.0125 for every monitor in the study — the failure is not that
the reservoir is too small, it is that nobody is looking at the right quantity.

A conceptor (Jaeger, "Controlling Recurrent Neural Networks by Conceptors",
arXiv:1403.3369) is a soft projection onto the subspace a reservoir occupies
while driven by a given signal class. From healthy reservoir states h_t with
correlation R = E[h h^T],

    C = R (R + a^-2 I)^-1

is a positive-semidefinite matrix with eigenvalues in [0, 1): directions the
healthy dynamics visit strongly get eigenvalue near 1, directions never
visited get near 0. `a` is the aperture.

The detection quantity is then how far the CURRENT state falls outside that
healthy ellipsoid:

    outside_t = 1 - ||C h_t||^2 / ||h_t||^2

which is ~0 while the trajectory stays in the healthy subspace and grows as it
rotates out, regardless of whether the next step remains predictable. That is
the mechanism claim, and it is the whole reason this arm exists.

Contract identical to every other monitor here, so a comparison measures the
MECHANISM and not the calibration: one-class (fit sees healthy only), strictly
causal, the shared `_MONITOR_SPLIT_SEED` fit/held split, robust-z on held-out
healthy with the `DEGENERATE_EPS` guard, one-sided CUSUM, channel-max fusion.

This is exploratory. It is reported as an arm, not as the primary monitor, and
on a corpus whose drift episodes must have real post-onset runway or the
result means nothing (see `experiments/collect_goal_drift_long.py`).
"""

from __future__ import annotations

import inspect

import numpy as np

from derail.common import (
    CHANNEL_SLICES,
    DEGENERATE_EPS,
    Episode,
    OnlineMonitor,
    Standardizer,
    rng_for,
)
from derail.monitor.esn import (
    _MONITOR_SPLIT_SEED,
    _WASHOUT,
    ESNEnsembleMonitor,
    _robust_loc_scale,
)

#: Reservoir and CUSUM settings are INHERITED from the ESN rather than
#: retyped, so the two arms cannot silently drift apart. A conceptor-vs-ESN
#: comparison is only a comparison of MECHANISMS (state geometry vs
#: prediction error) if everything upstream of the mechanism is identical;
#: re-declaring these numbers here is exactly how that guarantee gets lost.
_ESN_DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(ESNEnsembleMonitor.__init__).parameters.items()
    if p.default is not inspect.Parameter.empty
}


class _DeadChannel(Exception):
    """A channel whose reservoir states never leave the origin.

    Raised during fit and handled by disabling that channel, never propagated
    to the caller. A channel that carries no signal must contribute NOTHING to
    a max-fusion — it must not crash the monitor, and it must not be scaled up
    into the most sensitive detector in the system. Same contract as
    DESIGN.md Amendment 6, applied one level up: at the channel rather than
    the dimension.
    """


def _resolve_aperture_grid(eigenvalues: np.ndarray, n_points: int) -> np.ndarray:
    """Aperture candidates derived from the state spectrum, not assumed.

    A conceptor's aperture `a` enters only as `a^-2` against the eigenvalues
    of the state correlation matrix: direction i is kept in proportion to
    `lam_i / (lam_i + a^-2)`. So the scales that can possibly matter are set
    by the spectrum itself — `a^-2` far above `lam_max` projects everything
    out, far below `lam_min` keeps everything. The informative range is
    therefore `a in [lam_max^-1/2, lam_min^-1/2]`, which is computed here
    from the healthy data instead of being guessed. A fixed numeric grid
    would be wrong the moment the reservoir size, leak rate or channel width
    changed the scale of the states.
    """
    lam = np.sort(eigenvalues[eigenvalues > DEGENERATE_EPS])[::-1]
    if lam.size == 0:
        raise _DeadChannel(
            "state correlation has no eigenvalue above DEGENERATE_EPS — the "
            "reservoir never moved for this channel")

    # The grid must span the SIGNAL spectrum, not the numerical floor. A
    # reservoir driven by agent telemetry occupies far fewer directions than
    # it has units: measured effective rank here is ~1.4-14 out of 128, so the
    # smallest eigenvalues are round-off (~1e-12) and the condition number
    # reaches 1e11. Anchoring the low end of the grid at lam_min^-1/2 pushes
    # the search out to a ~1e7 aperture, where a^-2 is far below every real
    # eigenvalue, C collapses to the identity, and `1 - ||Ch||^2/||h||^2`
    # becomes ~0 for healthy AND drifting states alike — a blind detector.
    # Measured cost of anchoring at lam_min instead: ~0.22 episode AUC, with a
    # 0.21 spread across splits.
    #
    # The participation ratio (sum lam)^2 / sum lam^2 is the data's own answer
    # to "how many directions does this actually occupy". Truncating the grid
    # at that eigenvalue keeps the search inside the subspace the healthy
    # dynamics genuinely span. No target value is assumed; PR is measured.
    pr = float(lam.sum() ** 2 / np.sum(lam ** 2))
    k = int(np.clip(round(pr), 1, lam.size)) - 1
    lam_signal = float(lam[k])

    lo = float(lam[0]) ** -0.5
    hi = lam_signal ** -0.5
    if not np.isfinite(hi) or hi <= lo:
        hi = lo * np.e
    # One decade of padding either side so the optimum is interior to the
    # search rather than clipped to an endpoint.
    return np.geomspace(lo / 10.0, hi * 10.0, n_points)


def select_aperture(Rmat: np.ndarray, n_points: int) -> float:
    """Pick the aperture WITHOUT labels, by Jaeger's adaptation criterion.

    The aperture is the one real knob a conceptor has, and fixing it by hand
    is how an arm like this ends up quietly tuned to its test set. Jaeger
    (arXiv:1403.3369, §3.8) gives a label-free criterion: the useful aperture
    is the one at which the conceptor is most SENSITIVE to its own aperture,

        argmax_a  d ||C(a)||_F^2 / d log(a)

    i.e. the scale at which C is actively discriminating directions rather
    than saturating toward 0 (everything projected out) or toward I (nothing
    projected out). Both the criterion and the candidate range come from the
    HEALTHY fit-split correlation matrix alone — no injected episode, no
    held-out episode and no label influences the choice.
    """
    lam = np.linalg.eigvalsh(Rmat)
    grid = _resolve_aperture_grid(lam, n_points)
    # ||C(a)||_F^2 = sum_i (lam_i / (lam_i + a^-2))^2, evaluated spectrally:
    # no need to form C for each candidate.
    inv_sq = grid[:, None] ** -2.0
    norms = np.sum((lam[None, :] / (lam[None, :] + inv_sq)) ** 2, axis=1)
    grad = np.gradient(norms, np.log(grid))
    return float(grid[int(np.argmax(grad))])


class _ConceptorChannel:
    """One channel's reservoir + healthy conceptor + CUSUM."""

    def __init__(self, d_in: int, R: int, leak: float, spectral_radius: float,
                 input_scale: float, density: float, aperture: float | None,
                 cusum_k: float, seed: int, n_aperture_candidates: int,
                 tag: str) -> None:
        self.d_in, self.R = d_in, R
        self.leak = float(leak)
        #: None = select from healthy data by the criterion above.
        self.aperture = None if aperture is None else float(aperture)
        self.cusum_k = float(cusum_k)
        self.n_aperture_candidates = int(n_aperture_candidates)
        rng = rng_for(seed, "conceptor", tag)
        W = rng.standard_normal((R, R)) * (rng.random((R, R)) < density)
        sr = float(np.max(np.abs(np.linalg.eigvals(W))))
        if sr > 0.0:
            W *= spectral_radius / sr
        self.W = W
        self.Win = np.where(rng.random((R, d_in)) < 0.5, -1.0, 1.0) * input_scale
        self.C: np.ndarray | None = None
        #: True when the reservoir never leaves the origin for this channel;
        #: the channel then contributes 0.0 forever instead of crashing.
        self.dead = False
        self.loc, self.scale = 0.0, 1.0
        self.h = np.zeros(R)
        self.cusum = 0.0

    # ------------------------------------------------------------- states
    def states(self, U: np.ndarray) -> np.ndarray:
        """Teacher-forced leaky-tanh states, (T, R)."""
        T = U.shape[0]
        out = np.empty((T, self.R))
        h = np.zeros(self.R)
        lk = self.leak
        proj = U @ self.Win.T
        for t in range(T):
            h = (1.0 - lk) * h + lk * np.tanh(self.W @ h + proj[t])
            out[t] = h
        return out

    def outside(self, H: np.ndarray) -> np.ndarray:
        """1 - ||C h||^2 / ||h||^2 per row; 0 inside the healthy subspace."""
        num = np.einsum("ij,ij->i", H @ self.C.T, H @ self.C.T)
        den = np.einsum("ij,ij->i", H, H)
        return 1.0 - num / np.maximum(den, DEGENERATE_EPS)

    # -------------------------------------------------------------- fit
    def fit_conceptor(self, seqs: list[np.ndarray]) -> None:
        Rmat = np.zeros((self.R, self.R))
        n = 0
        for U in seqs:
            if U.shape[0] < _WASHOUT + 1:
                continue
            H = self.states(U)[_WASHOUT:]
            Rmat += H.T @ H
            n += H.shape[0]
        if n == 0:
            raise ValueError("conceptor: no usable healthy states")
        Rmat /= n
        try:
            if self.aperture is None:
                self.aperture = select_aperture(
                    Rmat, self.n_aperture_candidates)
        except _DeadChannel:
            self.dead = True
            self.C = None
            return
        self.C = Rmat @ np.linalg.inv(
            Rmat + (self.aperture ** -2) * np.eye(self.R))

    def calibrate(self, seqs: list[np.ndarray]) -> None:
        if self.dead:
            return
        vals = []
        for U in seqs:
            if U.shape[0] < _WASHOUT + 1:
                continue
            vals.append(self.outside(self.states(U)[_WASHOUT:]))
        if not vals:
            raise ValueError("conceptor: no usable held-out states")
        v = np.concatenate(vals)
        loc, scale = _robust_loc_scale(v)
        # Amendment 6: a stream with no healthy variation is left unscaled.
        self.loc = loc
        self.scale = 1.0 if scale < DEGENERATE_EPS else scale

    # --------------------------------------------------------- streaming
    def reset(self) -> None:
        self.h = np.zeros(self.R)
        self.cusum = 0.0

    def step(self, u: np.ndarray, t: int) -> float:
        if self.dead:
            return 0.0
        lk = self.leak
        self.h = (1.0 - lk) * self.h + lk * np.tanh(self.W @ self.h
                                                    + self.Win @ u)
        if t < _WASHOUT:
            return 0.0
        num = float(self.C @ self.h @ (self.C @ self.h))
        den = float(self.h @ self.h)
        z = ((1.0 - num / max(den, DEGENERATE_EPS)) - self.loc) / self.scale
        self.cusum = max(0.0, self.cusum + z - self.cusum_k)
        return self.cusum


class ConceptorMonitor(OnlineMonitor):
    """Channel-max conceptor monitor: scores state GEOMETRY, not prediction."""

    def __init__(
        self,
        standardizer: Standardizer,
        channels: tuple[str, ...] = ("e", "u", "m"),
        reservoir_size: int | None = None,
        leak_rate: float | None = None,
        spectral_radius: float | None = None,
        input_scale: float | None = None,
        density: float | None = None,
        aperture: float | None = None,
        cusum_k: float | None = None,
        n_aperture_candidates: int = 64,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        # None means "whatever the ESN uses", read from its signature.
        d = _ESN_DEFAULTS
        reservoir_size = d["reservoir_size"] if reservoir_size is None else reservoir_size
        leak_rate = d["leak_rate"] if leak_rate is None else leak_rate
        spectral_radius = d["spectral_radius"] if spectral_radius is None else spectral_radius
        input_scale = d["input_scale"] if input_scale is None else input_scale
        density = d["density"] if density is None else density
        cusum_k = d["cusum_k"] if cusum_k is None else cusum_k
        self.n_aperture_candidates = int(n_aperture_candidates)
        self.standardizer = standardizer
        self.channels = tuple(channels)
        self.name = name or (
            f"conceptor[{','.join(channels)}]"
            + ("auto" if aperture is None else f"a{aperture:g}"))
        self._cols: dict[str, np.ndarray] = {}
        self._chan: dict[str, _ConceptorChannel] = {}
        for i, c in enumerate(self.channels):
            sl = CHANNEL_SLICES[c]
            cols = np.arange(sl.start, sl.stop)
            self._cols[c] = cols
            self._chan[c] = _ConceptorChannel(
                cols.size, reservoir_size, leak_rate, spectral_radius,
                input_scale, density, aperture, cusum_k, seed,
                n_aperture_candidates, f"{c}-{i}")
        self._t = 0

    def fit(self, healthy_episodes: list[Episode]) -> None:
        if len(healthy_episodes) < 2:
            raise ValueError("ConceptorMonitor.fit needs >= 2 healthy episodes")
        # The SHARED split, so an A/B against the ESN measures the mechanism.
        perm = rng_for(_MONITOR_SPLIT_SEED, "monitor", "fit-held-split"
                       ).permutation(len(healthy_episodes))
        n_fit = min(max(int(round(0.85 * len(healthy_episodes))), 1),
                    len(healthy_episodes) - 1)
        fit_eps = [healthy_episodes[i] for i in perm[:n_fit]]
        held_eps = [healthy_episodes[i] for i in perm[n_fit:]]
        for c in self.channels:
            ch = self._chan[c]
            ch.fit_conceptor([self.standardizer.transform(e.X)[:, self._cols[c]]
                              for e in fit_eps])
            ch.calibrate([self.standardizer.transform(e.X)[:, self._cols[c]]
                          for e in held_eps])
        self.dead_channels = tuple(c for c in self.channels
                                   if self._chan[c].dead)
        if len(self.dead_channels) == len(self.channels):
            raise ValueError(
                f"{self.name}: every channel is degenerate — nothing to score")

    def start_episode(self) -> None:
        for ch in self._chan.values():
            if ch.C is None and not ch.dead:
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

    def make_episode(idx: int, drift_after: int | None = None) -> Episode:
        """AR(1) telemetry; optional slow ROTATION after a step.

        The rotation is the point: it barely changes one-step predictability
        but moves the trajectory into a different subspace.
        """
        rng = rng_for(MASTER_SEED, "conceptor-smoke", idx)
        T = 40
        X = np.empty((T, D_TOTAL))
        x = rng.standard_normal(D_TOTAL)
        direction = rng.standard_normal(D_TOTAL)
        direction /= np.linalg.norm(direction)
        for t in range(T):
            x = 0.9 * x + 0.3 * rng.standard_normal(D_TOTAL)
            X[t] = x
            if drift_after is not None and t > drift_after:
                X[t] = X[t] + 0.25 * (t - drift_after) * direction
        return Episode(X=X, episode_id=f"con-{idx:03d}", is_healthy=True,
                       failure_class=None, tau=None, t_fail=None, severity=None)

    train = [make_episode(i) for i in range(30)]
    std = Standardizer().fit(train)
    mon = ConceptorMonitor(std, ("e", "u", "m"), seed=0)
    mon.fit(train)

    clean = make_episode(900)
    s_clean = mon.score_episode(clean)
    assert s_clean.shape == (clean.T,) and np.all(np.isfinite(s_clean))
    assert np.all(s_clean[:_WASHOUT] == 0.0), "washout must score 0"
    assert np.all(s_clean >= 0.0), "CUSUM must be >= 0"
    assert np.array_equal(mon.score_episode(clean), s_clean), \
        "rescoring not deterministic"

    drift = make_episode(901, drift_after=15)
    s_dr = mon.score_episode(drift)
    pre = float(np.mean(s_dr[_WASHOUT:16]))
    post = float(np.mean(s_dr[16:]))
    assert post > pre, f"slow drift not elevated ({post:.3f} <= {pre:.3f})"

    print(f"  clean_max={s_clean.max():9.3f}  drift_pre={pre:9.3f}  "
          f"drift_post={post:9.3f}")
    print(f"PASS conceptor.py smoke test in {time.time() - t0:.1f}s "
          f"(causal, deterministic, geometry-sensitive)")
