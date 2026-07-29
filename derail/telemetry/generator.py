"""Telemetry generator: healthy-episode simulator + failure injector (Module 1).

Simulates episodes of an LLM agent as multi-channel step telemetry
x_t = [e_t; u_t; m_t] (layout in ``derail.common``) and injects one of five
derailment classes at a ground-truth onset step tau:

  goal_drift         -- gradual semantic rotation toward a distractor goal;
                        e_t leads, entropy normal/slightly reduced, m_t normal.
  looping            -- cycle over the last L semantic states + a repeated
                        short action cycle with near-identical latencies;
                        progress stalls; e_t and m_t lead.
  tool_cascade       -- error storms: retry pairs, inflated latencies,
                        semantic ping-pong attempt <-> error-processing; m_t leads.
  grounding_loss     -- the subtle one: semantics stay plausible, the
                        uncertainty channel shifts (entropy up, slope flips
                        positive, high-entropy fraction up); u_t leads,
                        e_t nearly blind.
  context_corruption -- predictability collapses: AR coherence drops,
                        innovations inflate, actions go near-uniform, entropy
                        VARIANCE inflates; step-to-step dynamics break.

All failures ramp in smoothly over ``ramp_steps`` and scale with severity, so
low-severity onsets are genuinely hard. Healthy episodes contain benign
irregularities (waypoint revisits, small corrections, entropy spikes, rare
transient tool errors with immediate recovery) so monitors cannot alarm on
any deviation.

All randomness flows through ``common.rng_for`` -- same seed => same dataset.
Run the smoke test with:  py -m derail.telemetry.generator
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from derail.common import (
    ACTION_TYPES,
    CHANNEL_SLICES,
    D_SEM,
    D_TOTAL,
    FAILURE_CLASSES,
    IDX_ACTION_ONEHOT,
    IDX_ENTROPY_SLOPE,
    IDX_ERROR_FLAG,
    IDX_HIGH_ENTROPY_FRAC,
    IDX_LATENCY_LOG,
    IDX_MAX_ENTROPY,
    IDX_MEAN_ENTROPY,
    IDX_OUTLEN_LOG,
    DatasetConfig,
    Episode,
    FailureSpec,
    SimConfig,
    Standardizer,
    rng_for,
)

__all__ = ["EpisodeGenerator", "sample_failure_spec", "make_dataset"]

# ---------------------------------------------------------------------------
# Internal constants of the healthy dynamics (interpretation of SimConfig)
# ---------------------------------------------------------------------------
_STEP_RATE = 0.35        # per-step pull of the semantic state toward its attractor
_REVISIT_P = 0.06        # per-step chance of a benign revisit of the previous waypoint
_CORRECTION_P = 0.05     # per-step chance of a small benign course correction
_BENIGN_SPIKE_P = 0.05   # per-step chance of a benign decision-point entropy spike

# Sticky-ish healthy action Markov chain over ACTION_TYPES
# (plan, tool_call, tool_result, synthesis); rows sum to 1.
_A_PLAN, _A_CALL, _A_RESULT, _A_SYNTH = 0, 1, 2, 3
_TRANS = np.array(
    [
        [0.08, 0.78, 0.02, 0.12],   # plan ->
        [0.04, 0.06, 0.88, 0.02],   # tool_call ->
        [0.18, 0.42, 0.06, 0.34],   # tool_result ->
        [0.30, 0.44, 0.06, 0.20],   # synthesis ->
    ]
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _ramp(j: int, ramp_steps: int) -> float:
    """Smooth onset weight in [0, 1] at the j-th post-tau step (j >= 0)."""
    lin = min(1.0, (j + 1) / max(1, ramp_steps))
    return lin * lin * (3.0 - 2.0 * lin)  # smoothstep: no step discontinuity


def _sample_action(rng: np.random.Generator, prev: int, run_len: int) -> int:
    """Markov step with a guard against 3+ identical consecutive actions."""
    a = int(rng.choice(4, p=_TRANS[prev]))
    if a == prev and run_len >= 2:
        row = _TRANS[prev].copy()
        row[prev] = 0.0
        row /= row.sum()
        a = int(rng.choice(4, p=row))
    return a


# ---------------------------------------------------------------------------
# Failure-spec sampling
# ---------------------------------------------------------------------------
def sample_failure_spec(
    failure_class: str, T: int, rng: np.random.Generator, cfg: SimConfig
) -> FailureSpec:
    """Sample onset/severity/ramp for one injected episode.

    tau ~ round(Uniform[tau_frac_min, tau_frac_max] * T_planned); severity ~
    Uniform[severity_min, severity_max]; ramp_steps longer for lower severity
    (subtle onsets ramp in slowly).
    """
    assert failure_class in FAILURE_CLASSES, failure_class
    tau = int(round(rng.uniform(cfg.tau_frac_min, cfg.tau_frac_max) * T))
    tau = max(1, min(T - 1, tau))
    severity = float(rng.uniform(cfg.severity_min, cfg.severity_max))
    sev_frac = (severity - cfg.severity_min) / max(
        1e-9, cfg.severity_max - cfg.severity_min
    )
    base = cfg.ramp_steps_max - sev_frac * (cfg.ramp_steps_max - cfg.ramp_steps_min)
    ramp_steps = int(
        np.clip(round(base + rng.normal(0.0, 0.7)), cfg.ramp_steps_min, cfg.ramp_steps_max)
    )
    return FailureSpec(
        failure_class=failure_class, tau=tau, severity=severity, ramp_steps=ramp_steps
    )


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------
class EpisodeGenerator:
    """Healthy-episode simulator + failure injector (shared dynamics)."""

    def __init__(self, cfg: SimConfig, seed: int) -> None:
        # The generator's channel layout is fixed to the study's semantic width
        # (D_SEM = 32); other d_sem values only fail later with a cryptic shape
        # error deep in generate(). Validate up front. Configurable
        # dimensions would need the whole channel layout parameterised, which is
        # out of scope; this rejects the unsupported value instead of pretending
        # to support it.
        if int(cfg.d_sem) != D_SEM:
            raise ValueError(
                f"SimConfig.d_sem must be {D_SEM} (the study's fixed semantic "
                f"width); got {cfg.d_sem}. Configurable dimensions are not "
                f"implemented.")
        self.cfg = cfg
        self.seed = int(seed)

    # -- per-episode RNG stream (deterministic in (seed, episode_id)) -------
    def _episode_rng(self, episode_id: str) -> np.random.Generator:
        return rng_for(self.seed, "episode", episode_id)

    def planned_length(self, episode_id: str) -> int:
        """Planned healthy length T_planned for this episode id (first draw of
        the episode stream, so it matches what generate() will use)."""
        rng = self._episode_rng(episode_id)
        return int(rng.integers(self.cfg.T_min, self.cfg.T_max + 1))

    # -----------------------------------------------------------------------
    def generate(self, episode_id: str, failure: FailureSpec | None) -> Episode:
        cfg = self.cfg
        d = cfg.d_sem
        rng = self._episode_rng(episode_id)
        T_planned = int(rng.integers(cfg.T_min, cfg.T_max + 1))

        if failure is None:
            T = T_planned
            tau = -1
            fc = None
            sev = 0.0
            ramp_steps = 1
        else:
            tau = int(failure.tau)
            fc = failure.failure_class
            sev = float(failure.severity)
            ramp_steps = max(1, int(failure.ramp_steps))
            # Failure horizon: smaller for higher severity (severe fails fast).
            sev_frac = (sev - cfg.severity_min) / max(
                1e-9, cfg.severity_max - cfg.severity_min
            )
            base_h = cfg.fail_horizon_max - sev_frac * (
                cfg.fail_horizon_max - cfg.fail_horizon_min
            )
            horizon = int(
                np.clip(
                    round(base_h + rng.normal(0.0, 1.5)),
                    cfg.fail_horizon_min,
                    cfg.fail_horizon_max,
                )
            )
            T = tau + horizon + 1  # t_fail = tau + horizon = T - 1

        # ---- healthy latents: goal + subtask waypoints + dwell schedule ----
        g = _unit(rng.normal(size=d))
        k = int(rng.integers(cfg.n_waypoints_min, cfg.n_waypoints_max + 1))
        W = np.stack([_unit(g + cfg.waypoint_sigma * rng.normal(size=d)) for _ in range(k)])
        seg = rng.dirichlet(np.full(k, 3.0)) * T_planned
        bounds = np.cumsum(seg)[:-1]  # waypoint switch times within the plan

        X = np.zeros((T, D_TOTAL))
        actions: list[int] = []

        s = _unit(W[0] + 0.25 * rng.normal(size=d))  # semantic latent state
        ar = np.zeros(d)                              # AR(1) semantic noise
        prev_a, run_len = -1, 0
        pending_retry = False
        last_error = False
        revisit_left = 0

        # ---- failure state (populated at onset) ----
        g2 = np.zeros(d)          # goal_drift distractor
        drift_amt = 0.0
        loop_e: list[np.ndarray] = []
        loop_a: list[int] = []
        loop_lat: list[float] = []
        loop_out: list[float] = []
        L = 0
        p_frozen, idx_frozen = 0.0, 0
        e_att = np.zeros(d)       # tool_cascade "attempt" state
        e_err = np.zeros(d)       # tool_cascade "error-processing" state
        c_err = 0.0               # tool_cascade terminal error prob coefficient
        c_ent = 0.0               # grounding_loss entropy shift coefficient

        for t in range(T):
            fail = failure is not None and t >= tau
            j = t - tau if fail else -1
            r = _ramp(j, ramp_steps) if fail else 0.0
            eff = sev * r  # severity-scaled onset weight
            w_loop = r * (0.55 + 0.45 * sev) if (fail and fc == "looping") else 0.0

            # ---------------- onset setup (once, at t == tau) ----------------
            if fail and j == 0:
                if fc == "goal_drift":
                    # distractor goal at a moderate angle (~63..103 deg) from g
                    v = rng.normal(size=d)
                    v = _unit(v - (v @ g) * g)
                    ang = rng.uniform(1.1, 1.8)
                    g2 = math.cos(ang) * g + math.sin(ang) * v
                elif fc == "looping":
                    L = int(rng.integers(2, 5))
                    lo = max(0, tau - L)
                    loop_e = [X[i, :D_SEM].copy() for i in range(lo, tau)]
                    loop_a = [actions[i] for i in range(lo, tau)]
                    loop_lat = [float(X[i, IDX_LATENCY_LOG]) for i in range(lo, tau)]
                    loop_out = [float(X[i, IDX_OUTLEN_LOG]) for i in range(lo, tau)]
                    L = len(loop_e)
                    if len(set(loop_a)) == 1:  # degenerate capture: force a retry pair
                        loop_a = [_A_CALL if i % 2 == 0 else _A_RESULT for i in range(L)]
                elif fc == "tool_cascade":
                    c_err = float(rng.uniform(0.6, 0.9))
                    e_att = X[tau - 1, :D_SEM].copy()
                    v = rng.normal(size=d)
                    v = _unit(v - (v @ e_att) * e_att)
                    ang = 0.35 + 0.40 * sev
                    e_err = math.cos(ang) * e_att + math.sin(ang) * v
                elif fc == "grounding_loss":
                    c_ent = float(rng.uniform(0.4, 1.0))
                if fc in ("looping", "tool_cascade"):
                    # progress stalls: freeze plan position at the onset
                    p_frozen = min(1.0, tau / max(1, T_planned - 1))
                    idx_frozen = min(int(np.searchsorted(bounds, tau + 0.5)), k - 1)

            # ---------------- progress + waypoint (with benign revisits) -----
            if fail and fc in ("looping", "tool_cascade"):
                p = p_frozen
                idx = idx_frozen
            else:
                p = min(1.0, t / max(1, T_planned - 1))
                idx = min(int(np.searchsorted(bounds, t + 0.5)), k - 1)
                if revisit_left > 0:
                    idx = max(0, idx - 1)
                    revisit_left -= 1
                elif idx > 0 and rng.random() < _REVISIT_P:
                    revisit_left = int(rng.integers(1, 3))
                    idx -= 1

            # ---------------- semantic channel e_t ----------------------------
            lam = min(0.95, cfg.pull_goal + (1.0 - cfg.pull_goal) * p**1.5)
            if fail and fc == "grounding_loss":
                lam *= 1.0 - 0.3 * eff  # pull toward goal reduced only mildly
            attract = _unit((1.0 - lam) * W[idx] + lam * g)
            if fail and fc == "goal_drift":
                # effective goal rotates toward the distractor at a rate ~ severity
                drift_amt = min(1.0, drift_amt + 0.30 * sev * r)
                attract = _unit((1.0 - drift_amt) * attract + drift_amt * g2)

            rho, sig, step_rate = cfg.ar_rho, cfg.ar_sigma, _STEP_RATE
            if fail and fc == "context_corruption":
                rho = rho - (rho - 0.2) * eff                 # coherence collapses
                sig = sig * (1.0 + (1.0 + 2.0 * sev) * r)     # innovations x2..x4
                step_rate *= 1.0 - 0.6 * eff                  # attractor pull weakens

            ar = rho * ar + sig * rng.normal(size=d)
            s = s + step_rate * (attract - s) + ar
            if fail and fc == "context_corruption":
                s = s + 0.03 * sev * r * rng.normal(size=d)   # slow walk off-manifold
            if not fail and rng.random() < _CORRECTION_P:
                s = s + 0.05 * rng.normal(size=d)             # benign small correction
            e_vec = _unit(s)

            if fail and fc == "looping":
                tgt = loop_e[j % L] + (0.03 + 0.05 * (1.0 - sev)) * rng.normal(size=d)
                e_vec = _unit((1.0 - w_loop) * e_vec + w_loop * _unit(tgt))
            elif fail and fc == "tool_cascade":
                w_sem = r * (0.45 + 0.45 * sev)
                tgt = (e_att if j % 2 == 0 else e_err) + 0.05 * rng.normal(size=d)
                e_vec = _unit((1.0 - w_sem) * e_vec + w_sem * _unit(tgt))

            # ---------------- action selection --------------------------------
            force: Optional[int] = None
            if fail:
                if fc == "looping" and rng.random() < r * (0.6 + 0.4 * sev):
                    force = loop_a[j % L]
                elif fc == "tool_cascade" and rng.random() < r * (0.6 + 0.4 * sev):
                    force = _A_CALL if j % 2 == 0 else _A_RESULT  # retry pair
                elif fc == "context_corruption" and rng.random() < r * (0.4 + 0.6 * sev):
                    force = int(rng.integers(0, 4))               # near-uniform
            if force is None and pending_retry:
                force = _A_CALL  # healthy retry after a transient error
            pending_retry = False

            if t == 0:
                a = _A_PLAN
            elif force is not None:
                a = force
            else:
                a = _sample_action(rng, prev_a, run_len)
            run_len = run_len + 1 if a == prev_a else 1
            prev_a = a
            actions.append(a)

            # ---------------- uncertainty channel u_t -------------------------
            noise_scale = cfg.entropy_noise
            if fail and fc == "looping":
                noise_scale *= 1.0 - 0.5 * w_loop             # confident repetition
            elif fail and fc == "context_corruption":
                noise_scale *= 1.0 + (1.5 + 2.5 * sev) * r    # variance inflates

            mean_ent = (
                cfg.entropy_base[ACTION_TYPES[a]]
                - cfg.entropy_completion_drop * p
                + rng.normal(0.0, noise_scale)
            )
            spike = rng.random() < _BENIGN_SPIKE_P            # benign decision point
            if spike:
                mean_ent += rng.uniform(0.3, 0.9)
            if fail:
                if fc == "goal_drift":
                    mean_ent -= 0.05 * eff                    # proceeds confidently
                elif fc == "looping":
                    mean_ent -= 0.20 * eff
                elif fc == "tool_cascade":
                    mean_ent += 0.35 * eff                    # moderate rise w/ retries
                elif fc == "grounding_loss":
                    mean_ent += c_ent * sev * r               # +(0.4..1.0)*severity
            mean_ent = max(0.05, mean_ent)

            max_ent = mean_ent + abs(rng.normal(0.0, 0.25))
            if spike:
                max_ent += rng.uniform(0.2, 0.6)
            if fail and fc == "grounding_loss" and rng.random() < 0.20 * r:
                max_ent += rng.uniform(0.6, 1.6) * (0.5 + 0.5 * sev)  # sporadic spikes

            slope_mu, slope_sd = -0.05, 0.10
            if fail:
                if fc == "grounding_loss":
                    slope_mu += (0.10 + 0.15 * sev) * r       # flattens -> positive
                elif fc == "looping":
                    slope_mu *= 1.0 - w_loop                  # no entropy decline
                    slope_sd *= 1.0 - 0.5 * w_loop
                elif fc == "context_corruption":
                    slope_sd *= 1.0 + 2.0 * eff
            slope = rng.normal(slope_mu, slope_sd)

            high_frac = 0.30 * mean_ent - 0.10 + rng.normal(0.0, 0.05)
            if fail and fc == "grounding_loss":
                high_frac += 0.15 * eff
            high_frac = float(np.clip(high_frac, 0.0, 1.0))

            # ---------------- metadata channel m_t ----------------------------
            mu_l, sd_l = cfg.latency_lognorm[ACTION_TYPES[a]]
            lat = rng.normal(mu_l, sd_l)
            mu_o, sd_o = cfg.outlen_lognorm[ACTION_TYPES[a]]
            if fail and fc == "context_corruption":
                sd_o *= 1.0 + (1.5 + 2.5 * sev) * r           # erratic output length
            out = rng.normal(mu_o, sd_o)

            err = 0.0
            if fail and fc == "tool_cascade":
                p_err = c_err * sev * r * (1.0 if a in (_A_CALL, _A_RESULT) else 0.3)
                if rng.random() < p_err:
                    err = 1.0                                  # consecutive errors allowed
                lat += math.log(1.0 + (1.0 + 4.0 * sev) * r)   # latency x2..x6 at full ramp
            elif a == _A_CALL and not last_error and rng.random() < cfg.healthy_error_rate:
                err = 1.0                                      # rare transient error
                pending_retry = True                           # retry the call once
                lat += 0.4                                     # slight timeout bump
            last_error = err == 1.0

            if fail and fc == "looping":                       # identical-ish stats
                lat = (1.0 - w_loop) * lat + w_loop * (loop_lat[j % L] + rng.normal(0.0, 0.05))
                out = (1.0 - w_loop) * out + w_loop * (loop_out[j % L] + rng.normal(0.0, 0.05))

            # ---------------- write the step ----------------------------------
            X[t, :D_SEM] = e_vec
            X[t, IDX_MEAN_ENTROPY] = mean_ent
            X[t, IDX_MAX_ENTROPY] = max_ent
            X[t, IDX_ENTROPY_SLOPE] = slope
            X[t, IDX_HIGH_ENTROPY_FRAC] = high_frac
            X[t, IDX_ACTION_ONEHOT.start + a] = 1.0
            X[t, IDX_LATENCY_LOG] = lat
            X[t, IDX_OUTLEN_LOG] = out
            X[t, IDX_ERROR_FLAG] = err

        if failure is None:
            return Episode(
                X=X, episode_id=episode_id, is_healthy=True, failure_class=None,
                tau=None, t_fail=None, severity=None,
            )
        return Episode(
            X=X, episode_id=episode_id, is_healthy=False, failure_class=fc,
            tau=tau, t_fail=T - 1, severity=sev,
        )


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def make_dataset(ds_cfg: DatasetConfig, sim_cfg: SimConfig) -> dict[str, list[Episode]]:
    """Build the train/val/cal/test splits.

    train/val are healthy-only; cal/test are healthy + injected
    (n_*_injected_per_class per failure class). Per-episode RNG derives from
    (master_seed, split, index); episode_id = f"{split}-{i:04d}".
    """
    gen = EpisodeGenerator(sim_cfg, ds_cfg.master_seed)
    plan: dict[str, tuple[int, int]] = {
        "train": (ds_cfg.n_train_healthy, 0),
        "val": (ds_cfg.n_val_healthy, 0),
        "cal": (ds_cfg.n_cal_healthy, ds_cfg.n_cal_injected_per_class),
        "test": (ds_cfg.n_test_healthy, ds_cfg.n_test_injected_per_class),
    }
    data: dict[str, list[Episode]] = {}
    for split, (n_healthy, n_inj_per_class) in plan.items():
        episodes: list[Episode] = []
        i = 0
        for _ in range(n_healthy):
            episodes.append(gen.generate(f"{split}-{i:04d}", None))
            i += 1
        for failure_class in FAILURE_CLASSES:
            for _ in range(n_inj_per_class):
                episode_id = f"{split}-{i:04d}"
                spec_rng = rng_for(ds_cfg.master_seed, split, i)
                spec = sample_failure_spec(
                    failure_class, gen.planned_length(episode_id), spec_rng, sim_cfg
                )
                episodes.append(gen.generate(episode_id, spec))
                i += 1
        data[split] = episodes
    return data


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _check_invariants(ep: Episode) -> None:
    X = ep.X
    assert X.shape == (ep.T, D_TOTAL), X.shape
    assert np.all(np.isfinite(X)), f"non-finite values in {ep.episode_id}"
    norms = np.linalg.norm(X[:, :D_SEM], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8), "semantic not unit-norm"
    assert np.all(X[:, IDX_MEAN_ENTROPY] >= 0.05 - 1e-12), "mean entropy < 0.05"
    assert np.all(X[:, IDX_MAX_ENTROPY] >= X[:, IDX_MEAN_ENTROPY] - 1e-12), "max < mean entropy"
    hf = X[:, IDX_HIGH_ENTROPY_FRAC]
    assert np.all((hf >= 0.0) & (hf <= 1.0)), "high-entropy fraction out of [0,1]"
    onehot = X[:, IDX_ACTION_ONEHOT]
    assert np.all(np.isin(onehot, (0.0, 1.0))) and np.all(onehot.sum(axis=1) == 1.0), "bad one-hot"
    errf = X[:, IDX_ERROR_FLAG]
    assert np.all(np.isin(errf, (0.0, 1.0))), "bad error flag"
    if ep.is_healthy:
        assert not np.any((errf[:-1] == 1.0) & (errf[1:] == 1.0)), "consecutive healthy errors"
        acts = onehot.argmax(axis=1)
        run = longest = 1
        for i in range(1, len(acts)):
            run = run + 1 if acts[i] == acts[i - 1] else 1
            longest = max(longest, run)
        assert longest <= 3, f"healthy action run of {longest}"
    else:
        assert ep.t_fail == ep.T - 1 and ep.tau is not None and 0 < ep.tau < ep.T


def _seg_shift(Z: np.ndarray, tau: int) -> dict[str, float]:
    """Mean |Delta| of standardized per-dim means, pre- vs post-tau, per group.

    Returns 0.0 for a group when either segment is empty, rather than taking the
    mean of an empty slice and returning NaN with a warning."""
    pre, post = Z[:tau], Z[tau:]
    width = Z.shape[1]
    out: dict[str, float] = {}
    for ch, sl in CHANNEL_SLICES.items():
        # A channel slice that lies beyond this episode's width (e.g. the
        # extended x dims on a 43-D episode) selects zero columns; report 0.0
        # rather than taking the mean of an empty slice.
        if pre.shape[0] == 0 or post.shape[0] == 0 or (sl.start or 0) >= width:
            out[ch] = 0.0
        else:
            out[ch] = float(np.mean(np.abs(post[:, sl].mean(axis=0)
                                           - pre[:, sl].mean(axis=0))))
    return out


def _min_lag_disp(E: np.ndarray, lo: int, hi: int) -> float:
    """min over lag in {2,3,4} of mean ||e_t - e_{t-lag}|| within [lo, hi)."""
    best = math.inf
    for lag in (2, 3, 4):
        if hi - lo > lag:
            d = np.linalg.norm(E[lo + lag : hi] - E[lo : hi - lag], axis=1)
            best = min(best, float(d.mean()))
    return best


def _cos_last(ep: Episode, tau: int) -> float:
    """cos(mean of last 3 semantic states, pre-tau mean direction) -- stays
    high for healthy progression, collapses under goal_drift's rotation."""
    E = ep.X[:, :D_SEM]
    pre = E[:tau].mean(axis=0)
    last = E[-3:].mean(axis=0)
    return float(_unit(last) @ _unit(pre))


def _smoke() -> None:
    import sys
    import time

    t0 = time.time()
    spec_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 424242
    cfg = SimConfig()
    gen = EpisodeGenerator(cfg, seed=20260713)

    n_healthy, n_per_class = 20, 4
    healthy = [gen.generate(f"smoke-h-{i:03d}", None) for i in range(n_healthy)]
    injected: dict[str, list[Episode]] = {}
    for fc in FAILURE_CLASSES:
        eps = []
        for i in range(n_per_class):
            eid = f"smoke-{fc}-{i:02d}"
            spec = sample_failure_spec(
                fc, gen.planned_length(eid), rng_for(spec_seed, "spec", fc, i), cfg
            )
            eps.append(gen.generate(eid, spec))
        injected[fc] = eps

    all_eps = healthy + [e for eps in injected.values() for e in eps]
    for ep in all_eps:
        _check_invariants(ep)

    # determinism: regenerating the same episode id yields identical telemetry
    assert np.array_equal(gen.generate("smoke-h-000", None).X, healthy[0].X), "not deterministic"

    # benign irregularities present in healthy runs
    total_err = sum(float(ep.X[:, IDX_ERROR_FLAG].sum()) for ep in healthy)
    assert total_err >= 1.0, "expected >=1 benign transient error across healthy episodes"

    # make_dataset API check (small sizes)
    ds = DatasetConfig(
        n_train_healthy=6, n_val_healthy=4, n_cal_healthy=4,
        n_cal_injected_per_class=1, n_test_healthy=4,
        n_test_injected_per_class=1, master_seed=123,
    )
    data = make_dataset(ds, cfg)
    assert sorted(data) == ["cal", "test", "train", "val"]
    assert len(data["train"]) == 6 and all(e.is_healthy for e in data["train"])
    assert len(data["val"]) == 4 and all(e.is_healthy for e in data["val"])
    assert len(data["cal"]) == 4 + 5 and len(data["test"]) == 4 + 5
    inj_classes = sorted(e.failure_class for e in data["test"] if not e.is_healthy)
    assert inj_classes == sorted(FAILURE_CLASSES)
    ids = [e.episode_id for split in data.values() for e in split]
    assert len(ids) == len(set(ids)), "duplicate episode ids"
    data2 = make_dataset(ds, cfg)
    assert all(
        np.array_equal(a.X, b.X) for a, b in zip(data["test"], data2["test"])
    ), "make_dataset not deterministic"
    for e in data["cal"] + data["test"]:
        _check_invariants(e)

    # ---- per-class pre/post-tau shift sanity table (standardized units) ----
    std = Standardizer().fit(healthy)
    base_lists: dict[str, list[float]] = {"e": [], "u": [], "m": []}
    base_cos: list[float] = []
    for ep in healthy:
        pseudo_tau = max(1, int(0.55 * ep.T))
        sh = _seg_shift(std.transform(ep.X), pseudo_tau)
        for ch in base_lists:
            base_lists[ch].append(sh[ch])
        base_cos.append(_cos_last(ep, pseudo_tau))
    base = {ch: float(np.mean(v)) for ch, v in base_lists.items()}
    healthy_cos = float(np.mean(base_cos))

    shifts: dict[str, dict[str, float]] = {}
    stats: dict[str, dict[str, float]] = {}
    for fc, eps in injected.items():
        acc = {"e": [], "u": [], "m": []}
        loop_ratios, disp_ratios, d_ents, err_jumps, cos_g = [], [], [], [], []
        for ep in eps:
            tau = int(ep.tau)  # type: ignore[arg-type]
            # Skip episodes whose pre- or post-onset segment is too short to
            # form a difference/mean, instead of feeding an empty slice to
            # _seg_shift/np.diff/mean and printing NaN under a PASS.
            if tau < 2 or ep.T - tau < 2:
                continue
            sh = _seg_shift(std.transform(ep.X), tau)
            for ch in acc:
                acc[ch].append(sh[ch])
            E = ep.X[:, :D_SEM]
            loop_ratios.append(
                _min_lag_disp(E, tau, ep.T) / max(1e-9, _min_lag_disp(E, 0, tau))
            )
            step_pre = np.linalg.norm(np.diff(E[:tau], axis=0), axis=1).mean()
            step_post = np.linalg.norm(np.diff(E[tau:], axis=0), axis=1).mean()
            disp_ratios.append(step_post / max(1e-9, step_pre))
            me = ep.X[:, IDX_MEAN_ENTROPY]
            d_ents.append(float(me[tau:].mean() - me[:tau].mean()))
            ef = ep.X[:, IDX_ERROR_FLAG]
            err_jumps.append(float(ef[tau:].mean() - ef[:tau].mean()))
            cos_g.append(_cos_last(ep, tau))
        shifts[fc] = {ch: float(np.mean(v)) for ch, v in acc.items()}
        stats[fc] = {
            "loop_ratio": float(np.mean(loop_ratios)),
            "disp_ratio": float(np.mean(disp_ratios)),
            "d_ent": float(np.mean(d_ents)),
            "d_err": float(np.mean(err_jumps)),
            "cos": float(np.mean(cos_g)),
        }

    print("per-channel standardized |Delta mean| pre-tau -> post-tau (mean over episodes)")
    print(f"{'class':<22}{'|dz| e':>8}{'|dz| u':>8}{'|dz| m':>8}   diagnostics")
    print(
        f"{'healthy (pseudo-tau)':<22}{base['e']:>8.2f}{base['u']:>8.2f}{base['m']:>8.2f}"
        f"   cos_pre={healthy_cos:+.2f}"
    )
    for fc in FAILURE_CLASSES:
        s, st = shifts[fc], stats[fc]
        sev_mean = float(np.mean([e.severity for e in injected[fc]]))
        print(
            f"{fc:<22}{s['e']:>8.2f}{s['u']:>8.2f}{s['m']:>8.2f}   "
            f"sev~{sev_mean:.2f}  cos_pre={st['cos']:+.2f} loop_ratio={st['loop_ratio']:.2f} "
            f"disp_ratio={st['disp_ratio']:.2f} dEnt={st['d_ent']:+.2f} dErr={st['d_err']:+.2f}"
        )

    # ---- per-class channel-signature assertions --------------------------
    def exc(fc: str, ch: str) -> float:  # shift in excess of the healthy baseline
        return max(0.0, shifts[fc][ch] - base[ch])

    # goal_drift: gradual semantic rotation -- e moves (trajectory abandons the
    # original goal direction), u and m stay near the healthy baseline.
    assert stats["goal_drift"]["cos"] < healthy_cos - 0.20, (
        f"goal_drift should rotate away from the pre-tau direction: "
        f"cos={stats['goal_drift']['cos']:+.2f} vs healthy {healthy_cos:+.2f}"
    )
    assert exc("goal_drift", "e") > 0.05 and exc("goal_drift", "m") < 0.20, (
        f"goal_drift should move e, not m: {shifts['goal_drift']} base={base}"
    )
    # grounding_loss: u leads, e nearly blind, m normal (anchors H2).
    assert exc("grounding_loss", "u") > 0.20, (
        f"grounding_loss should move u: {shifts['grounding_loss']} base={base}"
    )
    assert exc("grounding_loss", "u") > 2.5 * max(exc("grounding_loss", "e"), 0.02), (
        f"grounding_loss should move u >> e: {shifts['grounding_loss']} base={base}"
    )
    assert exc("grounding_loss", "m") < 0.20, (
        f"grounding_loss metadata should stay normal: {shifts['grounding_loss']}"
    )
    # tool_cascade: m leads (errors, latencies, retry pairs).
    assert exc("tool_cascade", "m") > 0.80, (
        f"tool_cascade should move m strongly: {shifts['tool_cascade']} base={base}"
    )
    assert stats["tool_cascade"]["d_err"] > 0.08, (
        f"tool_cascade should raise the error rate: dErr={stats['tool_cascade']['d_err']:+.2f}"
    )
    # looping: periodicity + stalled progression, entropy does not rise.
    assert stats["looping"]["loop_ratio"] < 0.80, (
        f"looping should be periodic post-tau: loop_ratio={stats['looping']['loop_ratio']:.2f}"
    )
    assert stats["looping"]["d_ent"] < 0.10, (
        f"looping entropy should not rise: dEnt={stats['looping']['d_ent']:+.2f}"
    )
    # context_corruption: step-to-step dynamics break (innovations inflate).
    assert stats["context_corruption"]["disp_ratio"] > 1.20, (
        f"context_corruption should break step dynamics: "
        f"disp_ratio={stats['context_corruption']['disp_ratio']:.2f}"
    )

    n_total = len(all_eps) + sum(len(v) for v in data.values()) + sum(len(v) for v in data2.values())
    print(
        f"PASS telemetry generator: {len(healthy)} healthy + "
        f"{n_per_class}x{len(FAILURE_CLASSES)} injected episodes ok, "
        f"invariants + determinism + make_dataset checks passed "
        f"({n_total} episodes total, {time.time() - t0:.1f}s)"
    )


if __name__ == "__main__":
    _smoke()
