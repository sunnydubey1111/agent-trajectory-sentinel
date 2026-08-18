"""Content-grounding monitor + grounded hybrids (exp/grounding-channel).

`GroundingMonitor` scores the g channel (telemetry v4, DESIGN.md amendment
3): per-dim one-sided robust z over the nine content-grounding dims,
fused by max. It is memoryless by design — content corruption is a
property of the step's data, not of temporal dynamics — and it is the
missing third information source next to the ESN (behavior) and
DeltaMahalanobis (state statistics).

Grounded hybrids extend the merged 2-way fusion (derail.monitor.hybrid), all
via `_GroundedBase` (quantile-equalized z_esn/z_maha/z_grd streams; see its
`fit()` for the equalization and lex-flag machinery each one shares):
  - HybridWeightedG   label-free max-union, quantile-equalized (see class
                      docstring). Max, not average: an earlier prototype
                      showed averaging a content signal into behavioral
                      channels dilutes its wins on exactly the classes it
                      exists for.
  - HybridContentGate content-first gating with a grounding override boost
                      (see class docstring) — the grounded default
                      `recommended_monitor` returns.
  - HybridAdaptive    soft sigmoid gate between behavioral and grounding
                      (see class docstring).
  - HybridLogisticG   learned 4-feature fusion (z_esn, z_maha, z_grd, lex),
                      quantile-equalized then clipped (see class docstring).

All causal `OnlineMonitor`s; g dims are 0 on v1 traces, so every monitor
here degrades to the ungrounded behavior when results are not recorded.
"""

from __future__ import annotations

import numpy as np

from derail.common import (D_TOTAL_EXT, D_TOTAL_GRD, GRD_SLICE, Episode,
                           OnlineMonitor)
from derail.monitor.hybrid import _HybridBase, _robust_stats

GRD_DIM_NAMES = ("query_dis", "reason_dis", "self_dis", "json_broken",
                 "char_anom", "consec_dis", "drift", "mem_dis", "lex_miss")
_EPS = 1e-9


class GroundingMonitor(OnlineMonitor):
    """Max of per-dim one-sided robust z over the content-grounding dims.

    fit() learns per-dim median/scale (1.4826*MAD, std fallback) on pooled
    healthy train steps; score_t = max_d max(0, (g_d - med_d) / scale_d)
    over the selected dims. One-sided because every g dim is oriented
    higher = more anomalous. `dims` selects a subset by name for ablation.
    """

    def __init__(self, dims: tuple[str, ...] = GRD_DIM_NAMES,
                 name: str | None = None) -> None:
        unknown = set(dims) - set(GRD_DIM_NAMES)
        if unknown:
            raise ValueError(f"unknown grounding dims: {sorted(unknown)}")
        self.dim_idx = np.array([GRD_SLICE.start + GRD_DIM_NAMES.index(d)
                                 for d in dims])
        self.name = name or ("grounding" if len(dims) == len(GRD_DIM_NAMES)
                             else f"grounding[{','.join(dims)}]")
        self._med: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        for ep in healthy_episodes:
            assert ep.X.shape[1] == D_TOTAL_GRD, \
                f"{self.name} needs grounding=True episodes (width 60)"
        G = np.concatenate([ep.X[:, self.dim_idx]
                            for ep in healthy_episodes], axis=0)
        stats = [_robust_stats(G[:, j]) for j in range(G.shape[1])]
        self._med = np.array([m for m, _ in stats])
        self._scale = np.maximum(np.array([s for _, s in stats]), _EPS)

    def start_episode(self) -> None:
        assert self._med is not None, "fit() first"

    #: Cap on any single per-dim robust z. A grounding dim that is CONSTANT on
    #: healthy train has MAD/std ~ 0, so its scale hits the floor and one binary
    #: event would explode to ~1e9 - then compound to 1e18/1e27 through the
    #: nested stream normalisations. Capping keeps every score finite
    #: and bounded; a capped z is still far above any healthy value, so
    #: detection is unaffected.
    _Z_CAP = 1e4

    def z_dims(self, x_t: np.ndarray) -> np.ndarray:
        """Per-dim one-sided robust z (attribution view of score_step)."""
        if self._med is None:
            raise RuntimeError("fit() first")
        z = (np.asarray(x_t, dtype=float)[self.dim_idx]
             - self._med) / self._scale
        z = np.clip(z, 0.0, self._Z_CAP)
        # Raised, not asserted: `python -O` strips asserts, and a NaN score
        # then flows into the alarm rule where `NaN > theta` is False - the
        # monitor reports "no alarm" for a step it could not score at all.
        # A guard that disappears under the flag a deployment is most likely
        # to use is not a guard.
        if not np.all(np.isfinite(z)):
            raise FloatingPointError(
                "non-finite grounding z; the monitor cannot score this step "
                "and must not silently report it as quiet")
        return z

    def score_step(self, x_t: np.ndarray) -> float:
        return float(np.max(self.z_dims(x_t), initial=0.0))


class _GroundedBase(_HybridBase):
    """2-way hybrid base plus a calibrated grounding stream."""

    def __init__(self, esn, maha, grd: GroundingMonitor, standardizer,
                 subs_prefit: bool = False) -> None:
        # Behavioural sub-monitors see the first 51 dims only; the grounding
        # dims (51-59) reach the score exclusively through `grd`, so they are
        # not double-counted in the behavioural Mahalanobis.
        super().__init__(esn, maha, standardizer, subs_prefit,
                         behav_slice=D_TOTAL_EXT)
        self.grd = grd
        self._grd_stats: tuple[float, float] | None = None

    def fit(self, healthy_episodes: list[Episode]) -> None:
        if not self._subs_prefit:
            self.grd.fit(healthy_episodes)
        super().fit(healthy_episodes)   # fits esn/maha + their calibration
        pool = np.concatenate([self.grd.score_episode(ep)
                               for ep in healthy_episodes])
        self._grd_stats = _robust_stats(pool)
        # Per-stream scale equalization at the "rare healthy episode" level:
        # the behavioral z's healthy per-episode maxima are heavy-tailed
        # (z ~ 5-15) while the grounding z's are near 0, so a shared alarm
        # threshold on a raw max-union swallows moderate grounding evidence
        # (measured: grounding alone detects context_corruption at 0.77,
        # a raw max-union at 0.00 — the archived "dilution" failure one
        # level up). Dividing each stream by the 95th percentile of its own
        # healthy TRAIN per-episode maxima makes "rare" the same size in
        # both streams before fusing. One-class (train only, label-free).
        maxima = {"e": [], "m": [], "b": [], "g": []}
        for ep in healthy_episodes:
            self.start_episode()
            ze, zm, zb, zg = [], [], [], []
            for x in ep.X:
                z_e, z_m, z_g = self._z3(x)
                ze.append(z_e)
                zm.append(z_m)
                zb.append(0.5 * z_e + 0.5 * z_m)
                zg.append(z_g)
            for k, v in (("e", ze), ("m", zm), ("b", zb), ("g", zg)):
                maxima[k].append(max(v))
        # 95th percentile, not the train max: measured on real_research7b,
        # max-normalization neither recovers context_corruption in-fusion
        # (structural: a shared 5% FA threshold prices out the moderate
        # grounding evidence) nor preserves the content gain as well
        # (pooled +0.21 vs +0.24 at the 95th).
        q = {k: max(float(np.quantile(v, 0.95, method="higher")), _EPS)
             for k, v in maxima.items()}
        self._q_e, self._q_m, self._q_b, self._q_g = (q["e"], q["m"],
                                                      q["b"], q["g"])
        # Grounding override trip point: the healthy-train MAXIMUM of the
        # normalized grounding stream — "no healthy training episode ever
        # reached this". Val-quantile split budgets collapse at small val
        # sizes (2.5% of 24 episodes rounds to theta = the max, killing the
        # behavioral stream); the train-max trip spends ~zero FA budget
        # instead of half of it.
        self._g_trip = max(float(np.max(maxima["g"])) / self._q_g, 1.0)
        # Binary lexical flag (wrong_document): a direct override, but only
        # in domains where it is PROVABLY clean on healthy train episodes
        # (zero flags). Where the healthy null is dirty (e.g. long research
        # episodes with wordy zero-overlap results), it self-disables —
        # detection there falls to the continuous dims / supervised fusion.
        # It must also never enter the continuous trip calibration: pass a
        # continuous-dims GroundingMonitor to hybrids (see make_grounded).
        from derail.common import IDX_GRD_LEX_MISS
        if healthy_episodes[0].X.shape[1] > IDX_GRD_LEX_MISS:
            self._lex_clean = not any(
                ep.X[:, IDX_GRD_LEX_MISS].max() > 0.0
                for ep in healthy_episodes)
        else:
            self._lex_clean = False

    def start_episode(self) -> None:
        super().start_episode()
        self.grd.start_episode()

    def _z3(self, x_t: np.ndarray) -> tuple[float, float, float]:
        z_e, z_m = self._z_scores(x_t)
        mg, sg = self._grd_stats
        z_g = (self.grd.score_step(x_t) - mg) / sg
        return z_e, z_m, float(z_g)

    def _lex(self, x_t: np.ndarray) -> float:
        """Lexical override contribution: the flag, iff clean-null domain."""
        from derail.common import IDX_GRD_LEX_MISS
        if not self._lex_clean or x_t.shape[0] <= IDX_GRD_LEX_MISS:
            return 0.0
        return float(x_t[IDX_GRD_LEX_MISS])

    def score_episode_streams(self, ep: Episode
                              ) -> tuple[np.ndarray, np.ndarray]:
        """(behavioral, grounding) per-step streams in healthy-rare units.

        For dual-budget deployment: each stream is thresholded on ITS OWN
        healthy-val quantile at a split FA budget, alarm = OR. This is a
        thresholding protocol, not a new score — total budget is preserved.
        """
        self.start_episode()
        zb, zg = [], []
        for x in ep.X:
            z_e, z_m, z_g = self._z3(x)
            zb.append((0.5 * z_e + 0.5 * z_m) / self._q_b)
            zg.append(z_g / self._q_g)
        return np.array(zb), np.array(zg)


class HybridWeightedG(_GroundedBase):
    """Label-free grounded fusion: quantile-equalized max-union.

    s_t = max(behavioral / q_b, grounding / q_g) where q_b, q_g are each
    stream's healthy-train 95th-percentile per-episode maximum (see
    _GroundedBase.fit). A stream crosses ~1.0 exactly when it exceeds its
    own healthy-rare level, so neither stream can drown the other.
    """

    name = "hybrid_weighted_g"

    def score_step(self, x_t: np.ndarray) -> float:
        z_e, z_m, z_g = self._z3(x_t)
        return max((0.5 * z_e + 0.5 * z_m) / self._q_b, z_g / self._q_g)


class HybridContentGate(_GroundedBase):
    """Content-first gating: behavioral ordering + a grounding override.

    s_t = z_b/q_b + BOOST * relu(z_g/q_g − 1): episodes stay ordered by the
    behavioral score UNLESS their grounding stream exceeds its own
    healthy-rare level, in which case they are boosted past any healthy
    behavioral tail. Diagnosed motivation: the plain max-union's val
    threshold is set by behavioral tail outliers (one healthy val episode
    at 8.26 q_b vs context evidence at 1.5-2.6 q_g — 16/17 grounding
    detections lost); the gate makes grounding evidence incommensurable
    with that tail instead of competing with it.
    """

    name = "hybrid_content_gate"
    BOOST = 10.0

    def score_step(self, x_t: np.ndarray) -> float:
        z_e, z_m, z_g = self._z3(x_t)
        zb = (0.5 * z_e + 0.5 * z_m) / self._q_b
        zg = z_g / self._q_g
        return (zb + self.BOOST * max(zg - self._g_trip, 0.0)
                + self.BOOST * self._lex(x_t))


class HybridAdaptive(_GroundedBase):
    """Grounding-confidence-adaptive weighting (soft gate).

    w_t = sigmoid(4 * (z_g/q_g − 1)); s_t = (1−w)*z_b/q_b + w*z_g/q_g.
    Below its healthy-rare level the grounding stream is nearly ignored;
    above it, it takes over smoothly.
    """

    name = "hybrid_adaptive"

    def score_step(self, x_t: np.ndarray) -> float:
        z_e, z_m, z_g = self._z3(x_t)
        zb = (0.5 * z_e + 0.5 * z_m) / self._q_b
        zg = z_g / self._q_g
        w = 1.0 / (1.0 + np.exp(-4.0 * (zg - 1.0)))
        return float((1.0 - w) * zb + w * zg + 10.0 * self._lex(x_t))


class HybridLogisticG(_GroundedBase):
    """Learned 3-feature fusion over quantile-equalized streams.

    Features are [z_esn/q_e, z_maha/q_m, z_grd/q_g] — each stream in units
    of its own healthy-rare level (95th-pct healthy-train per-episode max),
    then clipped. Without this the grounding feature's scale mismatch
    starves its coefficient (measured: coef_g 0.09, context_corruption
    detection 0 despite the standalone grounding monitor detecting 0.77).
    """

    name = "hybrid_logistic_g"

    def __init__(self, esn, maha, grd, standardizer,
                 subs_prefit: bool = False, clip: float = 10.0) -> None:
        super().__init__(esn, maha, grd, standardizer, subs_prefit)
        self.clip = float(clip)
        self.coef_ = np.array([0.25, 0.25, 0.25, 0.25])
        self.intercept_ = 0.0
        self.supervised_ = False

    def _feat(self, x_t: np.ndarray) -> tuple[float, float, float, float]:
        from derail.common import IDX_GRD_LEX_MISS
        z_e, z_m, z_g = self._z3(x_t)
        c = self.clip
        lex = (float(x_t[IDX_GRD_LEX_MISS])
               if x_t.shape[0] > IDX_GRD_LEX_MISS else 0.0)
        # lex enters RAW (0/1): the supervised weights decide its worth per
        # domain via cross-fit, so no clean-null gating is needed here.
        return (float(np.clip(z_e / self._q_e, -c, c)),
                float(np.clip(z_m / self._q_m, -c, c)),
                float(np.clip(z_g / self._q_g, -c, c)),
                lex)

    def step_features(self, ep: Episode) -> np.ndarray:
        self.start_episode()
        return np.array([self._feat(x) for x in ep.X], dtype=float)

    def fit_supervised(self, healthy_episodes: list[Episode],
                       injected_episodes: list[Episode]) -> None:
        from sklearn.linear_model import LogisticRegression

        # Reset supervised state before every fit: a degenerate-label
        # early return or a refit must not keep a previous fit's coefficients.
        self.coef_ = np.array([0.25, 0.25, 0.25, 0.25])
        self.intercept_ = 0.0
        self.supervised_ = False

        feats, labels = [], []
        for ep in healthy_episodes:
            feats.append(self.step_features(ep))
            labels.append(np.zeros(ep.T, dtype=int))
        for ep in injected_episodes:
            feats.append(self.step_features(ep))
            y = np.zeros(ep.T, dtype=int)
            y[ep.tau:] = 1
            labels.append(y)
        X = np.concatenate(feats, axis=0)
        y = np.concatenate(labels)
        if len(np.unique(y)) < 2:
            return
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X, y)
        self.coef_ = clf.coef_[0].astype(float)
        self.intercept_ = float(clf.intercept_[0])
        self.supervised_ = True

    def score_step(self, x_t: np.ndarray) -> float:
        return float(np.array(self._feat(x_t)) @ self.coef_
                     + self.intercept_)


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import time

    from derail.common import Standardizer, rng_for
    from derail.monitor.baselines import DeltaMahalanobisMonitor
    from derail.monitor.esn import ChannelMaxESNMonitor
    from derail.telemetry.adapter import episode_from_trace

    t0 = time.time()
    rng = rng_for(0, "grounding", "smoke")
    good_results = ['{"price": %d, "city": "Osaka"}' % (200 + i)
                    for i in range(40)]

    def _mk(eid: str, n: int, corrupt_from: int | None = None,
            mode: str = "json") -> Episode:
        steps = []
        for t in range(n):
            res = good_results[t % len(good_results)]
            if corrupt_from is not None and t >= corrupt_from:
                res = ('{"price": 215,, "city" "Osaka"}' if mode == "json"
                       else "■�� 9$$#@@!!�■ 0xFF")
            steps.append({
                "text": f"checking price [db_query({{\"sql\": \"S{t % 5}\"}})"
                        f" -> {res}]",
                "token_logprobs": (-rng.exponential(0.4, size=12)).tolist(),
                "action": "tool_call", "latency_s": 1.0, "output_tokens": 12,
            })
        tau = corrupt_from
        return episode_from_trace(steps, eid, tau=tau,
                                  failure_class=None if tau is None
                                  else "context_corruption",
                                  severity=None if tau is None else 0.5,
                                  use_sentence_transformers=False,
                                  grounding=True)

    train = [_mk(f"h{i}", 12) for i in range(20)]
    healthy_ep = _mk("ht", 12)
    bad_json = _mk("fj", 12, corrupt_from=6, mode="json")
    bad_garb = _mk("fg", 12, corrupt_from=6, mode="garbage")

    g = GroundingMonitor()
    g.fit(train)
    for ep in (bad_json, bad_garb):
        s_f, s_h = g.score_episode(ep), g.score_episode(healthy_ep)
        assert s_f[6:].mean() > s_h[6:].mean() + 1.0, ep.episode_id
    # streaming == batch
    g.start_episode()
    streamed = np.array([g.score_step(x) for x in bad_json.X])
    assert np.allclose(streamed, g.score_episode(bad_json))
    # ablation subsets behave
    gj = GroundingMonitor(dims=("json_broken",))
    gj.fit(train)
    assert gj.score_episode(bad_json)[6:].min() > 0.0
    assert gj.score_episode(bad_garb)[6:].max() == 0.0, \
        "json dim must not fire on garbage text (char_anom's job)"

    # Behavioural submodels live on the 51-dim view; the grounded
    # hybrid masks scoring to 51 via behav_slice, so the submodel standardizer
    # and reservoir must be 51-dim to match.
    def _v51(eps):
        return [Episode(X=e.X[:, :D_TOTAL_EXT].copy(), episode_id=e.episode_id,
                        is_healthy=e.is_healthy, failure_class=e.failure_class,
                        tau=e.tau, t_fail=e.t_fail, severity=e.severity)
                for e in eps]

    train51 = _v51(train)
    std60 = Standardizer().fit(train)          # 60-dim: grounded hybrid wrapper
    std51 = Standardizer().fit(train51)        # 51-dim: behavioural submodels
    esn = ChannelMaxESNMonitor(std51, channels=("e", "u", "m", "x"), K=4,
                               cusum=True, seed=9)
    maha = DeltaMahalanobisMonitor(std51)
    esn.fit(train51)
    maha.fit(train51)
    for cls in (HybridWeightedG, HybridLogisticG):
        grd = GroundingMonitor()
        grd.fit(train)
        mon = cls(esn, maha, grd, std60, subs_prefit=True)
        mon.fit(train)
        if isinstance(mon, HybridLogisticG):
            cal = [_mk(f"c{i}", 12, corrupt_from=6,
                       mode="json" if i % 2 else "garbage")
                   for i in range(6)]
            mon.fit_supervised(train[:8], cal)
            assert mon.supervised_
        s_f = mon.score_episode(bad_json)
        s_h = mon.score_episode(healthy_ep)
        assert s_f[6:].mean() > s_h[6:].mean(), mon.name
        mon.start_episode()
        st = np.array([mon.score_step(x) for x in bad_json.X])
        assert np.allclose(st, mon.score_episode(bad_json)), mon.name
        print(f"  {mon.name:<18} corrupt {s_f[6:].mean():8.2f}  "
              f"healthy {s_h[6:].mean():6.2f}")

    # recommended_monitor auto-upgrades to the grounded union on v4 data
    from derail.monitor.hybrid import recommended_monitor
    rec = recommended_monitor(std60, train, channels=("e", "u", "m", "x"),
                              K=4, seed=9)
    # name check, not isinstance: under `py -m derail.monitor.grounding`
    # this file is __main__ while recommended_monitor imports the package
    # copy — two class objects, same class.
    assert type(rec).__name__ == "HybridContentGate", type(rec).__name__
    assert (rec.score_episode(bad_json)[6:].mean()
            > rec.score_episode(healthy_ep)[6:].mean())

    print(f"PASS grounding smoke test ({time.time() - t0:.1f}s)")
