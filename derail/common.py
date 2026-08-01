"""Shared contract for the agent-trajectory-sentinel project.

Every module imports from here. This file is the single source of truth for:
  - the step-signal channel layout of x_t = [e_t; u_t; m_t]
  - the Episode / FailureSpec dataclasses
  - the OnlineMonitor abstract base class (causal, streaming scoring)
  - the Standardizer (z-scoring fit on healthy TRAIN steps only)
  - deterministic per-component RNG derivation
  - global configuration dataclasses (simulator, dataset, cost model)

Do not change constants or signatures here without updating DESIGN.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Channel layout of the step signal x_t  (total dimension D_TOTAL = 43)
# --------------------------------------------------------------------------
# semantic channel e_t: dims [0, 32) -- synthetic "embedding" of step output,
#   approximately unit-norm.
# uncertainty channel u_t: dims [32, 36). Token-level uncertainty, in nats.
#   The SIMULATOR generates these as sampled token entropy; the real-trace
#   adapter fills them with sampled-token SURPRISAL (-log p of the token the
#   model actually emitted), which is what provider APIs expose. The two are
#   not the same quantity and are never mixed in one corpus.
#   32: mean per-token uncertainty of the step (>= 0; -1 marks 'not measured')
#   33: max per-token uncertainty of the step
#   34: within-step slope (healthy: slightly negative)
#   35: high-uncertainty token fraction in [0, 1]
# metadata channel m_t: dims [36, 43):
#   36..39: action-type one-hot over ACTION_TYPES (plan, tool_call,
#           tool_result, synthesis)
#   40: log tool/step latency (log-seconds)
#   41: log output length (log-tokens)
#   42: error flag in {0.0, 1.0}

D_SEM = 32
D_UNC = 4
D_META = 7
D_TOTAL = D_SEM + D_UNC + D_META  # 43

SEM_SLICE = slice(0, D_SEM)                          # e_t
UNC_SLICE = slice(D_SEM, D_SEM + D_UNC)              # u_t
META_SLICE = slice(D_SEM + D_UNC, D_TOTAL)           # m_t

CHANNEL_SLICES: dict[str, slice] = {
    "e": SEM_SLICE,
    "u": UNC_SLICE,
    "m": META_SLICE,
    # "x" (derived, dims [43, 51)) is registered below once EXT_SLICE exists;
    # only episodes built with telemetry.adapter extended=True have it.
}

# Named indices inside u_t / m_t (absolute indices into x_t).
IDX_MEAN_ENTROPY = 32
IDX_MAX_ENTROPY = 33
IDX_ENTROPY_SLOPE = 34
IDX_HIGH_ENTROPY_FRAC = 35
IDX_ACTION_ONEHOT = slice(36, 40)
IDX_LATENCY_LOG = 40
IDX_OUTLEN_LOG = 41
IDX_ERROR_FLAG = 42

# --------------------------------------------------------------------------
# Extended derived channel x_t (telemetry v3): dims [43, 51).
# Computed CAUSALLY by telemetry.adapter from raw step dicts — real-trace and
# demo pipelines only. The simulator and the core 43-dim study are untouched:
# Episodes may be D_TOTAL or D_TOTAL_EXT wide, and monitors that select only
# ("e", "u", "m") behave identically on both. Nothing here uses lookahead.
# --------------------------------------------------------------------------
D_EXT = 8
D_TOTAL_EXT = D_TOTAL + D_EXT  # 51

EXT_SLICE = slice(D_TOTAL, D_TOTAL_EXT)  # x_t (derived)

IDX_COS_DRIFT = 43         # 1 - cos(e_t, e_{t-1}); 0 at t=0
IDX_TASK_SIM = 44          # cos(e_t, e_0) — similarity to the task anchor
IDX_TOOL_SUCCESS = 45      # per-step tool success rate in [0, 1]; 1 if no tools
IDX_RETRY_COUNT = 46       # tool calls in the step repeating an earlier (name, args)
IDX_TOOL_LATENCY = 47      # log(step latency / #tool calls); 0 if no tools
IDX_CTX_RATIO = 48         # approx cumulative tokens / context budget (capped)
IDX_REASON_DEPTH = 49      # number of tool calls in the step
IDX_SELF_CONSISTENCY = 50  # cos(e_t, mean(e_0..e_{t-1})); 1 at t=0

CHANNEL_SLICES["x"] = EXT_SLICE

# --------------------------------------------------------------------------
# OPTIONAL content-grounding channel g_t (telemetry v4, dims [51, 60)).
# Only episodes built with telemetry.adapter grounding=True have it (which
# implies extended=True); it measures whether tool-result CONTENT stays
# grounded, the information source that behavioral/statistical monitors
# lack for content-corruption failures. Every dim is causal and oriented
# so that HIGHER = more anomalous. Requires v2+ step text ("[name({args})
# -> result]"): on v1 traces (no recorded results) all nine dims are
# identically 0 (inert), never an error.
# --------------------------------------------------------------------------
D_GRD = 9
D_TOTAL_GRD = D_TOTAL_EXT + D_GRD  # 60

GRD_SLICE = slice(D_TOTAL_EXT, D_TOTAL_GRD)  # g_t (derived, content)

IDX_GRD_QUERY_DIS = 51    # mean 1 - cos(result, its own query+args); 0 if no tools
IDX_GRD_REASON_DIS = 52   # 1 - cos(step results, agent text sans tool bits)
IDX_GRD_SELF_DIS = 53     # 1 - cos(step results, running mean of past results)
IDX_GRD_JSON_BROKEN = 54  # fraction of JSON-looking results with a syntax
                          # error BEFORE end-of-string (truncation-consistent
                          # cuts are NOT flagged — see adapter._json_broken)
IDX_GRD_CHAR_ANOM = 55    # |alnum-ratio shift| vs the episode's running mean
                          # + non-printable/replacement-char fraction
IDX_GRD_CONSEC_DIS = 56   # 1 - cos(step results, PREVIOUS step's results) —
                          # sharper than the centroid for sudden corruption
IDX_GRD_DRIFT = 57        # causal EWMA (alpha 0.4) of self_dis — persistent
                          # grounding drift rather than one-step wobble
IDX_GRD_MEM_DIS = 58      # 1 - cos(agent text sans tool bits, mean of the
                          # LAST 3 steps' result embeddings) — reasoning vs
                          # recent retrieved context (windowed memory, not
                          # the whole-episode centroid)
IDX_GRD_LEX_MISS = 59     # lexical retrieval-relevance miss: for DOCUMENT
                          # results (>= 4 content words, not error/diagnostic
                          # strings), 1 - max(query-term, task-term coverage).
                          # Pure string ops — no embedding. Char-trigram
                          # cosine cannot separate off-topic documents (its
                          # healthy null is too noisy); exact content-word
                          # coverage can (wrong_document sits at 1.0).

CHANNEL_SLICES["g"] = GRD_SLICE

ACTION_TYPES: tuple[str, ...] = ("plan", "tool_call", "tool_result", "synthesis")

FAILURE_CLASSES: tuple[str, ...] = (
    "goal_drift",
    "looping",
    "tool_cascade",
    "grounding_loss",
    "context_corruption",
)

# Tool-layer failure classes injected by the real-agent harness (WS4,
# derail.harness.inject). Kept SEPARATE from the simulator's FAILURE_CLASSES
# — the simulator never generates these — but an Episode may be labeled with
# either set. Names shared with FAILURE_CLASSES (looping / tool_cascade /
# context_corruption) are intentionally not duplicated here.
REAL_FAILURE_CLASSES: tuple[str, ...] = (
    "wrong_document",
    "malformed_json",
    "rate_limit",
    "timeout",
    "sql_timeout",
    "mcp_unavailable",
    "browser_fail",
    # Imported corpora label failures with their own taxonomy, which does not
    # map onto ours. `external` says exactly that: a labelled failure whose
    # mechanism this project did not define and must not claim to know. The
    # importing corpus keeps its own label in its manifest, and per-class
    # numbers for such a corpus are grouped on that, never on this.
    "external",
)

MASTER_SEED = 20260713


# --------------------------------------------------------------------------
# RNG derivation: deterministic, component-isolated streams
# --------------------------------------------------------------------------
def stable_hash(*tags: object, mod: int = 2**63) -> int:
    """Platform- and process-stable hash of the tags (no PYTHONHASHSEED salt).

    Builtin hash() on strings is salted per interpreter process, so it must
    never seed anything reproducible. This is the same character-hash the
    rng streams use, exposed for callers that need an int seed rather than a
    Generator (e.g. trace collectors deriving a per-episode world seed).
    """
    h = 0
    for tag in tags:
        for ch in str(tag):
            h = (h * 1000003 + ord(ch)) % mod
    return h


def rng_for(seed: int, *tags: object) -> np.random.Generator:
    """Derive an independent np.random.Generator from a seed and hashable tags.

    Same (seed, tags) always yields the same stream; different tags yield
    statistically independent streams. Use one stream per episode / per
    ensemble member so that component changes don't perturb other components.
    """
    return np.random.default_rng(
        np.random.SeedSequence([seed, stable_hash(*tags)]))


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FailureSpec:
    """Ground-truth description of an injected failure."""

    failure_class: str            # one of FAILURE_CLASSES
    tau: int                      # 0-indexed onset step
    severity: float               # in (0, 1]; higher = more severe/obvious
    ramp_steps: int               # steps over which the shift ramps in


@dataclass
class Episode:
    """One agent episode. X[t] is the step signal x_t (shape (T, D_TOTAL))."""

    X: np.ndarray                 # (T, D_TOTAL) float64
    episode_id: str
    is_healthy: bool
    failure_class: Optional[str]  # None iff is_healthy
    tau: Optional[int]            # None iff is_healthy (0-indexed onset)
    t_fail: Optional[int]         # None iff is_healthy; == T - 1 for injected
    severity: Optional[float]     # None iff is_healthy
    T: int = 0                    # set in __post_init__ from X

    def __post_init__(self) -> None:
        self.T = int(self.X.shape[0])
        assert self.X.shape[1] in (D_TOTAL, D_TOTAL_EXT, D_TOTAL_GRD), \
            f"bad X width {self.X.shape}"
        if self.is_healthy:
            assert self.failure_class is None and self.tau is None
        else:
            assert (self.failure_class in FAILURE_CLASSES
                    or self.failure_class in REAL_FAILURE_CLASSES)
            assert self.tau is not None and 0 < self.tau < self.T
            assert self.t_fail == self.T - 1


# --------------------------------------------------------------------------
# Simulator configuration
# --------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Parameters of the healthy-episode generator and the failure injector.

    The generator module owns the interpretation of these; DESIGN.md section
    'Telemetry generator' specifies the required qualitative behavior per
    failure class.
    """

    d_sem: int = D_SEM
    # healthy episode length (steps), inclusive bounds
    T_min: int = 25
    T_max: int = 60
    # healthy semantic dynamics
    n_waypoints_min: int = 3
    n_waypoints_max: int = 6
    waypoint_sigma: float = 0.35   # spread of subtask waypoints around goal
    ar_rho: float = 0.75           # AR(1) coherence of semantic noise
    ar_sigma: float = 0.06         # innovation std of semantic noise
    pull_goal: float = 0.15        # progress-dependent pull toward goal
    # healthy uncertainty channel
    entropy_base: dict = field(default_factory=lambda: {
        "plan": 1.6, "tool_call": 1.1, "tool_result": 0.7, "synthesis": 1.4})
    entropy_completion_drop: float = 0.5   # linear decline with progress
    entropy_noise: float = 0.18
    # healthy metadata channel
    healthy_error_rate: float = 0.02       # rare transient tool errors
    latency_lognorm: dict = field(default_factory=lambda: {
        "plan": (-1.2, 0.4), "tool_call": (0.8, 0.6),
        "tool_result": (-0.5, 0.4), "synthesis": (-0.9, 0.4)})
    outlen_lognorm: dict = field(default_factory=lambda: {
        "plan": (4.5, 0.5), "tool_call": (3.0, 0.4),
        "tool_result": (4.0, 0.7), "synthesis": (5.0, 0.5)})
    # failure injection
    tau_frac_min: float = 0.3      # tau ~ Uniform[0.3 T, 0.7 T]
    tau_frac_max: float = 0.7
    fail_horizon_min: int = 8      # steps from tau to t_fail
    fail_horizon_max: int = 20
    severity_min: float = 0.35
    severity_max: float = 1.0
    ramp_steps_min: int = 2        # onset ramp length (longer for subtle)
    ramp_steps_max: int = 6


# --------------------------------------------------------------------------
# Dataset configuration (episode counts per split)
# --------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Split sizes. All monitor FITTING uses train (healthy only).

    val: healthy-only; threshold selection + score normalization checks.
    cal: healthy + injected; used ONLY by the oracle (isotonic) calibrator
         and escalation-policy tuning -- never by monitors. Its
         injected/healthy composition matches test (60*5/120 = 400/160 =
         71.4% injected) so the oracle calibrator is fit at the prevalence
         it is evaluated at.
    test: healthy + injected; all reported metrics.
    """

    n_train_healthy: int = 240
    n_val_healthy: int = 120
    n_cal_healthy: int = 120
    n_cal_injected_per_class: int = 60
    n_test_healthy: int = 160
    n_test_injected_per_class: int = 80
    master_seed: int = MASTER_SEED


# --------------------------------------------------------------------------
# Cost model (escalation experiments, H3)
# --------------------------------------------------------------------------
COST_STEP: float = 1.0    # cost of one executed agent step (1 LLM call)
COST_JUDGE: float = 1.0   # cost of one judge-LLM call


@dataclass
class JudgeConfig:
    """Modeled judge-LLM: a noisy oracle over 'is this episode derailed now'."""

    p_detect: float = 0.90   # P(positive | t >= tau) per call
    p_false: float = 0.02    # P(positive | healthy or t < tau) per call
    debounce: int = 2        # consecutive positive verdicts required to halt
    cooldown: int = 3        # steps to suppress re-escalation after a negative


# --------------------------------------------------------------------------
# Online monitor contract
# --------------------------------------------------------------------------
class OnlineMonitor(ABC):
    """A causal, streaming derailment scorer.

    Lifecycle:
        m = SomeMonitor(...); m.fit(healthy_train_episodes)
        for each episode: m.start_episode(); s_t = m.score_step(x_t) per step

    Constraints:
      - fit() sees HEALTHY episodes only (one-class).
      - score_step(x_t) may use only x_1..x_t of the current episode plus
        anything learned in fit(). No lookahead, no post-hoc smoothing.
      - Higher score = more anomalous. Scores need not be comparable across
        monitors; thresholds are calibrated per-monitor on healthy val.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(self, healthy_episodes: list[Episode]) -> None: ...

    @abstractmethod
    def start_episode(self) -> None: ...

    @abstractmethod
    def score_step(self, x_t: np.ndarray) -> float: ...

    def score_episode(self, episode: Episode) -> np.ndarray:
        """Convenience: stream the episode through score_step. Returns (T,)."""
        self.start_episode()
        return np.array([self.score_step(x) for x in episode.X], dtype=float)


# --------------------------------------------------------------------------
# Degenerate-scale contract (DESIGN.md Amendment 6)
# --------------------------------------------------------------------------
DEGENERATE_EPS = 1e-9      #: below this a scale counts as "no variation at all"


def safe_scale(scale, eps: float = DEGENERATE_EPS):
    """Replace a degenerate scale with 1.0 instead of a tiny floor.

    A quantity that never varied in healthy runs gives no basis for counting
    sigmas. Clamping its scale to a small epsilon would make an uninformative
    channel the most sensitive one in the system; returning 1.0 reports the
    RAW deviation in the quantity's own units, so a genuinely novel event
    stays visible without inventing a sigma count the data cannot support.
    Non-degenerate scales are returned unchanged.

    Contract and measured effect: DESIGN.md Amendment 6.
    """
    s = np.asarray(scale, dtype=float)
    out = np.where(s < eps, 1.0, s)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------
# Standardizer (fit on pooled healthy TRAIN steps only)
# --------------------------------------------------------------------------
class Standardizer:
    """Per-dimension z-scoring. Fit once on healthy train, share everywhere."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, episodes: list[Episode]) -> "Standardizer":
        X = np.concatenate([ep.X for ep in episodes], axis=0)
        self.mean_ = X.mean(axis=0)
        # A dim with any real variation keeps the 1e-3 floor (unchanged). Only
        # a dim with NO healthy variation at all is passed through unscaled,
        # instead of being divided by the floor — see safe_scale.
        sd = X.std(axis=0)
        self.std_ = np.where(sd < DEGENERATE_EPS, 1.0, np.maximum(sd, 1e-3))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Standardizer not fitted"
        return (X - self.mean_) / self.std_
