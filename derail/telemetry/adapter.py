"""Real-trace adapter: turn logged agent runs into Episodes (weakness A/B).

The whole study runs on the `Episode` interface — a (T, 43) array plus
metadata — so external validation is an adapter problem, not a rewrite. This
module converts JSONL traces exported from any agent framework (LangGraph,
AutoGen, CrewAI, a bespoke loop) into Episodes the monitors consume as-is.

Expected trace format: one JSON object per step, one file (or list) per run:

    {"text": "<the step's output text>",
     "token_logprobs": [-0.02, -1.31, ...],   # sampled-token logprobs (nats)
     "action": "tool_call",                    # free-form; mapped below
     "latency_s": 1.42,
     "output_tokens": 213,                     # optional; len(logprobs) if absent
     "error": false}

Channel construction:
  e_t  — a deterministic char-3-gram feature-hashing embedding (512 buckets ->
         seeded projection to 32, L2-normalized). This is the ONLY embedding
         used unless a caller passes use_sentence_transformers=True explicitly
         (a MiniLM embedding projected to 32 dims; measured to add nothing on
         this study's traces — never enabled implicitly).
  u_t  — SURPRISAL-based proxies for the entropy channel, since sampled-token
         logprobs are what APIs expose: mean/max of -logprob, within-step
         linear slope of -logprob, fraction of tokens with -logprob > 2.5
         nats. Retrain monitors on healthy runs from the SAME source; do not
         mix with simulator-trained monitors.
  m_t  — action one-hot via ACTION_MAP (unknown -> "tool_call"), log latency,
         log output length, error flag.
  x_t  — OPTIONAL derived channel (telemetry v3, dims [43, 51), via
         extended=True): cosine embedding drift vs the previous step,
         similarity to the task anchor (step 0), per-step tool success rate,
         retry count (exact repeats of earlier calls), log per-tool latency,
         approximate context-utilization ratio, reasoning depth (# tool
         calls), and self-consistency vs the episode's own running mean
         embedding. All computed causally from x_0..x_t; tool calls are
         parsed from the v2 step-text format "[name({args}) -> result]" so
         stored traces and live loops share one code path.

Ground-truth labels: pass tau/failure_class if you have them (for evaluation);
healthy monitoring deployment needs none.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from derail.common import (
    ACTION_TYPES,
    D_SEM,
    D_TOTAL,
    D_TOTAL_EXT,
    D_TOTAL_GRD,
    IDX_COS_DRIFT,
    IDX_CTX_RATIO,
    IDX_GRD_CHAR_ANOM,
    IDX_GRD_CONSEC_DIS,
    IDX_GRD_DRIFT,
    IDX_GRD_JSON_BROKEN,
    IDX_GRD_LEX_MISS,
    IDX_GRD_MEM_DIS,
    IDX_GRD_QUERY_DIS,
    IDX_GRD_REASON_DIS,
    IDX_GRD_SELF_DIS,
    IDX_REASON_DEPTH,
    IDX_RETRY_COUNT,
    IDX_SELF_CONSISTENCY,
    IDX_TASK_SIM,
    IDX_TOOL_LATENCY,
    IDX_TOOL_SUCCESS,
    Episode,
    rng_for,
)
from derail.preconditions import error_shaped
from derail.telemetry.events import (SCHEMA_VERSION, ToolCallEvent,
                                     canonical_args, make_tool_event,
                                     parse_step_events, parse_tool_bits)

_HASH_BUCKETS = 512
_HIGH_SURPRISAL_NATS = 2.5
_PROJECTION_SEED = 811_2026

# Telemetry v3: assumed context budget for the utilization ratio (tokens).
# Override with AGENTWATCH_CTX_BUDGET_TOKENS to match the deployment.
#
# This is a per-DEPLOYMENT fact, not a constant of the system, and getting it
# wrong makes a feature silently useless rather than wrong: `IDX_CTX_RATIO`
# is cumulative tokens over this budget, so on a 128k-context model an 8192
# budget saturates the ratio at its 2.0 clamp within a few steps and the dim
# carries no signal at all — dead rather than absent, which nothing reports.
# Every corpus here was collected against models of roughly this budget.
CTX_BUDGET_TOKENS = float(os.environ.get("AGENTWATCH_CTX_BUDGET_TOKENS",
                                         "8192"))
#: Sentence-embedding model, when `use_sentence_transformers` is on. Recorded
#: in provenance because changing it changes every semantic dim, so two
#: corpora embedded with different models are not comparable.
ST_MODEL_NAME = os.environ.get("AGENTWATCH_ST_MODEL", "all-MiniLM-L6-v2")
# Tool calls are read from the structured `tool_events` field when present
# and parsed out of the step text otherwise; see derail.telemetry.events.

ACTION_MAP = {
    # canonical
    "plan": "plan", "tool_call": "tool_call", "tool_result": "tool_result",
    "synthesis": "synthesis",
    # common framework vocabularies
    "think": "plan", "thought": "plan", "reasoning": "plan",
    "reflect": "plan", "act": "tool_call", "action": "tool_call",
    "function_call": "tool_call", "tool": "tool_call",
    "observation": "tool_result", "tool_response": "tool_result",
    "function_result": "tool_result", "final": "synthesis",
    "answer": "synthesis", "respond": "synthesis", "response": "synthesis",
}

#: Action names seen that `ACTION_MAP` does not know, in order of first sight.
#: An unmapped name is folded into `tool_call`, which is a guess: a genuinely
#: new step kind is then indistinguishable from a real tool call in the one-hot
#: dims. An explicit "unknown" category would be the honest encoding, but
#: `ACTION_TYPES` is part of the frozen feature schema (`common.D_TOTAL` and
#: the absolute `IDX_*` offsets), so adding one renumbers every dim downstream
#: and moves every published number. Recording the names instead makes the
#: guess visible to a caller who wants to check whether it was ever exercised.
UNMAPPED_ACTIONS: dict[str, int] = {}


def unmapped_actions() -> dict[str, int]:
    """Unknown action names encountered so far, and how often."""
    return dict(UNMAPPED_ACTIONS)


def _action_of(step: dict) -> str:
    raw = str(step.get("action", "")).lower()
    action = ACTION_MAP.get(raw)
    if action is None:
        if raw:
            UNMAPPED_ACTIONS[raw] = UNMAPPED_ACTIONS.get(raw, 0) + 1
        action = "tool_call"
    return action


def _latency_of(step: dict) -> float:
    """Step latency in seconds, floored at 1ms; negative values are rejected.

    A negative duration is not a small one, it is a broken record: clamping it
    to the floor turns a corrupt trace into a plausible-looking fast step and
    feeds it to the monitor as evidence.
    """
    value = float(step.get("latency_s", 1.0))
    if value < 0.0:
        raise TraceSchemaError(
            f"negative latency_s ({value!r}); a duration cannot be negative, "
            f"so this trace is corrupt rather than merely fast")
    return max(value, 1e-3)

# Lazily-loaded MiniLM model; only ever touched when a caller EXPLICITLY
# passes use_sentence_transformers=True. Installing sentence-transformers
# must never change the behavior of code that didn't ask for it.
_ST_MODEL = None


def _projection(d_in: int) -> np.ndarray:
    """Fixed seeded Gaussian projection d_in -> D_SEM (deterministic)."""
    rng = rng_for(_PROJECTION_SEED, "trace-projection", d_in)
    return rng.standard_normal((d_in, D_SEM)) / math.sqrt(d_in)


def _hash_embed(text: str) -> np.ndarray:
    """Deterministic char-3-gram hashing embedding, L2-normalized."""
    counts = np.zeros(_HASH_BUCKETS)
    s = f"  {text.lower()}  "
    for i in range(len(s) - 2):
        gram = s[i:i + 3]
        h = 0
        for ch in gram:
            h = (h * 1000003 + ord(ch)) % (2**31)
        counts[h % _HASH_BUCKETS] += -1.0 if (h >> 16) & 1 else 1.0
    v = counts @ _projection(_HASH_BUCKETS)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _st_embed(text: str) -> np.ndarray:
    """MiniLM embedding projected to D_SEM, L2-normalized (explicit opt-in)."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(ST_MODEL_NAME)
    e = np.asarray(_ST_MODEL.encode([text])[0], dtype=float)
    v = e @ _projection(e.size)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def embed_text(text: str, use_sentence_transformers: bool | None = None) -> np.ndarray:
    """Semantic channel for one step; hash embedding unless ST is EXPLICITLY
    requested with use_sentence_transformers=True (None and False both mean
    hash — installing the package must not change results)."""
    return _st_embed(text) if use_sentence_transformers else _hash_embed(text)


#: Value written into the surprisal dims when a step carries no logprobs.
#: Surprisal is -log p >= 0 for any real token, so a negative marker cannot
#: collide with an observed value. A positive floor such as 0.05 does collide,
#: and doubles as the clamp on genuine high-confidence steps, which makes
#: "missing" and "very confident" indistinguishable.
MISSING_SURPRISAL = -1.0


def logprobs_missing(step: dict) -> bool:
    """True when a step records no usable sampled-token logprobs.

    An explicit ``logprobs_available: false`` is honoured; otherwise absence or
    emptiness of ``token_logprobs`` means missing.
    """
    flag = step.get("logprobs_available")
    if flag is not None and not bool(flag):
        return True
    lps = step.get("token_logprobs")
    return lps is None or len(lps) == 0


def uncertainty_features(token_logprobs: Optional[Iterable[float]]) -> np.ndarray:
    """[mean, max, slope, high-fraction] of per-token surprisal (-logprob).

    Missing logprobs produce ``MISSING_SURPRISAL`` in the two level dims rather
    than a plausible small value, so a monitor can tell "no measurement" from
    "measured, very confident".  Observed values are no longer floored.
    """
    if token_logprobs is None:
        return np.array([MISSING_SURPRISAL, MISSING_SURPRISAL, 0.0, 0.0])
    s = np.asarray([-float(lp) for lp in token_logprobs], dtype=float)
    if s.size == 0:
        return np.array([MISSING_SURPRISAL, MISSING_SURPRISAL, 0.0, 0.0])
    if not np.all(np.isfinite(s)):
        raise ValueError("token_logprobs contains a non-finite value")
    mean, mx = float(np.mean(s)), float(np.max(s))
    if s.size >= 2:
        t = np.arange(s.size, dtype=float)
        slope = float(np.polyfit(t, s, 1)[0])
    else:
        slope = 0.0
    frac = float(np.mean(s > _HIGH_SURPRISAL_NATS))
    return np.array([mean, mx, slope, frac])


def step_signal(step: dict, use_sentence_transformers: bool | None = None
                ) -> np.ndarray:
    """One trace step dict -> the 43-dim x_t (see module docstring)."""
    x = np.zeros(D_TOTAL)
    x[:D_SEM] = embed_text(str(step.get("text", "")),
                           use_sentence_transformers)
    x[D_SEM:D_SEM + 4] = uncertainty_features(
        None if logprobs_missing(step) else step.get("token_logprobs"))
    action = _action_of(step)
    x[D_SEM + 4 + ACTION_TYPES.index(action)] = 1.0
    latency = _latency_of(step)
    n_tokens = int(step.get("output_tokens",
                            len(step.get("token_logprobs") or []) or 1))
    x[D_SEM + 4 + 4] = math.log(latency)
    x[D_SEM + 4 + 5] = math.log(max(n_tokens, 1))
    x[D_SEM + 4 + 6] = 1.0 if step.get("error") else 0.0
    return x


class ExtFeatureState:
    """Causal per-episode state for the derived x_t channel (telemetry v3).

    Create one per episode and pass it to every step_signal_ext call in step
    order. Everything it holds depends only on steps already seen.
    """

    def __init__(self) -> None:
        self.prev_emb: np.ndarray | None = None   # e_{t-1}
        self.anchor_emb: np.ndarray | None = None  # the task anchor
        self.emb_sum: np.ndarray | None = None     # sum of e_0..e_{t-1}
        self.n_steps = 0
        self.seen_calls: set[tuple[str, str]] = set()
        self.cum_tokens = 0.0


def task_anchor_text(step: dict, reasoning: str) -> str:
    """Text the task-similarity anchor is measured against.

    The original user task if the trace carries one.  Otherwise the first
    step's *reasoning* - never its tool results: anchoring on a result would
    compare a first-step wrong-document against itself and score it perfectly
    relevant.
    """
    task = step.get("task")
    if isinstance(task, str) and task.strip():
        return task
    return reasoning


def step_signal_ext(step: dict, state: ExtFeatureState,
                    use_sentence_transformers: bool | None = None
                    ) -> np.ndarray:
    """One step dict -> the 51-dim extended x_t (v3). Mutates `state`."""
    x = np.zeros(D_TOTAL_EXT)
    x[:D_TOTAL] = step_signal(step, use_sentence_transformers)
    e = x[:D_SEM].copy()  # unit-norm (see embed_text), so cos == dot

    tools, reasoning = parse_step_events(step)
    n_tools = len(tools)
    n_err = sum(1 for ev in tools if ev.is_error)
    keys = [(ev.name, ev.args_key) for ev in tools]
    n_retry = sum(1 for k in keys if k in state.seen_calls)
    state.seen_calls.update(keys)

    latency = _latency_of(step)
    n_tokens = int(step.get("output_tokens",
                            len(step.get("token_logprobs") or []) or 1))
    # `text` includes the rendered tool bits, whose tokens are already counted
    # in `output_tokens` for a step the model produced, so this over-counts a
    # tool-heavy step. It feeds only IDX_CTX_RATIO, a monotone utilisation
    # proxy clamped at 2.0, not a billing figure.
    state.cum_tokens += float(n_tokens) + len(str(step.get("text", ""))) / 4.0

    x[IDX_COS_DRIFT] = (0.0 if state.prev_emb is None
                        else 1.0 - float(e @ state.prev_emb))
    if state.anchor_emb is None:
        state.anchor_emb = embed_text(task_anchor_text(step, reasoning),
                                      use_sentence_transformers)
    x[IDX_TASK_SIM] = float(e @ state.anchor_emb)
    x[IDX_TOOL_SUCCESS] = 1.0 - (n_err / n_tools if n_tools else 0.0)
    x[IDX_RETRY_COUNT] = float(n_retry)
    # Measured tool time when the collector recorded it; otherwise the old
    # approximation from model-step time, which is all a legacy trace has
    #.
    measured = [ev.latency_s for ev in tools if ev.latency_s is not None]
    if measured:
        x[IDX_TOOL_LATENCY] = math.log(max(sum(measured) / len(measured), 1e-3))
    elif n_tools:
        x[IDX_TOOL_LATENCY] = math.log(latency / n_tools)
    x[IDX_CTX_RATIO] = min(state.cum_tokens / CTX_BUDGET_TOKENS, 2.0)
    x[IDX_REASON_DEPTH] = float(n_tools)
    if state.n_steps == 0 or state.emb_sum is None:
        x[IDX_SELF_CONSISTENCY] = 1.0
    else:
        norm = float(np.linalg.norm(state.emb_sum))
        x[IDX_SELF_CONSISTENCY] = (float(e @ (state.emb_sum / norm))
                                   if norm > 0 else 1.0)

    state.prev_emb = e
    state.emb_sum = e.copy() if state.emb_sum is None else state.emb_sum + e
    state.n_steps += 1
    return x


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else 1.0


def _json_broken(result: str, truncated: bool = True) -> bool:
    """True iff a JSON-looking result is NOT a valid JSON prefix.

    A stored tool result may be truncated, so a cut-off result must never
    count as corruption - a plain JSON-validity check false-positives on
    exactly that. A result is
    flagged only when it can NOT be completed to valid JSON: the scan finds a
    mismatched closer, or closing the open string/brackets (with trailing
    ','/':' repaired) still fails to parse — i.e. the breakage is structural,
    not a truncation cut.

    With structured telemetry the collector states whether the stored result
    was truncated; a complete result that does not parse is simply broken, so
    the prefix-completion allowance no longer hides genuinely malformed
    payloads.
    """
    s = result.strip()
    if not truncated:
        if not s or s[0] not in "{[":
            return False
        try:
            json.loads(s)
            return False
        except json.JSONDecodeError:
            return True
    if not s or s[0] not in "{[":
        return False
    try:
        json.loads(s)
        return False
    except json.JSONDecodeError:
        pass
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or stack[-1] != {"}": "{", "]": "["}[ch]:
                return True                      # mismatched closer
            stack.pop()
    t = (s + ('"' if in_str else "")).rstrip()
    if t.endswith(":"):
        t += " null"
    elif t.endswith(","):
        t = t[:-1]
    t += "".join("}" if c == "{" else "]" for c in reversed(stack))
    try:
        json.loads(t)
        return False                             # completable: truncation
    except json.JSONDecodeError:
        return True                              # structural breakage


def _char_stats(text: str) -> tuple[float, float]:
    """(alnum ratio, junk ratio) of a result string; junk = non-printable
    or replacement characters."""
    if not text:
        return 1.0, 0.0
    n = len(text)
    alnum = sum(ch.isalnum() or ch.isspace() for ch in text) / n
    junk = sum(1 for ch in text
               if ch == "�" or (not ch.isprintable()
                                     and ch not in "\n\t\r")) / n
    return alnum, junk


#: Function words dropped before the overlap test. English, because the tasks
#: in this project are English; supply another set through `_lex_miss(stop=)`
#: for another language.
#:
#: It matters less than it looks. The test is an INTERSECTION of two sets
#: produced by the same filter, so a stoplist that misses a language's function
#: words makes both sides larger and the overlap MORE likely — the dim
#: under-fires rather than false-alarms. The language-neutral `len(w) > 2` rule
#: below already removes most short function words in any language, which is
#: why a French decoy is still flagged correctly with an English stoplist.
_LEX_STOP = frozenset(
    "a an the of for in on with to and or is are was be this that how "
    "what".split())

#: Word characters in ANY script, minus underscore. `[a-z0-9]+` matched ASCII
#: letters only, which made this dim structurally inert outside the Latin
#: alphabet: a Cyrillic or Greek result tokenized to nothing, fell under
#: `min_words`, and scored 0.0 — reported as "relevant" because it could not be
#: read at all. Measured on the committed corpora: this changes the verdict on
#: 0 of 30,070 tool events, so it is a capability fix, not a result change.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: Document-like results the lexical reader could not tokenize into enough
#: words to judge. Non-zero means this dim is inert on that traffic — see
#: `lex_unreadable`.
LEX_UNREADABLE = 0


def lex_unreadable() -> int:
    """How many document-like results the lexical relevance dim could not read.

    A space-free script (Chinese, Japanese, Thai) tokenizes to one run per
    phrase, so it never reaches `min_words` and the dim silently returns 0.0.
    Segmenting it needs a per-language model, which this project deliberately
    does not carry; counting the cases is what keeps the gap visible instead of
    letting a permanently-quiet feature read as a permanently-clean one.
    """
    return LEX_UNREADABLE


def _content_words(text: str, stop: frozenset[str] = _LEX_STOP) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower())
            if w not in stop and len(w) > 2]


def _lex_miss(task_text: str, args: str, result: str,
              min_words: int = 4,
              stop: frozenset[str] = _LEX_STOP) -> float:
    """Binary lexical retrieval-relevance miss of one DOCUMENT result.

    1.0 iff a document-like result (>= min_words content words; not an
    error string, not a parenthesized tool diagnostic, not structured
    JSON) shares ZERO content words with BOTH its own query args and the
    task text. Pure string operations — the lightweight answer to
    wrong_document: char-trigram cosine has a hopelessly noisy healthy
    null, and even continuous term coverage leaves healthy tangential
    documents (~6% coverage) a hair from decoys (0%), which per-dim
    z-normalization cannot hold apart. The zero-overlap INDICATOR
    separates almost perfectly (measured: 28/28 wrong_document episodes
    flagged, 3/330 healthy across five datasets) — an on-topic result
    essentially always repeats at least one query or task content word.
    """
    r = result.strip()
    if not r or error_shaped(r) or r.startswith("("):
        return 0.0
    if r[0] in "{[":
        return 0.0      # structured data, not a document (json/char dims' job)
    rw = _content_words(r, stop)
    if len(rw) < min_words:
        # Long enough to be a document, but it did not tokenize into enough
        # words to judge — a space-free script, most likely. Counted so the
        # dim's silence on that traffic is observable rather than assumed to
        # mean "relevant"; see `lex_unreadable`.
        if len(r.split()) < min_words and len(r) >= 4 * min_words:
            global LEX_UNREADABLE
            LEX_UNREADABLE += 1
        return 0.0
    rset = set(rw)
    for src in (args, task_text):
        if set(_content_words(src, stop)) & rset:
            return 0.0
    return 1.0


class GrdFeatureState:
    """Causal per-episode state for the content-grounding channel g_t."""

    def __init__(self) -> None:
        self.task_text: str | None = None              # step-0 text
        self.result_emb_sum: np.ndarray | None = None  # sum of past step
        self.n_result_steps = 0                        # results' embeddings
        self.alnum_sum = 0.0
        self.n_alnum = 0
        self.prev_result_emb: np.ndarray | None = None  # last step's results
        self.recent_result_embs: list[np.ndarray] = []  # last 3 steps'
        self.drift_ewma = 0.0                           # EWMA of self_dis


def step_signal_grd(step: dict, ext_state: ExtFeatureState,
                    grd_state: GrdFeatureState,
                    use_sentence_transformers: bool | None = None
                    ) -> np.ndarray:
    """One step dict -> the 60-dim grounded x_t (v4). Mutates both states.

    All nine g dims are 0 when the step has no recorded tool results (v1
    traces, pure-reasoning steps) — inert, never an error. Higher = more
    anomalous throughout.
    """
    x = np.zeros(D_TOTAL_GRD)
    x[:D_TOTAL_EXT] = step_signal_ext(step, ext_state,
                                      use_sentence_transformers)
    events, reasoning = parse_step_events(step)
    if grd_state.task_text is None:
        # The task anchor is the task, never this step's tool results - a
        # first-step wrong document must not be compared against itself
        #.
        grd_state.task_text = task_anchor_text(step, reasoning)

    res_events = [ev for ev in events if ev.has_result and ev.result]
    tools = [(ev.name, ev.args_key, ev.result) for ev in res_events]
    if not tools:
        return x

    # 59: lexical retrieval-relevance miss (wrong_document; string ops only)
    x[IDX_GRD_LEX_MISS] = max(
        _lex_miss(grd_state.task_text, args, res) for _, args, res in tools)

    results_text = " ".join(res for _, _, res in tools)
    res_emb = embed_text(results_text, use_sentence_transformers)

    # 51: does each result look like an answer to its own query?
    dis = [1.0 - _cos(embed_text(res, use_sentence_transformers),
                      embed_text(f"{name} {args}", use_sentence_transformers))
           for name, args, res in tools]
    x[IDX_GRD_QUERY_DIS] = float(np.mean(dis))

    # 52: does the agent's own reasoning text relate to what the tools said?
    if reasoning:
        x[IDX_GRD_REASON_DIS] = 1.0 - _cos(
            res_emb, embed_text(reasoning, use_sentence_transformers))

    # 53: are this step's results consistent with the episode's past results?
    if grd_state.n_result_steps > 0 and grd_state.result_emb_sum is not None:
        x[IDX_GRD_SELF_DIS] = 1.0 - _cos(
            res_emb, grd_state.result_emb_sum / grd_state.n_result_steps)

    # 54: structural breakage of JSON-looking results (truncation-safe).
    x[IDX_GRD_JSON_BROKEN] = float(np.mean(
        [_json_broken(ev.result,
                      truncated=ev.truncated or ev.source == "text")
         for ev in res_events]))

    # 55: character-statistics anomaly of the result text.
    alnum, junk = _char_stats(results_text)
    if grd_state.n_alnum > 0:
        shift = abs(alnum - grd_state.alnum_sum / grd_state.n_alnum)
    else:
        shift = 0.0
    x[IDX_GRD_CHAR_ANOM] = shift + junk

    # 56: dissimilarity to the PREVIOUS step's results (sudden corruption).
    if grd_state.prev_result_emb is not None:
        x[IDX_GRD_CONSEC_DIS] = 1.0 - _cos(res_emb,
                                           grd_state.prev_result_emb)

    # 57: EWMA of the centroid dissimilarity — persistent grounding drift.
    grd_state.drift_ewma = (0.4 * x[IDX_GRD_SELF_DIS]
                            + 0.6 * grd_state.drift_ewma)
    x[IDX_GRD_DRIFT] = grd_state.drift_ewma

    # 58: reasoning vs the last-3-steps result memory (windowed, not
    # whole-episode centroid).
    if reasoning and grd_state.recent_result_embs:
        mem = np.mean(grd_state.recent_result_embs, axis=0)
        x[IDX_GRD_MEM_DIS] = 1.0 - _cos(
            embed_text(reasoning, use_sentence_transformers), mem)

    grd_state.result_emb_sum = (res_emb.copy()
                                if grd_state.result_emb_sum is None
                                else grd_state.result_emb_sum + res_emb)
    grd_state.n_result_steps += 1
    grd_state.alnum_sum += alnum
    grd_state.n_alnum += 1
    grd_state.prev_result_emb = res_emb
    grd_state.recent_result_embs = (grd_state.recent_result_embs
                                    + [res_emb])[-3:]
    return x


class TraceSchemaError(ValueError):
    """A trace step is missing a field or holds a value features cannot use."""


def validate_steps(steps: object, context: str = "<trace>") -> list[dict]:
    """Coerce and check one run's step records.

    Fields are coerced where a sane coercion exists (a numeric string token
    count, a missing text) and rejected where silence would corrupt features:
    non-finite latency or logprobs, negative counts, a non-list trace, or an
    empty run - each of which otherwise reaches the feature vectors as NaN, a
    TypeError deep in numpy, or a malformed one-dimensional array.
    """
    if not isinstance(steps, list):
        raise TraceSchemaError(f"{context}: trace must be a list of steps, "
                               f"got {type(steps).__name__}")
    if not steps:
        raise TraceSchemaError(f"{context}: trace has no steps")

    out: list[dict] = []
    for i, raw in enumerate(steps):
        where = f"{context}[step {i}]"
        if not isinstance(raw, dict):
            raise TraceSchemaError(f"{where}: step must be an object, "
                                   f"got {type(raw).__name__}")
        step = dict(raw)
        step["text"] = "" if step.get("text") is None else str(step["text"])

        lps = step.get("token_logprobs")
        if lps is not None:
            if not isinstance(lps, (list, tuple)):
                raise TraceSchemaError(f"{where}: token_logprobs must be a list")
            try:
                values = [float(v) for v in lps]
            except (TypeError, ValueError) as exc:
                raise TraceSchemaError(
                    f"{where}: token_logprobs holds a non-numeric value") from exc
            if any(not math.isfinite(v) for v in values):
                raise TraceSchemaError(f"{where}: token_logprobs holds a "
                                       f"non-finite value")
            step["token_logprobs"] = values

        latency = step.get("latency_s", 1.0)
        try:
            latency = float(1.0 if latency is None else latency)
        except (TypeError, ValueError) as exc:
            raise TraceSchemaError(f"{where}: latency_s is not numeric") from exc
        if not math.isfinite(latency) or latency < 0.0:
            raise TraceSchemaError(f"{where}: latency_s must be finite and "
                                   f"non-negative, got {latency!r}")
        step["latency_s"] = latency

        tokens = step.get("output_tokens")
        if tokens is not None:
            try:
                tokens = int(tokens)
            except (TypeError, ValueError) as exc:
                raise TraceSchemaError(
                    f"{where}: output_tokens is not an integer") from exc
            if tokens < 0:
                raise TraceSchemaError(f"{where}: output_tokens is negative")
            step["output_tokens"] = tokens

        step["error"] = bool(step.get("error", False))

        events = step.get("tool_events")
        if events is not None:
            if not isinstance(events, list):
                raise TraceSchemaError(f"{where}: tool_events must be a list")
            for k, ev in enumerate(events):
                if not isinstance(ev, dict):
                    raise TraceSchemaError(f"{where}: tool_events[{k}] must be "
                                           f"an object")
                if not str(ev.get("name", "")).strip():
                    raise TraceSchemaError(f"{where}: tool_events[{k}] has no name")
                lat = ev.get("latency_s")
                if lat is not None and not math.isfinite(float(lat)):
                    raise TraceSchemaError(f"{where}: tool_events[{k}] has a "
                                           f"non-finite latency")
        out.append(step)
    return out


def episode_from_trace(steps: list[dict], episode_id: str,
                       tau: int | None = None,
                       failure_class: str | None = None,
                       severity: float | None = None,
                       use_sentence_transformers: bool | None = None,
                       extended: bool = False,
                       grounding: bool = False) -> Episode:
    """Convert one run's step list into an Episode.

    Omit tau/failure_class for unlabeled (healthy-assumed or deployment)
    runs; pass them when you have ground truth for evaluation. With
    extended=True the Episode is (T, 51) including the derived x_t channel;
    grounding=True (implies extended) makes it (T, 60) with the content-
    grounding channel g_t appended (telemetry v4, DESIGN.md amendment 3).
    """
    steps = validate_steps(steps, context=episode_id)
    if grounding:
        ext_state, grd_state = ExtFeatureState(), GrdFeatureState()
        X = np.array([step_signal_grd(s, ext_state, grd_state,
                                      use_sentence_transformers)
                      for s in steps])
    elif extended:
        state = ExtFeatureState()
        X = np.array([step_signal_ext(s, state, use_sentence_transformers)
                      for s in steps])
    else:
        X = np.array([step_signal(s, use_sentence_transformers)
                      for s in steps])
    is_healthy = tau is None
    return Episode(X=X, episode_id=episode_id, is_healthy=is_healthy,
                   failure_class=failure_class, tau=tau,
                   t_fail=None if is_healthy else len(steps) - 1,
                   severity=severity)


def load_trace_jsonl(path: str | Path, episode_id: str | None = None,
                     **kwargs) -> Episode:
    """Read one JSONL trace file (one step object per line) into an Episode."""
    p = Path(path)
    steps = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TraceSchemaError(f"{p}:{lineno}: {exc}") from exc
    steps = validate_steps(steps, context=str(p))
    return episode_from_trace(steps, episode_id or p.stem, **kwargs)


if __name__ == "__main__":
    # Smoke test: fabricate a small trace and round-trip it.
    steps = []
    rng = rng_for(0, "adapter-smoke")
    for t in range(12):
        steps.append({
            "text": f"step {t}: querying the flights API for option {t % 3}",
            "token_logprobs": (-rng.exponential(0.8, size=30)).tolist(),
            "action": ["thought", "function_call", "observation",
                       "answer"][t % 4],
            "latency_s": float(rng.lognormal(0.0, 0.5)),
            "error": bool(t == 7),
        })
    ep = episode_from_trace(steps, "trace-smoke",
                            use_sentence_transformers=False)
    assert ep.X.shape == (12, D_TOTAL) and ep.is_healthy
    assert np.all(np.isfinite(ep.X))
    norms = np.linalg.norm(ep.X[:, :D_SEM], axis=1)
    assert np.allclose(norms, 1.0), "semantic channel not unit-norm"
    onehots = ep.X[:, D_SEM + 4:D_SEM + 8]
    assert np.all(onehots.sum(axis=1) == 1.0), "bad action one-hot"
    assert ep.X[7, -1] == 1.0 and ep.X[6, -1] == 0.0, "error flag misplaced"
    ep2 = episode_from_trace(steps, "trace-smoke",
                             use_sentence_transformers=False)
    assert np.array_equal(ep.X, ep2.X), "adapter not deterministic"
    labeled = episode_from_trace(steps, "trace-fail", tau=6,
                                 failure_class="tool_cascade", severity=0.5,
                                 use_sentence_transformers=False)
    assert not labeled.is_healthy and labeled.t_fail == 11

    # --- extended (v3) channel smoke ---
    bit = '[lookup_flight({"destination": "Osaka", "origin": "Bergen"}) -> $370]'
    err_bit = '[lookup_hotel({"city": "Osaka"}) -> Error: service unavailable]'
    ext_steps = [
        {"text": "planning the trip " + bit, "token_logprobs": [-0.1] * 20,
         "action": "tool_call", "latency_s": 2.0, "output_tokens": 20},
        {"text": "retrying " + bit + " " + err_bit,
         "token_logprobs": [-0.1] * 20, "action": "tool_call",
         "latency_s": 4.0, "output_tokens": 20},
        {"text": "totally different topic now: quantum llamas",
         "token_logprobs": [-0.1] * 20, "action": "synthesis",
         "latency_s": 1.0, "output_tokens": 20},
    ]
    epx = episode_from_trace(ext_steps, "trace-ext",
                             use_sentence_transformers=False, extended=True)
    assert epx.X.shape == (3, D_TOTAL_EXT) and np.all(np.isfinite(epx.X))
    assert epx.X[0, IDX_COS_DRIFT] == 0.0
    # with no task recorded the anchor is step 0's REASONING, never
    # its tool results, so step 0 is no longer trivially similar to itself.
    assert epx.X[0, IDX_TASK_SIM] < 1.0
    anchor_only = episode_from_trace(
        [{**ext_steps[0], "text": "planning the trip"}],
        "trace-anchor", use_sentence_transformers=False, extended=True)
    assert abs(anchor_only.X[0, IDX_TASK_SIM] - 1.0) < 1e-9, \
        "a step with no tool bits must anchor on its own text"
    with_task = episode_from_trace(
        [{**s, "task": "plan a trip from Bergen to Osaka"} for s in ext_steps],
        "trace-task", use_sentence_transformers=False, extended=True)
    assert abs(with_task.X[0, IDX_TASK_SIM]
               - float(with_task.X[0, :D_SEM]
                       @ embed_text("plan a trip from Bergen to Osaka", False))) < 1e-9, \
        "an explicit task must be the anchor"
    assert epx.X[0, IDX_SELF_CONSISTENCY] == 1.0
    assert epx.X[0, IDX_REASON_DEPTH] == 1.0 and epx.X[0, IDX_RETRY_COUNT] == 0.0
    assert epx.X[1, IDX_RETRY_COUNT] == 1.0, "repeat call not counted as retry"
    assert epx.X[1, IDX_REASON_DEPTH] == 2.0
    assert abs(epx.X[1, IDX_TOOL_SUCCESS] - 0.5) < 1e-9, "error rate wrong"
    assert epx.X[2, IDX_COS_DRIFT] > epx.X[1, IDX_COS_DRIFT], \
        "off-topic step should have higher cosine drift"

    # Task similarity at realistic step lengths. The three-gram hashing
    # embedding needs a sentence or two to be meaningful - on toy fixtures of
    # five words it is noise, which is why this check uses real-length text.
    # (Measured here: on-topic 0.72, partly-related 0.50, off-topic 0.11,
    # against a random-text null of ~0.13.)
    _task = ("Find the two most recent arXiv papers about echo state networks "
             "for anomaly detection, read their summaries, then search "
             "Wikipedia for Echo State Network and explain how reservoirs are "
             "trained.")
    _on = ("I searched arXiv for echo state network anomaly detection papers "
           "and found two recent works on reservoir computing for time-series "
           "anomaly detection; both train only the linear readout.")
    _off = ("Sourdough bread relies on a wild-yeast starter fermented over "
            "several days; bakers feed it flour and water to keep the culture "
            "active before shaping the loaf.")
    _sim_steps = [{"text": t, "token_logprobs": [-0.1] * 20, "task": _task,
                   "action": "synthesis", "latency_s": 1.0,
                   "output_tokens": 20} for t in (_on, _off)]
    _sim = episode_from_trace(_sim_steps, "trace-tasksim",
                              use_sentence_transformers=False, extended=True)
    assert _sim.X[1, IDX_TASK_SIM] < _sim.X[0, IDX_TASK_SIM] - 0.3, \
        f"off-topic step not separated from the task anchor: {_sim.X[:, IDX_TASK_SIM]}"
    assert np.all(np.diff(epx.X[:, IDX_CTX_RATIO]) > 0), "ctx ratio not increasing"
    epx2 = episode_from_trace(ext_steps, "trace-ext",
                              use_sentence_transformers=False, extended=True)
    assert np.array_equal(epx.X, epx2.X), "extended adapter not deterministic"
    base = np.array([step_signal(s, False) for s in ext_steps])
    assert np.array_equal(epx.X[:, :D_TOTAL], base), \
        "extended first 43 dims must equal step_signal"

    # v1 tool-bit format (no "-> result") must still count calls/retries
    v1 = '[lookup_flight({"destination": "Osaka", "origin": "Bergen"})]'
    v1_steps = [
        {"text": "calling " + v1, "token_logprobs": [-0.1] * 10,
         "action": "tool_call", "latency_s": 1.0, "output_tokens": 10},
        {"text": "again " + v1, "token_logprobs": [-0.1] * 10,
         "action": "tool_call", "latency_s": 1.0, "output_tokens": 10},
    ]
    epv1 = episode_from_trace(v1_steps, "trace-v1",
                              use_sentence_transformers=False, extended=True)
    assert epv1.X[0, IDX_REASON_DEPTH] == 1.0, "v1 bit not counted"
    assert epv1.X[1, IDX_RETRY_COUNT] == 1.0, "v1 retry not counted"
    assert epv1.X[0, IDX_TOOL_SUCCESS] == 1.0, "v1 success rate must be inert"

    # --- grounding (v4) channel smoke ---
    good = '[db_query({"sql": "SELECT price FROM rooms"}) -> {"price": 215, "city": "Osaka"}]'
    good2 = '[db_query({"sql": "SELECT price FROM rooms"}) -> {"price": 220, "city": "Osaka"}]'
    arr = '[vector_search({"q": "hotels"}) -> ["osaka grand", "namba inn"]]'
    trunc = '[db_query({"sql": "SELECT * FROM rooms"}) -> {"rows": [{"price": 215, "city": "Osa]'
    broken = '[db_query({"sql": "SELECT price"}) -> {"price": 215,, "city" "Osaka"}]'
    garbage = '[web_get({"url": "http://x"}) -> ■■��� 9$$#@@!!��■■�� 0x00FF]'

    def _gstep(text: str) -> dict:
        return {"text": "checking hotel prices " + text,
                "token_logprobs": [-0.1] * 15, "action": "tool_call",
                "latency_s": 1.0, "output_tokens": 15}

    g_steps = [_gstep(good), _gstep(good2), _gstep(arr), _gstep(trunc),
               _gstep(broken), _gstep(garbage)]
    epg = episode_from_trace(g_steps, "trace-grd",
                             use_sentence_transformers=False, grounding=True)
    assert epg.X.shape == (6, D_TOTAL_GRD) and np.all(np.isfinite(epg.X))
    assert np.array_equal(
        epg.X[:, :D_TOTAL_EXT],
        episode_from_trace(g_steps, "trace-grd",
                           use_sentence_transformers=False,
                           extended=True).X), \
        "grounding must not change the first 51 dims"
    # archive-bug regressions: valid JSON object/array and TRUNCATED JSON
    # must not be flagged; genuinely broken JSON must be.
    assert epg.X[0, IDX_GRD_JSON_BROKEN] == 0.0, "valid JSON flagged"
    assert epg.X[2, IDX_GRD_JSON_BROKEN] == 0.0, "valid JSON array flagged"
    assert epg.X[3, IDX_GRD_JSON_BROKEN] == 0.0, "truncated JSON flagged"
    assert epg.X[4, IDX_GRD_JSON_BROKEN] == 1.0, "broken JSON not flagged"
    # garbled result: character anomaly and self-inconsistency both spike
    assert epg.X[5, IDX_GRD_CHAR_ANOM] > 0.3, "garbage chars not caught"
    assert epg.X[5, IDX_GRD_SELF_DIS] > epg.X[1, IDX_GRD_SELF_DIS], \
        "garbage result should be less consistent with past results"
    assert epg.X[0, IDX_GRD_SELF_DIS] == 0.0, "first result step must be 0"
    # lexical relevance miss: an off-topic document result must spike dim
    # 59; on-topic results, terse results, and error strings must not.
    decoy = ('[web_search({"q": "hotel prices osaka"}) -> Sourdough bread '
             'relies on a wild-yeast starter fermented over several days.]')
    ontopic = ('[web_search({"q": "hotel prices osaka"}) -> Osaka hotel '
               'prices average 215 per night in the city center district.]')
    errres = '[web_search({"q": "hotel prices osaka"}) -> Error: timeout while contacting service]'
    lex_steps = [_gstep(ontopic), _gstep(decoy), _gstep(errres), _gstep(good)]
    epl = episode_from_trace(lex_steps, "trace-lex",
                             use_sentence_transformers=False, grounding=True)
    assert epl.X[1, IDX_GRD_LEX_MISS] == 1.0, "off-topic decoy not flagged"
    assert epl.X[0, IDX_GRD_LEX_MISS] == 0.0, "on-topic doc penalized"
    assert epl.X[2, IDX_GRD_LEX_MISS] == 0.0, "error string penalized"
    assert epl.X[3, IDX_GRD_LEX_MISS] == 0.0, "terse JSON result penalized"

    # v1 traces (no results): every g dim exactly 0 (inert, per contract)
    epg_v1 = episode_from_trace(v1_steps, "trace-grd-v1",
                                use_sentence_transformers=False,
                                grounding=True)
    assert np.all(epg_v1.X[:, D_TOTAL_EXT:] == 0.0), "v1 g dims must be inert"
    # determinism
    epg2 = episode_from_trace(g_steps, "trace-grd",
                              use_sentence_transformers=False, grounding=True)
    assert np.array_equal(epg.X, epg2.X), "grounding adapter not deterministic"

    print("PASS adapter smoke test (incl. extended v3 + grounding v4) | "
          "hash-embed backend (sentence-transformers only on explicit opt-in)")
