"""WS0.2 + WS0.3 — cost metering and record/replay ("cassettes").

Two provider-agnostic primitives every real-agent cell (WS1-WS5) is built
on. Both are plain-Python, JSON-in/JSON-out, no third-party deps, so they
work identically for Gemini calls and for real tool calls (Tavily, weather,
GitHub, SQL, ...).

CostMeter (WS0.2)
    Tracks $ spent from token usage against the confirmed Gemini list
    prices, and raises BudgetExceeded before a call that would blow a cap.
    Keeps the whole WS0-WS5 build under a hard dollar ceiling (target: $150,
    all-Flash).

Cassette (WS0.3)
    Records (request -> response) keyed by a canonical hash of the request,
    and replays the stored response instead of calling the live API/tool.
    In "auto" mode a recorded call is free forever after, so iterating on
    collection logic during development costs nothing after the first pass.

Neither touches the simulator or the monitor; this is collection-time infra.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ------------------------------------------------------------------ pricing
# Confirmed 2026-07 list prices, USD per 1M tokens (text tier). Pro is
# context-length tiered at 200k prompt tokens; Flash / Flash-Lite are flat.
# Source: ai.google.dev/gemini-api/docs/pricing.
_PRO_THRESHOLD = 200_000

GEMINI_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash":       {"in": 0.30, "out": 2.50, "cached_in": 0.03},
    "gemini-2.5-flash-lite":  {"in": 0.10, "out": 0.40, "cached_in": 0.01},
    # Pro: (<=200k, >200k) pairs — resolved per-call by prompt size.
    "gemini-2.5-pro":         {"in": 1.25, "in_hi": 2.50,
                               "out": 10.0, "out_hi": 15.0,
                               "cached_in": 0.125, "cached_in_hi": 0.25},
}


def price_call(model: str, in_tok: int, out_tok: int,
               cached_in_tok: int = 0) -> float:
    """USD for one call. cached_in_tok is billed at the cache-read rate and
    is assumed to be a subset already counted in in_tok."""
    p = GEMINI_PRICING.get(model)
    if p is None:
        raise KeyError(f"no pricing for model {model!r}; known: "
                       f"{sorted(GEMINI_PRICING)}")
    if in_tok < 0 or out_tok < 0 or cached_in_tok < 0:
        raise ValueError(f"token counts must be non-negative: "
                         f"in={in_tok}, out={out_tok}, cached={cached_in_tok}")
    if cached_in_tok > in_tok:
        raise ValueError(f"cached input ({cached_in_tok}) exceeds input ({in_tok})")
    # The Pro long-context tier is a *prompt*-size tier; adding output tokens
    # overpriced calls whose prompt was under the threshold.
    hi = "_hi" if in_tok > _PRO_THRESHOLD and "in_hi" in p else ""
    fresh_in = max(in_tok - cached_in_tok, 0)
    return (fresh_in * p[f"in{hi}"]
            + cached_in_tok * p[f"cached_in{hi}"]
            + out_tok * p[f"out{hi}"]) / 1_000_000.0


class BudgetExceeded(RuntimeError):
    """Raised when a charge would push cumulative spend past the cap."""


@dataclass
class CostMeter:
    """Cumulative spend tracker with an optional hard budget cap (USD)."""

    budget_usd: float | None = None
    spent_usd: float = 0.0
    n_calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, model: str, est_in_tok: int, est_out_tok: int) -> float:
        """Check a CONSERVATIVE estimate against the cap BEFORE the call is made.

        `charge` raises only after the request has already been billed by the
        provider; a genuine pre-spend guard has to run before the API call. `reserve` validates the model's pricing up front (a KeyError
        for an unknown model is raised, not swallowed, so an unpriced model can
        never run un-metered) and raises BudgetExceeded if the estimated cost
        would exceed the cap. It does NOT commit spend; call `charge` with the
        actual token counts afterwards to reconcile.
        """
        cost = price_call(model, int(est_in_tok), int(est_out_tok))
        with self._lock:
            if (self.budget_usd is not None
                    and self.spent_usd + cost > self.budget_usd):
                raise BudgetExceeded(
                    f"estimated call cost ${cost:.4f} would push cumulative "
                    f"${self.spent_usd + cost:.2f} over cap "
                    f"${self.budget_usd:.2f} — refusing BEFORE the request.")
        return cost

    def charge(self, model: str, in_tok: int, out_tok: int,
               cached_in_tok: int = 0) -> float:
        """Record one call's cost; raise BEFORE overspending the cap."""
        cost = price_call(model, in_tok, out_tok, cached_in_tok)
        with self._lock:
            if (self.budget_usd is not None
                    and self.spent_usd + cost > self.budget_usd):
                raise BudgetExceeded(
                    f"call would cost ${cost:.4f}, cumulative "
                    f"${self.spent_usd + cost:.2f} > cap ${self.budget_usd:.2f}"
                    f" — stopping before spending. Raise budget_usd or use "
                    f"a cassette in replay mode.")
            self.spent_usd += cost
            self.n_calls += 1
            self.in_tok += int(in_tok)
            self.out_tok += int(out_tok)
        return cost

    def summary(self) -> str:
        cap = f" / ${self.budget_usd:.2f} cap" if self.budget_usd else ""
        return (f"${self.spent_usd:.4f} spent{cap} over {self.n_calls} calls "
                f"({self.in_tok:,} in + {self.out_tok:,} out tokens)")


# ----------------------------------------------------------------- cassette
def _canonical(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def request_key(*parts: Any, namespace: str | None = None) -> str:
    """Deterministic cassette key for a request (model, prompt, params, ...).

    `namespace` folds the *implementation* behind the request into the key -
    tool class, its configuration, the schema version - so a changed tool can
    never replay a recording made by a different implementation.
    Keys are hex, which is also what `Cassette` accepts as a filename.
    """
    payload = list(parts) if namespace is None else [namespace, *parts]
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:32]


class Cassette:
    """A directory of recorded request->response pairs (one JSON file each).

    Modes:
      "auto"   -- replay if a recording exists, otherwise call and record
                  (the development default: pay once, then free).
      "record" -- always call the live function and (over)write the recording.
      "replay" -- never call; a missing recording is an error (CI / offline).
      "live"   -- passthrough; never read or write recordings (a real run).
    """

    #: A key must be a plain identifier - never a path.
    _KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

    def __init__(self, path: str | Path, mode: str = "auto",
                 ttl_s: float | None = None,
                 cache_errors: bool = False,
                 lock_timeout_s: float = 120.0) -> None:
        if mode not in ("auto", "record", "replay", "live"):
            raise ValueError(f"bad cassette mode {mode!r}")
        self.dir = Path(path)
        self.mode = mode
        # Time-sensitive corpora (weather, "latest" searches) must expire;
        # None keeps the historical forever-cache for reproducible corpora.
        self.ttl_s = None if ttl_s is None else float(ttl_s)
        # Whether error results enter the cache. Off by default: a transient
        # failure cached as a normal result would replay forever and freeze a
        # one-off outage into the corpus.
        self.cache_errors = bool(cache_errors)
        self.lock_timeout_s = float(lock_timeout_s)
        self.n_replayed = 0
        self.n_recorded = 0
        self.n_expired = 0
        self.n_migrated = 0
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        if mode != "live":
            self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- key safety
    def _file(self, key: str) -> Path:
        if not isinstance(key, str) or not self._KEY_RE.match(key):
            raise ValueError(
                f"unsafe cassette key {key!r}: expected [A-Za-z0-9._-]{{1,80}} "
                f"starting alphanumeric (no path separators or traversal)")
        root = self.dir.resolve()
        path = (root / f"{key}.json").resolve()
        if path.parent != root:
            raise ValueError(f"cassette key {key!r} escapes {root}")
        return path

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    # ------------------------------------------------------------ read/write
    def _read(self, path: Path) -> tuple[bool, Any]:
        """(hit, response). A corrupt or expired record counts as a miss."""
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, None
        if self.ttl_s is not None:
            age = time.time() - float(payload.get("recorded_at", 0.0))
            # `>=`, not `>`: a TTL of zero means "never replay", and a record
            # read in the same clock tick it was written has an age of exactly
            # 0.0, which `>` calls fresh. That made the boundary depend on how
            # fast the machine was --- the TTL test passed locally and failed
            # on CI, on the same commit, intermittently.
            if age >= self.ttl_s:
                self.n_expired += 1
                return False, None
        return True, payload.get("response")

    def _write(self, path: Path, key: str, result: Any) -> None:
        """Atomic write: a crash or a concurrent writer cannot leave a torn
        file behind, which the previous direct write_text could."""
        payload = _canonical({"key": key, "response": result,
                              "recorded_at": round(time.time(), 3)})
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(payload, "utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def call(self, key: str, fn: Callable[[], Any],
             legacy_keys: "tuple[str, ...]" = (),
             is_error: Callable[[Any], bool] | None = None) -> Any:
        """Return fn()'s (JSON-able) result, replaying/recording per mode.

        `legacy_keys` are older key derivations for the same request: a hit on
        one is migrated to `key`, so tightening the key scheme never throws
        away recordings that were paid for.

        `is_error` marks a result as a failure; failures are not recorded
        unless the cassette was built with `cache_errors=True`.
        """
        if self.mode == "live":
            return fn()
        path = self._file(key)

        def _lookup() -> tuple[bool, Any]:
            hit, response = self._read(path)
            if hit:
                self.n_replayed += 1
                return True, response
            for legacy in legacy_keys:
                lpath = self._file(legacy)
                lhit, lresponse = self._read(lpath)
                if lhit:
                    self.n_replayed += 1
                    self.n_migrated += 1
                    self._write(path, key, lresponse)
                    return True, lresponse
            return False, None

        if self.mode in ("auto", "replay"):
            hit, response = _lookup()
            if hit:
                return response
        if self.mode == "replay":
            raise KeyError(f"no recording for key {key} in {self.dir} "
                           f"(replay mode won't call the live function)")

        # One live call per key per process, so a concurrent miss cannot
        # duplicate a paid request.
        with self._key_lock(key):
            if self.mode == "auto":
                hit, response = _lookup()
                if hit:
                    return response
            result = fn()
            failed = bool(is_error(result)) if is_error is not None else False
            if failed and not self.cache_errors:
                return result             # transient failures are not cached
            self._write(path, key, result)
            self.n_recorded += 1
            return result

    def summary(self) -> str:
        extra = ""
        if self.n_expired:
            extra += f", {self.n_expired} expired"
        if self.n_migrated:
            extra += f", {self.n_migrated} migrated"
        return (f"cassette[{self.mode}] {self.dir.name}: "
                f"{self.n_replayed} replayed, {self.n_recorded} recorded{extra}")


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import tempfile

    # --- pricing: matches the hand-computed per-episode figures ---
    # 200k in + 20k out on Flash ~= $0.11 (the number quoted in planning).
    c = price_call("gemini-2.5-flash", 200_000, 20_000)
    assert abs(c - 0.11) < 1e-6, c
    # Flash-Lite far cheaper; Pro far dearer (and >200k trips the hi tier).
    assert price_call("gemini-2.5-flash-lite", 200_000, 20_000) < 0.03 + 1e-9
    lo = price_call("gemini-2.5-pro", 100_000, 10_000)          # <=200k tier
    hi = price_call("gemini-2.5-pro", 300_000, 30_000)          # >200k tier
    assert lo < hi and hi > 1.0, (lo, hi)
    # cached input is ~10x cheaper than fresh input.
    fresh = price_call("gemini-2.5-flash", 100_000, 0)
    cached = price_call("gemini-2.5-flash", 100_000, 0, cached_in_tok=100_000)
    assert cached < fresh / 5, (fresh, cached)

    # --- meter + budget cap ---
    m = CostMeter(budget_usd=0.50)
    m.charge("gemini-2.5-flash", 200_000, 20_000)   # ~$0.11
    m.charge("gemini-2.5-flash", 200_000, 20_000)   # ~$0.22 cumulative
    try:
        for _ in range(10):
            m.charge("gemini-2.5-flash", 200_000, 20_000)
        raise AssertionError("budget cap did not trigger")
    except BudgetExceeded:
        pass
    assert m.spent_usd <= 0.50 and m.n_calls >= 2

    # --- cassette record -> replay, and determinism of the key ---
    calls = {"n": 0}

    def live():
        calls["n"] += 1
        return {"text": "hello", "tool_uses": [], "output_tokens": 3}

    k1 = request_key("gemini-2.5-flash", [{"role": "user", "text": "hi"}], {"t": 0.0})
    k2 = request_key("gemini-2.5-flash", [{"role": "user", "text": "hi"}], {"t": 0.0})
    assert k1 == k2, "request_key not deterministic"

    with tempfile.TemporaryDirectory() as d:
        rec = Cassette(d, mode="auto")
        r1 = rec.call(k1, live)          # records (1 live call)
        r2 = rec.call(k1, live)          # replays (no live call)
        assert r1 == r2 and calls["n"] == 1, calls
        assert rec.n_recorded == 1 and rec.n_replayed == 1

        rep = Cassette(d, mode="replay")
        assert rep.call(k1, live) == r1 and calls["n"] == 1  # still no call
        try:
            rep.call(request_key("unseen"), live)
            raise AssertionError("replay of a missing key should error")
        except KeyError:
            pass

    print("PASS record_replay smoke test |", m.summary())
