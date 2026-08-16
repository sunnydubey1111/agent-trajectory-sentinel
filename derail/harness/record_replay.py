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

import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ------------------------------------------------------------------ pricing
# Loaded from `pricing/gemini.json`, not written here: list prices are
# time-sensitive, and a table in source has no field saying when it was true.
# The file carries `as_of` and a source URL, so a stale table is visible rather
# than inferred, and `PRICING_STALE_AFTER_DAYS` turns "old" into a warning.
# Override the path with AGENTWATCH_PRICING_FILE to price against a contract
# rate or a historical snapshot.
_PRICING_PATH = Path(os.environ.get(
    "AGENTWATCH_PRICING_FILE",
    str(Path(__file__).resolve().parents[2] / "pricing" / "gemini.json")))


def _load_pricing() -> dict:
    with _PRICING_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_PRICING = _load_pricing()
#: Prompt-token count above which the Pro long-context tier applies.
_PRO_THRESHOLD: int = int(_PRICING["pro_context_threshold_tokens"])
#: USD per 1M tokens, per model.
GEMINI_PRICING: dict[str, dict[str, float]] = {
    m: {k: float(v) for k, v in rates.items()}
    for m, rates in _PRICING["models"].items()
}
#: How old the table may be before `pricing_age_warning` complains.
PRICING_STALE_AFTER_DAYS = 180


def pricing_age_warning() -> str | None:
    """A message if the price table is older than the staleness window.

    Returned rather than printed so a caller can decide whether an estimate
    built on stale list prices is worth surfacing to the operator.
    """
    as_of = datetime.date.fromisoformat(str(_PRICING["as_of"]))
    age = (datetime.date.today() - as_of).days
    if age <= PRICING_STALE_AFTER_DAYS:
        return None
    return (f"price table is {age} days old (as_of {as_of}, source "
            f"{_PRICING.get('source', 'unknown')}); estimates may be wrong")


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
                    f" — refusing to record it. Raise budget_usd or use "
                    f"a cassette in replay mode. NOTE: this runs AFTER the "
                    f"provider returned, so that call has already been "
                    f"billed; it stops the NEXT one.")
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


def recorded_at_stamp() -> float:
    """The wall-clock stamp written into a recording, to the millisecond.

    Floored, never rounded. `round(t, 3)` goes to the NEAREST millisecond, so
    it can stamp a recording up to half a millisecond into the FUTURE. A read
    landing inside that window computes a negative age, and a negative age
    makes an expired record look fresh — which is how a zero TTL replayed on a
    fast machine and passed on a slow one, where the write itself takes longer
    than the window. Flooring cannot produce a stamp later than the moment of
    writing, and it is a separate function so that property can be tested
    without racing file I/O to observe it.
    """
    return math.floor(time.time() * 1000) / 1000


#: Where a live run may write. `traces/` is the committed research corpus that
#: `BASELINE_MANIFEST.json` hashes, so a serving run must never add files to
#: it: the dataset would then depend on who happened to run the demo. Override
#: with AGENTWATCH_RUNTIME_DIR; `runs/` is gitignored.
RUNTIME_DIR_ENV = "AGENTWATCH_RUNTIME_DIR"


def runtime_root() -> Path:
    """Root for runtime output, guaranteed outside the committed corpus."""
    env = os.environ.get(RUNTIME_DIR_ENV)
    root = (Path(env) if env
            else Path(__file__).resolve().parents[2] / "runs")
    return root


class Cassette:
    """A directory of recorded request->response pairs (one JSON file each).

    Modes:
      "auto"   -- replay if a recording exists, otherwise call and record
                  (the development default: pay once, then free).
      "record" -- always call the live function and (over)write the recording.
      "replay" -- never call; a missing recording is an error (CI / offline).
      "live"   -- passthrough; never read or write recordings (a real run).

    `serving=True` splits reading from writing: recordings are READ from
    `path` (the committed corpus, treated as read-only) and any new one is
    WRITTEN under `runtime_root()`. Collectors leave it False, because their
    recordings ARE the dataset; the demo and any other serving path set it, so
    running the demo cannot mutate the corpus a published number is computed
    from.
    """

    #: A key must be a plain identifier - never a path.
    _KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

    def __init__(self, path: str | Path, mode: str = "auto",
                 ttl_s: float | None = None,
                 cache_errors: bool = False,
                 lock_timeout_s: float = 120.0,
                 serving: bool = False) -> None:
        if mode not in ("auto", "record", "replay", "live"):
            raise ValueError(f"bad cassette mode {mode!r}")
        self.dir = Path(path)
        self.serving = bool(serving)
        #: Reads come from `dir` first, then `write_dir`; writes only ever go
        #: to `write_dir`. They are the same directory unless serving.
        self.write_dir = (runtime_root() / "cassettes" / self.dir.name
                          if self.serving else self.dir)
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
            self.write_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- key safety
    def _check_key(self, key: str) -> None:
        if not isinstance(key, str) or not self._KEY_RE.match(key):
            raise ValueError(
                f"unsafe cassette key {key!r}: expected [A-Za-z0-9._-]{{1,80}} "
                f"starting alphanumeric (no path separators or traversal)")

    def _in(self, root: Path, key: str) -> Path:
        self._check_key(key)
        root = root.resolve()
        path = (root / f"{key}.json").resolve()
        if path.parent != root:
            raise ValueError(f"cassette key {key!r} escapes {root}")
        return path

    def _file(self, key: str) -> Path:
        """Where a recording for `key` is READ from, preferring the source."""
        src = self._in(self.dir, key) if self.dir.exists() else None
        if src is not None and src.exists():
            return src
        return self._in(self.write_dir, key)

    def _wfile(self, key: str) -> Path:
        """Where a recording for `key` is WRITTEN. Never the source corpus."""
        return self._in(self.write_dir, key)

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    @contextlib.contextmanager
    def _hold(self, key: str):
        """Hold the per-key lock, honouring `lock_timeout_s`.

        A caller that asks for a timeout gets one; a stored-but-unread timeout
        would be an unbounded wait wearing a bound. `threading.Lock` is also
        per-PROCESS: two processes sharing a cassette directory can still
        record the same key concurrently. That is safe because `_write` is an
        atomic temp-file rename, so the loser is overwritten rather than the
        file torn — the lock only avoids duplicated live calls within a run.
        """
        lock = self._key_lock(key)
        if not lock.acquire(timeout=self.lock_timeout_s):
            raise TimeoutError(
                f"waited {self.lock_timeout_s:g}s for the cassette lock on "
                f"{key!r}; another thread is still recording it")
        try:
            yield
        finally:
            lock.release()

    # ------------------------------------------------------------ read/write
    def _read(self, path: Path) -> tuple[bool, Any]:
        """(hit, response). A corrupt or expired record counts as a miss.

        "Corrupt" has to mean every way the file can fail to be a recording,
        not only the two that raise from `json.loads`. Valid JSON that is a
        list, or an object whose `recorded_at` is a string, parses fine and
        then raises out of this method - violating the contract in the first
        line and turning a damaged cache into a crash instead of a miss.
        """
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return False, None
        if not isinstance(payload, dict):
            return False, None
        if self.ttl_s is not None:
            # Clamped at zero: a stamp from the future would otherwise make a
            # record look fresh no matter how old the caller says it may be.
            # That can happen without a broken clock - see `_write` - and it
            # can also happen with one, after an NTP step backwards.
            try:
                recorded_at = float(payload.get("recorded_at", 0.0))
            except (TypeError, ValueError):
                return False, None                 # not a usable timestamp
            age = max(0.0, time.time() - recorded_at)
            # `>=`, not `>`: a TTL of zero means "never replay", and a record
            # read in the same clock tick it was written has an age of exactly
            # 0.0, which `>` calls fresh.
            if age >= self.ttl_s:
                self.n_expired += 1
                return False, None
        return True, payload.get("response")

    def _write(self, path: Path, key: str, result: Any) -> None:
        """Atomic write: a crash or a concurrent writer cannot leave a torn
        file behind, which the previous direct write_text could."""
        payload = _canonical({"key": key, "response": result,
                              "recorded_at": recorded_at_stamp()})
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
        wpath = self._wfile(key)

        def _lookup() -> tuple[bool, Any]:
            hit, response = self._read(self._file(key))
            if hit:
                self.n_replayed += 1
                return True, response
            for legacy in legacy_keys:
                lhit, lresponse = self._read(self._file(legacy))
                if lhit:
                    self.n_replayed += 1
                    self.n_migrated += 1
                    self._write(wpath, key, lresponse)
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
        with self._hold(key):
            if self.mode == "auto":
                hit, response = _lookup()
                if hit:
                    return response
            result = fn()
            failed = bool(is_error(result)) if is_error is not None else False
            if failed and not self.cache_errors:
                return result             # transient failures are not cached
            self._write(wpath, key, result)
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

    # --- serving reads the source corpus but never writes to it ---
    with tempfile.TemporaryDirectory() as src, \
            tempfile.TemporaryDirectory() as rt:
        os.environ[RUNTIME_DIR_ENV] = rt
        try:
            collect = Cassette(src, mode="auto")
            collect.call(k1, live)                     # the "committed" one
            before = sorted(Path(src).iterdir())

            serve = Cassette(src, mode="auto", serving=True)
            assert serve.write_dir != serve.dir
            assert serve.call(k1, live) == r1, "must replay the source"
            fresh = request_key("gemini-2.5-flash", [{"role": "user",
                                                      "text": "new"}], {})
            serve.call(fresh, live)                    # records somewhere new
            assert sorted(Path(src).iterdir()) == before, \
                "a serving run wrote into the source corpus"
            written = list((Path(rt) / "cassettes").rglob("*.json"))
            assert len(written) == 1, written
            # ...and a later serving run replays what it recorded aside.
            n = calls["n"]
            assert Cassette(src, mode="auto", serving=True).call(fresh, live)
            assert calls["n"] == n, "runtime recording was not replayed"
        finally:
            os.environ.pop(RUNTIME_DIR_ENV, None)

    print("PASS record_replay smoke test |", m.summary())
