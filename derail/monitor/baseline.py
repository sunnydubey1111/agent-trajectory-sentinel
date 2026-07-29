"""Self-calibrating baseline for a deployment that has no calibration corpus.

A one-class monitor needs a reference distribution of healthy runs, and every
axis of the serving configuration changes that distribution: the model, the
decoding temperature, the system prompt, the tool roster, the telemetry
version. Collecting a fresh corpus by hand on every such change is the cost the
study measured and it is not something a deployment can pay repeatedly.

This module lets the baseline build itself:

* `ServingConfig.fingerprint()` identifies the configuration a baseline belongs
  to. When it changes, the old baseline is not merely stale, it describes a
  different system, so `RollingBaseline.reconfigure` retires it.
* `RollingBaseline` accumulates scores from completed runs over a rolling
  window and reports a threshold once it has enough of them to mean anything.
  "Enough" is not a guess: `metrics.min_calibration_episodes` gives the count
  below which the requested false-alarm budget is arithmetically unreachable.
* Admission is guarded. A run only joins the baseline if it passed the
  deterministic checks and did not itself alarm, so a monitor cannot learn that
  a failing run is normal. This is the poisoning that made the demo's own
  corpus useless until task-incomplete and wrong-total runs were excluded from
  it (DESIGN.md Amendment 7).
* The state is explicit — `warming_up`, `trusted`, `drifting`,
  `recalibrating` — because a caller must be able to tell "no alarm" from
  "not yet able to raise one".

Deterministic checks (`derail.verify.checks`) need none of this and stay live
from the first run, which is what makes a blind warm-up period acceptable.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Iterable

from derail.evaluation.metrics import min_calibration_episodes, pick_threshold

#: Baseline states, in the order a healthy deployment passes through them.
WARMING_UP = "warming_up"
TRUSTED = "trusted"
DRIFTING = "drifting"
RECALIBRATING = "recalibrating"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ServingConfig:
    """Everything that changes what a healthy run looks like.

    Two deployments with the same fingerprint may share a baseline; two with
    different fingerprints may not, however similar they appear.
    """

    model: str
    temperature: float
    prompt: str = ""
    tools: tuple[str, ...] = ()
    telemetry_schema: int | str = ""

    def fingerprint(self) -> str:
        payload = asdict(self)
        # The prompt is hashed rather than stored: it can be long, and only its
        # identity matters here.
        payload["prompt"] = _sha256(self.prompt) if self.prompt else ""
        payload["tools"] = sorted(self.tools)
        return _sha256(json.dumps(payload, sort_keys=True,
                                  separators=(",", ":"), default=str))


@dataclass
class RollingBaseline:
    """Healthy-score reference built from the deployment's own runs."""

    config: ServingConfig
    fa_budget: float = 0.10
    window: int = 200
    #: Runs whose own score already exceeds the working threshold are refused
    #: admission, so an alarming run cannot widen the null that judges it.
    admit_alarming: bool = False

    _scores: deque = field(default_factory=deque, init=False, repr=False)
    _fingerprint: str = field(default="", init=False)
    _recalibrating: bool = field(default=False, init=False)
    _rejected: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._scores = deque(maxlen=self.window)
        self._fingerprint = self.config.fingerprint()

    # -- identity ---------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def reconfigure(self, config: ServingConfig) -> bool:
        """Point the baseline at a configuration. True if it was retired.

        A changed fingerprint means the reference describes a different system,
        so keeping it would be worse than having none: the threshold would be
        confidently wrong rather than merely absent.
        """
        new = config.fingerprint()
        if new == self._fingerprint:
            return False
        self.config = config
        self._fingerprint = new
        self._scores.clear()
        self._rejected = 0
        self._recalibrating = True
        return True

    # -- accumulation -----------------------------------------------------
    @property
    def n(self) -> int:
        return len(self._scores)

    @property
    def n_required(self) -> int:
        """Runs needed before `fa_budget` is even arithmetically reachable."""
        return min_calibration_episodes(self.fa_budget)

    @property
    def rejected(self) -> int:
        """Runs refused admission, i.e. poisoning that was prevented."""
        return self._rejected

    def observe(self, score: float, *, checks_passed: bool) -> bool:
        """Offer one completed run's peak score. True if it joined the null.

        `checks_passed` is the deterministic verdict, which needs no baseline
        and is therefore available from the very first run — that is what makes
        guarded admission possible during warm-up.
        """
        if not checks_passed:
            self._rejected += 1
            return False
        if not self.admit_alarming and self.state == TRUSTED:
            theta = self.threshold()
            if theta is not None and score > theta:
                self._rejected += 1
                return False
        self._scores.append(float(score))
        if self._recalibrating and self.n >= self.n_required:
            self._recalibrating = False
        return True

    # -- serving ----------------------------------------------------------
    def threshold(self) -> float | None:
        """Alarm threshold, or None while the budget is unreachable."""
        if self.n < self.n_required:
            return None
        return float(pick_threshold([[s] for s in self._scores],
                                    fa_budget=self.fa_budget,
                                    warn_infeasible=False))

    @property
    def state(self) -> str:
        if self._recalibrating and self.n < self.n_required:
            return RECALIBRATING
        if self.n < self.n_required:
            return WARMING_UP
        if self.realized_fa() is not None and self.realized_fa() > 2 * self.fa_budget:
            return DRIFTING
        return TRUSTED

    def realized_fa(self) -> float | None:
        """Fraction of the window that would alarm at the current threshold.

        Reported rather than assumed: the study measured budgets missed by up
        to threefold, so a deployment must be able to see the rate it is
        actually getting.
        """
        theta = self.threshold()
        if theta is None or not self._scores:
            return None
        return sum(1 for s in self._scores if s > theta) / len(self._scores)

    def can_act(self) -> bool:
        """May the monitor drive an autonomous action right now?

        False while warming up or recalibrating: with no usable threshold an
        alarm carries no information, and acting on it would be worse than
        waiting. Deterministic checks are unaffected and keep running.
        """
        return self.state in (TRUSTED, DRIFTING)

    def snapshot(self) -> dict:
        return {"state": self.state, "n": self.n,
                "n_required": self.n_required,
                "rejected": self._rejected,
                "threshold": self.threshold(),
                "realized_fa": self.realized_fa(),
                "fingerprint": self._fingerprint[:12]}

    def extend(self, scores: Iterable[float]) -> None:
        """Seed from an existing corpus, bypassing admission checks.

        For the case where a curated healthy corpus already exists; a
        deployment starting cold simply never calls this.
        """
        for s in scores:
            self._scores.append(float(s))
        if self._recalibrating and self.n >= self.n_required:
            self._recalibrating = False


if __name__ == "__main__":       # self-test: no model, no network, no corpus
    cfg = ServingConfig(model="qwen2.5:7b", temperature=0.2,
                        prompt="you are a booking assistant",
                        tools=("lookup_flight", "lookup_hotel"),
                        telemetry_schema=4)
    b = RollingBaseline(cfg, fa_budget=0.10)

    assert b.n_required == 9, b.n_required          # ceil(1/0.10 - 1)
    assert b.state == WARMING_UP and not b.can_act()
    assert b.threshold() is None, "no threshold before the budget is reachable"

    for i in range(9):
        assert b.observe(0.5 + 0.01 * i, checks_passed=True)
    assert b.state == TRUSTED and b.can_act()
    assert b.threshold() is not None

    # A run that fails its checks never joins the null.
    before = b.n
    assert not b.observe(0.4, checks_passed=False)
    assert b.n == before and b.rejected == 1

    # Nor does one that alarms against the current threshold.
    assert not b.observe(99.0, checks_passed=True)
    assert b.rejected == 2

    # Tool roster is part of the identity: changing it retires the baseline.
    moved = ServingConfig(model="qwen2.5:7b", temperature=0.2,
                          prompt="you are a booking assistant",
                          tools=("lookup_flight",), telemetry_schema=4)
    assert b.reconfigure(moved)
    assert b.n == 0 and b.state == RECALIBRATING and not b.can_act()
    assert not b.reconfigure(moved), "same config must not retire again"

    # Tool ORDER is not identity; the same roster reordered is the same config.
    a = ServingConfig(model="m", temperature=0.2, tools=("b", "a"))
    c = ServingConfig(model="m", temperature=0.2, tools=("a", "b"))
    assert a.fingerprint() == c.fingerprint()
    assert ServingConfig(model="m", temperature=0.3).fingerprint() != \
        ServingConfig(model="m", temperature=0.2).fingerprint()

    print("PASS: monitor.baseline self-test")
