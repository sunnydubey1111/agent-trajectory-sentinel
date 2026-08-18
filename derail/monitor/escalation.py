"""Modeled judge-LLM, escalation policies, and cost accounting (Module 5, H3b).

IMPORTANT: the judge here is a STIPULATED noisy oracle, not an
empirical measurement. Its detection/false-alarm rates (JudgeConfig.p_detect /
p_false) and the per-call cost (COST_JUDGE relative to COST_STEP) are assumed
parameters, so every H3b escalation result is CONDITIONAL on those assumptions
and is a sensitivity analysis, not a measured cost saving.

That judge has since been MEASURED (`derail.experiments.run_judge_calibration`):
a real gemini-2.5-flash judge on a labelled subset of traces/ollama7b scores
p_detect = 0.548 (95% CI 0.44-0.65) and p_false = 0.052 (95% CI 0.020-0.126)
over 161 distinct prompts. BOTH stipulated values sit outside their measured
intervals - the real judge detects far less and false-alarms more. (An earlier
calibration on a larger negative stratum reported 0.057 over 172 prompts;
0.052 is the published rate, and experimental/judge_sensitivity.py explains
why its committed sweep still carries the older input.) The defaults
below are deliberately left at the stipulated values so that no published
number moves silently; the consequence is quantified instead, at one seed with
everything else held fixed, by `experimental.judge_sensitivity`
(results/tables/judge_sensitivity.csv): the call-saving survives, the
detection-recovery claim roughly halves. Set AGENTWATCH_JUDGE_P_DETECT /
AGENTWATCH_JUDGE_P_FALSE to re-run any study arm under the measured judge.

A monitor's causal score stream s_t (and optionally a running-confidence
stream) drives one of four escalation policies against the modeled judge-LLM
(`JudgeConfig`). Costs follow the contract in DESIGN.md:

    cost = COST_STEP * (steps actually executed)
         + COST_JUDGE * (judge calls)
         + COST_STEP * (T - 1 - t_h)   redo-penalty, ONLY for healthy
                                        episodes wrongfully halted at t_h.

For injected episodes the post-tau steps executed are already counted in the
first term (they ARE the waste; halting early saves them — no double count).

All policies are causal: the decision at step t uses only s_t / conf_t (values
available at step t) and past judge verdicts. Judge randomness is drawn from
a per-episode stream `rng_for(seed, "judge", episode_id)` so reruns with the
same seed are deterministic and episode outcomes do not perturb one another.

Debounce semantics: a halt requires `judge.debounce` positive verdicts on
consecutive *judged* steps; any step without a judge call (score below
threshold, or cooldown) breaks the streak. Cooldown semantics
(escalate_on_alarm only): after a negative verdict, escalation is suppressed
for the next `judge.cooldown` steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from derail.common import COST_JUDGE, COST_STEP, Episode, JudgeConfig, rng_for

POLICIES: tuple[str, ...] = (
    "never",
    "judge_every_step",
    "halt_on_alarm",
    "escalate_on_alarm",
)


def judge_verdict(episode: Episode, t: int, judge: JudgeConfig,
                  seed: int) -> bool:
    """One judge-LLM call at step t: 'is this episode derailed right now?'

    Returns True with probability judge.p_detect if the episode is injected
    and t >= tau (the derailment is underway), else True with probability
    judge.p_false.

    The random draw is keyed by (seed, episode_id, t) - NOT by position in a
    sequential per-episode stream - so the modeled verdict at a given
    (episode, step) is identical no matter how many times a policy called the
    judge earlier. Two policies that judge different subsets of steps
    now see the same judge behaviour at any shared step, which is what makes
    their cost/detection comparison fair.
    """
    derailed_now = (not episode.is_healthy) and t >= int(episode.tau)  # type: ignore[arg-type]
    p = judge.p_detect if derailed_now else judge.p_false
    u = float(rng_for(seed, "judge", episode.episode_id, t).random())
    return bool(u < p)


@dataclass
class PolicyOutcome:
    """Per-episode result of running one escalation policy."""

    episode_id: str
    is_healthy: bool
    failure_class: Optional[str]
    halted_at: Optional[int]   # step of halt (0-indexed), None if ran to end
    judge_calls: int
    cost: float                # per the module cost model (see docstring)
    detected: bool             # halted at t >= tau (injected episodes only)
    lead: Optional[int]        # T-1 - halted_at if detected else None
    wrongful_halt: bool        # halted while NOT yet derailed (see below)


def _judge_every_step(episode: Episode, judge: JudgeConfig,
                      seed: int) -> tuple[Optional[int], int]:
    """Judge at every executed step; halt on `debounce` consecutive positives."""
    streak = 0
    calls = 0
    for t in range(episode.T):
        calls += 1
        if judge_verdict(episode, t, judge, seed):
            streak += 1
            if streak >= judge.debounce:
                return t, calls
        else:
            streak = 0
    return None, calls


def _escalate_on_alarm(episode: Episode, trigger: np.ndarray, judge: JudgeConfig,
                       seed: int) -> tuple[Optional[int], int]:
    """Judge only on triggered steps; debounce to halt, cooldown after a negative."""
    streak = 0
    calls = 0
    cooldown_left = 0
    for t in range(episode.T):
        if cooldown_left > 0:
            cooldown_left -= 1
            streak = 0
            continue
        if not trigger[t]:
            streak = 0
            continue
        calls += 1
        if judge_verdict(episode, t, judge, seed):
            streak += 1
            if streak >= judge.debounce:
                return t, calls
        else:
            streak = 0
            cooldown_left = judge.cooldown
    return None, calls


def _outcome(episode: Episode, halted_at: Optional[int],
             judge_calls: int) -> PolicyOutcome:
    """Assemble a PolicyOutcome with the contract cost model.

    A WRONGFUL halt is any halt that stopped the episode while it was not yet
    derailed: a healthy episode halted at all, OR an injected episode halted
    BEFORE its onset tau. Both discard work that has to be redone, so both
    carry the redo penalty and both are reported. Charging healthy episodes
    only would make an injected episode halted before tau save steps, report
    detected=False and stay invisible - a free reward for halting too early.
    """
    executed = episode.T if halted_at is None else halted_at + 1
    cost = COST_STEP * executed + COST_JUDGE * judge_calls
    detected = (not episode.is_healthy and halted_at is not None
                and halted_at >= int(episode.tau))  # type: ignore[arg-type]
    wrongful_halt = (halted_at is not None and not detected)
    if wrongful_halt:
        cost += COST_STEP * (episode.T - 1 - halted_at)  # redo-penalty
    lead = (episode.T - 1 - halted_at) if detected else None
    return PolicyOutcome(
        episode_id=episode.episode_id,
        is_healthy=episode.is_healthy,
        failure_class=episode.failure_class,
        halted_at=halted_at,
        judge_calls=judge_calls,
        cost=float(cost),
        detected=detected,
        lead=lead,
        wrongful_halt=wrongful_halt,
    )


def run_policy(policy: str, episodes: list[Episode],
               scores: dict[str, np.ndarray],
               confidences: dict[str, np.ndarray] | None,
               theta_soft: float, judge: JudgeConfig, seed: int,
               conf_threshold: float | None = None) -> list[PolicyOutcome]:
    """Run one escalation policy over episodes; one PolicyOutcome per episode.

    Policies (all causal, stepped through t = 0..T-1):
      - "never": run every episode to completion; no judge, no halts.
      - "judge_every_step": call the judge at every step; halt on
        `judge.debounce` consecutive positive verdicts.
      - "halt_on_alarm": halt immediately at the first step with
        scores[episode_id][t] > theta_soft; no judge calls.
      - "escalate_on_alarm": while s_t > theta_soft (or, if conf_threshold is
        not None, while confidences[episode_id][t] > conf_threshold), call the
        judge; halt on `judge.debounce` consecutive positives; after a
        negative verdict, suppress escalation for `judge.cooldown` steps.

    `scores` / `confidences` map episode_id -> causal per-step stream of
    length T. Judge randomness comes from rng_for(seed, "judge", episode_id),
    so results are deterministic per (seed, episode) and comparable across
    runs. Unknown `policy` raises ValueError.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if policy == "escalate_on_alarm" and conf_threshold is not None:
        assert confidences is not None, "conf_threshold given but confidences is None"

    outcomes: list[PolicyOutcome] = []
    for ep in episodes:
        if policy == "never":
            halted_at: Optional[int] = None
            calls = 0
        elif policy == "judge_every_step":
            halted_at, calls = _judge_every_step(ep, judge, seed)
        elif policy == "halt_on_alarm":
            s = np.asarray(scores[ep.episode_id], dtype=float)
            assert s.shape == (ep.T,), f"score stream shape {s.shape} != ({ep.T},)"
            hits = np.flatnonzero(s > theta_soft)
            halted_at = int(hits[0]) if hits.size else None
            calls = 0
        else:  # escalate_on_alarm
            if conf_threshold is not None:
                stream = np.asarray(confidences[ep.episode_id], dtype=float)  # type: ignore[index]
                trigger = stream > conf_threshold
            else:
                stream = np.asarray(scores[ep.episode_id], dtype=float)
                trigger = stream > theta_soft
            assert stream.shape == (ep.T,), f"stream shape {stream.shape} != ({ep.T},)"
            halted_at, calls = _escalate_on_alarm(ep, trigger, judge, seed)
        outcomes.append(_outcome(ep, halted_at, calls))
    return outcomes


def cost_at(mean_cost: float, mean_judge_calls: float, ratio: float,
            *, recorded_ratio: float = 1.0) -> float:
    """Re-express a recorded mean cost at a different judge/step price ratio.

    The cost model is linear in `COST_JUDGE`, so a table produced at one ratio
    carries enough information to answer "what would this have been at
    another?" without re-running anything: the non-judge part is
    `mean_cost - recorded_ratio * mean_judge_calls`, and it does not move.
    """
    non_judge = mean_cost - recorded_ratio * mean_judge_calls
    return non_judge + ratio * mean_judge_calls


def cost_ratio_at(selective: tuple[float, float],
                  every_step: tuple[float, float], ratio: float,
                  *, recorded_ratio: float = 1.0) -> float:
    """Selective-policy cost as a fraction of judging every step, at `ratio`.

    Each argument is that policy's `(mean_cost, mean_judge_calls)` as recorded.
    Below 1.0 the selective policy is the cheaper one.
    """
    return (cost_at(*selective, ratio, recorded_ratio=recorded_ratio)
            / cost_at(*every_step, ratio, recorded_ratio=recorded_ratio))


def cost_ratio_break_even(selective: tuple[float, float],
                          every_step: tuple[float, float],
                          *, recorded_ratio: float = 1.0) -> float:
    """The judge/step price ratio at which selective escalation stops winning.

    Reported rather than assumed, because `COST_JUDGE == COST_STEP` is a choice
    of units and the conclusion drawn from it is not scale-free: a judge cheap
    enough relative to an agent step makes judging EVERY step the cheaper
    option, and no amount of selectivity recovers that.
    """
    s_cost, s_calls = selective
    e_cost, e_calls = every_step
    num = ((s_cost - recorded_ratio * s_calls)
           - (e_cost - recorded_ratio * e_calls))
    den = e_calls - s_calls
    if den == 0:
        return float("nan")
    return num / den


def summarize_policy(outcomes: list[PolicyOutcome]) -> dict:
    """Aggregate PolicyOutcomes.

    detection_rate is over injected episodes; mean_lead over detected episodes.
    wrongful_halt_rate is over ALL episodes and counts every wrongful halt - a
    healthy episode halted at all, OR an injected episode halted before its
    onset; healthy_wrongful_halt_rate keeps the healthy-only figure
    for continuity. Empty denominators yield NaN. mean_judge_calls isolates the
    monitoring overhead from the total cost, which is dominated by agent steps.
    """
    costs = np.array([o.cost for o in outcomes], dtype=float)
    calls = np.array([o.judge_calls for o in outcomes], dtype=float)
    injected = [o for o in outcomes if not o.is_healthy]
    healthy = [o for o in outcomes if o.is_healthy]
    leads = np.array([o.lead for o in outcomes if o.detected], dtype=float)
    return {
        "mean_cost": float(costs.mean()) if costs.size else float("nan"),
        "mean_judge_calls": float(calls.mean()) if calls.size else float("nan"),
        "detection_rate": (float(np.mean([o.detected for o in injected]))
                           if injected else float("nan")),
        "mean_lead": float(leads.mean()) if leads.size else float("nan"),
        "wrongful_halt_rate": (float(np.mean([o.wrongful_halt for o in outcomes]))
                               if outcomes else float("nan")),
        "healthy_wrongful_halt_rate": (
            float(np.mean([o.wrongful_halt for o in healthy]))
            if healthy else float("nan")),
        "early_injected_halt_rate": (
            float(np.mean([o.wrongful_halt for o in injected]))
            if injected else float("nan")),
    }


if __name__ == "__main__":
    import time

    from derail.common import D_TOTAL

    t0 = time.time()
    T, tau = 40, 20
    rng = rng_for(0, "smoke", "escalation")
    healthy_ep = Episode(X=rng.normal(size=(T, D_TOTAL)), episode_id="smoke-h0",
                         is_healthy=True, failure_class=None, tau=None,
                         t_fail=None, severity=None)
    injected_ep = Episode(X=rng.normal(size=(T, D_TOTAL)), episode_id="smoke-f0",
                          is_healthy=False, failure_class="looping", tau=tau,
                          t_fail=T - 1, severity=0.8)
    episodes = [healthy_ep, injected_ep]
    # Score streams: healthy stays low; injected steps up at tau.
    scores = {
        "smoke-h0": np.full(T, 0.1),
        "smoke-f0": np.where(np.arange(T) >= tau, 2.0, 0.1),
    }
    # Running confidences (nondecreasing, causal by construction).
    confidences = {eid: np.clip(np.maximum.accumulate(s) / 2.5, 0.0, 1.0)
                   for eid, s in scores.items()}
    judge = JudgeConfig()
    theta_soft = 1.0
    seed = 123

    summaries: dict[str, dict] = {}
    outcomes_by_policy: dict[str, list[PolicyOutcome]] = {}
    for pol in POLICIES:
        outs = run_policy(pol, episodes, scores, None, theta_soft, judge, seed)
        outcomes_by_policy[pol] = outs
        summaries[pol] = summarize_policy(outs)
        for o in outs:
            print(f"{pol:>18s} {o.episode_id}: halted_at={o.halted_at} "
                  f"judge_calls={o.judge_calls} cost={o.cost:.1f} "
                  f"detected={o.detected} lead={o.lead}")

    # "never": no halts, cost = COST_STEP * T per episode.
    for o in outcomes_by_policy["never"]:
        assert o.halted_at is None and o.judge_calls == 0
        assert o.cost == COST_STEP * T and not o.detected and o.lead is None

    # judge_every_step detects the injected episode.
    assert summaries["judge_every_step"]["detection_rate"] == 1.0

    # halt_on_alarm halts exactly at tau on the injected episode.
    h_alarm = {o.episode_id: o for o in outcomes_by_policy["halt_on_alarm"]}
    assert h_alarm["smoke-f0"].halted_at == tau and h_alarm["smoke-f0"].detected
    assert h_alarm["smoke-f0"].lead == T - 1 - tau
    assert h_alarm["smoke-h0"].halted_at is None

    # escalate_on_alarm detects and costs less than judge_every_step.
    assert summaries["escalate_on_alarm"]["detection_rate"] == 1.0
    assert (summaries["escalate_on_alarm"]["mean_cost"]
            < summaries["judge_every_step"]["mean_cost"])

    # Confidence-gated escalation path is deterministic and detects too.
    conf_outs = run_policy("escalate_on_alarm", episodes, scores, confidences,
                           theta_soft, judge, seed, conf_threshold=0.7)
    conf_outs2 = run_policy("escalate_on_alarm", episodes, scores, confidences,
                            theta_soft, judge, seed, conf_threshold=0.7)
    assert [o.__dict__ for o in conf_outs] == [o.__dict__ for o in conf_outs2]
    conf_by_id = {o.episode_id: o for o in conf_outs}
    assert conf_by_id["smoke-f0"].detected and not conf_by_id["smoke-h0"].detected

    # Wrongful-halt cost model: healthy halted at t_h costs
    # COST_STEP*(t_h+1) + COST_JUDGE*calls + COST_STEP*(T-1-t_h).
    forced = run_policy("halt_on_alarm", [healthy_ep],
                        {"smoke-h0": np.full(T, 5.0)}, None, theta_soft,
                        judge, seed)[0]
    assert forced.halted_at == 0 and forced.cost == COST_STEP * T
    assert not forced.detected and forced.wrongful_halt
    assert summarize_policy([forced])["wrongful_halt_rate"] == 1.0

    # early wrongful halt on an INJECTED episode (see _outcome's docstring
    # for why this must carry the redo penalty and count as wrongful):
    early = run_policy("halt_on_alarm", [injected_ep],
                       {"smoke-f0": np.where(np.arange(T) >= 5, 5.0, 0.1)},
                       None, theta_soft, judge, seed)[0]
    assert early.halted_at == 5 and not early.detected and early.wrongful_halt
    assert early.cost == COST_STEP * T, "pre-onset halt did not pay the redo penalty"
    summ_early = summarize_policy([early])
    assert summ_early["wrongful_halt_rate"] == 1.0
    assert summ_early["early_injected_halt_rate"] == 1.0

    # order-independence of judge_verdict (see its docstring for why):
    for t in range(T):
        assert (judge_verdict(injected_ep, t, judge, seed)
                == judge_verdict(injected_ep, t, judge, seed)), \
            "judge verdict not order-independent / not deterministic"

    print(f"PASS escalation smoke test in {time.time() - t0:.2f}s | "
          f"mean costs: " + ", ".join(
              f"{p}={summaries[p]['mean_cost']:.1f}" for p in POLICIES))
