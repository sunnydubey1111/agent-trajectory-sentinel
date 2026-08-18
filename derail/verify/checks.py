"""Deterministic checks over a run's own tool results.

Organic agent failures are often behaviourally silent: the agent looks up
every price correctly, sounds confident, and combines the numbers wrongly.
There is no anomaly for a trajectory monitor to see, but the answer is
*checkable*, and a check needs no healthy null, no threshold and no
per-deployment recalibration. Contract and measured results: DESIGN.md
Module 8.

What these checks may and may not read
--------------------------------------
Only what the agent itself observed: the tool calls it made and the results it
got back. NEVER the hidden world the task was generated from. That distinction
is the whole point — `_demo_expected_total(seed, world)` is the study's ORACLE
and is not deployable, whereas these checks run in production. They are
therefore strictly weaker than the oracle: an agent that looks up the wrong
city gets consistent arithmetic over the wrong inputs and passes
`total_consistency`, which `required_coverage` is there to catch instead.

Genericity
----------
The mechanisms are task-independent — recompute a stated total from observed
line items; confirm every declared-required call happened. What is per-task is
the small `TaskSpec` saying which tools return line items and with what
multiplicity. That spec is written once by whoever defines the task; it is not
harvested from 120 calibration runs.

The NUMBER READER underneath, and its dialect
---------------------------------------------
The mechanism is task-independent; the dialect the numbers are written in is
not, so it is declared on the spec beside the line items: `currency` and
`total_labels`, defaulting to US dollars and English. That matters because a
figure the reader cannot match is not read as "unpriced" — it is read as
absent, i.e. as "nothing to reconcile", which is a pass. Both readers
therefore refuse input outside their declared dialect
(`UnsupportedInputError`), and `TaskSpec.strict_currency=False` restores the
blind behaviour for a caller that has decided it is acceptable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from derail.preconditions import (DEFAULT_CURRENCY, currency_tokens,
                                  require_readable_money)
from derail.telemetry.events import (ToolCallEvent, canonical_args,
                                     parse_step_events)

#: The words the answer may use to name its total, most specific first.
DEFAULT_TOTAL_LABELS = ("grand total", "overall total", "total")
_AMOUNT = r"(\d[\d,]*(?:\.\d+)?)"


@lru_cache(maxsize=32)
def _readers(currency: str, total_labels: tuple[str, ...]
             ) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    """The three number readers for one (currency, label vocabulary).

    Cached because `total_consistency` calls them once per tool result and the
    patterns are pure functions of the spec.
    """
    sym = re.escape(currency)
    code = re.escape(_symbol_code(currency))
    labels = "|".join(re.escape(w).replace(r"\ ", r"\s+")
                      for w in total_labels)
    #: A money-ish figure in a tool result, e.g. "$502" or "$165/night".
    price = re.compile(rf"{sym}\s?{_AMOUNT}")
    #: The agent's stated figure: "$4,935" or "4935 USD".
    stated = re.compile(rf"{sym}\s?{_AMOUNT}|{_AMOUNT}\s*{code}", re.I)
    #: A figure the answer explicitly calls the total (see stated_total's
    #: docstring for why this beats "the last monetary figure"). Observed
    #: live: a repaired run replied "Total flight cost: $2755, hotel cost:
    #: $1836, ...". Verified to change no verdict on the 480 committed
    #: demo-task episodes.
    labelled = re.compile(
        rf"(?:{labels})(?!\s+\w+\s+cost)\D{{0,40}}?{sym}?\s?"
        rf"([\d,]+(?:\.\d+)?)", re.I)
    return price, stated, labelled


def _symbol_code(currency: str) -> str:
    """The ISO code that pairs with a currency symbol ('$' -> 'USD')."""
    return next((t for t in currency_tokens(currency) if t != currency),
                currency)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def first_price(result: str, strict: bool = True,
                currency: str = DEFAULT_CURRENCY) -> float | None:
    """The first `currency` figure in a tool result, or None.

    None means "this result quoted no price". `strict` keeps that true by
    refusing a result that quotes one this reader cannot see.
    """
    if strict:
        require_readable_money(result or "", "first_price", currency)
    price, _, _ = _readers(currency, DEFAULT_TOTAL_LABELS)
    m = price.search(result or "")
    return _to_float(m.group(1)) if m else None


def stated_total(text: str, strict: bool = True,
                 currency: str = DEFAULT_CURRENCY,
                 total_labels: tuple[str, ...] = DEFAULT_TOTAL_LABELS
                 ) -> float | None:
    """The total the agent asserts.

    Read from the figure the answer explicitly labels as a total; failing that,
    the last monetary figure. Position alone mis-reads an answer that ends on a
    line item, and a plain substring test mis-reads one whose line item happens
    to contain the right digits.

    None means "the agent stated no total" — a finding in itself — so `strict`
    refuses an answer whose total is stated in an unreadable currency rather
    than reporting one that was never made.
    """
    if strict:
        require_readable_money(text or "", "stated_total", currency)
    _, stated, labelled = _readers(currency, tuple(total_labels))
    m = labelled.search(text or "")
    if m is not None:
        v = _to_float(m.group(1))
        if v is not None:
            return v
    vals = []
    for m in stated.finditer(text or ""):
        v = _to_float(m.group(1) or m.group(2))
        if v is not None:
            vals.append(v)
    return vals[-1] if vals else None


@dataclass(frozen=True)
class LineItem:
    """One class of priced tool result and how often it enters the total."""

    tool: str
    multiplier: int = 1
    #: how many DISTINCT such lookups a complete run must make; None = skip
    required_calls: int | None = None
    #: Count each distinct (tool, args) lookup once: agents re-query the same
    #: item, and summing every result would double-count it. Set False for a
    #: task where repeating a lookup genuinely means buying it twice.
    distinct: bool = True


@dataclass(frozen=True)
class TaskSpec:
    """The per-task part: what a complete, correctly-totalled run looks like."""

    line_items: tuple[LineItem, ...]
    #: tools that must be called but contribute no money (e.g. weather)
    required_tools: tuple[str, ...] = ()
    #: (tool, n) pairs for tasks that name how many times a tool must be used
    required_counts: tuple[tuple[str, int], ...] = ()
    #: tools that mean the agent has stopped gathering and started combining.
    #: A required lookup still outstanding at that point is already missing,
    #: which is what lets coverage report before the run ends.
    combining_tools: tuple[str, ...] = ()
    #: (tool, pattern) pairs declaring the shape a successful result may take.
    #: A result matching none of them is malformed at the tool boundary and the
    #: agent should never have been handed it. Only declare a pattern for a
    #: tool whose output shape is genuinely closed; a free-text tool has no
    #: contract to violate and must be left out rather than approximated.
    result_contracts: tuple[tuple[str, str], ...] = ()
    #: relative tolerance when comparing the stated total to the recomputed one
    tolerance: float = 0.01
    #: Absolute floor under the relative tolerance, and the value the relative
    #: part is measured against when a candidate total is smaller than it.
    #: Both are task facts: the floor is the rounding a correct answer may
    #: carry (cents), the pivot stops a near-zero candidate from being held to
    #: a near-zero window. Declared rather than fixed in the comparison so a
    #: task priced in whole units or in millions can state its own.
    tolerance_floor: float = 0.5
    tolerance_pivot: float = 1.0
    #: Ceiling on the candidate-selection search. The search is combinatorial
    #: in how many prices a run observed, which is bounded for a booking task
    #: and unbounded for a long or adversarial episode. Past this the check
    #: reports `unverifiable` rather than spending the time or, worse, being
    #: quietly capped into answering a smaller question than it was asked.
    max_selection_candidates: int = 100_000
    name: str = "task"
    #: Refuse text pricing anything in a currency this spec's number reader
    #: cannot see, instead of silently reading it as priceless and passing.
    #: Set False only for a corpus known to contain such text, where a blind
    #: reading is a decision rather than an oversight.
    strict_currency: bool = True
    #: The task's currency and total-wording (see module docstring, "The
    #: NUMBER READER underneath"). A euro task in French passes its own.
    currency: str = DEFAULT_CURRENCY
    total_labels: tuple[str, ...] = DEFAULT_TOTAL_LABELS


@dataclass
class Finding:
    """One check failure. `step` is the earliest step it could be known at."""

    check: str
    step: int | None
    detail: str
    #: The same finding with computed VALUES removed. `detail` may contain the
    #: total recomputed from the agent's own tool results, which for a run that
    #: looked everything up and merely mis-added IS the correct answer; a
    #: repair prompt built from it therefore hands the answer over rather than
    #: only locating the fault. `terse` names the fault and nothing else, so
    #: the two effects can be measured apart. Defaults to `detail`.
    terse: str = ""

    def __post_init__(self) -> None:
        if not self.terse:
            self.terse = self.detail


@dataclass
class VerificationResult:
    findings: list[Finding] = field(default_factory=list)
    recomputed_total: float | None = None
    stated: float | None = None
    #: Why the totals check could not run, or None if it ran. Distinct from a
    #: pass: no findings and `unverifiable` set means "not checked", which
    #: `failed` deliberately does NOT report as a failure and a caller must
    #: not read as a clean bill.
    unverifiable: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.findings)

    @property
    def checked(self) -> bool:
        """True only if the totals check actually read evidence."""
        return self.unverifiable is None


def _events_per_step(steps: list[dict]) -> list[list[ToolCallEvent]]:
    """The tool calls of each step, read by the one telemetry reader.

    Delegated rather than reimplemented: a second reader here would answer
    differently from the one the monitor uses, and `tool_contract` depends on
    `has_result` to decide whether there is a result to match a pattern
    against at all. `parse_step_events` derives that, `truncated` and
    `args_key` from the record instead of assuming them.
    """
    return [parse_step_events(s)[0] for s in steps]


def required_coverage(steps: list[dict], spec: TaskSpec) -> list[Finding]:
    """Did every call the task requires actually happen?

    Reports at the earliest step where the answer is already determined: the
    step on which the agent starts combining (`spec.combining_tools`) while a
    requirement is outstanding, since from then on it is no longer gathering.
    Without such a step the finding lands on the final step, because a call
    not yet made is not the same as a call that will never be made.
    """
    per_step = _events_per_step(steps)
    distinct = {li.tool: li.distinct for li in spec.line_items}
    seen: dict[str, set] = {}
    counts: dict[str, int] = {}
    for evs in per_step:
        for e in evs:
            if e.is_error:
                continue
            if distinct.get(e.name, True):
                key = canonical_args(e.args)
                bucket = seen.setdefault(e.name, set())
                if key in bucket:
                    continue
                bucket.add(key)
            counts[e.name] = counts.get(e.name, 0) + 1
    findings = []
    last = len(steps) - 1
    # Dating a shortfall to `onset` (see docstring) is sound because the
    # counts below are whole-episode and only ever grow: a requirement unmet
    # at the end was unmet at the combining step too. The converse would need
    # a running tally, and it cannot happen.
    onset = last
    if spec.combining_tools:
        for i, evs in enumerate(per_step):
            if any(e.name in spec.combining_tools and not e.is_error
                   for e in evs):
                onset = i
                break
    for li in spec.line_items:
        if li.required_calls is None:
            continue
        got = counts.get(li.tool, 0)
        if got < li.required_calls:
            findings.append(Finding(
                "required_coverage", onset,
                f"{li.tool}: {got} successful call(s), task requires "
                f"{li.required_calls}"))
    for tool in spec.required_tools:
        if counts.get(tool, 0) == 0:
            findings.append(Finding("required_coverage", onset,
                                    f"{tool}: never called"))
    for tool, need in spec.required_counts:
        got = counts.get(tool, 0)
        if got < need:
            findings.append(Finding(
                "required_coverage", onset,
                f"{tool}: {got} successful call(s), task requires {need}"))
    return findings


def total_consistency(steps: list[dict], spec: TaskSpec) -> VerificationResult:
    """Does the stated total match a valid selection of the observed prices?

    An agent may look something up and then correctly decide not to use it —
    measured on llama3.1:8b, which priced six flights for a four-leg tour and
    totalled the right four. Requiring the total to equal the sum of EVERY
    observed price would call that run wrong, so where a line item declares
    how many it needs (`required_calls`), the check asks whether some selection
    of that many observed prices reproduces the stated total.

    That keeps what matters: a dropped line item, a double count or a spurious
    operation leaves no valid selection at all. Where no count is declared, all
    observed prices are used, as before. Nothing here reads the hidden world.

    When there is no evidence to reconcile, the result is marked
    `unverifiable` rather than returned clean: a spec that prices nothing has
    no contract here, and a priced spec that observed no price has a contract
    it could not test. For BOOKING_SPEC `required_coverage` catches the second
    case on every booking-shaped episode in the corpus, but that is a property
    of that spec's required counts, not a guarantee this check makes.
    """
    strict = spec.strict_currency
    per_step = _events_per_step(steps)
    priced = {li.tool: li for li in spec.line_items}
    prices: dict[str, list[float]] = {}
    seen_any = False
    counted: dict[str, set] = {}
    for evs in per_step:
        for e in evs:
            li = priced.get(e.name)
            if li is None or e.is_error:
                continue
            if li.distinct:
                key = canonical_args(e.args)
                bucket = counted.setdefault(e.name, set())
                if key in bucket:      # same item re-queried, not bought twice
                    continue
                bucket.add(key)
            p = first_price(e.result, strict=strict, currency=spec.currency)
            if p is None:
                continue
            prices.setdefault(e.name, []).append(p)
            seen_any = True

    # Total using everything observed — reported as `recomputed_total`, and the
    # only candidate when no line item declares a required count.
    total = sum(priced[t].multiplier * p
                for t, ps in prices.items() for p in ps)

    said = None
    for s in reversed(steps):
        said = stated_total(str(s.get("text", "")), strict=strict,
                            currency=spec.currency,
                            total_labels=spec.total_labels)
        if said is not None:
            break

    res = VerificationResult(recomputed_total=total if seen_any else None,
                             stated=said)
    last = len(steps) - 1
    if not priced:
        res.unverifiable = "the task spec declares no priced line items"
        return res
    if not seen_any:
        res.unverifiable = ("no priced tool result was observed, so there was "
                            "nothing to reconcile the stated total against")
        return res
    if said is None:
        res.findings.append(Finding("total_consistency", last,
                                    "no total stated in the final answer"))
        return res
    decided = _selection_matches(prices, priced, said, spec)
    if decided is None:
        res.unverifiable = (
            f"more than {spec.max_selection_candidates:,} candidate selections "
            f"of the observed prices; the stated total was not reconciled")
        return res
    if decided:
        return res
    res.findings.append(Finding(
        "total_consistency", last,
        f"stated {said:g} but no valid selection of the observed tool results "
        f"reproduces it (all of them sum to {total:g})",
        terse="the stated total does not match the figures this run looked up"))
    return res


def tolerance_for(spec: TaskSpec, value: float) -> float:
    """Absolute window allowed when comparing a stated total to `value`."""
    return max(spec.tolerance * max(abs(value), spec.tolerance_pivot),
               spec.tolerance_floor)


def _selection_matches(prices: dict[str, list[float]],
                       priced: dict[str, LineItem],
                       said: float, spec: TaskSpec) -> bool | None:
    """Can the stated total be built from the observed prices?

    None means the search exceeded `spec.max_selection_candidates` and the
    question was not decided - distinct from False, which is a real mismatch.

    Each line item contributes exactly `required_calls` of the prices seen for
    it, or all of them when no count is declared. Extra lookups the agent chose
    not to use are therefore allowed; a missing or double-counted one is not.

    The tolerance is computed against each CANDIDATE selection, not against the
    sum of everything observed. Scaling it by the grand total would let an
    agent widen its own acceptance window by making extra expensive lookups it
    never used - the window for a $900 candidate must not depend on a $5,000
    flight the agent priced and discarded.
    """
    from itertools import combinations
    from math import comb

    cap = spec.max_selection_candidates
    sums: set[float] = {0.0}
    for tool, ps in sorted(prices.items()):
        li = priced[tool]
        need = li.required_calls
        if need is None or need >= len(ps):
            options = {li.multiplier * sum(ps)}
        else:
            if comb(len(ps), need) > cap:
                return None
            options = {li.multiplier * sum(c)
                       for c in combinations(sorted(ps), need)}
        sums = {a + b for a in sums for b in options}
        if len(sums) > cap:
            return None
    return any(abs(said - v) <= tolerance_for(spec, v) for v in sums)


def tool_contract(steps: list[dict], spec: TaskSpec) -> list[Finding]:
    """Flag tool results that match none of their tool's declared shapes.

    The other two checks read what the agent *did* with its evidence. This one
    reads the evidence itself: a `lookup_flight` that answers anything but a
    price has failed at the boundary, whichever way its bytes were mangled, and
    a run built on it is unsound however carefully the agent then adds up.

    It reports at the step the result arrived, so it is the earliest signal
    available anywhere in the system - no null, no threshold, no waiting for an
    answer to check. It is deliberately silent on corruption that keeps a legal
    shape: a price altered from $361 to $605 is a well-formed price, and
    separating it from a real one needs a reference this layer does not have.
    """
    findings: list[Finding] = []
    contracts = {tool: re.compile(pattern)
                 for tool, pattern in spec.result_contracts}
    if not contracts:
        return findings
    for t, events in enumerate(_events_per_step(steps)):
        for event in events:
            pattern = contracts.get(event.name)
            # An error result is a declared outcome, not a malformed one; the
            # error flag already carries it to the monitors that want it.
            if pattern is None or event.is_error or not event.has_result:
                continue
            result = (event.result or "").strip()
            # fullmatch, not match: a contract describes the WHOLE result. A
            # malformed result that happens to start with a legal price would
            # satisfy `match` and pass, which is the case this check exists
            # for. Contracts are authored as complete shapes for that reason.
            if pattern.fullmatch(result):
                continue
            findings.append(Finding(
                check="tool_contract", step=t,
                detail=(f"{event.name} returned {result[:60]!r}, which is not "
                        f"a valid result for that tool"),
                terse=f"{event.name} returned a malformed result"))
    return findings


def verify(steps: list[dict], spec: TaskSpec) -> VerificationResult:
    """Run every check. No thresholds, no calibration, no healthy reference."""
    res = total_consistency(steps, spec)
    res.findings.extend(required_coverage(steps, spec))
    res.findings.extend(tool_contract(steps, spec))
    return res


#: The demo booking task: four flight legs, three hotels at two nights each,
#: and weather for the three hotel cities. Written from the task statement in
#: `demo._make_demo_task`, not learned from any corpus.
BOOKING_SPEC = TaskSpec(
    line_items=(
        LineItem("lookup_flight", multiplier=1, required_calls=4),
        LineItem("lookup_hotel", multiplier=2, required_calls=3),
    ),
    required_tools=("get_weather",),
    combining_tools=("calculator",),
    #: Taken from the tool implementations in `collect_traces._run_tool`, which
    #: can return only these shapes. `get_weather` is free text and declares no
    #: contract. Each alternative describes a COMPLETE result, because
    #: `tool_contract` fullmatches: the two message-shaped outcomes carry
    #: explanatory text after the phrase, so they end in `.*` rather than
    #: stopping at the prefix.
    result_contracts=(
        ("lookup_flight", r"\$\d+(\.\d+)?|No route found.*"),
        ("lookup_hotel", r"\$\d+(\.\d+)?/night|Unknown city\."),
        ("search_catalog", r"\$\d+(\.\d+)?|Item not found\."),
        ("calculator", r"-?\d+(\.\d+)?|Error: .*"),
    ),
    name="demo_booking",
)


#: The real-tools research task served by `derail.harness.demo_real`: two
#: arXiv searches, two Wikipedia lookups, one web search and one python call.
#: It has no computable ground-truth answer, so only coverage is checkable.
RESEARCH_SPEC = TaskSpec(
    line_items=(),
    required_counts=(("arxiv_search", 2), ("wikipedia_search", 2),
                     ("web_search", 1), ("python", 1)),
    name="demo_research",
)


if __name__ == "__main__":       # module self-test (no corpus, no network)
    ok = [
        {"text": '[lookup_flight({"a": 1}) -> $100]'},
        {"text": '[lookup_flight({"a": 2}) -> $200]'},
        {"text": '[lookup_flight({"a": 3}) -> $300]'},
        {"text": '[lookup_flight({"a": 4}) -> $400]'},
        {"text": '[lookup_hotel({"c": "x"}) -> $10/night]'},
        {"text": '[lookup_hotel({"c": "y"}) -> $20/night]'},
        {"text": '[lookup_hotel({"c": "z"}) -> $30/night]'},
        {"text": '[get_weather({"c": "x"}) -> sunny]'},
        {"text": "The grand total is $1120 USD."},          # 1000 + 2*60
    ]
    r = verify(ok, BOOKING_SPEC)
    assert not r.failed, r.findings
    assert r.recomputed_total == 1120.0

    bad_sum = list(ok[:-1]) + [{"text": "The grand total is $4935 USD."}]
    assert verify(bad_sum, BOOKING_SPEC).failed

    dropped = [s for s in ok[:-1] if '"a": 4' not in s["text"]]
    dropped.append({"text": "The grand total is $720 USD."})   # self-consistent
    r2 = verify(dropped, BOOKING_SPEC)
    assert any(f.check == "required_coverage" for f in r2.findings), r2.findings
    assert not any(f.check == "total_consistency" for f in r2.findings), \
        "a dropped leg the agent then totals consistently is a COVERAGE miss"

    # A re-queried item is priced once, not twice (measured false positive).
    requeried = list(ok[:-1]) + [
        {"text": '[lookup_hotel({"c": "z"}) -> $30/night]'},   # same as before
        {"text": "The grand total is $1120 USD."},
    ]
    r3 = verify(requeried, BOOKING_SPEC)
    assert not r3.failed, r3.findings
    assert r3.recomputed_total == 1120.0

    # required_counts: a task that names how many times a tool must be used.
    two_arxiv = [{"text": '[arxiv_search({"q": "a"}) -> paper A]'},
                 {"text": '[arxiv_search({"q": "b"}) -> paper B]'},
                 {"text": '[wikipedia_search({"q": "a"}) -> page A]'},
                 {"text": '[wikipedia_search({"q": "b"}) -> page B]'},
                 {"text": '[web_search({"q": "cmp"}) -> result]'},
                 {"text": '[python({"code": "print(2)"}) -> 2]'},
                 {"text": "Done."}]
    assert not required_coverage(two_arxiv, RESEARCH_SPEC)
    one_arxiv = two_arxiv[1:]
    f = required_coverage(one_arxiv, RESEARCH_SPEC)
    assert f and "arxiv_search" in f[0].detail, f

    # A tool result the agent never received cannot be invented by the check.
    assert first_price("Error: 429 rate limited") is None
    assert stated_total("no money here") is None

    # --- preconditions -------------------------------------------------
    from derail.preconditions import UnsupportedInputError

    # A priced spec that observed no price is not a pass.
    no_prices = [{"text": '[get_weather({"c": "x"}) -> sunny]'},
                 {"text": "The grand total is $1120 USD."}]
    r4 = total_consistency(no_prices, BOOKING_SPEC)
    assert not r4.findings and not r4.checked, r4
    assert "nothing to reconcile" in r4.unverifiable

    # Nor is a spec that prices nothing at all — RESEARCH_SPEC is coverage-only
    # and must not be readable as "the totals check passed".
    r5 = total_consistency(two_arxiv, RESEARCH_SPEC)
    assert not r5.findings and not r5.checked, r5

    # A real check that ran reports so.
    assert verify(ok, BOOKING_SPEC).checked

    # Money this reader cannot see is refused, not read as absent.
    euro = list(ok[:-1]) + [{"text": "The grand total is €1120."}]
    try:
        verify(euro, BOOKING_SPEC)
    except UnsupportedInputError as exc:
        assert "1120" in str(exc)
    else:
        raise AssertionError("a euro total must be refused, not read as none")
    import dataclasses
    blind = dataclasses.replace(BOOKING_SPEC, strict_currency=False)
    r6 = verify(euro, blind)
    assert not r6.failed and r6.stated == 1120.0, r6.findings
    # ^ what the guard is FOR: blind, the reader takes €1120 for $1120, finds
    #   it equals the dollars the tools returned, and passes the run.

    # A euro task declares its dialect — currency, total wording, and the
    # result shapes its tools return — and is checked properly in it.
    eur_spec = dataclasses.replace(
        BOOKING_SPEC, currency="€", total_labels=("montant total", "total"),
        result_contracts=(("lookup_flight", r"^€\d+(\.\d+)?$"),
                          ("lookup_hotel", r"^€\d+(\.\d+)?/night$")))
    eur_ok = [{"text": '[lookup_flight({"a": 1}) -> €100]'},
              {"text": '[lookup_flight({"a": 2}) -> €200]'},
              {"text": '[lookup_flight({"a": 3}) -> €300]'},
              {"text": '[lookup_flight({"a": 4}) -> €400]'},
              {"text": '[lookup_hotel({"c": "x"}) -> €10/night]'},
              {"text": '[lookup_hotel({"c": "y"}) -> €20/night]'},
              {"text": '[lookup_hotel({"c": "z"}) -> €30/night]'},
              {"text": '[get_weather({"c": "x"}) -> beau]'},
              {"text": "Le montant total est €1120."}]
    r7 = verify(eur_ok, eur_spec)
    assert not r7.failed, r7.findings
    assert r7.recomputed_total == 1120.0 and r7.stated == 1120.0
    eur_bad = list(eur_ok[:-1]) + [{"text": "Le montant total est €4935."}]
    assert verify(eur_bad, eur_spec).failed, "a wrong euro total must fail"
    try:
        verify(ok, eur_spec)          # the dollar run, read by the euro spec
    except UnsupportedInputError:
        pass
    else:
        raise AssertionError("a euro spec must refuse dollar figures")
    print("PASS: verify.checks self-test")
