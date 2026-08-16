"""Properties of the deterministic checks that are easy to lose silently.

Each test here pins a way the checks could go on returning a verdict while
quietly answering a weaker question than the one they were asked.
"""
from __future__ import annotations

import re

import pytest

from derail.verify.checks import (BOOKING_SPEC, LineItem, TaskSpec,
                                  _events_per_step, _selection_matches,
                                  tolerance_for, tool_contract,
                                  total_consistency, verify)


# ------------------------------------------------- structured events win
def test_an_empty_tool_events_list_is_a_report_of_no_calls():
    """A collector that wrote the key is reporting what it executed.

    Falling back to the text parser on an empty list is what lets a model
    that merely WROTE tool-like syntax have it read as an executed call.
    """
    step = {"text": '[lookup_flight({"a": 1}) -> $100]', "tool_events": []}
    assert _events_per_step([step]) == [[]]


def test_an_absent_tool_events_key_still_falls_back_to_the_text():
    """Corpora collected before the structured form must keep working."""
    step = {"text": '[lookup_flight({"a": 1}) -> $100]'}
    evs = _events_per_step([step])[0]
    assert [e.name for e in evs] == ["lookup_flight"]
    assert evs[0].source == "text"


def test_events_carry_a_real_has_result_not_a_pinned_true():
    """`tool_contract` skips an event with no result; pinning has_result True
    made it match a contract against the empty string instead."""
    step = {"text": "", "tool_events": [
        {"name": "lookup_flight", "args": {}, "result": None,
         "is_error": False}]}
    assert _events_per_step([step])[0][0].has_result is False


# --------------------------------------------------------- tool_contract
def test_a_contract_must_match_the_whole_result():
    """A malformed result that merely STARTS legally is still malformed."""
    step = {"text": "", "tool_events": [
        {"name": "lookup_flight", "args": {}, "result": "$100 and then junk",
         "is_error": False}]}
    assert [f.terse for f in tool_contract([step], BOOKING_SPEC)] == [
        "lookup_flight returned a malformed result"]


@pytest.mark.parametrize("tool,result", [
    ("lookup_flight", "$100"),
    ("lookup_flight", "No route found between Osaka and Lima"),
    ("lookup_hotel", "$120.50/night"),
    ("lookup_hotel", "Unknown city."),
    ("search_catalog", "$71.7"),
    ("search_catalog", "Item not found."),
    ("calculator", "255.798"),
    ("calculator", "-40"),
    ("calculator", "Error: division by zero"),
])
def test_every_declared_outcome_is_a_complete_shape(tool, result):
    """Each alternative must fullmatch, or the check false-positives on a
    legal outcome the tools really return."""
    step = {"text": "", "tool_events": [
        {"name": tool, "args": {}, "result": result, "is_error": False}]}
    assert tool_contract([step], BOOKING_SPEC) == []


# ------------------------------------------------------------- tolerance
def test_tolerance_is_measured_against_the_candidate_not_the_grand_total():
    """An unused expensive lookup must not widen the window for a candidate.

    The agent chooses what to look up, so a tolerance scaled by the sum of
    everything observed is a window the agent can widen at will.
    """
    spec = TaskSpec(line_items=(LineItem("f", multiplier=1, required_calls=1),),
                    tolerance=0.10, tolerance_floor=0.0)
    # Candidate selections are 100 and 5000. A stated 118 is 18 away from 100,
    # outside 10% of 100 but inside 10% of the 5100 grand total.
    steps = [{"text": '[f({"a": 1}) -> $100]'},
             {"text": '[f({"a": 2}) -> $5000]'},
             {"text": "Total $118."}]
    res = total_consistency(steps, spec)
    assert res.failed, "a 5000 lookup the agent did not use widened the window"


def test_tolerance_floor_and_pivot_are_task_facts_not_constants():
    spec = TaskSpec(line_items=(), tolerance=0.01,
                    tolerance_floor=2.0, tolerance_pivot=1000.0)
    assert tolerance_for(spec, 1.0) == 10.0      # pivot dominates
    assert tolerance_for(spec, 0.0) == 10.0
    tight = TaskSpec(line_items=(), tolerance=0.01,
                     tolerance_floor=0.0, tolerance_pivot=1.0)
    assert tolerance_for(tight, 100.0) == pytest.approx(1.0)


# ------------------------------------------- bounded combinatorial search
def test_an_undecidable_search_is_unverifiable_not_clean():
    """Past the cap the check must say so, not return a pass."""
    spec = TaskSpec(line_items=(LineItem("f", multiplier=1, required_calls=8),),
                    max_selection_candidates=10)
    steps = [{"text": f'[f({{"a": {i}}}) -> ${100 + i}]'} for i in range(20)]
    steps.append({"text": "Total $900."})
    res = total_consistency(steps, spec)
    assert not res.checked and "candidate selections" in res.unverifiable
    assert not res.failed, "undecided must not be reported as a failure"


def test_the_booking_spec_stays_well_inside_its_own_cap():
    steps = [{"text": f'[lookup_flight({{"a": {i}}}) -> ${100 + i}]'}
             for i in range(6)]
    steps += [{"text": f'[lookup_hotel({{"c": "{c}"}}) -> $50/night]'}
              for c in "xyz"]
    steps.append({"text": "Total $1000."})
    assert total_consistency(steps, BOOKING_SPEC).checked


def test_selection_search_reports_undecided_rather_than_guessing():
    spec = TaskSpec(line_items=(LineItem("f", multiplier=1, required_calls=5),),
                    max_selection_candidates=5)
    assert _selection_matches({"f": [1.0] * 20},
                              {"f": spec.line_items[0]}, 5.0, spec) is None


# ----------------------------------------------- the checks still verify
def test_a_correct_booking_run_still_passes_end_to_end():
    steps = [{"text": f'[lookup_flight({{"leg": {i}}}) -> $100]'}
             for i in range(4)]
    steps += [{"text": f'[lookup_hotel({{"c": "{c}"}}) -> $50/night]'}
              for c in "xyz"]
    steps.append({"text": '[get_weather({"c": "x"}) -> sunny]'})
    steps.append({"text": '[calculator({"e": "1"}) -> 700]'})
    steps.append({"text": "Grand total $700."})
    res = verify(steps, BOOKING_SPEC)
    assert not res.failed, [f.detail for f in res.findings]
    assert res.checked


def test_contract_patterns_are_authored_as_complete_shapes():
    """A pattern ending in a bare prefix would false-positive under fullmatch."""
    for tool, pattern in BOOKING_SPEC.result_contracts:
        for alt in re.split(r"(?<!\\)\|", pattern):
            assert not alt.startswith("^") and not alt.endswith("$"), (
                f"{tool}: {alt!r} carries anchors, but fullmatch already "
                f"anchors; the mixed form is how a prefix alternative hides")
