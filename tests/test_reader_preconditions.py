"""The deterministic readers must refuse input they cannot read.

Each reader here used to answer its CLEAN value on input outside the dialect
it parses — an unreadable currency read as "no money mentioned", a JSON error
envelope read as "the tool succeeded", no observed price read as "the total
reconciles". A clean answer on unread input is indistinguishable from a real
pass, which is what the "0 observed false positives" headline is made of.

These tests pin the refusals, and equally pin what must NOT be refused: the
committed corpora are dollar-denominated English, so every guard here is a
no-op on them and no published number may move because of one.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from conftest import REPO_ROOT
from derail.common import IDX_TOOL_SUCCESS
from derail.monitor.grounding_verify import (NumericGroundingMonitor,
                                             grounded_set)
from derail.preconditions import (UnsupportedInputError, error_shaped,
                                  require_readable_money,
                                  unsupported_currency)
from derail.telemetry import adapter
from derail.telemetry.events import parse_tool_bits
from derail.verify.checks import (BOOKING_SPEC, RESEARCH_SPEC, first_price,
                                  stated_total, total_consistency, verify)


# --------------------------------------------------------------- currency
@pytest.mark.parametrize("text", ["total $100", "4935 USD", "no money here",
                                  "", "a EUROPEAN tour", "eur is a prefix",
                                  "section 100 EURekas", "flight 100"])
def test_dollar_and_english_text_is_never_refused(text):
    assert unsupported_currency(text) is None
    require_readable_money(text, "test")


@pytest.mark.parametrize("text,found", [
    ("the hotel is €150/night", "€150"),
    ("£1,299.50 total", "£1,299.50"),
    ("costs 250 EUR", "250 EUR"),
    ("¥12000", "¥12000"),
    ("total 99.50 GBP", "99.50 GBP"),
])
def test_a_figure_in_an_unreadable_currency_is_named_not_ignored(text, found):
    assert unsupported_currency(text) == found
    with pytest.raises(UnsupportedInputError) as exc:
        require_readable_money(text, "test")
    assert found in str(exc.value)


def test_grounding_monitor_refuses_rather_than_clearing_a_foreign_figure():
    """The fabrication the $-only regex cannot see must not read as grounded."""
    m = NumericGroundingMonitor()
    m.start_episode()
    m.observe_tool_results("[lookup_hotel -> $100/night]")
    assert m.check_step("Hotel is $100/night.") == []
    with pytest.raises(UnsupportedInputError):
        m.check_step("Hotel in Prague is €150/night.")


def test_the_blind_reading_is_available_but_must_be_asked_for():
    lenient = NumericGroundingMonitor(strict=False)
    lenient.start_episode()
    lenient.observe_tool_results("[lookup_hotel -> $100/night]")
    assert lenient.check_step("Hotel in Prague is €150/night.") == []


# ------------------------------------------------- task parameters, not code
def test_the_demo_task_values_are_the_defaults():
    """Parameterising them must not move the shipped monitor."""
    m = NumericGroundingMonitor()
    assert m.currency == "$"
    assert m.nights_cap == 4
    assert m.tax_rates == (0.08, 0.085, 0.10)
    assert m.subset_multiples == (1, 2)
    assert grounded_set([100.0, 250.0]) == grounded_set(
        [100.0, 250.0], nights_cap=4, tax_rates=(0.08, 0.085, 0.10),
        subset_multiples=(1, 2))


def test_another_task_supplies_its_own_four_without_touching_the_module():
    """Three-night stays, VAT and euros, all through the constructor."""
    m = NumericGroundingMonitor(currency="€", nights_cap=3,
                                tax_rates=(0.21,), subset_multiples=(1, 3))
    m.start_episode()
    m.observe_tool_results("[lookup_hotel -> €100/night]")
    assert m.check_step("Three nights €300.") == [], "3 x 100 grounded"
    assert m.check_step("With VAT €363.") == [], "300 x 1.21 grounded"
    assert m.check_step("Prague is €150/night.") == [150.0], "not derivable"


def test_a_reader_configured_for_euros_refuses_dollars():
    m = NumericGroundingMonitor(currency="€")
    m.start_episode()
    m.observe_tool_results("[lookup_hotel -> €100/night]")
    with pytest.raises(UnsupportedInputError):
        m.check_step("Actually $150.")


def test_the_nights_cap_really_bounds_the_multiples():
    tight = NumericGroundingMonitor(nights_cap=1, subset_multiples=(1,))
    tight.start_episode()
    tight.observe_tool_results("[lookup_hotel -> $100/night]")
    assert tight.check_step("Two nights $200.") == [200.0], \
        "nights_cap=1 must not ground a 2x multiple"
    default = NumericGroundingMonitor()
    default.start_episode()
    default.observe_tool_results("[lookup_hotel -> $100/night]")
    assert default.check_step("Two nights $200.") == [], \
        "the demo default does ground it"


def test_a_task_spec_declares_its_own_currency_and_total_wording():
    """The verification parser's dialect is a spec field, not a constant."""
    eur = dataclasses.replace(
        BOOKING_SPEC, currency="€", total_labels=("montant total", "total"),
        result_contracts=(("lookup_flight", r"^€\d+(\.\d+)?$"),
                          ("lookup_hotel", r"^€\d+(\.\d+)?/night$")))
    steps = [{"text": '[lookup_flight({"a": %d}) -> €%d]' % (i, i * 100)}
             for i in (1, 2, 3, 4)]
    steps += [{"text": '[lookup_hotel({"c": "%s"}) -> €%d/night]' % (c, v)}
              for c, v in (("x", 10), ("y", 20), ("z", 30))]
    steps += [{"text": '[get_weather({"c": "x"}) -> beau]'},
              {"text": "Le montant total est €1120."}]
    res = verify(steps, eur)
    assert not res.failed, res.findings
    assert res.recomputed_total == 1120.0 and res.stated == 1120.0
    assert res.checked

    wrong = list(steps[:-1]) + [{"text": "Le montant total est €4935."}]
    assert verify(wrong, eur).failed, "a wrong euro total must still fail"


def test_a_euro_spec_refuses_a_dollar_run():
    eur = dataclasses.replace(BOOKING_SPEC, currency="€")
    with pytest.raises(UnsupportedInputError):
        verify([{"text": '[lookup_flight({"a": 1}) -> $100]'},
                {"text": "Total $100."}], eur)


def test_the_tax_rates_really_bound_what_a_total_may_carry():
    us = NumericGroundingMonitor()
    us.start_episode()
    us.observe_tool_results("[lookup_flight -> $100]")
    assert us.check_step("With tax $108.") == [], "0.08 is a declared rate"
    assert us.check_step("With VAT $121.") == [121.0], "0.21 is not"


def test_verify_refuses_a_total_stated_in_an_unreadable_currency():
    steps = [{"text": '[lookup_flight({"a": 1}) -> $100]'},
             {"text": "The grand total is €100."}]
    with pytest.raises(UnsupportedInputError):
        verify(steps, BOOKING_SPEC)


def test_blind_the_reader_takes_a_euro_total_for_a_dollar_one():
    """Why the guard exists, stated as a test rather than a comment."""
    steps = [{"text": '[lookup_flight({"a": 1}) -> $100]'},
             {"text": "The grand total is €100."}]
    blind = dataclasses.replace(BOOKING_SPEC, strict_currency=False)
    res = total_consistency(steps, blind)
    assert res.stated == 100.0, "€100 was read as $100"
    assert not res.findings, "and therefore reconciled against the dollars"


def test_the_readers_take_strict_off_only_when_told():
    with pytest.raises(UnsupportedInputError):
        first_price("costs 250 EUR")
    with pytest.raises(UnsupportedInputError):
        stated_total("total 250 EUR")
    # Blind, the two fail in opposite directions and both are wrong: the price
    # vanishes, so the run looks unpriced, while the stated total is read as
    # 250 DOLLARS. Pinned so the guard's value is not mistaken for pedantry.
    assert first_price("costs 250 EUR", strict=False) is None
    assert stated_total("total 250 EUR", strict=False) == 250.0


# ------------------------------------------------------- no evidence read
def test_no_priced_result_observed_is_unverifiable_not_a_pass():
    steps = [{"text": '[get_weather({"c": "x"}) -> sunny]'},
             {"text": "The grand total is $1120 USD."}]
    res = total_consistency(steps, BOOKING_SPEC)
    assert not res.findings
    assert not res.checked
    assert "nothing to reconcile" in res.unverifiable


def test_a_spec_that_prices_nothing_reports_not_applicable():
    steps = [{"text": '[arxiv_search({"q": "a"}) -> paper A]'},
             {"text": "Done."}]
    res = total_consistency(steps, RESEARCH_SPEC)
    assert not res.findings and not res.checked


def test_a_check_that_read_real_evidence_says_so():
    steps = [{"text": '[lookup_flight({"a": 1}) -> $100]'},
             {"text": '[lookup_flight({"a": 2}) -> $200]'},
             {"text": '[lookup_flight({"a": 3}) -> $300]'},
             {"text": '[lookup_flight({"a": 4}) -> $400]'},
             {"text": '[lookup_hotel({"c": "x"}) -> $10/night]'},
             {"text": '[lookup_hotel({"c": "y"}) -> $20/night]'},
             {"text": '[lookup_hotel({"c": "z"}) -> $30/night]'},
             {"text": '[get_weather({"c": "x"}) -> sunny]'},
             {"text": "The grand total is $1120 USD."}]
    res = verify(steps, BOOKING_SPEC)
    assert res.checked and not res.failed


# ------------------------------------------------------------ error shape
@pytest.mark.parametrize("result", [
    'Error: 429 rate limited',
    '   error: boom',
    '{"error": "rate limited"}',
    "NameError: name 'papers' is not defined",
    "Traceback (most recent call last):\n  File x",
    "HTTP 503 Service Unavailable",
    "500 Internal Server Error",
])
def test_a_failed_tool_result_is_read_as_a_failure(result):
    assert error_shaped(result)


@pytest.mark.parametrize("result", [
    "$402", "No route found", "Unknown city.", "sunny, 21C", "",
    "Item not found.", '{"title": "Echo State Networks"}', "42",
])
def test_a_successful_result_is_never_read_as_a_failure(result):
    """`lookup_flight` DECLARES 'No route found' valid; widening the reader to
    English prose would reclassify successful calls as errors."""
    assert not error_shaped(result)


def test_both_readers_share_one_definition():
    """`events` and `adapter` each carried their own copy of this rule."""
    import derail.telemetry.adapter as ad
    import derail.telemetry.events as ev
    assert ev.error_shaped is error_shaped
    assert ad.error_shaped is error_shaped


def test_a_json_error_envelope_no_longer_counts_as_a_successful_call():
    step = {"text": '[search({"q": "x"}) -> {"error": "quota exceeded"}]',
            "token_logprobs": [-0.1] * 12, "action": "tool_call",
            "latency_s": 1.0, "output_tokens": 12, "error": False}
    events, _ = parse_tool_bits(step["text"])
    assert len(events) == 1 and events[0].is_error
    ep = adapter.episode_from_trace([step, dict(step, text="done")], "e",
                                    extended=True,
                                    use_sentence_transformers=False)
    assert ep.X[0, IDX_TOOL_SUCCESS] == 0.0


# ------------------------------------------------- no committed data moves
def test_the_currency_guard_is_a_no_op_on_every_committed_trace():
    """No committed trace prices anything the $-only readers cannot see.

    Pins that, so a corpus added later cannot introduce one unnoticed and
    silently change what the guarded readers do to the published corpora.
    """
    traces = REPO_ROOT / "traces"
    offenders = []
    for path in sorted(traces.rglob("*.jsonl")):
        for line in path.read_text("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                step = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(step, dict):
                continue
            bad = unsupported_currency(str(step.get("text", "")))
            if bad:
                offenders.append(f"{path.relative_to(traces)}: {bad}")
    assert not offenders, ("committed traces now carry money the $-only "
                           f"readers cannot see: {offenders[:5]}")
