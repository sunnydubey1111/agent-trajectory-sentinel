"""Preconditions the deterministic readers assert before they answer.

Every reader in this repo that parses agent or tool text understands a
specific dialect: US-dollar figures, English labels, English error prefixes.
Fed something outside that dialect, a reader that does not check returns its
CLEAN value —
an ungrounded euro figure read as "all grounded", a JSON error result read as
"the tool succeeded", no observed price read as "the total checks out". A
reader that answers on input it cannot read is worse than one that fails,
because the answer is indistinguishable from a real pass, and the headline
"0 observed false positives" is made of exactly these answers.

So each reader asserts its own preconditions and refuses. The refusal is loud
(`UnsupportedInputError`) where a wrong answer would be silent, and explicit
(a recorded `unverifiable` reason) where the check legitimately has nothing to
do. No committed trace carries a non-dollar figure, so these guards are for
input this project has not been run on yet, not for input it already has.
"""
from __future__ import annotations

import re

__all__ = ["UnsupportedInputError", "unsupported_currency",
           "require_readable_money", "error_shaped", "currency_tokens",
           "DEFAULT_CURRENCY"]


class UnsupportedInputError(ValueError):
    """A reader was handed input outside the dialect it can parse.

    Raised instead of returning the clean value. Callers replaying a corpus
    that legitimately contains such input opt out per reader, which records
    the choice rather than hiding it.
    """


#: Currency symbol -> the ISO code that means the same thing, so a reader
#: configured for one marker also accepts the other spelling of it.
_SYMBOL_CODE = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
                "₹": "INR", "₽": "RUB", "₩": "KRW"}
_CODES = tuple(sorted(set(_SYMBOL_CODE.values()) |
                      {"CNY", "CHF", "CAD", "AUD", "SEK", "NOK", "BRL",
                       "MXN", "ZAR"}))

#: What the repo's readers understand unless told otherwise.
DEFAULT_CURRENCY = "$"

#: A monetary figure with its currency marker: a symbol before the number
#: (€100) or an ISO code after it (100 EUR). Deliberately narrow — it must look
#: like MONEY, so a bare "100" or a stray "£" in prose does not match.
_MONEY_TOKEN = re.compile(
    r"(?P<sym>[$€£¥₹₽₩])\s?-?\d[\d,]*(?:\.\d+)?"
    r"|(?<![A-Za-z0-9])-?\d[\d,]*(?:\.\d+)?\s?(?P<code>" + "|".join(_CODES) +
    r")(?![A-Za-z])",
    re.I)


def currency_tokens(symbol: str = DEFAULT_CURRENCY) -> frozenset[str]:
    """Both spellings of one currency: its symbol and its ISO code."""
    sym = symbol.strip()
    code = _SYMBOL_CODE.get(sym, sym.upper())
    return frozenset({sym, code})

#: Machine-shaped error markers a tool result can carry other than the English
#: word "error" at position 0. Restricted to forms that cannot be ordinary
#: content: a JSON error envelope, a Python traceback or exception class, an
#: HTTP failure status, an HTML error page. Fuzzy English ("failed",
#: "not found") is EXCLUDED on purpose — `lookup_flight` declares
#: "No route found" as a VALID result, so widening this to prose would
#: reclassify successful tool calls as errors.
_ERROR_SHAPED = re.compile(
    r'^\s*(?:'
    r'\{\s*"error"\s*:'                 # {"error": ...}
    r'|Traceback \(most recent call last\)'
    r'|[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\s*:'   # NameError: ...
    r'|HTTP\s+[45]\d\d\b'
    r'|[45]\d\d\s+[A-Z][a-z]'           # 500 Internal Server Error
    r'|<html\b|<!DOCTYPE html'
    r')')


def unsupported_currency(text: str,
                         allowed: str | frozenset[str] = DEFAULT_CURRENCY
                         ) -> str | None:
    """The first monetary figure in `text` the reader cannot read, or None.

    `allowed` is a currency symbol, or the token set `currency_tokens` builds.
    Returns the offending substring rather than a bool, so the refusal can name
    what it could not read.
    """
    ok = currency_tokens(allowed) if isinstance(allowed, str) else allowed
    ok = frozenset(t.upper() for t in ok)
    for m in _MONEY_TOKEN.finditer(text or ""):
        marker = (m.group("sym") or m.group("code") or "").upper()
        if marker not in ok:
            return m.group(0)
    return None


def require_readable_money(text: str, where: str,
                           allowed: str | frozenset[str] = DEFAULT_CURRENCY
                           ) -> None:
    """Refuse `text` if it prices anything in a currency this reader ignores.

    A figure the reader does not match is not read as unpriced — it is read as
    absent, which every downstream check treats as evidence of correctness.
    """
    bad = unsupported_currency(text, allowed)
    if bad is not None:
        ok = currency_tokens(allowed) if isinstance(allowed, str) else allowed
        raise UnsupportedInputError(
            f"{where}: cannot read {bad!r} — this reader understands "
            f"{'/'.join(sorted(ok))} figures only, and would otherwise report "
            f"a clean result by ignoring it")


def error_shaped(result: str) -> bool:
    """Did this tool result fail, judged from its text alone?

    The single definition of that question, shared by `telemetry.events` and
    `telemetry.adapter` so the two cannot drift. An English prefix test alone
    reads a `{"error": ...}` envelope, a traceback and an HTTP 500 as
    SUCCESSES, and feeds that to `IDX_ERROR_FLAG` and `IDX_TOOL_SUCCESS`.

    Structured `tool_events` carry a real `is_error` field and must use it
    instead; this is only for results recovered from rendered text.
    """
    r = (result or "").lstrip()
    return r[:5].lower() == "error" or bool(_ERROR_SHAPED.match(r))


if __name__ == "__main__":       # module self-test
    assert unsupported_currency("total $100") is None
    assert unsupported_currency("100 USD") is None
    assert unsupported_currency("total €100") == "€100"
    assert unsupported_currency("costs 250 EUR") == "250 EUR"
    assert unsupported_currency("£1,299.50 per night") == "£1,299.50"
    assert unsupported_currency("a EUROPEAN trip, 4 legs") is None, \
        "'EUR' inside a word must not trip the guard"
    assert unsupported_currency("section 100 EURekas") is None

    # A reader configured for another currency accepts it and refuses dollars.
    assert unsupported_currency("total €100", "€") is None
    assert unsupported_currency("total 100 EUR", "€") is None
    assert unsupported_currency("total $100", "€") == "$100"
    assert currency_tokens("$") == frozenset({"$", "USD"})
    assert currency_tokens("€") == frozenset({"€", "EUR"})

    assert error_shaped("Error: boom")
    assert error_shaped("  error: boom")
    assert error_shaped('{"error": "rate limited"}')
    assert error_shaped("NameError: name 'x' is not defined")
    assert error_shaped("Traceback (most recent call last):\n  File ...")
    assert error_shaped("HTTP 503 Service Unavailable")
    assert error_shaped("500 Internal Server Error")
    assert not error_shaped("$402")
    assert not error_shaped("No route found"), "a DECLARED result, not an error"
    assert not error_shaped("Unknown city."), "a DECLARED result, not an error"
    assert not error_shaped("")
    # The "error" PREFIX half keeps its known looseness: a result opening on
    # the word reads as an error whatever follows. Tightening it would
    # reclassify existing corpora, which is a separate question from the
    # silent passes this module addresses.
    assert error_shaped("Errors in the paper are discussed in section 4")

    try:
        require_readable_money("the total is €400", "test")
    except UnsupportedInputError as exc:
        assert "€400" in str(exc)
    else:
        raise AssertionError("must refuse a euro figure")

    print("PASS preconditions smoke: foreign money detected, error shapes "
          "recognised, declared results left alone.")
