"""Two assumptions that are invisible until the deployment stops matching them.

G2-7: the lexical relevance dim reads words with a fixed tokenizer and an
English stoplist. G2-8: the escalation economics price a judge call at exactly
one agent step. Neither is wrong; both are choices that a reader can mistake
for measurements, and one of them used to make a monitor dim silently inert.
"""
from __future__ import annotations

import pandas as pd
import pytest

from derail.monitor.escalation import (cost_at, cost_ratio_at,
                                       cost_ratio_break_even)
from derail.telemetry import adapter as A

TASK = "find recent papers on echo state networks"
QUERY = '{"query": "echo state networks"}'


# ----------------------------------------------------- G2-7: reading words
def test_an_on_topic_english_document_is_not_flagged():
    assert A._lex_miss(TASK, QUERY,
                       "Echo state networks are reservoir models.") == 0.0


def test_an_off_topic_english_document_is_flagged():
    assert A._lex_miss(TASK, QUERY,
                       "The 1998 World Cup final was played in Paris.") == 1.0


def test_a_cyrillic_document_can_be_judged_at_all():
    """`[a-z0-9]+` matched ASCII letters only, so a Cyrillic result tokenized
    to nothing, fell under min_words and scored 0.0 — reported as relevant
    because it could not be read. That is the failure this dim exists to
    catch, arriving as a clean bill."""
    task = "Найти статьи о сетях с эхо-состоянием"
    args = '{"query": "сети с эхо-состоянием"}'
    decoy = "Финал чемпионата мира 1998 года прошёл на стадионе в Париже."
    assert A._lex_miss(task, args, decoy) == 1.0


def test_a_cyrillic_on_topic_document_is_still_not_flagged():
    task = "Найти статьи о сетях с эхо-состоянием"
    args = '{"query": "сети эхо-состоянием"}'
    ok = "Сети с эхо-состоянием применяются для анализа временных рядов."
    assert A._lex_miss(task, args, ok) == 0.0


def test_accented_latin_survives_tokenization():
    assert A._content_words("réseaux à état d'écho") == [
        "réseaux", "état", "écho"]


def test_the_stoplist_is_a_parameter_not_a_fixture():
    """The overlap test intersects two sets built by the same filter, so a
    stoplist that misses a language's function words makes the dim UNDER-fire
    rather than false-alarm. A deployment can still supply its own."""
    french_stop = frozenset("les des une pour dans avec".split())
    assert "pour" not in A._content_words("pour les articles", french_stop)
    assert "pour" in A._content_words("pour les articles", frozenset())


def test_a_space_free_script_is_counted_rather_than_silently_ignored():
    """Japanese tokenizes to one run per phrase, never reaches min_words, and
    the dim stays 0.0. Segmenting it needs a per-language model this project
    does not carry, so the honest move is to make the silence countable."""
    before = A.lex_unreadable()
    verdict = A._lex_miss(
        "エコーステートネットワークに関する論文を探す", "{}",
        "1998年のワールドカップ決勝はパリのスタジアムで行われました。とても長い文章です。")
    assert verdict == 0.0
    assert A.lex_unreadable() == before + 1, (
        "the dim went quiet on unreadable input without recording that it "
        "could not read it")


def test_readable_input_does_not_inflate_the_unreadable_counter():
    before = A.lex_unreadable()
    A._lex_miss(TASK, QUERY, "The 1998 World Cup final was played in Paris.")
    assert A.lex_unreadable() == before


# ------------------------------------------------- G2-8: what 1.0 buys you
def test_cost_units_are_normalized_not_currency():
    from derail import common

    assert common.COST_STEP == 1.0 and common.COST_JUDGE == 1.0
    src = open(common.__file__, encoding="utf-8").read()
    assert "NORMALIZED UNITS, not money" in src, (
        "the cost constants must say what they are; a bare 1.0 reads as a "
        "measurement")


def test_re_pricing_a_recorded_row_is_exact_at_the_recorded_ratio():
    assert cost_at(58.31, 29.08, 1.0) == pytest.approx(58.31)


def test_a_more_expensive_judge_helps_the_selective_policy():
    selective, every = (35.75, 2.22), (58.31, 29.08)
    assert (cost_ratio_at(selective, every, 10.0)
            < cost_ratio_at(selective, every, 1.0))


def test_a_cheap_enough_judge_reverses_the_conclusion():
    """The point of the item: 1.0 is a choice of units, and the conclusion
    drawn at 1.0 is not scale-free."""
    selective, every = (35.75, 2.22), (58.31, 29.08)
    assert cost_ratio_at(selective, every, 1.0) < 1.0
    assert cost_ratio_at(selective, every, 0.01) > 1.0


def test_the_published_table_sits_well_above_its_own_break_even():
    """Computed from the committed table, so it cannot drift from it."""
    d = pd.read_csv("results/tables/h3_escalation.csv")
    sel = d[d["selected_on_cal"]].iloc[0]
    every = d[d["policy"] == "judge_every_step"].iloc[0]
    flip = cost_ratio_break_even(
        (float(sel["mean_cost"]), float(sel["mean_judge_calls"])),
        (float(every["mean_cost"]), float(every["mean_judge_calls"])))
    assert 0.0 < flip < 1.0
    assert flip == pytest.approx(0.16, abs=0.02)
    assert cost_ratio_at(
        (float(sel["mean_cost"]), float(sel["mean_judge_calls"])),
        (float(every["mean_cost"]), float(every["mean_judge_calls"])),
        1.0) == pytest.approx(float(sel["cost_ratio_vs_judge"]), abs=1e-6), (
        "the re-pricing helper disagrees with the column the study wrote")


def test_a_non_default_cost_ratio_cannot_reach_the_publication_path():
    """Same guard the judge overrides get: a sensitivity arm must not be able
    to land in results/ looking like a publication run."""
    import inspect

    from derail.experiments import run_experiment

    src = inspect.getsource(run_experiment.main)
    assert "AGENTWATCH_COST_STEP" in src and "AGENTWATCH_COST_JUDGE" in src
    assert "AGENTWATCH_RESULTS_ROOT" in src
