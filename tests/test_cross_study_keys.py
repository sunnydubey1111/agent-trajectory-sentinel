"""Can two studies be joined on the episodes they describe?

The per-study gates each check a number against its own file. None of them
checks that two files talk about the same episodes, and that gap let four
verification tables ship carrying the *same* 120 episode ids — including one
measured on a different model — with nothing in any row to tell them apart.

Every corpus numbers its episodes from zero, so `episode_id` is unique only
within a corpus. The key that survives a cross-study join is
`(dataset, episode_id)`, plus whatever a long-format table varies down its
rows. These tests hold that property for the committed tables.
"""
from __future__ import annotations

import itertools
import pathlib

import pandas as pd
import pytest

TABLES = pathlib.Path(__file__).resolve().parents[1] / "results" / "tables"

#: Columns a long-format table legitimately repeats an episode over. A table
#: absent from this map must be one row per (dataset, episode_id).
SUB_KEYS = {
    "repair_policies.csv": ("rung", "rep"),
    "organic_validation.csv": ("monitor",),
    "judge_calibration.csv": ("t",),
}

#: Tables that predate the dataset column and are not episode-level despite
#: carrying an id column. Empty on purpose: every exemption here is a study
#: whose results cannot be attributed to a corpus, so the list must stay empty.
EXEMPT: frozenset[str] = frozenset()


def _episode_tables() -> list[pathlib.Path]:
    return sorted(p for p in TABLES.glob("*.csv")
                  if "episode_id" in pd.read_csv(p, nrows=0).columns
                  and p.name not in EXEMPT)


def test_there_are_episode_level_tables_to_check():
    # Guards against the glob silently matching nothing and the two tests
    # below passing vacuously.
    assert len(_episode_tables()) >= 25


@pytest.mark.parametrize("path", _episode_tables(), ids=lambda p: p.name)
def test_every_episode_level_table_names_its_corpus(path):
    cols = pd.read_csv(path, nrows=0).columns
    assert "dataset" in cols, (
        f"{path.name} has episode_id but no dataset column, so its rows "
        f"cannot be attributed to a corpus and cannot be joined to another "
        f"study's rows")


@pytest.mark.parametrize("path", _episode_tables(), ids=lambda p: p.name)
def test_dataset_and_episode_id_identify_one_row(path):
    df = pd.read_csv(path)
    key = ["dataset", "episode_id", *SUB_KEYS.get(path.name, ())]
    missing = [c for c in key if c not in df.columns]
    assert not missing, f"{path.name} lacks declared key column(s) {missing}"
    dupes = df[df.duplicated(subset=key, keep=False)]
    assert dupes.empty, (
        f"{path.name}: {len(dupes)} rows share a key {tuple(key)}. Either the "
        f"study repeats an episode, or the column it varies over is missing "
        f"from SUB_KEYS in this test.")


#: Table pairs that score the SAME episodes from the same objective labeller.
#: Their `label` columns are one measurement, so a disagreement means one of
#: the two was not regenerated when the labeller changed.
LABEL_PAIRS = [
    ("provoked_fabrication.csv", "verification_provoked.csv"),
    ("fabrication_organic_demo7b_ext.csv", "organic_hallucination_ext.csv"),
    ("fabrication_organic_demo7b.csv", "organic_hallucination.csv"),
    ("verification_cold.csv", "organic_hallucination_cold.csv"),
    ("verification_holdout.csv", "organic_hallucination_holdout.csv"),
]


@pytest.mark.parametrize("left,right", LABEL_PAIRS,
                         ids=lambda s: s.replace(".csv", ""))
def test_studies_over_the_same_episodes_agree_on_their_labels(left, right):
    """Two studies, one corpus, one labeller — so one set of labels.

    This is the check nothing in the repo performed. `provoked_fabrication`
    shipped with pre-`incomplete` labels while `verification_provoked` carried
    the current ones: 7 of the same 120 episodes were healthy in one table and
    incomplete in the other.
    """
    a = pd.read_csv(TABLES / left)[["dataset", "episode_id", "label"]]
    b = pd.read_csv(TABLES / right)[["dataset", "episode_id", "label"]]
    m = a.merge(b, on=["dataset", "episode_id"], suffixes=("_l", "_r"))
    assert len(m) == len(a) == len(b), (
        f"{left} and {right} cover different episode sets "
        f"({len(a)} / {len(b)} rows, {len(m)} joined)")
    bad = m[m.label_l != m.label_r]
    assert bad.empty, (
        f"{left} and {right} disagree on {len(bad)} episodes' labels, so one "
        f"of them is stale against the labeller:\n{bad.head(10).to_string()}")


def test_the_verification_arms_are_distinguishable():
    """The defect that motivated this file, pinned so it cannot return.

    verification_cold / _holdout / _provoked / _organic_llama8b_cold hold one
    row per episode of four DIFFERENT corpora, one of them a different model.
    Before the dataset column they were byte-comparable on their key columns.
    """
    names = ["verification_cold.csv", "verification_holdout.csv",
             "verification_provoked.csv",
             "verification_organic_llama8b_cold.csv"]
    keys = {}
    for n in names:
        df = pd.read_csv(TABLES / n)
        assert df["dataset"].nunique() == 1, f"{n} mixes corpora"
        keys[n] = set(zip(df["dataset"], df["episode_id"]))

    for a, b in itertools.combinations(names, 2):
        assert not keys[a] & keys[b], (
            f"{a} and {b} still share {len(keys[a] & keys[b])} keys")
