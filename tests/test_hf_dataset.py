"""The published dataset must agree with the corpus it is derived from.

`data/episodes.jsonl` is a convenience view for `load_dataset`. If it ever
disagrees with `traces/`, the published dataset silently stops being the thing
the paper's numbers came from, and nothing else in the project would notice.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from devtools import hf_dataset

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return hf_dataset.build_episodes()


def test_episode_count_matches_the_manifests(records) -> None:
    expected = sum(len(json.loads(m.read_text("utf-8")))
                   for m in hf_dataset._corpora())
    assert len(records) == expected


def test_no_imported_or_scratch_corpus_is_published(records) -> None:
    """Underscore-prefixed directories are not ours; AFTraj must never ship."""
    assert not [r for r in records if r["corpus"].startswith("_")]
    assert all(not p.parent.name.startswith("_") for p in hf_dataset._corpora())


def test_every_record_keeps_its_manifest_metadata(records) -> None:
    by_id = {r["uid"]: r for r in records}
    for manifest_path in hf_dataset._corpora():
        for entry in json.loads(manifest_path.read_text("utf-8")):
            record = by_id[f"{manifest_path.parent.name}/{entry['episode_id']}"]
            assert record["T"] == entry["T"]
            assert record["tau"] == entry["tau"]
            assert record["failure_class"] == entry["failure_class"]
            assert record["has_logprobs"] == entry["has_logprobs"]
            assert record["corpus"] == manifest_path.parent.name


def test_step_count_matches_the_declared_length(records) -> None:
    wrong = [r["episode_id"] for r in records if len(r["steps"]) != r["T"]]
    assert not wrong, f"steps disagree with T for {wrong[:5]}"


def test_steps_are_the_trace_file_verbatim(records) -> None:
    """Spot-check both ends of the corpus rather than re-reading 2,823 files."""
    by_id = {r["uid"]: r for r in records}
    for manifest_path in hf_dataset._corpora():
        entries = json.loads(manifest_path.read_text("utf-8"))
        for entry in (entries[0], entries[-1]):
            raw = (manifest_path.parent / entry["file"]).read_text("utf-8")
            steps = [json.loads(x) for x in raw.splitlines() if x.strip()]
            uid = f"{manifest_path.parent.name}/{entry['episode_id']}"
            assert by_id[uid]["steps"] == steps


def test_the_file_key_is_dropped_but_nothing_else_is(records) -> None:
    """`file` is a repo path and is meaningless once flattened; the rest stays."""
    manifest_path = hf_dataset._corpora()[0]
    entry = json.loads(manifest_path.read_text("utf-8"))[0]
    record = next(r for r in records if r["episode_id"] == entry["episode_id"])
    assert "file" not in record
    assert set(entry) - {"file"} <= set(record)


def _flat(text: str) -> str:
    """Card text with wrapping and blockquote markers removed, for matching."""
    return " ".join(text.replace("\n>", "\n").split())


def test_card_states_the_gemini_terms_and_the_llama_notice() -> None:
    """The card is what a downloader reads; the obligations must survive it."""
    text = _flat(hf_dataset.card(2823, 25))
    assert "Built with Llama" in text
    assert "Acceptable Use" in text, "the Llama AUP must travel with the data"
    assert "develop models that compete with the Services" in text, (
        "Google's clause must be quoted, not paraphrased")
    assert "license: other" in text, "claiming plain MIT would be false"
    for section in ("## Licensing", "## Loading", "## Fields"):
        assert section in text, section


def test_uid_is_unique_and_episode_id_is_not() -> None:
    """Pins the reason `uid` exists, so nobody 'simplifies' it away later."""
    records = hf_dataset.build_episodes()
    uids = [r["uid"] for r in records]
    assert len(set(uids)) == len(records), "uid must be the primary key"
    assert len(set(r["episode_id"] for r in records)) < len(records), (
        "episode_id collides across corpora; if this ever stops being true, "
        "the card's warning about it should be removed too")


def test_the_card_warns_that_episode_id_collides() -> None:
    text = _flat(hf_dataset.card(2823, 25))
    assert "not unique across corpora" in text
    assert "Key on `uid`" in text


@pytest.mark.parametrize("planted,label", [
    ("sk-abcdefghijklmnopqrstuvwxyz012345", "OpenAI-style key"),
    ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "GitHub token"),
    ("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456", "Google API key"),
    ("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "Hugging Face token"),
    ("AKIAIOSFODNN7EXAMPLE", "AWS access key id"),
    ('password = "hunter2istoshortbutthisisnot"', "credential assignment"),
])
def test_the_secret_scan_catches_planted_credentials(tmp_path, planted, label
                                                     ) -> None:
    """A publish gate that cannot fail is not a gate."""
    victim = tmp_path / "trace.jsonl"
    victim.write_text(f'{{"text": "{planted}"}}\n', encoding="utf-8")
    findings = hf_dataset.scan_for_secrets([victim])
    assert findings, f"{label} slipped through the scan"


def test_the_secret_scan_does_not_fire_on_the_real_corpus() -> None:
    """Episode ids and sha256 hashes must not read as credentials."""
    sample = sorted((REPO_ROOT / "traces").rglob("*.jsonl"))[:300]
    assert sample, "no traces found to scan"
    assert hf_dataset.scan_for_secrets(sample) == []
