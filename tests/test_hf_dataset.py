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


def test_published_rows_have_one_uniform_schema(records) -> None:
    """The hub casts every row to one schema and fails the whole dataset if a
    later row disagrees. Corpora record different manifest fields, so this is
    the failure that actually happened once."""
    rows = hf_dataset.to_table_rows(records)
    shapes = {tuple(sorted(r)) for r in rows}
    assert len(shapes) == 1, f"{len(shapes)} different row shapes published"


def test_steps_and_metadata_round_trip(records) -> None:
    rows = {r["uid"]: r for r in hf_dataset.to_table_rows(records)}
    for record in (records[0], records[len(records) // 2], records[-1]):
        row = rows[record["uid"]]
        assert json.loads(row["steps"]) == record["steps"]
        assert row["n_steps"] == record["T"]


def test_metadata_carries_every_field_not_given_a_column(records) -> None:
    """Nothing may be silently dropped to make the schema fit."""
    core = {name for name, _ in hf_dataset.CORE_FIELDS} | {"steps"}
    rows = {r["uid"]: r for r in hf_dataset.to_table_rows(records)}
    for record in records[:200]:
        extra = json.loads(rows[record["uid"]]["metadata"])
        missing = {k for k in record if k not in core} - set(extra)
        assert not missing, f"{record['uid']} lost {missing}"


def test_the_parquet_file_matches_the_declared_schema(tmp_path, records) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    target = tmp_path / "episodes.parquet"
    hf_dataset.write_parquet(records[:50], target)
    table = pq.read_table(target)
    assert table.num_rows == 50
    assert table.schema.field("steps").type == pa.string()
    assert table.schema.field("tau").type == pa.int64()
    assert [f.name for f in table.schema][:3] == ["uid", "episode_id", "corpus"]


def test_the_card_points_at_the_parquet_it_writes() -> None:
    text = hf_dataset.card(2823, 25)
    assert "data/episodes.parquet" in text
    assert "data/episodes.jsonl" not in text, "stale path in the card"


def _frontmatter(text: str) -> dict[str, str]:
    """Scalar keys of the card's YAML header, without a yaml dependency."""
    assert text.startswith("---\n")
    block = text.split("---\n", 2)[1]
    out = {}
    for line in block.splitlines():
        if line.startswith(" ") or line.startswith("-") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def test_license_link_is_an_https_uri() -> None:
    """The hub validates this field and rejects a bare filename with a 400.

    That failure only surfaces at upload time, after the whole payload has been
    built and scanned, so it is pinned here instead.
    """
    meta = _frontmatter(hf_dataset.card(2823, 25, "someone/some-dataset"))
    assert meta["license_link"].startswith("https://"), meta["license_link"]
    assert "someone/some-dataset" in meta["license_link"], (
        "the link must resolve inside the dataset it is published to")


def test_card_follows_the_repo_id_it_is_built_for() -> None:
    text = hf_dataset.card(2823, 25, "someone/some-dataset")
    assert 'load_dataset("someone/some-dataset"' in text
    assert hf_dataset.DEFAULT_REPO_ID not in text


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


# --------------------------------------------------------------------------
# The scan must cover everything the upload sends. These pin the bug where it
# didn't: the scan skipped `_` directories at any depth, the upload passed
# `ignore_patterns=["_*/**", "_*"]` which fnmatch anchors at the folder root,
# and 1559 nested cassette files were published unscanned as a result.
# --------------------------------------------------------------------------
def _fake_traces(root: pathlib.Path) -> None:
    for rel in ("ollama7b/real-healthy-000.jsonl",
                "ollama7b/manifest.json",
                "ollama7b/_cassettes/deadbeef.json",          # nested scratch
                "ollama7b/_cassettes/_backend/cafe.json",     # deeper still
                "_cassettes/toplevel.json",                   # top-level scratch
                "_aftraj/imported.jsonl",
                "manifest.json"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def test_nested_scratch_is_never_published(tmp_path, monkeypatch) -> None:
    """The regression: `_cassettes` under a corpus, not just at the root."""
    traces = tmp_path / "traces"
    _fake_traces(traces)
    monkeypatch.setattr(hf_dataset, "TRACES", traces)
    build_dir = tmp_path / "build"
    (build_dir / "data").mkdir(parents=True)
    (build_dir / "data" / "episodes.parquet").write_bytes(b"x")

    _, trace_files = hf_dataset.publish_payload(build_dir)
    published = {p.relative_to(traces).as_posix() for p in trace_files}
    assert published == {"ollama7b/real-healthy-000.jsonl",
                         "ollama7b/manifest.json", "manifest.json"}


def test_the_uploaded_set_is_exactly_the_scanned_set(tmp_path,
                                                     monkeypatch) -> None:
    """Applies the hub's own fnmatch semantics to the patterns we pass.

    This is the assertion that would have caught the original bug: it fails if
    the patterns ever select a file the scan did not see.
    """
    from fnmatch import fnmatch

    traces = tmp_path / "traces"
    _fake_traces(traces)
    monkeypatch.setattr(hf_dataset, "TRACES", traces)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "README.md").write_text("x", encoding="utf-8")

    _, trace_files = hf_dataset.publish_payload(build_dir)
    patterns = hf_dataset._allow_patterns(trace_files, traces)
    on_disk = [p for p in traces.rglob("*") if p.is_file()]
    selected = {p for p in on_disk
                if any(fnmatch(p.relative_to(traces).as_posix(), pat)
                       for pat in patterns)}
    assert selected == set(trace_files)


def test_no_underscore_directory_survives_into_the_real_payload() -> None:
    """The same invariant, against the corpus actually being shipped."""
    build_dir = REPO_ROOT / "build" / "hf"
    if not build_dir.is_dir():
        pytest.skip("no build/hf; run `py -m devtools.hf_dataset --build`")
    _, trace_files = hf_dataset.publish_payload(build_dir)
    assert trace_files, "payload is empty"
    leaked = [p.relative_to(REPO_ROOT / "traces").as_posix()
              for p in trace_files
              if any(part.startswith("_")
                     for part in p.relative_to(REPO_ROOT / "traces").parts[:-1])]
    assert not leaked, f"scratch reached the payload: {leaked[:5]}"


def test_stale_build_artefacts_are_not_re_uploaded(tmp_path,
                                                   monkeypatch) -> None:
    """`_prune` deletes these after the fact; not sending them is better."""
    traces = tmp_path / "traces"
    _fake_traces(traces)
    monkeypatch.setattr(hf_dataset, "TRACES", traces)
    build_dir = tmp_path / "build"
    (build_dir / "data").mkdir(parents=True)
    (build_dir / "data" / "episodes.parquet").write_bytes(b"x")
    for stale in hf_dataset.STALE_PATHS:
        (build_dir / stale).write_text("old", encoding="utf-8")

    build_files, _ = hf_dataset.publish_payload(build_dir)
    names = {p.relative_to(build_dir).as_posix() for p in build_files}
    assert names == {"data/episodes.parquet"}


def test_allow_patterns_refuses_a_glob_metacharacter(tmp_path) -> None:
    """A literal list only stays literal if nothing in it is a pattern."""
    victim = tmp_path / "weird[1].jsonl"
    victim.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="glob metacharacter"):
        hf_dataset._allow_patterns([victim], tmp_path)


class _FakeApi:
    """Records what `_prune` would delete, without touching the hub."""

    def __init__(self) -> None:
        self.files: list[str] = []
        self.folders: list[str] = []

    def delete_file(self, path_in_repo, repo_id, repo_type, commit_message):
        self.files.append(path_in_repo)

    def delete_folder(self, path_in_repo, repo_id, repo_type, commit_message):
        self.folders.append(path_in_repo)


def test_prune_removes_the_already_published_scratch_directory() -> None:
    """Excluding a path from the payload cannot unpublish it; only this can."""
    api = _FakeApi()
    hf_dataset._prune(api, "someone/some-dataset")
    assert api.files == list(hf_dataset.STALE_PATHS)
    assert api.folders == list(hf_dataset.STALE_DIRS)
    assert "traces/ollama/_cassettes" in api.folders


def test_pruned_directories_are_ones_the_payload_excludes() -> None:
    """A directory may only be pruned if we would never upload it again.

    Pruning something still in the payload would delete and re-add it forever.
    """
    build_dir = REPO_ROOT / "build" / "hf"
    if not build_dir.is_dir():
        pytest.skip("no build/hf; run `py -m devtools.hf_dataset --build`")
    _, trace_files = hf_dataset.publish_payload(build_dir)
    published = {f"traces/{p.relative_to(REPO_ROOT / 'traces').as_posix()}"
                 for p in trace_files}
    for stale in hf_dataset.STALE_DIRS:
        assert not [p for p in published if p.startswith(stale + "/")], (
            f"{stale} is pruned but still in the payload")
