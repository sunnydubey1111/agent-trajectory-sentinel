"""Build (and optionally push) the Hugging Face dataset release.

The corpus in `traces/` is the source of record: one JSONL per episode plus a
per-corpus `manifest.json`, hashed in BASELINE_MANIFEST.json. That layout is
right for reproduction and wrong for discovery - `load_dataset` cannot read it,
so a visitor would have to clone the repo to see anything.

This module emits a second, derived view: `data/episodes.jsonl`, one JSON object
per episode carrying its metadata and its full step list, which `load_dataset`
reads directly. The raw traces are uploaded unchanged beside it. Nothing is
recomputed for the derived view - every field is copied from the manifest or the
trace - and `tests/test_hf_dataset.py` asserts the two agree on episode count,
tau, T and step content, so the convenience copy cannot drift from the record.

    py -m devtools.hf_dataset --build            # write build/hf/ and stop
    py -m devtools.hf_dataset --build --push     # ... and upload

Pushing needs a write token, read from the environment or a prior
`hf auth login`. No token is read from, or written to, this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACES = REPO_ROOT / "traces"
BUILD_DIR = REPO_ROOT / "build" / "hf"
DEFAULT_REPO_ID = "sunnydubey1111/agent-trajectory-sentinel"

#: Corpus directories that are not ours (imported for external validation) or
#: are scratch. A leading underscore is the marker used across the project.
def _corpora() -> list[pathlib.Path]:
    return sorted(p for p in TRACES.glob("*/manifest.json")
                  if not p.parent.name.startswith("_"))


#: Fields every corpus records, published as typed columns. Everything else a
#: manifest carries is corpus-specific - the booking corpora record
#: `expected_total` and `withhold_rate`, the framework corpora record
#: `provenance` and `framework` - so a row-per-episode table cannot give them
#: all one schema. They travel in `metadata` as JSON instead of being dropped.
CORE_FIELDS: tuple[tuple[str, str], ...] = (
    ("uid", "string"), ("episode_id", "string"), ("corpus", "string"),
    ("model", "string"), ("failure_class", "string"),
    ("tau", "int64"), ("T", "int64"), ("has_logprobs", "bool"),
)


def build_episodes() -> list[dict]:
    """One record per committed episode: manifest metadata plus its steps."""
    records: list[dict] = []
    for manifest_path in _corpora():
        corpus = manifest_path.parent.name
        for entry in json.loads(manifest_path.read_text("utf-8")):
            trace = manifest_path.parent / entry["file"]
            steps = [json.loads(line) for line in
                     trace.read_text("utf-8").splitlines() if line.strip()]
            record = dict(entry)
            record.pop("file", None)
            record["corpus"] = corpus
            # episode_id is unique WITHIN a corpus, not across the corpus set:
            # `autogen` and `autogen7b` both number their episodes from zero,
            # so 645 ids appear twice. A consumer keying on episode_id alone
            # would merge unrelated episodes, which is why the primary key is
            # published explicitly rather than left to be inferred.
            record["uid"] = f"{corpus}/{entry['episode_id']}"
            record["steps"] = steps
            records.append(record)
    return records


def to_table_rows(records: list[dict]) -> list[dict]:
    """Flatten to one uniform, typed row per episode.

    Written as Parquet with an explicit schema rather than JSONL: the hub
    infers a JSONL schema from the first rows, and since corpora disagree on
    which fields they record, inference picks a schema the later rows cannot
    be cast to. The steps list has the same problem one level down - a tool
    call's `args` is whatever that tool takes - so it is carried as JSON text
    rather than pretending it has a fixed shape.
    """
    rows = []
    for record in records:
        row = {name: record.get(name) for name, _ in CORE_FIELDS}
        extra = {k: v for k, v in record.items()
                 if k not in dict(CORE_FIELDS) and k != "steps"}
        row["n_steps"] = len(record["steps"])
        row["steps"] = json.dumps(record["steps"], ensure_ascii=False)
        row["metadata"] = json.dumps(extra, ensure_ascii=False, default=str)
        rows.append(row)
    return rows


def write_parquet(records: list[dict], target: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = to_table_rows(records)
    kinds = {"string": pa.string(), "int64": pa.int64(), "bool": pa.bool_()}
    schema = pa.schema(
        [pa.field(name, kinds[kind]) for name, kind in CORE_FIELDS]
        + [pa.field("n_steps", pa.int64()),
           pa.field("steps", pa.string()),
           pa.field("metadata", pa.string())])
    table = pa.Table.from_pylist(rows, schema=schema)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="zstd")


def card(n_episodes: int, n_corpora: int,
         repo_id: str = DEFAULT_REPO_ID) -> str:
    """The dataset card: HF frontmatter, then DATA_CARD.md unchanged.

    The body is not rewritten for HF. DATA_CARD.md is generated from the
    manifests and gated by `devtools.data_card --check`, so reusing it verbatim
    is the only way the published card cannot disagree with the corpus.
    """
    body = (REPO_ROOT / "DATA_CARD.md").read_text("utf-8")
    notice = (TRACES / "NOTICE_gemini.md").read_text("utf-8")
    frontmatter = "\n".join([
        "---",
        # Mixed, and saying "mit" here would be false: the code and the trace
        # format are MIT, but recorded model output carries the terms of the
        # model that produced it. See the Licensing section of the body.
        "license: other",
        "license_name: mixed-see-licensing",
        # The hub validates this as a URI and rejects a bare filename, so it
        # must be the resolved URL of LICENSING.md inside this dataset repo.
        f"license_link: https://huggingface.co/datasets/{repo_id}/blob/main/"
        "LICENSING.md",
        "pretty_name: AgentTrajectorySentinel agent-failure traces",
        "size_categories:",
        "- 1K<n<10K",
        "task_categories:",
        "- time-series-forecasting",
        "tags:",
        "- llm-agents",
        "- agent-monitoring",
        "- anomaly-detection",
        "- failure-detection",
        "- observability",
        "configs:",
        "- config_name: default",
        "  data_files: data/episodes.parquet",
        "---",
        "",
        f"# AgentTrajectorySentinel — {n_episodes} agent episodes "
        f"across {n_corpora} corpora",
        "",
        "Committed agent trajectories with step-level telemetry, used to fit and",
        "evaluate one-class monitors for real-time failure detection. Code,",
        "paper and the full evaluation harness:",
        f"<https://github.com/{repo_id.split('/')[0]}/"
        "agent-trajectory-sentinel>",
        "",
        "## Loading",
        "",
        "```python",
        "import json",
        "from datasets import load_dataset",
        "",
        f'ds = load_dataset("{repo_id}", split="train")',
        'row = ds[0]',
        'steps = json.loads(row["steps"])          # the trajectory',
        'meta  = json.loads(row["metadata"])       # corpus-specific fields',
        'steps[0].keys()   # text, action, latency_s, token_logprobs, ...',
        "```",
        "",
        "`steps` and `metadata` are JSON strings, not nested columns, and that is",
        "deliberate. Corpora disagree on which fields they record - the booking",
        "sets carry `expected_total`, the framework sets carry `provenance` - and",
        "a tool call's `args` is whatever that tool takes. There is no single",
        "table schema that fits them, so the varying parts are carried as text",
        "rather than dropped to make one fit. Nothing is lost: `metadata` holds",
        "every manifest field outside the typed columns.",
        "",
        "`data/episodes.parquet` is a convenience view. The authoritative layout",
        "is `traces/<corpus>/`, uploaded unchanged - one JSONL per episode plus",
        "the `manifest.json` every published number is computed from.",
        "",
        "## Fields",
        "",
        "| field | meaning |",
        "|---|---|",
        "| `uid` | **primary key**, `corpus/episode_id` |",
        "| `episode_id` | id within its corpus — **not unique across corpora**: "
        "`autogen` and `autogen7b` both number from zero, so 645 ids appear "
        "twice. Key on `uid`. |",
        "| `corpus` | which corpus the episode belongs to |",
        "| `n_steps` | number of steps, same as `T` |",
        "| `steps` | JSON string: the trajectory (see Loading) |",
        "| `metadata` | JSON string: every manifest field outside these columns |",
        "| `model` | the agent model that produced it |",
        "| `failure_class` | `null` for healthy, else the injected class |",
        "| `tau` | 0-indexed onset step; `null` for healthy |",
        "| `T` | number of steps |",
        "| `has_logprobs` | whether the `u` (uncertainty) channel is populated |",
        "| `steps` | the trajectory: per-step text, action, timings, tool events |",
        "",
        "---",
        "",
    ])
    return frontmatter + body + "\n\n---\n\n" + notice


# Credential shapes worth refusing to publish. Deliberately narrow: a pattern
# loose enough to catch "any long hex string" would fire on every episode id
# and hash in the corpus, and a gate that always fires is a gate nobody reads.
_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style key"),
    (r"sk-ant-[A-Za-z0-9\-_]{20,}", "Anthropic key"),
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"AIza[A-Za-z0-9_\-]{30,}", "Google API key"),
    (r"hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"xox[baprs]-[A-Za-z0-9\-]{10,}", "Slack token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"(?i)\b(authorization|api[_-]?key|password|passwd|secret)\b\s*[:=]\s*"
     r"[\"']([A-Za-z0-9_\-]{16,})[\"']", "credential assignment"),
)


def scan_for_secrets(paths: list[pathlib.Path]) -> list[str]:
    """Credential-shaped strings in what is about to be published.

    Publishing is irreversible in the way a git push is not: a Hugging Face
    dataset is cloned by strangers within minutes. This runs before every
    upload and refuses the push rather than warning about it.
    """
    import re

    compiled = [(re.compile(p), label) for p, label in _SECRET_PATTERNS]
    findings: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, label in compiled:
            for match in pattern.finditer(text):
                snippet = match.group(0)[:12]
                findings.append(f"{path.name}: {label} near {snippet!r}")
                break                      # one report per pattern per file
    return findings


def build(out_dir: pathlib.Path, repo_id: str = DEFAULT_REPO_ID) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)

    records = build_episodes()
    target = out_dir / "data" / "episodes.parquet"
    write_parquet(records, target)

    n_corpora = len(_corpora())
    (out_dir / "README.md").write_text(card(len(records), n_corpora, repo_id),
                                       encoding="utf-8", newline="\n")
    (out_dir / "LICENSING.md").write_text(
        "See the Licensing section of the dataset card. In short: the code and\n"
        "trace format are MIT; recorded model output carries the terms of the\n"
        "model that produced it (qwen2.5 Apache-2.0, llama3.1 Community License\n"
        "with its Acceptable Use Policy, gemini-2.5-flash under the Gemini API\n"
        "terms - see NOTICE_gemini.md).\n", encoding="utf-8", newline="\n")

    return {"episodes": len(records), "corpora": n_corpora,
            "bytes": target.stat().st_size}


#: Files an earlier release published that must not survive a re-upload.
#: `data/episodes.jsonl` had a schema the hub could not cast, and uploading
#: never deletes, so leaving it keeps a broken 32 MB file in the repo beside
#: the parquet that replaced it.
STALE_PATHS: tuple[str, ...] = ("data/episodes.jsonl",)


def _prune(api, repo_id: str) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    for path in STALE_PATHS:
        try:
            api.delete_file(path_in_repo=path, repo_id=repo_id,
                            repo_type="dataset",
                            commit_message=f"Remove superseded {path}")
            print(f"[hf] removed stale {path}")
        except EntryNotFoundError:
            pass
        except Exception as exc:                     # already gone, or no rights
            print(f"[hf] could not remove {path}: {exc}")


def push(out_dir: pathlib.Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    payload = ([p for p in out_dir.rglob("*") if p.is_file()]
               + [p for p in TRACES.rglob("*")
                  if p.is_file() and not any(part.startswith("_")
                                             for part in p.relative_to(TRACES).parts[:-1])])
    findings = scan_for_secrets(payload)
    if findings:
        for finding in findings[:20]:
            print(f"  {finding}", file=sys.stderr)
        raise SystemExit(f"refusing to publish: {len(findings)} credential-shaped "
                         f"string(s) in the payload")
    print(f"[hf] secret scan clean over {len(payload)} files")

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)          # falls back to a prior `hf auth login`
    api.create_repo(repo_id, repo_type="dataset", private=private,
                    exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id,
                      repo_type="dataset",
                      commit_message="Publish the agent-failure trace corpus")
    _prune(api, repo_id)
    # The authoritative layout, uploaded unchanged beside the derived view.
    api.upload_folder(folder_path=str(TRACES), repo_id=repo_id,
                      repo_type="dataset", path_in_repo="traces",
                      ignore_patterns=["_*/**", "_*"],
                      commit_message="Add the per-episode trace files")
    print(f"[hf] pushed to https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m devtools.hf_dataset")
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--scan-only", action="store_true",
                        help="run the credential scan over the payload and stop")
    parser.add_argument("--push", action="store_true",
                        help="upload; needs HF_TOKEN or a prior `hf auth login`")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--out", default=str(BUILD_DIR))
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    summary = build(out_dir, args.repo_id)
    print(f"[hf] {summary['episodes']} episodes from {summary['corpora']} "
          f"corpora -> {summary['bytes'] / 1e6:.1f} MB")
    print(f"[hf] built {out_dir}")
    if args.scan_only:
        payload = ([p for p in out_dir.rglob("*") if p.is_file()]
                   + [p for p in TRACES.rglob("*") if p.is_file()])
        findings = scan_for_secrets(payload)
        for finding in findings[:20]:
            print(f"  {finding}")
        print(f"[hf] scanned {len(payload)} files: "
              f"{len(findings)} credential-shaped string(s)")
        return 1 if findings else 0
    if args.push:
        push(out_dir, args.repo_id, args.private)
    else:
        print("[hf] not pushed (pass --push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
