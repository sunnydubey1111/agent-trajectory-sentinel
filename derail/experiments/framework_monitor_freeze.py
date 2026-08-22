"""Freezes the monitor used to score the framework x real-tool validation
study, before any of its episodes are collected -- produces the immutable
artifact that scoring later reads and never regenerates.

**Canonical monitor, traced repo-wide, not chosen by convenience.**
`README.md:128` states outright: "The primary monitor is `esn_cusum_max`:
one ESN-CUSUM detector per channel[, fused by max]." That name resolves to
exactly one class: `derail.monitor.esn.ChannelMaxESNMonitor` (K=8,
cusum=True, one `ESNEnsembleMonitor` sub-monitor per channel, seed=seed*100+i,
fused by `max`) -- confirmed by reading the class, and confirmed as the thing
`README.md`/`CLAIMS.md`/`devtools/claims_ledger.py` actually cite: `h1.auc`,
`h1.detection`, `hybrid.esn`, the AFTraj external-validation claim, and the
real-traces claims keyed `esn_cusum_max[e,m]` all resolve through this class.

**No single already-fit "the" monitor object exists project-wide**, and that
is not a gap -- it is how every headline `esn_cusum_max` number in this
project is already produced: `run_real_traces.py` (per-corpus), the hybrid
grand-mean (8 datasets), the AFTraj external arm, and `ollama_llama8b`'s
cross-family transfer arm each *refit* this same class fresh on the
population that specific study needs, via the same 60/20/20 healthy
train/val/test split and a 5%-FA-budget threshold picked on the val split
alone (`run_real_traces.py`, confirmed by reading it). Fitting fresh on this
study's own population, the same way, is consistent with that established
practice -- not a new procedure invented for it.

**Channels are data-driven, per the project's own stated rule**
(`run_real_traces.py`'s docstring: "traces without logprobs fall back to
e+m"), and `esn_cusum_max[e,m]` is itself an already-published, separately
named variant (`devtools/claims_ledger.py`'s real-traces claims). This freeze
uses `channels=("e","m")` specifically because its SCORING TARGET -- the
LangGraph/AutoGen framework episodes -- has no u channel at all
(`ChatOllama`/`OllamaChatCompletionClient` do not surface logprobs;
documented in `harness/frameworks.py`'s own module docstring). Fitting with
u included would produce a monitor that cannot even score what this study
needs it to score.

Run once, before any episode of the study exists. Never regenerated after
scoring starts.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np

from derail.common import D_META, D_SEM, D_TOTAL, D_UNC, DEGENERATE_EPS, rng_for
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import (_MONITOR_SPLIT_SEED, ChannelMaxESNMonitor,
                                ESNEnsembleMonitor)
from derail.telemetry.adapter import episode_from_trace
from derail.telemetry.events import SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_PATH = REPO_ROOT / "results" / "framework_monitor_freeze.json"

#: Same five corpora fixed as the rollback/retry study's source pool.
FIT_CORPORA = (
    "real_research7b", "real_ollama7b", "real_research7b_long",
    "real_research7b_long_ext", "real_research7b_long_drift",
)

#: The scoring target (LangGraph/AutoGen) has no u channel -- see module
#: docstring. Matches the already-published `esn_cusum_max[e,m]` variant.
CHANNELS = ("e", "m")
FA_BUDGET = 0.05
SPLIT_SEED_BASE = 52026
SPLIT_SEED_TAGS = ("e1", "monitor-split")

#: Full dependency chain that can alter a fitted score: the monitor class
#: itself, the feature schema/robust-scaling primitives, the feature
#: construction that turns a trace into Episode.X, and the threshold-picking
#: function. Anything else touching a score lives inside one of these.
DEPENDENCY_FILES = (
    "derail/monitor/esn.py",
    "derail/common.py",
    "derail/telemetry/adapter.py",
    "derail/telemetry/events.py",
    "derail/evaluation/metrics.py",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=10).stdout.strip()
    except Exception:                                      # noqa: BLE001
        return ""


def _blob_hashes(paths: tuple[str, ...]) -> dict:
    return {p: _git("rev-parse", f"HEAD:{p}") for p in paths}


def _sha256_file(path: Path) -> str:
    """Content hash of a trace file, over its LF-normalised bytes.

    Raw bytes make this a property of the CHECKOUT, not the corpus: git
    stores these files with LF and hands a Windows working tree CRLF, so a
    freeze taken on one platform rejects the identical corpus on the other
    -- all 257 frozen files read as drifted on Linux CI. Line endings are
    stripped by splitlines() before a step is parsed, so they cannot change
    an episode; normalising anchors the check to the committed content, as
    devtools/artifact_manifest._sha256 already does.
    """
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_healthy_episodes(corpora: tuple[str, ...]):
    """(episodes, ids, source_file_hashes) for every healthy episode,
    corpora in fixed order, ids sorted within each -- deterministic."""
    episodes, ids, file_hashes = [], [], {}
    for corpus in corpora:
        manifest_path = TRACES_ROOT / corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        healthy = sorted(
            (e for e in manifest if e.get("failure_class") is None),
            key=lambda e: e["episode_id"])
        for entry in healthy:
            trace_path = TRACES_ROOT / corpus / entry["file"]
            full_id = f"{corpus}/{entry['episode_id']}"
            file_hashes[full_id] = _sha256_file(trace_path)
            steps = [json.loads(line) for line in
                    trace_path.read_text("utf-8").splitlines() if line.strip()]
            ep = episode_from_trace(steps, full_id, use_sentence_transformers=False,
                                    extended=True)
            episodes.append(ep)
            ids.append(full_id)
    return episodes, ids, file_hashes


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _hash_file_hashes(file_hashes: dict) -> str:
    payload = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_feature_arrays(episodes: list) -> str:
    """SHA-256 over the exact bytes of the concatenated, ORDERED feature
    matrices actually handed to fit()/scoring -- catches a feature-
    construction change even when episode IDs and their source files are
    unchanged (e.g. a channel-slicing or adapter bug)."""
    h = hashlib.sha256()
    for ep in episodes:                     # already in deterministic order
        h.update(np.ascontiguousarray(ep.X).tobytes())
    return h.hexdigest()


def _split_60_20_20(episodes: list, ids: list[str]):
    perm = rng_for(SPLIT_SEED_BASE, *SPLIT_SEED_TAGS).permutation(len(episodes))
    n_train = int(round(0.6 * len(episodes)))
    n_val = int(round(0.2 * len(episodes)))
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    pick = lambda idxs: ([episodes[i] for i in idxs], [ids[i] for i in idxs])
    return pick(idx_train), pick(idx_val), pick(idx_test)


def _esn_hyperparameters() -> dict:
    """Hyperparameters of the inner ESNEnsembleMonitor sub-monitors
    ChannelMaxESNMonitor builds one per channel -- these, not the outer
    class's own K/cusum/seed (recorded separately), are what shapes each
    channel's reservoir dynamics."""
    sig = inspect.signature(ESNEnsembleMonitor.__init__)
    return {name: p.default for name, p in sig.parameters.items()
            if name not in ("self", "standardizer", "channels", "K", "cusum",
                           "seed", "name")
            and p.default is not inspect.Parameter.empty}


def build_freeze_artifact() -> dict:
    episodes, ids, file_hashes = _load_healthy_episodes(FIT_CORPORA)
    if len(episodes) < 10:
        raise SystemExit(f"only {len(episodes)} healthy episodes across "
                         f"{FIT_CORPORA} -- too few to fit/threshold; "
                         f"refusing to freeze a degenerate monitor")

    (train, train_ids), (val, val_ids), (test, test_ids) = _split_60_20_20(episodes, ids)

    from derail.common import Standardizer
    standardizer = Standardizer()
    standardizer.fit(train)
    monitor = ChannelMaxESNMonitor(standardizer, K=8, cusum=True, seed=0,
                                   name="esn_cusum_max", channels=CHANNELS)
    monitor.fit(train)

    val_streams = []
    for ep in val:
        monitor.start_episode()
        val_streams.append(np.array([monitor.score_step(ep.X[t])
                                     for t in range(ep.X.shape[0])]))
    theta_b5 = float(pick_threshold(val_streams, fa_budget=FA_BUDGET))

    artifact = {
        "purpose": "Monitor freeze for the framework/real-tool validation "
                   "study -- immutable, generated once before any of its "
                   "episodes are scored; never regenerated after.",
        "monitor_name": "esn_cusum_max",
        "monitor_class": "derail.monitor.esn.ChannelMaxESNMonitor",
        "canonical_source": "README.md:128 (\"The primary monitor is "
                            "esn_cusum_max\"); devtools/claims_ledger.py "
                            "h1.auc/h1.detection/hybrid.esn and the "
                            "real-traces esn_cusum_max[e,m] claims all "
                            "resolve to this class.",
        "channels": list(CHANNELS),
        "channel_selection_reason": "The scoring target (LangGraph/AutoGen) "
                                    "has no u channel (no logprobs) -- "
                                    "matches the already-published "
                                    "esn_cusum_max[e,m] variant, per "
                                    "run_real_traces.py's own data-driven "
                                    "channel rule.",
        "outer_monitor_config": {"K": 8, "cusum": True, "seed": 0},
        "inner_esn_hyperparameters": _esn_hyperparameters(),
        "scaling_config": {"function": "_robust_loc_scale",
                           "degenerate_eps": DEGENERATE_EPS},
        "monitor_split_seed": _MONITOR_SPLIT_SEED,
        "outer_split_seed_base": SPLIT_SEED_BASE,
        "outer_split_seed_tags": list(SPLIT_SEED_TAGS),
        "outer_split_method": "60/20/20 healthy train/val/test, matching "
                              "run_real_traces.py's established real-corpus "
                              "methodology",
        "feature_schema_version": SCHEMA_VERSION,
        "feature_dims": {"D_TOTAL": D_TOTAL, "D_SEM": D_SEM,
                         "D_UNC": D_UNC, "D_META": D_META},
        "fa_budget": FA_BUDGET,
        "theta_b5": theta_b5,
        "fit_corpora": list(FIT_CORPORA),
        "training_episode_ids": train_ids,
        "training_episode_count": len(train_ids),
        "calibration_episode_ids": val_ids,
        "calibration_episode_count": len(val_ids),
        "held_out_test_episode_ids": test_ids,
        "training_input_hash": _hash_ids(train_ids),
        "calibration_input_hash": _hash_ids(val_ids),
        "source_file_sha256": {i: file_hashes[i] for i in train_ids + val_ids},
        "source_file_hash_digest": _hash_file_hashes(
            {i: file_hashes[i] for i in train_ids + val_ids}),
        "feature_array_hash_train": _hash_feature_arrays(train),
        "feature_array_hash_val": _hash_feature_arrays(val),
        "dependency_code_commit": _git("rev-parse", "HEAD"),
        "dependency_blob_hashes": _blob_hashes(DEPENDENCY_FILES),
        "dependency_files_dirty": bool(_git("status", "--porcelain",
                                            *DEPENDENCY_FILES)),
    }
    return artifact


def load_frozen_monitor(freeze_path: Path = OUT_PATH):
    """Re-fit the monitor described by an existing freeze artifact.

    No model is persisted (esn.py fits deterministically from code+seed+data,
    confirmed -- no pickle exists anywhere in this project), so "loading" a
    frozen monitor means re-running the same deterministic fit against the
    same frozen inputs. Asserts the result still matches the frozen
    `training_input_hash`, `source_file_hash_digest` and every dependency
    blob hash -- if the environment has drifted since the freeze (a corpus
    edited, a healthy episode added, ANY dependency file touched), this
    raises instead of silently scoring against a different monitor than what
    was frozen.

    Returns (monitor, theta_b5, artifact).
    """
    from derail.common import Standardizer

    artifact = json.loads(freeze_path.read_text("utf-8"))
    for path, frozen_blob in artifact["dependency_blob_hashes"].items():
        current_blob = _git("rev-parse", f"HEAD:{path}")
        if current_blob != frozen_blob:
            raise RuntimeError(
                f"dependency {path} has drifted since the freeze "
                f"({frozen_blob!r} -> {current_blob!r}) -- do not score "
                f"against a monitor whose dependency chain no longer "
                f"matches its own freeze artifact.")

    episodes, ids, file_hashes = _load_healthy_episodes(tuple(artifact["fit_corpora"]))
    by_id = dict(zip(ids, episodes))
    train_ids = artifact["training_episode_ids"]
    val_ids = artifact["calibration_episode_ids"]
    if _hash_ids(train_ids) != artifact["training_input_hash"]:
        raise RuntimeError("frozen training population id set has drifted")
    current_source_hashes = {i: file_hashes[i] for i in train_ids + val_ids}
    if _hash_file_hashes(current_source_hashes) != artifact["source_file_hash_digest"]:
        raise RuntimeError(
            "one or more source trace files have changed content since the "
            "freeze -- refusing to score against a drifted population.")

    train = [by_id[i] for i in train_ids]
    if _hash_feature_arrays(train) != artifact["feature_array_hash_train"]:
        raise RuntimeError(
            "recomputed feature arrays no longer match the frozen "
            "feature_array_hash_train -- feature construction has changed "
            "even though source files have not.")

    standardizer = Standardizer()
    standardizer.fit(train)
    monitor = ChannelMaxESNMonitor(
        standardizer, K=artifact["outer_monitor_config"]["K"],
        cusum=artifact["outer_monitor_config"]["cusum"],
        seed=artifact["outer_monitor_config"]["seed"],
        name=artifact["monitor_name"], channels=tuple(artifact["channels"]))
    monitor.fit(train)
    return monitor, float(artifact["theta_b5"]), artifact


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing freeze artifact (must not "
                         "be used after any episode has been scored)")
    args = ap.parse_args(argv)

    if OUT_PATH.exists() and not args.force:
        raise SystemExit(
            f"{OUT_PATH} already exists -- refusing to regenerate. The "
            f"freeze artifact is generated once, before scoring starts, and "
            f"never after. Pass --force only if scoring has not yet started "
            f"against it.")

    artifact = build_freeze_artifact()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"theta_b5 = {artifact['theta_b5']}")
    print(f"train={artifact['training_episode_count']} "
         f"val={artifact['calibration_episode_count']} "
         f"test={len(artifact['held_out_test_episode_ids'])}")


if __name__ == "__main__":
    main()
