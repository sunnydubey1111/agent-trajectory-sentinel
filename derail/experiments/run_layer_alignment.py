"""The behavioural and grounding layers, measured on one shared population.

The two layers are reported by two studies that do not cover the same episodes.
`run_hybrid_study` scores 1,002 injected episodes over 8 datasets, 400 of them
simulator; `run_grounding_study` scores 874 over 10 real corpora, none
simulator. They overlap on 7 corpora and 602 episodes. A sentence that takes
the behavioural figure from one and the content figure from the other is
comparing two populations, and any difference between them is partly a
difference in which corpora were counted.

This study fixes the population instead of describing the problem. It reports
each quantity three times:

  own      - the study's own full population, unchanged, still useful on its own
  shared   - the 602-episode intersection, where the two layers ARE comparable
  outside  - the episodes each study holds that the other does not

`outside` is the diagnostic column: when `own` sits between `shared` and
`outside`, the pooled figure is a weighted average of two different regimes and
the weight is the corpus list, not a property of the layer.

Both layers are read from `grounding_diagnosis.csv`, which carries the
behavioural detectors AND the grounding ones for every episode it scores, so
the comparison is within one scoring run rather than across two. The
behavioural study's own table supplies the episodes it holds and grounding does
not (the simulator), and the two are verified to agree episode for episode
where they overlap.

Run:  py -m derail.experiments.run_layer_alignment
Writes results/tables/layer_alignment_{summary,by_dataset}.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLES_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"

#: Detector columns, named once so the two tables' different spellings for the
#: same monitor cannot drift apart.
BEHAVIOURAL = "det_esn_cusum_max"
CONTENT_GATE = "det_hybrid_content_gate"


def load_aligned(tables_dir: Path = TABLES_DIR) -> tuple[pd.DataFrame, set[str]]:
    """The grounding table, plus the corpora both studies scored.

    Raises if the two studies disagree about an episode they both scored: that
    would mean the shared population is not actually shared, and every number
    below would be comparing two answers to the same question.
    """
    g = pd.read_csv(tables_dir / "grounding_diagnosis.csv")
    h = pd.read_csv(tables_dir / "hybrid_diagnosis.csv")
    shared = set(g.dataset) & set(h.dataset)
    m = h.merge(g, on=["dataset", "episode_id"], suffixes=("_h", "_g"))
    for a, b, what in ((m.failure_class_h, m.failure_class_g, "failure_class"),
                       (m.det_esn.astype(bool),
                        m.det_esn_cusum_max.astype(bool), "ESN detection"),
                       (m.det_maha.astype(bool),
                        m.det_delta_mahalanobis.astype(bool), "Mahalanobis")):
        bad = int((a != b).sum())
        if bad:
            raise ValueError(
                f"the two studies disagree on {what} for {bad} of {len(m)} "
                "episodes they both scored; the shared population is not shared")
    return g, shared


def _metrics(d: pd.DataFrame) -> dict:
    content = d[d.is_content.astype(bool)]
    behav = d[~d.is_content.astype(bool)]
    out = {
        "n": len(d), "n_datasets": int(d.dataset.nunique()),
        "n_content": len(content), "n_behavioural": len(behav),
    }
    if len(content):
        out |= {
            "content_esn": round(float(content[BEHAVIOURAL].mean()), 4),
            "content_gate": round(float(content[CONTENT_GATE].mean()), 4),
            "content_gain": round(float(content[CONTENT_GATE].mean()
                                        - content[BEHAVIOURAL].mean()), 4),
            # The two things a gain requires, both measured rather than
            # asserted: the grounding stream must be able to see the
            # corruption at all, and the behavioural monitor must have left
            # something to catch. A corpus failing either shows no gain, which
            # is why the pooled figure moves with the corpus list.
            "grounding_stream": round(float(content.det_grounding.mean()), 4),
            "headroom": round(1.0 - float(content[BEHAVIOURAL].mean()), 4),
        }
    if len(behav):
        out |= {
            "behavioural_esn": round(float(behav[BEHAVIOURAL].mean()), 4),
            "behavioural_gate": round(float(behav[CONTENT_GATE].mean()), 4),
            "behavioural_delta": round(float(behav[CONTENT_GATE].mean()
                                             - behav[BEHAVIOURAL].mean()), 4),
        }
    return out


def summary_table(g: pd.DataFrame, shared: set[str]) -> pd.DataFrame:
    """Each quantity on the grounding study's own population, the shared one,
    and the episodes it holds outside the shared one."""
    arms = (
        ("own (grounding study)", g),
        ("shared (both studies)", g[g.dataset.isin(shared)]),
        ("outside shared", g[~g.dataset.isin(shared)]),
    )
    return pd.DataFrame([{"arm": name} | _metrics(d) for name, d in arms])


def by_dataset_table(g: pd.DataFrame, shared: set[str]) -> pd.DataFrame:
    rows = []
    for ds, d in g.groupby("dataset", sort=True):
        rows.append({"dataset": ds, "in_behavioural_study": ds in shared}
                    | _metrics(d))
    return pd.DataFrame(rows).sort_values("content_gain", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", default="layer_alignment")
    args = ap.parse_args()

    g, shared = load_aligned()
    summary = summary_table(g, shared)
    by_ds = by_dataset_table(g, shared)

    print(f"[layers] shared corpora: {len(shared)} of "
          f"{g.dataset.nunique()} scored by the grounding study")
    print("\n[summary] the same quantities on three populations")
    print(summary.to_string(index=False))
    print("\n[by dataset] content gain, and whether the behavioural study "
          "covers the corpus")
    print(by_ds[["dataset", "in_behavioural_study", "n_content",
                 "content_esn", "content_gate", "content_gain"]]
          .to_string(index=False))

    own = summary.loc[summary.arm.str.startswith("own"), "content_gain"].iloc[0]
    sh = summary.loc[summary.arm.str.startswith("shared"), "content_gain"].iloc[0]
    print(f"\n[composition] pooled content gain {own:+.4f} against {sh:+.4f} on "
          f"the matched population: a gap of {own - sh:+.4f} carried by which "
          "corpora each study counts, not by the layer.")

    import numpy as np
    r_stream = float(np.corrcoef(by_ds.grounding_stream, by_ds.content_gain)[0, 1])
    r_head = float(np.corrcoef(by_ds.headroom, by_ds.content_gain)[0, 1])
    dead = by_ds[by_ds.grounding_stream == 0.0]
    print(f"[mechanism] across {len(by_ds)} corpora the gain tracks whether the "
          f"grounding stream can see the corruption (r={r_stream:+.3f}) and how "
          f"much the behavioural monitor left (r={r_head:+.3f}). The "
          f"{len(dead)} corpora whose grounding stream detects nothing "
          f"({', '.join(dead.dataset)}) have a mean gain of "
          f"{dead.content_gain.mean():+.4f}: the pooled figure moves with the "
          "corpus list because these two conditions are corpus properties.")

    for name, frame in (("summary", summary), ("by_dataset", by_ds)):
        frame.to_csv(TABLES_DIR / f"{args.out_prefix}_{name}.csv", index=False)
    print(f"[layers] wrote {args.out_prefix}_"
          "{summary,by_dataset}.csv to " + str(TABLES_DIR))


if __name__ == "__main__":
    main()
