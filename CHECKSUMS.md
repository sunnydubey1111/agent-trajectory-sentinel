# Checksums

Every file this repository publishes is hashed with SHA-256 in
`BASELINE_MANIFEST.json`. Text files are hashed after CRLF to LF
normalisation, matching the `.gitattributes` policy, so a checkout on
Windows and one on Linux produce identical digests and a line-ending
flip is never mistaken for a data change.

```
py -m devtools.artifact_manifest --check     # verify every file
py -m devtools.artifact_manifest --doc       # regenerate this summary
```

**Root digest:** `3327584d738be91164aa18ed215a77cb952058b7bab50765f154b956295d0f26`

A single SHA-256 over every path and per-file hash in the manifest, in
sorted order. Two checkouts agreeing on this value agree on every
tracked byte.

| section | files | covers |
|---|---:|---|
| `code` | 112 | `derail/`, `verification/`, `experimental/`, `devtools/`, `tests/` |
| `docs` | 14 | `*.md`, the paper sources, and the requirements files |
| `results` | 186 | every table, figure and results JSON the claims cite |
| `traces` | 8,168 | every committed agent episode and replay cassette |
| **total** | **8,480** | |

Per-episode trace hashes are additionally recorded in each corpus's own
`manifest.json` where the collector wrote them (`trace_sha256`).
