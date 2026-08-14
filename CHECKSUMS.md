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

**Root digest:** `15e73d1d8549a970a6b0ed074ec237415b94bf48eead83cdc133a25ae2b0fa6b`

A single SHA-256 over every path and per-file hash in the manifest, in
sorted order. Two checkouts agreeing on this value agree on every
tracked byte.

| section | files | covers |
|---|---:|---|
| `code` | 130 | `derail/`, `verification/`, `experimental/`, `devtools/`, `tests/` |
| `docs` | 13 | `*.md`, `LICENSE`, `NOTICE`, the paper sources, and the requirements files |
| `results` | 198 | every table, figure and results JSON the claims cite |
| `traces` | 9,094 | every committed agent episode and replay cassette |
| **total** | **9,435** | |

Per-episode trace hashes are additionally recorded in each corpus's own
`manifest.json` where the collector wrote them (`trace_sha256`).
