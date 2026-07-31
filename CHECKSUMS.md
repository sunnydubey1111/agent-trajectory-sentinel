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

**Root digest:** `251c27d144f1d56ed2978146dca677c46edf41f8d969cee3a0a10926b86fdc9f`

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
