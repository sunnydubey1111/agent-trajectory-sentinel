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

**Root digest:** `f7d9c6fa53825e1ced815168534c7f91236e91e7b0c342108ae535e4a6dfa3e6`

A single SHA-256 over every path and per-file hash in the manifest, in
sorted order. Two checkouts agreeing on this value agree on every
tracked byte.

| section | files | covers |
|---|---:|---|
| `code` | 123 | `derail/`, `verification/`, `experimental/`, `devtools/`, `tests/` |
| `docs` | 15 | `*.md`, the paper sources, and the requirements files |
| `results` | 196 | every table, figure and results JSON the claims cite |
| `traces` | 8,168 | every committed agent episode and replay cassette |
| **total** | **8,502** | |

Per-episode trace hashes are additionally recorded in each corpus's own
`manifest.json` where the collector wrote them (`trace_sha256`).
