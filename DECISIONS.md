# Decisions

Departures from, and resolutions of, `cellular_response_prototype_brief.md`. Each entry states
what was decided, why, and what would reverse it.

## D1. JAX + Equinox instead of PyTorch (2026-10-16)

Owner's choice. Flows use FlowJAX (Equinox-native rational-quadratic spline coupling flows),
optimization uses Optax, and checkpoints use `eqx.tree_serialise_leaves`. JAX runs on CPU:
`jax-metal` is experimental and lags JAX releases. The brief-sized flow (6 layers, 8 knots, 2×256,
64-d) has 1.84M parameters and produced finite log-probabilities on CPU. The optional CPA/chemCPA
comparator (PyTorch) would need a separate environment.

## D2. Compute envelope is a laptop (2026-10-16)

MacBook Pro M3 Pro (6P+6E cores), 18 GB RAM, ~600 GB free SSD, ~10 MB/s measured download, no
CUDA GPU. This replaces the brief's 24 GB GPU / 64 GB RAM planning envelope. Consequences: sparse
counts on disk; HVG/PCA fitted on bounded balanced subsamples; models stay small; throughput is
benchmarked on 100k cells before long runs. Reversed if flow runs exceed ~1 h each.

## D3. Pinned source and selective access (2026-10-16)

`tahoebio/Tahoe-100M` at revision `349e2a55f5538a3e73b32a06c6bdd8e9e87011e8` (CC0-1.0 on the
card; `LICENSE.md` checksummed). Metadata is downloaded with a resumable `httpx` client because
`hf_hub_download` stalled on the xet-backed 2.3 GB file. Expression rows are fetched by row group
using `data/interim/cell_locations.parquet`, which was reconstructed from metadata and verified
against all 98,252 row-group footers. The Tara plugin advertised on the dataset page is not used.
Anonymous access hit HTTP 429; set `HF_TOKEN` for cohort downloads.

## D4. Retained population = `pass_filter == "full"` (2026-10-16)

Only `full` cells (95,624,334) have expression rows; the 5,024,456 `minimal` cells are metadata-only.
All outcomes are defined among `full` cells. `minimal` counts are reported per (well, line) as
attrition and never imputed. No additional QC filter is applied in v0.1. Any later filter will be
chosen on training data with treatment-sensitivity checks.

## D5. Compound identity (2026-10-16)

Joins use `compound_key`, a case- and whitespace-normalized name, because sample and obs tables
carry trailing spaces (`'Erdafitinib '`, `'Selinexor '`) that `drug_metadata` and the expression
shards do not. Split grouping uses the largest-fragment InChIKey: 22 groups of salts, solvates, or
stereo-free duplicates (including Doxorubicin/Epirubicin) are held out together. Two SMILES fail to
parse (`Sacubitril/Valsartan`, `Verteporfin`). They stay in protocol A with a "no fingerprint" flag
and are excluded from protocol C.

## D6. Cohort (2026-10-16; freeze at M1a)

- **Lines:** the 12 top-ranked by feasible plates, then minimum vehicle cells, then treated-well
  coverage: PANC-1, SW480, A498, HOP62, J82, NCI-H460, LoVo, A549, CFPAC-1, LOX-IMVI, NCI-H2347,
  HT-29 (lung 4, bowel 3, pancreas 2, kidney, bladder, skin). Selection used metadata only.
- **Plates:** compound groups B (plates 4–6) and C (plates 7–9), plus plate 14. That gives 185
  panel compounds (excluding the Adagrasib anchor), close to the brief's ~160, and includes the
  documented 6/14 repeat. Groups A and D are
  a later scale extension.
- **Caps:** ≤256 treated cells per (well, line), all vehicle cells. Selection is hash-deterministic
  over row groups and then cells (`cellresponse.data.select`).
- **Smoke:** plate 6, the top 4 lines, 12 group-B compounds chosen by hash (not by response),
  ≤128 treated and ≤512 vehicle cells per (well, line).
- **Footprint:** ~47 GB of row groups and ~2.44M cells for the prototype; ~1 GB for the smoke cohort.

## D7. Query blocks and anchors (2026-10-16)

A query is one (line, plate) block at the plate's single panel dose with that plate's vehicle
wells as the shared reference. The Adagrasib 0.05 µM anchor is excluded from panels, because on
most plates it sits at a different dose than the panel. It is used as a cross-plate diagnostic.
No cross-plate, cross-dose, or cross-group ranking is evaluated.

## D8. Plate 13 deferred (2026-10-16)

Plate 13 (mixed doses; 58 repeat wells and 36 unique conditions) is not in the v0.1 cohort and
has no query blocks. It is reserved for a later replication and cross-group comparability study.

## D9. Eligibility thresholds unchanged (2026-10-16)

≥128 retained treated cells per (well, line); ≥256 vehicle cells per (plate, line); ≥10 held-out
candidates per query. Under these thresholds, 591 of 650 (line, plate) blocks are feasible.

## D10. Tooling (2026-10-16)

The installed uv is 0.4.9, which predates `[dependency-groups]`, so dev dependencies live in
`[tool.uv] dev-dependencies`. The CLI uses argparse. The package is `cellresponse` (as in the
brief) and the repository is `expression`.
