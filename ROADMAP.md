# Roadmap

Development plan for the brief in `cellular_response_prototype_brief.md`, adapted to the
facts established in M0 (`reports/data_audit.md`, `reports/design_audit.md`) and to the
constraints in `DECISIONS.md`: a MacBook Pro (M3 Pro, 18 GB RAM, CPU JAX) and JAX/Equinox.

Each milestone ends at a **landmark**: a concrete, checkable artifact plus a go/no-go gate.
A failed gate is a result to record, not a reason to change the test set.

## Status

| Milestone | State | Landmark |
| --- | --- | --- |
| M0 Audit | **Done** | Reproducible audit, verified cell-location map, feasible panels, footprints |
| M1a Cohort freeze + data prep | **Done** | Prototype cohort on disk; counts match for 2,436,792 cells |
| M1b Benchmark contract | **Done** | Splits, goals, queries, and the representation gate (direct module scores) |
| M2 Baselines | Planned | Validation metrics for untreated, shared-shift, ridge, Gaussian |
| M3 Distribution models | Planned | Mixture and flow selected on validation |
| M4 Final evaluation | Planned | Locked rankings, test + replicate audits, three seeds |
| M5 Portfolio | Planned | Explorer, model card, technical report |

## What M0 changed

- **The design is blocked by plate.** 379 compounds form four groups of ~94; each group sits on
  three plates, one dose per plate. Dose is confounded with plate and group with plate triplet.
  The primary search unit is therefore one (line, plate) block ranking ~93 compounds at one dose
  against that plate's two vehicle wells. Cross-dose and cross-group rankings are out of scope.
- **Replication exists in two places.** Plate 14 repeats all 94 plate-6 compounds (group B, 5 µM).
  Plate 13, which the brief does not mention, holds 58 repeat wells and 36 plate-13-only conditions.
  Adagrasib 0.05 µM is on all 14 plates.
- **Selective download works.** Every retained cell maps exactly to `(shard, row_group, row)`.
  A 1.4M-cell cohort is ~27 GB of row groups rather than the 338 GB atlas.
- **Compound identity needs structure groups.** 22 salt/free-base or stereo-free duplicate groups
  must stay together in every split.

## M1a: cohort freeze and data preparation

Cohort (see `DECISIONS.md` D6): the 12 best-supported lines (PANC-1, SW480, A498, HOP62, J82,
NCI-H460, LoVo, A549, CFPAC-1, LOX-IMVI, NCI-H2347, HT-29); compound groups B and C (plates 4–9)
plus plate 14; ≤256 treated cells per (well, line); all vehicle cells.

1. `cellresponse prepare --config configs/smoke.yaml`: fetch the smoke cohort (~1 GB, 10k cells),
   parse with marker checks, write sparse count shards plus `cells`, `conditions`, `compounds`,
   `genes` Parquet and a manifest with checksums.
2. Profile throughput (rows/s, MB/s, 429 rate) and adjust download concurrency. Use an HF token.
3. `cellresponse prepare --config configs/prototype.yaml`: ~47 GB of row groups, ~2.4M cells,
   resumable, written as zstd Parquet subsets (~6 GB local), then raw row groups discarded.

**Landmark M1a:** prototype subset reproducible from manifests. Re-running `prepare` produces
identical cell IDs and checksums, and per-cell totals match `tscp_count` for 100% of cells.
*Laptop budget:* ~1.5 h transfer, <10 GB disk, <8 GB RAM peak.

## M1b: benchmark contract (freeze before any preprocessing is fitted)

1. **Splits** (`cellresponse split`), written to `splits.parquet` with seed and hash:
   - A: unseen (line, compound-structure-group) pairs, 70/15/15, all doses/wells kept together,
     balanced so every test line and compound has training partners.
   - A2: query blocks per (line, plate) over held-out candidates; ≥10 per query or the block is dropped.
   - B: plate 14 held out of all fitting; rows labeled "seen condition, new plate" vs "unseen pair, new plate".
   - C: unseen compound structure groups (secondary; separate preprocessing).
   - Vehicle wells split into reference and evaluation roles per (plate, line): one well each.
2. **Goals** (`goals.py`): 3–5 versioned modules fixed from literature and training data only,
   such as G2M/proliferation down, p53/DNA-damage response, an apoptosis/stress constraint, and an
   interferon or differentiation program. Thresholds come from training vehicle cells.
3. **Representation** (`features/`): library size → normalize 1e4 → log1p → 2,000 HVGs balanced
   across training conditions → PCA-64 fitted on a balanced training subsample → scaling. Fitted
   per split protocol; never on test-treated cells. PCA is fitted on a ≤200k-cell dense subsample
   (≤1.6 GB) to fit in RAM.
4. **Representation gate:** on validation, raw versus PCA-reconstructed goal scores need median
   |Δ event fraction| ≤ 0.05 and rank correlation ≥ 0.8. On failure, switch inverse search to
   directly modeled module scores.
5. **Queries:** `queries/dev.json` and `queries/test.json` with context, reference IDs, panel,
   goal, and scoring. No outcomes.

**Landmark M1b:** `pytest` leakage suite passes (pair grouping, structure groups, plate-14
exclusion, train-only fitting, control/target disjointness, query files without outcomes), and
the representation gate result is recorded in `reports/representation.md`.

**Gate result (2026-10-17).** Median absolute event-fraction error is within 0.05 for all four
goals. Rank correlation is not: 0.64 for G2/M, 0.68 for p53, 0.70 for apoptosis, and 0.83 for
interferon-gamma. Only 17–29% of each module's genes are among the 2,000 HVGs. Inverse search
will model the module scores directly. PCA-64 remains the representation for distributional
baselines. The test queries were not changed after the gate. Rankings after this point use
the soft membership in D11, not the hard event this gate measured.

## M2: baselines (CPU; NumPy/scikit-learn plus small JAX models)

Untreated resampling; shared compound-dose shift; ridge mean with empirical residuals (vehicle
residuals and a pooled treated-residual bank); chemical-neighbor transfer for protocol C;
conditional Gaussian (diagonal + rank-8) in Equinox. Energy score, mean-shift error, marginal
Wasserstein, covariance discrepancies, sliced Wasserstein, and search metrics on **validation only**.

**Landmark M2:** `reports/baselines.md` with per-condition validation metrics, plus the synthetic
metric suite (correct-mean/wrong-variance, correct-marginals/wrong-dependence, equal-means/
different-tails ranking) passing. *Laptop budget:* minutes per model.

## M3: distribution models (JAX/Equinox/FlowJAX on CPU)

Four-component conditional mixture; FlowJAX rational-quadratic spline coupling flow (6 layers,
8 knots, 2×256 conditioners, 64-d residuals; 1.84M parameters, checked in M0). Condition encoder:
control mean/log-SD, Morgan-2048 → 128 projection, `log1p(dose/0.05)`. Training samples conditions
uniformly, then wells, then cells. AdamW 3e-4, clip 1.0, early stopping on validation energy score.
≤12 configurations total.

**Landmark M3:** frozen configs and checkpoints; numerical checks (inverse consistency, finite
log-det) pass; validation comparison against the best M2 baseline written up. Benchmark the step
time on 100k cells first and extrapolate before any long run (target: <1 h per run).

## M4: final evaluation

Three seeds for the finalists. Rankings for `queries/test.json` are generated and hashed **before**
outcomes are loaded. Primary metric: top-1 observed utility versus the strongest validation-
selected baseline (gate: +5 pp with a paired interval and consistent seeds). Supporting: energy
score (−5% gate), mean-response non-inferiority, plate 6→14 transfer, plate-13 repeats, and the
Adagrasib anchor across plates.

**Landmark M4:** `reports/benchmark.md` including failures and cases where no candidate beats
vehicle; replicate audit reported regardless of direction.

## M5: portfolio

Static explorer (`app/`) over precomputed predictions: context + goal → ranked panel, untreated/
predicted/measured distributions, support flags, retrospective reveal. Model card and technical
write-up. Examples chosen by rule (median, strong, poor, high-disagreement, no-response).

**Landmark M5:** a fresh clone reproduces the reported tables from manifests, and the explorer
refuses unsupported comparisons (cross-dose, cross-group, unseen lines).

## Later, only if earlier gates pass

Plate-13 cross-group calibration; groups A and D (adds ~40 GB); 128-PC or autoencoder
representation; set encoder for controls; count (NB) decoder; protocol D (unseen lines).
