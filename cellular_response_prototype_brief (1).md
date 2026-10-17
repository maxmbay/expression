# Context-Conditioned Perturbation Search

**Prototype design and implementation handoff**  
Prepared for Maxwell Bay · Revised 2026-10-17 · Version 2.0

## 1. Project mandate

Build a public research portfolio project that **ranks supported perturbations by how well their predicted cellular-response distributions satisfy a specified goal in a fixed biological context**. A forward distribution model is the engine; inverse search is the product and principal benchmark. Use Tahoe-100M first. Deliver an audited benchmark, strong simple baselines, a compact conditional generative model, and a goal-driven explorer with retrospective validation.

Version 2.0 changes the scope from forward prediction to decision support. The first release searches a finite compound panel at a fixed measured dose and assay context. Joint compound–dose search across batches, novel molecule generation, and causal discovery are later extensions requiring additional evidence.

The owner is a senior data scientist with a neuroscience PhD, strong probabilistic modeling experience, and production ML experience. The project should demonstrate scientific judgment, distributional modeling, rigorous evaluation, and competent data engineering. A credible negative result is preferable to an impressive-looking but invalid benchmark.

**Central research question:** Given a compatible vehicle-control population and a prespecified target state, can a model rank withheld treatments within the same cell line, assay context, and dose better than strong baselines—and does modeling population heterogeneity improve that ranking?

**Primary hypothesis:** Distribution-aware search improves observed target-state attainment for selected held-out treatments compared with mean-based and simple distributional alternatives. Improved forward energy score is supporting evidence, not sufficient evidence of useful search.

**Default scope:** A single GPU, approximately 1–2 million selected cells, 12 cell lines, and approximately 160 compounds. These are design targets, not confirmed coverage or measured resource requirements. Audit the data before fixing the cohort. Do not begin by downloading or training on the entire atlas.

**First deliverable for the implementing agent:** a small, executable data and identifiability audit, feasible candidate panels, and a frozen split/query manifest. If comparable panels cannot be formed, narrow the question or choose a better-supported dataset before model implementation.

## 2. Evidence, corrections, and unresolved facts

Sources were reviewed on 2026-10-16/17. No expression dataset was downloaded and no model was trained while preparing this brief. Hyperparameters, cohort sizes, resource budgets, and acceptance criteria below are proposed defaults.

### Verified experimental context

The experiment used approximately 379 distinct compounds at three concentrations, 0.05, 0.5, and 5 µM, with a 24-hour endpoint. Fifty cancer cell lines were pooled in three-dimensional cultures. The design included 14 plates and vehicle-control wells; plates 6 and 14 form a documented technical-repeat comparison. These are limited technical repeats, not extensive independent biological replication. [S2]

**Correction to the initial brainstorming:** the approximately 1,100 figure refers to drug–dose conditions, not 1,100 chemically distinct drugs. The publication describes a filtered analysis with 47 cell lines, 379 compounds, and 1,135 drug–dose combinations. Dataset releases and filtering stages differ. [S3]

### Earlier-paper status and related evidence

The owner recalls an earlier discussion of confounding in LINCS/L1000. The exact paper was not recovered from prior-chat retrieval; do not attribute the following sources to that conversation. Two verified related studies concern batch effects in L1000 profiles [S9] and shared toxicity/proliferation signatures that can obscure mechanism-specific similarity [S10]. A closer candidate, *Inductive Biases for Disentangled Representation Learning with Correlated Treatment–Nuisance Factors*, was found through indexed OpenReview text; full retrieval was blocked and its publication status remains unverified [S11]. Resolve that reference if the owner supplies another clue, without blocking the audit.

Tahoe remains the primary dataset, subject to the audit below. L1000 measures bulk profiles; a collection of its replicate profiles is not a sample of single-cell response heterogeneity [S12]. Do not mix it into the primary distribution target. Any later use as auxiliary supervision needs separate assay, context, and provenance handling.

### Preferred data access

Start with the official `tahoebio/Tahoe-100M` Hugging Face dataset. Its expression configuration currently displays approximately 95.6 million rows; the dataset card labels the release CC0-1.0. Pin a revision and record the downloaded license. Do not interpret the site's total row count across configurations as the number of cells. [S1]

Arc also distributes AnnData matrices and Parquet metadata. Its documentation now points to a Google Cloud Marketplace bucket and flags requester-pays requirements; the older bucket was scheduled for deprecation in March 2026. Prefer the Hugging Face route initially, and audit actual transfer costs and permissions before using cloud alternatives. [S4]

### Facts the agent must resolve before modeling

- Exact release, file/shard organization, download sizes, and whether selection can avoid scanning most of the atlas.
- Exact mapping among treatment well, `sample`, plate, sublibrary, cell identifier, and cell line.
- Which cells already passed which QC filters; whether raw counts are available in every selected source.
- Availability of adequate plate- and cell-line-matched vehicle controls.
- Exact repeated conditions on plates 6 and 14, including concentration and processing differences.
- Whether more independent repeats exist; never infer experimental replication from multiple sublibraries or many cells.
- Chemical aliases, duplicate structures, mixtures, invalid SMILES, and actual condition coverage.

If the documented structure differs from observed files, preserve the discrepancy in `reports/data_audit.md` and revise the configuration. Do not silently force the data into this design.

## 3. Estimand and boundaries of the scientific claim

Let `X` be the measured RNA count vector of a cell, `c` its cell line, `d` a compound, `a` its dose, and `C0(c,b)` a vehicle-control reference population from compatible experimental context `b`.

**Vehicle endpoint is not measured pretreatment state.** Treat Tahoe controls as contemporaneous vehicle endpoints unless metadata explicitly establishes time-zero sampling. They supply a population reference under vehicle, not the same cells before exposure. Record culture/media, exposure duration, vehicle, co-treatments, and prior-treatment history when available; unknown history stays unknown.

Version 0.1 predicts:

`P(Z | C0(c,b), structure(d), a, assay protocol)`, where `Z = T(X)`.

`T` is a fixed, training-fitted transformation: library normalization, log transformation, gene selection, PCA, and scaling. The target is the **joint distribution in this transcriptional representation**, including changes in location, spread, dependence, and population shape.

This is deliberately narrower than a complete RNA-count simulator. Inverse PCA provides projected expression summaries, not faithful full-dimensional expression samples. The interface and paper must say so. Section 13 specifies the count-model extension.

The target population consists of cells captured and retained after the assay's processing and QC. It does not include all cells that existed before treatment. Treatment-induced death, proliferation, capture, and QC can change which cells appear. Consequently:

- Do not interpret the output as absolute survival, viability, or a population growth rate.
- Do not infer an individual cell's counterfactual trajectory: control and treated cells are unpaired.
- Do not claim to separate pure biological variability from measurement variation using this dataset alone.
- Do not claim transfer to patient tissue, other assay protocols, time points, or drug combinations.
- Do not call a prediction for an unseen pair a prediction for an unseen drug.

Available untreated reference cells are part of the intended inference input. Models must never receive treated target expression, target cell-cycle scores, target aggregate QC, or target-derived embeddings as conditioning features.

### Experimental design and confounding audit

The target is a conditional population response. Cell identity is often a legitimate effect modifier: conditioning on it is necessary, while predicting it can inflate an inappropriate global metric. Do not automatically erase all cell-line information or call every biological difference a nuisance.

| Threat | How it could mislead inverse search | Required response |
| --- | --- | --- |
| Unequal line–drug coverage | A desired baseline is mistaken for a drug-induced change | Fix line/context within each ranking; measure improvement against vehicle; retain a no-treatment baseline |
| Compound/dose allocated by plate or batch | Drug identity acts as a batch label | Build compound × dose × plate incidence tables and overlap graphs; identify estimable comparisons; use shared matched controls and comparable candidate panels |
| Media, pretreatment, exposure-time differences | A protocol difference is attributed to the candidate | Include known history in the context contract; stratify compatible protocols; report missing metadata |
| Shared stress, proliferation, or death signatures | Generic damage resembles the desired signature | Prespecify undesirable-expression constraints when appropriate; distinguish downstream mediators from pretreatment confounders; do not automatically regress them away |
| Survival/capture/QC selection | Removing an unwanted subpopulation appears to reprogram it | Define outcomes among captured, retained cells; report recovery and attrition; no viability claim from transcriptomes alone |
| Leakage or population imbalance | Strong prediction reflects shared assay information or abundant lines | Hold out intervention groups, fit transforms on training data, macro-average, and disclose shared pooled wells |

Audit overlap before selecting a correction method. Matched controls help with shared shifts, but cannot by themselves separate treatment-specific batch interactions or repair perfect treatment–batch confounding. A nuisance embedding, adversarial objective, or normalization algorithm cannot establish an unidentified effect without additional assumptions or data. Record those assumptions explicitly.

Produce `reports/design_audit.md`: experimental-unit hierarchy; incidence/overlap plots; known and missing protocol fields; support by line/compound/dose; which comparisons survive; and which prospective claims remain unsupported. Probe predictability of batch and treatment from representations as a diagnostic, not a proof of causal validity. Treatment-label permutation must respect genuinely exchangeable experimental blocks; do not manufacture an exchangeability assumption when each treatment occupies a unique batch.

### Inverse-search contract

Do not train `p(drug | desired_expression)` as the primary solution: that can inherit experimental assignment frequencies and cell-context shortcuts. Enumerate a supported candidate set, generate each candidate's conditional response, and score the same goal. The model need not be mathematically invertible.

A frozen query contains:

- Context: line, reference-control IDs, endpoint time, protocol and known treatment history, plus support status.
- Candidate panel: explicit compound IDs/structures and measured dose; no candidate-specific reference substitutions inside a ranking.
- Goal: versioned gene/module definitions, direction, threshold, and optional undesired-state constraint; all fixed before test outcomes are examined.
- Scoring: utility definition, constraint thresholds, model/sample seeds, and any disagreement penalty chosen on validation.

Default utility is the fraction of captured cells jointly satisfying a desired module threshold and an undesired-module limit. For `u_q(z)` equal to that event indicator:

`U_q(d,a) = E[u_q(Z_d,a) | C0, context] - E[u_q(Z_vehicle) | C0, context]`.

The vehicle term is common to candidates; it makes improvement interpretable and supports a no-treatment option. Differences of marginal expectations are estimable population contrasts under the stated design assumptions. They do not identify which individual cells would improve under treatment. A model-predicted cell fraction is also not the probability that a new experiment will meet a target fraction; the latter needs an experiment-level uncertainty model and adequate repeats.

Use three to five prespecified, interpretable module goals initially, with exact gene membership and scoring code committed before final evaluation. Derive thresholds from permitted controls/training data. Choose the goals for interpretability and validation representation fidelity, not because test compounds achieve them. Optional later utilities include distance to an independent target distribution or cross-context selectivity. Keep target-reference samples separate from evaluation labels; matching a different cell type's global expression is not itself evidence of reprogramming.

**Representation gate:** score observed outcomes using actual normalized expression/module values, not their PCA reconstructions. On validation, compare raw and reconstructed goal scores, including event fractions and candidate rankings. Proposed gate: median absolute error in event fractions at most 0.05 and rank correlation at least 0.8 where rankings have nontrivial variation; report tails and exclusions too. These are engineering defaults. If the gate fails, use a directly modeled low-dimensional space of the prespecified module scores for inverse search, or revise the representation before test unblinding. Do not claim support for arbitrary gene-expression goals.

Return predicted utility, expected target fraction, ensemble disagreement, Monte Carlo error, and support/constraint flags separately. Default ranking uses expected utility; an optional validation-fixed disagreement penalty is a heuristic, not a calibrated lower confidence bound. Enumerating a few hundred candidates needs no reinforcement learning or molecule generator.

## 4. Scope and staged cohort

| Stage | Proposed cohort | Purpose |
| --- | --- | --- |
| Smoke test | 4 lines, 12 compounds, one observed dose, at most 128 cells per condition | Validate parsing, splits, losses, sampling, and metrics |
| Main prototype | 12 lines, approximately 160 compounds, available standard doses, at most 256 treated cells per condition per well | Test conditional distribution prediction on roughly 1–2 million cells |
| Replication audit | Verified overlapping conditions on plates 6 and 14, restricted to eligible lines | Assess technical reproducibility and transfer between plates |
| Scale extension | More compounds, lines, and cells, using the same pipeline | Establish learning curves only after the prototype is informative |

Select lines by control availability and coverage, with tissue diversity as a secondary consideration. Select compounds using metadata to obtain dense, comparable panels within plate/dose blocks; structural diversity is secondary. Require at least ten eligible held-out candidates per primary ranking query. Adjust cohort size or split proportions before freezing if needed. Do not select drugs based on favorable held-out responses.

Proposed eligibility: at least 128 retained treated cells in an evaluated condition and at least 256 suitable controls across available reference wells. Retain the available individual control wells separately. Reserve enough controls for evaluation; do not count cells reused by resampling as additional observations.

These thresholds make distributional evaluation practical but introduce selection toward well-recovered conditions. Report exclusions and recovery counts. Excluded or nearly depleted conditions are not evidence of no response. Do not score a rare-cell claim on a cohort capped at 256 cells per condition.

## 5. Data implementation contract

### Source parsing

The official expression rows contain aligned `genes` and `expressions` arrays, a cell key, treatment/sample identifiers, cell-line identifiers, and plate information. The leading array element is a marker, not a biological count. Use gene metadata to resolve token IDs. `DMSO_TF` identifies vehicle controls; match on line and plate. Dose is available through sample metadata, including `drugname_drugconc`. Some mechanism annotations were model-assisted; omit them from primary predictors. [S1]

The Arc documentation also lists sublibrary and QC fields, including `pass_filter`. Preserve them for auditing and sensitivity analyses. [S4]

Implementation requirements:

1. Discover dataset configurations and schemas at the pinned revision; source examples may contain typos. Verify the actual loader arguments on a tiny read.
2. Parse treatment metadata with a safe structured parser; never use Python `eval`.
3. Normalize concentration units to µM. Keep the original string and parsed values.
4. Verify that the marker is removed by position/schema, then reject remaining negative or non-finite counts. Do not indiscriminately drop all negative values to conceal parsing errors.
5. Map gene IDs once; validate array alignment, duplicate IDs, and unique cell identifiers.
6. Maintain sparse counts on disk and densify only minibatches or bounded preprocessing chunks.
7. Use deterministic hash-based sampling within groups. Record both available and retained counts.
8. Preserve source QC first. Additional thresholds must be chosen from training data and documented; test sensitivity because treatment itself can affect QC measurements.

### Canonical local tables

| Artifact | Minimum fields |
| --- | --- |
| `cells.parquet` | cell key, sample/well key, line ID, compound ID, dose, control flag, plate, sublibrary, source shard/row reference, QC flags, full library size |
| `conditions.parquet` | line, compound, dose, well/sample, plate, available/retained cell counts, control-reference IDs, split role |
| `compounds.parquet` | compound ID, source name, canonical structure, structure hash, scaffold group, fingerprint validity |
| `genes.parquet` | original token ID, stable gene ID, symbol, feature-panel membership and order |
| `splits.parquet` | cell/condition IDs, protocol, train/validation/test/audit/control-reference role, split seed |
| `manifest.json` | source URLs/revision, file checksums, license, configuration hash, preprocessing hash, exclusion counts |

Create an explicit condition key such as `(line_id, compound_id, dose_uM, sample_id)`. Do not use a drug-name string as the only join key. Treatment wells contain multiple lines: the well is an experimental unit, while line-specific outcomes are nested measurements.

### Access and storage strategy

Download metadata first, then inspect Parquet statistics and shard layout. A streaming filter may still read every preceding row; streaming is not indexed access. Compare actual bytes needed for a representative subset before selecting a transfer method. If selective retrieval is impractical, perform one resumable pass with deterministic stratified sampling and checkpoint the resulting subset.

Do not materialize all genes for all cells as a dense matrix. Store counts as CSR/AnnData or equivalent sparse shards; keep metadata in Parquet and transformed features in chunked arrays. A 1.5-million-by-64 float32 feature array is approximately 384 MB, while a dense 1.5-million-by-2,000 float32 expression array is approximately 12 GB before overhead. Raw download size may be far larger than either.

## 6. Evaluation splits: freeze before preprocessing

### A. Primary: unseen line–compound pairs

Partition eligible `(line, compound)` pairs approximately 70/15/15 into training, validation, and test. Keep **all doses, wells, and repeats for a pair together**. Constrain the split so every test line and compound has sufficient training coverage with other partners. Publish the deterministic construction algorithm and resulting coverage table.

This is matrix completion across known biological contexts and known compounds. Other cell lines from the same pooled treatment well may appear in training. That is permitted for this explicitly defined benchmark, but it means the test is not an independent new experiment. Name the setting accurately and accompany it with the replication audit below.

### A2. Primary search panels within pair holdout

Construct queries within a fixed `(line, plate, dose, endpoint/protocol)` block. Rank only test-held-out candidates for that query, all using exactly the same eligible control reference. Require at least ten candidates and disclose panel size. The corresponding line–compound pairs remain withheld across every dose and repeat under protocol A. Check candidate availability from metadata, not outcome magnitude. A small smoke test may omit search-performance claims.

This estimates retrospective ranking within the documented assay, not arbitrary prospective drug choice. Dose may be entangled with plate: do not combine separate fixed-dose rankings into an evaluated dose optimizer. For a prospective common-reference search across plates, fix a single reference for all candidates and separately validate reference transfer and cross-batch comparability using overlap/repeats. If those checks are unsupported, label that use unevaluated and leave it out of the headline demo.

### B. Technical transfer: held-out replicate plate

Reserve all treated cells from plate 14 outside model fitting, hyperparameter tuning, gene selection, and PCA fitting. Use plate 6 as the corresponding reference where verified overlap exists. Exact plate assignment may change only after the metadata audit confirms the documented pair.

Separate audit rows into:

- **Seen condition, new plate:** the same line–compound–dose was seen in training on the other plate. Measures technical transfer, not biological novelty.
- **Unseen pair, new plate:** the pair was withheld everywhere. Measures a harder combination of pair generalization and technical transfer.

If plate 14 alone does not cover enough test pairs, report the smaller eligible audit explicitly. Do not present a few successful drugs as a broad replication result. All dose copies of an unseen pair remain withheld, including plate 6 copies.

### C. Secondary: unseen compounds

Create a separate protocol that holds out entire standardized compounds across all lines, doses, and plates. Start with a structure-grouped split; add scaffold-grouped evaluation if enough distinct groups survive. Document handling of salts, aliases, acyclic molecules, and large scaffold groups. Do not advertise a generic scaffold function as a perfect chemical novelty test.

This protocol requires a separately fitted model and preprocessing pipeline. No held-out compound expression may enter unsupervised fitting. Metadata/structure alone is allowed. Report structural nearest-neighbor similarity to training compounds to contextualize performance.

### D. Stretch: unseen cell line with controls

Hold out all treated cells from selected lines; provide a fixed untreated reference from each new line at inference. Describe this as **control-conditioned cell-line transfer**, not zero-information transfer. No learned line-ID lookup is available for a new line. Refit all learned preprocessing on permitted training data.

### Control-reference rules

Partition controls into reference and evaluation roles by well when adequate controls exist; otherwise use disjoint cell subsets and label the weaker design. Reference controls are available to all models under the same rules. They may come from a test plate because the intended task assumes matched untreated measurements.

For a held-out line or plate, reference controls can condition predictions but cannot update PCA, feature selection, model parameters, or hyperparameters. Evaluate robustness to using a pooled historical reference instead. For control-as-target training examples, use cross-fitting so a cell never helps construct its own reference summary.

### Prohibited shortcuts

- Random cell splits as the headline result.
- Shared cells between reference and evaluation populations.
- Treating different sublibraries of a well as independent biological repeats.
- Fitting PCA/HVGs/batch correction on test-treated cells, even without labels.
- Using a pretrained Tahoe model without establishing its overlap with held-out data.
- Choosing preprocessing, favorable subsets, or showcase cases after inspecting final test scores.

## 7. Representation and prediction model

### Fixed representation for version 0.1

Use training cells and permitted training controls only:

1. Remove non-biological marker entries; compute full library size across retained biological genes.
2. Normalize each cell to a total of 10,000 and apply `log1p`. Keep raw counts unchanged in storage.
3. Select 2,000 highly variable genes using a documented, count-compatible method with balanced contribution across training conditions. Do not let abundant lines dominate selection. Keep a separate fixed biological marker panel for diagnostics.
4. Fit centered PCA with 64 components on a balanced training subsample. Do not regress out cell cycle: it may be part of the response.
5. Scale PCs using training standard deviations with a documented variance floor. Preserve all transform parameters and gene order.

Fit a 128-component sensitivity run only if 64 components discard substantial validation treatment signal. Compare models within a shared representation; likelihood values across different representations are not directly comparable.

For a condition, compute the mean and log standard deviation of the reference controls in PC space. Denote the mean `m0` and define the treated residual `R = Z - m0`. This subtracts a population mean; it does not pair individual cells.

Before fitting a generative model, quantify how well the representation preserves validation treatment mean shifts, covariance changes, and prespecified marker/pathway summaries. PCA can retain cell-line identity while losing subtle responses. Low reconstruction fidelity limits biological interpretation; it is not a reason to tune on test data.

### Main model: conditional residual normalizing flow

Use a small rational-quadratic spline coupling flow:

- 64-dimensional residual target.
- Six coupling blocks, alternating masks/permutations.
- Eight spline bins, linear tails, two 256-unit hidden layers per conditioner.
- Standard normal base distribution; positive minimum widths, heights, and derivatives.
- Conditional inputs: control mean and log-SD; a 128-dimensional MLP projection of a 2,048-bit radius-2 molecular fingerprint; transformed dose; control/treatment flag.
- Dose feature: `log1p(dose_uM / 0.05)`, with explicit vehicle handling. Do not assume monotone gene responses or interpolate outside supported concentrations in the demo.
- Exclude line, drug, plate, sample, and sublibrary ID embeddings in the primary model. The controls can still carry batch information; excluding IDs does not solve confounding.

Use an established spline implementation and pin its version after a numerical smoke test. Start in float32. Mixed precision is optional only after checking inverse consistency and finite log determinants.

Objective:

`L = mean_conditions[ mean_cells[ -log p_theta(Z_i - m0 | h_condition) ] ]`.

Sample conditions uniformly, then cells within condition. If multiple wells exist, sample wells before cells so cell-rich wells do not dominate. Train control-as-target cases with cross-fitted controls as a negative-control diagnostic.

Initial training defaults: AdamW, learning rate `3e-4`, weight decay `1e-4`, effective batch size 1,024, gradient clipping at 1.0, at most 50 passes through the selected cohort. Save checkpoints and stop early on validation distributional score. These are starting values, not validated settings. Restrict the first search to model depth, width, and learning rate; keep the validation budget comparable across models.

Generate residuals, add `m0`, and return samples in the fixed PC representation. Do not identify one generated cell with one real cell. Inverse-PCA displays must be labeled projected normalized expression; do not clip negative reconstructed values inside evaluation to make them look biological.

### Why this architecture first

It gives tractable conditional density, easy sampling, modest training requirements, and a direct comparison with Gaussian alternatives. It separates the scientific question—does distribution shape generalize?—from training a large expression encoder. A flow is a hypothesis to test, not a requirement for a successful project.

## 8. Required baselines and ablations

All baselines receive the same reference controls, training data, feature information, and split. Fit strong baselines before the flow. Existing perturbation benchmarks provide reason to take simple methods seriously, although results on genetic perturbations do not establish what will happen on Tahoe's chemical perturbations. [S5]

| Model | Implementation | What it tests |
| --- | --- | --- |
| Untreated resampling | Sample the matched control population | Whether predicting any treatment effect helps |
| Shared treatment shift | Estimate a compound–dose mean residual across other training lines; add it to target-line controls | Whether cell-specific nonlinear interactions are necessary; only valid when compound is seen |
| Ridge mean + empirical residuals | Predict condition mean residual from control summaries, chemical features, dose, and a documented low-rank interaction map; add centered target-control residuals | Whether a strong mean model plus realistic untreated heterogeneity is sufficient |
| Chemical-neighbor transfer | Transfer residual samples from structurally similar training compounds, matching dose/context as closely as allowed | A demanding baseline for unseen compounds |
| Conditional Gaussian | Same condition encoder; predict mean and positive diagonal plus rank-8 covariance | Whether changing covariance explains the gain |
| Conditional mixture | Four Gaussian components, same inputs; regularized variances and weights | Whether a simpler multimodal model matches the flow |
| Conditional flow | Main model from section 7 | Whether more flexible joint density adds reproducible value |

Use two ridge residual variants if feasible: untreated empirical residuals and a pooled training-treated residual bank, selected on validation. This prevents an artificially weak distributional comparator.

Required ablations: no chemical features; pooled versus plate-matched controls; mean-only control summaries versus mean plus dispersion; Gaussian versus mixture versus flow under comparable capacity. Repeat the winning baseline and main model with three seeds.

Optional external comparator: CPA/chemCPA trained on exactly the same cohort and split. CPA models compositional perturbation responses; chemCPA adds chemical information for new compounds. [S6] Arc State and its evaluation tooling are useful references, but Tahoe-pretrained weights must not be presented as an uncontaminated held-out baseline. [S7]

## 9. Evaluation specification

### Primary: quality of the selected perturbations

Freeze query definitions, panels, scoring rules, and model selection before final test evaluation. The query manifest must never contain candidate treated outcomes as inference inputs. Generate rankings first; reveal held-out measurements only to the evaluator.

For each query, measure observed goal attainment and improvement over vehicle for the top-ranked candidate. Compare with random selection, the no-treatment option, mean-based ranking, and ranking from each strong distributional baseline. A mean-based scorer applies the goal to the predicted mean; distinguish this from estimating the fraction of cells satisfying it. The central test is whether distribution-aware decisions improve observed attainment, not whether a more complex sampler wins a visual comparison.

Report top-1 observed utility as the primary decision metric; also report average observed utility of the selected top five, hit enrichment at a predeclared threshold, ranking correlation, and regret relative to the best measured candidate in the same panel. The best observed candidate is a noisy retrospective reference, not a known biological optimum. Do not report the best of five as equivalent to selecting one: it assumes a larger experimental budget. Always include the no-treatment option and report how often the system selects it.

Macro-average queries, with sensitivity to shared lines, compounds, plates, and goals. Bootstrap at meaningful experimental/group levels and disclose dependencies; several goals on the same plate are not independent experiments. Where repeats support it, evaluate a locked ranking against the independent repeat measurements and report ranking/utility stability. Same-well cell resampling measures sampling uncertainty only.

Proposed practical success gate: at least a five-percentage-point improvement in observed top-1 target attainment over the strongest validation-selected baseline on the locked queries, a paired interval favoring improvement, and consistent direction across three seeds. For utilities with other scales, predeclare an analogous threshold using validation. Also require no material degradation in the stated undesirable-state criterion. Report absolute performance and cases where no candidate beats vehicle. Limited panels/repeats may support only a descriptive result; do not manufacture significance.

### Supporting forward score

Use a condition-macro-averaged **energy score** in the fixed standardized PC representation. For real cells `y_j` and generated samples `x_i`:

`ES = mean_(i,j) ||x_i - y_j||_2 - 0.5 * mean_(i != k) ||x_i - x_k||_2`.

Lower is better. Exclude diagonal self-pairs in the generated-sample term. The same observed population must be used for every model. Match sample counts—default 128 real and 128 generated per condition—and repeat deterministic subsamples five times for Monte Carlo stability. Do not confuse these repetitions with additional experiments.

Energy score can be insensitive to some dependence errors in high dimensions, so the following diagnostics are required rather than optional decorations:

| Aspect | Required diagnostic |
| --- | --- |
| Mean response | Error and correlation of treated-minus-control PC means; projected gene-level mean shifts as secondary output |
| Marginal distributions | CRPS or Wasserstein-1 on prespecified PC directions and sufficiently well-reconstructed biological summaries |
| Spread/dependence | Variance-ratio errors and shrinkage covariance/correlation discrepancies; avoid interpreting unstable sample covariance as truth |
| Population shape | Sliced Wasserstein distance over fixed training-independent random directions; examples of mode or tail failures |
| Density | Held-out NLL for comparable density models in the same representation; empirical-resampling baselines receive no invented likelihood |
| Negative controls | Control prediction and randomized-treatment-label checks; representation and plate diagnostics |
| Biological response | Prespecified cell-cycle/stress/pathway summaries and response ranking, with projection-fidelity checks |

Do not lead with correlation of whole expression profiles: baseline cell identity can dominate it. Rank response genes using training information or use clearly labeled post-hoc descriptive analysis. Cell-level differential-expression p-values do not create biological replication.

### Replicate reference and uncertainty

Report three distinct quantities:

1. **Predictive variation across cells:** variation in generated transcriptional states; includes assay-associated variation.
2. **Model disagreement:** variation across three independently trained models; an uncertainty diagnostic, not a calibrated posterior.
3. **Finite-sample uncertainty in scores:** paired resampling over held-out compounds/conditions, with cell subsampling nested inside. Include sensitivity to shared wells and lines; limited plates prevent a strong claim about arbitrary future-batch uncertainty.

Calculate observed-versus-observed discrepancies using disjoint halves of a condition and independently using verified technical repeats. Call these sampling/replicate reference levels, not universal upper bounds. A same-well cell split cannot measure between-experiment reproducibility.

For interval diagnostics, evaluate empirical coverage on held-out marginal summaries at 50%, 80%, and 95%. Marginal cell-distribution intervals are not confidence intervals for a condition's mean. Do not present an ensemble envelope as calibrated biological uncertainty.

### Reporting and decision criteria

Report per-condition results, macro averages, and strata by line, compound similarity, dose, control availability, and response strength. Response-strength bins chosen from held-out observations are descriptive only, not a basis for choosing the winning model.

Supporting forward-model target: at least a 5% reduction in macro energy score relative to the strongest validation-selected distributional baseline on the primary test, with a paired interval favoring improvement, no material mean-response deterioration (default tolerance 5%), and consistent direction across seeds. Report absolute differences too; relative changes are unstable near zero. This threshold is an engineering target, not an established field standard.

Forward-model improvement alone does not establish useful inverse search. The technical-repeat audit must be reported even if it contradicts the main result. Failure to beat baselines is an acceptable completed study; do not respond by repeatedly changing the test cohort. Broad novel-drug claims require separate success under protocol C.

## 10. Software and reproducibility

Preferred stack: Python, PyTorch, Polars, PyArrow, NumPy/SciPy, AnnData/Scanpy for established single-cell utilities, scikit-learn, RDKit, YAML configuration, `uv` lockfile, and pytest for critical invariants. Use MLflow or a small local run manifest, whichever is simpler. No workflow orchestrator is necessary initially.

Organize the repository into modules rather than an analysis notebook:

| Path | Responsibility |
| --- | --- |
| `src/cellresponse/data/` | discovery, source adapters, schema validation, sampling, control matching |
| `src/cellresponse/splits.py` | group-aware protocol construction and leakage checks |
| `src/cellresponse/features/` | count transformation, gene panel, PCA, chemistry, controls |
| `src/cellresponse/models/` | shared interface, baselines, Gaussian, mixture, flow |
| `src/cellresponse/train.py` | reproducible training and checkpointing |
| `src/cellresponse/evaluate.py` | per-condition metrics, resampling, aggregation |
| `src/cellresponse/predict.py` | inference from allowed context only |
| `src/cellresponse/search.py` | query validation, candidate support, shared-reference ranking, abstention |
| `src/cellresponse/goals.py` | versioned module definitions, thresholds, utility and fidelity checks |
| `queries/` | frozen development/test query and candidate manifests without treated outcomes |
| `configs/` | smoke, prototype, split-specific experiments |
| `tests/` | parser, split, numerical, and metric contracts |
| `reports/` | audit, benchmark, model card, figures |
| `app/` | explorer consuming exported predictions |

Suggested CLI contract, to implement rather than assume already exists:

```bash
uv run cellresponse audit --config configs/smoke.yaml
uv run cellresponse prepare --config configs/prototype.yaml
uv run cellresponse split --config configs/pair_holdout.yaml
uv run cellresponse features --config configs/pair_holdout.yaml
uv run cellresponse train --model ridge_residual --config configs/pair_holdout.yaml
uv run cellresponse train --model gaussian --config configs/pair_holdout.yaml
uv run cellresponse train --model flow --config configs/pair_holdout.yaml
uv run cellresponse evaluate --run-id RUN_ID --partition validation
uv run cellresponse evaluate --run-id RUN_ID --partition test
uv run cellresponse search --run-id RUN_ID --queries queries/test.json --output rankings.parquet
uv run cellresponse evaluate-search --rankings rankings.parquet --partition test
uv run cellresponse export-demo --run-id RUN_ID
```

Minimum inference interface:

```python
predict_distribution(
    control_reference,       # raw counts + gene IDs, or validated cached features
    canonical_smiles,
    dose_uM,
    n_samples=512,
    seed=0,
) -> PredictionBundle
```

`PredictionBundle` contains PC samples, control summaries, feature/model versions, requested condition, support warnings, and optional projected-expression summaries. No target expression is passed to this function. For unseen chemistry or unsupported doses, return a flagged experimental prediction or reject the request; never silently present it as an evaluated setting.

Search API: `rank_perturbations(control_reference, context, candidates, goal_spec, model_bundle, seed=0) -> RankingBundle`. Validate compatibility before sampling. Return candidate ranks, utility components, uncertainty diagnostics, provenance, and support status; return an explicit unsupported result for invalid comparisons. Use identical reference inputs across candidates. Changing test outcome files must not change produced rankings.

Log source revision, exact cell IDs, split hash, feature hash, configuration, code commit, random seeds, package versions, training duration, hardware, peak memory, and evaluation sampling seeds. One fitted feature artifact belongs to one split protocol. Do not cache a global PCA and reuse it across biological holdouts.

Critical tests: marker removal, count alignment, dose conversion, unique keys, control/target disjointness, all-dose pair grouping, compound grouping, plate-14 exclusion, train-only feature fitting, stable fingerprint mapping, flow inverse/log-probability consistency, and synthetic metric cases. The metric suite must detect correct-mean/wrong-variance and correct-marginals/wrong-dependence examples. Search tests must detect candidate-specific reference swaps, post-treatment metadata inputs, outcome-file leakage, incompatible context/dose comparisons, and objective changes after query freezing. Include a synthetic case where equal means but different tails correctly change the ranking for a threshold goal.

## 11. Compute and execution plan

Assume access to one 24-GB CUDA GPU, 64 GB host RAM, and 200 GB of working SSD as a planning envelope. This is not a measured minimum or a statement that all source downloads fit. CPU-only baseline work is possible; the agent must profile data transfer and preprocessing before forecasting total runtime.

| Milestone | Work | Exit artifact |
| --- | --- | --- |
| M0: audit | Inspect metadata and a tiny expression sample; verify controls, design overlap, replicates, chemistry, and access costs | `data_audit.md`, `design_audit.md`, source manifest, feasible candidate panels |
| M1: benchmark contract | Freeze cohort/splits and goals; build transformation; validate goal fidelity | Versioned split/features/query manifests plus leakage-check report |
| M2: falsifiable baseline | Fit untreated, shared-shift, ridge-residual, and Gaussian models | Per-condition validation metrics and baseline report |
| M3: distribution model | Fit mixture and flow; run limited ablations; choose hyperparameters on validation | Frozen model configurations and selected checkpoints |
| M4: final evaluation | Run three seeds; lock rankings before revealing outcomes; run test/replicate audits | Forward and search benchmarks, uncertainty diagnostics, failure analysis |
| M5: portfolio | Export representative cases; build explorer and technical narrative | Reproducible repository, model card, demo, report |

Planning estimate: roughly 4–6 weeks of part-time work for the full prototype, highly dependent on data access and integration. This is not a delivery promise. First benchmark 100,000 cells and extrapolate measured throughput. Limit initial tuning to approximately 12 configurations total, then three seeds for finalists. Avoid large-scale training until data and metric gates pass.

Keep a small synthetic fixture in the repository so CI can run without downloading Tahoe. Checkpoint prepared subsets and expensive feature artifacts. Publish download manifests and preparation code; do not put the full dataset in Git.

## 12. Public demo and final research narrative

The explorer opens with a supported context and a goal. It ranks a fixed candidate panel, shows predicted goal attainment and undesirable-state fractions, and offers a separate retrospective reveal of held-out outcomes. Display panel support, uncertainty diagnostics, and the no-treatment option. Do not label transcriptomic constraints as clinical safety.

The candidate-detail view should make the distributional question visible:

- Select a supported cell line, compound, and observed dose.
- Display untreated, predicted, and measured distributions in fixed PC coordinates and selected biological summaries.
- Compare the flow directly with the strongest baseline, including failures.
- Show changes in spread and covariance alongside changes in means.
- Label every case by evaluation setting: seen condition, held-out pair, replicate transfer, or held-out compound.
- Display recovery counts, control availability, projection fidelity, and model disagreement separately.
- Include a short plain-language statement of the estimand and supported claims.

Use deterministic precomputed examples initially. A broad free-form molecule input is unnecessary for version 0.1. If UMAP is included, use a training-fitted map and treat it as illustration; visual overlap is not an evaluation metric.

Select examples by a reproducible rule: median performance, one strong result, one poor result, and one large disagreement case. Include a no-response condition. The final report should answer what improved, which distributional properties improved, when the approach failed, and whether the improvements survived the technical-repeat audit.

## 13. Extensions, in order

1. **Better representation only if needed.** Compare 64/128-PC features and a training-only autoencoder. Require preserved treatment signal and fair evaluation in shared spaces. An architecture change does not excuse changing the test protocol.
2. **Control population encoder.** Replace control moments with a permutation-invariant set encoder if validation shows that multimodal baseline states matter. Keep a moments baseline and cross-fitting.
3. **Count-generating model.** Use a conditional latent-variable model with an explicit count observation distribution. For example, a conditional prior `p_theta(z | controls, structure, dose)` and a negative-binomial decoder with gene-wise dispersion and an explicitly modeled or supplied library-size variable. Train with a variational objective; do not call its ELBO an exact likelihood. Generated counts, conditional expression proportions, and latent biological state are different outputs. Conditioning on observed library size limits claims about treatment-induced total RNA changes. Validate counts and normalized responses separately.
4. **Unseen chemistry and cell lines.** Expand protocols C/D with adequate structural and biological coverage before claiming general drug-response prediction.
5. **Scale study.** Vary number of compounds, lines, and cells per condition separately. More cells within a well improve distribution estimation but do not provide new independent interventions.
6. **External validation.** Find a compatible independent chemical perturbation dataset and define the protocol mismatch before attempting transfer. This requires new research; no external dataset is assumed interchangeable here.

Then consider joint compound–dose search only after verifying cross-dose/cross-plate comparisons, followed by uncertainty-aware experimental prioritization with real repeat data. Defer drug combinations, patient predictions, unconstrained molecule generation, and mechanistic causal interpretation until the foundations are demonstrated.

## 14. Risks and decision rules

| Risk | Required response |
| --- | --- |
| Search exploits batch allocation or unequal candidate references | Restrict to compatible panels and one reference per query; reject unsupported prospective comparisons |
| Target signature rewards generic stress or selective cell loss | Inspect prespecified undesirable programs and recovery; distinguish captured-cell state from survival/reprogramming |
| Forward scores improve but chosen treatments do not | Report the inverse-search hypothesis as unsupported and retain the best decision baseline |
| Compound/dose effects confounded with plate | Preserve matched controls, audit incidence structure, report both within-assay completion and technical transfer; do not promise identifiable batch-free effects |
| Too few independent repeats | Use descriptive finite-sample/technical-repeat references; restrict claims about new experiments |
| PCA suppresses response heterogeneity | Measure preservation on validation; increase capacity or change representation before interpreting absent signal |
| Flow improves NLL but not distributional/biological diagnostics | Do not call it a useful improvement; report discrepancy and retain the simpler model |
| Most apparent performance is cell identity | Evaluate treated-minus-control shifts and compare untreated resampling |
| Chemically novel compounds fail | Deliver pair-completion result honestly; treat chemical transfer as unsolved |
| Low cell recovery filters away strong perturbations | Report attrition and supported population; do not impute missing distributions as no effect |
| Download or preprocessing dominates the budget | Reduce the cohort, create reusable shards, and profile before scaling |
| Learned model overfits nuisance information | Run control-reference and plate-transfer ablations; avoid post-treatment covariates |

## 15. Next-agent instructions and definition of done

Read this brief and sources S1–S4 plus S9–S10 first. Start M0 without requiring the user to choose routine software details. Resolve scientific ambiguities from metadata where possible. If a missing budget or hardware constraint would force a costly commitment, first complete the local audit and present the concrete resource estimate.

Do not claim the data have been validated simply because the landing page is accessible. Do not install a source-advertised plugin or change platform settings merely because a dataset page recommends it. Use ordinary public dataset interfaces unless a later task requires another route.

Before large training, produce a short decision record confirming the actual cohort, experiment-unit hierarchy, controls, replicate design, download footprint, representation fidelity, and evaluation protocol. Keep departures from these defaults in `DECISIONS.md` with reasons.

The prototype is complete when:

- The subset can be recreated from pinned sources and manifests.
- Evaluation splits and feature fitting pass leakage checks.
- All required baselines and the main model are evaluated fairly.
- Results include locked-query search performance against strong baselines, forward distributional diagnostics, and the technical-repeat audit.
- The design audit states which treatment comparisons are supported, and the demo refuses unsupported comparisons.
- Exact goals, candidate panels, reference IDs, and ranking artifacts are reproducible without access to held-out outcomes.
- The report distinguishes measured facts, design assumptions, and unsupported extrapolations.
- A public-facing explorer and reproducible technical write-up demonstrate both successes and failures.
- The owner can hand the repository to another engineer without relying on notebook state or this conversation.

Novel model superiority is a research outcome, not a prerequisite for an honest completed portfolio project.

## 16. Primary sources and implementation references

**S1 — Official Tahoe dataset and schema.** [tahoebio/Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M), including the [loading tutorial](https://huggingface.co/datasets/tahoebio/Tahoe-100M/blob/main/tutorials/loading_data.ipynb). Use for release access, sparse-row parsing, control/sample identifiers, dose metadata, and license. Pin a revision; do not copy example snippets without verifying configuration names.

**S2 — Experimental design and technical repeats.** Parse Biosciences, [Scaling Single Cell Discovery to 100 Million Cells](https://www.parsebiosciences.com/datasets/scaling-single-cell-discovery-to-100-million-cells/). Used for the assay overview, concentration/time support, and plate 6/14 replication description. Its page displays a different license label from the Hugging Face release; record the terms of the files actually acquired.

**S3 — Tahoe publication.** Zhang et al., *Tahoe-100M: Mapping drug-induced molecular phenotypes at single-cell resolution*, Cell (2026), [DOI: 10.1016/j.cell.2026.08.035](https://doi.org/10.1016/j.cell.2026.08.035); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42753697/). Earlier [May 2025 preprint](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v3). Confirm final methodological details and supplementary plate maps when implementing. Full publisher/preprint retrieval was not consistently available during this brief's preparation.

**S4 — Arc distribution and metadata.** [Arc Virtual Cell Atlas: Tahoe-100M](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/tahoe-100M/README.md). Use for AnnData alternatives, processing/QC fields, and current data-access migration instructions. Do not rely on deprecated bucket examples.

**S5 — Strong baseline motivation.** Ahlmann-Eltze, Huber, and Anders (2025), [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/), Nature Methods. This is evidence about the evaluated genetic-perturbation settings, not proof that chemical-response distribution models cannot improve.

**S6 — Related perturbation models.** [CPA paper](https://doi.org/10.15252/msb.202211517), [CPA implementation](https://github.com/theislab/cpa), and [chemCPA implementation](https://github.com/theislab/chemCPA). Use as related work and optional carefully matched benchmarks.

**S7 — Modern reference implementations.** [Arc State](https://github.com/ArcInstitute/state) and [cell-eval](https://github.com/ArcInstitute/cell-eval). Inspect metric definitions, split assumptions, checkpoint training provenance, and licenses before reuse. The bespoke metric contract above remains authoritative for this prototype.

**S8 — Distribution transport context.** [Learning single-cell perturbation responses using neural optimal transport](https://doi.org/10.1038/s41592-023-01969-x), Nature Methods (2023). Useful related work on unpaired populations; an inferred transport map should not be mistaken for an observed cell lineage or uniquely identified counterfactual.


**S9 — Batch effects in L1000.** Filzen et al. (2017), [Representing high throughput expression profiles via perturbation barcodes reveals compound targets](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005335), PLOS Computational Biology. Their L1000 study reports batch-related variability despite within-batch standardization. Related evidence, not a claim that every LINCS release has the same design or that this is the owner's remembered paper.

**S10 — Shared toxicity/proliferation signatures.** Szalai et al. (2019), [Signatures of cell death and proliferation in perturbation transcriptomics data—from confounding factor to effective prediction](https://pubmed.ncbi.nlm.nih.gov/31552418/), Nucleic Acids Research; [analysis code](https://github.com/bence-szalai/Cell-death-signatures). Links LINCS signatures with viability data and shows that shared toxicity can obscure mechanism-specific similarity. This motivates checking what a goal rewards; it does not justify treating every downstream stress response as a removable pretreatment confounder.

**S11 — Possible related reference; verification incomplete.** [Inductive Biases for Disentangled Representation Learning with Correlated Treatment–Nuisance Factors](https://openreview.net/pdf/00011154060c61d882ef3e632e6237ec7de7f0a3.pdf), anonymous OpenReview manuscript in indexed results. Its indexed abstract addresses correlated treatment/nuisance assignments using experimental-design-inspired VAE constraints. Full PDF access was blocked during preparation; authors, final version, and review status are unverified. Treat as a reading lead, not validated evidence that a particular correction solves this project's identification problems.

**S12 — L1000 assay definition.** Broad Connectivity Map, [What is L1000?](https://clue.io/connectopedia/what_is_l1000). The platform measures 978 landmark transcripts and infers additional expression features. Its bulk assay measurements do not directly supply cell-level distributions. Use only for appropriately scoped auxiliary tasks after a separate design audit.
