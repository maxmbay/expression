# Representation

The gate does not pass. Inverse search will model the four module scores directly. PCA-64 stays available for baselines that do not need the goal event. The test set is unchanged.

Fitting uses protocol-A training treated cells and reference vehicle cells on plates
other than 14. Validation treated cells and plate 14 are not in the fit.
Protocol C has its own HVG and PCA fit, with the same plate-14 exclusion;
the gate below is the protocol-A validation gate, which is the primary benchmark.

Seurat-v3-style residual: on a balanced subsample of at most 32 cells per training condition, compute the mean and variance of raw counts, fit a degree-2 polynomial of log10(variance) on log10(mean) for genes with mean at least 0.001, and keep the 2,000 genes with the largest residual. Ties break toward the lower column.

Counts are normalized to 10,000 per cell and transformed with log1p before PCA.
PCA is centered, with 64 components, fit on at most 200,000 balanced cells.
PC scales are training standard deviations floored at 0.0001.

## Fits

| fit | cells | conditions | hvg_cells | pca_cells | variance_explained |
| --- | --- | --- | --- | --- | --- |
| fit_a | 1,391,159 | 4770 | 152,640 | 195,570 | 0.37 |
| fit_c | 1,340,654 | 4572 | 146,304 | 196,596 | 0.372 |

## Gate

A goal passes when at least half of its vocabulary genes are among the 2,000 HVGs,
the median absolute difference of raw and reconstructed event fractions is at most 0.05,
and the median Spearman correlation of those fractions across compounds, within a
(line, plate) block of at least 10 compounds, is at least 0.8. Blocks with no variation
are excluded from the correlation. The event threshold is the training reference-vehicle
quantile of the raw score on the same gene subset.

| goal | role | genes_in_hvg | threshold | median_abs_delta | p90_abs_delta | spearman | ranked_blocks | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g2m_down | objective | 33/200 | 0.142 | 0.0234 | 0.0508 | 0.645 | 54 | False |
| p53_up | objective | 41/199 | 0.582 | 0.0352 | 0.0859 | 0.681 | 54 | False |
| apoptosis_constraint | constraint | 37/161 | 0.555 | 0.0391 | 0.0859 | 0.703 | 54 | False |
| interferon_gamma_up | objective | 58/199 | 0.606 | 0.0078 | 0.0312 | 0.833 | 52 | False |

Decision recorded for later milestones: `direct_module_scores`.

Genes from the Hallmark list that are not in the Tahoe vocabulary:

- p53_up is missing H2AW from the vocabulary.
- interferon_gamma_up is missing DDX58 from the vocabulary.
