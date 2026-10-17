# Experiment log

Append-only. New entries go at the bottom.

## 2026-10-17 — M1b frozen, utility softened

Pushed the cohort fetch fix and the benchmark contract (`4a74ab9`). Protocol A is 1474/282/320 train/validation/test pairs. The representation gate failed on rank correlation (0.64–0.83) and on module coverage in the 2,000 HVGs (17–29%), so inverse search uses the four module scores. PCA-64 stays for distributional comparisons.

Utility is no longer a hard percentile cut. A cell's membership is a logistic centered at the training-solvent quantile, with width equal to the solvent standard deviation. Objective utility subtracts the apoptosis membership gain with weight 1 (`DECISIONS.md` D11). Query manifests were rewritten with that rule; panels and reference cells are unchanged. Test-treated expression is still unread.

Next: score training, control, and validation cells, then fit untreated, shared shift, ridge-plus-residuals, and a conditional Gaussian on the validation queries.
