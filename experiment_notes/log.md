# Experiment log

Append-only. New entries go at the bottom.

## 2026-10-17 — M1b frozen, utility softened

Pushed the cohort fetch fix and the benchmark contract (`4a74ab9`). Protocol A is 1474/282/320 train/validation/test pairs. The representation gate failed on rank correlation (0.64–0.83) and on module coverage in the 2,000 HVGs (17–29%), so inverse search uses the four module scores. PCA-64 stays for distributional comparisons.

Utility is no longer a hard percentile cut. A cell's membership is a logistic centered at the training-solvent quantile, with width equal to the solvent standard deviation. Objective utility subtracts the apoptosis membership gain with weight 1 (`DECISIONS.md` D11). Query manifests were rewritten with that rule; panels and reference cells are unchanged. Test-treated expression is still unread.

Next: score training, control, and validation cells, then fit untreated, shared shift, ridge-plus-residuals, and a conditional Gaussian on the validation queries.

## 2026-10-17 — M2 baselines

Module scores for 2,128,309 cells are cached (protocol-A test-treated cells excluded). Soft curves were fit on 188,654 training reference cells.

On 162 validation queries, mean top-1 observed utility is shared shift 0.0265, ridge 0.0187, gaussian 0.0047, untreated 0. Shared shift also has the best energy score. It picks nothing on 26% of queries. Spearman is 0.63 overall and 0.80 for G2/M. Protocol C neighbor transfer is near zero except a small p53 gain (0.018). Ties at utility zero now select doing nothing, so the untreated model is a real null.

The validation-selected model for a later test ranking is the shared shift. A flow still has to beat it on validation before it can be the selected system.

## 2026-10-17 — M3 flow

A 4-layer rational-quadratic coupling flow on the four residual module scores, trained for 6 epochs on 60,000 training cells. Inverse error 1.2e-7. Holdout NLL fell from 5.03 to 4.76. Mean validation top-1 utility was 0.002, against 0.026 for the shared shift, and it ranked near chance (Spearman about 0.1). The selected model stays the shared shift. Three flow seeds were not run: the first seed is not a finalist.
