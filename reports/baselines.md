# Baselines

Validation queries only. Utility is the soft membership gain of the objective minus
the apoptosis membership gain. Doing nothing has utility zero. Test-treated cells
were not scored and do not enter these rankings.

## Solvent curves

| goal | threshold | scale | n |
| --- | --- | --- | --- |
| g2m_down | 0.151 | 0.134 | 188,654 |
| p53_up | 0.31 | 0.0694 | 188,654 |
| apoptosis_constraint | 0.347 | 0.0807 | 188,654 |
| interferon_gamma_up | 0.352 | 0.111 | 188,654 |

## Search

| model | top1 | top5 | regret | nothing | spearman | queries |
| --- | --- | --- | --- | --- | --- | --- |
| shared_shift | 0.0265 | 0.01 | 0.0186 | 0.259 | 0.631 | 162 |
| ridge | 0.0187 | 0.00702 | 0.0264 | 0.228 | 0.477 | 162 |
| gaussian | 0.00468 | -0.00818 | 0.0405 | 0.525 | 0.116 | 162 |
| untreated | 0 | -0.0105 | 0.0451 | 1 |  | 162 |

## By goal

| model | goal_id | top1 | spearman |
| --- | --- | --- | --- |
| gaussian | g2m_down | 0.00474 | 0.1 |
| ridge | g2m_down | 0.026 | 0.631 |
| shared_shift | g2m_down | 0.0402 | 0.798 |
| untreated | g2m_down | 0 |  |
| gaussian | interferon_gamma_up | 0.00378 | 0.145 |
| ridge | interferon_gamma_up | 0.0129 | 0.424 |
| shared_shift | interferon_gamma_up | 0.0137 | 0.562 |
| untreated | interferon_gamma_up | 0 |  |
| gaussian | p53_up | 0.0055 | 0.104 |
| ridge | p53_up | 0.0172 | 0.377 |
| shared_shift | p53_up | 0.0256 | 0.533 |
| untreated | p53_up | 0 |  |

## Distributional distance to validation cells

| model | energy | mean_shift | marginal_w1 | sliced_w1 | covariance |
| --- | --- | --- | --- | --- | --- |
| shared_shift | 0.00316 | 0.0367 | 0.0208 | 0.0202 | 0.00777 |
| ridge | 0.00532 | 0.0402 | 0.0248 | 0.0228 | 0.00969 |
| untreated | 0.0107 | 0.0556 | 0.0279 | 0.026 | 0.00744 |
| gaussian | 0.0312 | 0.0897 | 0.0446 | 0.0432 | 0.0109 |

## Protocol C neighbor transfer

A held-out structure borrows the training shift of its nearest Morgan neighbor.
These blocks are not the protocol-A queries.

Held-out groups: 55. Training groups: 117.

| goal_id | top1 | spearman | queries |
| --- | --- | --- | --- |
| g2m_down | 0.00296 | 0.18 | 72 |
| interferon_gamma_up | -0.0109 | 0.0997 | 72 |
| p53_up | 0.0183 | 0.101 | 72 |
