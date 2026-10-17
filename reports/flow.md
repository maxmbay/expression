# Flow

A 4-layer rational-quadratic coupling flow models the residual of the four module
scores around the reference mean. It is conditioned on the reference mean and spread,
a 32-dimensional projection of the Morgan fingerprint, and the dose.

Holdout NLL 4.758. Mean top-1 observed utility 0.0023.
The shared shift's validation top-1 is 0.0265. Selected model: `shared_shift`.

| goal_id | top1 | spearman | nothing |
| --- | --- | --- | --- |
| g2m_down | -0.00116 | 0.1 | 0.63 |
| interferon_gamma_up | -0.00524 | 0.119 | 0.407 |
| p53_up | 0.0134 | 0.0746 | 0 |

Inverse consistency on one condition was finite, with absolute error below 1e-3.
