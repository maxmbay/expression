# Benchmark

Rankings were written to `reports/test_rankings.json` and hashed (`c5edff06c46241acd280691fed8d4f68ac832eb9ef1906f96a2382df7358ee72`)
before any test-treated module score was computed. The selected model is the
shared shift. It has no random seed. The flow was not a finalist.

## Test queries

| model | top1 | regret | nothing | spearman | queries |
| --- | --- | --- | --- | --- | --- |
| shared_shift | 0.0224 | 0.0183 | 0.275 | 0.644 | 207 |
| ridge | 0.0216 | 0.0191 | 0.222 | 0.491 | 207 |
| untreated | 0 | 0.0407 | 1 |  | 207 |

Shared shift minus the next best model: 0.0008.
The +5 percentage-point gate does not pass.

## Plate 14

| split | model | top1 | queries |
| --- | --- | --- | --- |
| seen_condition_new_plate | ridge | 0.029 | 36 |
| seen_condition_new_plate | shared_shift | 0.034 | 36 |
| seen_condition_new_plate | untreated | 0 | 36 |
| test | ridge | 0.0119 | 36 |
| test | shared_shift | 0.0126 | 36 |
| test | untreated | 0 | 36 |
| validation | ridge | 0.0214 | 21 |
| validation | shared_shift | 0.018 | 21 |
| validation | untreated | 0 | 21 |
