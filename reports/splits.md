# Splits

Protocol A holds out `(cell line, structure group)` pairs. A structure group is the
largest-fragment InChIKey, so a salt and its free base stay together.
The hash is Blake2s of `seed`, `A`, the line, and the structure group, joined by NUL
bytes. Values below 0.70 are train, below 0.85 validation, and the rest test. If a line
or a structure group would otherwise have no training pair, its lowest-hash held-out pair
moves to train. Every dose, well, and plate of a pair, plate 14 included, shares it.

Protocol C hashes the structure group alone (`C` instead of `A`) and does not repair:
a held-out group is absent from every line. Adagrasib and the two compounds with no
fingerprint (`Sacubitril/Valsartan`, `Verteporfin`) are excluded from protocol C.
Scaffold holdout is not a second protocol. The most common Bemis-Murcko scaffold in this
cohort is benzene, shared by 17 compounds, and 7 compounds are acyclic, so a scaffold
split would not be a chemical-novelty test.

Each plate has two vehicle wells. The well with the smaller Blake2s hash is the
reference population; the other is the evaluation population and is disjoint. Plate 14
is outside both fit masks, including its reference well. Fit masks are training treated
cells on the other plates, plus reference vehicle cells on those plates.

Split file sha256: `f2f2fe6acf48ff6d678ac6d32f8d3edff86c04b7f9639358b65fc3b00127593f`. Seed `0`.

## Protocol A pairs

| protocol_a | n | fraction |
| --- | --- | --- |
| test | 320 | 0.154 |
| train | 1474 | 0.71 |
| validation | 282 | 0.136 |

## Protocol A by line

| cell_line | train | validation | test |
| --- | --- | --- | --- |
| CVCL_0023 | 130 | 20 | 23 |
| CVCL_0320 | 137 | 17 | 19 |
| CVCL_0359 | 115 | 31 | 27 |
| CVCL_0399 | 117 | 18 | 38 |
| CVCL_0459 | 126 | 20 | 27 |
| CVCL_0480 | 122 | 23 | 28 |
| CVCL_0546 | 121 | 31 | 21 |
| CVCL_1056 | 112 | 28 | 33 |
| CVCL_1119 | 119 | 27 | 27 |
| CVCL_1285 | 125 | 22 | 26 |
| CVCL_1381 | 124 | 22 | 27 |
| CVCL_1550 | 126 | 23 | 24 |

## Protocol C groups

| protocol_c | n | fraction |
| --- | --- | --- |
| test | 22 | 0.129 |
| train | 117 | 0.684 |
| validation | 32 | 0.187 |

## Plate 14 treated cells

| protocol_b | n |
| --- | --- |
| anchor | 3072 |
| control | 37,601 |
| seen_condition_new_plate | 204,218 |
| unseen_pair_new_plate | 78,041 |

## Vehicle wells

| plate_num | sample | control_role |
| --- | --- | --- |
| 4 | smp_1878 | evaluation |
| 4 | smp_1877 | reference |
| 5 | smp_1973 | evaluation |
| 5 | smp_1974 | reference |
| 6 | smp_2070 | evaluation |
| 6 | smp_2069 | reference |
| 7 | smp_2165 | evaluation |
| 7 | smp_2166 | reference |
| 8 | smp_2262 | evaluation |
| 8 | smp_2261 | reference |
| 9 | smp_2358 | evaluation |
| 9 | smp_2357 | reference |
| 14 | smp_2838 | evaluation |
| 14 | smp_2837 | reference |

## Queries

- `dev`: 216 queries over 54 contexts, 18 blocks below the 10-candidate minimum.
- `test`: 276 queries over 69 contexts, 3 blocks below the 10-candidate minimum.
- `plate14`: 124 queries over 12 contexts, 5 blocks below the 10-candidate minimum.

Query files name reference and evaluation cell ids, the panel, and the scoring rule.
They do not contain expression, scores, or thresholds. Thresholds are fit from training
reference-vehicle cells when the representation is fit (`goals` version `hallmark-2020`).

Compounds in the cohort: 186. Structure groups: 174.
