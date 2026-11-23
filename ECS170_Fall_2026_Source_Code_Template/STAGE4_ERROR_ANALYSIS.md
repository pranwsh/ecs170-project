# Classification error analysis (one-epoch validation pilots)

These are descriptive observations on the common 2,500-review validation partition, not supplied-test tuning. Positive = 1. Review lengths count all preprocessing tokens before the 256-token cutoff. Cells show errors / reviews (error rate). The groups are observational and do not establish that truncation causes errors.

| cell | <=128 tokens | 129–256 | >256 | false positives | false negatives |
|---|---:|---:|---:|---:|---:|
| RNN | 164/390 (42.1%) | 547/1179 (46.4%) | 405/931 (43.5%) | 482 | 634 |
| LSTM | 106/390 (27.2%) | 434/1179 (36.8%) | 362/931 (38.9%) | 473 | 429 |
| GRU | 112/390 (28.7%) | 375/1179 (31.8%) | 333/931 (35.8%) | 267 | 553 |

The GRU misclassifies more positive reviews as negative than the reverse (553 vs 267); it also has a higher observed error rate on >256-token reviews than on shorter reviews. Reviews over the cutoff lose their tail, but length is associated with other properties and no causal conclusion follows. Rates use 390, 1,179 and 931 examples respectively. Counts are from saved validation predictions and original review texts; this is not a qualitative manual review or confidence/calibration analysis. Pilot findings may not generalize to multi-epoch fits.
