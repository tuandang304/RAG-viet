# Consolidated evaluation report


### Main baseline table (P1d, full test sets)

| Method | ViQuAD clean (n=7301) | ViQuAD noisy (n=3814) | DANGDOCAO clean (n=4315) | DANGDOCAO noisy (n=4315) |
|---|---|---|---|---|
| Router (4-way, ours) | 0.8541 | 0.6405 | 0.8196 | 0.6218 |
| Best fixed (dev-tuned) | 0.8211 *** | 0.6457 * | 0.7869 *** | 0.6287 *** |
| Fixed equal 4-way | 0.8369 *** | 0.5694 *** | 0.8051 *** | 0.4277 *** |
| RRF | 0.7988 *** | 0.5045 *** | 0.7816 *** | 0.3055 *** |
| Fixed equal 3-way | 0.8482 *** | 0.3961 *** | 0.8152 ** | 0.1470 *** |
| Toneless only | 0.5760 *** | 0.5905 *** | 0.6094 *** | 0.6095 *** |
| Dense only | 0.8064 *** | 0.2953 *** | 0.7907 *** | 0.0715 *** |
| BM25 only | 0.6621 *** | 0.1558 *** | 0.6748 *** | 0.0481 *** |
| Sparse only | 0.7594 *** | 0.3669 *** | 0.7526 *** | 0.1435 *** |

_NDCG@10; significance of router vs baseline: *** p<0.001, ** p<0.01, * p<0.05, ns not sig, — undefined._

### OOD LLM-noise generalization (P2.7, DANGDOCAO full)

| Method | missing_tone (n=4315) | typo_telex (n=4315) | informal (n=4315) | code_switch (n=4315) |
|---|---|---|---|---|
| Router (4-way, ours) | 0.6140 | 0.5351 | 0.7955 | 0.7827 |
| Best fixed (dev-tuned) | 0.6204 *** | 0.5430 *** | 0.7612 *** | 0.7366 *** |
| Fixed equal 4-way | 0.4389 *** | 0.4800 *** | 0.7773 *** | 0.7618 *** |
| RRF | 0.3284 *** | 0.4344 *** | 0.7477 *** | 0.7323 *** |
| Fixed equal 3-way | 0.1719 *** | 0.3343 *** | 0.7803 *** | 0.7821 ns |
| Toneless only | 0.5955 *** | 0.4622 *** | 0.5703 *** | 0.5143 *** |
| Dense only | 0.0885 *** | 0.2106 *** | 0.7725 *** | 0.7656 *** |
| BM25 only | 0.0573 *** | 0.1365 *** | 0.6128 *** | 0.5780 *** |
| Sparse only | 0.1607 *** | 0.3075 *** | 0.6999 *** | 0.6712 *** |

_NDCG@10; significance of router vs baseline: *** p<0.001, ** p<0.01, * p<0.05, ns not sig, — undefined._

### Partial-noise curve (P2.6) — NDCG@10 vs % syllables stripped

**viaquad**

| Noise % | Router (4-way, ours) | Best fixed (dev-tuned) | Fixed equal 3-way | Toneless only | Dense only |
|---|---|---|---|---|---|
| 0% | 0.8447 | 0.8232 | 0.8429 | 0.6000 | 0.8129 |
| 25% | 0.8052 | 0.7988 | 0.7958 | 0.6001 | 0.7138 |
| 50% | 0.7681 | 0.7698 | 0.7233 | 0.6000 | 0.6140 |
| 75% | 0.7119 | 0.7106 | 0.6054 | 0.5992 | 0.4903 |
| 100% | 0.6571 | 0.6673 | 0.4290 | 0.6000 | 0.3244 |

**dangdocao**

| Noise % | Router (4-way, ours) | Best fixed (dev-tuned) | Fixed equal 3-way | Toneless only | Dense only |
|---|---|---|---|---|---|
| 0% | 0.8284 | 0.7887 | 0.8221 | 0.6043 | 0.7931 |
| 25% | 0.7953 | 0.7721 | 0.7744 | 0.6041 | 0.7045 |
| 50% | 0.7438 | 0.7432 | 0.6838 | 0.6043 | 0.5440 |
| 75% | 0.6713 | 0.6735 | 0.4369 | 0.6045 | 0.2788 |
| 100% | 0.6150 | 0.6235 | 0.1304 | 0.6031 | 0.0557 |


### Diacritic restoration vs toneless channel (P1c)

| Domain | Router on noisy | Router on restored | Δ (restore−noisy) |
|---|---|---|---|
| ViQuAD | 0.6564 | 0.8257 | +0.1693 |
| DANGDOCAO | 0.6148 | 0.8198 | +0.2050 |

_Same 500 queries (seed 42) for both columns. Restoration is a strong but costly baseline: 1 LLM call/query vs a single BM25 lookup for the toneless channel._

### RAGAS end-to-end answer quality

**ragas_clean** (n=50)

| Method | context_precision | context_recall | faithfulness | answer_relevancy |
|---|---|---|---|---|
| dynamic_mlp | 0.8008 | 0.9800 | 0.8939 | 0.6913 |
| fixed_equal_4 | 0.7855 | 0.9400 | 0.8762 | 0.7684 |
| toneless_only | 0.5684 | 0.7800 | 0.7283 | 0.6329 |
| dense_only | 0.7819 | 0.9800 | 0.9205 | 0.7678 |

**ragas_noisy** (n=40)

| Method | context_precision | context_recall | faithfulness |
|---|---|---|---|
| dynamic_mlp | 0.5737 | 0.7750 | 0.7481 |
| fixed_equal_4 | 0.5766 | 0.7250 | 0.7552 |
| toneless_only | 0.5327 | 0.7250 | 0.7761 |

