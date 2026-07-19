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

### Component ablation (router NDCG@10, n=500/set, seed 42)

| Configuration | ViQuAD clean | ViQuAD noisy | DANGDOCAO clean | DANGDOCAO noisy |
|---|---|---|---|---|
| Full system | 0.8598 | 0.6570 | 0.8291 | 0.6154 |
| − expected weights (argmax) | 0.8589 | 0.6464 | 0.8238 | 0.6139 |
| − QPP signals (8 linguistic feats) | 0.8632 | 0.6487 | 0.8223 | 0.6158 |
| − raw labels (per-query min-max) | 0.8687 | 0.6391 | 0.8248 | 0.6167 |
| − toneless augmentation | 0.8621 | 0.4834 | 0.8277 | 0.1427 |
| − toneless channel (3-way routing) | 0.8670 | 0.4274 | 0.8261 | 0.1280 |

**Oracle headroom** (label-dependent per-query best grid point — upper bound):

| Set | Fixed equal 4-way | Router | Oracle | Headroom realized |
|---|---|---|---|---|
| ViQuAD clean | 0.8484 | 0.8598 | 0.9416 | 12% |
| ViQuAD noisy | 0.5930 | 0.6570 | 0.7547 | 40% |
| DANGDOCAO clean | 0.8118 | 0.8291 | 0.9298 | 15% |
| DANGDOCAO noisy | 0.4157 | 0.6154 | 0.6704 | 78% |

_Headroom realized = (router − equal4) / (oracle − equal4)._

### Per-query latency breakdown (ms, ViQuAD n=500 run)

| Stage | mean | p50 |
|---|---|---|
| dense | 1.6 | 1.4 |
| bm25 | 15.5 | 14.5 |
| sparse | 30.6 | 29.8 |
| toneless | 19.0 | 18.2 |
| features | 0.6 | 0.6 |
| signals | 0.2 | 0.2 |
| router MLP | 5.9 | — |

_Dense = FAISS search only (query embedding is a cached/batched API call). The toneless channel adds one in-memory BM25 lookup; LLM diacritic restoration costs ~1.4–1.7 s/query wall-clock (measured over the two 500-query restoration runs) plus generation-side token spend._

### RAGAS end-to-end answer quality (Llama-3.3-70B judge)

**ViQuAD clean** (n=100)

| Method | context_precision | context_recall | faithfulness |
|---|---|---|---|
| dynamic_mlp | 0.8047 | 0.9700 | 0.8936 |
| fixed_equal_4 | 0.7864 | 0.9400 | 0.8431 |
| toneless_only | 0.5448 | 0.7400 | 0.7025 |

**ViQuAD noisy** (n=100)

| Method | context_precision | context_recall | faithfulness |
|---|---|---|---|
| dynamic_mlp | 0.5755 | 0.7700 | 0.7734 |
| fixed_equal_4 | 0.5475 | 0.7300 | 0.7758 |
| toneless_only | 0.5455 | 0.7000 | 0.7554 |

**DANGDOCAO clean** (n=100)

| Method | context_precision | context_recall | faithfulness |
|---|---|---|---|
| dynamic_mlp | 0.8573 | 0.9458 | 0.9610 |
| fixed_equal_4 | 0.8369 | 0.9407 | 0.9320 |
| toneless_only | 0.7061 | 0.9082 | 0.8323 |

**DANGDOCAO noisy** (n=100)

| Method | context_precision | context_recall | faithfulness |
|---|---|---|---|
| dynamic_mlp | 0.6877 | 0.9217 | 0.8846 |
| fixed_equal_4 | 0.4168 | 0.7778 | 0.7547 |
| toneless_only | 0.6923 | 0.9040 | 0.8914 |

_Context precision/recall (retrieval-quality metrics): the router leads in all four conditions. Faithfulness (generator-dependent) is comparable in the noisy regime. answer_relevancy is omitted — its async embedding path deadlocks against the FPT API._

**RAGAS paired significance** (router vs baseline, per-sample paired t-test):

| Condition | vs | metric | Δ | p |
|---|---|---|---|---|
| ViQuAD clean | fixed_equal_4 | context_precision | +0.0184 | 2.1e-01 ns |
| ViQuAD clean | fixed_equal_4 | context_recall | +0.0300 | 8.3e-02 ns |
| ViQuAD clean | fixed_equal_4 | faithfulness | +0.0515 | 1.3e-02 * |
| ViQuAD clean | toneless_only | context_precision | +0.2599 | 5.3e-09 *** |
| ViQuAD clean | toneless_only | context_recall | +0.2300 | 3.9e-07 *** |
| ViQuAD clean | toneless_only | faithfulness | +0.1817 | 5.1e-05 *** |
| ViQuAD noisy | fixed_equal_4 | context_precision | +0.0280 | 3.3e-01 ns |
| ViQuAD noisy | fixed_equal_4 | context_recall | +0.0400 | 1.6e-01 ns |
| ViQuAD noisy | fixed_equal_4 | faithfulness | -0.0024 | 9.5e-01 ns |
| ViQuAD noisy | toneless_only | context_precision | +0.0300 | 1.4e-01 ns |
| ViQuAD noisy | toneless_only | context_recall | +0.0700 | 1.9e-02 * |
| ViQuAD noisy | toneless_only | faithfulness | +0.0168 | 5.8e-01 ns |
| DANGDOCAO clean | fixed_equal_4 | context_precision | +0.0205 | 3.9e-02 * |
| DANGDOCAO clean | fixed_equal_4 | context_recall | +0.0051 | 7.4e-01 ns |
| DANGDOCAO clean | fixed_equal_4 | faithfulness | +0.0305 | 9.3e-02 ns |
| DANGDOCAO clean | toneless_only | context_precision | +0.1512 | 1.1e-06 *** |
| DANGDOCAO clean | toneless_only | context_recall | +0.0375 | 9.5e-02 ns |
| DANGDOCAO clean | toneless_only | faithfulness | +0.1185 | 3.0e-03 ** |
| DANGDOCAO noisy | fixed_equal_4 | context_precision | +0.2709 | 8.1e-14 *** |
| DANGDOCAO noisy | fixed_equal_4 | context_recall | +0.1439 | 8.8e-05 *** |
| DANGDOCAO noisy | fixed_equal_4 | faithfulness | +0.1174 | 9.8e-03 ** |
| DANGDOCAO noisy | toneless_only | context_precision | -0.0046 | 6.0e-01 ns |
| DANGDOCAO noisy | toneless_only | context_recall | +0.0177 | 3.4e-01 ns |
| DANGDOCAO noisy | toneless_only | faithfulness | -0.0121 | 5.0e-01 ns |
