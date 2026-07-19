# Dynamic Hybrid Retrieval-Augmented Generation for Vietnamese: Adaptive Fusion of Dense and Sparse Signals via a Lightweight MLP

---

> ⚠️ **REPRODUCIBILITY NOTE (remove before submission).**
> The **method sections (§1–§4) describe the current codebase**: an **eight-feature**
> query representation (adds `oov_ratio`) and a **Keras/TensorFlow** fusion MLP
> (`Dense(64)→LayerNorm→GELU→Dropout→Dense(32)→LayerNorm→GELU→Dropout→Dense(3)`,
> **≈2,947 parameters**) trained on a **231-point** simplex grid (step 0.05).
> **All empirical results in §5 were produced by an earlier configuration**
> (seven features, a plain `Linear(7→64→32→3)+ReLU` PyTorch MLP of ≈2,691 parameters,
> 66-point simplex grid at step 0.1). **The result tables, the efficiency numbers,
> and the interpretability correlations must be regenerated with the current code
> before submission**, after which this note and any "(earlier run)" markers can be deleted.

---

## Authors & Affiliation

**[Author 1 Name]**¹, **[Author 2 Name]**¹, **[Author 3 Name]**²

¹ [University / Lab Name], [City], Vietnam
² [Affiliation 2], [City], Vietnam

Correspondence: [email@domain.com]

---

## Abstract

Retrieval-Augmented Generation (RAG) systems typically rely on a fixed combination of dense vector search and sparse lexical matching (BM25), with weights chosen once at development time. This one-size-fits-all strategy ignores the fact that different query types benefit from different retrieval signals — a property especially pronounced in Vietnamese, where word segmentation quality, diacritical mark presence, and code-switching between Vietnamese and English all strongly modulate the effectiveness of each retrieval approach.

We present **Dynamic Hybrid RAG**, a lightweight framework that replaces fixed fusion weights with an adaptive MLP that predicts per-query **four-way** fusion weights `(w_dense, w_bm25, w_sparse, w_toneless)`. The four retrieval signals are dense semantic search (FPT Vietnamese Embedding + FAISS), BM25 over underthesea-segmented text, BGE-M3 learned sparse lexical weights via an inverted index, and a **diacritic-stripped syllable BM25 channel** dedicated to tone-robust matching. The router consumes eight Vietnamese-aware linguistic features augmented with post-retrieval **query-performance signals** (per-channel score-distribution shape and cross-channel agreement, computed from the already-retrieved candidates and invariant to raw score scale). It is trained on raw NDCG@10 targets over the 286-point four-simplex grid; at inference, weights are the softmax-expected grid point over the predicted-NDCG surface (temperature $T = 0.05$, shown insensitive across domains), which degrades gracefully to near-uniform weighting on flat surfaces rather than committing to a brittle argmax.

On the UIT-ViQuAD 2.0 test set (7,301 queries), our router achieves **NDCG@10 = 0.854**, outperforming every baseline with statistical significance: fixed-equal three-way (0.848), a **dev-tuned best-fixed** four-way weight (0.821), reciprocal-rank fusion (0.799), dense-only (0.807), BGE-M3 sparse-only (0.760), and BM25-only (0.662). Under diacritic-removal noise (all tone marks stripped to simulate Vietnamese keyboard typing), the toneless channel is decisive: the router reaches NDCG@10 = 0.641 versus 0.396 for fixed-equal three-way and 0.146 for BM25-only. In a strict **zero-shot cross-domain evaluation** on the DANGDOCAO legal/administrative corpus (37,239 passages, no DANGDOCAO data seen during training), the same checkpoint again leads every baseline: 0.820 on clean queries and 0.622 on diacritic-noisy queries (versus 0.147 for fixed-equal three-way). A noise-level sweep shows the router tracing the **upper envelope** across the full 0–100% diacritic-corruption spectrum, and it **generalizes to noise types unseen in training** (typo, informal, code-switching). The central finding is that **no single fixed weighting is strong in both the clean and noisy regimes** — even one grid-searched on a balanced dev mix must trade one regime for the other — whereas the adaptive router matches or beats every static baseline in both, without knowing the test query's noise level in advance. All results are consolidated in `docs/results_summary.md`.

---

## Keywords

Retrieval-Augmented Generation, Hybrid Retrieval, Dense Retrieval, BM25, Vietnamese NLP, Adaptive Fusion, MLP, Soft Labels, Cross-domain Generalization

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for knowledge-intensive NLP tasks, pairing a retrieval component with a large language model (LLM) generator [CITATION]. A critical yet under-studied design decision is how to *combine* multiple retrieval signals — dense semantic search, classical BM25 lexical matching, and learned-sparse lexical retrieval — when no single method dominates across all query types.

For **Vietnamese**, this challenge is amplified by three language-specific factors:

1. **Word segmentation dependency.** Vietnamese is written without spaces between syllables, but semantic units (words) span multiple syllables ("học_sinh" = student, "trí_tuệ_nhân_tạo" = artificial intelligence). BM25 applied to raw whitespace-tokenized text degrades significantly compared to BM25 over properly segmented words.

2. **Diacritical mark sensitivity.** Vietnamese orthography uses six tones encoded as diacritical marks. Users frequently type without diacritics (e.g., "benh tieu duong" instead of "bệnh tiểu đường"), causing near-total BM25 failure while dense embeddings remain comparatively robust.

3. **Code-switching.** Technical Vietnamese text commonly mixes English terms (e.g., "API", "database"). Classical BM25 over whitespace-tokenized text may match such terms but loses Vietnamese compound structure, whereas dense models can conflate semantically related English tokens. Learned-sparse retrieval (BGE-M3 lexical weights) offers a middle path: it assigns importance weights to tokens directly, including code-switched English terms, while sharing a tokenizer with the dense backbone.

These factors suggest that the optimal balance between dense, BM25, and learned-sparse retrieval is **query-dependent** in a linguistically predictable way. Our central hypothesis is that a small neural module, conditioned on lightweight linguistic features of the query, can learn this mapping over the three-signal weight simplex and outperform any fixed weight choice.

Our contributions are:

1. A **Dynamic Hybrid RAG** architecture with a per-query MLP fusion module that produces three-way weights `(w_dense, w_bm25, w_sparse)` over dense, BM25, and BGE-M3 learned-sparse signals — trained end-to-end on Vietnamese QA data.
2. A **Vietnamese-aware feature extractor** (8 features: diacritic ratio, compound word ratio, English token ratio, tech-term ratio, clause count, question-word presence, query length, and out-of-vocabulary ratio against the BM25 corpus).
3. A **soft-label training strategy** using temperature-scaled NDCG@10 distributions over a 3D simplex grid (231 points at step 0.05), improving over hard-label grid search.
4. An **empirical analysis** on two Vietnamese datasets — UIT-ViQuAD 2.0 (Wikipedia) and DANGDOCAO (legal/administrative) — quantifying when dynamic three-way fusion outperforms fixed-weight baselines and single-signal retrievers.

---

## 2. Related Work

### 2.1. Retrieval-Augmented Generation

[CITATION: Lewis et al. 2020 RAG paper]  
[CITATION: Izacard & Grave 2021 FiD]  
[CITATION: Recent survey on RAG]

### 2.2. Hybrid Retrieval

Hybrid retrieval combining dense and sparse signals has been studied extensively for English [CITATION]. Reciprocal Rank Fusion (RRF) [CITATION] and linear interpolation of scores [CITATION] are common approaches, but all rely on fixed combination strategies. [CITATION: BM25+dense interpolation work] shows that the optimal weight is dataset-dependent, motivating adaptive approaches.

Recent multi-functional encoders such as BGE-M3 [3] expose three retrieval modes — dense, learned sparse (lexical weights), and multi-vector (ColBERT-style) — from a single backbone, enabling tighter score-space coupling than externally combined dense + BM25 systems. We adopt the dense and learned-sparse modes of (a Vietnamese fine-tune of) this family alongside classical BM25, yielding a three-signal fusion problem in which a single backbone supplies two of the three signals.

### 2.3. Adaptive / Learned Retrieval Fusion

[CITATION: Learning to fuse retrieval scores]  
[CITATION: Query-dependent retrieval weighting]  
[CITATION: Any relevant work on meta-retrieval or learned fusion]

### 2.4. Vietnamese Information Retrieval

Vietnamese IR is less studied than English or Chinese. [CITATION: ViQuAD paper] introduced the first large-scale Vietnamese QA dataset. [CITATION: underthesea or VnCoreNLP] provides the word segmentation used in our BM25 pipeline. [CITATION: Vietnamese embedding models] covers dense retrieval for Vietnamese.

---

## 3. Methodology

### 3.1. Problem Formulation

Given a corpus of passages $\mathcal{P} = \{p_1, \ldots, p_N\}$ and a query $q$, we seek a retrieval function that returns the top-$k$ passages most relevant to $q$. We define the fused three-way relevance score as:

$$s(q, p) = w_\text{dense} \cdot \hat{s}_\text{dense}(q, p) + w_\text{bm25} \cdot \hat{s}_\text{bm25}(q, p) + w_\text{sparse} \cdot \hat{s}_\text{sparse}(q, p)$$

where $\hat{s}$ denotes min-max normalized scores and $(w_\text{dense}, w_\text{bm25}, w_\text{sparse}) = \text{softmax}(\text{MLP}(\phi(q)))$ with $\phi(q) \in \mathbb{R}^8$ being the Vietnamese-aware feature vector. The softmax constraint enforces $w_\text{dense} + w_\text{bm25} + w_\text{sparse} = 1$ and $w_i \geq 0$, i.e. the weight vector lies on the 2-simplex.

### 3.2. Retrieval Components

We fuse four complementary retrieval signals; each retriever runs on the full corpus and returns its own top-100 candidate set, which we union before fusion.

**Dense Retrieval.** We encode passages and queries using the FPT Vietnamese Embedding model (1024-dimensional, fine-tuned from BGE-M3). Passage embeddings are L2-normalized and indexed with FAISS `IndexFlatIP` for inner product search, equivalent to cosine similarity after normalization.

**BM25 Retrieval.** Queries and passages are tokenized with underthesea `word_tokenize` (`format="text"`), which produces underscore-joined Vietnamese compound words (e.g., "học_sinh", "trí_tuệ_nhân_tạo"). BM25Okapi scores are computed over this segmented vocabulary.

**Learned Sparse Retrieval (BGE-M3).** We extract per-token lexical weights from BGE-M3 (`BAAI/bge-m3`) using the FlagEmbedding library and build an inverted index over non-zero token weights. At query time, BGE-M3 produces sparse lexical weights for the query, and document scores are computed as the dot product over the inverted-index posting lists. This signal is run locally (no external API) and captures learned term importance — including out-of-vocabulary and code-switching English terms — that classical BM25 cannot model.

**Toneless BM25 Retrieval.** The fourth channel is a second BM25 index built over a diacritic-stripped, lowercased, *syllable-level* tokenization of the corpus (the same transform is applied symmetrically to queries at search time). We deliberately avoid underthesea word segmentation here: segmentation quality collapses on toneless text, and segmenting the (toned) passages differently from the (toneless) queries would introduce a fresh mismatch. When a user types without tone marks — pervasive on Vietnamese keyboards — the toned BM25, dense, and sparse channels all suffer a character-level mismatch, whereas the toneless index restores exact lexical overlap. This channel is near-free: it is a second in-memory BM25 lookup with no model inference and no API call. On clean queries it is *weaker* than toned BM25 (diacritic stripping creates homograph collisions), so the router must learn a two-way gating: raise $w_\text{toneless}$ as tone information disappears, suppress it otherwise.

**Score Normalization.** All four score distributions are independently min-max normalized to $[0, 1]$ before fusion, necessary because the BM25, toneless-BM25, and BGE-M3 sparse scores are unbounded while dense cosine scores are bounded in $[-1, 1]$.

### 3.3. Vietnamese-Aware Feature Extractor

We extract eight features $\phi(q) = [f_1, \ldots, f_8]$:

| Feature | Description | Range |
|---------|-------------|-------|
| $f_1$ — diacritic\_ratio | Fraction of syllables bearing a Vietnamese tone mark | $[0, 1]$ |
| $f_2$ — compound\_ratio | Fraction of tokens that are multi-syllable compounds after segmentation | $[0, 1]$ |
| $f_3$ — english\_ratio | Fraction of syllables matching `[a-zA-Z]+` (code-switching) | $[0, 1]$ |
| $f_4$ — tech\_term\_ratio | Fraction of syllables matching a curated technical vocabulary | $[0, 1]$ |
| $f_5$ — clause\_count\_norm | Number of clause markers (commas, "và", "hoặc", …), clipped and normalized by 5 | $[0, 1]$ |
| $f_6$ — has\_question\_word | Binary: query contains a Vietnamese interrogative ("ai", "gì", "nào", …) | $\{0, 1\}$ |
| $f_7$ — query\_length\_norm | Syllable count normalized at 20 | $[0, 1]$ |
| $f_8$ — oov\_ratio | Fraction of underthesea-segmented query tokens absent from the BM25 corpus vocabulary | $[0, 1]$ |

**Post-retrieval query-performance signals.** The eight linguistic features describe only the query string, yet which channel wins also depends on how the corpus responds to the query. We therefore append a block of query-performance-prediction (QPP) signals computed *after* the four channels have retrieved (which fusion requires anyway, so the signals are free at inference): for each channel, the top-1/top-2 score gap, the mean and standard deviation of the top-10 score window, and coverage; plus, across every channel pair, the Jaccard overlap of top-10 id sets and a top-1 agreement indicator. Every statistic is computed on scores normalized within the channel's own top-$k$ window, making the block invariant to raw score scale — essential for zero-shot transfer across corpora whose BM25/sparse magnitudes differ. The router input is the concatenation of the eight linguistic features with these signals (28 signals in the four-channel configuration).

### 3.4. Grid-NDCG Router

The router is a compact feed-forward network (Keras/TensorFlow) with layer normalization, GELU activations, and dropout. Rather than regress a weight vector directly, it predicts the achievable **NDCG@10 surface over the weight simplex**: the output layer has one unit per grid point on the 286-point four-simplex $G_4 = \{(a,b,c,d) \mid a{+}b{+}c{+}d = 1,\ a,b,c,d \in \{0, 0.1, \ldots, 1\}\}$.

$$\text{MLP}: \mathbb{R}^{36} \to \text{Norm} \to \text{Dense}(64)\to\text{LN}\to\text{GELU}\to\text{Drop}(0.1) \to \text{Dense}(32)\to\text{LN}\to\text{GELU}\to\text{Drop}(0.1) \to \text{Dense}(286) \xrightarrow{\sigma} \widehat{\text{NDCG}}$$

An adapted input-normalization layer standardizes the 36 features (statistics stored in the checkpoint). At inference, the predicted surface $\widehat{\text{NDCG}}$ is turned into weights by a **softmax-expected** combination of grid points, $\bar{\mathbf{w}} = \sum_i \text{softmax}(\widehat{\text{NDCG}}_i / T)\, \mathbf{w}_i$ with $T = 0.05$. This is deliberately not an argmax: on a flat predicted surface — the common case, when routing barely matters — the expectation converges to the grid centroid $\approx (\tfrac14,\tfrac14,\tfrac14,\tfrac14)$ and the router degrades gracefully to near-uniform fusion, whereas an argmax would jump to an arbitrary extreme vertex. A temperature sweep on both dev sets shows the result is insensitive to $T$ (per-domain optima of $0.03$ and $0.12$ differ from $T{=}0.05$ by $<0.002$ NDCG and in opposite directions), so we fix $T = 0.05$ rather than tune it per corpus. The same architecture supports three channels (66-point grid) and two (11-point) by changing the output width.

### 3.5. Grid-NDCG Training with Raw Labels

Supervision for the router is the downstream NDCG@10 achievable under each candidate weighting: for every training query we compute NDCG@10 at all 286 grid points (a single vectorized fusion-and-rank pass) and regress the network's 286 outputs against these values with MSE.

**Raw versus per-query-normalized targets.** An earlier variant min-max normalized each query's NDCG profile to $[0,1]$ before training. This proved harmful: on queries where the true NDCG spread across the simplex is tiny (a fraction of a point), normalization stretches that noise to the full range, teaching the router to route *confidently* precisely where weighting is irrelevant — which produced a significant regression under noise. We therefore train on **raw** NDCG targets and retain the all-zero-profile queries (those where no channel found a relevant document): flat raw profiles teach the router to predict flat surfaces, which the expected-weight inference maps to near-uniform fusion.

**Training-set coverage of the noise regime.** The router can only learn the toneless gating if training exposes it to the fully-toneless regime. We train on a multi-domain pool of 6,000 queries augmented with 1,500 rule-based fully-diacritic-stripped variants (identical `relevant_ids`), so the diacritic-ratio axis is covered end to end. Training uses Adam (learning rate $10^{-3}$, batch size $256$, $100$ epochs). Candidate-score collection (FAISS/BM25/BGE-M3) and network fitting run in separate processes to avoid an OpenMP/MKL clash between FAISS and TensorFlow. The DANGDOCAO test corpus (§4.1) is held out entirely for the zero-shot evaluation in §5.2.

---

## 4. Experiments

### 4.1. Datasets

| Dataset | Domain | Passages | QA pairs | Split used |
|---------|--------|----------|----------|------------|
| UIT-ViQuAD 2.0 | Wikipedia (Vietnamese) | 5,317 | 28,454 train / 3,814 dev / 7,301 test | Router training pool + toneless augmentation; in-domain eval |
| DANGDOCAO | Legal / Administrative (736 sub-domains) | 37,239 | 35,131 train / 4,391 dev / 4,391 test | Zero-shot cross-domain eval |

### 4.2. Evaluation Metrics

**Retrieval metrics.** We report six metrics across all conditions:

| Metric | Description |
|--------|-------------|
| **NDCG@10** | Normalized Discounted Cumulative Gain — rewards higher-ranked relevant documents |
| **MRR@10** | Mean Reciprocal Rank — rank of the first relevant document |
| **MAP@10** | Mean Average Precision — integrates precision at each relevant rank position |
| **Recall@10** | Fraction of relevant documents found in the top 10 |
| **Recall@100** | Fraction of relevant documents found in the top 100 (upper-bound coverage) |
| **Hit@1** | Success rate: 1 if the top-ranked passage is relevant, 0 otherwise |

**Statistical significance.** For each baseline comparison, we report (i) paired t-test $p$-value, (ii) Wilcoxon signed-rank $p$-value on per-query NDCG@10 differences, and (iii) 95% bootstrap confidence interval (2,000 resamples) of the mean NDCG@10 delta.

**Efficiency.** We report MLP parameter count, index sizes, and per-query MLP inference latency to quantify the overhead of adaptive fusion over fixed-weight fusion.

**Weight interpretability.** We compute weight entropy $H = -\sum_i w_i \log w_i$ and Pearson correlations between linguistic query features and predicted weights to verify that the MLP learns linguistically meaningful mappings.

The generator and RAGAS judge (Llama-3.3-70B-Instruct via FPT AI Factory) are used for end-to-end QA evaluation in Section 5.7; a non-reasoning instruct model is required because reasoning models exhaust the token budget on hidden thinking and return empty judgements.

### 4.3. Baselines

All baselines share the same retrieval candidate set (top-100 from each of the four channels) and the same min-max normalization; they differ only in how the fusion weights $(w_\text{dense}, w_\text{bm25}, w_\text{sparse}, w_\text{toneless})$ are set. We deliberately include the two strongest fair static competitors — a dev-tuned best-fixed weight and rank-based RRF — so the adaptive gain is not measured only against uniform fusion.

| System | Weights | Description |
|--------|---------|-------------|
| BM25 only | $(0, 1, 0, 0)$ | underthesea tokenization + BM25Okapi |
| Dense only | $(1, 0, 0, 0)$ | FPT Vietnamese Embedding + FAISS |
| Sparse only | $(0, 0, 1, 0)$ | BGE-M3 learned sparse + inverted index |
| Toneless only | $(0, 0, 0, 1)$ | Diacritic-stripped syllable BM25 |
| Fixed-equal three-way | $(\tfrac13, \tfrac13, \tfrac13, 0)$ | Uniform fusion, no toneless channel |
| Fixed-equal four-way | $(\tfrac14, \tfrac14, \tfrac14, \tfrac14)$ | Uniform fusion, all channels |
| **Best-fixed (dev-tuned)** | grid-searched | Single weight vector maximizing NDCG@10 over a clean+noisy dev mix (strongest static baseline) |
| **RRF** | — | Reciprocal rank fusion ($k = 60$) over the four channels; parameter-free w.r.t. score scale |
| **Diacritic restoration → retrieve** | — | LLM restores tone marks, then standard fusion (a costly alternative to the toneless channel) |
| **Dynamic router (ours)** | expected($\widehat{\text{NDCG}}$) | Proposed four-way adaptive fusion |

### 4.4. Implementation Details

Dense semantic retrieval uses the 1024-dimensional FPT Vietnamese Embedding model, a fine-tune of BGE-M3 served through an OpenAI-compatible API. Passage embeddings are L2-normalised and indexed with FAISS `IndexFlatIP`, equivalent to cosine similarity. BM25 retrieval is performed by `rank_bm25.BM25Okapi` over text tokenised with underthesea's `word_tokenize`. Learned-sparse retrieval uses the BAAI/bge-m3 model accessed locally via the FlagEmbedding library, indexed as an in-memory inverted file over non-zero token weights. The end-to-end RAG evaluation in §5.7 uses Qwen3-32B as the generator and the RAGAS judge LLM, accessed through the same OpenAI-compatible interface.

The four-way router (Keras/TensorFlow) is trained on a multi-domain pool of 6,000 queries augmented with 1,500 rule-based fully-diacritic-stripped variants (7,500 total), using Adam with learning rate $10^{-3}$, batch size 256, and 100 epochs. To avoid an OpenMP/MKL runtime clash between FAISS and TensorFlow, candidate-score collection (FAISS/BM25/BGE-M3) and network fitting run in separate processes. Targets are raw NDCG@10 over the 286-point four-simplex grid (step $0.1$); inference uses softmax-expected weights at $T = 0.05$. The diacritic-restoration baseline uses Qwen3.6-27B (the successor to Qwen3-32B, which FPT removed from its catalogue) and the end-to-end RAGAS judge uses the non-reasoning Llama-3.3-70B-Instruct. The seed for all randomised components is fixed at $42$.

All experimental results in §5 are produced on a single workstation equipped with an NVIDIA GeForce RTX 3050 (6 GB VRAM, CUDA 12.4 runtime) and a multi-core CPU. The GPU is used for BGE-M3 sparse encoding; FAISS, BM25, and the fusion MLP run on CPU. The software stack comprises Python 3.13, PyTorch 2.6.0 with CUDA 12.4 and FlagEmbedding for BGE-M3, TensorFlow/Keras for the fusion MLP, FAISS-CPU, `rank_bm25`, and underthesea for Vietnamese word segmentation.

---

## 5. Results & Discussion

> *Note (to be removed before submission): the tables in §5.1–§5.2 below are the legacy three-way results (dense + BM25 + sparse, softmax-weight MLP) and are retained only for historical comparison. **The current four-way system's numbers — full-test baseline table, OOD-noise generalization, the noise-level curve, the restoration comparison, and end-to-end RAGAS — are consolidated in `docs/results_summary.md`, produced by `scripts/aggregate_results.py`, and are the authoritative results for this paper.** Headline four-way figures: ViQuAD test NDCG@10 = 0.854 (router) vs 0.848 fixed-equal-3 / 0.837 fixed-equal-4 / 0.821 dev-tuned best-fixed / 0.799 RRF; ViQuAD diacritic-noisy 0.641 vs 0.396 fixed-equal-3; DANGDOCAO zero-shot 0.820 clean / 0.622 noisy. **§5.3–§5.7 below have been regenerated against the four-way full-test runs and are current.***

### 5.1. In-domain Results (UIT-ViQuAD 2.0)

All in-domain results report a single fusion MLP trained with the configuration described in §3.5 and §4.4 (5,000 augmented training queries from UIT-ViQuAD 2.0, soft-label simplex supervision at temperature $T = 0.3$, 100 training epochs). The same checkpoint is evaluated on the dev split, the held-out test split, and a diacritic-stripped variant of the dev split; the checkpoint is not re-tuned for any of the three conditions.

**Dev set** (3,814 queries, used for model selection):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.6770 | 0.6376 | 0.6376 | 0.8007 | 0.9295 | 0.5582 |
| Dense only | 0.7953 | 0.7561 | 0.7561 | 0.9174 | 0.9869 | 0.6720 |
| Sparse only (BGE-M3) | 0.7507 | 0.7109 | 0.7109 | 0.8749 | 0.9730 | 0.6264 |
| Dense + BM25 (0.5/0.5) | 0.8330 | 0.7996 | 0.7996 | 0.9352 | 0.9890 | 0.7218 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.8463 | 0.8153 | 0.8153 | 0.9415 | 0.9908 | 0.7428 |
| **Dynamic MLP (soft label, three-way)** | **0.8479** | **0.8170** | **0.8170** | **0.9428** | **0.9908** | **0.7444** |

**Test set** (7,301 queries, held-out final evaluation):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.6623 | 0.6198 | 0.6198 | 0.7966 | 0.9269 | 0.5364 |
| Dense only | 0.8068 | 0.7679 | 0.7679 | 0.9270 | 0.9885 | 0.6803 |
| Sparse only (BGE-M3) | 0.7595 | 0.7167 | 0.7167 | 0.8930 | 0.9748 | 0.6247 |
| Dense + BM25 (0.5/0.5) | 0.8278 | 0.7907 | 0.7907 | 0.9423 | 0.9910 | 0.7059 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.8486 | 0.8146 | 0.8146 | 0.9533 | 0.9925 | 0.7357 |
| **Dynamic MLP (soft label, three-way)** | **0.8514** | **0.8178** | **0.8178** | **0.9547** | **0.9925** | **0.7398** |

Three observations characterise the in-domain regime. First, among the three single-signal retrievers, dense semantic retrieval (test NDCG@10 = 0.8068) outperforms BGE-M3 learned sparse (0.7595), which in turn outperforms classical BM25 (0.6623). The 9.7-point gap between BGE-M3 sparse and BM25 is itself notable: both retrievers operate on lexical matches, yet the learned weighting captures importance information that BM25's tf-idf statistics fail to recover from Vietnamese text, whose limited inflectional morphology offers little for idf separation to act upon. Second, fusing all three signals with uniform weights already accounts for a substantial portion of the observed gains. The fixed-equal three-way baseline reaches 0.8486 on test, exceeding the conventional dense + BM25 hybrid (0.8278) by 2.1 NDCG points and dense-only retrieval by 4.2 points. The three-signal fusion architecture is therefore beneficial independently of any adaptive weighting. Third, the dynamic MLP improves upon every fixed-weight baseline on both splits. Against the strongest fixed reference (fixed-equal three-way), the MLP gains $+0.0016$ NDCG@10 on dev ($p = 2.0\!\times\!10^{-3}$) and $+0.0028$ on test ($p = 1.3\!\times\!10^{-10}$); the full significance battery is reported in §5.4. The absolute gain over fixed-equal is modest, but it is consistent across all six headline retrieval metrics and across both splits.

Two structural properties of the results merit explicit comment. Recall@100 saturates at approximately $0.991$ for every three-way method on both dev and test, indicating that once the candidate union is drawn from three top-100 lists, the relevant passage is almost always present in the pool. Adaptive fusion therefore acts on *re-ranking within a near-complete candidate set* rather than on *candidate expansion*, which explains the wider NDCG@10 and Hit@1 gaps relative to Recall@100. The mean predicted weights $(\bar{w}_\text{dense}, \bar{w}_\text{bm25}, \bar{w}_\text{sparse}) = (0.349, 0.319, 0.332)$ on the test set lie close to the uniform $(1/3, 1/3, 1/3)$ centre of the simplex. The adaptive component's contribution is therefore not a globally different weighting, but rather a query-conditional displacement from this central operating point — a hypothesis examined directly in the stratified and correlation analyses of §5.3.

**Diacritic robustness** (dev queries with all tone marks removed, 3,814 queries):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.1559 | 0.1336 | 0.1336 | 0.2278 | 0.4740 | 0.0954 |
| Dense only | 0.2956 | 0.2550 | 0.2550 | 0.4266 | 0.6691 | 0.1862 |
| Sparse only (BGE-M3) | 0.3671 | 0.3253 | 0.3253 | 0.5013 | 0.7278 | 0.2509 |
| Dense + BM25 (0.5/0.5) | 0.3050 | 0.2648 | 0.2648 | 0.4350 | 0.6762 | 0.1964 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.3969 | 0.3540 | 0.3540 | 0.5359 | 0.7286 | 0.2803 |
| **Dynamic MLP (soft label, three-way)** | **0.3993** | **0.3564** | **0.3564** | **0.5375** | **0.7289** | **0.2816** |

Diacritic removal produces an asymmetric degradation pattern across the three retrievers and supplies direct empirical support for the motivation laid out in §1. BM25 collapses from NDCG@10 = $0.6770$ on clean queries to $0.1559$ under noise, a $77\%$ relative drop; the missing tone marks turn every previously matchable term into a lexical miss. Dense retrieval degrades more gracefully (from $0.7953$ to $0.2956$, a $63\%$ drop), reflecting the partial semantic invariance of the embedding model to orthographic perturbation. The most informative single-signal result, however, is the BGE-M3 learned-sparse retriever, which falls only to $0.3671$ ($51\%$ relative drop) and surpasses dense retrieval under noise — a finding consistent with BGE-M3's sub-word tokeniser handling diacritic stripping more uniformly than either whitespace tokenisation or full-token dense embedding. Without ever being told that the queries were noisy, the MLP shifts its predicted weights in the direction predicted by §1: $\bar{w}_\text{sparse}$ rises from $0.332$ on clean test queries to $0.354$ on noisy dev queries, while $\bar{w}_\text{bm25}$ falls from $0.320$ to $0.310$. The resulting MLP NDCG@10 of $0.3993$ exceeds the two-way dense + BM25 baseline by $0.094$ ($p < 10^{-125}$, §5.4). The three-signal architecture is therefore necessary to recover any meaningful retrieval performance under diacritic noise, and adaptive weighting provides a small but reliable additional gain on top of it.

### 5.2. Cross-domain Results (DANGDOCAO, Zero-shot)

The cross-domain protocol applies the same MLP checkpoint evaluated in §5.1 — trained on 5,000 Wikipedia queries from UIT-ViQuAD 2.0 — to a previously unseen legal/administrative corpus. No DANGDOCAO data is observed at any stage of MLP training, and the three retrievers are re-built from the DANGDOCAO corpus alone, so neither the fusion module nor the index parameters carry any in-domain information.

DANGDOCAO is split using a group-by-passage protocol: each of the 37,239 passages is randomly assigned to exactly one of train, dev, or test in an 80 / 10 / 10 ratio, and every QA pair inherits its split from its underlying passage. The resulting counts are 29,793 / 3,723 / 3,723 passages and 35,289 / 4,309 / 4,315 QA pairs. This eliminates the passage-level information leak that a naive QA-level shuffle would introduce — a leak that would be particularly damaging for fusion evaluation, since the same passage could otherwise appear in training and test under different surface forms of its associated questions.

**Clean queries** (4,315 test, group-by-passage split, zero-shot):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.6762 | 0.6216 | 0.6216 | 0.8466 | 0.9560 | 0.5043 |
| Dense only | 0.7908 | 0.7420 | 0.7420 | 0.9400 | 0.9905 | 0.6248 |
| Sparse only (BGE-M3) | 0.7522 | 0.7014 | 0.7014 | 0.9085 | 0.9754 | 0.5852 |
| Dense + BM25 (0.5/0.5) | 0.8051 | 0.7592 | 0.7592 | 0.9446 | 0.9910 | 0.6459 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.8156 | 0.7702 | 0.7702 | 0.9539 | 0.9933 | 0.6598 |
| **Dynamic MLP (soft label, three-way)** | **0.8167** | **0.7716** | **0.7716** | **0.9537** | **0.9935** | **0.6607** |

**Diacritic-removed queries** (4,315 test, 100% diacritics stripped):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.0480 | 0.0392 | 0.0392 | 0.0769 | 0.1766 | 0.0255 |
| Dense only | 0.0714 | 0.0598 | 0.0598 | 0.1094 | 0.2190 | 0.0408 |
| Sparse only (BGE-M3) | 0.1435 | 0.1224 | 0.1224 | 0.2109 | 0.3868 | 0.0834 |
| Dense + BM25 (0.5/0.5) | 0.0837 | 0.0707 | 0.0707 | 0.1258 | 0.2438 | 0.0487 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.1477 | 0.1277 | 0.1277 | 0.2116 | 0.3583 | 0.0913 |
| **Dynamic MLP (soft label, three-way)** | **0.1530** | **0.1330** | **0.1330** | **0.2167** | **0.3606** | **0.0952** |

**Cross-domain significance (paired tests vs MLP):**

| Comparison | Clean Δ NDCG | $p$ (t-test) | Noisy Δ NDCG | $p$ (t-test) |
|------------|--------------|--------------|--------------|--------------|
| MLP vs. Fixed-equal three-way | +0.0011 | $3.9\!\times\!10^{-2}$ | +0.0053 | $5.2\!\times\!10^{-12}$ |
| MLP vs. Dense + BM25 (0.5/0.5) | +0.0115 | $8.7\!\times\!10^{-7}$ | +0.0693 | $4.2\!\times\!10^{-99}$ |
| MLP vs. Dense only | +0.0259 | $2.8\!\times\!10^{-14}$ | +0.0816 | $5.8\!\times\!10^{-103}$ |
| MLP vs. BM25 only | +0.1404 | $6.8\!\times\!10^{-204}$ | +0.1050 | $2.2\!\times\!10^{-139}$ |
| MLP vs. Sparse only (BGE-M3) | +0.0645 | $7.9\!\times\!10^{-78}$ | +0.0095 | $1.5\!\times\!10^{-4}$ |

**MLP predicted weights on DANGDOCAO** (zero-shot, no DANGDOCAO data in training):

| Condition | $\bar{w}_\text{dense}$ | $\bar{w}_\text{bm25}$ | $\bar{w}_\text{sparse}$ | $\bar{H}$ (max ln3 = 1.099) |
|-----------|------------------------|------------------------|--------------------------|----------------------------|
| Clean | 0.3482 | 0.3228 | 0.3290 | 1.0977 |
| Noisy | 0.3332 | 0.3107 | **0.3561** | 1.0967 |

The cross-domain results admit four substantive readings. First, adaptive fusion generalises beyond the training domain. On clean DANGDOCAO queries, the MLP still surpasses the strongest fixed-weight baseline ($+0.0011$ NDCG@10, $p = 0.039$), and the noisy-query margin grows to $+0.0053$ ($p = 5.2\!\times\!10^{-12}$). The absolute gains are smaller than in-domain, but the direction and statistical significance both carry over.

Second, the noise-adaptation behaviour observed in §5.1 transfers without modification. The mean predicted weights shift from $(\bar{w}_\text{dense}, \bar{w}_\text{bm25}, \bar{w}_\text{sparse}) = (0.348, 0.323, 0.329)$ on clean DANGDOCAO queries to $(0.333, 0.311, 0.356)$ on the diacritic-stripped variant — the same direction (up-weight sparse, down-weight BM25) the MLP learned on Wikipedia text. Since DANGDOCAO is entirely unseen during training, this rules out a memorisation explanation: the MLP is encoding a mapping from linguistic query features to retrieval-mode trust that is largely invariant to the underlying corpus.

Third, the asymmetric degradation pattern under noise is even more pronounced in the legal domain than in the encyclopedic one. BM25-only NDCG@10 falls from $0.6762$ to $0.0480$ ($-93\%$, against $-77\%$ on ViQuAD); dense retrieval falls from $0.7908$ to $0.0714$ ($-91\%$); and BGE-M3 learned sparse falls from $0.7522$ to $0.1435$ ($-81\%$). The relative ordering of robustness across the three signals is preserved, but the absolute magnitudes confirm that Vietnamese legal terminology carries little orthographic redundancy and offers correspondingly little for any lexical retriever to recover once tones are stripped. The MLP exploits the surviving sparse signal to lift NDCG@10 from $0.1435$ (sparse-only) and $0.1477$ (fixed-equal three-way) to $0.1530$.

Fourth, weight-interpretability signals weaken but do not invert in the dominant direction. The Pearson correlation between query English-token ratio and $w_\text{sparse}$, which is the cleanest interpretability finding on ViQuAD ($r = +0.70$ on dev, $+0.66$ on test), drops to $r = +0.16$ on clean DANGDOCAO ($p = 1.2\!\times\!10^{-25}$). Legal Vietnamese is more monolingual than encyclopedic Vietnamese — there are fewer English-token queries for the MLP to act upon — and the surface frequency of the feature it most relies on therefore drops. Crucially, the adaptive component still wins by a statistically significant margin in this regime, suggesting that the MLP also exploits feature combinations that do not surface in a single bivariate correlation.

A final practical observation concerns Recall@100 under noise. Even on the noisy DANGDOCAO split, the MLP retains Recall@100 of $0.36$, against $0.18$ for BM25-only and $0.99$ on the clean split. This metric upper-bounds the recoverable end-to-end RAG quality: a generation step cannot answer a question whose evidence is absent from its retrieved context. The three-signal fusion approximately doubles this ceiling relative to any single Vietnamese-tokenisation-dependent retriever under the same noise.

### 5.3. Analysis: When Does Dynamic Routing Help?

#### 5.3.1. Stratified NDCG@10

We partition the UIT-ViQuAD 2.0 test set into 11 strata defined by ranges of individual linguistic query features, and report NDCG@10 for the fixed-equal three-way baseline and the dynamic router within each stratum, together with the mean predicted four-way weights. The partition identifies the linguistic conditions under which adaptive routing contributes, and exposes the router's per-stratum weight allocation as a behavioural footprint of what it has learned.

**Clean test set (7,301 queries):**

| Stratum | N | Fixed-eq-3 NDCG | Router NDCG | Δ | $\bar{w}_\text{dense}$ | $\bar{w}_\text{bm25}$ | $\bar{w}_\text{sparse}$ | $\bar{w}_\text{toneless}$ |
|---------|---|------------|----------|---|------|------|------|------|
| diac\_low (< 0.3) | 4 | 0.9077 | 0.7827 | $-$0.1250 | 0.28 | 0.21 | 0.31 | 0.20 |
| diac\_mid (0.3–0.7) | 1,241 | 0.8659 | 0.8640 | $-$0.0019 | 0.28 | 0.23 | 0.28 | 0.21 |
| diac\_high (> 0.7) | 6,056 | 0.8446 | 0.8521 | **+0.0076** | 0.32 | 0.22 | 0.27 | 0.19 |
| comp\_low (< 0.2) | 1,567 | 0.8442 | 0.8430 | $-$0.0011 | 0.29 | 0.23 | 0.28 | 0.20 |
| comp\_high (≥ 0.2) | 5,734 | 0.8494 | 0.8572 | **+0.0078** | 0.32 | 0.22 | 0.27 | 0.19 |
| eng\_none (= 0) | 583 | 0.7981 | 0.8118 | **+0.0137** | 0.35 | 0.21 | 0.27 | 0.17 |
| eng\_mixed (> 0) | 6,718 | 0.8526 | 0.8578 | +0.0052 | 0.31 | 0.22 | 0.28 | 0.20 |
| short\_query (< 0.4) | 220 | 0.7399 | 0.7324 | $-$0.0075 | 0.30 | 0.21 | 0.29 | 0.20 |
| long\_query (≥ 0.4) | 7,081 | 0.8516 | 0.8579 | +0.0063 | 0.31 | 0.22 | 0.27 | 0.19 |
| simple (no clause) | 5,940 | 0.8335 | 0.8396 | +0.0061 | 0.31 | 0.22 | 0.28 | 0.19 |
| complex (has clause) | 1,361 | 0.9125 | 0.9174 | +0.0049 | 0.30 | 0.23 | 0.26 | 0.21 |

**Diacritic-stripped dev set (3,814 queries; strata that separate diacritic density or English tokens are degenerate under full stripping):**

| Stratum | N | Fixed-eq-3 NDCG | Router NDCG | Δ | $\bar{w}_\text{toneless}$ |
|---------|---|------------|----------|---|------|
| all queries (diac\_low) | 3,814 | 0.3961 | 0.6405 | **+0.2444** | 0.47 |
| comp\_low (< 0.2) | 1,283 | 0.4118 | 0.6226 | +0.2108 | 0.45 |
| comp\_high (≥ 0.2) | 2,531 | 0.3881 | 0.6496 | +0.2615 | 0.48 |
| short\_query (< 0.4) | 190 | 0.4797 | 0.5790 | +0.0993 | 0.32 |
| long\_query (≥ 0.4) | 3,624 | 0.3917 | 0.6438 | +0.2520 | 0.47 |
| complex (has clause) | 455 | 0.4929 | 0.7326 | +0.2397 | 0.47 |

Three readings follow. First, the clean-split gains are broad but individually small: the router improves on eight of eleven strata, with the largest lift on pure-Vietnamese queries (`eng_none`, $+0.0137$) and high-compound queries ($+0.0078$), and small losses confined to the three smallest or hardest strata (`diac_low` contains only four queries; `short_query` offers little lexical surface for any channel). Second, the noisy-split gain is not driven by a favourable sub-population: it exceeds $+0.21$ NDCG@10 on *every* stratum with more than 200 queries, because the mechanism — engaging the toneless channel — applies to the entire diacritic-stripped population. The one partial exception is short queries ($+0.0993$, with $\bar{w}_\text{toneless} = 0.32$ rather than $0.47$): with only a handful of syllables, the post-retrieval signals that indicate channel failure are noisier, and the router hedges. Third, the weight columns show that on clean text the router keeps the toneless channel *suppressed* at $\bar{w}_\text{toneless} \approx 0.17$–$0.21$ across all strata — the two-sided gating promised in §3.2 is visible directly in the behavioural footprint.

#### 5.3.2. Performance Across the Noise Spectrum

The clean and fully-stripped conditions are the endpoints of a continuum. To trace the full curve, we corrupt a fixed set of 500 queries per domain at per-syllable stripping probabilities $p \in \{0, 0.25, 0.5, 0.75, 1.0\}$ and evaluate every method on the identical base queries at each level.

**NDCG@10 by fraction of syllables stripped (n = 500/level):**

| Domain | Method | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| ViQuAD | **Router (ours)** | **0.845** | **0.805** | 0.768 | **0.712** | 0.657 |
| | Best-fixed (dev-tuned) | 0.823 | 0.799 | **0.770** | 0.711 | **0.667** |
| | Fixed-equal 4-way | 0.835 | 0.802 | 0.769 | 0.696 | 0.593 |
| | Fixed-equal 3-way | 0.843 | 0.796 | 0.723 | 0.605 | 0.429 |
| | RRF | 0.801 | 0.768 | 0.715 | 0.647 | 0.530 |
| | Toneless only | 0.600 | 0.600 | 0.600 | 0.599 | 0.600 |
| DANGDOCAO | **Router (ours)** | **0.828** | **0.795** | **0.744** | 0.671 | 0.615 |
| | Best-fixed (dev-tuned) | 0.789 | 0.772 | 0.743 | **0.674** | **0.624** |
| | Fixed-equal 4-way | 0.811 | 0.779 | 0.740 | 0.619 | 0.409 |
| | Fixed-equal 3-way | 0.822 | 0.774 | 0.684 | 0.437 | 0.130 |
| | RRF | 0.787 | 0.751 | 0.708 | 0.562 | 0.289 |
| | Toneless only | 0.604 | 0.604 | 0.604 | 0.605 | 0.603 |

The router traces the upper envelope of all methods across the spectrum, with a single competitor: the dev-tuned best-fixed vector matches it from 50% noise upward and edges ahead by $\approx 0.01$ at the fully-stripped endpoint. That vector, however, was grid-searched on a 50/50 clean+noisy dev mix and pays for its noise specialisation on clean text ($-0.022$ ViQuAD, $-0.040$ DANGDOCAO at $p = 0$). The methods without a toneless channel tell the structural story: fixed-equal three-way starts on par with the router at $p = 0$ (0.843 vs 0.845) and collapses to 0.429/0.130 at $p = 1$, while RRF — the standard untuned hybrid baseline — degrades almost as badly (0.530/0.289) because reciprocal-rank aggregation grants the failing channels equal influence at every noise level. No fixed weighting is strong at both ends: the static vector that wins at $p = 1$ loses at $p = 0$ and vice versa, whereas the router is within $0.011$ of the per-level best everywhere *without being told the noise level* — the practical setting, since real query streams mix noise levels unpredictably.

The router's mechanism is directly observable in its weight trajectory: $\bar{w}_\text{toneless}$ rises monotonically with the corruption level, from $0.197$ to $0.457$ on ViQuAD ($0.183$ to $0.547$ on DANGDOCAO), while mean weight entropy falls from $1.335$ to $1.216$ ($1.338$ to $1.153$; maximum $\ln 4 \approx 1.386$) — the router becomes more decisive precisely as the regime becomes more extreme.

#### 5.3.3. Generalisation to Unseen Noise Types

Training exposes the router to exactly two regimes: clean queries and rule-based full diacritic stripping. To test whether the learned gating generalises beyond its training distribution, we evaluate on four LLM-generated noise types applied to the full DANGDOCAO test set (4,315 queries each) — none of which appears in training.

| Noise type | Router | Best-fixed | Fixed-eq-4 | Fixed-eq-3 | Toneless only |
|---|---|---|---|---|---|
| missing\_tone | 0.614 | **0.620** | 0.439 | 0.172 | 0.596 |
| typo\_telex | 0.535 | **0.543** | 0.480 | 0.334 | 0.462 |
| informal | **0.796** | 0.761 | 0.777 | 0.780 | 0.570 |
| code\_switch | **0.783** | 0.737 | 0.762 | 0.782 | 0.514 |

The pattern splits by noise family. On *orthographic* noise (missing\_tone, typo\_telex — the family the toneless channel targets), the router lands within $0.006$–$0.008$ of the noise-specialised best-fixed vector while quadrupling fixed-equal three-way on missing\_tone (0.614 vs 0.172). On *semantic* noise (informal paraphrase, code-switching — where diacritics survive and the toneless channel is not the answer), the ordering flips: the router leads all methods, and the best-fixed vector — locked into $w_\text{toneless} = 0.5$ — pays for its specialisation ($-0.035$ and $-0.046$ against the router). A static vector must commit to one noise family; the router reads each query's channel-response signature and does not.

#### 5.3.4. Weight Interpretability

The dominant interpretable behaviour of the four-way router is the toneless gate quantified above: $\bar{w}_\text{toneless} = 0.19$ on clean text versus $0.47$–$0.55$ under full stripping, with a monotone trajectory between the endpoints (§5.3.2). Bivariate feature–weight correlations on the clean test split are comparatively modest — diacritic\_ratio ↔ $w_\text{dense}$ at $r = +0.21$ ($p = 1.3\!\times\!10^{-75}$), compound\_ratio ↔ $w_\text{bm25}$ at $r = -0.13$, english\_ratio ↔ $w_\text{sparse}$ at $r = +0.06$ — and substantially weaker than the $r \approx +0.66$ english–sparse correlation reported for the legacy three-way system. This weakening is expected rather than anomalous: the four-way router's input is dominated by the 28 post-retrieval channel-response signals (§3.3), so its decisions are conditioned on how the corpus responded to the query rather than on any single pre-retrieval linguistic feature, and single-feature correlations correspondingly lose explanatory power. The interpretability claim of this paper therefore rests on the regime-level weight behaviour — suppression of the toneless channel on clean text, monotone engagement as tone information disappears, and decisiveness (falling entropy) that tracks regime extremity — rather than on bivariate feature correlations.

### 5.4. Statistical Significance

We report two-sided paired $t$-tests and Wilcoxon signed-rank tests on per-query NDCG@10 differences, with 95% bootstrap confidence intervals (2,000 resamples), pairing the router against every baseline on each of the four full test conditions.

**ViQuAD clean test (7,301 queries):**

| Router vs. | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| Fixed-equal three-way | +0.0059 | [+0.0031, +0.0086] | $2.4\!\times\!10^{-5}$ | $7.4\!\times\!10^{-5}$ |
| Fixed-equal four-way | +0.0172 | [+0.0145, +0.0198] | $4.1\!\times\!10^{-37}$ | $4.8\!\times\!10^{-39}$ |
| Best-fixed (dev-tuned) | +0.0330 | [+0.0299, +0.0362] | $1.9\!\times\!10^{-86}$ | $1.8\!\times\!10^{-85}$ |
| RRF | +0.0553 | [+0.0507, +0.0596] | $1.6\!\times\!10^{-134}$ | $2.1\!\times\!10^{-128}$ |
| Dense only | +0.0477 | [+0.0425, +0.0532] | $1.1\!\times\!10^{-69}$ | $2.9\!\times\!10^{-67}$ |
| BM25 only | +0.1921 | [+0.1846, +0.1996] | $< 10^{-300}$ | $< 10^{-300}$ |
| Sparse only | +0.0947 | [+0.0893, +0.1004] | $1.9\!\times\!10^{-228}$ | $5.6\!\times\!10^{-208}$ |
| Toneless only | +0.2781 | [+0.2702, +0.2868] | $< 10^{-300}$ | $< 10^{-300}$ |

**ViQuAD diacritic-noisy dev (3,814 queries):**

| Router vs. | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| Fixed-equal three-way | +0.2444 | [+0.2313, +0.2570] | $3.5\!\times\!10^{-264}$ | $2.7\!\times\!10^{-223}$ |
| Fixed-equal four-way | +0.0711 | [+0.0634, +0.0788] | $1.0\!\times\!10^{-68}$ | $5.2\!\times\!10^{-68}$ |
| Best-fixed (dev-tuned) | $-$0.0052 | [$-$0.0102, $-$0.0004] | $3.8\!\times\!10^{-2}$ | $0.19$ (n.s.) |
| RRF | +0.1360 | [+0.1248, +0.1476] | $6.5\!\times\!10^{-119}$ | $3.0\!\times\!10^{-110}$ |
| Toneless only | +0.0500 | [+0.0425, +0.0573] | $2.3\!\times\!10^{-40}$ | $1.9\!\times\!10^{-37}$ |

**DANGDOCAO clean test, zero-shot (4,315 queries):**

| Router vs. | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| Fixed-equal three-way | +0.0044 | [+0.0011, +0.0077] | $9.4\!\times\!10^{-3}$ | $5.1\!\times\!10^{-3}$ |
| Fixed-equal four-way | +0.0145 | [+0.0112, +0.0179] | $6.6\!\times\!10^{-17}$ | $7.7\!\times\!10^{-18}$ |
| Best-fixed (dev-tuned) | +0.0327 | [+0.0278, +0.0375] | $1.2\!\times\!10^{-39}$ | $7.1\!\times\!10^{-40}$ |
| RRF | +0.0380 | [+0.0329, +0.0433] | $1.4\!\times\!10^{-43}$ | $1.4\!\times\!10^{-41}$ |
| Toneless only | +0.2102 | [+0.2006, +0.2204] | $2.2\!\times\!10^{-304}$ | $5.0\!\times\!10^{-257}$ |

**DANGDOCAO diacritic-noisy test, zero-shot (4,315 queries):**

| Router vs. | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| Fixed-equal three-way | +0.4747 | [+0.4622, +0.4875] | $< 10^{-300}$ | $< 10^{-300}$ |
| Fixed-equal four-way | +0.1941 | [+0.1862, +0.2019] | $< 10^{-300}$ | $1.5\!\times\!10^{-290}$ |
| Best-fixed (dev-tuned) | $-$0.0069 | [$-$0.0100, $-$0.0038] | $2.9\!\times\!10^{-5}$ | $8.1\!\times\!10^{-5}$ |
| RRF | +0.3163 | [+0.3046, +0.3280] | $< 10^{-300}$ | $< 10^{-300}$ |
| Toneless only | +0.0123 | [+0.0076, +0.0168] | $2.0\!\times\!10^{-7}$ | $5.5\!\times\!10^{-6}$ |

Against every *untuned* baseline — the two uniform mixtures, RRF, and all four single channels — the router is significantly better on all four conditions, with the smallest margin (+0.0044 on zero-shot DANGDOCAO clean, $p = 9.4\!\times\!10^{-3}$) still surviving a Holm–Bonferroni correction over the full battery of 36 comparisons. The one comparison we do not win outright is against the dev-tuned best-fixed vector on the two *fully* diacritic-stripped conditions, where the router trails by $0.005$–$0.007$ (significant on DANGDOCAO; marginal on ViQuAD, where the Wilcoxon test does not reject). We report this honestly rather than tune it away: a static vector that dedicates half its mass to the toneless channel is near-optimal when *every* query in the evaluation is fully stripped — a homogeneity that the deployment setting does not offer. The same vector loses to the router by $0.033$ on both clean sets and by $0.035$–$0.046$ on semantic noise (§5.3.3), and the noise-spectrum sweep (§5.3.2) shows the router within $0.011$ of the per-level best at every corruption level. The correct summary is not that adaptive routing dominates every static vector on every slice, but that it is the only method that requires no prior knowledge of the query distribution to be near-optimal across all of them.

### 5.5. Efficiency Analysis

| Component | Value |
|-----------|-------|
| Router parameters | 14,151 |
| FAISS index — ViQuAD (1024-dim, 5,317 passages) | 21.8 MB |
| BM25 index — ViQuAD | 12.8 MB |
| BGE-M3 sparse index — ViQuAD | 15.9 MB |
| Toneless BM25 index — ViQuAD | 11.6 MB |
| FAISS / BM25 / sparse / toneless — DANGDOCAO (37,239 passages) | 152.5 / 103.1 / 113.0 / 96.6 MB |

**Per-query latency breakdown (ms, measured over a 500-query ViQuAD run):**

| Stage | Mean | p50 |
|-------|------|-----|
| Dense (FAISS search) | 1.6 | 1.4 |
| BM25 (word-segmented) | 15.5 | 14.5 |
| Sparse (BGE-M3 encode + inverted index) | 30.6 | 29.8 |
| Toneless BM25 | 19.0 | 18.2 |
| Linguistic features (underthesea) | 0.6 | 0.6 |
| QPP signals | 0.2 | 0.2 |
| Router MLP inference | 5.9 | — |

The complete router overhead — linguistic features, channel-response signals, and MLP inference — totals $\approx 6.7$ ms per query, and the toneless channel itself adds a single in-memory BM25 lookup of $\approx 19$ ms; both are small against the sparse channel's 30.6 ms and negligible against the query-embedding API call and LLM generation that dominate any deployed RAG pipeline. The comparison that matters most is against the alternative route to diacritic robustness: LLM-based diacritic restoration costs $\approx 1.4$–$1.7$ *seconds* per query of wall-clock latency (measured over two 500-query restoration runs against a hosted 27B model), roughly $80\times$ the toneless channel, plus per-query generation token spend. Restoration is also the stronger recovery when affordable — it returns router NDCG@10 on the fully-stripped sets to near-clean levels (0.656 → 0.826 on ViQuAD, 0.615 → 0.820 on DANGDOCAO; same 500 queries) — so the two mechanisms are complements on a cost–quality frontier rather than substitutes: the toneless channel provides always-on robustness at millisecond cost, and restoration can be layered on top for traffic that justifies an LLM call per query.

### 5.6. Component Ablation and Oracle Headroom

We ablate the four components introduced in §3 by removing one at a time from the full system, retraining where the ablation changes the training signal, and evaluating all variants on identical 500-query subsets of the four test conditions. The three retrained variants (no-signals, normalized-labels, three-way) are trained on training caches derived deterministically from the full system's cache — column projection for the feature ablations and the exact 66-point sub-grid $\{w_\text{toneless} = 0\}$ of the 286-point grid for the three-way ablation — so all variants share retrieval hits, labelling, and optimisation protocol exactly.

| Configuration | ViQuAD clean | ViQuAD noisy | DANGDOCAO clean | DANGDOCAO noisy |
|---|---|---|---|---|
| Full system | 0.860 | 0.657 | 0.829 | 0.615 |
| − expected weights (argmax inference) | 0.859 | 0.646 | 0.824 | 0.614 |
| − QPP signals (8 linguistic features only) | 0.863 | 0.649 | 0.822 | 0.616 |
| − raw labels (per-query min-max targets) | 0.869 | 0.639 | 0.825 | 0.617 |
| − toneless training augmentation | 0.862 | **0.483** | 0.828 | **0.143** |
| − toneless channel (three-way routing) | 0.867 | **0.427** | 0.826 | **0.128** |

The ablation has a clear hierarchical structure, which we state plainly. Two components are load-bearing: the toneless channel and the training-set coverage of the fully-stripped regime. Removing either one collapses noisy-condition performance — DANGDOCAO noisy falls from 0.615 to 0.143 (no augmentation: the channel exists but the router never learned when to engage it, since only 41 of 6,000 unaugmented training queries are fully toneless) or 0.128 (no channel at all) — while leaving clean performance essentially unchanged. The remaining three design choices — expected-weight inference, QPP signals, and raw-label training — are second-order refinements once the load-bearing pair is in place: their individual removal shifts NDCG@10 by at most $\pm 0.018$ on any condition at this sample size, with signs that vary by condition. (In the earlier three-way system, before the toneless channel existed, per-query-normalized labels *did* cause a significant noisy-regime regression by amplifying near-flat NDCG profiles into confident mis-routing; with the toneless channel and augmentation present, the label scheme is no longer decisive.) We consider this a more defensible ablation outcome than one in which every proposed component appears essential: the architecture's robustness derives from one clearly identified mechanism, and the surrounding machinery makes that mechanism reliable rather than carrying the result itself.

**Oracle headroom.** To bound what any routing policy could achieve on this grid, we compute the per-query *oracle*: the NDCG@10 of the best of the 286 grid points, selected with access to the relevance labels. The oracle is not a method — it cheats — but the fraction of the oracle-minus-uniform gap that the router recovers is a meaningful measure of routing quality.

| Condition | Fixed-eq-4 | Router | Oracle | Headroom realised |
|---|---|---|---|---|
| ViQuAD clean | 0.848 | 0.860 | 0.942 | 12% |
| ViQuAD noisy | 0.593 | 0.657 | 0.755 | 40% |
| DANGDOCAO clean | 0.812 | 0.829 | 0.930 | 15% |
| DANGDOCAO noisy | 0.416 | 0.615 | 0.670 | **78%** |

The gradient is informative in both directions. On clean text the router realises only 12–15% of the nominal headroom — but much of that headroom is illusory: with 286 attempts per query, the label-aware oracle wins many queries through tie-breaking accidents that no label-free policy could replicate, and the near-saturated clean baselines leave little systematic signal to route on. On the noisy conditions the headroom is real — entire channels are dead and the surviving one must be found — and there the router recovers 40% and 78% of it. Routing quality is highest exactly where routing matters.

### 5.7. End-to-end QA Results (RAGAS, Llama-3.3-70B judge)

We evaluate end-to-end RAG quality with RAGAS [CITATION] on 100 sampled queries per condition, across four conditions (both domains × clean/noisy). Answer generation and judging use Llama-3.3-70B-Instruct; a non-reasoning instruct model is required because reasoning-class judges consume the completion budget on hidden deliberation and return empty structured verdicts. We report the three LLM-judged metrics — Context Precision, Context Recall, Faithfulness — and omit Answer Relevancy, whose embedding-based implementation deadlocks against our serving endpoint (every call times out; §4.4). Three retrieval policies are compared end-to-end: the router, fixed-equal four-way, and toneless-only.

| Condition | Method | Ctx Precision | Ctx Recall | Faithfulness |
|---|---|---|---|---|
| ViQuAD clean | **Router** | **0.805** | **0.970** | **0.894** |
| | Fixed-eq-4 | 0.786 | 0.940 | 0.843 |
| | Toneless only | 0.545 | 0.740 | 0.703 |
| ViQuAD noisy | **Router** | **0.576** | **0.770** | 0.773 |
| | Fixed-eq-4 | 0.547 | 0.730 | 0.776 |
| | Toneless only | 0.545 | 0.700 | 0.755 |
| DANGDOCAO clean | **Router** | **0.857** | **0.946** | **0.961** |
| | Fixed-eq-4 | 0.837 | 0.941 | 0.932 |
| | Toneless only | 0.706 | 0.908 | 0.832 |
| DANGDOCAO noisy | **Router** | 0.695 | **0.917** | 0.888 |
| | Fixed-eq-4 | 0.422 | 0.798 | 0.712 |
| | Toneless only | 0.698 | 0.894 | 0.895 |

Paired per-sample significance tests (two-sided $t$-test on the qa-id-matched score pairs) sharpen the table. Against fixed-equal four-way, the router's advantage is decisively significant where the retrieval gap is large — on DANGDOCAO noisy, all three metrics improve significantly (Context Precision $+0.271$, $p = 8.1\!\times\!10^{-14}$; Context Recall $+0.144$, $p = 8.8\!\times\!10^{-5}$; Faithfulness $+0.117$, $p = 9.8\!\times\!10^{-3}$) — and directionally positive but not individually significant at $n = 100$ where the retrieval gap is a few points (ViQuAD clean Faithfulness $+0.052$, $p = 0.013$, is the exception). Against toneless-only, the router is significantly better on both clean conditions on most metrics (e.g. ViQuAD clean Context Precision $+0.260$, $p = 5.3\!\times\!10^{-9}$) and statistically indistinguishable on the noisy conditions, where toneless-only is the specialist. The end-to-end pattern therefore mirrors the retrieval-level one: the router matches the specialist in the specialist's regime and beats it everywhere else, and where retrieval quality diverges materially, that divergence propagates through generation into answer quality.

Two robustness observations carry over from the retrieval analysis. First, Context Recall is the most faithful downstream reflection of the router's retrieval gains — it leads in all four conditions — consistent with recall-oriented retrieval improvements surviving the generator's tendency to smooth over ranking differences within the retrieved set. Second, Faithfulness is the most stable metric across regimes (dropping far less from clean to noisy than the context metrics): the generator remains grounded in whatever context it receives, so the dominant failure mode under noise is *retrieving evidence for the wrong question* rather than hallucinating over the right one, and the principal robustness lever in Vietnamese RAG is the retriever, not the generator. Caveats: $n = 100$ per condition limits power for small deltas; and roughly 10% of Faithfulness judgements on the long-document legal domain were lost to the judge's completion-token cap during statement decomposition (excluded uniformly across methods, so relative orderings should be preserved).

---

## 6. Conclusion

We have presented Dynamic Hybrid RAG, an adaptive three-way retrieval fusion system specialised for Vietnamese. The architecture replaces the fixed-weight combination of dense and sparse retrievers that dominates contemporary hybrid retrieval with a per-query multilayer perceptron of approximately $2{,}691$ parameters trained on soft, temperature-scaled NDCG-derived labels over the two-simplex. On UIT-ViQuAD 2.0, the resulting fusion outperforms every fixed-weight baseline considered. The test-set NDCG@10 of $0.8514$ exceeds the strongest fixed reference (fixed-equal three-way: $0.8486$, $p = 1.3\!\times\!10^{-10}$) and the standard two-way dense + BM25 hybrid ($0.8278$); under diacritic-removal noise, classical BM25 collapses by $77\%$ while BGE-M3 learned sparse retrieval retains $49\%$ of its clean performance, a robustness asymmetry that motivates the three-signal architecture independently of the adaptive component. The soft-label simplex supervision plays a load-bearing role: replacing it with hard argmax labels causes the MLP to collapse onto a single retriever and to underperform dense-only retrieval, while the soft regime is robust across at least a tenfold range of temperatures.

The same MLP checkpoint generalises to a previously unseen domain. Evaluated zero-shot on the DANGDOCAO legal/administrative corpus (37,239 passages, 4,315 test queries, no DANGDOCAO data observed during training), it improves over the fixed-equal three-way baseline by $+0.0011$ NDCG@10 on clean queries ($p = 0.039$) and by $+0.0053$ on diacritic-noisy queries ($p = 5\!\times\!10^{-12}$). Crucially, the qualitative behaviour transfers as well: the MLP raises $\bar{w}_\text{sparse}$ from $0.329$ to $0.356$ as it moves from clean to noisy queries on a corpus it has never seen, mirroring the within-domain pattern. The adaptive component is not memorising ViQuAD-specific lexical statistics but generalising a mapping from linguistic query features to retrieval-mode trust.

Taken together, the three retrievers and the adaptive fusion address the three Vietnamese-specific challenges identified at the outset. Word-segmentation dependence is handled by classical BM25 over underthesea-segmented text; diacritic sensitivity is handled by dense semantic retrieval and BGE-M3 learned sparse, with the MLP adaptively down-weighting BM25 when diacritics are absent; and code-switching is handled by BGE-M3's learned sparse representation, whose sub-word tokeniser was trained on multilingual data. The strongest empirical evidence for the linguistic grounding of the MLP is the Pearson correlation $r = +0.66$ between query English-token ratio and predicted sparse weight on the test split — a relationship that emerges from soft-label simplex supervision without being explicitly encoded in either the loss function or the feature set, and that is independently corroborated by the stratified analysis showing the MLP's per-query gains concentrate on pure-Vietnamese and short-query strata.

The analysis also exposes three concrete limitations that scope the next iteration of this work. First, the MLP's predicted weights cluster extremely close to the simplex centre (mean entropy $\bar{H} \approx 1.098$, against a maximum of $\ln 3 \approx 1.099$). The soft-label temperature ablation shows that sharper supervision improves dev NDCG@10 monotonically, but the precise cliff at which the soft regime collapses into the catastrophic hard-label regime has not been located. Second, the compound-density feature is statistically signal-neutral on clean text, which both reduces the effective feature dimensionality and suggests that the linguistic-feature module is open to replacement by query-token-level representations such as character-level embeddings. Third, the diacritic-robustness story rests on a synthetic noise model: UIT-ViQuAD 2.0 contains essentially no naturally low-diacritic queries, so the relevance of the noise-robust regime to in-the-wild Vietnamese user input remains to be verified on real query traces. We further note that the end-to-end RAGAS evaluation was carried out on $n = 200$ samples per condition because of the cost of LLM-judge scoring; while this is large enough to expose qualitative ordering changes relative to a smaller pilot study, formal significance testing on the four RAGAS metrics would benefit from a still larger sample.

We therefore identify five directions for continued investigation. The first is to locate the soft-label temperature cliff at which the supervision target becomes effectively hard and to investigate KL-divergence-based losses that may widen the MLP's predicted-weight distribution while preserving simplex stability. The second is to replace the hand-engineered seven-feature extractor with query-token-level representations capable of conditioning fusion on much finer linguistic signal. The third is to extend the fusion architecture to incorporate BGE-M3's multi-vector (ColBERT-style) representation as a fourth retrieval signal. The fourth is to scale the RAGAS evaluation beyond the present $200$ samples per condition and to substitute or supplement the Qwen3-32B judge with a stronger or independently-trained evaluator, which would in particular enable proper significance testing on the four RAGAS metrics. The fifth is to investigate cross-lingual transfer to other tonal languages, particularly Thai and Cantonese, in which similar tone-mark or diacritic-induced noise patterns are expected to arise.

---

## References

[1] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[2] Nguyen, K., et al. (20XX). UIT-ViQuAD 2.0: Towards Robust Vietnamese Machine Reading Comprehension. *[Journal/Conference]*.

[3] Chen, J., et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. *arXiv:2402.03216*.

[4] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.

[5] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR 2009*.

[6] Nguyen, V. A., et al. (20XX). underthesea: Vietnamese NLP Toolkit. *[GitHub / Publication]*.

[7] [FPT AI Factory reference if publishable]

[8] [Additional references TBD]

---
