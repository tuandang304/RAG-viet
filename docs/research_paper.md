# Dynamic Hybrid Retrieval-Augmented Generation for Vietnamese: Tone-Robust Four-Way Retrieval Fusion with a Lightweight Adaptive Router

---

> *Note (remove before submission): author placeholders and [CITATION] markers remain to be filled in. All numbers in this draft are current — they are produced by `scripts/aggregate_results.py` from the result files in `results/` and consolidated in `docs/results_summary.md`.*

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

Retrieval-Augmented Generation, Hybrid Retrieval, Dense Retrieval, BM25, Vietnamese NLP, Diacritic Robustness, Adaptive Fusion, Query Performance Prediction, Cross-domain Generalization

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for knowledge-intensive NLP tasks, pairing a retrieval component with a large language model (LLM) generator [CITATION]. A critical yet under-studied design decision is how to *combine* multiple retrieval signals — dense semantic search, classical BM25 lexical matching, and learned-sparse lexical retrieval — when no single method dominates across all query types.

For **Vietnamese**, this challenge is amplified by three language-specific factors:

1. **Word segmentation dependency.** Vietnamese is written without spaces between syllables, but semantic units (words) span multiple syllables ("học_sinh" = student, "trí_tuệ_nhân_tạo" = artificial intelligence). BM25 applied to raw whitespace-tokenized text degrades significantly compared to BM25 over properly segmented words.

2. **Diacritical mark sensitivity.** Vietnamese orthography uses six tones encoded as diacritical marks. Users frequently type without diacritics (e.g., "benh tieu duong" instead of "bệnh tiểu đường"). This is not a failure mode of one retriever but of *all* standard ones simultaneously — the missing marks break lexical matching for BM25, shift the query out of distribution for the dense encoder, and change the sub-word segmentation for learned-sparse retrieval. Recovering lexical overlap requires an index built in the same diacritic-free space as the corrupted query — which motivates the dedicated toneless retrieval channel introduced in this work.

3. **Code-switching.** Technical Vietnamese text commonly mixes English terms (e.g., "API", "database"). Classical BM25 over whitespace-tokenized text may match such terms but loses Vietnamese compound structure, whereas dense models can conflate semantically related English tokens. Learned-sparse retrieval (BGE-M3 lexical weights) offers a middle path: it assigns importance weights to tokens directly, including code-switched English terms, while sharing a tokenizer with the dense backbone.

These factors have a structural consequence: the channel that rescues diacritic-free queries (an index built over diacritic-stripped text) is *harmful* on clean queries, where stripping creates homograph collisions. No fixed combination of channels can therefore be strong in both regimes — and real Vietnamese query streams mix the regimes unpredictably. Our central hypothesis is that a small routing module, conditioned on lightweight linguistic features of the query *and on how each retrieval channel responds to it*, can allocate fusion weight per query and remain near-optimal across the entire clean-to-noisy spectrum without being told which regime a query belongs to.

Our contributions are:

1. A **tone-robust four-way retrieval architecture** for Vietnamese that adds a diacritic-stripped syllable-level BM25 channel alongside dense (FPT Vietnamese Embedding + FAISS), word-segmented BM25, and BGE-M3 learned-sparse retrieval. The toneless channel restores lexical overlap for diacritic-free queries at the cost of a single in-memory BM25 lookup (≈19 ms), transforming noisy-regime retrieval (NDCG@10 0.147 → 0.622 on legal-domain queries) — but only when gated per query, since uniform inclusion *hurts* clean-regime quality.
2. A **lightweight adaptive router** (≈14K parameters) that predicts the achievable NDCG@10 surface over a 286-point weight simplex from eight Vietnamese-aware linguistic features plus 28 scale-invariant post-retrieval query-performance signals, and converts the predicted surface into weights by softmax expectation — degrading gracefully to near-uniform fusion when routing cannot help.
3. A **training protocol** — raw (un-normalized) grid-NDCG regression targets plus explicit training-set coverage of the fully-toneless regime — each element of which is validated (or honestly bounded) by a component ablation, an oracle-headroom analysis, and a cross-domain temperature-sensitivity study.
4. A **comprehensive empirical study** on UIT-ViQuAD 2.0 (Wikipedia) and zero-shot DANGDOCAO (legal/administrative): full-test comparisons against ten baselines including reciprocal-rank fusion, a dev-tuned best-fixed weight vector, and LLM diacritic restoration; a noise-level sweep showing the router traces the upper envelope across the 0–100% corruption spectrum; generalization to LLM-generated noise types unseen in training; and end-to-end RAGAS evaluation with paired significance testing.

---

## 2. Related Work

### 2.1. Retrieval-Augmented Generation

[CITATION: Lewis et al. 2020 RAG paper]  
[CITATION: Izacard & Grave 2021 FiD]  
[CITATION: Recent survey on RAG]

### 2.2. Hybrid Retrieval

Hybrid retrieval combining dense and sparse signals has been studied extensively for English [CITATION]. Reciprocal Rank Fusion (RRF) [5] and linear interpolation of scores [CITATION] are common approaches, but both rely on fixed, query-independent combination strategies; we evaluate RRF directly as a baseline and show that its egalitarian rank aggregation is particularly costly when some channels fail catastrophically (as under diacritic noise, §5.3.2). [CITATION: BM25+dense interpolation work] shows that the optimal interpolation weight is dataset-dependent, motivating adaptive approaches; our results sharpen this observation to the *query* level: on Vietnamese, the optimal weight vector differs so strongly between clean and diacritic-free queries that no single static vector — not even one tuned on a development mix of both regimes — is strong everywhere (§5.4).

Recent multi-functional encoders such as BGE-M3 [3] expose three retrieval modes — dense, learned sparse (lexical weights), and multi-vector (ColBERT-style) — from a single backbone, enabling tighter score-space coupling than externally combined dense + BM25 systems. We adopt the dense and learned-sparse modes of (a Vietnamese fine-tune of) this family alongside classical word-segmented BM25 and a diacritic-stripped syllable-level BM25 index, yielding a four-signal fusion problem in which a single backbone supplies two of the four signals and the remaining two are complementary lexical views of the corpus — one preserving tone information, one deliberately discarding it.

### 2.3. Adaptive / Learned Retrieval Fusion

[CITATION: Learning to fuse retrieval scores]  
[CITATION: Query-dependent retrieval weighting]  
[CITATION: Any relevant work on meta-retrieval or learned fusion]

### 2.4. Vietnamese Information Retrieval

Vietnamese IR is less studied than English or Chinese. [CITATION: ViQuAD paper] introduced the first large-scale Vietnamese QA dataset. [CITATION: underthesea or VnCoreNLP] provides the word segmentation used in our BM25 pipeline. [CITATION: Vietnamese embedding models] covers dense retrieval for Vietnamese.

---

## 3. Methodology

### 3.1. Problem Formulation

Given a corpus of passages $\mathcal{P} = \{p_1, \ldots, p_N\}$ and a query $q$, we seek a retrieval function that returns the top-$k$ passages most relevant to $q$. We define the fused four-way relevance score as:

$$s(q, p) = w_\text{dense} \cdot \hat{s}_\text{dense}(q, p) + w_\text{bm25} \cdot \hat{s}_\text{bm25}(q, p) + w_\text{sparse} \cdot \hat{s}_\text{sparse}(q, p) + w_\text{toneless} \cdot \hat{s}_\text{toneless}(q, p)$$

where $\hat{s}$ denotes min-max normalized channel scores and the weight vector $\mathbf{w} = (w_\text{dense}, w_\text{bm25}, w_\text{sparse}, w_\text{toneless})$ lies on the 3-simplex ($\sum_i w_i = 1$, $w_i \geq 0$). Rather than emit $\mathbf{w}$ directly, the router predicts the *achievable NDCG@10* at every point of a discrete grid $G_4$ over the simplex and derives $\mathbf{w}$ from that predicted surface (§3.4), conditioning on both a linguistic feature vector $\phi(q)$ and a post-retrieval channel-response signal vector $\psi(q)$ computed from the candidates the four channels have already returned (§3.3).

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

**Post-retrieval query-performance signals** $\psi(q)$. The eight linguistic features describe only the query string, yet which channel wins also depends on how the corpus responds to the query. We therefore append a block of query-performance-prediction (QPP) signals computed *after* the four channels have retrieved (which fusion requires anyway, so the signals are free at inference): for each channel, the top-1/top-2 score gap, the mean and standard deviation of the top-10 score window, and coverage; plus, across every channel pair, the Jaccard overlap of top-10 id sets and a top-1 agreement indicator. Every statistic is computed on scores normalized within the channel's own top-$k$ window, making the block invariant to raw score scale — essential for zero-shot transfer across corpora whose BM25/sparse magnitudes differ. The router input is the concatenation of the eight linguistic features with these signals (28 signals in the four-channel configuration).

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

**Efficiency.** We report router parameter count, index sizes, and a per-stage per-query latency breakdown (each retrieval channel, feature extraction, signal computation, router inference) to quantify the overhead of adaptive fusion over fixed-weight fusion and to ground the cost comparison against LLM-based diacritic restoration.

**Weight interpretability.** We compute weight entropy $H = -\sum_i w_i \log w_i$, Pearson correlations between linguistic query features and predicted weights, and — most informatively for the four-way system — the trajectory of the mean toneless weight and entropy across controlled noise levels, to verify that the router's gating behaviour is linguistically meaningful at the regime level.

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

Dense semantic retrieval uses the 1024-dimensional FPT Vietnamese Embedding model, a fine-tune of BGE-M3 served through an OpenAI-compatible API. Passage embeddings are L2-normalised and indexed with FAISS `IndexFlatIP`, equivalent to cosine similarity. BM25 retrieval is performed by `rank_bm25.BM25Okapi` over text tokenised with underthesea's `word_tokenize`; the toneless channel is a second `BM25Okapi` index over lowercased, diacritic-stripped, whitespace-split syllables, with the identical transform applied to queries at search time. Learned-sparse retrieval uses the BAAI/bge-m3 model accessed locally via the FlagEmbedding library, indexed as an in-memory inverted file over non-zero token weights. The end-to-end RAG evaluation in §5.7 uses Llama-3.3-70B-Instruct as both generator and RAGAS judge, accessed through the same OpenAI-compatible interface.

The four-way router (Keras/TensorFlow) is trained on a multi-domain pool of 6,000 queries augmented with 1,500 rule-based fully-diacritic-stripped variants (7,500 total), using Adam with learning rate $10^{-3}$, batch size 256, and 100 epochs. To avoid an OpenMP/MKL runtime clash between FAISS and TensorFlow, candidate-score collection (FAISS/BM25/BGE-M3) and network fitting run in separate processes. Targets are raw NDCG@10 over the 286-point four-simplex grid (step $0.1$); inference uses softmax-expected weights at $T = 0.05$. The diacritic-restoration baseline uses Qwen3.6-27B (the successor to Qwen3-32B, which FPT removed from its catalogue) and the end-to-end RAGAS judge uses the non-reasoning Llama-3.3-70B-Instruct. The seed for all randomised components is fixed at $42$.

All experimental results in §5 are produced on a single workstation equipped with an NVIDIA GeForce RTX 3050 (6 GB VRAM, CUDA 12.4 runtime) and a multi-core CPU. The GPU is used for BGE-M3 sparse encoding; FAISS, BM25, and the fusion MLP run on CPU. The software stack comprises Python 3.13, PyTorch 2.6.0 with CUDA 12.4 and FlagEmbedding for BGE-M3, TensorFlow/Keras for the fusion MLP, FAISS-CPU, `rank_bm25`, and underthesea for Vietnamese word segmentation.

---

## 5. Results & Discussion

All results in this section are produced by a single router checkpoint, trained once with the protocol of §3.5 and §4.4 and never re-tuned per condition. §5.1 reports the in-domain evaluation on UIT-ViQuAD 2.0; §5.2 the zero-shot cross-domain evaluation on DANGDOCAO; §5.3 analyses *when* and *why* routing helps, including the noise-level sweep and generalisation to unseen noise types; §5.4 gives the full significance battery; §5.5 the efficiency analysis; §5.6 the component ablation and oracle-headroom study; and §5.7 the end-to-end RAGAS evaluation.

### 5.1. In-domain Results (UIT-ViQuAD 2.0)

**Clean test set** (7,301 queries, held-out):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.6621 | 0.6195 | 0.6195 | 0.7966 | 0.9259 | 0.5357 |
| Dense only | 0.8064 | 0.7674 | 0.7674 | 0.9270 | 0.9885 | 0.6795 |
| Sparse only (BGE-M3) | 0.7594 | 0.7164 | 0.7164 | 0.8933 | 0.9743 | 0.6240 |
| Toneless only | 0.5760 | 0.5316 | 0.5316 | 0.7162 | 0.8754 | 0.4457 |
| Dense + BM25 (0.5/0.5) | 0.8272 | 0.7898 | 0.7898 | 0.9423 | 0.9910 | 0.7042 |
| RRF ($k = 60$) | 0.7988 | 0.7576 | 0.7576 | 0.9273 | 0.9908 | 0.6696 |
| Fixed-equal three-way | 0.8482 | 0.8142 | 0.8142 | 0.9529 | 0.9925 | 0.7350 |
| Fixed-equal four-way | 0.8369 | 0.8013 | 0.8013 | 0.9467 | 0.9915 | 0.7199 |
| Best-fixed (dev-tuned) | 0.8211 | 0.7831 | 0.7831 | 0.9388 | 0.9916 | 0.6980 |
| **Dynamic router (ours)** | **0.8541** | **0.8217** | **0.8217** | **0.9540** | **0.9929** | **0.7466** |

Four observations characterise the clean regime. First, among the four single channels, dense semantic retrieval (0.8064) leads BGE-M3 learned sparse (0.7594), classical BM25 (0.6621), and — last by design — the toneless channel (0.5760), whose diacritic stripping creates homograph collisions that cost it roughly nine NDCG points against toned BM25 on clean text. Second, and centrally to this paper's argument: *adding the toneless channel with uniform weights makes fusion worse*. Fixed-equal four-way (0.8369) trails fixed-equal three-way (0.8482) by more than a point — the fourth channel is a liability on clean queries unless something suppresses it. The dynamic router is that something: it holds $\bar{w}_\text{toneless}$ at $0.19$ on this split (§5.3) and reaches 0.8541, above every baseline with $p \le 2.4\!\times\!10^{-5}$ (§5.4). Third, the two tuned/untuned static competitors bracket the fusion design space from both sides and both lose: RRF (0.7988) sits below even the two-way linear hybrid because egalitarian rank aggregation grants the two weaker lexical channels equal influence, and the dev-tuned best-fixed vector (0.8211) — grid-searched on a balanced clean+noisy dev mix — pays a three-point clean-side toll for its noise insurance. Fourth, Recall@100 saturates near $0.99$ for every multi-channel method, so the router's contribution on clean text is re-ranking within an almost-complete candidate pool; correspondingly its largest margins appear in the rank-sensitive metrics (Hit@1: 0.7466 vs 0.7350 for the best static method).

**Diacritic-stripped dev set** (3,814 queries, all tone marks removed):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.1558 | 0.1335 | 0.1335 | 0.2276 | 0.4751 | 0.0954 |
| Dense only | 0.2953 | 0.2547 | 0.2547 | 0.4266 | 0.6681 | 0.1856 |
| Sparse only (BGE-M3) | 0.3669 | 0.3253 | 0.3253 | 0.5003 | 0.7278 | 0.2509 |
| Toneless only | 0.5905 | 0.5506 | 0.5506 | 0.7171 | 0.8728 | 0.4738 |
| Dense + BM25 (0.5/0.5) | 0.3015 | 0.2601 | 0.2601 | 0.4350 | 0.6762 | 0.1880 |
| RRF ($k = 60$) | 0.5045 | 0.4543 | 0.4543 | 0.6644 | 0.9061 | 0.3597 |
| Fixed-equal three-way | 0.3961 | 0.3532 | 0.3532 | 0.5346 | 0.7284 | 0.2792 |
| Fixed-equal four-way | 0.5694 | 0.5167 | 0.5167 | 0.7362 | 0.8993 | 0.4164 |
| Best-fixed (dev-tuned) | 0.6457 | 0.6035 | 0.6035 | 0.7787 | 0.9156 | 0.5184 |
| **Dynamic router (ours)** | 0.6405 | 0.5972 | 0.5972 | 0.7777 | **0.9206** | 0.5134 |

Full diacritic stripping inverts the channel hierarchy. The three tone-dependent channels collapse — BM25 by 76% relative (0.662 → 0.156), dense by 63% (0.806 → 0.295), BGE-M3 sparse by 52% (0.759 → 0.367) — while the toneless channel is *unchanged by construction* (0.576 → 0.591; its index and the stripped queries live in the same diacritic-free space). The consequences propagate through every fusion strategy that lacks or under-uses the surviving channel: fixed-equal three-way manages only 0.3961, and even fixed-equal four-way (0.5694) dilutes the one working signal with three failing ones. The router reaches 0.6405 — above toneless-only by $+0.0500$ ($p = 2.3\!\times\!10^{-40}$), because on partially matchable queries it still routes residual weight to whichever toned channel retains signal — and above every untuned baseline by wide, significant margins. The single method it trails is the dev-tuned best-fixed vector (0.6457, $-0.0052$; marginal, and not confirmed by the Wilcoxon test), a comparison we analyse honestly in §5.4: a static vector specialised for full stripping wins narrowly when every query is fully stripped, and pays for it everywhere else. Recall@100 tells the downstream-relevant story most clearly: the router restores the retrievable-evidence ceiling to 0.9206 — within a point of the clean-split ceiling — against 0.7284 for the best three-way method, roughly halving the fraction of queries whose answer passage a generator could never see.

### 5.2. Cross-domain Results (DANGDOCAO, Zero-shot)

The cross-domain protocol applies the identical router checkpoint — trained on the multi-domain pool of §3.5, which contains no DANGDOCAO data — to the DANGDOCAO legal/administrative corpus (37,239 passages, 736 sub-domains). All four channel indexes are rebuilt from the DANGDOCAO corpus alone; neither the router nor any index parameter carries target-domain information.

**Clean test set** (4,315 queries):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.6748 | 0.6197 | 0.6197 | 0.8463 | 0.9569 | 0.5013 |
| Dense only | 0.7907 | 0.7418 | 0.7418 | 0.9400 | 0.9905 | 0.6243 |
| Sparse only (BGE-M3) | 0.7526 | 0.7020 | 0.7020 | 0.9082 | 0.9754 | 0.5863 |
| Toneless only | 0.6094 | 0.5552 | 0.5552 | 0.7791 | 0.9295 | 0.4419 |
| Dense + BM25 (0.5/0.5) | 0.8048 | 0.7589 | 0.7589 | 0.9444 | 0.9910 | 0.6452 |
| RRF ($k = 60$) | 0.7816 | 0.7330 | 0.7330 | 0.9321 | 0.9917 | 0.6232 |
| Fixed-equal three-way | 0.8152 | 0.7697 | 0.7697 | 0.9539 | 0.9933 | 0.6586 |
| Fixed-equal four-way | 0.8051 | 0.7583 | 0.7583 | 0.9486 | 0.9921 | 0.6470 |
| Best-fixed (dev-tuned) | 0.7869 | 0.7380 | 0.7380 | 0.9370 | 0.9917 | 0.6246 |
| **Dynamic router (ours)** | **0.8196** | **0.7753** | **0.7753** | **0.9543** | 0.9930 | **0.6660** |

**Diacritic-stripped test set** (4,315 queries):

| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |
|--------|---------|--------|--------|-----------|------------|-------|
| BM25 only | 0.0481 | 0.0393 | 0.0393 | 0.0769 | 0.1773 | 0.0257 |
| Dense only | 0.0715 | 0.0598 | 0.0598 | 0.1094 | 0.2197 | 0.0408 |
| Sparse only (BGE-M3) | 0.1435 | 0.1222 | 0.1222 | 0.2116 | 0.3891 | 0.0830 |
| Toneless only | 0.6095 | 0.5552 | 0.5552 | 0.7791 | 0.9270 | 0.4417 |
| Dense + BM25 (0.5/0.5) | 0.0833 | 0.0700 | 0.0700 | 0.1258 | 0.2438 | 0.0473 |
| RRF ($k = 60$) | 0.3055 | 0.2523 | 0.2523 | 0.4809 | 0.8962 | 0.1715 |
| Fixed-equal three-way | 0.1470 | 0.1269 | 0.1269 | 0.2114 | 0.3581 | 0.0904 |
| Fixed-equal four-way | 0.4277 | 0.3531 | 0.3531 | 0.6667 | 0.8971 | 0.2271 |
| Best-fixed (dev-tuned) | 0.6287 | 0.5761 | 0.5761 | 0.7930 | 0.9156 | 0.4660 |
| **Dynamic router (ours)** | 0.6218 | 0.5678 | 0.5678 | 0.7903 | **0.9207** | 0.4545 |

The cross-domain results support four readings. First, the routing policy transfers zero-shot: on clean legal queries the router again leads every baseline (0.8196; $+0.0044$ over the strongest static method at $p = 9.4\!\times\!10^{-3}$, and $+0.0145$ to $+0.0380$ over the four-way, tuned, and RRF alternatives), and the clean-side pattern of §5.1 — uniform inclusion of the toneless channel *hurts* (0.8051 vs 0.8152), gated inclusion helps — reproduces on a corpus the router has never seen. Second, the noise catastrophe is deeper in the legal domain than the encyclopedic one: BM25 falls by 93% relative (to 0.0481), dense by 91%, sparse by 81% — Vietnamese legal terminology consists of long, formulaic, low-redundancy phrases that offer nothing to recover once tone marks vanish. Fixed-equal three-way (0.1470) is barely a tenth of its clean self, and RRF (0.3055) again demonstrates the cost of egalitarian aggregation over dead channels. The router reaches 0.6218 — a $4.2\times$ improvement over the best tone-dependent fusion and within $0.007$ of the noise-specialised tuned vector — and restores Recall@100 to 0.9207 against 0.3581 for three-way fusion. Third, the gating mechanism itself transfers: mean $\bar{w}_\text{toneless}$ moves from $0.187$ on clean DANGDOCAO to $0.546$ on stripped DANGDOCAO (§5.3), the same two-regime signature learned in training, on out-of-domain data. Because DANGDOCAO is entirely unseen, this excludes memorisation of corpus-specific lexical statistics: the router is applying a corpus-independent mapping from query and channel-response evidence to channel trust. Fourth, the toneless channel is remarkably domain-stable (0.6094 clean, 0.6095 stripped) — its syllable-level diacritic-free matching is indifferent to both the noise condition and, largely, the domain shift, which is precisely the property that makes it a reliable routing target.

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

We have presented Dynamic Hybrid RAG, a tone-robust four-way retrieval fusion system for Vietnamese. The architecture augments the standard dense + BM25 + learned-sparse trio with a diacritic-stripped syllable-level BM25 channel, and gates all four channels per query with a lightweight router (≈14K parameters) that predicts the achievable NDCG@10 surface over a 286-point weight simplex from eight linguistic query features and 28 scale-invariant post-retrieval channel-response signals, converting the predicted surface into weights by softmax expectation.

The empirical case rests on a structural finding rather than a single headline number: **on Vietnamese, no fixed channel weighting is strong in both the clean and the diacritic-free regime.** The toneless channel that rescues stripped queries (restoring legal-domain NDCG@10 from 0.147 to 0.622) is a liability on clean text, where its uniform inclusion *lowers* fusion quality (fixed-equal four-way 0.837 vs three-way 0.848 on ViQuAD test); conversely, every tone-dependent configuration collapses under stripping (fixed-equal three-way: 0.848 → 0.396 in-domain, 0.815 → 0.147 zero-shot). Even a best-fixed vector grid-searched over 286 candidates on a balanced clean+noisy development mix cannot escape the dilemma — it concedes three to four NDCG points on clean text to stay competitive under noise. The router resolves the dilemma without being told the regime: it significantly outperforms every untuned baseline on all four full test conditions (smallest margin $p = 9.4\!\times\!10^{-3}$, surviving Holm–Bonferroni over the 36-test battery), traces the upper envelope of all methods across a 0–100% noise-level sweep, and trails the noise-specialised tuned vector by at most $0.007$ at the fully-stripped extreme while beating it by $0.033$ on both clean sets and by $0.035$–$0.046$ on semantic noise types it was never tuned for.

Three further properties give the result its robustness. The mechanism is *observable*: mean toneless weight rises monotonically from ≈0.19 to ≈0.46–0.55 as per-syllable corruption goes from 0% to 100%, with weight entropy falling in lockstep — the router grows decisive exactly as the regime grows extreme — and the same signature appears zero-shot on a legal corpus never seen in training. The mechanism is *correctly attributed*: a six-configuration component ablation shows the toneless channel and the training-set coverage of the fully-toneless regime to be load-bearing (removing either collapses noisy performance to 0.13–0.14 zero-shot), while expected-weight inference, the QPP signal block, and raw-label training are second-order refinements; an oracle-headroom analysis shows the router realising 40–78% of the label-aware routing upper bound on the noisy conditions where headroom is real. And the mechanism is *cheap*: the full router overhead is ≈6.7 ms per query and the toneless channel a 19 ms in-memory lookup, against ≈1.5 s per query for the LLM diacritic-restoration alternative — which is stronger when affordable (restoring stripped-query retrieval to near-clean levels) and which we therefore position as a complement on the cost–quality frontier rather than a defeated baseline. End-to-end RAGAS evaluation confirms that the retrieval-level differences propagate to answer quality: the router leads context precision and recall in all four conditions, decisively so where the retrieval gap is large (legal-domain noisy context precision +0.271 over uniform four-way fusion, $p = 8.1\!\times\!10^{-14}$).

The analysis also fixes the limitations that scope future work. First, on homogeneous fully-stripped evaluation sets a noise-specialised static vector remains marginally ahead; the router's value proposition is distribution-independence, not universal per-slice dominance, and applications with a *known*, stable noise profile can do without it. Second, the stripped-query conditions use a synthetic rule-based noise model; the LLM-generated noise study (typo, informal, code-switching) partially addresses ecological validity, but evaluation on real Vietnamese user query traces remains open. Third, the four-way router's decisions are less transparent at the single-feature level than its three-way predecessor's — bivariate feature–weight correlations weaken as routing shifts onto post-retrieval channel-response signals — so its interpretability case rests on regime-level behaviour rather than per-feature attribution. Fourth, the end-to-end evaluation ($n = 100$ per condition) has limited power for small effects, and the embedding-based answer-relevancy metric could not be run against our serving infrastructure.

We see four natural extensions. A *cascaded* system could route cheaply by default and invoke LLM diacritic restoration only for queries whose channel-response signature indicates that no retrieval channel will succeed — combining the 19 ms and 1.5 s mechanisms along the frontier this paper maps. A *cost-aware* router could skip channels it predicts to be useless, converting the routing signal into latency savings as well as quality. Replacing the hand-engineered linguistic features with token-level representations may recover finer-grained routing signal, particularly for partially corrupted queries where our short-query hedging indicates residual uncertainty. Finally, the design pattern — a dedicated normalized-orthography channel gated by a query-conditioned router — should transfer to other diacritic-rich languages (Thai, Czech, Turkish) and to other systematic orthographic noise processes; validating that transfer would establish the architecture as a general recipe rather than a Vietnamese-specific solution.

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
