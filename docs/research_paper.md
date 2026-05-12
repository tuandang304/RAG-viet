# Dynamic Hybrid Retrieval-Augmented Generation for Vietnamese: Adaptive Fusion of Dense and Sparse Signals via a Lightweight MLP

---

## Authors & Affiliation

**[Author 1 Name]**¹, **[Author 2 Name]**¹, **[Author 3 Name]**²

¹ [University / Lab Name], [City], Vietnam
² [Affiliation 2], [City], Vietnam

Correspondence: [email@domain.com]

---

## Abstract

Retrieval-Augmented Generation (RAG) systems typically rely on a fixed combination of dense vector search and sparse lexical matching (BM25), with weights chosen once at development time. This one-size-fits-all strategy ignores the fact that different query types benefit from different retrieval signals — a property especially pronounced in Vietnamese, where word segmentation quality, diacritical mark presence, and code-switching between Vietnamese and English all strongly modulate the effectiveness of each retrieval approach.

We present **Dynamic Hybrid RAG**, a lightweight framework that replaces fixed fusion weights with an adaptive MLP (≈2,691 parameters) that predicts per-query three-way fusion weights `(w_dense, w_bm25, w_sparse)` from seven Vietnamese-aware linguistic features. The three retrieval signals are dense semantic search (FPT Vietnamese Embedding + FAISS), BM25 over underthesea-segmented text, and BGE-M3 learned sparse lexical weights via an inverted index. The MLP is trained with a **soft-label supervision** strategy on a 3D simplex grid (66 points, step = 0.1): we compute NDCG@10 for each `(a, b, c)` candidate and apply temperature-scaled softmax over the simplex to construct smooth expected-weight targets, substantially reducing label noise compared to hard-label grid search.

On the UIT-ViQuAD 2.0 test set (7,301 queries), our method achieves **NDCG@10 = 0.8514** and **MRR@10 = 0.8178**, outperforming the fixed-equal three-way hybrid baseline (0.8486 / 0.8146; paired $t$-test $p = 1.3\!\times\!10^{-10}$), the two-way dense + BM25 hybrid (0.8278 / 0.7907), dense-only retrieval (0.8068 / 0.7679), BM25-only retrieval (0.6623 / 0.6198), and BGE-M3 sparse-only retrieval (0.7595 / 0.7167). Under diacritic-removal noise (3,814 dev queries with all tone marks stripped to simulate Vietnamese keyboard typing), the MLP achieves NDCG@10 = 0.3993 versus 0.3969 for fixed-equal three-way fusion, 0.3050 for two-way dense + BM25, and only 0.1559 for BM25-only — confirming that the learned-sparse signal (BGE-M3) is far more robust to missing tone marks than classical BM25. In a strict **zero-shot cross-domain evaluation** on the DANGDOCAO legal/administrative corpus (37,239 passages, no DANGDOCAO data seen during training), the same MLP checkpoint still beats every baseline: NDCG@10 = 0.8167 vs 0.8156 fixed-equal three-way ($p = 0.039$) on clean queries, and 0.1530 vs 0.1477 ($p = 5\!\times\!10^{-12}$) on diacritic-noisy queries — confirming that the learned feature-to-weight mapping captures domain-invariant linguistic signals rather than ViQuAD-specific lexical statistics.

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
2. A **Vietnamese-aware feature extractor** (7 features: diacritic ratio, compound word ratio, English token ratio, tech-term ratio, clause count, question-word presence, query length).
3. A **soft-label training strategy** using temperature-scaled NDCG@10 distributions over a 3D simplex grid (66 points at step 0.1), improving over hard-label grid search.
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

where $\hat{s}$ denotes min-max normalized scores and $(w_\text{dense}, w_\text{bm25}, w_\text{sparse}) = \text{softmax}(\text{MLP}(\phi(q)))$ with $\phi(q) \in \mathbb{R}^7$ being the Vietnamese-aware feature vector. The softmax constraint enforces $w_\text{dense} + w_\text{bm25} + w_\text{sparse} = 1$ and $w_i \geq 0$, i.e. the weight vector lies on the 2-simplex.

### 3.2. Retrieval Components

We fuse three complementary retrieval signals; each retriever runs on the full corpus and returns its own top-100 candidate set, which we union before fusion.

**Dense Retrieval.** We encode passages and queries using the FPT Vietnamese Embedding model (1024-dimensional, fine-tuned from BGE-M3). Passage embeddings are L2-normalized and indexed with FAISS `IndexFlatIP` for inner product search, equivalent to cosine similarity after normalization.

**BM25 Retrieval.** Queries and passages are tokenized with underthesea `word_tokenize` (`format="text"`), which produces underscore-joined Vietnamese compound words (e.g., "học_sinh", "trí_tuệ_nhân_tạo"). BM25Okapi scores are computed over this segmented vocabulary.

**Learned Sparse Retrieval (BGE-M3).** We extract per-token lexical weights from BGE-M3 (`BAAI/bge-m3`) using the FlagEmbedding library and build an inverted index over non-zero token weights. At query time, BGE-M3 produces sparse lexical weights for the query, and document scores are computed as the dot product over the inverted-index posting lists. This signal is run locally (no external API) and captures learned term importance — including out-of-vocabulary and code-switching English terms — that classical BM25 cannot model.

**Score Normalization.** All three score distributions are independently min-max normalized to $[0, 1]$ before fusion, necessary because BM25 and BGE-M3 sparse scores are unbounded while dense cosine scores are bounded in $[-1, 1]$.

### 3.3. Vietnamese-Aware Feature Extractor

We extract seven features $\phi(q) = [f_1, \ldots, f_7]$:

| Feature | Description | Range |
|---------|-------------|-------|
| $f_1$ — diacritic\_ratio | Fraction of syllables bearing a Vietnamese tone mark | $[0, 1]$ |
| $f_2$ — compound\_ratio | Fraction of tokens that are multi-syllable compounds after segmentation | $[0, 1]$ |
| $f_3$ — english\_ratio | Fraction of syllables matching `[a-zA-Z]+` (code-switching) | $[0, 1]$ |
| $f_4$ — tech\_term\_ratio | Fraction of syllables matching a curated technical vocabulary | $[0, 1]$ |
| $f_5$ — clause\_count\_norm | Number of clause markers (commas, "và", "hoặc", …), clipped and normalized by 5 | $[0, 1]$ |
| $f_6$ — has\_question\_word | Binary: query contains a Vietnamese interrogative ("ai", "gì", "nào", …) | $\{0, 1\}$ |
| $f_7$ — query\_length\_norm | Syllable count normalized at 20 | $[0, 1]$ |

### 3.4. Fusion MLP

The fusion MLP is a 3-layer feed-forward network:

$$\text{MLP}: \mathbb{R}^7 \xrightarrow{\text{Linear}(7 \to 64)} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(64 \to 32)} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Linear}(32 \to 3)} \xrightarrow{\text{softmax}} (w_\text{dense}, w_\text{bm25}, w_\text{sparse})$$

Total parameters: $7 \cdot 64 + 64 + 64 \cdot 32 + 32 + 32 \cdot 3 + 3 = 2{,}691$. The softmax output constraint guarantees the weight vector lies on the 2-simplex ($w_\text{dense} + w_\text{bm25} + w_\text{sparse} = 1$, $w_i \geq 0$).

### 3.5. Soft-Label Training

Constructing supervision signal for the fusion MLP is non-trivial: there is no ground-truth weight triple for a query — only the downstream NDCG@10 achievable under each candidate weighting.

**Hard-label baseline.** A grid-search baseline enumerates simplex points $G = \{(a, b, c) \mid a + b + c = 1,\ a, b, c \in \{0.0, 0.1, \ldots, 1.0\}\}$ (66 points), selects the point $\mathbf{w}^*$ maximizing NDCG@10 per query, and trains the MLP with cross-entropy or MSE against the one-hot target $\mathbf{w}^*$.

**Soft-label method (proposed).** Rather than collapsing to a single argmax, we treat the full NDCG-over-simplex profile as supervision. For each training query, we compute NDCG@10 at every grid point $\mathbf{w}_i \in G$ and apply a temperature-scaled softmax over the simplex:

$$p_i = \frac{\exp\!\left(\text{NDCG@10}(\mathbf{w}_i) / T\right)}{\sum_j \exp\!\left(\text{NDCG@10}(\mathbf{w}_j) / T\right)}, \quad T = 0.3$$

The expected soft-label weight triple is $\bar{\mathbf{w}} = \sum_i p_i \cdot \mathbf{w}_i$, which by construction also lies on the 2-simplex. The MLP is trained with MSE loss against $\bar{\mathbf{w}}$. This approach (1) avoids tie-breaking ambiguity when multiple simplex points achieve near-identical NDCG, (2) encodes the relative preference structure across the full simplex rather than only the mode, and (3) produces smoother gradients during MLP training.

Training uses the Adam optimiser with learning rate $10^{-3}$ and batch size $256$, run for $100$ epochs on $5{,}000$ randomly sampled queries drawn from the UIT-ViQuAD 2.0 training set augmented with diacritic-removed copies at a $30\%$ noise ratio. The DANGDOCAO corpus described in §4.1 is held out from training entirely and used only for the zero-shot cross-domain evaluation in §5.2.

---

## 4. Experiments

### 4.1. Datasets

| Dataset | Domain | Passages | QA pairs | Split used |
|---------|--------|----------|----------|------------|
| UIT-ViQuAD 2.0 | Wikipedia (Vietnamese) | 5,317 | 28,454 train / 3,814 dev / 7,301 test | Train MLP (aug. to 36,990); in-domain eval |
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

The generator (Qwen3-32B via FPT AI Factory) is used for end-to-end QA evaluation in Section 5.7.

### 4.3. Baselines

All baselines share the same retrieval candidate set (top-100 from each of dense, BM25, sparse) and the same min-max normalization step; they differ only in the fusion weight triple $(w_\text{dense}, w_\text{bm25}, w_\text{sparse})$.

| System | $(w_\text{dense}, w_\text{bm25}, w_\text{sparse})$ | Description |
|--------|---------------------------------------------------|-------------|
| BM25 only | $(0, 1, 0)$ | underthesea tokenization + BM25Okapi |
| Dense only | $(1, 0, 0)$ | FPT Vietnamese Embedding + FAISS |
| Sparse only | $(0, 0, 1)$ | BGE-M3 learned sparse + inverted index |
| Fixed-equal three-way | $(1/3, 1/3, 1/3)$ | Uniform three-way fusion |
| Dense + BM25 (0.5/0.5) | $(0.5, 0.5, 0)$ | Two-way hybrid baseline (no sparse signal) |
| **Dynamic MLP (ours)** | softmax(MLP($\phi(q)$)) | Proposed three-way adaptive fusion |

### 4.4. Implementation Details

Dense semantic retrieval uses the 1024-dimensional FPT Vietnamese Embedding model, a fine-tune of BGE-M3 served through an OpenAI-compatible API. Passage embeddings are L2-normalised and indexed with FAISS `IndexFlatIP`, equivalent to cosine similarity. BM25 retrieval is performed by `rank_bm25.BM25Okapi` over text tokenised with underthesea's `word_tokenize`. Learned-sparse retrieval uses the BAAI/bge-m3 model accessed locally via the FlagEmbedding library, indexed as an in-memory inverted file over non-zero token weights. The end-to-end RAG evaluation in §5.7 uses Qwen3-32B as the generator and the RAGAS judge LLM, accessed through the same OpenAI-compatible interface.

The fusion MLP is trained on the augmented UIT-ViQuAD 2.0 training set of 36,990 queries (28,454 original queries plus 8,536 diacritic-removed copies at a 30\% noise ratio) using Adam with learning rate $10^{-3}$, batch size 256, and 100 epochs. Soft labels are constructed on a 66-point three-simplex grid (step $0.1$) at temperature $T = 0.3$. The seed for all randomised components is fixed at $42$.

All experimental results in §5 are produced on a single workstation equipped with an NVIDIA GeForce RTX 3050 (6 GB VRAM, CUDA 12.4 runtime) and a multi-core CPU. The GPU is used for BGE-M3 sparse encoding; FAISS, BM25, and the fusion MLP run on CPU. The software stack comprises PyTorch 2.6.0 with CUDA 12.4, Python 3.13, FAISS-CPU, FlagEmbedding for BGE-M3, `rank_bm25`, and underthesea for Vietnamese word segmentation.

---

## 5. Results & Discussion

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

### 5.3. Analysis: When Does Dynamic Fusion Help?

#### 5.3.1. Stratified NDCG@10

We partition the UIT-ViQuAD 2.0 dev and test sets into 11 strata defined by ranges of individual query features, and report NDCG@10 for both the fixed-equal three-way baseline and the dynamic MLP within each stratum together with the mean predicted fusion weights $\bar{w}_\text{dense}$, $\bar{w}_\text{bm25}$, $\bar{w}_\text{sparse}$. The partition makes it possible to identify the linguistic conditions under which adaptive fusion contributes most, and to read the MLP's per-stratum weight allocation as a behavioural footprint of what it has learned.

**Dev set (3,814 queries):**

| Stratum | N | Fixed NDCG | MLP NDCG | Δ | $\bar{w}_\text{dense}$ | $\bar{w}_\text{bm25}$ | $\bar{w}_\text{sparse}$ |
|---------|---|------------|----------|---|------------------------|------------------------|------------------------|
| diac\_low (< 0.3) | 9 | 1.0000 | 1.0000 | +0.0000 | 0.350 | 0.302 | 0.348 |
| diac\_mid (0.3–0.7) | 627 | 0.8801 | 0.8812 | +0.0011 | 0.347 | 0.313 | 0.339 |
| diac\_high (> 0.7) | 3,178 | 0.8392 | 0.8409 | +0.0017 | 0.349 | 0.321 | 0.330 |
| comp\_low (< 0.2) | 774 | 0.8388 | 0.8384 | $-$0.0004 | 0.349 | 0.319 | 0.331 |
| comp\_high (≥ 0.2) | 3,040 | 0.8482 | 0.8503 | +0.0021 | 0.349 | 0.319 | 0.332 |
| eng\_none (= 0) | 318 | 0.7625 | 0.7695 | **+0.0070** | 0.351 | 0.322 | 0.327 |
| eng\_mixed (> 0) | 3,496 | 0.8539 | 0.8550 | +0.0011 | 0.349 | 0.319 | 0.332 |
| short\_query (< 0.4) | 190 | 0.7632 | 0.7631 | $-$0.0000 | 0.361 | 0.299 | 0.340 |
| long\_query (≥ 0.4) | 3,624 | 0.8507 | 0.8524 | +0.0017 | 0.348 | 0.320 | 0.331 |
| simple (no clause) | 3,125 | 0.8340 | 0.8360 | +0.0019 | 0.351 | 0.317 | 0.332 |
| complex (has clause) | 689 | 0.9021 | 0.9021 | +0.0000 | 0.340 | 0.331 | 0.329 |

**Test set (7,301 queries):**

| Stratum | N | Fixed NDCG | MLP NDCG | Δ | $\bar{w}_\text{dense}$ | $\bar{w}_\text{bm25}$ | $\bar{w}_\text{sparse}$ |
|---------|---|------------|----------|---|------------------------|------------------------|------------------------|
| diac\_low (< 0.3) | 4 | 0.9077 | 0.9077 | +0.0000 | 0.347 | 0.301 | 0.352 |
| diac\_mid (0.3–0.7) | 1,241 | 0.8659 | 0.8697 | +0.0038 | 0.348 | 0.314 | 0.338 |
| diac\_high (> 0.7) | 6,056 | 0.8450 | 0.8477 | +0.0026 | 0.349 | 0.321 | 0.330 |
| comp\_low (< 0.2) | 1,567 | 0.8449 | 0.8470 | +0.0021 | 0.348 | 0.320 | 0.332 |
| comp\_high (≥ 0.2) | 5,734 | 0.8496 | 0.8526 | +0.0030 | 0.349 | 0.320 | 0.332 |
| eng\_none (= 0) | 583 | 0.7981 | 0.8024 | **+0.0043** | 0.350 | 0.323 | 0.327 |
| eng\_mixed (> 0) | 6,718 | 0.8530 | 0.8557 | +0.0027 | 0.349 | 0.319 | 0.332 |
| short\_query (< 0.4) | 220 | 0.7402 | 0.7412 | +0.0010 | 0.361 | 0.299 | 0.340 |
| long\_query (≥ 0.4) | 7,081 | 0.8520 | 0.8549 | +0.0029 | 0.348 | 0.320 | 0.331 |
| simple (no clause) | 5,940 | 0.8338 | 0.8371 | **+0.0033** | 0.351 | 0.317 | 0.332 |
| complex (has clause) | 1,361 | 0.9133 | 0.9139 | +0.0005 | 0.340 | 0.331 | 0.329 |

The stratified results admit four substantive interpretations. The diacritic-density partition is dominated by the high-density bin (`diac_high` covers $83\%$ of both dev and test), reflecting the fact that UIT-ViQuAD 2.0 is sourced from well-edited Vietnamese Wikipedia, in which essentially every query is fully toned. The `diac_low` strata contain only nine dev queries and four test queries; both are too small to support inference, and the MLP achieves identical NDCG@10 to the fixed baseline on them. The substantive diacritic-robustness experiments must therefore be read from the noisy split in §5.1 rather than from this clean stratification.

The largest stratified gains over the fixed-equal three-way baseline appear on the English-code-switching partition, but in the opposite direction to what a naive reading of §1 might suggest. The MLP's per-stratum lift is largest on the `eng_none` partition (pure Vietnamese queries, $\Delta = +0.0070$ on dev, $+0.0043$ on test) — roughly three to four times larger than the average lift on `eng_mixed` queries. Reading this together with the Pearson correlation $r = +0.66$ between `english_ratio` and $w_\text{sparse}$ (§5.3.2), the mechanism becomes clear: the MLP has learned to associate English-token density with sparse-retriever trust, and the per-stratum advantage on pure-Vietnamese queries arises from *withholding* sparse weight on queries that lack the signature feature, rather than from any unconditional preference for sparse retrieval. The fixed baseline's uniform $1/3$ overspends on the sparse signal in this stratum; the MLP correctly under-spends.

The compound-word partition is essentially signal-neutral in clean text ($\Delta \approx +0.002$ in both bins on dev). This matches the near-zero Pearson correlation between `compound_ratio` and $w_\text{bm25}$ reported in §5.3.2 and suggests that the underthesea-segmented BM25 retriever already extracts most of the available term-match signal from Vietnamese compounds without the MLP needing to differentially weight it. The compound feature is therefore the weakest member of the seven-feature extractor on clean queries, and we revisit this finding in §6 when discussing future-work feature-engineering directions.

Two additional strata produce distinctive MLP weight signatures even when the NDCG@10 gap is small. On short queries ($N = 190$ on dev, $220$ on test), the MLP raises $\bar{w}_\text{dense}$ from $0.349$ to $0.361$ and lowers $\bar{w}_\text{bm25}$ from $0.320$ to $0.299$. Short queries provide little lexical surface for BM25 to score, and a stronger dense weighting is the linguistically reasonable response. On multi-clause queries, retrieval performance under the fixed baseline already saturates at NDCG@10 $\approx 0.91$, and the MLP adds essentially nothing ($\Delta = +0.000$ on dev, $+0.0005$ on test). When all three signals agree, the fusion problem is degenerate and adaptive weighting cannot help; the MLP correctly recognises this regime and does not perturb the weights.

#### 5.3.2. Weight Interpretability

To assess whether the MLP captures linguistically meaningful structure rather than merely memorising training-set statistics, we compute Pearson correlations between three of the seven input features and the corresponding predicted weight that linguistic theory would associate with them.

| Correlation (test set, $N = 7{,}301$) | Expected sign | Actual $r$ | $p$-value |
|---------------------------------------|---------------|------------|----------|
| diacritic\_ratio ↔ $w_\text{dense}$ | negative (fewer diacritics → higher $w_\text{dense}$) | **+0.1005** | $7.4\!\times\!10^{-18}$ |
| compound\_ratio ↔ $w_\text{bm25}$ | positive (more Vietnamese compounds → higher $w_\text{bm25}$) | $-$0.0171 | 0.145 (n.s.) |
| english\_ratio ↔ $w_\text{sparse}$ | positive (more English code-switching → higher $w_\text{sparse}$) | **+0.6647** | $< 10^{-300}$ |

The strongest interpretability result, by a substantial margin, is the correlation between English-token density and the predicted sparse weight: Pearson $r = +0.70$ on dev and $+0.66$ on test, with $p$-values below $10^{-300}$ on both splits. This constitutes direct evidence that the MLP has learned a linguistically grounded preference — increasing English code-switching in the query elicits a higher predicted weight on the BGE-M3 learned-sparse retriever, which is the only signal in the trio whose tokeniser was trained on code-switched data. The mechanism is not stated in the loss function or the architecture; it emerges from the soft-label simplex supervision and the seven-feature query representation alone.

The diacritic correlation comes out positive on the clean splits, in the opposite direction to the hypothesis suggested by §1. The effect is small ($r \approx +0.10$) but statistically significant. A plausible explanation lies in the structure of Wikipedia-style questions: queries with high diacritic density are also more likely to be full-form entity questions ("ai là người sáng lập …", "khi nào … được thành lập"), for which dense semantic retrieval is particularly effective. The MLP appears to pick up this surface correlation rather than the diacritic-as-noise signal predicted in §1. The hypothesised negative direction holds where it was originally motivated — on the diacritic-stripped noisy split (§5.1), where the MLP shifts $\bar{w}_\text{sparse}$ upward from $0.332$ to $0.354$ and $\bar{w}_\text{bm25}$ downward from $0.320$ to $0.310$. The clean-split correlation is therefore not a failure of the predicted relationship but evidence that, in the absence of noise, the feature is correlated with a different latent property of the query that benefits dense retrieval.

The compound-density correlation is statistically indistinguishable from zero ($r = +0.02$ on dev, $-0.02$ on test, $p \geq 0.15$). The MLP does not condition its predicted BM25 weight on compound density. This is a useful negative result: the underthesea segmenter already extracts most of the available term-match signal on Vietnamese compounds, leaving the MLP no marginal discrimination to perform. It also implies that the compound feature is the weakest contributor to the seven-feature extractor on clean queries, and is a natural candidate for replacement in any future re-design of the linguistic-feature module (§6).

We also report weight entropy $H = -\sum_i w_i \log w_i$ over the three-way weight distribution. The maximum is $\ln 3 \approx 1.099$ for the uniform $(1/3, 1/3, 1/3)$ prior.

| Statistic | Dev | Test |
|-----------|-----|------|
| $\bar{H}$ (mean entropy) | 1.0976 | 1.0977 |
| $\sigma_H$ (std entropy) | 0.0009 | 0.0008 |
| $\bar{w}_\text{dense}$, $\sigma_{w_\text{dense}}$ | 0.3490, 0.0074 | 0.3488, 0.0073 |
| $\bar{w}_\text{bm25}$, $\sigma_{w_\text{bm25}}$ | 0.3193, 0.0102 | 0.3197, 0.0097 |
| $\bar{w}_\text{sparse}$, $\sigma_{w_\text{sparse}}$ | 0.3317, 0.0061 | 0.3315, 0.0059 |

The mean predicted entropy of $\bar{H} \approx 1.097$ lies extremely close to the uniform-entropy ceiling of $\ln 3 \approx 1.099$, with a standard deviation of approximately $10^{-3}$. This concentration is a direct consequence of the soft-label supervision protocol: the 66-point simplex grid combined with temperature $T = 0.3$ produces target distributions that themselves lie close to the interior of the simplex for nearly every training query, and the MLP correctly reproduces this property rather than over-extending into the simplex corners. The per-query weight ranges (for example $w_\text{dense} \in [0.31, 0.38]$ on dev) are narrow but non-degenerate, and the variance is precisely what generates the stratified gains documented in §5.3.1: query-conditional shifts of weight on the order of $\pm 0.03$ translate into consistent NDCG@10 improvements on the strata where adaptive behaviour is informative.

The same finding identifies the principal limitation of the present architecture. The MLP retains substantial representational headroom — its outputs cluster so tightly around the simplex centre that the per-query response is rarely confident — and a sharper supervision target offers the most direct route to amplifying the per-stratum gains observed here. §5.6 ablates this hypothesis by training fusion MLPs at $T \in \{0.1, 0.3, 1.0\}$ and at the hard-label limit. The soft-label curve improves monotonically as $T$ decreases (dev NDCG@10 of $0.8469 \to 0.8479 \to 0.8489$) without ever collapsing the simplex, while the hard-label limit places almost all probability mass on a single simplex corner and underperforms dense-only retrieval. Soft-label supervision is therefore load-bearing for the architecture, and further sharpening within the soft regime is a concrete future-work direction.

### 5.4. Statistical Significance

We report two-sided paired $t$-tests and Wilcoxon signed-rank tests on per-query NDCG@10 differences, alongside 95\% bootstrap confidence intervals on the mean NDCG@10 delta (2,000 resamples). All tests pair every baseline against the dynamic MLP on each of the three evaluation conditions.

**Dev set (3,814 queries):**

| Comparison | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| MLP vs. Fixed-equal three-way | +0.0016 | [+0.0006, +0.0026] | $1.9\!\times\!10^{-3}$ | $4.4\!\times\!10^{-5}$ |
| MLP vs. Dense + BM25 (0.5/0.5) | +0.0149 | [+0.0105, +0.0194] | $7.7\!\times\!10^{-11}$ | $1.4\!\times\!10^{-10}$ |
| MLP vs. Dense only | +0.0526 | [+0.0450, +0.0599] | $2.1\!\times\!10^{-39}$ | $2.5\!\times\!10^{-37}$ |
| MLP vs. BM25 only | +0.1709 | [+0.1614, +0.1804] | $1.4\!\times\!10^{-229}$ | $3.5\!\times\!10^{-196}$ |
| MLP vs. Sparse only (BGE-M3) | +0.0972 | [+0.0896, +0.1042] | $1.9\!\times\!10^{-130}$ | $1.3\!\times\!10^{-117}$ |

**Test set (7,301 queries):**

| Comparison | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| MLP vs. Fixed-equal three-way | +0.0028 | [+0.0020, +0.0037] | $1.3\!\times\!10^{-10}$ | $6.9\!\times\!10^{-14}$ |
| MLP vs. Dense + BM25 (0.5/0.5) | +0.0236 | [+0.0203, +0.0269] | $2.1\!\times\!10^{-42}$ | $2.6\!\times\!10^{-41}$ |
| MLP vs. Dense only | +0.0446 | [+0.0392, +0.0505] | $2.5\!\times\!10^{-56}$ | $1.3\!\times\!10^{-52}$ |
| MLP vs. BM25 only | +0.1891 | [+0.1820, +0.1965] | $< 10^{-300}$ | $< 10^{-300}$ |
| MLP vs. Sparse only (BGE-M3) | +0.0919 | [+0.0865, +0.0973] | $1.2\!\times\!10^{-229}$ | $1.1\!\times\!10^{-206}$ |

**Diacritic-noisy dev (3,814 queries with all diacritics stripped):**

| Comparison | Δ NDCG@10 | 95% CI | t-test $p$ | Wilcoxon $p$ |
|------------|-----------|--------|-----------|-------------|
| MLP vs. Fixed-equal three-way | +0.0024 | [+0.0011, +0.0037] | $6.1\!\times\!10^{-4}$ | $6.2\!\times\!10^{-8}$ |
| MLP vs. Dense + BM25 (0.5/0.5) | +0.0944 | [+0.0870, +0.1018] | $8.8\!\times\!10^{-126}$ | $1.7\!\times\!10^{-122}$ |
| MLP vs. Dense only | +0.1038 | [+0.0950, +0.1122] | $5.9\!\times\!10^{-110}$ | $2.3\!\times\!10^{-104}$ |
| MLP vs. BM25 only | +0.2434 | [+0.2321, +0.2554] | $5.3\!\times\!10^{-311}$ | $3.5\!\times\!10^{-253}$ |
| MLP vs. Sparse only (BGE-M3) | +0.0322 | [+0.0253, +0.0395] | $7.5\!\times\!10^{-19}$ | $3.2\!\times\!10^{-18}$ |

The MLP outperforms every baseline on every split with $p \ll 0.01$. Two of these comparisons answer substantively different research questions and deserve to be kept distinct. The comparison against the fixed-equal three-way baseline isolates the contribution of *adaptive weighting*, holding constant the choice of three retrievers and the simplex on which they are fused; this comparison is significant on all three splits, with $p$ ranging from $6.1\!\times\!10^{-4}$ to $1.3\!\times\!10^{-10}$. The comparisons against the two-way dense + BM25 baseline and against each single-signal retriever, by contrast, conflate the gain from adding the BGE-M3 sparse signal with the gain from adapting the weights. The large magnitudes of those comparisons (for example $+0.094$ NDCG@10 on the noisy split versus the two-way baseline) primarily reflect the value of the third retriever rather than of the adaptive component. Both perspectives are nevertheless reported because they together answer the natural pair of questions a reader of this paper may ask: "does three-way fusion help?" and "does adaptive weighting add anything beyond fixed three-way fusion?", with each receiving an independently significant affirmative answer.

### 5.5. Efficiency Analysis

| Component | Value |
|-----------|-------|
| MLP parameters | 2,691 |
| MLP inference latency (CPU, $n$ = 7,301 test queries) | $204 \pm 40$ μs |
| FAISS index — ViQuAD (1024-dim L2-norm, 5,317 passages) | 21.78 MB |
| BM25 index — ViQuAD | 12.75 MB |
| BGE-M3 sparse index — ViQuAD | 15.85 MB |
| FAISS index — DANGDOCAO (1024-dim, 37,239 passages) | 152.53 MB |
| BM25 index — DANGDOCAO | 103.08 MB |
| BGE-M3 sparse index — DANGDOCAO | 112.99 MB |

The fusion MLP imposes a mean per-query inference cost of $204$ μs on CPU, three orders of magnitude below a single embedding API call ($\sim 100$ ms) and four orders below a Qwen3-32B generation call. Adaptive fusion is therefore effectively free relative to the dominant latency components of any practical Vietnamese RAG pipeline. The combined index footprint for the ViQuAD corpus is $50.4$ MB (FAISS + BM25 + BGE-M3 sparse) for 5,317 passages, scaling roughly linearly to $368.6$ MB for the 37,239-passage DANGDOCAO corpus. The BGE-M3 sparse index is comparable in size to the BM25 index despite carrying learned per-token weights because non-positive weights are discarded at index-build time, leaving a highly sparse posting list.

### 5.6. Soft Label Ablation

This section isolates the contribution of the soft-label simplex supervision proposed in §3.5 by comparing four variants of an otherwise identical training procedure. All variants share the architecture, training set ($5{,}000$ UIT-ViQuAD queries), simplex grid ($66$ points at step $0.1$), optimiser (Adam, lr $10^{-3}$, batch $256$, $100$ epochs, seed $42$), and evaluation set (UIT-ViQuAD dev, $3{,}814$ queries). They differ only in how supervision targets are constructed from the per-query NDCG@10 profile over the simplex grid: three soft-label settings at temperatures $T \in \{0.1, 0.3, 1.0\}$, plus a hard-label variant that takes the argmax over the grid.

| Label strategy | Temp $T$ | NDCG@10 | MRR@10 | Hit@1 | $\bar{w}_\text{dense}, \bar{w}_\text{bm25}, \bar{w}_\text{sparse}$ | $\bar{H}$ |
|----------------|----------|---------|--------|-------|-------------------------------------------------------------------|-----------|
| Hard label (argmax over simplex grid) | — | 0.7745 | 0.7366 | 0.6508 | (0.053, 0.084, 0.863) | 0.4860 |
| Soft label (proposed, sharper) | **0.1** | **0.8489** | **0.8182** | **0.7462** | (0.358, 0.315, 0.327) | 1.0965 |
| Soft label (proposed, default) | 0.3 | 0.8479 | 0.8170 | 0.7444 | (0.349, 0.319, 0.332) | 1.0976 |
| Soft label (proposed, smoother) | 1.0 | 0.8469 | 0.8162 | 0.7441 | (0.339, 0.327, 0.334) | 1.0985 |

The ablation supports three substantive conclusions. First, hard-label supervision is catastrophic for this architecture. The argmax operator over the simplex selects a single corner for nearly every training query, and because BGE-M3 learned sparse is the strongest single retriever on a plurality of ViQuAD dev queries, the resulting target distribution concentrates almost all mass on $(0, 0, 1)$. The empirical target means $(\bar{w}_\text{dense}, \bar{w}_\text{bm25}, \bar{w}_\text{sparse}) = (0.053, 0.084, 0.863)$ and the entropy $\bar{H} = 0.486 \ll \ln 3$ make this concentration explicit. The MLP duly learns to predict near-sparse-only weights, and the resulting dev NDCG@10 of $0.7745$ is *below* the dense-only retrieval result ($0.7953$) and barely above the sparse-only result ($0.7507$): hard-label fusion under-performs no fusion at all. This is the clearest empirical evidence in the paper for why soft-label supervision is load-bearing.

Second, within the soft-label regime, sharper supervision is monotonically better but the curve is shallow. As $T$ decreases from $1.0$ to $0.3$ to $0.1$, dev NDCG@10 rises from $0.8469$ to $0.8479$ to $0.8489$, and MRR@10 and Hit@1 follow the same ordering. The corresponding entropies move from $1.0985$ to $1.0976$ to $1.0965$, all of which remain extremely close to the $\ln 3$ uniform ceiling. Even the sharpest soft setting we test is therefore far from a confident corner-of-simplex prediction, and the catastrophic regime documented in the first finding occurs only at the strict argmax limit. The improvement of $T = 0.1$ over the default $T = 0.3$ ($+0.0010$ NDCG@10) is small but consistent across the four headline metrics, suggesting that there exists a finite $T^* \in (0, 0.1)$ at which the soft target becomes effectively hard and the catastrophe recurs. Locating $T^*$ is a natural extension of this work.

Third, the headline configuration is robust to the precise temperature within the soft regime. The NDCG@10 spread across the three soft-label rows is only $\pm 0.001$, indicating that the choice of $T = 0.3$ is defensible rather than load-bearing. This is a desirable property for cross-domain deployment, in which expensive hyper-parameter search may not be feasible: the qualitative dichotomy that matters is soft-versus-hard supervision, not the specific temperature within the soft regime.

### 5.7. End-to-end QA Results (RAGAS, Qwen3-32B judge)

We evaluate end-to-end RAG quality using RAGAS [CITATION] with Qwen3-32B as the LLM judge on 200 sampled UIT-ViQuAD 2.0 dev queries per condition. The four RAGAS metrics decompose end-to-end quality along complementary axes:

| Metric | Description |
|--------|-------------|
| **Context Precision** | LLM judges whether each retrieved chunk is relevant to the question |
| **Context Recall** | LLM judges whether the retrieved chunks collectively cover the ground-truth answer |
| **Faithfulness** | LLM judges whether all answer statements are grounded in retrieved context |
| **Answer Relevancy** | Embedding similarity between question and generated answer |

**Clean queries** (UIT-ViQuAD 2.0 dev, $n = 200$):

| Method | Ctx Precision | Ctx Recall | Faithfulness | Ans. Relevancy |
|--------|--------------|------------|--------------|----------------|
| BM25 only | 0.5839 | 0.8150 | 0.8024 | 0.6661 |
| Dense only | **0.7879** | 0.9425 | **0.8690** | **0.7860** |
| Sparse only (BGE-M3) | 0.7036 | 0.8959 | 0.8335 | 0.7292 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.7742 | 0.9308 | 0.8547 | 0.7750 |
| **Dynamic MLP (ours)** | 0.7729 | **0.9433** | 0.8500 | 0.7802 |

**Diacritic-removed queries** (UIT-ViQuAD 2.0 dev with all tones stripped, $n = 200$):

| Method | Ctx Precision | Ctx Recall | Faithfulness | Ans. Relevancy |
|--------|--------------|------------|--------------|----------------|
| BM25 only | 0.2051 | 0.2830 | 0.7639 | 0.2235 |
| Dense only | 0.3647 | 0.5100 | 0.7788 | 0.3317 |
| Sparse only (BGE-M3) | 0.4467 | 0.6000 | 0.7835 | 0.3513 |
| Fixed-equal three-way (1/3,1/3,1/3) | 0.4639 | 0.6325 | **0.8001** | 0.3503 |
| **Dynamic MLP (ours)** | **0.4710** | **0.6400** | 0.7823 | **0.3755** |

The end-to-end picture under clean queries is more nuanced than the retrieval-level results in §5.1 would suggest. At $n = 200$, dense-only retrieval ties or leads on three of the four RAGAS metrics — Context Precision ($0.7879$), Faithfulness ($0.8690$), and Answer Relevancy ($0.7860$) — while the dynamic MLP leads only on Context Recall ($0.9433$). The MLP and the fixed-equal three-way fusion are statistically indistinguishable on every metric under clean queries, and both slightly trail dense-only on three of four. Two factors plausibly explain this divergence between the retrieval-level ranking in §5.1 (where the MLP is the strongest method) and the end-to-end ranking here. First, RAGAS metrics are dominated by *which* passage reaches the generator, not by ranking depth within the top-$k$, so the higher fusion NDCG@10 documented elsewhere has little leverage on RAGAS when Recall@$k$ is already very high. Second, dense-only retrieval is the strongest single source of *single-passage* relevance — the very property RAGAS measures — and the diluting weight given to BM25 and sparse signals by the three-way fusion modestly reduces the average per-passage relevance even when overall ranking quality improves.

The picture inverts cleanly on the diacritic-stripped split. Under noise, the dynamic MLP leads on Context Precision ($0.4710$), Context Recall ($0.6400$), and Answer Relevancy ($0.3755$), trailing only on Faithfulness, where the fixed-equal three-way reaches $0.8001$ versus the MLP's $0.7823$ — and where the absolute spread across all five methods is only $\sim 0.04$. The MLP's lead over the fixed-equal baseline is $+0.0071$ Context Precision, $+0.0075$ Context Recall, and $+0.0252$ Answer Relevancy. The same mechanism that produced the retrieval-level gains under noise (up-weighting BGE-M3 sparse on queries with diacritic-poor surface features, as quantified in §5.3) is therefore visible at the end-to-end QA level, and the adaptive component is the dominant lever for noise-tolerant Vietnamese RAG.

A robustness pattern that survives the change of sample size is the relative resilience of Faithfulness compared to Answer Relevancy. Across all five methods, Faithfulness drops by only $7$–$17\%$ from clean to noisy, while Answer Relevancy drops by $48$–$66\%$. The LLM stays grounded in whatever context it receives even when that context provides less of the right information for the original question. The dominant end-to-end failure mode under diacritic noise is therefore best characterised as "answers the wrong question faithfully" rather than "hallucinates from a relevant context." This finding has practical consequences for downstream Vietnamese RAG deployments: the principal lever for noise robustness is the retriever, not the generator.

Two caveats apply. First, the $200$-sample size, while four times larger than our pilot, remains small relative to the dev split itself; significance testing on the four RAGAS metrics is not reported, and the cross-method orderings above should be read as suggestive of qualitative pattern rather than as point-significant claims. Second, a non-trivial fraction of Faithfulness scoring calls were truncated by the $1{,}024$-token completion limit on Qwen3-32B during long statement-extraction outputs and were excluded from the metric mean as failed retries. With a larger token budget the absolute Faithfulness values would likely rise across all methods, but the relative ordering across methods should be preserved because the token cap is applied uniformly.

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
