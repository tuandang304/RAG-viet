# Proposal: Dynamic Hybrid Retrieval-Augmented Generation for Vietnamese

## 1. Project Objective
The objective of this project is to develop **Dynamic Hybrid RAG**, a lightweight framework designed specifically for Vietnamese Retrieval-Augmented Generation. Instead of relying on a one-size-fits-all combination of retrieval signals (dense vector search, sparse lexical matching like BM25, and learned-sparse representations), this project proposes an adaptive Multi-Layer Perceptron (MLP). This MLP will dynamically predict per-query three-way fusion weights based on Vietnamese-specific linguistic features. 

## 2. Key Findings & Motivations
From initial research and empirical analysis, several key insights motivate this work:
*   **Vietnamese Specific Challenges:** Retrieval effectiveness in Vietnamese is highly sensitive to three main factors: word segmentation dependency (words spanning multiple syllables), diacritical mark presence (users often type without tone marks), and frequent code-switching with English (especially in technical domains).
*   **Signal Complementarity:** Classical BM25 heavily relies on proper word segmentation and fails drastically when diacritics are missing. Dense embeddings are robust to diacritic noise but can blur specific lexical details. Learned-sparse representations (like BGE-M3) offer a middle ground, preserving token-level importance and handling code-switched terms better than BM25.
*   **Superiority of Dynamic Fusion:** An adaptive weighting strategy significantly outperforms any single-signal retriever (dense-only, BM25-only, or sparse-only) as well as fixed-weight hybrid baselines. 
*   **Domain Generalization:** A dynamic MLP trained on a specific domain (e.g., Wikipedia via UIT-ViQuAD 2.0) can successfully generalize to unseen domains (e.g., legal/administrative texts in DANGDOCAO) in a zero-shot setting, proving it learns domain-invariant linguistic signals rather than dataset-specific lexical statistics.

## 3. Implementation Methodology
The implementation will be carried out through the following technical components:
*   **Retrieval Components:** 
    *   *Dense Retrieval:* Using FPT Vietnamese Embedding (1024-dim) indexed with FAISS for semantic search.
    *   *BM25 Retrieval:* Using `underthesea` for Vietnamese word segmentation and `BM25Okapi` for lexical matching.
    *   *Learned Sparse Retrieval:* Using BGE-M3 lexical weights via an inverted index to capture learned term importance.
*   **Feature Extraction:** A dedicated module to extract 7 lightweight linguistic features from the query: diacritic ratio, compound word ratio, English token ratio, tech-term ratio, clause count, question-word presence, and normalized query length.
*   **Adaptive MLP Fusion:** A 3-layer feed-forward neural network (approx. 2,691 parameters) taking the 7 features as input and outputting a softmax distribution over the 3 retrieval signals (dense, BM25, sparse) to guarantee weights sum to 1.
*   **Soft-Label Training Strategy:** To train the MLP without ground-truth weights, we will compute NDCG@10 across a 3D simplex grid of possible weights (66 points). We will then apply a temperature-scaled softmax to create smooth, expected-weight targets, minimizing MSE loss against these expected targets.

## 4. Contributions
While hybrid retrieval is well-studied for English, our approach introduces several novelties:
*   **Dynamic vs. Fixed Weights:** Unlike standard approaches (e.g., Reciprocal Rank Fusion or fixed linear interpolation) that use the same static weights for all queries, our system computes custom fusion weights *per query*.
*   **Vietnamese-Aware Design:** The system explicitly models the nuances of the Vietnamese language through a custom 7-feature extractor, rather than applying language-agnostic statistical methods.
*   **Soft-Label Simplex Supervision:** Instead of using hard argmax labels (which cause the model to collapse onto a single retriever and perform poorly), we introduce a temperature-scaled soft-label strategy over a 3D simplex grid, avoiding tie-breaking ambiguity and ensuring smooth gradient updates.

## 5. Timeline
*   **Week 1: Project Setup & Data Preparation**
    *   Literature review and baseline setup.
    *   Acquire and preprocess datasets: UIT-ViQuAD 2.0 (Wikipedia) and DANGDOCAO (Legal/Administrative).
*   **Week 2: Implementation of Base Retrievers**
    *   Set up Dense retrieval (FPT embeddings + FAISS).
    *   Set up BM25 retrieval (underthesea segmentation + BM25Okapi).
    *   Set up BGE-M3 Learned Sparse retrieval (inverted index).
*   **Week 3: Feature Extractor Development**
    *   Implement the Vietnamese-aware 7-feature extractor.
    *   Run feature extraction over the training and dev query sets.
*   **Week 4: Fusion Model & Training Pipeline**
    *   Build the 3-layer MLP fusion module.
    *   Implement the 3D simplex grid search and the soft-label supervision target generator.
*   **Week 5: Training & In-Domain Evaluation**
    *   Train the MLP on augmented UIT-ViQuAD 2.0 data (including diacritic-stripped queries).
    *   Evaluate NDCG, MRR, MAP, and Recall against fixed baselines on the in-domain test set.
*   **Week 6: Cross-Domain & Robustness Testing**
    *   Conduct zero-shot cross-domain evaluation on the DANGDOCAO corpus.
    *   Perform robustness analysis using diacritic-removed (noisy) query sets.
*   **Week 7: End-to-End RAG Evaluation**
    *   Integrate the retrieval pipeline with a generator (e.g., Qwen3-32B).
    *   Evaluate end-to-end QA quality using RAGAS metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy).
*   **Week 8: Finalization & Documentation**
    *   Conduct ablation studies (e.g., varying the soft-label temperature).
    *   Finalize code refactoring, synthesize results, and complete the final research report/paper.
