# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AI Engineering Pair  
**Team Members:** Vinuj & Antigravity  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We developed a high-precision, country-partitioned entity resolution pipeline combining universal script transliteration (`anyascii`), character $n$-gram TF-IDF cosine blocking, an 18-dimensional fuzzy/syntactic feature extractor, and a calibrated LightGBM binary classifier. Operating under the precision-weighted macro $F_{0.5}$ metric, our blocking stage achieves **98.63% validation candidate recall**, and our matching stage delivers **0.9682 validation macro $F_{0.5}$** under a precision-optimized 1-to-many decision threshold.

---

## 2. Methodology

### 2.1 Problem Analysis
Key noise patterns and challenges identified during exploratory data analysis:
1. **Multilingual & Multi-Script Text**: Records contain text in Hindi (Devanagari), Tamil, Bengali, French, and English, frequently mixing transliterated scripts within the same address or entity (e.g. *Raj Investments எல்எல்பி* vs *ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி* vs *एसएस फूड प्राइवेट लिमिटेड*).
2. **Legal Suffix Inconsistency**: Entities interchange legal entity designations (`Pvt Ltd`, `Private Limited`, `LLP`, `LLC`, `Corp`, `Inc`, `SAS`, `SARL`).
3. **Scrambled & Omitted Addresses**: Address components (PIN codes, unit numbers, street names, cities) appear in non-standard permutations with frequent OCR/typo noise.
4. **Metric Sensitivity**: The macro-averaged $F_{0.5}$ metric penalizes false merges twice as heavily as missed matches ($\beta = 0.5$). Singletons earn a perfect 1.0 for predicting empty matches and 0.0 for any incorrect prediction, requiring strict false-positive suppression.
5. **Scale**: The test set comprises 1.73M $S_1$ queries and ~9.97M records across $S_2$ and $S_3$, requiring strict partition-based blocking.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Hybrid Blocking (TF-IDF Character $n$-grams) + GBDT Binary Classifier + Greedy 1-to-Many Optimization.  
**Core Innovation:** Combining multi-script universal phonetic/ASCII normalization with sparse character-boundary $n$-gram matrix dot-products for candidate generation, paired with a non-linear LightGBM classifier optimizing directly for the $F_{0.5}$ macro-average.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys & representation:** 
  1. Universal transliteration via `anyascii` to project all non-Latin characters to ASCII.
  2. Removal of common web domain extensions (`.com`, `.net`, `.in`, `.fr`).
  3. Character word-boundary $n$-grams (`ngram_range=(2, 4)`) with sublinear term-frequency scaling.
  4. Strict country partitioning (`India`, `US`, `France`), eliminating cross-country comparisons.
  5. Sparse matrix multiplication chunking ($5,000$ queries per chunk) to retrieve Top-$K$ ($K=50$) candidates per $S_1$ entity.
- **Candidate pairs generated:** 50 candidates per $S_1$ entity.
- **How true matches were preserved:**
  - Evaluated on 51,804 ground-truth matches on a 15,000 validation split.
  - The character $n$-gram TF-IDF blocking recalled **51,093 / 51,804 true matches (98.63% recall)**.

---

## 4. Matching Model

**Features used (18 dimensions):**
- **Name features:** `n_token_set`, `n_token_sort`, `n_ratio`, `n_partial`, `nn_token_set`, `nn_token_sort` (with legal suffixes removed), and character 3-gram Jaccard index (`ngram_jaccard`).
- **Address features:** `a_token_set`, `a_token_sort`, `a_ratio`, `a_partial`, and address presence flag `has_both_addr`.
- **Numeric features:** Extracted digit sequences overlap (`num_common`) and Jaccard similarity (`num_jaccard`).
- **Composite bounds & blocking:** `tfidf_score` (cosine similarity), `max_name_sim`, `max_addr_sim`, `max_any_sim`, `min_sim`.

**Model type:** LightGBM Binary Classifier (`n_estimators=500`, `num_leaves=63`, `max_depth=8`, `scale_pos_weight=13.48`).  
**Threshold selection method:** Grid sweep optimizing macro $F_{0.5}$ on out-of-fold validation set with greedy 1-to-many candidate uniqueness constraint.

---

## 5. Results & Error Analysis

- **$F_{0.5}$ Score (macro):** **0.9682** at optimal decision threshold of **0.90**.
- **Threshold progression:**
  - Threshold 0.50: $F_{0.5} = 0.9482$
  - Threshold 0.70: $F_{0.5} = 0.9591$
  - Threshold 0.80: $F_{0.5} = 0.9640$
  - Threshold 0.90: $F_{0.5} = 0.9682$
- **Feature Importance:**
  1. `tfidf_score` (2902 splits)
  2. `n_ratio` & `n_partial` (2712 & 2519 splits)
  3. `a_partial` & `a_token_sort` (2190 & 2016 splits)
  4. `max_any_sim` & `ngram_jaccard` (1564 & 1558 splits)
- **Common false positives (wrong merges):** Franchise branches sharing corporate parent names in identical postal regions but differing in sub-unit numbers.
- **Common false negatives (missed matches):** Ultra-short single-word acronyms (e.g., 2-character abbreviations) with omitted addresses where TF-IDF overlap was sparse.

---

## 6. Conclusion
By replacing exact-word inverted indexing with character-level multi-script transliteration and TF-IDF blocking, we lifted candidate recall from <91% to 98.63%. Combining this with an 18-feature LightGBM model and precision-weighted thresholding elevated the macro $F_{0.5}$ score from 0.38 to **0.9682**, establishing a competitive, scalable solution.

---

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/`: Contains preprocessing, TF-IDF blocking, feature extraction, and LightGBM inference modules.
- `requirements.txt`: Pinned dependencies (`lightgbm`, `rapidfuzz`, `anyascii`, `scikit-learn`, `scipy`, `pandas`, `numpy`).
- Execution entry point reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
