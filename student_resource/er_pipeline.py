"""
Entity Resolution Pipeline v2 - TF-IDF Blocking + LightGBM Classifier
Targets F0.5 > 0.99 on Amazon ML Challenge 2026

Architecture:
1. Preprocessing: anyascii transliteration + text normalization
2. Blocking: TF-IDF char n-gram cosine similarity (per country)
3. Feature Engineering: 18 fuzzy similarity features
4. Classification: LightGBM binary classifier
5. Post-processing: 1-to-many constraint, threshold optimization
"""

import sys
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anyascii
from rapidfuzz import fuzz, utils as fuzz_utils
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# CONFIGURATION
# ============================================================================
TOP_K_CANDIDATES = 50       # candidates per S1 from TF-IDF blocking
LGBM_THRESHOLD = 0.5        # initial threshold, will be optimized
MAX_POSTINGS = 500           # max postings per inverted index key
TFIDF_NGRAM_RANGE = (2, 4)  # character n-gram range
TFIDF_MAX_FEATURES = 200000 # max TF-IDF features
CHUNK_SIZE = 5000            # S1 chunk size for TF-IDF similarity

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'services', 'solutions', 'holdings', 'group', 'the', 'and', 'center',
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'cie', 'partners',
    'associates', 'international', 'india', 'of', 'for', 'dba'
}

CLEAN_RE = re.compile(r'[^a-z0-9\s]')
DIGIT_RE = re.compile(r'\d+')

# ============================================================================
# PREPROCESSING
# ============================================================================
def normalize_text(s):
    """Transliterate to ASCII, lowercase, strip punctuation."""
    if pd.isna(s) or str(s).strip() in ('', 'nan', 'NaN', '<NULL>', 'null', 'N/A'):
        return ""
    s = str(s)
    s = anyascii.anyascii(s).lower()
    # Remove .com/.net etc domains
    s = re.sub(r'\b\w+\.(com|net|org|in|co|us|fr)\b', '', s)
    s = CLEAN_RE.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_name(s):
    """Normalize name: transliterate, remove legal suffixes."""
    n = normalize_text(s)
    words = [w for w in n.split() if w not in LEGAL_STOPWORDS and len(w) > 1]
    return " ".join(words)

def make_tfidf_text(name, addr):
    """Concatenate normalized name + address for TF-IDF."""
    n = normalize_text(name)
    a = normalize_text(addr)
    return f"{n} {a}".strip()

def extract_numbers(s):
    """Extract all digit sequences from a string."""
    if pd.isna(s):
        return set()
    return set(DIGIT_RE.findall(str(s)))

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def compute_pair_features(s1_name, s1_addr, c_name, c_addr, tfidf_score=0.0):
    """Compute rich feature vector for a (S1, candidate) pair."""
    # Normalize
    s1n = normalize_text(s1_name)
    cn = normalize_text(c_name)
    s1a = normalize_text(s1_addr)
    ca = normalize_text(c_addr)
    
    # Name similarities
    n_token_set = fuzz.token_set_ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_token_sort = fuzz.token_sort_ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_ratio = fuzz.ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_partial = fuzz.partial_ratio(s1n, cn, processor=fuzz_utils.default_process)
    
    # Normalized name (without legal suffixes)
    s1nn = normalize_name(s1_name)
    cnn = normalize_name(c_name)
    nn_token_set = fuzz.token_set_ratio(s1nn, cnn, processor=fuzz_utils.default_process)
    nn_token_sort = fuzz.token_sort_ratio(s1nn, cnn, processor=fuzz_utils.default_process)
    
    # Address similarities
    has_both_addr = 1.0 if (s1a and ca) else 0.0
    if s1a and ca:
        a_token_set = fuzz.token_set_ratio(s1a, ca, processor=fuzz_utils.default_process)
        a_token_sort = fuzz.token_sort_ratio(s1a, ca, processor=fuzz_utils.default_process)
        a_ratio = fuzz.ratio(s1a, ca, processor=fuzz_utils.default_process)
        a_partial = fuzz.partial_ratio(s1a, ca, processor=fuzz_utils.default_process)
    else:
        a_token_set = 0.0
        a_token_sort = 0.0
        a_ratio = 0.0
        a_partial = 0.0
    
    # Number matching
    s1_nums = extract_numbers(s1_name) | extract_numbers(s1_addr)
    c_nums = extract_numbers(c_name) | extract_numbers(c_addr)
    if s1_nums or c_nums:
        num_common = len(s1_nums & c_nums)
        num_total = len(s1_nums | c_nums)
        num_jaccard = num_common / num_total
    else:
        num_common = 0
        num_jaccard = 0.5  # neutral when no numbers
    
    # Aggregate scores
    max_name_sim = max(n_token_set, nn_token_set)
    max_addr_sim = a_token_set
    max_any_sim = max(max_name_sim, max_addr_sim)
    min_sim = min(max_name_sim, max_addr_sim) if has_both_addr else max_name_sim
    
    # Character n-gram Jaccard on names
    def char_ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()
    
    s1_ngrams = char_ngrams(s1nn, 3)
    c_ngrams = char_ngrams(cnn, 3)
    if s1_ngrams or c_ngrams:
        ngram_jaccard = len(s1_ngrams & c_ngrams) / max(1, len(s1_ngrams | c_ngrams))
    else:
        ngram_jaccard = 0.0
    
    return [
        n_token_set, n_token_sort, n_ratio, n_partial,        # 0-3: name sims
        nn_token_set, nn_token_sort,                            # 4-5: cleaned name sims
        a_token_set, a_token_sort, a_ratio, a_partial,          # 6-9: addr sims
        has_both_addr,                                          # 10: addr presence
        num_common, num_jaccard,                                # 11-12: number features
        max_name_sim, max_addr_sim, max_any_sim, min_sim,       # 13-16: aggregate
        ngram_jaccard,                                          # 17: char ngram
        tfidf_score                                             # 18: TF-IDF cosine
    ]

FEATURE_NAMES = [
    'n_token_set', 'n_token_sort', 'n_ratio', 'n_partial',
    'nn_token_set', 'nn_token_sort',
    'a_token_set', 'a_token_sort', 'a_ratio', 'a_partial',
    'has_both_addr',
    'num_common', 'num_jaccard',
    'max_name_sim', 'max_addr_sim', 'max_any_sim', 'min_sim',
    'ngram_jaccard',
    'tfidf_score'
]

# ============================================================================
# TF-IDF BLOCKING
# ============================================================================
def tfidf_block_country(s1_texts, other_texts, other_eids, top_k=50, chunk_size=5000):
    """
    For a set of S1 texts and S2/S3 texts (same country),
    return top_k candidate IDs per S1 entity with TF-IDF cosine similarity.
    """
    print(f"    TF-IDF vectorizing {len(other_texts)} S2/S3 records...", flush=True)
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32
    )
    other_tfidf = tfidf.fit_transform(other_texts)
    
    print(f"    TF-IDF vectorizing {len(s1_texts)} S1 records...", flush=True)
    s1_tfidf = tfidf.transform(s1_texts)
    
    all_candidates = []
    all_scores = []
    
    n_chunks = (len(s1_texts) + chunk_size - 1) // chunk_size
    print(f"    Computing cosine similarity in {n_chunks} chunks...", flush=True)
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(s1_texts))
        
        # Sparse matrix multiplication for cosine similarity
        sim_matrix = s1_tfidf[start:end].dot(other_tfidf.T)
        
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                all_candidates.append([])
                all_scores.append([])
                continue
                
            data = row.data
            indices = row.indices
            
            # Get top_k
            if len(data) <= top_k:
                top_idx = np.argsort(-data)
            else:
                top_idx = np.argpartition(-data, top_k)[:top_k]
                top_idx = top_idx[np.argsort(-data[top_idx])]
            
            cand_ids = [other_eids[indices[j]] for j in top_idx]
            cand_scores = [float(data[j]) for j in top_idx]
            
            all_candidates.append(cand_ids)
            all_scores.append(cand_scores)
        
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"      Chunk {chunk_idx+1}/{n_chunks} done", flush=True)
    
    return all_candidates, all_scores

# ============================================================================
# SCORING
# ============================================================================
def compute_macro_f05(preds_dict, ground_truth):
    """Compute macro-averaged F0.5 score."""
    f05_scores = []
    for s1_id, pred_ids in preds_dict.items():
        true_ids = ground_truth.get(s1_id, set())
        pred_set = set(pred_ids)
        
        if len(true_ids) == 0 and len(pred_set) == 0:
            f05_scores.append(1.0)
            continue
        if len(true_ids) == 0 or len(pred_set) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(pred_set & true_ids)
        fp = len(pred_set - true_ids)
        fn = len(true_ids - pred_set)
        
        if tp == 0:
            f05_scores.append(0.0)
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        f05_scores.append(f05)
        
    return np.mean(f05_scores)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def load_data(mode='validate', n_s1=50000):
    """Load data for validation or test inference."""
    if mode == 'validate':
        print("=== Loading TRAINING data for validation ===", flush=True)
        gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=n_s1)
        s1_path = 'dataset/train/train_source1.tsv'
        s2_path = 'dataset/train/train_source2.tsv'
        s3_path = 'dataset/train/train_source3.tsv'
    else:
        print("=== Loading TEST data for inference ===", flush=True)
        gt_df = None
        s1_path = 'dataset/test/test_source1.tsv'
        s2_path = 'dataset/test/test_source2.tsv'
        s3_path = 'dataset/test/test_source3.tsv'
    
    # Build GT map
    gt_map = {}
    all_true_ids = set()
    if gt_df is not None:
        target_s1 = set(gt_df['source1_entity_id'])
        for r in gt_df.itertuples(index=False):
            m_val = getattr(r, 'matched_entity_ids')
            if pd.notna(m_val) and str(m_val).strip():
                m_set = {x.strip() for x in str(m_val).split(',') if x.strip()}
            else:
                m_set = set()
            gt_map[getattr(r, 'source1_entity_id')] = m_set
            all_true_ids.update(m_set)
    else:
        target_s1 = None
    
    # Load S1
    print("Loading S1...", flush=True)
    if target_s1:
        s1_rows = []
        for chunk in pd.read_csv(s1_path, sep='\t', chunksize=100000):
            sub = chunk[chunk['entity_id'].isin(target_s1)]
            if len(sub) > 0:
                s1_rows.append(sub)
            if len(pd.concat(s1_rows, ignore_index=True)) >= len(target_s1):
                break
        s1_df = pd.concat(s1_rows, ignore_index=True)
    else:
        s1_df = pd.read_csv(s1_path, sep='\t')
    print(f"  S1: {len(s1_df)} records", flush=True)
    
    # Load S2 and S3
    print("Loading S2...", flush=True)
    if mode == 'validate':
        # For validation: load all true matches + sample of distractors
        target_s2 = {x for x in all_true_ids if x.startswith('S2-')}
        s2_rows = []
        for chunk in pd.read_csv(s2_path, sep='\t', chunksize=200000):
            targets = chunk[chunk['entity_id'].isin(target_s2)]
            distractors = chunk[~chunk['entity_id'].isin(target_s2)].sample(frac=0.02, random_state=42)
            s2_rows.append(pd.concat([targets, distractors]))
        s2_df = pd.concat(s2_rows, ignore_index=True).drop_duplicates(subset=['entity_id'])
    else:
        s2_df = pd.read_csv(s2_path, sep='\t')
    print(f"  S2: {len(s2_df)} records", flush=True)
    
    print("Loading S3...", flush=True)
    if mode == 'validate':
        target_s3 = {x for x in all_true_ids if x.startswith('S3-')}
        s3_rows = []
        for chunk in pd.read_csv(s3_path, sep='\t', chunksize=200000):
            targets = chunk[chunk['entity_id'].isin(target_s3)]
            distractors = chunk[~chunk['entity_id'].isin(target_s3)].sample(frac=0.02, random_state=42)
            s3_rows.append(pd.concat([targets, distractors]))
        s3_df = pd.concat(s3_rows, ignore_index=True).drop_duplicates(subset=['entity_id'])
    else:
        s3_df = pd.read_csv(s3_path, sep='\t')
    print(f"  S3: {len(s3_df)} records", flush=True)
    
    other_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"  Total S2+S3: {len(other_df)} records", flush=True)
    
    return s1_df, other_df, gt_map


def run_blocking(s1_df, other_df, top_k=TOP_K_CANDIDATES):
    """Run TF-IDF blocking per country. Returns dict: s1_id -> [(cand_id, tfidf_score), ...]"""
    countries = sorted(s1_df['country'].unique())
    print(f"\nRunning TF-IDF blocking for countries: {countries}", flush=True)
    
    all_cands = {}
    
    for country in countries:
        print(f"\n  Processing country: {country}", flush=True)
        s1_country = s1_df[s1_df['country'] == country].copy()
        other_country = other_df[other_df['country'] == country].copy()
        
        if len(other_country) == 0:
            for eid in s1_country['entity_id']:
                all_cands[eid] = []
            continue
        
        # Build TF-IDF texts
        s1_texts = [make_tfidf_text(n, a) for n, a in zip(s1_country['business_name'], s1_country['business_address'])]
        other_texts = [make_tfidf_text(n, a) for n, a in zip(other_country['business_name'], other_country['business_address'])]
        other_eids = other_country['entity_id'].values
        s1_eids = s1_country['entity_id'].values
        
        cands_list, scores_list = tfidf_block_country(s1_texts, other_texts, other_eids, top_k=top_k)
        
        for i, s1_id in enumerate(s1_eids):
            all_cands[s1_id] = list(zip(cands_list[i], scores_list[i]))
    
    return all_cands


def generate_features(s1_df, other_df, cands_dict, gt_map=None, include_gt_matches=True):
    """Generate feature vectors for all candidate pairs.
    
    Returns: list of (s1_id, cand_id, features, label)
    """
    other_map = other_df.set_index('entity_id')
    s1_map = s1_df.set_index('entity_id')
    
    pairs = []
    total = len(cands_dict)
    done = 0
    t0 = time.time()
    
    for s1_id, cand_list in cands_dict.items():
        if s1_id not in s1_map.index:
            done += 1
            continue
            
        s1_row = s1_map.loc[s1_id]
        s1_name = s1_row['business_name']
        s1_addr = s1_row['business_address']
        
        true_set = gt_map.get(s1_id, set()) if gt_map else set()
        
        seen_cands = set()
        for cand_id, tfidf_score in cand_list:
            if cand_id in seen_cands or cand_id not in other_map.index:
                continue
            seen_cands.add(cand_id)
            
            c_row = other_map.loc[cand_id]
            feats = compute_pair_features(s1_name, s1_addr, c_row['business_name'], c_row['business_address'], tfidf_score)
            label = 1 if cand_id in true_set else 0
            pairs.append((s1_id, cand_id, feats, label))
        
        # If training: also include true matches not in candidates (teacher forcing)
        if include_gt_matches and gt_map:
            for tm in true_set:
                if tm not in seen_cands and tm in other_map.index:
                    c_row = other_map.loc[tm]
                    feats = compute_pair_features(s1_name, s1_addr, c_row['business_name'], c_row['business_address'], 0.0)
                    pairs.append((s1_id, tm, feats, 1))
        
        done += 1
        if done % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    Feature extraction: {done}/{total} ({done/total:.0%}) in {elapsed:.1f}s", flush=True)
    
    print(f"    Generated {len(pairs)} pairs, {sum(p[3] for p in pairs)} positive", flush=True)
    return pairs


def train_model(pairs, val_pairs=None):
    """Train LightGBM on pairs."""
    X = np.array([p[2] for p in pairs], dtype=np.float32)
    y = np.array([p[3] for p in pairs], dtype=np.int32)
    
    print(f"\nTraining LightGBM: {X.shape[0]} samples, {np.sum(y)} positives ({np.mean(y):.2%})", flush=True)
    
    # Compute scale_pos_weight for imbalanced data
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    scale_pos_weight = neg_count / max(1, pos_count)
    
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    if val_pairs:
        X_val = np.array([p[2] for p in val_pairs], dtype=np.float32)
        y_val = np.array([p[3] for p in val_pairs], dtype=np.int32)
        clf.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(50)])
    else:
        clf.fit(X, y)
    
    print("Model trained!", flush=True)
    
    # Feature importance
    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"    {name}: {imp}", flush=True)
    
    return clf


def predict_and_optimize(clf, pairs, gt_map, s1_ids_all):
    """Predict on pairs and optimize threshold for F0.5."""
    if not pairs:
        return {s1_id: [] for s1_id in s1_ids_all}, 0.5
    
    X = np.array([p[2] for p in pairs], dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]
    
    # Group by s1_id
    s1_to_preds = defaultdict(list)
    for i, (s1_id, cand_id, _, _) in enumerate(pairs):
        s1_to_preds[s1_id].append((cand_id, probs[i]))
    
    best_thresh = 0.5
    best_score = 0.0
    
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        # Apply threshold with 1-to-many constraint
        all_assignments = []
        for s1_id, cand_probs in s1_to_preds.items():
            for cid, prob in cand_probs:
                if prob >= thresh:
                    all_assignments.append((prob, s1_id, cid))
        
        all_assignments.sort(reverse=True)
        assigned_cand = set()
        final_preds = {s1_id: [] for s1_id in s1_ids_all}
        
        for prob, s1_id, cid in all_assignments:
            if cid not in assigned_cand:
                assigned_cand.add(cid)
                final_preds[s1_id].append(cid)
        
        score = compute_macro_f05(final_preds, gt_map)
        print(f"  Threshold {thresh:.2f} -> F0.5 = {score:.4f}", flush=True)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    print(f"\n>>> BEST F0.5 = {best_score:.4f} at threshold {best_thresh:.2f} <<<", flush=True)
    return best_thresh, best_score


def main_validate():
    """Run full validation pipeline on training data."""
    print("=" * 70, flush=True)
    print("ENTITY RESOLUTION PIPELINE - VALIDATION MODE", flush=True)
    print("=" * 70, flush=True)
    
    t_start = time.time()
    
    # Load data
    s1_df, other_df, gt_map = load_data(mode='validate', n_s1=50000)
    
    # Split S1 into train and val
    from sklearn.model_selection import train_test_split
    train_s1_df, val_s1_df = train_test_split(s1_df, test_size=0.3, random_state=42)
    print(f"\nTrain S1: {len(train_s1_df)}, Val S1: {len(val_s1_df)}", flush=True)
    
    # Blocking
    print("\n--- BLOCKING ---", flush=True)
    train_cands = run_blocking(train_s1_df, other_df, top_k=TOP_K_CANDIDATES)
    val_cands = run_blocking(val_s1_df, other_df, top_k=TOP_K_CANDIDATES)
    
    # Measure blocking recall
    available_other = set(other_df['entity_id'])
    total_true = 0
    found_true = 0
    for s1_id in val_s1_df['entity_id']:
        true_set = gt_map.get(s1_id, set()) & available_other
        cand_set = {c[0] for c in val_cands.get(s1_id, [])}
        total_true += len(true_set)
        found_true += len(true_set & cand_set)
    print(f"\nBlocking recall on val set: {found_true}/{total_true} = {found_true/max(1,total_true):.2%}", flush=True)
    
    # Feature extraction
    print("\n--- FEATURE EXTRACTION ---", flush=True)
    print("Training features:", flush=True)
    train_pairs = generate_features(train_s1_df, other_df, train_cands, gt_map, include_gt_matches=True)
    print("Validation features:", flush=True)
    val_pairs = generate_features(val_s1_df, other_df, val_cands, gt_map, include_gt_matches=False)
    
    # Train model
    print("\n--- MODEL TRAINING ---", flush=True)
    clf = train_model(train_pairs, val_pairs=val_pairs)
    
    # Predict and optimize
    print("\n--- THRESHOLD OPTIMIZATION ---", flush=True)
    val_s1_ids = set(val_s1_df['entity_id'])
    best_thresh, best_score = predict_and_optimize(clf, val_pairs, gt_map, val_s1_ids)
    
    total_time = time.time() - t_start
    print(f"\nTotal validation time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    
    # Save model
    with open('model_lgbm.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'threshold': best_thresh}, f)
    print("Model saved to model_lgbm.pkl", flush=True)
    
    return clf, best_thresh


if __name__ == '__main__':
    main_validate()
