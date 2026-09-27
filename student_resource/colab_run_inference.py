"""
=============================================================================
Amazon ML Challenge 2026 - Accelerated Colab Inference Pipeline
=============================================================================

Instructions for Google Colab:
------------------------------
1. Set Runtime Type:
   Go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU (or A100/V100).

2. Mount Google Drive or Upload Files:
   Put the following files in your Colab working directory (or Google Drive):
     - model_lgbm.pkl
     - test_source1.tsv
     - test_source2.tsv
     - test_source3.tsv
     - (Optional) existing partial candidate_pairs.tsv & matching_results.tsv

3. Install requirements in a Colab cell:
   !pip install -q rapidfuzz anyascii lightgbm scipy scikit-learn

4. Run this script in Colab:
   !python colab_run_inference.py --data-dir ./ --output-dir ./output --model-path ./model_lgbm.pkl --resume
=============================================================================
"""

import sys
import os
import re
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import anyascii
from rapidfuzz import fuzz, utils as fuzz_utils
import lightgbm as lgb
import torch

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Hyperparameters
TOP_K_CANDIDATES = 50
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MAX_FEATURES = 150000
BLOCK_CHUNK_SIZE = 500       # Batch queries for vector/matrix ops
EVAL_BATCH_SIZE = 5000       # Batch size for LightGBM predict_proba

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'services', 'solutions', 'holdings', 'group', 'the', 'and', 'center',
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'cie', 'partners',
    'associates', 'international', 'india', 'of', 'for', 'dba'
}

CLEAN_RE = re.compile(r'[^a-z0-9\s]')
DIGIT_RE = re.compile(r'\d+')

def normalize_text(s):
    if pd.isna(s) or str(s).strip() in ('', 'nan', 'NaN', '<NULL>', 'null', 'N/A'):
        return ""
    s = str(s)
    s = anyascii.anyascii(s).lower()
    s = re.sub(r'\b\w+\.(com|net|org|in|co|us|fr)\b', '', s)
    s = CLEAN_RE.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_name(s):
    n = normalize_text(s)
    words = [w for w in n.split() if w not in LEGAL_STOPWORDS and len(w) > 1]
    return " ".join(words)

def make_tfidf_text(name, addr):
    n = normalize_text(name)
    a = normalize_text(addr)
    return f"{n} {a}".strip()

def extract_numbers(s):
    if pd.isna(s) or not s:
        return set()
    return set(DIGIT_RE.findall(str(s)))

def compute_pair_features(s1n, s1nn, s1a, s1_nums, cn, cnn, ca, c_nums, tfidf_score=0.0):
    """Compute the 19 features required by model_lgbm.pkl."""
    n_token_set = fuzz.token_set_ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_token_sort = fuzz.token_sort_ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_ratio = fuzz.ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_partial = fuzz.partial_ratio(s1n, cn, processor=fuzz_utils.default_process)
    
    nn_token_set = fuzz.token_set_ratio(s1nn, cnn, processor=fuzz_utils.default_process)
    nn_token_sort = fuzz.token_sort_ratio(s1nn, cnn, processor=fuzz_utils.default_process)
    
    has_both_addr = 1.0 if (s1a and ca) else 0.0
    if s1a and ca:
        a_token_set = fuzz.token_set_ratio(s1a, ca, processor=fuzz_utils.default_process)
        a_token_sort = fuzz.token_sort_ratio(s1a, ca, processor=fuzz_utils.default_process)
        a_ratio = fuzz.ratio(s1a, ca, processor=fuzz_utils.default_process)
        a_partial = fuzz.partial_ratio(s1a, ca, processor=fuzz_utils.default_process)
    else:
        a_token_set = a_token_sort = a_ratio = a_partial = 0.0
    
    if s1_nums or c_nums:
        num_common = len(s1_nums & c_nums)
        num_jaccard = num_common / len(s1_nums | c_nums)
    else:
        num_common = 0
        num_jaccard = 0.5
    
    max_name_sim = max(n_token_set, nn_token_set)
    max_addr_sim = a_token_set
    max_any_sim = max(max_name_sim, max_addr_sim)
    min_sim = min(max_name_sim, max_addr_sim) if has_both_addr else max_name_sim
    
    # 3-gram character jaccard
    s1_ngrams = set(s1nn[i:i+3] for i in range(len(s1nn) - 2)) if len(s1nn) >= 3 else set()
    c_ngrams = set(cnn[i:i+3] for i in range(len(cnn) - 2)) if len(cnn) >= 3 else set()
    ngram_jaccard = len(s1_ngrams & c_ngrams) / max(1, len(s1_ngrams | c_ngrams)) if (s1_ngrams or c_ngrams) else 0.0
    
    return [
        n_token_set, n_token_sort, n_ratio, n_partial,
        nn_token_set, nn_token_sort,
        a_token_set, a_token_sort, a_ratio, a_partial,
        has_both_addr,
        num_common, num_jaccard,
        max_name_sim, max_addr_sim, max_any_sim, min_sim,
        ngram_jaccard,
        tfidf_score
    ]

def load_existing_done_ids(match_path):
    """Load IDs that have already been written to matching_results.tsv."""
    if not os.path.exists(match_path):
        return set()
    done = set()
    with open(match_path, 'r', encoding='utf-8') as f:
        header = next(f, None)
        for line in f:
            parts = line.split('\t')
            if parts:
                done.add(parts[0].strip())
    return done

def process_country_partition(country, s1_df, other_df, clf, threshold, cand_file, match_file, done_s1_ids, device):
    print(f"\n{'='*30} Processing Country: {country} {'='*30}", flush=True)
    s1_c = s1_df[s1_df['country'] == country]
    other_c = other_df[other_df['country'] == country]
    
    if len(s1_c) == 0:
        print(f"No S1 records for {country}, skipping.", flush=True)
        return
    
    # Filter out already completed entities if resuming
    if done_s1_ids:
        already_done_mask = s1_c['entity_id'].isin(done_s1_ids)
        n_already = already_done_mask.sum()
        if n_already > 0:
            print(f"  [Resume] Skipping {n_already} already processed {country} records.", flush=True)
            s1_c = s1_c[~already_done_mask]
            if len(s1_c) == 0:
                print(f"  All records for {country} are already completed!", flush=True)
                return

    if len(other_c) == 0:
        print(f"  No S2/S3 candidates found for {country}. Writing empty matches...", flush=True)
        for eid in s1_c['entity_id']:
            cand_file.write(f"{eid}\t\n")
            match_file.write(f"{eid}\t\n")
        cand_file.flush()
        match_file.flush()
        return

    n_s1 = len(s1_c)
    n_other = len(other_c)
    print(f"  Entities: {n_s1:,} queries in S1 vs {n_other:,} candidate pool in S2/S3", flush=True)
    
    # Pre-extract and pre-clean data into fast memory arrays
    print("  Pre-cleaning text fields...", flush=True)
    s1_eids = s1_c['entity_id'].values
    s1_raw_names = s1_c['business_name'].values
    s1_raw_addrs = s1_c['business_address'].values
    
    s1_clean_names = [normalize_text(n) for n in s1_raw_names]
    s1_norm_names = [normalize_name(n) for n in s1_raw_names]
    s1_clean_addrs = [normalize_text(a) for a in s1_raw_addrs]
    s1_num_sets = [extract_numbers(n) | extract_numbers(a) for n, a in zip(s1_clean_names, s1_clean_addrs)]
    s1_texts = [f"{n} {a}".strip() for n, a in zip(s1_clean_names, s1_clean_addrs)]
    
    other_eids = other_c['entity_id'].values
    other_raw_names = other_c['business_name'].values
    other_raw_addrs = other_c['business_address'].values
    
    other_clean_names = [normalize_text(n) for n in other_raw_names]
    other_norm_names = [normalize_name(n) for n in other_raw_names]
    other_clean_addrs = [normalize_text(a) for a in other_raw_addrs]
    other_num_sets = [extract_numbers(n) | extract_numbers(a) for n, a in zip(other_clean_names, other_clean_addrs)]
    other_texts = [f"{n} {a}".strip() for n, a in zip(other_clean_names, other_clean_addrs)]

    print("  Fitting TF-IDF on candidate pool...", flush=True)
    t0 = time.time()
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32
    )
    other_tfidf = tfidf.fit_transform(other_texts)
    print(f"  Candidate TF-IDF shape: {other_tfidf.shape} in {time.time()-t0:.1f}s", flush=True)
    
    print("  Transforming S1 query texts...", flush=True)
    t0 = time.time()
    s1_tfidf = tfidf.transform(s1_texts)
    print(f"  S1 TF-IDF shape: {s1_tfidf.shape} in {time.time()-t0:.1f}s", flush=True)
    
    # Transpose candidate pool for sparse dot-product
    other_tfidf_T = other_tfidf.T.tocsc()
    
    print(f"  Executing high-throughput streaming inference (Chunk Size: {BLOCK_CHUNK_SIZE})...", flush=True)
    t_start = time.time()
    total_processed = 0
    
    # Evaluation buffers for batching LightGBM predict_proba
    buffer_query_indices = []
    buffer_s1_ids = []
    buffer_c_ids = []
    buffer_features = []
    query_candidates_map = {}  # q_idx -> list of top-k c_ids for candidate_pairs.tsv

    def flush_eval_buffer():
        """Evaluate accumulated candidate pairs through LightGBM in one fast batch call."""
        if not buffer_features:
            return
        
        feat_matrix = np.array(buffer_features, dtype=np.float32)
        probs = clf.predict_proba(feat_matrix)[:, 1]
        
        # Group matches by query
        query_matches = defaultdict(list)
        for q_idx, cid, prob in zip(buffer_query_indices, buffer_c_ids, probs):
            if prob >= threshold:
                query_matches[q_idx].append(cid)
        
        # Write batch results to files
        for q_idx, s1_id in enumerate(buffer_batch_s1_ids):
            cands = query_candidates_map.get(q_idx, [])
            cand_str = ",".join(cands) if cands else ""
            cand_file.write(f"{s1_id}\t{cand_str}\n")
            
            matches = query_matches.get(q_idx, [])
            match_str = ",".join(matches) if matches else ""
            match_file.write(f"{s1_id}\t{match_str}\n")
        
        # Reset buffers
        buffer_query_indices.clear()
        buffer_s1_ids.clear()
        buffer_c_ids.clear()
        buffer_features.clear()
        query_candidates_map.clear()

    # Iterate through S1 in chunks
    for c_start in range(0, n_s1, BLOCK_CHUNK_SIZE):
        c_end = min(c_start + BLOCK_CHUNK_SIZE, n_s1)
        sim_slice = s1_tfidf[c_start:c_end].dot(other_tfidf_T)
        
        buffer_batch_s1_ids = s1_eids[c_start:c_end]
        query_candidates_map.clear()
        
        for local_q_idx in range(sim_slice.shape[0]):
            abs_idx = c_start + local_q_idx
            s1_id = s1_eids[abs_idx]
            
            row = sim_slice.getrow(local_q_idx)
            if row.nnz == 0:
                query_candidates_map[local_q_idx] = []
                continue
            
            data = row.data
            indices = row.indices
            
            if len(data) <= TOP_K_CANDIDATES:
                top_order = np.argsort(-data)
            else:
                top_order = np.argpartition(-data, TOP_K_CANDIDATES)[:TOP_K_CANDIDATES]
                top_order = top_order[np.argsort(-data[top_order])]
            
            chosen_c_indices = indices[top_order]
            chosen_scores = data[top_order]
            
            cand_id_list = [other_eids[idx] for idx in chosen_c_indices]
            query_candidates_map[local_q_idx] = cand_id_list
            
            # Query pre-processed fields
            s1n = s1_clean_names[abs_idx]
            s1nn = s1_norm_names[abs_idx]
            s1a = s1_clean_addrs[abs_idx]
            s1_nums = s1_num_sets[abs_idx]
            
            # Generate features directly using fast array indexing
            for cand_pos, (c_idx, score) in enumerate(zip(chosen_c_indices, chosen_scores)):
                feats = compute_pair_features(
                    s1n, s1nn, s1a, s1_nums,
                    other_clean_names[c_idx],
                    other_norm_names[c_idx],
                    other_clean_addrs[c_idx],
                    other_num_sets[c_idx],
                    float(score)
                )
                buffer_query_indices.append(local_q_idx)
                buffer_s1_ids.append(s1_id)
                buffer_c_ids.append(other_eids[c_idx])
                buffer_features.append(feats)
        
        # Batch predict for this slice
        flush_eval_buffer()
        
        total_processed += (c_end - c_start)
        if total_processed % 5000 == 0 or c_end == n_s1:
            elapsed = time.time() - t_start
            qps = total_processed / max(1.0, elapsed)
            rem_sec = (n_s1 - total_processed) / max(1.0, qps)
            cand_file.flush()
            match_file.flush()
            print(f"    Processed {total_processed:,}/{n_s1:,} ({total_processed/n_s1:.1%}) "
                  f"| Speed: {qps:.1f} queries/s | ETA: {rem_sec/60:.1f} min", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Accelerated Colab Test Inference")
    parser.add_argument('--data-dir', default='dataset/test', help='Directory containing test TSVs')
    parser.add_argument('--output-dir', default='output', help='Output directory for TSV results')
    parser.add_argument('--model-path', default='model_lgbm.pkl', help='Path to trained model pickle')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from existing partial outputs')
    args = parser.parse_args()

    # GPU Check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print(f"Loading model weights from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file {args.model_path} not found. Please upload model_lgbm.pkl.")
        sys.exit(1)
        
    with open(args.model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['model']
    threshold = model_data.get('threshold', 0.90)
    print(f"Loaded LightGBM model. Decision threshold: {threshold}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = os.path.join(args.output_dir, 'candidate_pairs.tsv')
    match_path = os.path.join(args.output_dir, 'matching_results.tsv')
    
    done_s1_ids = set()
    if args.resume and os.path.exists(match_path):
        done_s1_ids = load_existing_done_ids(match_path)
        print(f"Found existing matching_results.tsv with {len(done_s1_ids):,} completed entities.")
        cand_mode = 'a'
        match_mode = 'a'
    else:
        cand_mode = 'w'
        match_mode = 'w'
    
    cand_file = open(cand_path, cand_mode, encoding='utf-8')
    match_file = open(match_path, match_mode, encoding='utf-8')
    
    if cand_mode == 'w':
        cand_file.write("source1_entity_id\tcandidate_entity_ids\n")
        match_file.write("source1_entity_id\tmatched_entity_ids\n")
        cand_file.flush()
        match_file.flush()

    print(f"\nLoading test sets from {args.data_dir}...")
    s1_path = os.path.join(args.data_dir, 'test_source1.tsv')
    s2_path = os.path.join(args.data_dir, 'test_source2.tsv')
    s3_path = os.path.join(args.data_dir, 'test_source3.tsv')
    
    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            print(f"ERROR: File {p} not found! Check --data-dir.")
            sys.exit(1)

    s1_df = pd.read_csv(s1_path, sep='\t')
    print(f"Loaded S1: {len(s1_df):,} records")
    s2_df = pd.read_csv(s2_path, sep='\t')
    s3_df = pd.read_csv(s3_path, sep='\t')
    other_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"Loaded S2+S3 candidate pool: {len(other_df):,} unique records")
    
    # Process partitions in order
    countries = ['France', 'US', 'India']
    for country in countries:
        process_country_partition(
            country, s1_df, other_df, clf, threshold,
            cand_file, match_file, done_s1_ids, device
        )

    cand_file.close()
    match_file.close()
    
    print("\n" + "="*50)
    print("Inference successfully finished!")
    print(f"Results saved to:")
    print(f"  - {cand_path}")
    print(f"  - {match_path}")
    print("="*50)

if __name__ == '__main__':
    main()
