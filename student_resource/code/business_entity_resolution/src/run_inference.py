"""
Memory-Optimized Full Test Inference Pipeline for Amazon ML Challenge 2026
Streams both:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv
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

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TOP_K_CANDIDATES = 50
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MAX_FEATURES = 150000
BLOCK_CHUNK_SIZE = 100       # 100 S1 queries at a time for sparse dot product

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
    if pd.isna(s):
        return set()
    return set(DIGIT_RE.findall(str(s)))

def compute_pair_features(s1_name, s1_addr, c_name, c_addr, tfidf_score=0.0):
    s1n = normalize_text(s1_name)
    cn = normalize_text(c_name)
    s1a = normalize_text(s1_addr)
    ca = normalize_text(c_addr)
    
    n_token_set = fuzz.token_set_ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_token_sort = fuzz.token_sort_ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_ratio = fuzz.ratio(s1n, cn, processor=fuzz_utils.default_process)
    n_partial = fuzz.partial_ratio(s1n, cn, processor=fuzz_utils.default_process)
    
    s1nn = normalize_name(s1_name)
    cnn = normalize_name(c_name)
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
    
    s1_nums = extract_numbers(s1_name) | extract_numbers(s1_addr)
    c_nums = extract_numbers(c_name) | extract_numbers(c_addr)
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
    
    def char_ngrams(s, n=3):
        return set(s[i:i+n] for i in range(len(s) - n + 1)) if len(s) >= n else set()
    
    s1_ngrams = char_ngrams(s1nn, 3)
    c_ngrams = char_ngrams(cnn, 3)
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

def process_country_stream(country, s1_df, other_df, clf, threshold, cand_file, match_file):
    print(f"\n================ Processing country: {country} ================", flush=True)
    s1_c = s1_df[s1_df['country'] == country]
    other_c = other_df[other_df['country'] == country]
    
    if len(s1_c) == 0:
        return
    
    if len(other_c) == 0:
        for eid in s1_c['entity_id']:
            cand_file.write(f"{eid}\t\n")
            match_file.write(f"{eid}\t\n")
        return
        
    print(f"  Preparing texts for {len(s1_c)} S1 vs {len(other_c)} S2/S3...", flush=True)
    s1_names = s1_c['business_name'].values
    s1_addrs = s1_c['business_address'].values
    s1_eids = s1_c['entity_id'].values
    
    other_names = other_c['business_name'].values
    other_addrs = other_c['business_address'].values
    other_eids = other_c['entity_id'].values
    
    other_c_map = other_c.set_index('entity_id')
    
    s1_texts = [make_tfidf_text(n, a) for n, a in zip(s1_names, s1_addrs)]
    other_texts = [make_tfidf_text(n, a) for n, a in zip(other_names, other_addrs)]
    
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
    
    print("  Transforming S1 queries...", flush=True)
    t0 = time.time()
    s1_tfidf = tfidf.transform(s1_texts)
    print(f"  S1 TF-IDF shape: {s1_tfidf.shape} in {time.time()-t0:.1f}s", flush=True)
    
    n_queries = len(s1_eids)
    print(f"  Streaming inference in chunks of {BLOCK_CHUNK_SIZE} queries...", flush=True)
    t_start = time.time()
    
    for c_start in range(0, n_queries, BLOCK_CHUNK_SIZE):
        c_end = min(c_start + BLOCK_CHUNK_SIZE, n_queries)
        
        # Multiply only the small slice with the full candidates matrix
        sim_slice = s1_tfidf[c_start:c_end] @ other_tfidf.T
        
        for q_idx in range(sim_slice.shape[0]):
            abs_idx = c_start + q_idx
            s1_id = s1_eids[abs_idx]
            s1_n = s1_names[abs_idx]
            s1_a = s1_addrs[abs_idx]
            
            row = sim_slice.getrow(q_idx)
            if row.nnz == 0:
                cand_file.write(f"{s1_id}\t\n")
                match_file.write(f"{s1_id}\t\n")
                continue
                
            data = row.data
            indices = row.indices
            
            if len(data) <= TOP_K_CANDIDATES:
                top_idx = np.argsort(-data)
            else:
                top_idx = np.argpartition(-data, TOP_K_CANDIDATES)[:TOP_K_CANDIDATES]
                top_idx = top_idx[np.argsort(-data[top_idx])]
                
            c_ids = [other_eids[indices[j]] for j in top_idx]
            c_scores = [float(data[j]) for j in top_idx]
            
            # Write candidate_pairs
            cand_file.write(f"{s1_id}\t{','.join(c_ids)}\n")
            
            # Feature extraction for candidates
            batch_feats = []
            valid_cids = []
            for cid, sc in zip(c_ids, c_scores):
                if cid in other_c_map.index:
                    crow = other_c_map.loc[cid]
                    feats = compute_pair_features(s1_n, s1_a, crow['business_name'], crow['business_address'], sc)
                    batch_feats.append(feats)
                    valid_cids.append(cid)
                    
            if not batch_feats:
                match_file.write(f"{s1_id}\t\n")
                continue
                
            probs = clf.predict_proba(np.array(batch_feats, dtype=np.float32))[:, 1]
            matched = [cid for cid, prob in zip(valid_cids, probs) if prob >= threshold]
            match_file.write(f"{s1_id}\t{','.join(matched)}\n")
            
        if (c_end) % 10000 == 0 or c_end == n_queries:
            elapsed = time.time() - t_start
            rate = c_end / max(1.0, elapsed)
            print(f"    Completed {c_end}/{n_queries} ({c_end/n_queries:.1%}) [{rate:.1f} queries/s]", flush=True)

def run_test_pipeline(test_dir, output_dir, model_path='model_lgbm.pkl'):
    print(f"Loading trained model from {model_path}...", flush=True)
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['model']
    threshold = model_data.get('threshold', 0.90)
    print(f"Loaded LightGBM model. Using decision threshold: {threshold}", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    match_path = os.path.join(output_dir, 'matching_results.tsv')
    
    print(f"Reading test datasets from {test_dir}...", flush=True)
    s1_df = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t')
    print(f"Loaded test S1: {len(s1_df)} records", flush=True)
    
    s2_df = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t')
    s3_df = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t')
    other_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"Loaded test S2+S3: {len(other_df)} records", flush=True)
    
    # Process smaller partitions first
    countries = ['France', 'US', 'India']
    print(f"Country processing sequence: {countries}", flush=True)
    
    cand_file = open(cand_path, 'w', encoding='utf-8')
    match_file = open(match_path, 'w', encoding='utf-8')
    
    cand_file.write("source1_entity_id\tcandidate_entity_ids\n")
    match_file.write("source1_entity_id\tmatched_entity_ids\n")
    
    for country in countries:
        process_country_stream(country, s1_df, other_df, clf, threshold, cand_file, match_file)
        
    cand_file.close()
    match_file.close()
    print("\nInference complete! Outputs saved to:", flush=True)
    print(f"  - {cand_path}", flush=True)
    print(f"  - {match_path}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='dataset/test', help='Path to test data directory')
    parser.add_argument('--output-dir', default='output', help='Path to output directory')
    parser.add_argument('--model-path', default='model_lgbm.pkl', help='Path to trained model pickle')
    args = parser.parse_args()
    
    run_test_pipeline(args.data_dir, args.output_dir, args.model_path)
