import sys
import re
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import anyascii
from rapidfuzz import fuzz, utils
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=== Starting End-to-End ER Pipeline Validation ===", flush=True)

# 1. Load sample of training data (e.g. 30,000 S1 entities and corresponding S2/S3)
GT_PATH = 'dataset/train/train_ground_truth.tsv'
S1_PATH = 'dataset/train/train_source1.tsv'
S2_PATH = 'dataset/train/train_source2.tsv'
S3_PATH = 'dataset/train/train_source3.tsv'

N_S1_TRAIN = 25000

print(f"Reading {N_S1_TRAIN} Ground Truth records...", flush=True)
gt_df = pd.read_csv(GT_PATH, sep='\t', nrows=N_S1_TRAIN)
target_s1 = set(gt_df['source1_entity_id'])

# Build GT dictionary: s1_id -> set of true matched s2/s3 ids
gt_map = {}
all_true_matches = set()
for r in gt_df.itertuples(index=False):
    m_val = getattr(r, 'matched_entity_ids')
    if pd.notna(m_val) and str(m_val).strip():
        m_set = {x.strip() for x in str(m_val).split(',') if x.strip()}
    else:
        m_set = set()
    gt_map[getattr(r, 'source1_entity_id')] = m_set
    all_true_matches.update(m_set)

print(f"Loaded {len(gt_map)} S1 GT records with {sum(len(v) for v in gt_map.values())} true links.", flush=True)

# Load S1
print("Reading S1 records...", flush=True)
s1_rows = []
for chunk in pd.read_csv(S1_PATH, sep='\t', chunksize=100000):
    sub = chunk[chunk['entity_id'].isin(target_s1)]
    if len(sub) > 0:
        s1_rows.append(sub)
    if len(pd.concat(s1_rows, ignore_index=True)) == len(target_s1):
        break
s1_df = pd.concat(s1_rows, ignore_index=True)
print(f"Loaded {len(s1_df)} S1 records.", flush=True)

# Load S2 and S3: include all true matches plus random sample to test realistic distractor ratio
target_s2 = {x for x in all_true_matches if x.startswith('S2-')}
target_s3 = {x for x in all_true_matches if x.startswith('S3-')}

print(f"Reading S2 records (targets: {len(target_s2)} + distractors)...", flush=True)
s2_rows = []
total_s2_read = 0
for chunk in pd.read_csv(S2_PATH, sep='\t', chunksize=200000):
    # Grab targets and some non-targets
    sub_targets = chunk[chunk['entity_id'].isin(target_s2)]
    sub_distractors = chunk[~chunk['entity_id'].isin(target_s2)].sample(frac=0.04, random_state=42)
    s2_rows.append(pd.concat([sub_targets, sub_distractors]))
    total_s2_read += len(chunk)
    if total_s2_read >= 1000000:
        break
s2_df = pd.concat(s2_rows, ignore_index=True)

print(f"Reading S3 records (targets: {len(target_s3)} + distractors)...", flush=True)
s3_rows = []
total_s3_read = 0
for chunk in pd.read_csv(S3_PATH, sep='\t', chunksize=200000):
    sub_targets = chunk[chunk['entity_id'].isin(target_s3)]
    sub_distractors = chunk[~chunk['entity_id'].isin(target_s3)].sample(frac=0.04, random_state=42)
    s3_rows.append(pd.concat([sub_targets, sub_distractors]))
    total_s3_read += len(chunk)
    if total_s3_read >= 1000000:
        break
s3_df = pd.concat(s3_rows, ignore_index=True)

other_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
print(f"Total S2+S3 universe in validation: {len(other_df)} records.", flush=True)

# Split S1 into Train (70%) and Val (30%)
train_s1_df, val_s1_df = train_test_split(s1_df, test_size=0.3, random_state=42)
print(f"S1 split: Train={len(train_s1_df)}, Val={len(val_s1_df)}", flush=True)

# 2. Advanced Multi-Key Blocking
LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'services', 'solutions', 'holdings', 'group', 'the', 'and', 'center',
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'cie'
}

CLEAN_RE = re.compile(r'[^a-z0-9\s]')
DIGIT_RE = re.compile(r'\d+')

def extract_keys(name, addr):
    keys = set()
    s_name = "" if pd.isna(name) else str(name)
    s_addr = "" if pd.isna(addr) else str(addr)
    
    name_ascii = anyascii.anyascii(s_name).lower()
    addr_ascii = anyascii.anyascii(s_addr).lower()
    
    # Strip common web extensions
    name_ascii = re.sub(r'\.(com|net|org|in|co|us|fr)\b', '', name_ascii)
    
    name_clean = CLEAN_RE.sub(' ', name_ascii)
    addr_clean = CLEAN_RE.sub(' ', addr_ascii)
    
    name_words = [w for w in name_clean.split() if w not in LEGAL_STOPWORDS and len(w) > 1]
    if name_words:
        keys.add(('core_name', " ".join(sorted(name_words))))
        if len(name_words) >= 2:
            keys.add(('name_prefix_2', f"{name_words[0]}_{name_words[1]}"))
        for w in name_words:
            if len(w) >= 3:
                keys.add(('name_word', w))
                
    addr_words = [w for w in addr_clean.split() if len(w) > 1]
    raw_nums = DIGIT_RE.findall(addr_ascii) + DIGIT_RE.findall(name_ascii)
    nums = [str(int(x)) for x in raw_nums if len(x) <= 8]
    non_nums = [w for w in addr_words if not w.isdigit() and len(w) >= 3]
    
    for num in nums[:3]:
        for w in non_nums[:3]:
            keys.add(('num_word', f"{num}_{w}"))
        keys.add(('num', num))
            
    for w in non_nums:
        if len(w) >= 4:
            keys.add(('addr_word', w))
            
    return keys

print("Indexing S2+S3 universe...", flush=True)
t0 = time.time()
index = defaultdict(list)
key_counts = defaultdict(int)

other_eids = other_df['entity_id'].values
other_countries = other_df['country'].values
other_names = other_df['business_name'].values
other_addrs = other_df['business_address'].values

records_keys = []
for i in range(len(other_df)):
    c = str(other_countries[i])
    k_set = extract_keys(other_names[i], other_addrs[i])
    records_keys.append((c, k_set))
    for k in k_set:
        key_counts[(c, k)] += 1

MAX_POSTINGS = 300
for i in range(len(other_df)):
    e_id = other_eids[i]
    c, k_set = records_keys[i]
    for k in k_set:
        ck = (c, k)
        if key_counts[ck] <= MAX_POSTINGS:
            index[ck].append(e_id)

print(f"Index built in {time.time()-t0:.1f}s with {len(index)} distinct keys.", flush=True)

# 3. Candidate Generation and Feature Extraction Function
other_record_map = other_df.set_index('entity_id')

def get_features_for_pair(s1_row, cand_id, shared_keys):
    cand_row = other_record_map.loc[cand_id]
    
    s1_name = "" if pd.isna(s1_row['business_name']) else str(s1_row['business_name'])
    s1_addr = "" if pd.isna(s1_row['business_address']) else str(s1_row['business_address'])
    cand_name = "" if pd.isna(cand_row['business_name']) else str(cand_row['business_name'])
    cand_addr = "" if pd.isna(cand_row['business_address']) else str(cand_row['business_address'])
    
    # Transliterated versions
    s1_name_asc = anyascii.anyascii(s1_name).lower()
    cand_name_asc = anyascii.anyascii(cand_name).lower()
    s1_addr_asc = anyascii.anyascii(s1_addr).lower()
    cand_addr_asc = anyascii.anyascii(cand_addr).lower()
    
    n_set = fuzz.token_set_ratio(s1_name_asc, cand_name_asc, processor=utils.default_process)
    n_sort = fuzz.token_sort_ratio(s1_name_asc, cand_name_asc, processor=utils.default_process)
    n_ratio = fuzz.ratio(s1_name_asc, cand_name_asc, processor=utils.default_process)
    
    a_set = fuzz.token_set_ratio(s1_addr_asc, cand_addr_asc, processor=utils.default_process) if (s1_addr and cand_addr) else 0.0
    a_sort = fuzz.token_sort_ratio(s1_addr_asc, cand_addr_asc, processor=utils.default_process) if (s1_addr and cand_addr) else 0.0
    a_ratio = fuzz.ratio(s1_addr_asc, cand_addr_asc, processor=utils.default_process) if (s1_addr and cand_addr) else 0.0
    
    s1_nums = set(DIGIT_RE.findall(s1_addr_asc) + DIGIT_RE.findall(s1_name_asc))
    cand_nums = set(DIGIT_RE.findall(cand_addr_asc) + DIGIT_RE.findall(cand_name_asc))
    num_common = len(s1_nums & cand_nums)
    num_jaccard = num_common / max(1, len(s1_nums | cand_nums)) if (s1_nums or cand_nums) else 0.5
    
    has_addr_both = 1.0 if (s1_addr and cand_addr) else 0.0
    max_sim = max(n_set, a_set)
    min_sim = min(n_set, a_set)
    
    return [
        n_set, n_sort, n_ratio,
        a_set, a_sort, a_ratio,
        num_common, num_jaccard,
        has_addr_both, max_sim, min_sim,
        shared_keys
    ]

# Generate training pairs
print("Generating candidate pairs and features for Training set...", flush=True)
t0 = time.time()
X_train = []
y_train = []

for r in train_s1_df.itertuples(index=False):
    s1_id = getattr(r, 'entity_id')
    c = str(getattr(r, 'country'))
    true_set = gt_map.get(s1_id, set())
    
    k_set = extract_keys(getattr(r, 'business_name'), getattr(r, 'business_address'))
    cand_counts = Counter()
    for k in k_set:
        ck = (c, k)
        if ck in index:
            for eid in index[ck]:
                cand_counts[eid] += 1
                
    # Always include true matches if they are in the available other_df
    # and top 20 candidates
    top_cands = [eid for eid, cnt in cand_counts.most_common(20)]
    for tm in true_set:
        if tm in other_record_map.index and tm not in cand_counts:
            top_cands.append(tm)
            cand_counts[tm] = 1
            
    s1_row_dict = {
        'business_name': getattr(r, 'business_name'),
        'business_address': getattr(r, 'business_address')
    }
    
    for cand_id in top_cands:
        feats = get_features_for_pair(s1_row_dict, cand_id, cand_counts.get(cand_id, 1))
        label = 1 if cand_id in true_set else 0
        X_train.append(feats)
        y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"X_train shape: {X_train.shape}, positive fraction: {np.mean(y_train):.2%}, generated in {time.time()-t0:.1f}s", flush=True)

# Train LightGBM Model
print("Training LightGBM classifier...", flush=True)
clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
print("LightGBM model trained successfully!", flush=True)

# Evaluate on Validation S1 entities
print("Evaluating on Validation S1 entities...", flush=True)
t0 = time.time()

FEATURE_NAMES = [
    'n_set', 'n_sort', 'n_ratio',
    'a_set', 'a_sort', 'a_ratio',
    'num_common', 'num_jaccard',
    'has_addr_both', 'max_sim', 'min_sim',
    'shared_keys'
]

# Run prediction for val entities
val_predictions = {}
val_candidates_dict = {}

for r in val_s1_df.itertuples(index=False):
    s1_id = getattr(r, 'entity_id')
    c = str(getattr(r, 'country'))
    
    k_set = extract_keys(getattr(r, 'business_name'), getattr(r, 'business_address'))
    cand_counts = Counter()
    for k in k_set:
        ck = (c, k)
        if ck in index:
            for eid in index[ck]:
                cand_counts[eid] += 1
                
    top_cands = [eid for eid, cnt in cand_counts.most_common(25)]
    val_candidates_dict[s1_id] = top_cands
    
    if not top_cands:
        val_predictions[s1_id] = []
        continue
        
    s1_row_dict = {
        'business_name': getattr(r, 'business_name'),
        'business_address': getattr(r, 'business_address')
    }
    
    cand_feats = [
        get_features_for_pair(s1_row_dict, cid, cand_counts.get(cid, 1))
        for cid in top_cands
    ]
    
    probs = clf.predict_proba(cand_feats)[:, 1]
    
    # Store candidates with their probabilities
    cands_with_prob = [(cid, prob) for cid, prob in zip(top_cands, probs)]
    val_predictions[s1_id] = cands_with_prob

print(f"Validation inference completed in {time.time()-t0:.1f}s", flush=True)

# Macro F0.5 scoring function
def compute_macro_f05(preds_dict, ground_truth):
    f05_scores = []
    for s1_id, pred_ids in preds_dict.items():
        true_ids = ground_truth.get(s1_id, set())
        pred_set = set(pred_ids)
        
        # Singletons: both empty -> 1.0, one empty one not -> 0.0
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

# Sweep probability thresholds to find optimal F0.5
print("\nSweeping threshold for F0.5 optimization:")
best_thresh = 0.5
best_score = 0.0

for thresh in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95]:
    # Apply 1-to-many constraint:
    # An S2/S3 entity can only be assigned to ONE S1 entity (the one with highest prob)
    pair_matches = []
    for s1_id, cands_prob in val_predictions.items():
        for cid, prob in cands_prob:
            if prob >= thresh:
                pair_matches.append((prob, s1_id, cid))
                
    # Sort pairs by prob descending
    pair_matches.sort(reverse=True, key=lambda x: x[0])
    
    assigned_cand = set()
    final_preds = {s1_id: [] for s1_id in val_predictions}
    for prob, s1_id, cid in pair_matches:
        if cid not in assigned_cand:
            assigned_cand.add(cid)
            final_preds[s1_id].append(cid)
            
    score = compute_macro_f05(final_preds, gt_map)
    print(f"Threshold {thresh:.2f} -> Macro F0.5 = {score:.4f}", flush=True)
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\n>>> BEST VALIDATION MACRO F0.5 SCORE: {best_score:.4f} at threshold {best_thresh:.2f} <<<", flush=True)
