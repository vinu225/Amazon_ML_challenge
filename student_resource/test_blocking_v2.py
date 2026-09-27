import sys
import re
import time
import pandas as pd
from collections import defaultdict
import numpy as np
import anyascii

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'co', 'company', 'enterprises', 'enterprise',
    'services', 'solutions', 'holdings', 'group', 'the', 'and', 'center',
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'sa', 'cie'
}

CLEAN_RE = re.compile(r'[^a-z0-9\s]')
DIGIT_RE = re.compile(r'\d+')

def extract_blocking_keys(name, addr):
    keys = set()
    s_name = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name)
    s_addr = "" if addr is None or (isinstance(addr, float) and np.isnan(addr)) else str(addr)
    
    # Transliterate everything to ASCII lowercase
    name_ascii = anyascii.anyascii(s_name).lower()
    addr_ascii = anyascii.anyascii(s_addr).lower()
    
    # Clean domains
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

print("Loading 20,000 train GT rows...", flush=True)
gt_sample = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)
target_s1 = set(gt_sample['source1_entity_id'])

gt_map = {}
for r in gt_sample.itertuples(index=False):
    m_val = getattr(r, 'matched_entity_ids')
    if pd.notna(m_val):
        m_set = {x.strip() for x in str(m_val).split(',') if x.strip()}
    else:
        m_set = set()
    gt_map[getattr(r, 'source1_entity_id')] = m_set

# Load S1
s1_rows = []
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=100000):
    sub = chunk[chunk['entity_id'].isin(target_s1)]
    if len(sub) > 0:
        s1_rows.append(sub)
    if len(pd.concat(s1_rows, ignore_index=True)) == len(target_s1):
        break
s1_df = pd.concat(s1_rows, ignore_index=True)
print(f"Loaded S1 sample: {len(s1_df)} records", flush=True)

# Sample S2 and S3 (500k each)
print("Loading S2 and S3 sample (500k each)...", flush=True)
t0 = time.time()
s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=500000)
s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=500000)
other_df = pd.concat([s2_df, s3_df], ignore_index=True)
print(f"Loaded {len(other_df)} records in {time.time()-t0:.1f}s", flush=True)

# Build inverted index
print("Building inverted index...", flush=True)
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
    k_set = extract_blocking_keys(other_names[i], other_addrs[i])
    records_keys.append((c, k_set))
    for k in k_set:
        key_counts[(c, k)] += 1

MAX_POSTINGS = 250
for i in range(len(other_df)):
    e_id = other_eids[i]
    c, k_set = records_keys[i]
    for k in k_set:
        ck = (c, k)
        if key_counts[ck] <= MAX_POSTINGS:
            index[ck].append(e_id)

print(f"Built index with {len(index)} distinct keys in {time.time()-t0:.1f}s", flush=True)

# Query S1 against index
print("Querying S1 records against index...", flush=True)
t0 = time.time()
recalled_links = 0
total_candidates = 0
candidates_per_s1 = []

available_s2_s3 = set(other_eids)
evaluated_true_links = 0

s1_eids = s1_df['entity_id'].values
s1_countries = s1_df['country'].values
s1_names = s1_df['business_name'].values
s1_addrs = s1_df['business_address'].values

for i in range(len(s1_df)):
    s1_id = s1_eids[i]
    c = str(s1_countries[i])
    true_matches = gt_map.get(s1_id, set()) & available_s2_s3
    evaluated_true_links += len(true_matches)
    
    k_set = extract_blocking_keys(s1_names[i], s1_addrs[i])
    cands = set()
    for k in k_set:
        ck = (c, k)
        if ck in index:
            cands.update(index[ck])
            if len(cands) > 100:
                break
                
    total_candidates += len(cands)
    candidates_per_s1.append(len(cands))
    recalled_links += len(true_matches & cands)

query_time = time.time() - t0
print(f"Queried {len(s1_df)} S1 records in {query_time:.2f}s ({len(s1_df)/query_time:.0f} recs/s)", flush=True)
print(f"Evaluated true links present in sample: {evaluated_true_links}", flush=True)
print(f"Recalled true links: {recalled_links} ({recalled_links/max(1,evaluated_true_links):.2%})", flush=True)
print(f"Average candidates per S1: {np.mean(candidates_per_s1):.1f} (median: {np.median(candidates_per_s1)})", flush=True)
