import sys
import pandas as pd
from rapidfuzz import fuzz, utils
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print('Reading 500 GT rows...')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=500)
target_s1 = set(gt['source1_entity_id'])
target_s2_s3 = set()
for m in gt['matched_entity_ids']:
    if pd.notna(m):
        for id_ in str(m).split(','):
            if id_.strip():
                target_s2_s3.add(id_.strip())

s1_matches = []
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
    sub = chunk[chunk['entity_id'].isin(target_s1)]
    if len(sub) > 0:
        s1_matches.append(sub)
        if len(pd.concat(s1_matches)) == len(target_s1):
            break
s1_df = pd.concat(s1_matches).set_index('entity_id')

s2_targets = {x for x in target_s2_s3 if x.startswith('S2-')}
s3_targets = {x for x in target_s2_s3 if x.startswith('S3-')}

s2_matches = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=200000):
    sub = chunk[chunk['entity_id'].isin(s2_targets)]
    if len(sub) > 0:
        s2_matches.append(sub)
        if sum(len(x) for x in s2_matches) == len(s2_targets):
            break
s2_df = pd.concat(s2_matches) if s2_matches else pd.DataFrame()

s3_matches = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=200000):
    sub = chunk[chunk['entity_id'].isin(s3_targets)]
    if len(sub) > 0:
        s3_matches.append(sub)
        if sum(len(x) for x in s3_matches) == len(s3_targets):
            break
s3_df = pd.concat(s3_matches) if s3_matches else pd.DataFrame()

all_other = pd.concat([s2_df, s3_df]).set_index('entity_id')

name_ratios = []
addr_ratios = []
max_ratios = []
low_max_pairs = []

for idx, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_df.index:
        continue
    s1_row = s1_df.loc[s1_id]
    s1_name = str(s1_row['business_name']) if pd.notna(s1_row['business_name']) else ''
    s1_addr = str(s1_row['business_address']) if pd.notna(s1_row['business_address']) else ''
    
    matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
    for m in matches:
        m = m.strip()
        if not m or m not in all_other.index:
            continue
        m_row = all_other.loc[m]
        m_name = str(m_row['business_name']) if pd.notna(m_row['business_name']) else ''
        m_addr = str(m_row['business_address']) if pd.notna(m_row['business_address']) else ''
        
        n_rat = fuzz.token_set_ratio(s1_name, m_name, processor=utils.default_process)
        a_rat = fuzz.token_set_ratio(s1_addr, m_addr, processor=utils.default_process)
        mx = max(n_rat, a_rat)
        
        name_ratios.append(n_rat)
        addr_ratios.append(a_rat)
        max_ratios.append(mx)
        if mx < 60:
            low_max_pairs.append((s1_name, s1_addr, m_name, m_addr, n_rat, a_rat))

print(f"Total positive pairs evaluated: {len(max_ratios)}")
print(f"Name token_set_ratio: mean={np.mean(name_ratios):.1f}, median={np.median(name_ratios):.1f}, min={np.min(name_ratios):.1f}")
print(f"Addr token_set_ratio: mean={np.mean(addr_ratios):.1f}, median={np.median(addr_ratios):.1f}, min={np.min(addr_ratios):.1f}")
print(f"MAX(Name, Addr) ratio: mean={np.mean(max_ratios):.1f}, median={np.median(max_ratios):.1f}, min={np.min(max_ratios):.1f}")
print(f"Count with max_ratio >= 75: {sum(x >= 75 for x in max_ratios)}/{len(max_ratios)} ({np.mean([x >= 75 for x in max_ratios]):.2%})")
print(f"Count with max_ratio >= 70: {sum(x >= 70 for x in max_ratios)}/{len(max_ratios)} ({np.mean([x >= 70 for x in max_ratios]):.2%})")
print(f"Count with max_ratio >= 60: {sum(x >= 60 for x in max_ratios)}/{len(max_ratios)} ({np.mean([x >= 60 for x in max_ratios]):.2%})")
print(f"Count with max_ratio >= 50: {sum(x >= 50 for x in max_ratios)}/{len(max_ratios)} ({np.mean([x >= 50 for x in max_ratios]):.2%})")

print(f"\n--- Low max pairs (mx < 60): {len(low_max_pairs)} ---")
for p in low_max_pairs[:10]:
    print(f"S1: {p[0]} | {p[1]}")
    print(f"M:  {p[2]} | {p[3]}")
    print(f"    n_rat={p[4]:.1f}, a_rat={p[5]:.1f}")
    print()
