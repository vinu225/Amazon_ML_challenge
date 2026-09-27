import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=50)
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

total_pairs = 0
name_exact = 0
name_sub = 0
addr_exact = 0
addr_sub = 0
has_num = 0
num_match = 0
token_overlap_zero = 0

def clean(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    return re.sub(r'[^a-z0-9]', ' ', s).strip()

def get_tokens(s):
    return set(clean(s).split())

for idx, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_df.index:
        continue
    s1_row = s1_df.loc[s1_id]
    s1_name = str(s1_row['business_name'])
    s1_addr = str(s1_row['business_address'])
    s1_toks = get_tokens(s1_name + " " + s1_addr)
    s1_nums = set(re.findall(r'\d+', s1_name + " " + s1_addr))
    
    matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
    for m in matches:
        m = m.strip()
        if not m or m not in all_other.index:
            continue
        m_row = all_other.loc[m]
        m_name = str(m_row['business_name'])
        m_addr = str(m_row['business_address'])
        m_toks = get_tokens(m_name + " " + m_addr)
        m_nums = set(re.findall(r'\d+', m_name + " " + m_addr))
        
        total_pairs += 1
        if clean(s1_name) == clean(m_name):
            name_exact += 1
        if clean(s1_addr) == clean(m_addr) and len(clean(s1_addr)) > 3:
            addr_exact += 1
        overlap = s1_toks & m_toks
        # remove single character tokens and trivial words
        overlap = {w for w in overlap if len(w) > 2 and w not in {'inc', 'llc', 'ltd', 'pvt', 'the', 'and'}}
        if len(overlap) == 0:
            token_overlap_zero += 1
            print(f"NO OVERLAP PAIR:")
            print(f"  S1: {s1_name} | {s1_addr}")
            print(f"  M:  {m_name} | {m_addr}")
        
print("="*50)
print(f"Total pairs checked: {total_pairs}")
print(f"Name exact clean match: {name_exact}/{total_pairs} ({name_exact/max(1,total_pairs):.2%})")
print(f"Addr exact clean match: {addr_exact}/{total_pairs} ({addr_exact/max(1,total_pairs):.2%})")
print(f"Zero token overlap: {token_overlap_zero}/{total_pairs} ({token_overlap_zero/max(1,total_pairs):.2%})")
