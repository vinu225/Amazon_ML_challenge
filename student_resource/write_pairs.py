import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=30)
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

with open('scratch_pair_inspect.txt', 'w', encoding='utf-8') as out:
    for idx, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if s1_id not in s1_df.index:
            continue
        s1_row = s1_df.loc[s1_id]
        out.write(f"=== S1: {s1_id} [{s1_row['country']}] ===\n")
        out.write(f"  NAME:    {s1_row['business_name']}\n")
        out.write(f"  ADDRESS: {s1_row['business_address']}\n")
        matches = str(row['matched_entity_ids']).split(',') if pd.notna(row['matched_entity_ids']) else []
        for m in matches:
            m = m.strip()
            if m in all_other.index:
                m_row = all_other.loc[m]
                out.write(f"  MATCH {m} [{m_row['country']}]:\n")
                out.write(f"    NAME:    {m_row['business_name']}\n")
                out.write(f"    ADDRESS: {m_row['business_address']}\n")
        out.write("\n")

print("Wrote scratch_pair_inspect.txt successfully!")
