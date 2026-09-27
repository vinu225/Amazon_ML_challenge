import sys
import re
import pandas as pd
from collections import Counter
import anyascii
from test_blocking_v2 import extract_blocking_keys, s1_df, other_df, gt_map, available_s2_s3, index

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

other_map = other_df.set_index('entity_id')

s1_eids = s1_df['entity_id'].values
s1_countries = s1_df['country'].values
s1_names = s1_df['business_name'].values
s1_addrs = s1_df['business_address'].values

missed_count = 0
found_count = 0

for i in range(len(s1_df)):
    s1_id = s1_eids[i]
    c = str(s1_countries[i])
    true_matches = gt_map.get(s1_id, set()) & available_s2_s3
    if not true_matches:
        continue
    
    k_set = extract_blocking_keys(s1_names[i], s1_addrs[i])
    cand_scores = Counter()
    for k in k_set:
        ck = (c, k)
        if ck in index:
            for eid in index[ck]:
                cand_scores[eid] += 1
                
    # Check if true matches are in cand_scores AT ALL
    for tm in true_matches:
        if tm in cand_scores:
            found_count += 1
        else:
            missed_count += 1
            if missed_count <= 5:
                tm_row = other_map.loc[tm]
                print(f"=== MISSED #{missed_count} ===")
                print(f"S1: {s1_names[i]} | {s1_addrs[i]}")
                print(f"TM: {tm_row['business_name']} | {tm_row['business_address']}")
                print("S1 keys:", sorted(list(k_set))[:8])
                print("TM keys:", sorted(list(extract_blocking_keys(tm_row['business_name'], tm_row['business_address'])))[:8])
                print()

total = found_count + missed_count
print(f"TOTAL: Found in index: {found_count}/{total} ({found_count/max(1,total):.2%}) | Missed from index: {missed_count}/{total}")
