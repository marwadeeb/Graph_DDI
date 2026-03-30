"""
step2_dedup_interactions.py — deduplicate DDI pairs under the assumption that
A interacts with B === B interacts with A.

For each pair (drugbank_id, interacting_drugbank_id), the canonical form is
(min_id, max_id) so we keep only one row per unordered pair.
When both directions exist with different descriptions, we concatenate them
with " | " so no information is lost.

Input:  data/step1_full/drug_interactions.csv   (2,911,156 directed rows)
Output: data/step2_dedup/drug_interactions_dedup.csv

Usage:
    python step2_dedup_interactions.py
"""
import os
import time
import pandas as pd

WORKING_DIR  = r"D:\DDI\drugbank_all_full_database.xml"
INPUT_CSV    = os.path.join(WORKING_DIR, "data", "step1_full", "drug_interactions.csv")
OUTPUT_DIR   = os.path.join(WORKING_DIR, "data", "step2_dedup")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "drug_interactions_dedup.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print(f"[step2] Reading {INPUT_CSV} ...")
    t0 = time.time()

    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    print(f"  Loaded {len(df):,} directed rows in {time.time()-t0:.1f}s")

    # Build canonical (drug_a, drug_b) where drug_a <= drug_b lexicographically
    df["drug_a"] = df[["drugbank_id", "interacting_drugbank_id"]].min(axis=1)
    df["drug_b"] = df[["drugbank_id", "interacting_drugbank_id"]].max(axis=1)

    # Group by canonical pair; merge descriptions from both directions
    # (most pairs have identical descriptions in both directions, but not all)
    def merge_descriptions(series):
        unique = series[series != ""].unique()
        if len(unique) == 0:
            return ""
        if len(unique) == 1:
            return unique[0]
        return " | ".join(unique)

    print("  Deduplicating pairs ...")
    t1 = time.time()
    dedup = (
        df.groupby(["drug_a", "drug_b"], sort=False)
          .agg(description=("description", merge_descriptions))
          .reset_index()
    )
    print(f"  Deduplication done in {time.time()-t1:.1f}s")

    # Rename columns to keep naming consistent with downstream use
    dedup = dedup.rename(columns={"drug_a": "drugbank_id_a", "drug_b": "drugbank_id_b"})

    # Add integer primary key (1-based)
    dedup.insert(0, "interaction_id", range(1, len(dedup) + 1))

    dedup.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    reduction = (1 - len(dedup) / len(df)) * 100
    print(f"\n[step2] Done in {elapsed:.1f}s")
    print(f"  Directed pairs   : {len(df):>12,}")
    print(f"  Undirected pairs : {len(dedup):>12,}  ({reduction:.1f}% reduction)")
    print(f"  Output           : {OUTPUT_CSV}")
    print(f"  File size        : {os.path.getsize(OUTPUT_CSV)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
