"""
filter_approved.py — filter all CSVs to FDA-approved drugs only.

A drug is considered FDA-approved if it has at least one row in drug_attributes
with attr_type='group' and value='approved'.

Strategy:
  1. Build the approved_ids set from drug_attributes.csv
  2. Filter every table that has a 'drugbank_id' column
  3. For drug_interactions, also use the deduplicated pairs from step2_output/
     and keep only pairs where BOTH drugs are FDA-approved
  4. Filter lookup tables (categories, pathways, interactants, polypeptides, etc.)
     to only entries actually referenced by the filtered drug tables

Inputs:  data/step1_full/*.csv  (all 27 tables)
         data/step2_dedup/drug_interactions_dedup.csv
Outputs: data/step3_approved/*.csv  (filtered subset)

Usage:
    python pipeline/filter_approved.py
    (run after dedup_interactions.py)
"""
import os
import time
import pandas as pd

WORKING_DIR   = r"D:\DDI\drugbank_all_full_database.xml"
INPUT_DIR     = os.path.join(WORKING_DIR, "data", "step1_full")
STEP2_DIR     = os.path.join(WORKING_DIR, "data", "step2_dedup")
OUTPUT_DIR    = os.path.join(WORKING_DIR, "data", "step3_approved")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read(name, source_dir=INPUT_DIR):
    path = os.path.join(source_dir, f"{name}.csv")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write(df, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  {name}.csv  —  {len(df):,} rows  ({os.path.getsize(path)/1024:.0f} KB)")
    return path


def main():
    t0 = time.time()
    print("[step3] Building FDA-approved drug ID set ...")

    attrs = read("drug_attributes")
    approved_ids = set(
        attrs.loc[(attrs["attr_type"] == "group") & (attrs["value"] == "approved"),
                  "drugbank_id"]
    )
    print(f"  FDA-approved drugs: {len(approved_ids):,}")

    print("\n[step3] Writing filtered tables to step3_output/ ...")

    # ── Drug-centric tables (filter by drugbank_id) ────────────────────────────
    drugs         = read("drugs")
    drugs_f       = drugs[drugs["drugbank_id"].isin(approved_ids)]
    write(drugs_f, "drugs")

    drug_ids_f    = read("drug_ids")
    drug_ids_f    = drug_ids_f[drug_ids_f["drugbank_id"].isin(approved_ids)]
    write(drug_ids_f, "drug_ids")

    drug_attrs_f  = read("drug_attributes")
    drug_attrs_f  = drug_attrs_f[drug_attrs_f["drugbank_id"].isin(approved_ids)]
    write(drug_attrs_f, "drug_attributes")

    drug_props_f  = read("drug_properties")
    drug_props_f  = drug_props_f[drug_props_f["drugbank_id"].isin(approved_ids)]
    write(drug_props_f, "drug_properties")

    salts_f       = read("salts")
    salts_f       = salts_f[salts_f["drugbank_id"].isin(approved_ids)]
    write(salts_f, "salts")

    products_f    = read("products")
    products_f    = products_f[products_f["drugbank_id"].isin(approved_ids)]
    write(products_f, "products")

    dce_f         = read("drug_commercial_entities")
    dce_f         = dce_f[dce_f["drugbank_id"].isin(approved_ids)]
    write(dce_f, "drug_commercial_entities")

    mixtures_f    = read("mixtures")
    mixtures_f    = mixtures_f[mixtures_f["drugbank_id"].isin(approved_ids)]
    write(mixtures_f, "mixtures")

    prices_f      = read("prices")
    prices_f      = prices_f[prices_f["drugbank_id"].isin(approved_ids)]
    write(prices_f, "prices")

    drug_cats_f   = read("drug_categories")
    drug_cats_f   = drug_cats_f[drug_cats_f["drugbank_id"].isin(approved_ids)]
    write(drug_cats_f, "drug_categories")

    dosages_f     = read("dosages")
    dosages_f     = dosages_f[dosages_f["drugbank_id"].isin(approved_ids)]
    write(dosages_f, "dosages")

    atc_f         = read("atc_codes")
    atc_f         = atc_f[atc_f["drugbank_id"].isin(approved_ids)]
    write(atc_f, "atc_codes")

    patents_f     = read("patents")
    patents_f     = patents_f[patents_f["drugbank_id"].isin(approved_ids)]
    write(patents_f, "patents")

    snp_f         = read("drug_snp_data")
    snp_f         = snp_f[snp_f["drugbank_id"].isin(approved_ids)]
    write(snp_f, "drug_snp_data")

    reactions_f   = read("reactions")
    reactions_f   = reactions_f[reactions_f["drugbank_id"].isin(approved_ids)]
    write(reactions_f, "reactions")

    di_f          = read("drug_interactants")
    di_f          = di_f[di_f["drugbank_id"].isin(approved_ids)]
    write(di_f, "drug_interactants")

    # ── drug_interactions: use deduplicated pairs, both sides must be approved ─
    ddi_dedup     = read("drug_interactions_dedup", source_dir=STEP2_DIR)
    ddi_f         = ddi_dedup[
        ddi_dedup["drugbank_id_a"].isin(approved_ids) &
        ddi_dedup["drugbank_id_b"].isin(approved_ids)
    ]
    write(ddi_f, "drug_interactions_dedup")

    # ── external_identifiers: filter drug-level rows; keep polypeptide rows ───
    ext_ids       = read("external_identifiers")
    ext_ids_drug  = ext_ids[
        (ext_ids["entity_type"] == "drug") &
        ext_ids["entity_id"].isin(approved_ids)
    ]
    ext_ids_salt  = ext_ids[
        (ext_ids["entity_type"] == "salt") &
        ext_ids["entity_id"].isin(set(salts_f["salt_id"]))
    ]
    # polypeptide external_ids filtered below after polypeptide filtering
    ext_ids_partial = pd.concat([ext_ids_drug, ext_ids_salt], ignore_index=True)

    # ── reference_associations: filter by drugbank_id ─────────────────────────
    ref_assoc     = read("reference_associations")
    ref_assoc_f   = ref_assoc[ref_assoc["drugbank_id"].isin(approved_ids)]
    write(ref_assoc_f, "reference_associations")

    # references: keep only those cited by the filtered reference_associations
    active_ref_pks = set(ref_assoc_f["ref_pk"])
    refs          = read("references")
    refs_f        = refs[refs["ref_pk"].isin(active_ref_pks)]
    write(refs_f, "references")

    # ── categories: keep only those used by filtered drug_categories ──────────
    active_cat_ids = set(drug_cats_f["category_id"])
    cats          = read("categories")
    cats_f        = cats[cats["category_id"].isin(active_cat_ids)]
    write(cats_f, "categories")

    # ── pathways + pathway_members ────────────────────────────────────────────
    pm            = read("pathway_members")
    # Keep pathway if it contains at least one approved drug member
    approved_pathways = set(
        pm.loc[(pm["member_type"] == "drug") & pm["member_id"].isin(approved_ids),
               "smpdb_id"]
    )
    pathways_f    = read("pathways")
    pathways_f    = pathways_f[pathways_f["smpdb_id"].isin(approved_pathways)]
    write(pathways_f, "pathways")

    pm_f          = pm[pm["smpdb_id"].isin(approved_pathways)]
    write(pm_f, "pathway_members")

    # ── interactants + drug_interactants + polypeptides ───────────────────────
    active_interactant_ids = set(di_f["interactant_id"])

    interactants_f = read("interactants")
    interactants_f = interactants_f[
        interactants_f["interactant_id"].isin(active_interactant_ids)
    ]
    write(interactants_f, "interactants")

    ip            = read("interactant_polypeptides")
    ip_f          = ip[ip["interactant_id"].isin(active_interactant_ids)]
    write(ip_f, "interactant_polypeptides")

    active_poly_ids = set(ip_f["polypeptide_id"])

    polypeptides_f = read("polypeptides")
    polypeptides_f = polypeptides_f[
        polypeptides_f["polypeptide_id"].isin(active_poly_ids)
    ]
    write(polypeptides_f, "polypeptides")

    poly_attrs_f  = read("polypeptide_attributes")
    poly_attrs_f  = poly_attrs_f[
        poly_attrs_f["polypeptide_id"].isin(active_poly_ids)
    ]
    write(poly_attrs_f, "polypeptide_attributes")

    # Complete external_identifiers with polypeptide rows
    ext_ids_poly  = ext_ids[
        (ext_ids["entity_type"] == "polypeptide") &
        ext_ids["entity_id"].isin(active_poly_ids)
    ]
    ext_ids_f     = pd.concat([ext_ids_partial, ext_ids_poly], ignore_index=True)
    write(ext_ids_f, "external_identifiers")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_bytes = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")
    )
    print(f"\n[step3] Done in {elapsed:.1f}s")
    print(f"  FDA-approved drugs       : {len(approved_ids):,}")
    print(f"  Drugs in output          : {len(drugs_f):,}")
    print(f"  Approved-only DDI pairs  : {len(ddi_f):,}")
    print(f"  Active pathways          : {len(pathways_f):,}")
    print(f"  Active polypeptides      : {len(polypeptides_f):,}")
    print(f"  Total output size        : {total_bytes/1024/1024:.1f} MB")
    print(f"  Output directory         : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
