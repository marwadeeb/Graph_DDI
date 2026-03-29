"""
validate.py — post-parse validation of all 27 CSV files.

Checks performed:
  1. All 27 CSV files exist and are non-empty
  2. Row counts are in expected ranges
  3. No drug in drug_interactions references an unknown drugbank_id
  4. All category_ids in drug_categories exist in categories
  5. All interactant_ids in drug_interactants exist in interactants
  6. All polypeptide_ids in interactant_polypeptides exist in polypeptides
  7. All ref_pks in reference_associations exist in references
  8. Primary drug IDs in drug_ids match drugs table
  9. drugbank_id column is never NULL in any table that has one
 10. drug_type values are only 'small molecule' or 'biotech'
 11. ATC code level structure is consistent (l1_code shorter than atc_code)
 12. SNP snp_type values are only 'effect' or 'adverse_reaction'
 13. entity_type in external_identifiers is in the allowed set
 14. drug_interactants role values are in the allowed set
 15. pathway member_type values are in the allowed set
 16. XSD coverage check: counts key tables against expected minimums

Usage:
    python validate.py
"""
import csv
import os
import sys
from config import OUTPUT_DIR, SCHEMA

ERRORS = []
WARNINGS = []


def err(msg):
    ERRORS.append(msg)
    print(f"  [ERROR]   {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN]    {msg}")


def ok(msg):
    print(f"  [OK]      {msg}")


# ── helpers ───────────────────────────────────────────────────────────────────

def csv_path(table):
    return os.path.join(OUTPUT_DIR, f"{table}.csv")


def read_csv(table):
    path = csv_path(table)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def count_rows(table):
    """Count data rows using csv.reader to handle multi-line quoted fields correctly."""
    path = csv_path(table)
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1   # subtract header


def col_set(rows, col):
    return {r[col] for r in rows if r.get(col)}


# ── check 1: file existence and size ─────────────────────────────────────────

def check_files_exist():
    print("\n[1] File existence & size")
    for table in SCHEMA:
        path = csv_path(table)
        if not os.path.exists(path):
            err(f"{table}.csv missing")
        else:
            size = os.path.getsize(path)
            if size < 10:
                warn(f"{table}.csv exists but is very small ({size} bytes)")
            else:
                ok(f"{table}.csv  ({size:,} bytes)")


# ── check 2: row counts ───────────────────────────────────────────────────────

EXPECTED_MINIMUMS = {
    "drugs":                   19_000,   # ~19,842 drugs
    "drug_ids":                20_000,   # primary + legacy IDs
    "drug_attributes":        500_000,   # groups/synonyms/organisms/etc.
    "drug_properties":        200_000,   # calculated + experimental
    "external_identifiers":   100_000,
    "references":              30_000,   # globally deduplicated refs
    "reference_associations":  80_000,   # general + interactant refs
    "products":               400_000,
    "drug_interactions":    2_500_000,   # directed DDI edges
    "interactants":             4_000,
    "polypeptides":             5_000,   # unique UniProt proteins
    "drug_interactants":       30_000,
    "categories":               3_000,
    "drug_categories":         80_000,
    "pathways":                10_000,
    "pathway_members":      1_000_000,   # pathway drug+enzyme members
}

def check_row_counts():
    print("\n[2] Row count minimums")
    counts = {}
    for table in SCHEMA:
        try:
            n = count_rows(table)
            counts[table] = n
            minimum = EXPECTED_MINIMUMS.get(table, 0)
            if n < minimum:
                err(f"{table}: {n:,} rows (expected >= {minimum:,})")
            else:
                ok(f"{table}: {n:,} rows")
        except Exception as e:
            err(f"{table}: could not count rows — {e}")
    return counts


# ── check 3: DDI referential integrity ───────────────────────────────────────

def check_ddi_ids(drug_ids_set):
    print("\n[3] drug_interactions — referential integrity (sample check)")
    try:
        rows = read_csv("drug_interactions")
        unknown_src  = sum(1 for r in rows if r["drugbank_id"] not in drug_ids_set)
        unknown_tgt  = sum(1 for r in rows if r["interacting_drugbank_id"] not in drug_ids_set)
        if unknown_src:
            warn(f"drug_interactions: {unknown_src:,} rows with unknown source drugbank_id")
        else:
            ok("drug_interactions: all source IDs found in drugs table")
        # Target IDs can legitimately be absent if the other drug is also in DB
        if unknown_tgt:
            warn(f"drug_interactions: {unknown_tgt:,} rows with unknown target drugbank_id "
                 f"(may be withdrawn/experimental drugs not in this export)")
        else:
            ok("drug_interactions: all target IDs found in drugs table")
    except Exception as e:
        err(f"check_ddi_ids failed: {e}")


# ── check 4: category FK ─────────────────────────────────────────────────────

def check_category_fk():
    print("\n[4] drug_categories -> categories FK")
    try:
        cat_ids   = col_set(read_csv("categories"), "category_id")
        dc_rows   = read_csv("drug_categories")
        missing   = sum(1 for r in dc_rows if r["category_id"] not in cat_ids)
        if missing:
            err(f"drug_categories: {missing:,} rows with unknown category_id")
        else:
            ok(f"drug_categories: all {len(dc_rows):,} category_id values found")
    except Exception as e:
        err(f"check_category_fk failed: {e}")


# ── check 5: interactant FK ───────────────────────────────────────────────────

def check_interactant_fk():
    print("\n[5] drug_interactants -> interactants FK")
    try:
        int_ids  = col_set(read_csv("interactants"), "interactant_id")
        di_rows  = read_csv("drug_interactants")
        missing  = sum(1 for r in di_rows if r["interactant_id"] not in int_ids)
        if missing:
            err(f"drug_interactants: {missing:,} rows with unknown interactant_id")
        else:
            ok(f"drug_interactants: all {len(di_rows):,} interactant_id values found")
    except Exception as e:
        err(f"check_interactant_fk failed: {e}")


# ── check 6: polypeptide FK ───────────────────────────────────────────────────

def check_polypeptide_fk():
    print("\n[6] interactant_polypeptides -> polypeptides FK")
    try:
        poly_ids = col_set(read_csv("polypeptides"), "polypeptide_id")
        ip_rows  = read_csv("interactant_polypeptides")
        missing  = sum(1 for r in ip_rows if r["polypeptide_id"] not in poly_ids)
        if missing:
            err(f"interactant_polypeptides: {missing:,} rows with unknown polypeptide_id")
        else:
            ok(f"interactant_polypeptides: all {len(ip_rows):,} polypeptide_id values found")
    except Exception as e:
        err(f"check_polypeptide_fk failed: {e}")


# ── check 7: reference_associations FK ───────────────────────────────────────

def check_ref_fk():
    print("\n[7] reference_associations -> references FK")
    try:
        ref_pks  = col_set(read_csv("references"), "ref_pk")
        ra_rows  = read_csv("reference_associations")
        missing  = sum(1 for r in ra_rows if r["ref_pk"] not in ref_pks)
        if missing:
            err(f"reference_associations: {missing:,} rows with unknown ref_pk")
        else:
            ok(f"reference_associations: all {len(ra_rows):,} ref_pk values found")
    except Exception as e:
        err(f"check_ref_fk failed: {e}")


# ── check 8: drug_ids primary coverage ───────────────────────────────────────

def check_drug_ids_coverage(drug_ids_set):
    print("\n[8] drug_ids — primary ID coverage")
    try:
        di_rows  = read_csv("drug_ids")
        primary_in_di = {r["legacy_id"] for r in di_rows
                         if r.get("is_primary", "").lower() == "true"}
        missing = drug_ids_set - primary_in_di
        if missing:
            warn(f"drug_ids: {len(missing):,} primary drug IDs not found in drug_ids table")
        else:
            ok(f"drug_ids: all {len(drug_ids_set):,} primary IDs represented")
    except Exception as e:
        err(f"check_drug_ids_coverage failed: {e}")


# ── check 9: NULL drugbank_id ─────────────────────────────────────────────────

def check_no_null_ids():
    print("\n[9] NULL drugbank_id check")
    tables_with_did = [t for t in SCHEMA if "drugbank_id" in SCHEMA[t]]
    for table in tables_with_did:
        try:
            rows   = read_csv(table)
            nulls  = sum(1 for r in rows if not r.get("drugbank_id"))
            if nulls:
                err(f"{table}: {nulls:,} rows with NULL/empty drugbank_id")
            else:
                ok(f"{table}: no NULL drugbank_id ({len(rows):,} rows)")
        except Exception as e:
            err(f"{table} NULL check failed: {e}")


# ── check 10: drug_type values ────────────────────────────────────────────────

def check_drug_type():
    print("\n[10] drugs — drug_type values")
    try:
        rows   = read_csv("drugs")
        types  = {r["drug_type"] for r in rows}
        valid  = {"small molecule", "biotech"}
        bad    = types - valid
        if bad:
            err(f"drugs: unexpected drug_type values: {bad}")
        else:
            ok(f"drugs: drug_type values = {types}")
    except Exception as e:
        err(f"check_drug_type failed: {e}")


# ── check 11: ATC code structure ──────────────────────────────────────────────

def check_atc_codes():
    print("\n[11] atc_codes — level hierarchy consistency")
    try:
        rows  = read_csv("atc_codes")
        bad   = 0
        for r in rows:
            code   = r.get("atc_code") or ""
            l1c    = r.get("l1_code") or ""
            l4c    = r.get("l4_code") or ""
            # l1 code should be shorter than full code; l4 should be 1 char
            if code and l1c and len(l1c) >= len(code):
                bad += 1
            if l4c and len(l4c) > 1:
                bad += 1
        if bad:
            warn(f"atc_codes: {bad} rows with unexpected level code lengths")
        else:
            ok(f"atc_codes: level hierarchy looks correct ({len(rows):,} rows)")
    except Exception as e:
        err(f"check_atc_codes failed: {e}")


# ── check 12: snp_type values ─────────────────────────────────────────────────

def check_snp_types():
    print("\n[12] drug_snp_data — snp_type values")
    try:
        rows  = read_csv("drug_snp_data")
        types = {r["snp_type"] for r in rows}
        valid = {"effect", "adverse_reaction"}
        bad   = types - valid
        if bad:
            err(f"drug_snp_data: unexpected snp_type values: {bad}")
        else:
            ok(f"drug_snp_data: snp_type values = {types} ({len(rows):,} rows)")
    except Exception as e:
        err(f"check_snp_types failed: {e}")


# ── check 13: external_identifiers entity_type ───────────────────────────────

def check_entity_types():
    print("\n[13] external_identifiers — entity_type values")
    try:
        rows  = read_csv("external_identifiers")
        types = {r["entity_type"] for r in rows}
        valid = {"drug", "drug_link", "polypeptide", "salt"}
        bad   = types - valid
        if bad:
            err(f"external_identifiers: unexpected entity_type values: {bad}")
        else:
            ok(f"external_identifiers: entity_type values = {types} ({len(rows):,} rows)")
    except Exception as e:
        err(f"check_entity_types failed: {e}")


# ── check 14: drug_interactants role values ───────────────────────────────────

def check_interactant_roles():
    print("\n[14] drug_interactants — role values")
    try:
        rows  = read_csv("drug_interactants")
        roles = {r["role"] for r in rows}
        valid = {"target", "enzyme", "carrier", "transporter"}
        bad   = roles - valid
        if bad:
            err(f"drug_interactants: unexpected role values: {bad}")
        else:
            ok(f"drug_interactants: role values = {roles} ({len(rows):,} rows)")
    except Exception as e:
        err(f"check_interactant_roles failed: {e}")


# ── check 15: pathway member_type values ──────────────────────────────────────

def check_pathway_member_types():
    print("\n[15] pathway_members — member_type values")
    try:
        rows  = read_csv("pathway_members")
        types = {r["member_type"] for r in rows}
        valid = {"drug", "enzyme"}
        bad   = types - valid
        if bad:
            err(f"pathway_members: unexpected member_type values: {bad}")
        else:
            ok(f"pathway_members: member_type values = {types} ({len(rows):,} rows)")
    except Exception as e:
        err(f"check_pathway_member_types failed: {e}")


# ── check 16: XSD coverage summary ───────────────────────────────────────────

def check_xsd_coverage(counts):
    print("\n[16] XSD coverage summary")
    items = [
        ("drugs",            "drug entries (XSD: drug-type)"),
        ("drug_ids",         "drugbank-id elements"),
        ("drug_attributes",  "multi-valued string attrs (groups/synonyms/etc.)"),
        ("drug_properties",  "calculated + experimental properties"),
        ("external_identifiers", "external IDs + links (drug/polypeptide/salt)"),
        ("references",       "globally deduplicated references"),
        ("reference_associations", "reference context associations"),
        ("salts",            "salt forms"),
        ("products",         "marketed products"),
        ("drug_commercial_entities", "packagers + manufacturers + brands"),
        ("mixtures",         "drug mixtures"),
        ("prices",           "price entries"),
        ("categories",       "unique MeSH categories"),
        ("drug_categories",  "drug-category assignments"),
        ("dosages",          "dosage records"),
        ("atc_codes",        "ATC code entries"),
        ("patents",          "patent records"),
        ("drug_interactions","directed DDI edges"),
        ("drug_snp_data",    "SNP pharmacogenomics records"),
        ("pathways",         "unique pathways"),
        ("pathway_members",  "pathway drug/enzyme members"),
        ("reactions",        "metabolic reactions"),
        ("interactants",     "unique binding entities (BE-IDs)"),
        ("drug_interactants","drug–protein interaction records"),
        ("polypeptides",     "unique UniProt polypeptides"),
        ("interactant_polypeptides", "interactant–polypeptide links"),
        ("polypeptide_attributes",   "polypeptide synonyms/Pfam/GO"),
    ]
    for table, desc in items:
        n = counts.get(table, "?")
        print(f"    {n:>10,}  {desc}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("DrugBank CSV Validation Report")
    print("=" * 65)

    check_files_exist()

    counts = check_row_counts()

    # Load drug IDs set (used in multiple checks)
    try:
        drugs_rows   = read_csv("drugs")
        drug_ids_set = {r["drugbank_id"] for r in drugs_rows if r.get("drugbank_id")}
    except Exception as e:
        err(f"Could not load drugs.csv: {e}")
        drug_ids_set = set()

    check_ddi_ids(drug_ids_set)
    check_category_fk()
    check_interactant_fk()
    check_polypeptide_fk()
    check_ref_fk()
    check_drug_ids_coverage(drug_ids_set)
    check_no_null_ids()
    check_drug_type()
    check_atc_codes()
    check_snp_types()
    check_entity_types()
    check_interactant_roles()
    check_pathway_member_types()
    check_xsd_coverage(counts)

    print("\n" + "=" * 65)
    print(f"Validation complete: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
    if ERRORS:
        print("\nERRORS:")
        for e in ERRORS:
            print(f"  [X] {e}")
    if WARNINGS:
        print("\nWARNINGS:")
        for w in WARNINGS:
            print(f"  [!] {w}")
    if not ERRORS and not WARNINGS:
        print("  [OK] All checks passed — data looks clean!")
    print("=" * 65)

    return len(ERRORS)


if __name__ == "__main__":
    sys.exit(main())
