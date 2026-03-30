"""
step4_build_graph.py — build structural node features + edge index from the FDA-approved subset.

Reads all relevant tables from data/step3_approved/ and produces:
  node_mapping.csv        node_idx | drugbank_id | name
  node_features.csv       node_idx + ~191 scaled feature columns  [N x F]
  node_features_raw.csv   same columns, unscaled                  [N x F]
  edge_index.csv          src_idx | dst_idx | interaction_id      [E x 3]
  feature_names.json      feature name list + group labels

Run step4_embed.py afterwards to generate text embeddings and the combined feature matrix.

Feature groups (printed at runtime):
  A  Drug masses          average_mass, monoisotopic_mass
  B  Drug type & state    is_biotech; state_solid/liquid/gas
  C  Calculated props     logP, logS, MW, HBD, HBA, RotB, PSA, charge,
                          rings, bioavailability, rule5, ghose, mddrl,
                          refractivity, polarizability, pKa_acid, pKa_basic
  D  Experimental props   exp_logP, exp_logS, exp_melting_pt, exp_boiling_pt,
                          exp_water_sol, exp_pKa
  E  Group flags          is_withdrawn, is_investigational, is_vet_approved,
                          is_nutraceutical, is_illicit
  F  Aggregate counts     n_targets, n_enzymes, n_carriers, n_transporters,
                          n_categories, n_atc_codes, n_patents, n_products,
                          n_food_interactions, n_synonyms, n_pathways
  G  ATC anatomical       atc_A … atc_V  (14 one-hot cols)
  H  Classification       kingdom_organic, kingdom_inorganic;
                          superclass_* (top-10 one-hot)
  I  MeSH categories      cat_* (top-50 most frequent, multi-hot)
  J  Pathway membership   pathway_* (top-50 most drug-populated, multi-hot)
  K  Sequence features    seq_length, aa_A … aa_Y (21 cols, biotech drugs only)

Missing values: continuous → median imputation; binary/count/one-hot → 0.
Continuous features standardised (mean=0, std=1); others left as-is.

Usage:
    python pipeline/step4_build_graph.py          # preview + save
    python pipeline/step4_build_graph.py --dry-run # preview only, no files written
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKING_DIR = r"D:\DDI\drugbank_all_full_database.xml"
INPUT_DIR   = os.path.join(WORKING_DIR, "data", "step3_approved")
OUTPUT_DIR  = os.path.join(WORKING_DIR, "data", "step4_graph")
DRY_RUN     = "--dry-run" in sys.argv

# ── helpers ───────────────────────────────────────────────────────────────────

def load(name):
    return pd.read_csv(os.path.join(INPUT_DIR, f"{name}.csv"), dtype=str,
                       keep_default_na=False)

def to_float(series):
    return pd.to_numeric(series, errors="coerce")

def coverage(col, n):
    return f"{col.notna().sum():>5,} / {n}  ({col.notna().mean()*100:5.1f}%)"

def sep(label=""):
    w = 72
    if label:
        pad = (w - len(label) - 2) // 2
        print(f"\n{'-'*pad} {label} {'-'*(w-pad-len(label)-2)}")
    else:
        print("-" * w)

# ── load tables ───────────────────────────────────────────────────────────────

print("[step4] Loading tables from step3_approved ...")
drugs        = load("drugs")
drug_attrs   = load("drug_attributes")
drug_props   = load("drug_properties")
drug_ints    = load("drug_interactants")
drug_cats    = load("drug_categories")
categories   = load("categories")
atc          = load("atc_codes")
patents      = load("patents")
products     = load("products")
pathways     = load("pathways")
path_members = load("pathway_members")
ddi          = load("drug_interactions_dedup")

N = len(drugs)
drug_ids = drugs["drugbank_id"].tolist()
id_to_idx = {db_id: i for i, db_id in enumerate(drug_ids)}

# ── feature building ──────────────────────────────────────────────────────────

feat_cols   = {}   # col_name -> pd.Series (float, aligned to drugs index)
feat_groups = {}   # col_name -> group letter

def add(name, series, group):
    feat_cols[name]   = series.reset_index(drop=True)
    feat_groups[name] = group

# ── Group A: masses ──────────────────────────────────────────────────────────
add("average_mass",       to_float(drugs["average_mass"]),       "A")
add("monoisotopic_mass",  to_float(drugs["monoisotopic_mass"]),   "A")

# ── Group B: drug type + state ───────────────────────────────────────────────
add("is_biotech",    (drugs["drug_type"] == "biotech").astype(float), "B")
add("state_solid",   (drugs["state"] == "solid").astype(float),       "B")
add("state_liquid",  (drugs["state"] == "liquid").astype(float),      "B")
add("state_gas",     (drugs["state"] == "gas").astype(float),         "B")

# ── Group C: calculated properties ──────────────────────────────────────────
CALC_NUM = {
    "logP":                    "calc_logP",
    "logS":                    "calc_logS",
    "Molecular Weight":        "calc_mol_weight",
    "H Bond Donor Count":      "calc_hbd",
    "H Bond Acceptor Count":   "calc_hba",
    "Rotatable Bond Count":    "calc_rotb",
    "Polar Surface Area (PSA)":"calc_psa",
    "Physiological Charge":    "calc_charge",
    "Number of Rings":         "calc_n_rings",
    "Bioavailability":         "calc_bioavail",
    "Rule of Five":            "calc_rule5",
    "Ghose Filter":            "calc_ghose",
    "MDDR-Like Rule":          "calc_mddrl",
    "Refractivity":            "calc_refractivity",
    "Polarizability":          "calc_polarizability",
    "pKa (strongest acidic)":  "calc_pka_acid",
    "pKa (strongest basic)":   "calc_pka_basic",
    # "Water Solubility" is not present as a calculated property in this export;
    # use calc_logS (log mol/L) instead — already included above.
}
calc_props = drug_props[drug_props["property_class"] == "calculated"].copy()
calc_pivot = (
    calc_props[calc_props["kind"].isin(CALC_NUM)]
    .groupby(["drugbank_id", "kind"])["value"]
    .first()
    .unstack("kind")
)
calc_pivot = calc_pivot.reindex(drug_ids)
for kind, col in CALC_NUM.items():
    ser = to_float(calc_pivot.get(kind, pd.Series(index=drug_ids, dtype=float)))
    add(col, ser, "C")

# ── Group D: experimental properties ────────────────────────────────────────
EXP_NUM = {
    "logP":              "exp_logP",
    "logS":              "exp_logS",
    "Melting Point":     "exp_melting_pt",
    "Boiling Point":     "exp_boiling_pt",
    "Water Solubility":  "exp_water_sol",
    "pKa":               "exp_pKa",
}
exp_props = drug_props[drug_props["property_class"] == "experimental"].copy()

def extract_numeric(s):
    """Extract first float from strings like '179-181 °C', '>300', '~2.5'."""
    import re
    m = re.search(r"[-+]?\d+\.?\d*", str(s).replace(",", ""))
    return float(m.group()) if m else np.nan

exp_pivot = (
    exp_props[exp_props["kind"].isin(EXP_NUM)]
    .groupby(["drugbank_id", "kind"])["value"]
    .first()
    .unstack("kind")
)
exp_pivot = exp_pivot.reindex(drug_ids)
for kind, col in EXP_NUM.items():
    raw = exp_pivot.get(kind, pd.Series(index=drug_ids, dtype=object))
    ser = raw.apply(extract_numeric)
    add(col, ser, "D")

# ── Group E: group flags ─────────────────────────────────────────────────────
grp = drug_attrs[drug_attrs["attr_type"] == "group"]
grp_pivot = grp.groupby(["drugbank_id", "value"]).size().unstack("value", fill_value=0)
grp_pivot = grp_pivot.reindex(drug_ids, fill_value=0)
for flag in ["withdrawn", "investigational", "vet_approved", "nutraceutical", "illicit"]:
    ser = grp_pivot.get(flag, pd.Series(0, index=drug_ids)).astype(float)
    add(f"is_{flag}", ser, "E")

# ── Group F: aggregate counts ────────────────────────────────────────────────
def count_map(df, id_col, val_col=None, val_filter=None):
    sub = df[df[val_col] == val_filter] if (val_col and val_filter) else df
    counts = sub.groupby(id_col).size().reindex(drug_ids, fill_value=0).astype(float)
    return counts

add("n_targets",          count_map(drug_ints, "drugbank_id", "role", "target"),       "F")
add("n_enzymes",          count_map(drug_ints, "drugbank_id", "role", "enzyme"),        "F")
add("n_carriers",         count_map(drug_ints, "drugbank_id", "role", "carrier"),       "F")
add("n_transporters",     count_map(drug_ints, "drugbank_id", "role", "transporter"),   "F")
add("n_categories",       count_map(drug_cats, "drugbank_id"),                          "F")
add("n_atc_codes",        count_map(atc,       "drugbank_id"),                          "F")
add("n_patents",          count_map(patents,    "drugbank_id"),                          "F")
add("n_products",         count_map(products,   "drugbank_id"),                          "F")

food = drug_attrs[drug_attrs["attr_type"] == "food_interaction"]
add("n_food_interactions", count_map(food, "drugbank_id"),                              "F")

syns = drug_attrs[drug_attrs["attr_type"] == "synonym"]
add("n_synonyms",         count_map(syns, "drugbank_id"),                               "F")

drug_path_members = path_members[path_members["member_type"] == "drug"]
add("n_pathways",         count_map(drug_path_members, "member_id"),                    "F")

# ── Group G: ATC anatomical one-hot (l4_code A–V) ────────────────────────────
ATC_GROUPS = sorted(atc["l4_code"].dropna().unique())
atc_onehot = atc.groupby(["drugbank_id", "l4_code"]).size().unstack("l4_code", fill_value=0)
atc_onehot = atc_onehot.reindex(drug_ids, fill_value=0)
for code in ATC_GROUPS:
    ser = atc_onehot.get(code, pd.Series(0, index=drug_ids)).astype(float)
    add(f"atc_{code}", ser, "G")

# ── Group H: classification ──────────────────────────────────────────────────
# Kingdom (normalize case)
kingdom_norm = drugs["classification_kingdom"].str.lower().str.strip()
add("kingdom_organic",    (kingdom_norm.str.contains("organic", na=False) &
                           ~kingdom_norm.str.contains("inorganic", na=False)).astype(float), "H")
add("kingdom_inorganic",  kingdom_norm.str.contains("inorganic", na=False).astype(float),    "H")

# Top-10 superclasses by frequency
sc = drugs["classification_superclass"].str.strip()
sc_top10 = [s for s in sc.value_counts().head(11).index.tolist() if s.strip()][:10]
for sup in sc_top10:
    safe = sup.lower().replace(" ", "_").replace(",", "").replace("/", "_")[:30]
    add(f"sc_{safe}", (sc == sup).astype(float), "H")

# ── Group I: MeSH therapeutic categories multi-hot (top-50) ─────────────────
# Join drug_categories → categories to get names, pick top-50 by drug count
cat_counts = drug_cats.groupby("category_id")["drugbank_id"].nunique()
top50_cat_ids = cat_counts.nlargest(50).index.tolist()
top50_cats = categories[categories["category_id"].isin(top50_cat_ids)].set_index("category_id")

drug_cat_pivot = (
    drug_cats[drug_cats["category_id"].isin(top50_cat_ids)]
    .groupby(["drugbank_id", "category_id"])
    .size()
    .unstack("category_id", fill_value=0)
    .reindex(drug_ids, fill_value=0)
)
for cat_id in top50_cat_ids:
    cat_name = top50_cats.loc[cat_id, "category_name"] if cat_id in top50_cats.index else cat_id
    safe = cat_name.lower().replace(" ", "_").replace(",", "").replace("/", "_")
    safe = "cat_" + safe[:28]
    ser = drug_cat_pivot.get(cat_id, pd.Series(0, index=drug_ids)).astype(float)
    add(safe, ser, "I")

# ── Group J: Pathway membership multi-hot (top-50 most drug-populated) ───────
pathway_drug_counts = drug_path_members.groupby("smpdb_id")["member_id"].nunique()
top50_pathways = pathway_drug_counts.nlargest(50).index.tolist()
top50_pway_names = pathways[pathways["smpdb_id"].isin(top50_pathways)].set_index("smpdb_id")

drug_pway_pivot = (
    drug_path_members[drug_path_members["smpdb_id"].isin(top50_pathways)]
    .groupby(["member_id", "smpdb_id"])
    .size()
    .unstack("smpdb_id", fill_value=0)
    .reindex(drug_ids, fill_value=0)
)
for smpdb_id in top50_pathways:
    pway_name = top50_pway_names.loc[smpdb_id, "name"] if smpdb_id in top50_pway_names.index else smpdb_id
    safe = pway_name.lower().replace(" ", "_").replace(",", "").replace("/", "_").replace("-", "_")
    safe = "pway_" + safe[:26]
    ser = drug_pway_pivot.get(smpdb_id, pd.Series(0, index=drug_ids)).astype(float)
    add(safe, ser, "J")

# ── Group K: Sequence features (biotech drugs only) ──────────────────────────
# FASTA sequences stored in drug_attributes where attr_type='sequence'
# Compute: sequence length + % of each of the 20 standard amino acids
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

seq_rows = drug_attrs[drug_attrs["attr_type"] == "sequence"].copy()

def parse_fasta_seq(fasta):
    """Strip FASTA header lines and return the bare sequence string."""
    lines = [l for l in fasta.strip().splitlines() if not l.startswith(">")]
    return "".join(lines).upper()

seq_map = {}  # drugbank_id -> sequence string
for _, row in seq_rows.iterrows():
    seq = parse_fasta_seq(row["value"])
    if seq:
        seq_map[row["drugbank_id"]] = seq_map.get(row["drugbank_id"], "") + seq

seq_lengths = pd.Series(
    {db_id: len(seq_map[db_id]) if db_id in seq_map else 0 for db_id in drug_ids},
    index=drug_ids, dtype=float
)
add("seq_length", seq_lengths, "K")

for aa in AA_LIST:
    aa_pct = pd.Series(
        {db_id: (seq_map[db_id].count(aa) / len(seq_map[db_id])
                 if db_id in seq_map and len(seq_map[db_id]) > 0 else 0.0)
         for db_id in drug_ids},
        index=drug_ids, dtype=float
    )
    add(f"aa_{aa}", aa_pct, "K")

# ── Assemble raw matrix ───────────────────────────────────────────────────────
feature_names = list(feat_cols.keys())
X_raw = pd.DataFrame(feat_cols)

# Impute continuous features with median; binary/count/one-hot stay 0
CONTINUOUS_GROUPS = {"A", "C", "D"}
for col in feature_names:
    if feat_groups[col] in CONTINUOUS_GROUPS:
        med = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(med)
    else:
        X_raw[col] = X_raw[col].fillna(0.0)

# Standardise: mean=0, std=1 for continuous; leave binary/count as-is
X_scaled = X_raw.copy()
for col in feature_names:
    if feat_groups[col] in CONTINUOUS_GROUPS:
        mu, sd = X_raw[col].mean(), X_raw[col].std()
        if sd > 0:
            X_scaled[col] = (X_raw[col] - mu) / sd

# ── Edge index ───────────────────────────────────────────────────────────────
valid_mask = ddi["drugbank_id_a"].isin(id_to_idx) & ddi["drugbank_id_b"].isin(id_to_idx)
ddi_valid = ddi[valid_mask].reset_index(drop=True)

src = ddi_valid["drugbank_id_a"].map(id_to_idx).values.astype(np.int64)
dst = ddi_valid["drugbank_id_b"].map(id_to_idx).values.astype(np.int64)
edge_interaction_ids = ddi_valid["interaction_id"].astype(np.int64).values

E = len(src)

# ── Preview ───────────────────────────────────────────────────────────────────
sep("STEP 4 — GRAPH BUILD PREVIEW")
print(f"  Nodes (drugs)  : {N:,}")
print(f"  Edges (DDI)    : {E:,}  (undirected pairs, both approved)")
print(f"  Features       : {len(feature_names)}")
print(f"  Avg degree     : {E*2/N:.1f}  (counting each edge once)")
print(f"  Edge density   : {E / (N*(N-1)/2) * 100:.3f}%")

sep("FEATURE INVENTORY")
print(f"  {'#':>3}  {'Feature':<38} {'Grp'}  {'Coverage':>22}  {'Mean':>9}  {'Std':>9}")
sep()
for i, col in enumerate(feature_names):
    raw_ser = feat_cols[col]
    cov = coverage(raw_ser, N)
    mean_v = X_raw[col].mean()
    std_v  = X_raw[col].std()
    print(f"  {i+1:>3}  {col:<38} {feat_groups[col]}    {cov}  {mean_v:>9.3f}  {std_v:>9.3f}")

sep("OUTPUT SHAPES")
print(f"  node_features.csv     (scaled) : [{N} rows x {len(feature_names)} features]")
print(f"  node_features_raw.csv (raw)    : [{N} rows x {len(feature_names)} features]")
print(f"  edge_index.csv                 : [{E} rows x 3 cols] (src, dst, interaction_id)")
print(f"  node_mapping.csv               : [{N} rows]")

sep("SAMPLE NODES (first 3)")
sample = drugs[["drugbank_id","name","drug_type","state"]].head(3).copy()
sample["n_features_nonzero"] = (X_raw.values[:3] != 0).sum(axis=1)
print(sample.to_string(index=False))

sep("FEATURE GROUP SUMMARY")
from collections import Counter
gc = Counter(feat_groups.values())
labels = {"A":"Drug masses","B":"Type & state","C":"Calculated props",
          "D":"Experimental props","E":"Group flags","F":"Counts",
          "G":"ATC one-hot","H":"Classification",
          "I":"MeSH categories (multi-hot)","J":"Pathway membership (multi-hot)",
          "K":"Sequence features"}
for g in sorted(gc):
    print(f"  Group {g} ({labels[g]}): {gc[g]} features")
sep()

# ── Save ──────────────────────────────────────────────────────────────────────
if DRY_RUN:
    print("\n[step4] Dry-run mode — no files written.")
    sys.exit(0)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# node_mapping.csv
node_map = pd.DataFrame({
    "node_idx":    range(N),
    "drugbank_id": drug_ids,
    "name":        drugs["name"].tolist(),
})
node_map_path = os.path.join(OUTPUT_DIR, "node_mapping.csv")
node_map.to_csv(node_map_path, index=False)

# node_features.csv  (scaled)
node_feat_path = os.path.join(OUTPUT_DIR, "node_features.csv")
X_out = X_scaled.copy()
X_out.insert(0, "node_idx", range(N))
X_out.to_csv(node_feat_path, index=False)

# node_features_raw.csv  (unscaled)
node_feat_raw_path = os.path.join(OUTPUT_DIR, "node_features_raw.csv")
X_out_raw = X_raw.copy()
X_out_raw.insert(0, "node_idx", range(N))
X_out_raw.to_csv(node_feat_raw_path, index=False)

# edge_index.csv
edge_path = os.path.join(OUTPUT_DIR, "edge_index.csv")
pd.DataFrame({
    "src_idx":        src,
    "dst_idx":        dst,
    "interaction_id": edge_interaction_ids,
}).to_csv(edge_path, index=False)

# feature_names.json
meta_path = os.path.join(OUTPUT_DIR, "feature_names.json")
with open(meta_path, "w") as f:
    json.dump({"features": feature_names, "groups": feat_groups}, f, indent=2)

sep("SAVED TO " + OUTPUT_DIR)
for path in [node_map_path, node_feat_path, node_feat_raw_path, edge_path, meta_path]:
    kb = os.path.getsize(path) / 1024
    label = os.path.basename(path)
    print(f"  {label:<30} {kb:>8.0f} KB")
sep()
print("[step4] Done.")
