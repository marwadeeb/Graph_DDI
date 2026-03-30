"""
step4_embed.py — generate pharmacological text embeddings for all FDA-approved drugs.

For each drug, builds a rich text document by concatenating:
  - name, drug_type
  - description, indication, mechanism_of_action, pharmacodynamics,
    toxicity, metabolism, absorption
  - food interactions, affected organisms       (drug_attributes)
  - MeSH therapeutic category names             (categories + drug_categories)
  - ATC code names (level-1 subgroup)           (atc_codes)

Embeds with sentence-transformers: pritamdeka/S-PubMedBert-MS-MARCO (768-dim).
Model downloads once on first run (~400 MB). Runs fully on CPU, no API key needed.

Inputs:
  data/step3_approved/*.csv
  data/step4_graph/node_mapping.csv      (for node_idx alignment)
  data/step4_graph/node_features.csv     (structural features, for combined output)

Outputs:
  data/step4_graph/node_embeddings.csv          [N x 769]  (node_idx + 768 dims)
  data/step4_graph/node_features_combined.csv   [N x (F+768)]  structural + embeddings

Usage:
    pip install sentence-transformers
    python pipeline/step4_embed.py
    python pipeline/step4_embed.py --model all-MiniLM-L6-v2   # faster, 384-dim
"""
import os, sys, time, warnings

# Disable TensorFlow integration in transformers before any imports.
# Prevents a Keras 3 / tf-keras version conflict on systems with TF installed.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORKING_DIR  = r"D:\DDI\drugbank_all_full_database.xml"
INPUT_DIR    = os.path.join(WORKING_DIR, "data", "step3_approved")
GRAPH_DIR    = os.path.join(WORKING_DIR, "data", "step4_graph")

# Default model — biomedical PubMedBERT fine-tuned for semantic similarity
DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
BATCH_SIZE    = 64

# Allow override via --model flag
model_name = DEFAULT_MODEL
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--model" and i + 1 < len(sys.argv) - 1:
        model_name = sys.argv[i + 2]

# ── helpers ───────────────────────────────────────────────────────────────────

def load(name, src=INPUT_DIR):
    return pd.read_csv(os.path.join(src, f"{name}.csv"), dtype=str, keep_default_na=False)

def sep(label=""):
    w = 72
    if label:
        pad = (w - len(label) - 2) // 2
        print(f"\n{'-'*pad} {label} {'-'*(w-pad-len(label)-2)}")
    else:
        print("-" * w)

# ── load tables ───────────────────────────────────────────────────────────────

sep("STEP 4b — TEXT EMBEDDING")
print(f"  Model : {model_name}")
print(f"  Input : {INPUT_DIR}")
print(f"  Output: {GRAPH_DIR}")

print("\n[embed] Loading tables ...")
drugs       = load("drugs")
drug_attrs  = load("drug_attributes")
drug_cats   = load("drug_categories")
categories  = load("categories")
atc         = load("atc_codes")
node_map    = load("node_mapping", src=GRAPH_DIR)

drug_ids  = node_map["drugbank_id"].tolist()
N         = len(drug_ids)
drugs     = drugs.set_index("drugbank_id").reindex(drug_ids)

# ── build per-drug lookup structures ─────────────────────────────────────────

# Food interactions
food_map = (
    drug_attrs[drug_attrs["attr_type"] == "food_interaction"]
    .groupby("drugbank_id")["value"]
    .apply(list)
    .to_dict()
)

# Affected organisms
org_map = (
    drug_attrs[drug_attrs["attr_type"] == "affected_organism"]
    .groupby("drugbank_id")["value"]
    .apply(list)
    .to_dict()
)

# MeSH category names
cat_id_to_name = categories.set_index("category_id")["category_name"].to_dict()
cat_map = (
    drug_cats.groupby("drugbank_id")["category_id"]
    .apply(lambda ids: [cat_id_to_name.get(i, "") for i in ids])
    .to_dict()
)

# ATC level-1 subgroup names (most specific)
atc_map = (
    atc.groupby("drugbank_id")["l1_name"]
    .apply(lambda names: list(names.dropna().unique()))
    .to_dict()
)

# ── build text documents ──────────────────────────────────────────────────────

TEXT_FIELDS = [
    "description", "indication", "mechanism_of_action",
    "pharmacodynamics", "toxicity", "metabolism", "absorption",
]

def build_text(db_id):
    row = drugs.loc[db_id] if db_id in drugs.index else {}
    parts = []

    name = row.get("name", "") if hasattr(row, "get") else ""
    dtype = row.get("drug_type", "") if hasattr(row, "get") else ""
    if name:
        parts.append(f"Drug: {name}.")
    if dtype:
        parts.append(f"Type: {dtype}.")

    for field in TEXT_FIELDS:
        val = row.get(field, "") if hasattr(row, "get") else ""
        if val and str(val).strip():
            parts.append(str(val).strip())

    cats = cat_map.get(db_id, [])
    if cats:
        parts.append("Therapeutic categories: " + ", ".join(c for c in cats if c) + ".")

    atcs = atc_map.get(db_id, [])
    if atcs:
        parts.append("ATC classes: " + ", ".join(atcs) + ".")

    orgs = org_map.get(db_id, [])
    if orgs:
        parts.append("Affects: " + ", ".join(orgs) + ".")

    foods = food_map.get(db_id, [])
    if foods:
        parts.append("Food interactions: " + "; ".join(foods[:5]) + ".")  # cap at 5

    return " ".join(parts) if parts else name or db_id

print("[embed] Building drug text documents ...")
texts = [build_text(db_id) for db_id in drug_ids]

# Quick coverage stats
non_empty    = sum(1 for t in texts if len(t) > 50)
avg_len      = int(np.mean([len(t) for t in texts]))
median_len   = int(np.median([len(t) for t in texts]))
print(f"  Drugs with rich text (>50 chars) : {non_empty:,} / {N:,}")
print(f"  Avg text length                  : {avg_len:,} chars")
print(f"  Median text length               : {median_len:,} chars")
print(f"\n  Sample (DB00001):\n  {texts[0][:300]}...")

# ── load model + embed ────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("\n[embed] ERROR: sentence-transformers not installed.")
    print("  Run: pip install sentence-transformers")
    sys.exit(1)

sep(f"EMBEDDING {N:,} DRUGS")
print(f"  Loading model: {model_name}")
print("  (First run downloads ~400 MB — cached afterwards)")

t0 = time.time()
model = SentenceTransformer(model_name)
embed_dim = model.get_sentence_embedding_dimension()
print(f"  Model loaded in {time.time()-t0:.1f}s  |  embedding dim = {embed_dim}")

print(f"\n  Encoding in batches of {BATCH_SIZE} ...")
t1 = time.time()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2-normalize — standard for cosine similarity
)
print(f"  Encoded {N:,} drugs in {time.time()-t1:.1f}s")
print(f"  Embedding matrix shape: {embeddings.shape}")

# ── save ──────────────────────────────────────────────────────────────────────

os.makedirs(GRAPH_DIR, exist_ok=True)

# node_embeddings.csv
emb_cols = [f"emb_{i}" for i in range(embed_dim)]
emb_df   = pd.DataFrame(embeddings, columns=emb_cols)
emb_df.insert(0, "node_idx", range(N))
emb_path = os.path.join(GRAPH_DIR, "node_embeddings.csv")
emb_df.to_csv(emb_path, index=False)

# node_features_combined.csv  (structural + embeddings)
struct_path = os.path.join(GRAPH_DIR, "node_features.csv")
if os.path.exists(struct_path):
    struct_df = pd.read_csv(struct_path)
    combined  = pd.concat([struct_df, emb_df.drop(columns=["node_idx"])], axis=1)
    comb_path = os.path.join(GRAPH_DIR, "node_features_combined.csv")
    combined.to_csv(comb_path, index=False)
    comb_size = os.path.getsize(comb_path) / 1024 / 1024
else:
    comb_path  = None
    comb_size  = 0
    print("  [warn] node_features.csv not found — run step4_build_graph.py first for combined output.")

emb_size  = os.path.getsize(emb_path) / 1024 / 1024

sep("OUTPUT")
print(f"  node_embeddings.csv          : {emb_size:.1f} MB  [{N} x {embed_dim}]")
if comb_path:
    n_struct = len(struct_df.columns) - 1  # minus node_idx
    print(f"  node_features_combined.csv   : {comb_size:.1f} MB  [{N} x {n_struct + embed_dim}]")
    print(f"    = {n_struct} structural + {embed_dim} embedding dimensions")
sep()
print(f"[embed] Done in {time.time()-t0:.1f}s total.")
