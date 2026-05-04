"""
ddi_query.py
------------------
Drug name resolution utilities for DDI detection.

Provides:
  - get_drugs_df()   — load approved drugs table (drugbank_id, name)
  - get_synonym_map() — synonym -> (drugbank_id, canonical_name) mapping
  - resolve_drug()   — resolve a drug name / DrugBank ID to (id, canonical_name)

Used by app.py for all /api/check and /api/chat requests.
"""

import os
import pandas as pd

# Load .env if present (for local dev)
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

_load_env()

# ---------------------------------------------------------------------------
WORKING_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROVED_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")

_drugs_df         = None
_synonym_map      = None  # synonym (lowercase) -> (drugbank_id, canonical_name)
_brand_components = None  # brand_lower -> [(drugbank_id, canonical_name), ...]
_brand_display    = None  # brand_lower -> original-case display name


def get_drugs_df():
    global _drugs_df
    if _drugs_df is None:
        _drugs_df = pd.read_csv(
            os.path.join(APPROVED_DIR, "drugs.csv"),
            usecols=["drugbank_id", "name"]
        )
    return _drugs_df


def get_brand_components(brand_name: str) -> list:
    """Return all drugs sharing a brand name (for combination products).
    e.g. get_brand_components("Viadur") -> [("DB00041", "Leuprolide"), ("DB00281", "Lidocaine")]
    Returns [] if not a brand or brand maps to a single drug.
    """
    get_synonym_map()  # ensure loaded
    if not _brand_components:
        return []
    return _brand_components.get(brand_name.strip().lower(), [])


def get_synonym_map():
    global _synonym_map, _brand_components, _brand_display
    if _synonym_map is None:
        df = get_drugs_df()
        name_lookup = dict(zip(df["drugbank_id"], df["name"]))
        syn_df = pd.read_csv(
            os.path.join(APPROVED_DIR, "drug_attributes.csv"),
            usecols=["drugbank_id", "attr_type", "value"]
        )
        syn_df = syn_df[syn_df["attr_type"] == "synonym"]
        _synonym_map = {}
        for _, row in syn_df.iterrows():
            key = str(row["value"]).strip().lower()
            did = row["drugbank_id"]
            if did in name_lookup:
                _synonym_map[key] = (did, name_lookup[did])
        
        _brand_components = {}
        _brand_display    = {}
        try:
            prod_df = pd.read_csv(
                os.path.join(APPROVED_DIR, "products.csv"),
                usecols=["drugbank_id", "name"]
            )
            prod_df = prod_df.dropna(subset=["name"])
            for _, row in prod_df.iterrows():
                raw  = str(row["name"]).strip()
                key  = raw.lower()
                did  = row["drugbank_id"]
                if not key or did not in name_lookup:
                    continue
                canonical = name_lookup[did]
                # Track original-case display name (first occurrence wins)
                if key not in _brand_display:
                    _brand_display[key] = raw
                # Build per-brand component list (for combination products)
                if key not in _brand_components:
                    _brand_components[key] = []
                existing_ids = [x[0] for x in _brand_components[key]]
                if did not in existing_ids:
                    _brand_components[key].append((did, canonical))
                # Synonym map: first drug per brand wins (primary resolution)
                if key not in _synonym_map:
                    _synonym_map[key] = (did, canonical)
        except Exception as e:
            print(f"  [warn] Could not load brand names from products.csv: {e}")
    return _synonym_map


def resolve_drug(query: str) -> tuple:
    """
    Resolve a drug name or DrugBank ID to (drugbank_id, canonical_name).
    Accepts: 'DB00682', 'Warfarin', 'warfarin', 'aspirin' (synonym).
    Raises ValueError if not found in the approved drug set.
    """
    df = get_drugs_df()
    q  = query.strip()

    if not q:
        raise ValueError("Drug not found in approved set: ''")

    # Exact DrugBank ID
    if q.upper().startswith("DB"):
        row = df[df["drugbank_id"] == q.upper()]
        if not row.empty:
            return row.iloc[0]["drugbank_id"], row.iloc[0]["name"]

    # Case-insensitive exact name match
    row = df[df["name"].str.lower() == q.lower()]
    if not row.empty:
        return row.iloc[0]["drugbank_id"], row.iloc[0]["name"]

    # Partial name match (substring)
    row = df[df["name"].str.lower().str.contains(q.lower(), na=False)]
    if not row.empty:
        if len(row) > 1:
            print(f"  [warn] Multiple matches for '{q}', using first: {row.iloc[0]['name']}")
        return row.iloc[0]["drugbank_id"], row.iloc[0]["name"]

    # Synonym lookup (e.g. "aspirin" -> "Acetylsalicylic acid")
    syn_map = get_synonym_map()
    if q.lower() in syn_map:
        did, canonical = syn_map[q.lower()]
        return did, canonical

    raise ValueError(f"Drug not found in approved set: '{query}'")
