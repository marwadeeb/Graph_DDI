"""
parse_pharmacological.py — extracts pharmacological classification and regulatory data.

Tables populated:
  categories       — normalized MeSH category entities (deduplicated globally)
  drug_categories  — drug ↔ category junction
  dosages          — dosage forms, routes, strengths
  atc_codes        — ATC classification with 4-level hierarchy
  patents          — patent records
"""
from config import NP
from utils import t, clean


def extract(drug_el, primary_id, state):
    new_cats, drug_cats = _categories(drug_el, primary_id, state)
    return {
        "categories":     new_cats,
        "drug_categories": drug_cats,
        "dosages":         _dosages(drug_el, primary_id),
        "atc_codes":       _atc_codes(drug_el, primary_id),
        "patents":         _patents(drug_el, primary_id),
    }


# ── categories (globally deduplicated) ───────────────────────────────────────

def _categories(drug_el, primary_id, state):
    new_cat_rows = []
    drug_cat_rows = []

    cats_el = drug_el.find(f"{NP}categories")
    if cats_el is None:
        return new_cat_rows, drug_cat_rows

    for cat in cats_el.findall(f"{NP}category"):
        name    = t(cat, "category")
        mesh_id = t(cat, "mesh-id")
        if not name:
            continue

        # Dedup key: normalised name + mesh_id (mesh_id may be empty)
        key = (name.lower(), mesh_id or "")
        if key not in state.cats_seen:
            state.cat_counter += 1
            cid = state.cat_counter
            state.cats_seen[key] = cid
            new_cat_rows.append({
                "category_id":   cid,
                "category_name": name,
                "mesh_id":       mesh_id,
            })
        drug_cat_rows.append({
            "drugbank_id": primary_id,
            "category_id": state.cats_seen[key],
        })

    return new_cat_rows, drug_cat_rows


# ── dosages ───────────────────────────────────────────────────────────────────

def _dosages(drug_el, primary_id):
    rows = []
    dos_el = drug_el.find(f"{NP}dosages")
    if dos_el is None:
        return rows
    for dos in dos_el.findall(f"{NP}dosage"):
        rows.append({
            "drugbank_id": primary_id,
            "form":        t(dos, "form"),
            "route":       t(dos, "route"),
            "strength":    t(dos, "strength"),
        })
    return rows


# ── atc_codes ─────────────────────────────────────────────────────────────────

def _atc_codes(drug_el, primary_id):
    rows = []
    atc_list = drug_el.find(f"{NP}atc-codes")
    if atc_list is None:
        return rows

    for atc in atc_list.findall(f"{NP}atc-code"):
        code   = clean(atc.get("code"))
        levels = atc.findall(f"{NP}level")
        # XSD guarantees exactly 4 levels; guard against malformed data
        def lc(i): return clean(levels[i].get("code")) if i < len(levels) else None
        def ln(i): return clean(levels[i].text) if i < len(levels) else None

        rows.append({
            "drugbank_id": primary_id,
            "atc_code":    code,
            # level[0] = most-specific pharmacological subgroup (e.g. B01AE)
            # level[3] = top anatomical group (e.g. B)
            "l1_code": lc(0), "l1_name": ln(0),
            "l2_code": lc(1), "l2_name": ln(1),
            "l3_code": lc(2), "l3_name": ln(2),
            "l4_code": lc(3), "l4_name": ln(3),
        })
    return rows


# ── patents ───────────────────────────────────────────────────────────────────

def _patents(drug_el, primary_id):
    rows = []
    pats = drug_el.find(f"{NP}patents")
    if pats is None:
        return rows
    for pat in pats.findall(f"{NP}patent"):
        rows.append({
            "drugbank_id":       primary_id,
            "number":            t(pat, "number"),
            "country":           t(pat, "country"),
            "approved_date":     t(pat, "approved"),
            "expires_date":      t(pat, "expires"),
            "pediatric_extension": t(pat, "pediatric-extension"),
        })
    return rows
