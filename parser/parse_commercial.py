"""
parse_commercial.py — extracts commercial and regulatory data.

Tables populated:
  salts                  — drug salt forms
  products               — marketed drug products (FDA NDC, DPD, EMA)
  drug_commercial_entities — packagers, manufacturers, international brands (merged)
  mixtures               — drug mixture / combination products
  prices                 — drug pricing data
  external_identifiers   — salt legacy DrugBank IDs (entity_type='salt')
"""
from config import NP
from utils import t, a, clean


def extract(drug_el, primary_id, state):
    salts, salt_ext_ids = _salts(drug_el, primary_id, state)
    return {
        "salts":                   salts,
        "products":                _products(drug_el, primary_id, state),
        "drug_commercial_entities": _commercial_entities(drug_el, primary_id),
        "mixtures":                _mixtures(drug_el, primary_id),
        "prices":                  _prices(drug_el, primary_id),
        "external_identifiers":    salt_ext_ids,
    }


# ── salts ─────────────────────────────────────────────────────────────────────

def _salts(drug_el, primary_id, state):
    salt_rows = []
    ext_id_rows = []

    salts_el = drug_el.find(f"{NP}salts")
    if salts_el is None:
        return salt_rows, ext_id_rows

    for salt in salts_el.findall(f"{NP}salt"):
        # Identify primary salt ID
        salt_id = None
        for sid_el in salt.findall(f"{NP}drugbank-id"):
            if sid_el.get("primary", "false").lower() == "true":
                salt_id = clean(sid_el.text)
                break
        if salt_id is None:
            ids = salt.findall(f"{NP}drugbank-id")
            salt_id = clean(ids[0].text) if ids else None
        if not salt_id:
            continue

        salt_rows.append({
            "salt_id":          salt_id,
            "drugbank_id":      primary_id,
            "name":             t(salt, "name"),
            "unii":             t(salt, "unii"),
            "cas_number":       t(salt, "cas-number"),
            "inchikey":         t(salt, "inchikey"),
            "average_mass":     t(salt, "average-mass"),
            "monoisotopic_mass": t(salt, "monoisotopic-mass"),
        })

        # All IDs for this salt go into external_identifiers
        for sid_el in salt.findall(f"{NP}drugbank-id"):
            v = clean(sid_el.text)
            if v:
                is_primary = sid_el.get("primary", "false").lower() == "true"
                ext_id_rows.append({
                    "entity_type": "salt",
                    "entity_id":   salt_id,
                    "resource":    "DrugBank" + (" (primary)" if is_primary else " (legacy)"),
                    "identifier":  v,
                })

    return salt_rows, ext_id_rows


# ── products ──────────────────────────────────────────────────────────────────

def _products(drug_el, primary_id, state):
    rows = []
    prods_el = drug_el.find(f"{NP}products")
    if prods_el is None:
        return rows

    for prod in prods_el.findall(f"{NP}product"):
        state.product_counter += 1
        rows.append({
            "product_id":           state.product_counter,
            "drugbank_id":          primary_id,
            "name":                 t(prod, "name"),
            "labeller":             t(prod, "labeller"),
            "ndc_id":               t(prod, "ndc-id"),
            "ndc_product_code":     t(prod, "ndc-product-code"),
            "dpd_id":               t(prod, "dpd-id"),
            "ema_product_code":     t(prod, "ema-product-code"),
            "ema_ma_number":        t(prod, "ema-ma-number"),
            "started_marketing_on": t(prod, "started-marketing-on"),
            "ended_marketing_on":   t(prod, "ended-marketing-on"),
            "dosage_form":          t(prod, "dosage-form"),
            "strength":             t(prod, "strength"),
            "route":                t(prod, "route"),
            "fda_application_number": t(prod, "fda-application-number"),
            "generic":              t(prod, "generic"),
            "over_the_counter":     t(prod, "over-the-counter"),
            "approved":             t(prod, "approved"),
            "country":              t(prod, "country"),
            "source":               t(prod, "source"),
        })
    return rows


# ── drug_commercial_entities (packagers + manufacturers + brands merged) ───────

def _commercial_entities(drug_el, primary_id):
    rows = []

    # Packagers
    pkgrs = drug_el.find(f"{NP}packagers")
    if pkgrs is not None:
        for pk in pkgrs.findall(f"{NP}packager"):
            name = t(pk, "name")
            if name:
                rows.append({
                    "drugbank_id": primary_id, "entity_type": "packager",
                    "name": name, "url": t(pk, "url"),
                    "company": None, "generic_flag": None,
                })

    # Manufacturers  (text content + generic attr + url attr)
    mfrs = drug_el.find(f"{NP}manufacturers")
    if mfrs is not None:
        for mf in mfrs.findall(f"{NP}manufacturer"):
            name = clean(mf.text)
            if name:
                rows.append({
                    "drugbank_id": primary_id, "entity_type": "manufacturer",
                    "name": name,
                    "url":  clean(mf.get("url")),
                    "company": None,
                    "generic_flag": clean(mf.get("generic")),
                })

    # International brands
    brands = drug_el.find(f"{NP}international-brands")
    if brands is not None:
        for br in brands.findall(f"{NP}international-brand"):
            name = t(br, "name")
            if name:
                rows.append({
                    "drugbank_id": primary_id, "entity_type": "brand",
                    "name": name, "url": None,
                    "company": t(br, "company"), "generic_flag": None,
                })

    return rows


# ── mixtures ──────────────────────────────────────────────────────────────────

def _mixtures(drug_el, primary_id):
    rows = []
    mixes = drug_el.find(f"{NP}mixtures")
    if mixes is not None:
        for mx in mixes.findall(f"{NP}mixture"):
            name = t(mx, "name")
            if name:
                rows.append({
                    "drugbank_id":             primary_id,
                    "name":                    name,
                    "ingredients":             t(mx, "ingredients"),
                    "supplemental_ingredients": t(mx, "supplemental-ingredients"),
                })
    return rows


# ── prices ────────────────────────────────────────────────────────────────────

def _prices(drug_el, primary_id):
    rows = []
    prices_el = drug_el.find(f"{NP}prices")
    if prices_el is None:
        return rows

    for price in prices_el.findall(f"{NP}price"):
        cost_el = price.find(f"{NP}cost")
        cost = None
        currency = None
        if cost_el is not None:
            cost     = clean(cost_el.text)
            currency = clean(cost_el.get("currency"))
        rows.append({
            "drugbank_id": primary_id,
            "description": t(price, "description"),
            "cost":        cost,
            "currency":    currency,
            "unit":        t(price, "unit"),
        })
    return rows
