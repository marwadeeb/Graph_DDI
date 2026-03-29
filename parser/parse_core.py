"""
parse_core.py — extracts core drug data.

Tables populated:
  drugs                — one row per drug (all scalar fields + inlined classification)
  drug_ids             — all DrugBank IDs (primary + secondary/legacy)
  drug_attributes      — multi-valued string lists (groups, synonyms, organisms,
                          food_interactions, sequences, ahfs_codes, pdb_entries,
                          classification alt_parents, substituents)
  drug_properties      — calculated + experimental properties (merged)
  external_identifiers — drug-level cross-database IDs and external links
"""
from config import NP
from utils import t, a, clean


def extract(drug_el, primary_id, state):
    """Return dict[table_name -> list[row_dict]]."""
    return {
        "drugs":                _drugs(drug_el, primary_id),
        "drug_ids":             _drug_ids(drug_el, primary_id),
        "drug_attributes":      _drug_attributes(drug_el, primary_id),
        "drug_properties":      _drug_properties(drug_el, primary_id),
        "external_identifiers": _external_identifiers(drug_el, primary_id),
    }


# ── drugs ─────────────────────────────────────────────────────────────────────

def _drugs(drug_el, primary_id):
    cls_el = drug_el.find(f"{NP}classification")
    cls = {}
    if cls_el is not None:
        cls = {
            "classification_description":   t(cls_el, "description"),
            "classification_direct_parent":  t(cls_el, "direct-parent"),
            "classification_kingdom":        t(cls_el, "kingdom"),
            "classification_superclass":     t(cls_el, "superclass"),
            "classification_class":          t(cls_el, "class"),
            "classification_subclass":       t(cls_el, "subclass"),
        }

    row = {
        "drugbank_id":      primary_id,
        "name":             t(drug_el, "name"),
        "drug_type":        drug_el.get("type"),
        "description":      t(drug_el, "description"),
        "cas_number":       t(drug_el, "cas-number"),
        "unii":             t(drug_el, "unii"),
        "average_mass":     t(drug_el, "average-mass"),
        "monoisotopic_mass": t(drug_el, "monoisotopic-mass"),
        "state":            t(drug_el, "state"),
        "indication":               t(drug_el, "indication"),
        "pharmacodynamics":         t(drug_el, "pharmacodynamics"),
        "mechanism_of_action":      t(drug_el, "mechanism-of-action"),
        "toxicity":                 t(drug_el, "toxicity"),
        "metabolism":               t(drug_el, "metabolism"),
        "absorption":               t(drug_el, "absorption"),
        "half_life":                t(drug_el, "half-life"),
        "protein_binding":          t(drug_el, "protein-binding"),
        "route_of_elimination":     t(drug_el, "route-of-elimination"),
        "volume_of_distribution":   t(drug_el, "volume-of-distribution"),
        "clearance":                t(drug_el, "clearance"),
        "synthesis_reference":      t(drug_el, "synthesis-reference"),
        "fda_label_url":            t(drug_el, "fda-label"),
        "msds_url":                 t(drug_el, "msds"),
        "created_date":  drug_el.get("created"),
        "updated_date":  drug_el.get("updated"),
        # Classification scalars (empty dict means all will be None)
        "classification_description":  cls.get("classification_description"),
        "classification_direct_parent": cls.get("classification_direct_parent"),
        "classification_kingdom":      cls.get("classification_kingdom"),
        "classification_superclass":   cls.get("classification_superclass"),
        "classification_class":        cls.get("classification_class"),
        "classification_subclass":     cls.get("classification_subclass"),
    }
    return [row]


# ── drug_ids ──────────────────────────────────────────────────────────────────

def _drug_ids(drug_el, primary_id):
    rows = []
    for id_el in drug_el.findall(f"{NP}drugbank-id"):
        val = clean(id_el.text)
        if val:
            rows.append({
                "drugbank_id": primary_id,
                "legacy_id":   val,
                "is_primary":  id_el.get("primary", "false").lower() == "true",
            })
    return rows


# ── drug_attributes ───────────────────────────────────────────────────────────

def _attr(did, atype, value, v2=None, v3=None):
    return {"drugbank_id": did, "attr_type": atype,
            "value": value, "value2": v2, "value3": v3}


def _drug_attributes(drug_el, primary_id):
    rows = []
    did = primary_id

    # Groups
    grps = drug_el.find(f"{NP}groups")
    if grps is not None:
        for g in grps.findall(f"{NP}group"):
            v = clean(g.text)
            if v:
                rows.append(_attr(did, "group", v))

    # Synonyms (with language + coder attributes)
    syns = drug_el.find(f"{NP}synonyms")
    if syns is not None:
        for s in syns.findall(f"{NP}synonym"):
            v = clean(s.text)
            if v:
                rows.append(_attr(did, "synonym", v,
                                  clean(s.get("language")),
                                  clean(s.get("coder"))))

    # Affected organisms
    ao = drug_el.find(f"{NP}affected-organisms")
    if ao is not None:
        for o in ao.findall(f"{NP}affected-organism"):
            v = clean(o.text)
            if v:
                rows.append(_attr(did, "affected_organism", v))

    # Food interactions
    fi = drug_el.find(f"{NP}food-interactions")
    if fi is not None:
        for f_ in fi.findall(f"{NP}food-interaction"):
            v = clean(f_.text)
            if v:
                rows.append(_attr(did, "food_interaction", v))

    # Sequences (biotech drugs — FASTA strings)
    seqs = drug_el.find(f"{NP}sequences")
    if seqs is not None:
        for seq in seqs.findall(f"{NP}sequence"):
            v = clean(seq.text)
            if v:
                rows.append(_attr(did, "sequence", v, clean(seq.get("format"))))

    # AHFS codes
    ahfs = drug_el.find(f"{NP}ahfs-codes")
    if ahfs is not None:
        for code in ahfs.findall(f"{NP}ahfs-code"):
            v = clean(code.text)
            if v:
                rows.append(_attr(did, "ahfs_code", v))

    # PDB entries
    pdb = drug_el.find(f"{NP}pdb-entries")
    if pdb is not None:
        for entry in pdb.findall(f"{NP}pdb-entry"):
            v = clean(entry.text)
            if v:
                rows.append(_attr(did, "pdb_entry", v))

    # Classification multi-valued: alternative-parents + substituents
    cls_el = drug_el.find(f"{NP}classification")
    if cls_el is not None:
        for ap in cls_el.findall(f"{NP}alternative-parent"):
            v = clean(ap.text)
            if v:
                rows.append(_attr(did, "classification_alt_parent", v))
        for sub in cls_el.findall(f"{NP}substituent"):
            v = clean(sub.text)
            if v:
                rows.append(_attr(did, "classification_substituent", v))

    return rows


# ── drug_properties ───────────────────────────────────────────────────────────

def _drug_properties(drug_el, primary_id):
    rows = []

    # Calculated properties
    calc = drug_el.find(f"{NP}calculated-properties")
    if calc is not None:
        for prop in calc.findall(f"{NP}property"):
            rows.append({
                "drugbank_id":    primary_id,
                "property_class": "calculated",
                "kind":           t(prop, "kind"),
                "value":          t(prop, "value"),
                "source":         t(prop, "source"),
            })

    # Experimental properties
    exp = drug_el.find(f"{NP}experimental-properties")
    if exp is not None:
        for prop in exp.findall(f"{NP}property"):
            rows.append({
                "drugbank_id":    primary_id,
                "property_class": "experimental",
                "kind":           t(prop, "kind"),
                "value":          t(prop, "value"),
                "source":         t(prop, "source"),
            })

    return rows


# ── external_identifiers (drug-level) ─────────────────────────────────────────

def _external_identifiers(drug_el, primary_id):
    rows = []

    # Cross-database identifiers (UniProtKB, ChEMBL, PubChem, KEGG, etc.)
    ext_ids = drug_el.find(f"{NP}external-identifiers")
    if ext_ids is not None:
        for ei in ext_ids.findall(f"{NP}external-identifier"):
            resource   = t(ei, "resource")
            identifier = t(ei, "identifier")
            if resource and identifier:
                rows.append({
                    "entity_type": "drug",
                    "entity_id":   primary_id,
                    "resource":    resource,
                    "identifier":  identifier,
                })

    # External links (RxList, PDRhealth, Drugs.com) — stored as identifier=url
    ext_links = drug_el.find(f"{NP}external-links")
    if ext_links is not None:
        for lnk in ext_links.findall(f"{NP}external-link"):
            resource = t(lnk, "resource")
            url      = t(lnk, "url")
            if resource and url:
                rows.append({
                    "entity_type": "drug_link",
                    "entity_id":   primary_id,
                    "resource":    resource,
                    "identifier":  url,
                })

    return rows
