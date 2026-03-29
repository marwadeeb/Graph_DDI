"""
parse_proteins.py — extracts drug–protein binding data (targets, enzymes,
                    carriers, transporters) and the underlying polypeptides.

Tables populated:
  interactants             — BE-ID binding entity records (deduplicated)
  drug_interactants        — drug ↔ interactant junction (role, position, actions inlined)
  polypeptides             — UniProt protein records (deduplicated by UniProt ID)
  interactant_polypeptides — interactant ↔ polypeptide junction
  polypeptide_attributes   — polypeptide synonyms, Pfam domains, GO classifiers (merged)
  external_identifiers     — polypeptide-level cross-database IDs (entity_type='polypeptide')
  references               — references cited in interactant entries (global dedup)
  reference_associations   — links refs to (drugbank_id, interactant_id) context
"""
from config import NP
from utils import t, a, clean, extract_ref_list

# Roles and their corresponding XML container/child tag pairs
_ROLES = [
    ("target",      "targets",      "target"),
    ("enzyme",      "enzymes",      "enzyme"),
    ("carrier",     "carriers",     "carrier"),
    ("transporter", "transporters", "transporter"),
]


def extract(drug_el, primary_id, state):
    results = {
        "interactants":             [],
        "drug_interactants":        [],
        "polypeptides":             [],
        "interactant_polypeptides": [],
        "polypeptide_attributes":   [],
        "external_identifiers":     [],
        "references":               [],
        "reference_associations":   [],
    }

    for role, container_tag, child_tag in _ROLES:
        container = drug_el.find(f"{NP}{container_tag}")
        if container is None:
            continue
        for item in container.findall(f"{NP}{child_tag}"):
            _process_interactant(item, primary_id, role, state, results)

    return results


# ── interactant processing ────────────────────────────────────────────────────

def _process_interactant(item_el, primary_id, role, state, out):
    interactant_id = t(item_el, "id")
    if not interactant_id:
        return

    # interactants table (deduplicated by BE-ID)
    if interactant_id not in state.interactants_seen:
        state.interactants_seen.add(interactant_id)
        out["interactants"].append({
            "interactant_id": interactant_id,
            "name":           t(item_el, "name"),
            "organism":       t(item_el, "organism"),
        })

    # Actions: collect all <action> children, pipe-delimited
    actions_el = item_el.find(f"{NP}actions")
    action_list = []
    if actions_el is not None:
        for act in actions_el.findall(f"{NP}action"):
            v = clean(act.text)
            if v:
                action_list.append(v)

    # drug_interactants junction
    out["drug_interactants"].append({
        "drugbank_id":        primary_id,
        "interactant_id":     interactant_id,
        "role":               role,
        "position":           clean(item_el.get("position")),
        "known_action":       t(item_el, "known-action"),
        "actions":            "|".join(action_list) if action_list else None,
        "inhibition_strength": t(item_el, "inhibition-strength"),   # enzyme only
        "induction_strength":  t(item_el, "induction-strength"),    # enzyme only
    })

    # References for this interactant entry
    refs_el = item_el.find(f"{NP}references")
    new_refs, ref_pks = extract_ref_list(refs_el, state)
    out["references"].extend(new_refs)
    for rpk in ref_pks:
        out["reference_associations"].append({
            "ref_pk":         rpk,
            "drugbank_id":    primary_id,
            "interactant_id": interactant_id,
        })

    # Polypeptides inside this interactant
    for poly_el in item_el.findall(f"{NP}polypeptide"):
        _process_polypeptide(poly_el, interactant_id, state, out)


# ── polypeptide processing ────────────────────────────────────────────────────

def _process_polypeptide(poly_el, interactant_id, state, out):
    poly_id = clean(poly_el.get("id"))
    if not poly_id:
        return

    # interactant_polypeptides (always write — same polypeptide under different interactants)
    out["interactant_polypeptides"].append({
        "interactant_id": interactant_id,
        "polypeptide_id": poly_id,
    })

    # polypeptides table (deduplicated by UniProt ID)
    if poly_id not in state.polypeptides_seen:
        state.polypeptides_seen.add(poly_id)

        organism_el = poly_el.find(f"{NP}organism")
        organism_name  = clean(organism_el.text) if organism_el is not None else None
        ncbi_tax_id    = clean(organism_el.get("ncbi-taxonomy-id")) if organism_el is not None else None

        aa_seq_el   = poly_el.find(f"{NP}amino-acid-sequence")
        gene_seq_el = poly_el.find(f"{NP}gene-sequence")

        out["polypeptides"].append({
            "polypeptide_id":      poly_id,
            "source":              clean(poly_el.get("source")),
            "name":                t(poly_el, "name"),
            "general_function":    t(poly_el, "general-function"),
            "specific_function":   t(poly_el, "specific-function"),
            "gene_name":           t(poly_el, "gene-name"),
            "locus":               t(poly_el, "locus"),
            "cellular_location":   t(poly_el, "cellular-location"),
            "transmembrane_regions": t(poly_el, "transmembrane-regions"),
            "signal_regions":      t(poly_el, "signal-regions"),
            "theoretical_pi":      t(poly_el, "theoretical-pi"),
            "molecular_weight":    t(poly_el, "molecular-weight"),
            "chromosome_location": t(poly_el, "chromosome-location"),
            "organism":            organism_name,
            "ncbi_taxonomy_id":    ncbi_tax_id,
            "amino_acid_sequence": clean(aa_seq_el.text) if aa_seq_el is not None else None,
            "gene_sequence":       clean(gene_seq_el.text) if gene_seq_el is not None else None,
        })

        # polypeptide_attributes: synonyms
        syns_el = poly_el.find(f"{NP}synonyms")
        if syns_el is not None:
            for syn in syns_el.findall(f"{NP}synonym"):
                v = clean(syn.text)
                if v:
                    out["polypeptide_attributes"].append({
                        "polypeptide_id": poly_id,
                        "attr_type": "synonym", "value": v, "value2": None,
                    })

        # polypeptide_attributes: Pfam domains
        pfams_el = poly_el.find(f"{NP}pfams")
        if pfams_el is not None:
            for pfam in pfams_el.findall(f"{NP}pfam"):
                pid  = t(pfam, "identifier")
                pname = t(pfam, "name")
                if pid:
                    out["polypeptide_attributes"].append({
                        "polypeptide_id": poly_id,
                        "attr_type": "pfam", "value": pid, "value2": pname,
                    })

        # polypeptide_attributes: GO classifiers
        go_el = poly_el.find(f"{NP}go-classifiers")
        if go_el is not None:
            for go in go_el.findall(f"{NP}go-classifier"):
                cat  = t(go, "category")
                desc = t(go, "description")
                if cat:
                    out["polypeptide_attributes"].append({
                        "polypeptide_id": poly_id,
                        "attr_type": "go_classifier", "value": cat, "value2": desc,
                    })

        # external_identifiers: polypeptide-level
        ext_ids = poly_el.find(f"{NP}external-identifiers")
        if ext_ids is not None:
            for ei in ext_ids.findall(f"{NP}external-identifier"):
                resource   = t(ei, "resource")
                identifier = t(ei, "identifier")
                if resource and identifier:
                    out["external_identifiers"].append({
                        "entity_type": "polypeptide",
                        "entity_id":   poly_id,
                        "resource":    resource,
                        "identifier":  identifier,
                    })
