"""
parse_references.py — extracts drug-level general references.

Tables populated:
  references            — globally deduplicated reference records
                          (articles, textbooks, links, attachments merged)
  reference_associations — links each reference to its context:
                            interactant_id=NULL means drug general reference;
                            interactant_id=BE-ID means interactant reference
                            (interactant refs are added by parse_proteins.py)
"""
from config import NP
from utils import extract_ref_list


def extract(drug_el, primary_id, state):
    refs_el = drug_el.find(f"{NP}general-references")
    new_refs, ref_pks = extract_ref_list(refs_el, state)

    assocs = [
        {"ref_pk": rpk, "drugbank_id": primary_id, "interactant_id": None}
        for rpk in ref_pks
    ]

    return {
        "references":            new_refs,
        "reference_associations": assocs,
    }
