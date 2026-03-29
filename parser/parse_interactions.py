"""
parse_interactions.py — extracts drug-drug interactions and SNP data.

Tables populated:
  drug_interactions  — directed DDI edges (both A→B and B→A preserved; descriptions differ)
  drug_snp_data      — SNP pharmacogenomic effects + SNP adverse reactions (merged with snp_type)
"""
from config import NP
from utils import t, clean


def extract(drug_el, primary_id, state):
    return {
        "drug_interactions": _drug_interactions(drug_el, primary_id),
        "drug_snp_data":     _snp_data(drug_el, primary_id),
    }


# ── drug_interactions ─────────────────────────────────────────────────────────

def _drug_interactions(drug_el, primary_id):
    rows = []
    ddi_el = drug_el.find(f"{NP}drug-interactions")
    if ddi_el is None:
        return rows
    for ddi in ddi_el.findall(f"{NP}drug-interaction"):
        other_id = t(ddi, "drugbank-id")
        if other_id:
            rows.append({
                "drugbank_id":           primary_id,
                "interacting_drugbank_id": other_id,
                "description":           t(ddi, "description"),
            })
    return rows


# ── drug_snp_data (effects + adverse reactions merged) ────────────────────────

def _snp_data(drug_el, primary_id):
    rows = []

    # SNP effects
    snp_eff = drug_el.find(f"{NP}snp-effects")
    if snp_eff is not None:
        for eff in snp_eff.findall(f"{NP}effect"):
            rows.append({
                "drugbank_id":           primary_id,
                "snp_type":              "effect",
                "protein_name":          t(eff, "protein-name"),
                "gene_symbol":           t(eff, "gene-symbol"),
                "uniprot_id":            t(eff, "uniprot-id"),
                "rs_id":                 t(eff, "rs-id"),
                "allele":                t(eff, "allele"),
                "defining_change":       t(eff, "defining-change"),
                "adverse_reaction":      None,
                "description":           t(eff, "description"),
                "pubmed_id":             t(eff, "pubmed-id"),
            })

    # SNP adverse drug reactions
    snp_adr = drug_el.find(f"{NP}snp-adverse-drug-reactions")
    if snp_adr is not None:
        for rxn in snp_adr.findall(f"{NP}reaction"):
            rows.append({
                "drugbank_id":           primary_id,
                "snp_type":              "adverse_reaction",
                "protein_name":          t(rxn, "protein-name"),
                "gene_symbol":           t(rxn, "gene-symbol"),
                "uniprot_id":            t(rxn, "uniprot-id"),
                "rs_id":                 t(rxn, "rs-id"),
                "allele":                t(rxn, "allele"),
                "defining_change":       None,
                "adverse_reaction":      t(rxn, "adverse-reaction"),
                "description":           t(rxn, "description"),
                "pubmed_id":             t(rxn, "pubmed-id"),
            })

    return rows
