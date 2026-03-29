"""
parse_pathways.py — extracts pathway and reaction data.

Tables populated:
  pathways        — deduplicated pathway entities (by smpdb_id)
  pathway_members — drugs + enzyme uniprot-ids within pathways (member_type discriminator)
  reactions       — metabolic reaction records; reaction enzymes serialised as
                    pipe-delimited triples  "drugbank_id|name|uniprot_id|..."
"""
from config import NP
from utils import t, clean


def extract(drug_el, primary_id, state):
    new_pathways, pathway_members = _pathways(drug_el, state)
    reactions = _reactions(drug_el, primary_id, state)
    return {
        "pathways":        new_pathways,
        "pathway_members": pathway_members,
        "reactions":       reactions,
    }


# ── pathways ──────────────────────────────────────────────────────────────────

def _pathways(drug_el, state):
    new_pathway_rows = []
    member_rows = []

    pways = drug_el.find(f"{NP}pathways")
    if pways is None:
        return new_pathway_rows, member_rows

    for pw in pways.findall(f"{NP}pathway"):
        smpdb_id = t(pw, "smpdb-id")
        if not smpdb_id:
            continue

        # Write pathway entity only once (deduplicate by smpdb_id)
        if smpdb_id not in state.pathways_seen:
            state.pathways_seen.add(smpdb_id)
            new_pathway_rows.append({
                "smpdb_id": smpdb_id,
                "name":     t(pw, "name"),
                "category": t(pw, "category"),
            })

        # pathway_members: drugs listed in this pathway
        drugs_el = pw.find(f"{NP}drugs")
        if drugs_el is not None:
            for d in drugs_el.findall(f"{NP}drug"):
                mid = t(d, "drugbank-id")
                if mid:
                    member_rows.append({
                        "smpdb_id":   smpdb_id,
                        "member_type": "drug",
                        "member_id":   mid,
                        "member_name": t(d, "name"),
                    })

        # pathway_members: enzymes (only uniprot-id; no name in XSD)
        enzymes_el = pw.find(f"{NP}enzymes")
        if enzymes_el is not None:
            for uid_el in enzymes_el.findall(f"{NP}uniprot-id"):
                uid = clean(uid_el.text)
                if uid:
                    member_rows.append({
                        "smpdb_id":   smpdb_id,
                        "member_type": "enzyme",
                        "member_id":   uid,
                        "member_name": None,
                    })

    return new_pathway_rows, member_rows


# ── reactions ─────────────────────────────────────────────────────────────────

def _reactions(drug_el, primary_id, state):
    rows = []
    rxns = drug_el.find(f"{NP}reactions")
    if rxns is None:
        return rows

    for rxn in rxns.findall(f"{NP}reaction"):
        # Left / right elements
        left  = rxn.find(f"{NP}left-element")
        right = rxn.find(f"{NP}right-element")

        # Serialize enzymes as pipe-delimited triples: "db_id|name|uniprot_id"
        enz_parts = []
        enz_el = rxn.find(f"{NP}enzymes")
        if enz_el is not None:
            for enz in enz_el.findall(f"{NP}enzyme"):
                eid   = t(enz, "drugbank-id") or ""
                ename = t(enz, "name") or ""
                euid  = t(enz, "uniprot-id") or ""
                enz_parts.append(f"{eid}|{ename}|{euid}")

        state.reaction_counter += 1
        rows.append({
            "reaction_id":       state.reaction_counter,
            "drugbank_id":       primary_id,
            "sequence":          t(rxn, "sequence"),
            "left_element_id":   t(left, "drugbank-id") if left is not None else None,
            "left_element_name": t(left, "name")        if left is not None else None,
            "right_element_id":  t(right, "drugbank-id") if right is not None else None,
            "right_element_name": t(right, "name")       if right is not None else None,
            "enzymes":           "||".join(enz_parts) if enz_parts else None,
        })

    return rows
