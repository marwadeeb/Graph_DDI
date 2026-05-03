"""
build_pyg_hetero.py
Augments ddi_graph.pt with drug-target relationships to produce hetero_ddi_graph.pt

Node types: drug (4795, 980-dim), protein (2708 human proteins, 5-dim)
Edge types: (drug,ddi,drug), (drug,targets,protein), (protein,rev_targets,drug)
Protein features: [is_target, is_enzyme, is_transporter, is_carrier, log_degree]

Run from repo root with venv active:
    python pipeline/build_pyg_hetero.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP3_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")
GRAPH_DIR = os.path.join(WORKING_DIR, "data", "step4_graph")
DDI_GRAPH_PATH = os.path.join(GRAPH_DIR, "ddi_graph.pt")
HETERO_GRAPH_PATH = os.path.join(GRAPH_DIR, "hetero_ddi_graph.pt")

#Load existing homogeneous graph
homo = torch.load(DDI_GRAPH_PATH, map_location="cpu", weights_only=False)
num_drugs = homo.num_nodes
drug_ids = homo.drugbank_ids
drug_names = homo.drug_names
id_to_idx = {db_id: i for i, db_id in enumerate(drug_ids)}
drug_x = homo.x
ddi_edge_idx = homo.edge_index
print(f"Drug nodes: {num_drugs:,} | DDI pairs: {ddi_edge_idx.shape[1]//2:,} | Drug feat dim: {drug_x.shape[1]}")

#Load drug-protein interaction tables
drug_ints = pd.read_csv(os.path.join(STEP3_DIR, "drug_interactants.csv"), dtype=str)
interactants = pd.read_csv(os.path.join(STEP3_DIR, "interactants.csv"), dtype=str)
iact_poly = pd.read_csv(os.path.join(STEP3_DIR, "interactant_polypeptides.csv"), dtype=str)

human_iacts = interactants[interactants["organism"] == "Humans"][["interactant_id"]].drop_duplicates()
merged = (
    drug_ints
    .merge(iact_poly, on="interactant_id")
    .merge(human_iacts, on="interactant_id")
    [["drugbank_id", "polypeptide_id", "role"]]
    .drop_duplicates(subset=["drugbank_id", "polypeptide_id", "role"])
)
merged = merged[merged["drugbank_id"].isin(id_to_idx)]
print(f"Drug-protein pairs (human): {len(merged):,} | Drugs with targets: {merged['drugbank_id'].nunique():,} | Proteins: {merged['polypeptide_id'].nunique():,}")

#Build protein node index and features
all_proteins = sorted(merged["polypeptide_id"].unique())
num_proteins = len(all_proteins)
prot_to_idx = {p: i for i, p in enumerate(all_proteins)}
ROLES = ["target", "enzyme", "transporter", "carrier"]

role_mat = np.zeros((num_proteins, len(ROLES)), dtype=np.float32)
for _, row in merged.iterrows():
    if row["role"] in ROLES:
        role_mat[prot_to_idx[row["polypeptide_id"]], ROLES.index(row["role"])] = 1.0

degree = merged.drop_duplicates(["polypeptide_id", "drugbank_id"]).groupby("polypeptide_id").size()
max_deg = degree.max()
log_deg = np.array(
    [np.log1p(degree.get(p, 0)) / np.log1p(max_deg) for p in all_proteins], dtype=np.float32
).reshape(-1, 1)

protein_x = torch.tensor(np.concatenate([role_mat, log_deg], axis=1))
print(f"Protein nodes: {num_proteins:,} | Features: {protein_x.shape[1]}")

#Build drug-protein edge index
dp_pairs = merged[["drugbank_id", "polypeptide_id"]].drop_duplicates()
drug_protein_src = torch.tensor([id_to_idx[d] for d in dp_pairs["drugbank_id"]], dtype=torch.long)
drug_protein_dst = torch.tensor([prot_to_idx[p] for p in dp_pairs["polypeptide_id"]], dtype=torch.long)
drug_protein_edge_idx = torch.stack([drug_protein_src, drug_protein_dst], dim=0)
protein_drug_edge_idx = torch.stack([drug_protein_dst, drug_protein_src], dim=0)
print(f"Drug→protein edges: {drug_protein_edge_idx.shape[1]:,}")

#Assemble HeteroData
data = HeteroData()
data["drug"].x = drug_x
data["drug"].num_nodes = num_drugs
data["drug"].drugbank_ids = drug_ids
data["drug"].drug_names = drug_names
data["protein"].x = protein_x
data["protein"].num_nodes = num_proteins
data["protein"].polypeptide_ids = all_proteins
data["drug", "ddi", "drug"].edge_index = ddi_edge_idx
data["drug", "targets", "protein"].edge_index = drug_protein_edge_idx
data["protein", "rev_targets", "drug"].edge_index = protein_drug_edge_idx

#Sanity checks
assert data["drug"].x.shape == (4795, 980)
assert data["protein"].x.shape[1] == 5
assert not data["drug"].x.isnan().any()
assert not data["protein"].x.isnan().any()
assert data["drug", "ddi", "drug"].edge_index.max() < 4795
assert data["drug", "targets", "protein"].edge_index[0].max() < 4795
assert data["drug", "targets", "protein"].edge_index[1].max() < num_proteins
print("All checks passed.")

#Save
os.makedirs(GRAPH_DIR, exist_ok=True)
torch.save(data, HETERO_GRAPH_PATH)
print(f"Saved: hetero_ddi_graph.pt ({os.path.getsize(HETERO_GRAPH_PATH)/1e6:.1f} MB)")
