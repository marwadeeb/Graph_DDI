# Model Architecture — DDI Checker

GNN model, node features, PyG data objects, and baseline details.

---

## GNN Model — HeteroGraphSAGE + NCN

| Component | Detail |
|---|---|
| **Model** | HeteroGraphSAGE + EnhancedLinkPredictor (NCN-style decoder) |
| **Node types** | Drug (4,795 · 980-dim features), Protein (2,708 human proteins · 5-dim) |
| **Edge types** | `(drug, ddi, drug)`, `(drug, targets, protein)`, `(protein, rev_targets, drug)` |
| **Encoder** | 3 × HeteroConv(SAGEConv), hidden=256, out=64, dropout=0.3 |
| **Decoder** | NCN pooling: `[z_u ‖ z_v ‖ mean(shared DDI neighbours) ‖ mean(shared proteins)]` |
| **Loss** | nnPU (non-negative PU loss — handles unlabelled negatives) |
| **Prior π** | 0.0717 (observed positive rate in training set) |
| **Threshold** | 0.43 (set on validation set) |
| **Warm AUROC** | **0.9738** · Warm AUPR **0.9589** |
| **Cold AUROC** | **0.9175** · Cold AUPR **0.8824** |

Reference: Wang et al., "Neural Common Neighbor with Completion for Link Prediction", ICLR 2024.

Training notebook: [`pipeline/hetero_model.ipynb`](../pipeline/hetero_model.ipynb)

---

## Node Features (980-dim)

### Step 4a — Structural features (212-dim)

| Group | Count | Features | Source |
|---|---|---|---|
| A — Masses | 2 | `average_mass`, `monoisotopic_mass` | `drugs.csv` |
| B — Type & state | 4 | `is_biotech`, `state_solid/liquid/gas` | `drugs.csv` |
| C — Calculated props | 17 | logP, logS, MW, HBD, HBA, RotB, PSA, charge, rings, bioavailability, Rule of Five, Ghose, MDDR-like, refractivity, polarizability, pKa acid/basic | `drug_properties.csv` |
| D — Experimental props | 6 | logP, logS, melting point, boiling point, water solubility, pKa | `drug_properties.csv` |
| E — Group flags | 5 | `is_withdrawn`, `is_investigational`, `is_vet_approved`, `is_nutraceutical`, `is_illicit` | `drug_attributes.csv` |
| F — Counts | 11 | n_targets, n_enzymes, n_carriers, n_transporters, n_categories, n_atc_codes, n_patents, n_products, n_food_interactions, n_synonyms, n_pathways | Interaction tables |
| G — ATC anatomical | 14 | `atc_A` ... `atc_V` one-hot | `atc_codes.csv` |
| H — ClassyFire | 12 | `kingdom_organic/inorganic`; top-10 superclass one-hot | `drugs.csv` |
| I — MeSH categories | 50 | Top-50 therapeutic categories multi-hot | `drug_categories.csv` |
| J — Pathways | 49 | Top-50 SMPDB pathways multi-hot | `pathway_members.csv` |
| K — Sequence | 21 | `seq_length` + 20 amino acid percentages | `drug_attributes.csv` |
| L — CYP enzyme roles | 21 | Binary flags: `cyp{subtype}_{role}` for 3A4, 2D6, 2C9, 2C19, 1A2, 2B6, 2E1 × substrate/inhibitor/inducer | `drug_interactants.csv` |

Missing continuous → median imputation + standardization. Binary/count/one-hot → 0-filled.

### Step 4b — Text embeddings (768-dim)

`pritamdeka/S-PubMedBert-MS-MARCO` encodes: name + description + indication + mechanism of action + pharmacodynamics + toxicity + metabolism + absorption + food interactions + MeSH categories + ATC subgroups. L2-normalized, 768-dim.

| File | Shape | Description |
|---|---|---|
| `node_embeddings.csv` | 4,795 × 769 | `node_idx` + 768 PubMedBERT dims |
| `node_features_combined.csv` | 4,795 × 981 | `node_idx` + 212 structural + 768 text = **980 total** |

---

## PyG Data Objects

### Homogeneous graph (`ddi_graph.pt`)

```python
data = torch.load("data/step4_graph/ddi_graph.pt")
# data.x           [4795, 980]     float32 — node features
# data.edge_index  [2, 1648498]    long    — COO both directions
# data.edge_attr   [1648498, 1]    long    — interaction_id per edge
# data.drugbank_ids list of 4795 IDs
# data.drug_names  list of 4795 names
```

Use `--structural-only` for a 212-dim ablation (`ddi_graph_structural.pt`).

### Heterogeneous graph (`hetero_ddi_graph.pt`)

Adds protein nodes (2,708 human UniProt proteins) and drug→protein target edges.
Node types: `drug` (980-dim), `protein` (5-dim).
Edge types: `(drug, ddi, drug)`, `(drug, targets, protein)`, `(protein, rev_targets, drug)`.

---

## Feature Ablation Results

| Ablation | AUROC | AUPR | Drop |
|---|---|---|---|
| Full features (980-dim) | 0.9738 | 0.9589 | — |
| Remove structural [0:212] | 0.8993 | 0.8549 | −0.0745 |
| Remove PubMedBERT [212:980] | 0.9661 | 0.9464 | −0.0077 |

Structural features dominate; BERT embeddings add a small but consistent improvement.

---

## CN Pooling Ablation

| Ablation | AUROC | Drop |
|---|---|---|
| Full decoder | 0.9738 | — |
| Remove shared DDI neighbours | 0.9719 | −0.0019 |
| Remove shared protein targets | 0.9717 | −0.0021 |
| Remove both (plain MLP) | 0.9724 | −0.0014 |

---

## Model Evolution

| Model | Warm AUROC | Cold AUROC | Note |
|---|---|---|---|
| Homo SAGE | 0.9615 | — | Drug-only graph |
| Hetero V1 (no NCN) | 0.9726 | — | + protein nodes, MLP decoder |
| **Hetero + NCN (ours)** | **0.9738** | **0.9175** | + NCN pooling, nnPU loss |

---

*For responsible ML analysis (fairness, robustness, privacy) see [responsible_ml.md](responsible_ml.md).*
