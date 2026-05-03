# Pipeline — DDI Checker

Step-by-step data pipeline from raw DrugBank XML to deployed model.
For tech stack and infrastructure see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Pipeline Overview

| Step | Script | Output | Size |
|---|---|---|---|
| 1 — Parse & Validate | `parser/run_all.py` | `data/step1_full/` — 27 CSVs, all drugs | 707 MB · gitignored |
| 2 — Dedup Interactions | `pipeline/dedup_interactions.py` | `data/step2_dedup/` — undirected DDI | 171 MB · gitignored |
| 3 — FDA-Approved Subset | `pipeline/filter_approved.py` | `data/step3_approved/` — 4,795 drugs | 351 MB · **on GitHub** |
| 4a — Build Graph | `pipeline/build_graph.py` | `data/step4_graph/` — 212 structural features + edge index | 19 MB · **on GitHub** |
| 4b — Text Embeddings | `pipeline/embed_drugs.py` | `data/step4_graph/` — 768-dim PubMedBERT + combined 980-dim | 94 MB · **on GitHub** |
| 5a — PyG Homo Graph | `pipeline/build_pyg_homo.py` | `data/step4_graph/ddi_graph.pt` — homogeneous | 58 MB · **on GitHub** |
| 5b — PyG Hetero Graph | `pipeline/build_pyg_hetero.py` | `data/step4_graph/hetero_ddi_graph.pt` — drug + protein nodes | 46 MB · **on GitHub** |
| 6 — RAG Vector Index | `pipeline/build_rag_index.py` | `data/rag_index/` — FAISS index of 824K DDI descriptions | ~2.5 GB · gitignored |
| 7 — RAG / Dict Query | `pipeline/rag_query.py` | CLI/API — O(1) dict lookup + optional FAISS | — |
| 8 — RAG Evaluation | `pipeline/evaluate_rag.py` | `data/evaluation/` — precision/recall/F1 | — |
| 9 — Baselines + Split | `pipeline/run_baselines.py` | `data/evaluation/` — graph heuristics + LR + cold-start split files | — |
| 10 — Responsible ML | `pipeline/responsible_ml.py` | `data/evaluation/` — bias JSON · robustness JSON | — |
| GNN Training | `hetero_model.ipynb` | `data/step4_graph/bestHeteroModel.pt` | 6 MB · **on GitHub** |
| Serving | `app.py` | Flask REST API + Web UI on port 7860 | — |

## Run Commands

```bash
python parser/run_all.py                         # ~2 min
python pipeline/dedup_interactions.py            # ~3.5 min
python pipeline/filter_approved.py              # ~20 s
python pipeline/build_graph.py                  # ~15 s
python pipeline/embed_drugs.py                  # ~25 min (CPU)
python pipeline/build_pyg_homo.py               # ~2 s
python pipeline/build_pyg_hetero.py             # ~30 s
python pipeline/run_baselines.py                # ~30 s  (graph heuristics + LR + cold split)
python pipeline/responsible_ml.py               # ~5 s   (bias + robustness)
# GNN training: run hetero_model.ipynb in Jupyter
```

---

## Step 1 — DrugBank XML Parsing

DrugBank distributes its data as a single monolithic XML file (v5.1: **1.1 GB**, **19,842 drug entries**).
The schema is deeply nested, highly polymorphic, and semi-structured.

We use **lxml `iterparse`** to process one `<drug>` element at a time, keeping peak memory bounded.

```
main_parser.py          streaming loop — fires per-drug event, calls all parsers
├── parse_core.py       → drugs, drug_ids, drug_attributes, properties, external_identifiers
├── parse_references.py → references, reference_associations
├── parse_commercial.py → salts, products, mixtures, prices, commercial_entities
├── parse_pharmacological.py → categories, atc_codes, dosages, patents
├── parse_interactions.py    → drug_interactions (DDI), drug_snp_data
├── parse_pathways.py   → pathways, pathway_members, reactions
└── parse_proteins.py   → interactants, polypeptides, polypeptide_attributes
```

`parser/validate.py` runs 16 post-parse checks: PK uniqueness, FK integrity, row-count thresholds, non-null constraints, duplicate detection.

---

## Step 2 — Deduplication

Raw XML stores directed pairs `(A→B)` and `(B→A)` separately. `dedup_interactions.py` collapses these to canonical `frozenset{A,B}` pairs, reducing 2.9M directed records to **824,249 unique undirected pairs**.

---

## Steps 6 & 7 — RAG / Dict Pipeline

### Primary path — O(1) dict lookup
`rag_query.py` builds an in-memory `frozenset → description` dict from `drug_interactions_dedup.csv` at startup (~3 s load).

### Optional path — FAISS semantic search
824,249 DDI descriptions embedded with PubMedBERT → FAISS IndexFlatIP. Loaded on demand (~30 s first call).

| File | Description |
|---|---|
| `data/rag_index/faiss.index` | FAISS IndexFlatIP — 824,249 × 768 vectors |
| `data/rag_index/metadata.pkl` | Per-vector metadata: interaction_id, names, text |

### LLM (Groq)
`llama-3.3-70b-versatile` via Groq free tier. Used for NER extraction from chat input and natural-language explanation of results. Requires `GROQ_API_KEY` in `.env`.

---

## Step 9 — Baselines

### Graph heuristics
- **Common Neighbors:** `CN = A @ A`
- **Adamic-Adar:** `AA = A @ D @ A` where `D = diag(1/log(degree))`
- **Jaccard:** `JC = CN / (deg_u + deg_v - CN)`

### Logistic Regression
212-dim node vectors → 424-dim symmetric pair features: `[|feat_A − feat_B|, feat_A * feat_B]`  
Pipeline: `StandardScaler` + `LogisticRegression(max_iter=1000, C=1.0)`

`edge_split.npz` saved at `data/evaluation/edge_split.npz` for fair comparison with GNN training split.

---

## File Structure

```
├── parser/                          Step 1 — XML → CSV
│   ├── config.py / state.py / utils.py
│   ├── parse_core.py / parse_references.py / parse_commercial.py
│   ├── parse_pharmacological.py / parse_interactions.py
│   ├── parse_pathways.py / parse_proteins.py
│   ├── main_parser.py               streaming iterparse loop
│   ├── validate.py                  16-check validation suite
│   └── run_all.py                   entry point
├── pipeline/                        Steps 2-10
│   ├── dedup_interactions.py
│   ├── filter_approved.py
│   ├── build_graph.py
│   ├── embed_drugs.py
│   ├── build_pyg_homo.py
│   ├── build_pyg_hetero.py
│   ├── build_rag_index.py
│   ├── rag_query.py                 drug name resolver + FAISS + LLM
│   ├── evaluate_rag.py
│   ├── run_baselines.py
│   ├── responsible_ml.py
│   ├── error_analysis.py
│   └── gnn_predictor.py             GNN inference interface (Flask ↔ model)
├── hetero_model.ipynb               GNN training
├── app.py                           Flask REST API
└── data/
    ├── step1_full/                  [gitignored]
    ├── step2_dedup/                 [gitignored]
    ├── step3_approved/              [tracked via LFS]
    ├── step4_graph/                 [tracked via LFS]
    ├── rag_index/                   [gitignored — ~2.5 GB]
    └── evaluation/                  [tracked]
```
