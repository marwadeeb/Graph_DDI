---
title: DDI Checker
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DDI Checker — Drug Interaction Detection

Detect drug-drug interactions (DDI) using a two-stage pipeline:
**DrugBank dictionary lookup** (O(1), 824 K documented pairs) +
**GNN link prediction** (novel pairs not yet in DrugBank).

**Live demo:** https://huggingface.co/spaces/marwadeeb/ddi-checker

---

## Quick Start (local — Docker Compose)

Create a `.env` file first (see `.env.example`):

```bash
cp .env.example .env
# edit .env and fill in GROQ_API_KEY
```

Then start the app:

```bash
docker compose up --build -d
# App is available at http://localhost:7860
```

To stop:

```bash
docker compose down
```
 
> The server starts in ~3 s. Only `GROQ_API_KEY` is required for the AI chat feature.

---

## Pages

| URL | Description |
|---|---|
| `/` | Landing page — animated pipeline demo + drug graph visualization |
| `/checker` | Drug pair checker — type two drug names, get interaction details |
| `/chat` | Chat interface — ask in plain English, NER extracts drug names |
| `/results` | Model performance — cold-start (primary) + warm evaluation |
| `/responsible` | Responsible ML — explainability, fairness, privacy, robustness |
| `/dashboard` | Live dashboard — query stats, system health, recent activity |
| `/about` | About the system — tech stack, how it works, key numbers |

---

## Key Results


**Cold-start evaluation** (10 % of drugs held out entirely — the GNN's real use case)

| Model | Cold AUC-ROC | Note |
|---|---|---|
| **GNN — HeteroGraphSAGE + NCN** | *run pending* | Uses node features + graph structure |
| Logistic Regression | *run pending* | Uses node features only |
| Graph heuristics | **0.5000** | Analytically guaranteed — zero training edges = zero score |

> Graph heuristics score 0.98 on warm evaluation because DrugBank is extremely dense (avg degree 344). On cold-start they collapse to 0.50 = random chance. The GNN's purpose is novel pair prediction — see `/results` for full analysis.

---

## Pipeline (at a glance)

```
DrugBank XML (19,842 drugs · 2.9M DDI)
    ↓ parser/run_all.py           [Step 1]  parse → 27 CSVs
    ↓ step2_dedup_interactions    [Step 2]  directed → undirected DDI pairs
    ↓ step3_fda_approved          [Step 3]  filter to 4,795 approved drugs
    ↓ step4_build_graph           [Step 4a] 212 structural node features
    ↓ step4_embed                 [Step 4b] + 768-dim PubMedBERT = 980 features
    ↓ step5_pyg_data              [Step 5a] PyG homogeneous ddi_graph.pt
    ↓ step5_hetero_graph          [Step 5b] + drug-protein edges → hetero_ddi_graph.pt
    ↓ step9_baseline              [Step 9]  graph heuristics + LR + cold-start split
    ↓ step10_responsible_ml       [Step 10] bias + robustness analysis
    ↓ hetero_model.ipynb                    GNN training (HeteroGraphSAGE + NCN)
    ↓ app.py                               Flask REST API + Web UI
```

Full run commands:

```bash
python parser/run_all.py                        # ~2 min
python pipeline/step2_dedup_interactions.py     # ~3.5 min
python pipeline/step3_fda_approved.py           # ~20 s
python pipeline/step4_build_graph.py            # ~15 s
python pipeline/step4_embed.py                  # ~25 min (CPU)
python pipeline/step5_pyg_data.py               # ~2 s
python pipeline/step5_hetero_graph.py           # ~30 s  (drug-protein hetero graph)
python pipeline/step9_baseline.py               # ~30 s  (graph heuristics + LR + cold split)
python pipeline/step10_responsible_ml.py        # ~5 s   (bias + robustness)
python pipeline/step10_responsible_ml.py --section gnn_auc  # per-category GNN AUC (needs model .pt files)
# GNN training: run hetero_model.ipynb in Jupyter
```

---

## API

```bash
# Single pair check
curl -X POST http://localhost:7860/api/check \
  -H "Content-Type: application/json" \
  -d '{"drug_a": "Warfarin", "drug_b": "Aspirin"}'

# Drug autocomplete
curl "http://localhost:7860/api/drug/search?q=war&limit=5"

# Health check
curl http://localhost:7860/health
```

Accepts drug names **or** DrugBank IDs (`DB00682`).
Case-insensitive · brand names partially supported via synonym table.

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full pipeline, tech stack, 27-table schema, feature groups, API reference |
| [`docs/responsible_ml.md`](docs/responsible_ml.md) | RM1 explainability · RM2 bias · RM3 privacy · RM4 robustness |
| [`docs/security.md`](docs/security.md) | Attack surface audit, input validation posture, dependency scanning |

---

## Deployment

Deployed on HuggingFace Spaces via Docker. The `Dockerfile` installs dependencies, copies
`data/step3_approved/` and `data/step4_graph/` (tracked via Git LFS), and starts `app.py`
on port 7860 with Gunicorn (1 worker, 4 threads).

---

## Data

**Source:** DrugBank Full Database v5.1 · CC BY-NC 4.0  
**Note:** DrugBank requires registration at drugbank.ca — the raw XML is not redistributed here.
The derived CSVs (`data/step3_approved/`, `data/step4_graph/`) are tracked in this repo via Git LFS.
