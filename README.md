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

**Live demo:** https://marwadeeb-ddi-checker.hf.space

> **⏳ First visit:** HuggingFace Spaces hibernates after a period of inactivity. If the demo
> shows a loading screen, wait ~30–60 seconds for the container to wake before using it.

> **📚 Project scope:** This repository documents the full ML pipeline, baselines, ablation
> results, and Responsible ML analysis for academic evaluation purposes. A production deployment
> would strip out the evaluation pages and expose only the checker and chat interfaces.

> **⏳ First visit:** HuggingFace Spaces hibernates after a period of inactivity. If the demo
> shows a loading screen, wait ~30–60 seconds for the container to wake before using it.

> **📚 Project scope:** This repository documents the full ML pipeline, baselines, ablation
> results, and Responsible ML analysis for academic evaluation purposes. A production deployment
> would strip out the evaluation pages and expose only the checker and chat interfaces.

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
| `/results` | Model performance — cold-start evaluation + full baseline comparison |
| `/responsible` | Responsible ML — explainability, fairness, privacy, robustness |
| `/dashboard` | Live dashboard — query stats, system health, recent activity |
| `/about` | About the system — tech stack, how it works, key numbers |

---

## Key Results

**Cold-start evaluation** (10 % of drugs held out entirely — the GNN's real use case)

| Model | Cold AUC-ROC | Cold Avg Precision | Note |
|---|---|---|---|
| **GNN — HeteroGraphSAGE + NCN** | **0.9175** | **0.8824** | Features + graph context |
| Logistic Regression | 0.8974 | 0.9032 | Node features only |
| Graph heuristics (all) | 0.5000 | 0.5000 | Analytically guaranteed — zero training edges = zero score |

> Graph heuristics score ~0.97 on warm eval because DrugBank is extremely dense (avg degree ~344). On cold-start they collapse to 0.50 = random chance. The GNN holds at 0.9175, outperforming LR by +0.0201 AUROC — see `/results` for full analysis.

---

## Pipeline (at a glance)

```
DrugBank XML (19,842 drugs · 2.9M DDI)
    ↓ parser/run_all.py            [Step 1]  parse → 27 CSVs
    ↓ dedup_interactions.py        [Step 2]  directed → undirected DDI pairs
    ↓ filter_approved.py           [Step 3]  filter to 4,795 approved drugs
    ↓ build_graph.py               [Step 4a] 212 structural node features
    ↓ embed_drugs.py               [Step 4b] + 768-dim PubMedBERT = 980 features
    ↓ build_pyg_homo.py            [Step 5a] PyG homogeneous ddi_graph.pt
    ↓ build_pyg_hetero.py          [Step 5b] + drug-protein edges → hetero_ddi_graph.pt
    ↓ run_baselines.py             [Step 9]  graph heuristics + LR + cold-start split
    ↓ responsible_ml.py            [Step 10] bias + robustness analysis
    ↓ pipeline/hetero_model.ipynb            GNN training (HeteroGraphSAGE + NCN)
    ↓ app.py                                 Flask REST API + Web UI
```

Full run commands:

```bash
python parser/run_all.py                        # ~2 min
python pipeline/dedup_interactions.py           # ~3.5 min
python pipeline/filter_approved.py             # ~20 s
python pipeline/build_graph.py                  # ~15 s
python pipeline/embed_drugs.py                  # ~25 min (CPU)
python pipeline/build_pyg_homo.py               # ~2 s
python pipeline/build_pyg_hetero.py             # ~30 s  (drug-protein hetero graph)
python pipeline/run_baselines.py                # ~30 s  (graph heuristics + LR + cold split)
python pipeline/responsible_ml.py               # ~5 s   (bias + robustness)
# GNN training: run pipeline/hetero_model.ipynb in Jupyter
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
| [`readme_correction.md`](readme_correction.md) | **Grading map** — rubric code → exact file/page location |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Tech stack index → pipeline · model · API · data schema |
| [`docs/pipeline.md`](docs/pipeline.md) | Step-by-step pipeline, run commands, file structure |
| [`docs/model_architecture.md`](docs/model_architecture.md) | GNN architecture, 980-dim features, ablation results |
| [`docs/api_reference.md`](docs/api_reference.md) | All REST endpoints, request/response format |
| [`docs/data_schema.md`](docs/data_schema.md) | 27 normalised DrugBank tables, key statistics |
| [`docs/responsible_ml.md`](docs/responsible_ml.md) | RM1 explainability · RM2 bias · RM3 privacy · RM4 robustness |
| [`docs/security.md`](docs/security.md) | Attack surface audit, input validation posture |

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
