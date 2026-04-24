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

## Quick Start

```bash
git clone https://huggingface.co/spaces/marwadeeb/ddi-checker
cd ddi-checker
pip install -r requirements.txt

# Copy your .env (needs GROQ_API_KEY for NLP features)
echo "GROQ_API_KEY=gsk_..." > .env

python app.py          # starts on http://localhost:7860
```

The server is ready in ~3 s (dict lookup loads instantly).
The FAISS RAG index is loaded on demand only.

---

## Pages

| URL | Description |
|---|---|
| `/` | Drug pair checker — type two drug names, get interaction details |
| `/chat` | Chat interface — ask in plain English, NER extracts drug names |
| `/results` | Model comparison table + Responsible ML analysis |

---

## Key Results

Link-prediction evaluation · 80/20 train-test split · seed 42 · 1:1 pos:neg ratio

| Model | AUC-ROC | Category |
|---|---|---|
| Jaccard (graph heuristic) | **0.9845** | Non-AI baseline (TM2A) |
| Adamic-Adar (graph heuristic) | 0.9828 | Non-AI baseline (TM2A) |
| Common Neighbors | 0.9818 | Non-AI baseline (TM2A) |
| Logistic Regression (node features) | 0.9502 | Non-graph ML baseline (TM10G) |
| **GNN (graph + features)** | **TBD** | Final model |
| Random chance | 0.5000 | Trivial baseline |

---

## Pipeline (at a glance)

```
DrugBank XML (19,842 drugs · 2.9M DDI)
    ↓ parser/run_all.py          [Step 1] parse → 27 CSVs
    ↓ step2_dedup_interactions   [Step 2] directed → undirected DDI pairs
    ↓ step3_fda_approved         [Step 3] filter to 4,795 approved drugs
    ↓ step4_build_graph          [Step 4a] 212 structural node features
    ↓ step4_embed                [Step 4b] + 768-dim PubMedBERT = 980 features
    ↓ step5_pyg_data             [Step 5] PyTorch Geometric Data object
    ↓ step6_rag_index            [Step 6] FAISS index of 824K DDI descriptions
    ↓ step7_rag_query            [Step 7] dict lookup + optional FAISS query
    ↓ step8_evaluate_rag         [Step 8] precision/recall/F1 evaluation
    ↓ step9_baseline             [Step 9] graph heuristics + LR baselines
    ↓ step10_responsible_ml      [Step 10] bias + robustness analysis
    ↓ app.py                              Flask REST API + Web UI
```

Full run commands:

```bash
python parser/run_all.py                       # ~2 min
python pipeline/step2_dedup_interactions.py    # ~3.5 min
python pipeline/step3_fda_approved.py          # ~20 s
python pipeline/step4_build_graph.py           # ~15 s
python pipeline/step4_embed.py                 # ~25 min (CPU)
python pipeline/step5_pyg_data.py             # ~2 s
python pipeline/step6_rag_index.py             # ~3-4 hrs (CPU, resumable)
python pipeline/step9_baseline.py              # ~30 s (graph heuristics + LR)
python pipeline/step10_responsible_ml.py       # ~5 s  (bias + robustness)
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
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full pipeline, 27-table schema, feature groups, API reference |
| [`docs/responsible_ml.md`](docs/responsible_ml.md) | RM1 explainability · RM2 bias · RM3 privacy · RM4 robustness |

---

## Deployment

Deployed on HuggingFace Spaces via Docker. The `Dockerfile` installs dependencies, copies
`data/step3_approved/` and `data/step4_graph/` (tracked via Git LFS), and starts `app.py`.

The FAISS index (`data/rag_index/`, ~2.5 GB) is **not** included in the deployment image —
the system operates fully without it using the in-memory DrugBank dict.

---

## Data

**Source:** DrugBank Full Database v5.1 · CC BY-NC 4.0  
**Note:** DrugBank requires registration at drugbank.ca — the raw XML is not redistributed here.
The derived CSVs (`data/step3_approved/`, `data/step4_graph/`) are tracked in this repo via Git LFS.
