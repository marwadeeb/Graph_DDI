# Grading Map — DDI Checker
**Team:** Marwa Deeb · Laure Mohsen
**Project:** Graph-Enhanced Clinical Decision Support for Drug–Drug Interaction Detection
**Track:** Type A · Graph: Yes
**Repository:** https://github.com/marwadeeb/Graph_DDI
**Live Demo:** https://marwadeeb-ddi-checker.hf.space

This file serves as a quick-reference index mapping each rubric criterion to its exact location in the repository or live application. A more detailed, properly written account of all rubric items, including methodology, equations, ablation tables, results, and limitations, is provided in the submitted project report titled "EECE 690 - Rubric Report.pdf". 

> **Note:** The live application runs on HuggingFace Spaces free tier and may hibernate after inactivity. Allow 30–60 seconds on first load.

---

## Live Pages

| Page | URL |
|------|-----|
| Landing | https://marwadeeb-ddi-checker.hf.space |
| Checker | https://marwadeeb-ddi-checker.hf.space/checker |
| Chat | https://marwadeeb-ddi-checker.hf.space/chat |
| Results | https://marwadeeb-ddi-checker.hf.space/results |
| Responsible ML | https://marwadeeb-ddi-checker.hf.space/responsible |
| Dashboard | https://marwadeeb-ddi-checker.hf.space/dashboard |
| About | https://marwadeeb-ddi-checker.hf.space/about |

---

## Problem & Fit

| Code | Criterion | Location |
|------|-----------|----------|
| **PF1** | Specific problem | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → Problem Definition · [`README.md`](README.md) intro · Report §1.1 |
| **PF2A** | User / decision / deployer | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → "Context & Limitations" → "Intended Users" · Report §1.2 |
| **PF3A** | Why ML, not a simpler approach | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → "Why a GNN" · [`/results`](https://marwadeeb-ddi-checker.hf.space/results) cold-start collapse · Report §1.3 |
| **PF4** | Impact and significance | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) stat block · [`/`](https://marwadeeb-ddi-checker.hf.space) landing hero · Report §1.4 |
| **PF5** | Track fit + success criteria | [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → "Evaluation Notes" · Report §1.5 |

---

## Technical Rigor

| Code | Criterion | Location |
|------|-----------|----------|
| **TM1** | Task formulation + data | [`docs/pipeline.md`](docs/pipeline.md) · [`README.md`](README.md) Pipeline section · Report §2.1 |
| **TM2A** | Non-AI baseline | [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → "Baseline Comparison" · [`pipeline/step9_baseline.py`](pipeline/step9_baseline.py) · Report §2.2 — 5 baselines: Feature Cosine (0.6041), Degree Product (0.9534), Common Neighbors (0.9738), Adamic–Adar (0.9748), Jaccard (0.9763) |
| **TM3** | Method choice + substance | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → "Graph Model Architecture" · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) · [`docs/model_architecture.md`](docs/model_architecture.md) · Report §2.3 |
| **TM4** | Preprocessing / features / leakage | [`docs/model_architecture.md`](docs/model_architecture.md) feature tables · [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) → RM3 · [`pipeline/step4_build_graph.py`](pipeline/step4_build_graph.py) · Report §2.4 |
| **TM5** | Splits / metrics / protocol | [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → "Evaluation Notes" · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) split cells · Report §2.5 |
| **TM6** | Error analysis | [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → "Error Analysis" · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) error analysis cell · Report §2.6 |
| **TM7** | Limitations + trade-offs | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → "Context & Limitations" · [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → "Evaluation Notes" · Report §2.7 |
| **TM9G** | Graph as core object | [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) model definition cells · [`pipeline/step5_hetero_graph.py`](pipeline/step5_hetero_graph.py) · Report §2.8 |
| **TM10G** | Graph vs. non-graph justified | [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → Cold-Start table · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) cold-start cell · Report §2.9 |

---

## Responsible ML (within Technical Rigor)

| Code | Criterion | Location |
|------|-----------|----------|
| **RM1** | Explainability | [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) → RM1 · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) explainability cells · [`pipeline/gnn_predictor.py`](pipeline/gnn_predictor.py) `explain()` · Report §2.10 |
| **RM2** | Fairness / bias | [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) → RM2 · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) fairness cell · Report §2.11 |
| **RM3** | Privacy / leakage | [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) → RM3 · Report §2.12 |
| **RM4** | Robustness / shift | [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) → RM4 · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) robustness cell · Report §2.13 |

---

## Deployment & Engineering

| Code | Criterion | Location |
|------|-----------|----------|
| **EN1** | Dockerized REST API | [`Dockerfile`](Dockerfile) · [`docker-compose.yml`](docker-compose.yml) · [`README.md`](README.md) → Quick Start |
| **EN2** | Separation of data / model / serving | `data/` (artifacts) · `pipeline/` (training) · `app.py` (serving) · [`pipeline/gnn_predictor.py`](pipeline/gnn_predictor.py) (inference module) · Report §3.2 |
| **EN3** | Reproducible environment + run path | [`README.md`](README.md) → Pipeline section · [`requirements.txt`](requirements.txt) · Report §3.3 |
| **EN4** | Functional UI / demo flow | https://marwadeeb-ddi-checker.hf.space — 6 pages, autocomplete, three-state results, PDF export, live dashboard · Report §3.4 |
| **EN5** | Running deployed artifact | https://marwadeeb-ddi-checker.hf.space — Docker on HuggingFace Spaces · `/health` endpoint · Report §3.5 |

---

## GitHub & Documentation

| Code | Criterion | Location |
|------|-----------|----------|
| **GD1** | Repository structure | [`README.md`](README.md) structure section · Report §4.1 |
| **GD2** | README: setup + run | [`README.md`](README.md) → Quick Start + Pipeline + API |
| **GD3** | Method / architecture docs | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (index) · [`docs/model_architecture.md`](docs/model_architecture.md) · [`docs/pipeline.md`](docs/pipeline.md) · [`docs/api_reference.md`](docs/api_reference.md) · [`docs/data_schema.md`](docs/data_schema.md) · [`docs/responsible_ml.md`](docs/responsible_ml.md) |
| **GD4** | Results / logs / ablations | [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) (embedded outputs) · [`/results`](https://marwadeeb-ddi-checker.hf.space/results) · [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) · `data/evaluation/` |
| **GD5** | Data + limits + notes | [`README.md`](README.md) → Data section · [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → Limitations · [`docs/responsible_ml.md`](docs/responsible_ml.md) |

---

## Creativity & Initiative

| Code | Criterion | Location |
|------|-----------|----------|
| **CI1** | Originality | [`/about`](https://marwadeeb-ddi-checker.hf.space/about) → Architecture · Report §5 |
| **CI2** | Design trade-offs | [`/results`](https://marwadeeb-ddi-checker.hf.space/results) → Model Evolution table · [`/responsible`](https://marwadeeb-ddi-checker.hf.space/responsible) → RM4 · Report §5 |
| **CI3** | Beyond minimum | 6-page deployed app · AI chat with Groq LLaMA NER · live dashboard · PDF export · cold-start evaluation |
| **CI4** | Purposeful extras | Per-prediction explanation box on checker · three-state source labeling · [`docs/security.md`](docs/security.md) |

---

## Bonus

| Code | Criterion | Evidence |
|------|-----------|----------|
| **BX2** | RM beyond minimum | All 4 RM topics with quantitative tables — see `/responsible` and Report §2.10–2.13 |
| **BX3** | Exceptional extension | NCN decoder on heterogeneous biomedical graph · nnPU PU learning · fully deployed 6-page clinical app with AI chatbot |

---

## Demo Pairs for `/checker`

| Expected output | Drug A | Drug B |
|-----------------|--------|--------|
| `documented` | Warfarin | Aspirin |
| `documented` | Sildenafil | Nitroglycerin |
| `documented` | Simvastatin | Clarithromycin |
| `gnn_predicted` | Metformin | Vitamin D |
| `gnn_predicted` | Atomoxetine | Methylphenidate |
| `not_found` | Glatiramer | Famciclovir |
