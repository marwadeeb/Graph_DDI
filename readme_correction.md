# Grading Map — DDI Checker
**Team:** Marwa Deeb · Laure Mohsen  
**Project:** Application-Oriented Graph-Based ML on Cloud  
**Track:** Type A · Graph project (TM9G/TM10G applicable)

This file maps every rubric criterion to its exact location in the repository, live app, or notebook.
Use it as a checklist when grading. If something is not linked here, it does not exist.

---

## Problem & Fit (PF)

| Code | Criterion | Where to find it |
|------|-----------|-----------------|
| PF1 | Specific problem / question | [`README.md`](README.md) intro · [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Why this problem" section |
| PF2A | User / decision-maker / deployer | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Context & Limitations" → "Intended users" · [`docs/responsible_ml.md`](docs/responsible_ml.md) |
| PF3A | Why ML, why not simpler | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Why a GNN" section · [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) cold-start collapse of graph heuristics to 0.50 |
| PF4 | Impact / significance | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) intro stat block: 824 K DDI pairs · [`/`](https://huggingface.co/spaces/marwadeeb/ddi-checker) landing page hero text |
| PF5 | Track fit + success criteria | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Evaluation Notes" section — AUROC primary metric, AUPR secondary; cold-start as primary evaluation scenario |

---

## Technical Rigor & Responsible ML (TM / RM)

| Code | Criterion | Where to find it |
|------|-----------|-----------------|
| TM1 | Task + data formulation | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) Pipeline Overview table · [`README.md`](README.md) Pipeline section |
| TM2A | Non-AI baseline | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Baseline Comparison" — Feature Cosine (0.6041), Degree Product (0.9534), graph heuristics (Jaccard 0.9763, AA 0.9748, CN 0.9738) · [`pipeline/step9_baseline.py`](pipeline/step9_baseline.py) |
| TM3 | Method choice + substance | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Graph Model Architecture" card · [`hetero_model.ipynb`](hetero_model.ipynb) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| TM4 | Preprocessing / features / leakage | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) Feature Engineering section · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM3 "Data Leakage Prevention" subsection · [`pipeline/step4_build_graph.py`](pipeline/step4_build_graph.py) |
| TM5 | Splits / metrics / protocol | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Evaluation Notes" · [`hetero_model.ipynb`](hetero_model.ipynb) split cells · AUROC + AUPR, seed 42, 80/20 warm + 10% cold-start hold-out |
| TM6 | Error analysis | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Error Analysis" section — confusion matrix (TP 55,406 / FP 3,055 / FN 10,566 / TN 62,917), precision/recall, degree distribution of errors |
| TM7 | Limits + trade-offs | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Context & Limitations" · [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Evaluation Notes" · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM2 coverage gap · RM4 σ=0.5 degradation |
| TM9G | Graph as core object | [`hetero_model.ipynb`](hetero_model.ipynb) · [`pipeline/step5_hetero_graph.py`](pipeline/step5_hetero_graph.py) — heterogeneous drug+protein graph; DDI as link prediction on graph edges |
| TM10G | Graph vs non-graph justified | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) cold-start section: graph heuristics → 0.5000 (random), GNN → 0.9175 (+0.0201 vs LR) — graph is necessary for cold-start |
| RM1 | Explainability | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM1 section — 5-step pipeline: dict lookup (verbatim DrugBank text), LR weight attribution, GNN feature ablation (−0.0745 structural), CN pooling ablation, source transparency label |
| RM2 | Fairness / bias | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM2 section — ATC coverage gap (4.1×), degree-split stratification (0.8844 vs 0.9867 AUC), protein-coverage stratification · [`docs/responsible_ml.md`](docs/responsible_ml.md) |
| RM3 | Privacy / leakage | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM3 section — two subsections: Privacy by Design + Data Leakage Prevention (masked edges, fixed features, clean negatives) |
| RM4 | Robustness / distribution shift | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM4 section — GNN perturbation table (edge dropout 20/40%, feature noise σ=0.1/0.5); input resolver robustness test suite |

---

## Deployment & Engineering (EN)

| Code | Criterion | Where to find it |
|------|-----------|-----------------|
| EN1 | Dockerized API | [`Dockerfile`](Dockerfile) · [`docker-compose.yml`](docker-compose.yml) · `README.md` Quick Start section |
| EN2 | Separation of data / model / serving | `data/` (pipeline outputs) · `pipeline/` (training scripts) · `app.py` (serving only, loads pre-built artifacts) — model never retrained at serve time |
| EN3 | Reproducible env + run path | [`README.md`](README.md) Pipeline section with exact commands · [`requirements.txt`](requirements.txt) · [`Dockerfile`](Dockerfile) · seed 42 in [`hetero_model.ipynb`](hetero_model.ipynb) |
| EN4 | UI / demo flow | Live: https://huggingface.co/spaces/marwadeeb/ddi-checker — 7 pages: checker, chat, results, responsible, dashboard, about, landing |
| EN5 | Running artifact / service | https://huggingface.co/spaces/marwadeeb/ddi-checker (live, Docker-deployed on HuggingFace Spaces) |

---

## GitHub & Documentation (GD)

| Code | Criterion | Where to find it |
|------|-----------|-----------------|
| GD1 | Repo structure | [`README.md`](README.md) · `parser/`, `pipeline/`, `data/`, `templates/`, `docs/`, `app.py` |
| GD2 | README: setup + run | [`README.md`](README.md) "Quick Start" + "Pipeline" + "API" sections |
| GD3 | Method / arch docs | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full pipeline, tech stack, feature engineering, API reference |
| GD4 | Results / logs / ablations | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) live page · [`hetero_model.ipynb`](hetero_model.ipynb) training cells + output · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) ablation tables |
| GD5 | Data + limits + notes | [`README.md`](README.md) Data section · [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) Context & Limitations · [`docs/responsible_ml.md`](docs/responsible_ml.md) RM2 bias |

---

## Creativity & Initiative (CI)

| Code | Criterion | Where to find it |
|------|-----------|-----------------|
| CI1 | Originality | Heterogeneous drug+protein graph with NCN decoder (Wang et al. ICLR 2024) applied to DDI; nnPU loss for positive-unlabelled learning; PubMedBERT drug embeddings |
| CI2 | Design trade-offs | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) Model Evolution table (Homo → Hetero V1 → Hetero+NCN) · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM4 robustness vs. accuracy trade-off · [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) limitations (precision vs. recall trade-off) |
| CI3 | Beyond minimum | 7-page web app · AI chat assistant with NER · live dashboard · PDF export from checker · responsible ML with 4 full topics + ablation tables |
| CI4 | Purposeful extras | Source transparency label on every prediction (`documented` / `gnn_predicted` / `not_found`) · cold-start evaluation (the harder, more realistic scenario) · GNN vs. non-graph justified via performance collapse |

---

## Bonus (BX)

| Code | Criterion | Status |
|------|-----------|--------|
| BX2 | RM beyond minimum | All 4 RM topics addressed with quantitative evidence (ablation tables, fairness tables, perturbation tests) |
| BX3 | Exceptional extension | NCN (Neural Common Neighbours) decoder from ICLR 2024 applied to heterogeneous biomedical graph; nnPU loss for PU learning; full 7-page deployed app |

---

## Quick Reference — Live Pages

| Page | URL |
|------|-----|
| Landing | https://huggingface.co/spaces/marwadeeb/ddi-checker |
| Checker | https://huggingface.co/spaces/marwadeeb/ddi-checker/checker |
| Chat | https://huggingface.co/spaces/marwadeeb/ddi-checker/chat |
| Results | https://huggingface.co/spaces/marwadeeb/ddi-checker/results |
| Responsible ML | https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible |
| Dashboard | https://huggingface.co/spaces/marwadeeb/ddi-checker/dashboard |
| About | https://huggingface.co/spaces/marwadeeb/ddi-checker/about |

> **Note:** First visit may require ~30–60 s to wake from HuggingFace Spaces hibernation.
