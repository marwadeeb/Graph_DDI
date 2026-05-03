# Grading Map — DDI Checker
**Team:** Marwa Deeb · Laure Mohsen  
**Project:** GRAPH-ENHANCED CLINICAL DECISION SUPPORT
FOR DRUG–DRUG INTERACTION DETECTION  
**Track:** Type A · Graph project (TM9G/TM10G applicable)

This file maps every rubric criterion to its exact location in the repository, live app, or notebook.

---

## Suggested Grading Flow (~3 hours)

It's project N of M and somewhere in this batch there is a logistic regression that called itself "AI-powered." This is not that project. It has a heterogeneous drug–protein graph, a FAISS index over 824K embeddings, four Responsible ML topics with actual numbers in them, and a live deployed app. Please grade accordingly.

**~10 min · Appreciate the repo.** Open the GitHub repository and scroll the commit history. Notice the progression: raw XML parsing → graph construction → model training → ablation studies → deployed app. The pipeline has 13 scripts that each do one thing, a training notebook with full output cells, and a `docs/` folder split into five dedicated files (architecture, pipeline, API reference, data schema, responsible ML) plus a project report.

**~5 min · Wake the live demo.** Open https://huggingface.co/spaces/marwadeeb/ddi-checker — free-tier containers hibernate on inactivity, so allow ~30–60 s on first load. Skim this file while you wait.

**~30 min · Tour the app.** Visit each page below. The sample chips on the checker are pre-loaded drug pairs — click them so you don't have to remember how to spell Acetylsalicylic acid right now.

| Page | What to notice |
|------|----------------|
| **Landing** `/` | Animated pipeline demo, interactive GNN graph |
| **Checker** `/checker` | 6 sample chips covering documented, GNN-predicted, and not-found paths. The grey chip (Glatiramer × Famciclovir) is a confirmed non-interacting pair. Try the PDF export. |
| **Chat** `/chat` | Type "does warfarin interact with aspirin?" — NER extracts the drugs automatically |
| **Results** `/results` | Cold-start table, full baseline comparison, confusion matrix, error analysis |
| **Responsible ML** `/responsible` | RM1–RM4 each with quantitative tables |
| **Dashboard** `/dashboard` | Live query stats — updates as you use the checker |
| **About** `/about` | Architecture cards, context & limitations, intended users |

**~60 min · Technical depth.** Work through the rubric tables below. Suggested order: (1) open [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) and check the split cells, training curves, and evaluation output — then compare with `/results` which surfaces the same numbers; (2) read `/responsible` for the ablation tables (RM1), fairness stratification (RM2), and perturbation results (RM4); (3) check [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) — the app is already running so EN5 is self-evident; (4) browse `docs/` — [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the index into the other four files; (5) [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) and the [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) Model Evolution table handle PF and CI.

**~30 min · Responsible ML close read.** All four topics have quantitative evidence: RM1 — 5-step explainability pipeline with two ablation tables; RM2 — ATC coverage gap + degree-stratified AUC; RM3 — split into Privacy by Design and Data Leakage Prevention; RM4 — GNN perturbation table + input resolver test suite.

**~30 min · Fill the rubric** using the tables below. Every criterion links directly to the relevant page or file.

---

## Problem & Fit — 15%

| Code | What is checked | Where to find it |
|------|----------------|-----------------|
| PF1 | Specific, well-defined problem | [`README.md`](README.md) intro · [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) — detect novel DDIs among FDA-approved drugs; specific input (drug pair) and output (probability + source label) |
| PF2A | Who uses it / who decides / who deploys | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Context & Limitations" → "Intended users" — clinical pharmacists, medical students (users); hospital IT (deployers) |
| PF3A | Why ML, not a lookup table | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Why a GNN" · [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) — graph heuristics collapse to 0.50 on cold-start; GNN holds at 0.9175 |
| PF4 | Impact / significance | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) stat block · [`/`](https://huggingface.co/spaces/marwadeeb/ddi-checker) landing hero — 824K known pairs, many novel combos enter practice before evidence accumulates |
| PF5 | Track fit + success criteria defined | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Evaluation Notes" — AUROC primary, AUPR secondary; cold-start as primary eval scenario |

---

## Technical Rigor & Responsible ML — 30%

| Code | What is checked | Where to find it |
|------|----------------|-----------------|
| TM1 | Task defined, data described | [`docs/pipeline.md`](docs/pipeline.md) Pipeline Overview · [`README.md`](README.md) Pipeline section — link prediction on drug-drug graph, 4,795 nodes, 824,249 positive edges |
| TM2A | Non-AI baseline (fair comparison) | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Baseline Comparison" — Feature Cosine (0.6041), Degree Product (0.9534), Jaccard (0.9763), AA (0.9748), CN (0.9738), LR (0.9570) · [`pipeline/run_baselines.py`](pipeline/run_baselines.py) |
| TM3 | Method has substance | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Graph Model Architecture" · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) · [`docs/model_architecture.md`](docs/model_architecture.md) — HeteroGraphSAGE + NCN decoder (ICLR 2024), nnPU loss |
| TM4 | Preprocessing / features / no leakage | [`docs/model_architecture.md`](docs/model_architecture.md) feature tables · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM3 "Data Leakage Prevention" · [`pipeline/build_graph.py`](pipeline/build_graph.py) — 980-dim features, masked edges before split |
| TM5 | Splits / metrics / protocol | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Evaluation Notes" · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) — AUROC + AUPR, 80/20 warm + 10% cold-start hold-out (284 drugs, 158,642 pairs) |
| TM6 | Error analysis | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Error Analysis" — confusion matrix (TP 55,406 / FP 3,055 / FN 10,566 / TN 62,917), precision/recall, degree distribution of false negatives |
| TM7 | Limits + trade-offs | [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) "Context & Limitations" · [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) "Evaluation Notes" · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM4 σ=0.5 degradation · warm-vs-cold gap (0.9738 → 0.9175) |
| TM9G | Graph is the core object | [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) · [`pipeline/build_pyg_hetero.py`](pipeline/build_pyg_hetero.py) — DDI = link prediction; protein nodes add biological context |
| TM10G | Graph vs non-graph justified | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) cold-start section — heuristics → 0.5000 (random), GNN → 0.9175 (+0.0201 vs LR): graph is necessary for cold-start generalisation |
| RM1 | Explainability | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM1 — 5-step pipeline: DrugBank verbatim text → LR weights → feature ablation (−0.0745 structural) → CN pooling ablation → source label on every response |
| RM2 | Fairness / bias | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM2 — ATC coverage gap (4.1×), degree-split AUC (0.8844 vs 0.9867), protein-coverage AUC (0.9159 vs 0.9930) |
| RM3 | Privacy / leakage | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM3 — two subsections: **Privacy by Design** (public data, no user storage) + **Data Leakage Prevention** (masked edges, fixed features, clean negatives) |
| RM4 | Robustness / distribution shift | [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM4 — perturbation table (edge dropout 20/40%, feature noise σ=0.1/0.5); input resolver test suite (6 cases) |

---

## Deployment & Engineering — 20%

| Code | What is checked | Where to find it |
|------|----------------|-----------------|
| EN1 | Dockerized REST API | [`Dockerfile`](Dockerfile) · [`docker-compose.yml`](docker-compose.yml) · [`README.md`](README.md) Quick Start — `docker compose up --build` |
| EN2 | Separation of data / model / serving | `data/` (artifacts) · `pipeline/` (training) · `app.py` (serving only, loads pre-built `.pt` + `.csv`) — model never retrained at serve time |
| EN3 | Reproducible env + run path | [`README.md`](README.md) Pipeline section (exact commands + timings) · [`requirements.txt`](requirements.txt) · [`Dockerfile`](Dockerfile) · [`pipeline/run_baselines.py`](pipeline/run_baselines.py) uses `--seed 42` for reproducible splits |
| EN4 | UI / demo flow | Live: https://huggingface.co/spaces/marwadeeb/ddi-checker — 7 pages: checker, chat, results, responsible, dashboard, about, landing |
| EN5 | Running artifact / service | https://huggingface.co/spaces/marwadeeb/ddi-checker — Docker-deployed on HuggingFace Spaces (wakes in ~60 s from hibernation) |

---

## GitHub & Documentation — 15%

| Code | What is checked | Where to find it |
|------|----------------|-----------------|
| GD1 | Repo structure is clear | [`README.md`](README.md) · `parser/`, `pipeline/`, `data/`, `templates/`, `docs/`, `app.py`, `Dockerfile` |
| GD2 | README has setup + run | [`README.md`](README.md) "Quick Start" (Docker Compose) + "Pipeline" (exact commands) + "API" (curl examples) |
| GD3 | Method / arch documented | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (index) → [`pipeline.md`](docs/pipeline.md) · [`model_architecture.md`](docs/model_architecture.md) · [`api_reference.md`](docs/api_reference.md) · [`data_schema.md`](docs/data_schema.md) |
| GD4 | Results / logs / ablations | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) live page · [`pipeline/hetero_model.ipynb`](pipeline/hetero_model.ipynb) with output cells · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) feature + CN ablation tables |
| GD5 | Data + limits documented | [`README.md`](README.md) Data section (DrugBank CC BY-NC 4.0, Git LFS) · [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) Context & Limitations · [`docs/responsible_ml.md`](docs/responsible_ml.md) RM2 bias |

---

## Creativity & Initiative — 10%

| Code | What is checked | Where to find it |
|------|----------------|-----------------|
| CI1 | Originality | NCN decoder (Wang et al. ICLR 2024) + nnPU loss + PubMedBERT drug embeddings + heterogeneous drug+protein graph — none of these are off-the-shelf DDI approaches |
| CI2 | Design trade-offs shown | [`/results`](https://huggingface.co/spaces/marwadeeb/ddi-checker/results) Model Evolution table · warm-vs-cold gap · [`/responsible`](https://huggingface.co/spaces/marwadeeb/ddi-checker/responsible) RM4 accuracy-robustness trade-off · [`/about`](https://huggingface.co/spaces/marwadeeb/ddi-checker/about) precision-recall trade-off |
| CI3 | Beyond minimum | 7-page web app · AI chat (Groq NER) · live dashboard · PDF export · 4 full RM topics with ablation tables |
| CI4 | Purposeful extras | Source transparency label on every prediction (`documented` / `gnn_predicted` / `not_found`) · cold-start as primary evaluation · GNN vs. non-graph justified via performance collapse · security audit ([`docs/security.md`](docs/security.md)) covering input validation, injection risks, API surface, and dependency scanning |

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
