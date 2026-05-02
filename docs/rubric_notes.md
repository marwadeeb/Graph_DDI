# Rubric Reference — DDI Checker

Internal notes on each rubric criterion, what it checks, and where the project satisfies it.
For the professor-facing grading map see [`readme_correction.md`](../readme_correction.md).

---

## Problem & Fit — 15%

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **PF1** | Is the problem specific and well-defined? | Yes — detect novel DDIs among FDA-approved drugs not yet in DrugBank. Specific input (drug pair), specific output (interaction probability + source label). |
| **PF2A** | Who uses it, who decides, who deploys? | `/about` Context section: clinical pharmacists, medical students, researchers (users); hospital IT / deployment team (deployers). Explicitly stated. |
| **PF3A** | Why ML and not a lookup table? | Graph heuristics score 0.50 on cold-start (random chance). LR node features score 0.8974 but miss graph context. GNN reaches 0.9175 — only ML can generalise to unseen drugs. |
| **PF4** | Is this important? | 4,795 approved drugs × 824K known pairs, but many novel drug combos enter clinical practice before evidence accumulates. DDI is a leading cause of preventable hospital admissions. |
| **PF5** | Does the method fit the track, is success defined? | Type A applied project, graph track. Success = AUROC on cold-start split (harder, more realistic than warm). |

---

## Technical Rigor — 30%

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **TM1** | Task defined, data described | Link prediction on bipartite drug-drug graph. 4,795 nodes, 824,249 positive edges, 1:1 neg sampling. |
| **TM2A** | Non-AI baseline exists and is fair | Three graph heuristics (Jaccard, AA, CN), Degree Product, Feature Cosine, Logistic Regression. All evaluated on same splits. |
| **TM3** | Method has substance | HeteroGraphSAGE + NCN decoder (ICLR 2024 paper). nnPU loss for positive-unlabelled learning. Heterogeneous graph (drug + protein nodes). |
| **TM4** | Preprocessing documented, no leakage | 212 structural features (CYP450, ATC, physicochemical) + 768-dim PubMedBERT = 980-dim. Masked edges before split. Fixed features. |
| **TM5** | Splits correct, metrics justified | 80/20 random warm split + 10% cold-start hold-out (284 drugs, 158,642 pairs). AUROC + AUPR. Seed 42 throughout. |
| **TM6** | Error analysis done | Confusion matrix, precision/recall, degree distribution of false negatives. High-degree drugs harder to rank — documented. |
| **TM7** | Limits and trade-offs acknowledged | Cold-start gap (0.9175 < 0.9738 warm), precision/recall trade-off, ATC coverage bias, σ=0.5 noise degradation. |
| **TM9G** | Graph is the core object | Heterogeneous PyG graph. DDI = link prediction task. Protein nodes add biological context to drug representations. |
| **TM10G** | Graph vs non-graph justified | Graph heuristics: 0.9763 warm / 0.5000 cold. GNN: 0.9738 warm / 0.9175 cold. Non-graph (LR): 0.9570 warm / 0.8974 cold. Graph is necessary for cold-start generalisation. |

---

## Responsible ML — within TM 30%

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **RM1** | Explainability addressed | 5-step pipeline on `/responsible`: DrugBank text (verbatim) → LR weights → GNN feature ablation (structural −0.0745, BERT −0.0077) → CN pooling ablation → source label on every response. **Done.** |
| **RM2** | Bias / fairness measured | ATC coverage gap (NERVOUS 864 vs VARIOUS 213 mean degree, 4.1× ratio). Degree-split AUC (0.8844 high / 0.9867 low). Protein-coverage AUC (0.9159 / 0.9930). **Done.** |
| **RM3** | Privacy and leakage | Privacy by Design (public data only, no user data stored). Data Leakage Prevention (masked edges, fixed features, clean negatives). **Done.** |
| **RM4** | Robustness / distribution shift | Edge dropout 20% (−0.0024), 40% (−0.0021), Feature noise σ=0.1 (−0.0035), σ=0.5 (−0.0455 — documented limit). Input resolver tested on 6 cases. **Done.** |

---

## Deployment & Engineering — 20%

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **EN1** | Dockerized REST API | `Dockerfile` + `docker-compose.yml`. `docker compose up --build` brings up the full app. |
| **EN2** | Data / model / serving separated | `data/` (pipeline artifacts, tracked LFS), `pipeline/` (training), `app.py` (serving only — loads `.pt` and `.csv` at startup, no retraining). |
| **EN3** | Reproducible env + run path | `requirements.txt`, `Dockerfile`, pipeline commands in `README.md`, seed 42 in notebook. |
| **EN4** | UI / demo | 7-page web app (landing, checker, chat, results, responsible, dashboard, about). Live on HuggingFace Spaces. |
| **EN5** | Running artifact | https://huggingface.co/spaces/marwadeeb/ddi-checker — always-on (wakes from hibernation in ~60 s). |

---

## GitHub & Documentation — 15%

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **GD1** | Repo structure clear | `parser/`, `pipeline/`, `data/`, `templates/`, `docs/`, `app.py`, `Dockerfile`, `README.md` |
| **GD2** | README has setup + run | Quick Start (Docker Compose), Pipeline commands, API examples. |
| **GD3** | Method / arch docs | `docs/ARCHITECTURE.md` — full pipeline table, tech stack, feature groups, API reference. |
| **GD4** | Results / logs / ablations | `/results` live page, `hetero_model.ipynb` notebook cells + outputs, `/responsible` ablation tables. |
| **GD5** | Data + limits documented | `README.md` Data section (DrugBank CC BY-NC 4.0, Git LFS note), `/about` Context & Limitations, RM2 bias tables. |

---

## Creativity & Initiative — 10%

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **CI1** | Originality | NCN decoder (ICLR 2024), nnPU loss, PubMedBERT drug embeddings, heterogeneous graph. |
| **CI2** | Design trade-offs shown | Model evolution table (warm vs cold trade-off); precision vs recall; robustness vs accuracy; ATC bias vs model performance. |
| **CI3** | Beyond minimum | 7-page app, AI chat (Groq NER), live dashboard, PDF export, "not found" result path. |
| **CI4** | Purposeful extras | Source transparency (`documented`/`gnn_predicted`/`not_found`) on every API response; cold-start as primary evaluation. |

---

## Bonus

| Code | What the prof is checking | Our evidence |
|------|--------------------------|-------------|
| **BX2** | RM beyond minimum | All 4 RM topics with quantitative tables (not just "we thought about it"). |
| **BX3** | Exceptional extension | Full deployed 7-page app; ICLR 2024 decoder; nnPU; heterogeneous biomedical graph. |
