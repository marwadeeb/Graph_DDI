# Architecture — DDI Checker

Technical reference index. Each topic is in its own document:

| Document | Contents |
|---|---|
| **[pipeline.md](pipeline.md)** | Step-by-step pipeline (parse → graph → train → serve), run commands, file structure |
| **[model_architecture.md](model_architecture.md)** | GNN architecture (HeteroGraphSAGE + NCN), node features (980-dim), PyG data objects, ablation results |
| **[api_reference.md](api_reference.md)** | All REST endpoints, request/response formats, `source` field values |
| **[data_schema.md](data_schema.md)** | 27 normalised tables from DrugBank XML, key statistics, FK conventions |
| **[responsible_ml.md](responsible_ml.md)** | RM1 explainability · RM2 fairness · RM3 privacy · RM4 robustness |

For operational setup (Docker, env vars, quick start) see the main [README](../README.md).

---

## Technology Stack

### Backend
| Library | Version | Role |
|---|---|---|
| Python | 3.10 | Runtime |
| Flask | 2.x | REST API + Jinja2 server-side rendering |
| PyTorch | 2.2 | GNN inference + tensor ops |
| PyTorch Geometric (PyG) | 2.x | Heterogeneous graph data structures + SAGEConv |
| scikit-learn | 1.x | Logistic Regression baseline, StandardScaler |
| sentence-transformers | 2.x | PubMedBERT embeddings (`S-PubMedBert-MS-MARCO`, 768-dim) |
| Groq Python SDK | 0.9+ | LLM calls — NER extraction + plain-language explanations |
| Pandas / NumPy | 2.x / 1.x | Feature engineering, data pipeline, evaluation |

### Frontend
| Technology | Role |
|---|---|
| Vanilla HTML / CSS / JS | Zero framework — no build step required |
| Jinja2 templates | Server-side rendering, auto-escaped output (XSS safe by default) |
| CSS Custom Properties + Grid/Flexbox | Consistent design system, glassmorphism cards |
| Native Fetch API | Async drug autocomplete, live dashboard polling (`/api/stats`) |

### Infrastructure
| Tool | Role |
|---|---|
| Docker | Containerised deployment — `Dockerfile` in repo root |
| HuggingFace Spaces | Live demo hosting (https://marwadeeb-ddi-checker.hf.space) |
| Git LFS | Large binary files: `*.pt` model files (2–46 MB each) |
| GitHub | Source control |
