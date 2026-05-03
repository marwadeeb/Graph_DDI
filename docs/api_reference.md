# API Reference — DDI Checker

REST API served by `app.py` on port 7860.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Landing page |
| GET | `/checker` | DDI Checker UI |
| GET | `/chat` | Chat interface |
| GET | `/results` | Model performance page |
| GET | `/responsible` | Responsible ML page (RM1–RM4) |
| GET | `/dashboard` | Live dashboard |
| GET | `/about` | About page |
| GET | `/health` | Liveness / readiness check |
| GET | `/api/stats` | Live stats (queries, hit rate, top pairs) |
| POST | `/api/check` | Check a single drug pair |
| POST | `/api/check/batch` | Check up to 50 pairs |
| POST | `/api/check/compare` | Side-by-side dict vs GNN comparison |
| POST | `/api/chat` | Chat endpoint (NER → lookup → explanation) |
| GET | `/api/drug/search?q=<query>&limit=<n>` | Drug name autocomplete |

---

## `POST /api/check`

**Request:**
```json
{ "drug_a": "Warfarin", "drug_b": "Aspirin" }
```
Accepts drug names **or** DrugBank IDs (`DB00682`). Case-insensitive.

**Response:**
```json
{
  "drug_a": { "query": "Warfarin", "resolved": "Warfarin", "id": "DB00682" },
  "drug_b": { "query": "Aspirin",  "resolved": "Acetylsalicylic acid", "id": "DB00945" },
  "source":  "documented",
  "found":   true,
  "interaction_description": "Acetylsalicylic acid may increase the anticoagulant activities...",
  "gnn":     null,
  "error":   null
}
```

**`source` values:**
| Value | Meaning |
|---|---|
| `"documented"` | Pair found in DrugBank (O(1) dict lookup) |
| `"gnn_predicted"` | Novel pair — GNN link prediction above threshold |
| `"not_found"` | Not in DrugBank; GNN score below threshold |
| `"drug_not_found"` | Drug name could not be resolved |

---

## `POST /api/check/batch`

**Request:**
```json
{
  "pairs": [
    { "drug_a": "Warfarin", "drug_b": "Aspirin" },
    { "drug_a": "Metformin", "drug_b": "Lisinopril" }
  ]
}
```
Max 50 pairs per request.

**Response:**
```json
{ "results": [ { /* same structure as /api/check */ }, ... ] }
```

---

## `GET /api/drug/search`

```
GET /api/drug/search?q=war&limit=5
```

Returns a list of drug name suggestions for autocomplete:
```json
[
  { "name": "Warfarin", "id": "DB00682" },
  ...
]
```

---

## `GET /health`

```json
{ "status": "ok", "model_loaded": true, "drugs_loaded": 4795 }
```

---

## `GET /api/stats`

Live dashboard metrics:
```json
{
  "total_queries": 142,
  "documented_hits": 89,
  "gnn_predictions": 31,
  "not_found": 22,
  "avg_latency_ms": 14.2,
  "top_pairs": [ ... ]
}
```

---

## Authentication

No authentication required. The API is public and read-only.
`GROQ_API_KEY` is required server-side for `/api/chat` (NER + explanation); all other endpoints work without it.

---

*For pipeline details see [pipeline.md](pipeline.md). For model details see [model_architecture.md](model_architecture.md).*
