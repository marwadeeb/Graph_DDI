"""
app.py
------
Flask API for the DDI Clinical Decision Support System.

Endpoints:
    GET  /health                      -- liveness check
    POST /api/check                   -- single drug pair DDI check (RAG)
    POST /api/check/batch             -- multiple pairs in one request
    POST /api/check/compare           -- side-by-side: Exact / TF-IDF / RAG
    GET  /api/drug/search?q=<name>    -- drug name lookup / autocomplete

All responses are JSON.  The RAG pipeline (step7) is loaded once at startup.
"""

import os, sys, time, json, threading
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------------------------
# Bootstrap: add pipeline/ to path and load .env
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

def load_env():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()

load_env()
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Lazy-loaded RAG components (loaded on first request or at startup)
_rag_ready    = False
_drug_names   = None   # list of approved drug names for search (sorted by popularity)
_drug_pop     = None   # dict drugbank_id -> interaction count
_rag_lock     = threading.Lock()

# Lazy-loaded baseline components
_lookup_set   = None   # set of (id_a, id_b) known DDI pairs — exact lookup
_tfidf_vec    = None   # TF-IDF vectorizer
_tfidf_mat    = None   # TF-IDF document matrix
_tfidf_meta   = None   # metadata aligned with matrix rows
_baseline_lock = threading.Lock()
TFIDF_THRESHOLD = 0.30


def _init_rag():
    global _rag_ready, _drug_names
    if _rag_ready:
        return
    with _rag_lock:
        if _rag_ready:   # re-check inside lock
            return
        print("[DDI] Loading RAG pipeline ...", flush=True)
        t0 = time.time()

        import step7_rag_query as rag
        rag.get_embed_model()
        rag.get_index()

        drugs_path = os.path.join(BASE_DIR, "data", "step3_approved", "drugs.csv")
        ddi_path   = os.path.join(BASE_DIR, "data", "step3_approved", "drug_interactions_dedup.csv")
        if os.path.exists(drugs_path):
            df = pd.read_csv(drugs_path, usecols=["drugbank_id", "name"])
            # rank by number of known interactions (most connected drugs first)
            if os.path.exists(ddi_path):
                ddi = pd.read_csv(ddi_path, usecols=["drugbank_id_a", "drugbank_id_b"])
                counts = pd.concat([ddi["drugbank_id_a"], ddi["drugbank_id_b"]]).value_counts()
                _drug_pop = counts.to_dict()
                df["_pop"] = df["drugbank_id"].map(_drug_pop).fillna(0)
                df = df.sort_values("_pop", ascending=False).drop(columns="_pop")
            _drug_names = df.to_dict("records")
        else:
            _drug_names = []

        _rag_ready = True
        print(f"[DDI] RAG pipeline ready in {time.time()-t0:.1f}s", flush=True)


def _init_baselines():
    """Load exact-lookup set and TF-IDF index (lazy, called by /compare)."""
    global _lookup_set, _tfidf_vec, _tfidf_mat, _tfidf_meta
    if _lookup_set is not None:
        return
    with _baseline_lock:
        if _lookup_set is not None:   # re-check inside lock
            return
        print("[DDI] Loading baselines ...", flush=True)
        t0 = time.time()

        approved = os.path.join(BASE_DIR, "data", "step3_approved")
        ddi   = pd.read_csv(os.path.join(approved, "drug_interactions_dedup.csv"))
        drugs = pd.read_csv(os.path.join(approved, "drugs.csv"),
                            usecols=["drugbank_id", "name"])
        nm = dict(zip(drugs["drugbank_id"], drugs["name"]))

        # Exact lookup set (fast, small)
        _lookup_set = set(zip(ddi["drugbank_id_a"], ddi["drugbank_id_b"]))

        # TF-IDF index
        from sklearn.feature_extraction.text import TfidfVectorizer
        ddi["name_a"] = ddi["drugbank_id_a"].map(nm)
        ddi["name_b"] = ddi["drugbank_id_b"].map(nm)
        ddi = ddi.dropna(subset=["name_a", "name_b", "description"])

        texts = [f"{r['name_a']} interaction with {r['name_b']} is: {r['description']}"
                 for _, r in ddi.iterrows()]
        _tfidf_meta = list(zip(ddi["drugbank_id_a"], ddi["drugbank_id_b"],
                               ddi["name_a"], ddi["name_b"]))

        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
        _tfidf_vec = vec
        _tfidf_mat = vec.fit_transform(texts)

    print(f"[DDI] Baselines ready in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Liveness check — returns 200 as soon as Flask is up."""
    return jsonify({
        "status": "ok",
        "rag_loaded": _rag_ready,
    })


@app.route("/api/check", methods=["POST"])
def check_pair():
    """
    Two-stage DDI check (per project design):

      Stage 1 — RAG (DrugBank retrieval + LLM):
          If found → return as DOCUMENTED interaction (trustworthy, DrugBank-sourced).
      Stage 2 — GNN (only if RAG says no):
          If GNN probability >= GNN_THRESHOLD → return as PREDICTED (with disclaimer).
      If both say no → return not_found.

    Request body (JSON):
        { "drug_a": "Warfarin", "drug_b": "Aspirin", "top_k": 3 }

    Response:
        {
            "drug_a":   { "query": "...", "resolved": "...", "id": "..." },
            "drug_b":   { ... },
            "source":   "rag_documented" | "gnn_predicted" | "not_found",
            "found":    true | false,
            "interaction_type":        "...",
            "interaction_description": "...",
            "retrieval_confidence":    0.91,
            "evidence":  [...],
            "gnn":       { "probability": 0.87, "mock": true } | null,
            "disclaimer": "..." | null,
            "fallback":   null | "exact_lookup" | "tfidf",
            "error":      null
        }
    """
    _init_rag()

    GNN_THRESHOLD = 0.5

    body   = request.get_json(silent=True) or {}
    drug_a = (body.get("drug_a") or "").strip()
    drug_b = (body.get("drug_b") or "").strip()
    top_k  = int(body.get("top_k", 3))

    if not drug_a or not drug_b:
        return jsonify({"error": "Both 'drug_a' and 'drug_b' are required."}), 400

    import step7_rag_query as rag

    try:
        id_a, name_a = rag.resolve_drug(drug_a)
        id_b, name_b = rag.resolve_drug(drug_b)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    # ── Stage 1: RAG ──────────────────────────────────────────────────────
    retrieved = rag.retrieve(name_a, name_b, top_k=top_k)
    retrieval_confidence = round(
        float(sum(r["score"] for r in retrieved) / len(retrieved)), 4
    ) if retrieved else 0.0

    evidence_list = [
        {"rank": i+1, "score": round(r["score"], 4), "text": r["text"]}
        for i, r in enumerate(retrieved)
    ]

    fallback_method = None
    try:
        rag_result = rag.call_llm(name_a, name_b, retrieved)
    except Exception as llm_err:
        print(f"[DDI] LLM unavailable ({llm_err}), using offline fallback", flush=True)
        _init_baselines()
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        if (id_a, id_b) in _lookup_set or (id_b, id_a) in _lookup_set:
            fallback_method = "exact_lookup"
            rag_result = {
                "found": True,
                "interaction_type": "known",
                "interaction_description": (
                    f"{name_a} and {name_b} have a documented interaction in DrugBank "
                    "(offline fallback — LLM unavailable)."
                ),
            }
        else:
            q_vec = _tfidf_vec.transform([f"{name_a} interaction with {name_b} is:"])
            best  = float(_cos_sim(q_vec, _tfidf_mat).flatten().max())
            fallback_method = "tfidf"
            rag_result = {
                "found": best >= TFIDF_THRESHOLD,
                "interaction_type": "probable" if best >= TFIDF_THRESHOLD else None,
                "interaction_description": (
                    f"TF-IDF similarity {best:.3f} (offline fallback — LLM unavailable)."
                ),
            }

    # ── Stage 1 hit: documented interaction ──────────────────────────────
    if rag_result.get("found"):
        return jsonify({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "source":                  "rag_documented",
            "found":                   True,
            "interaction_type":        rag_result.get("interaction_type"),
            "interaction_description": rag_result.get("interaction_description"),
            "retrieval_confidence":    retrieval_confidence,
            "evidence":   evidence_list,
            "gnn":        None,
            "disclaimer": None,
            "fallback":   fallback_method,
            "error":      None,
        })

    # ── Stage 2: GNN (RAG found nothing) ─────────────────────────────────
    gnn_result = None
    try:
        import gnn_predictor
        gnn_result = gnn_predictor.predict(id_a, id_b)
    except Exception as e:
        gnn_result = {"found": None, "probability": None, "note": str(e), "mock": True}

    gnn_prob  = gnn_result.get("probability") if gnn_result else None
    gnn_found = (gnn_prob is not None) and (gnn_prob >= GNN_THRESHOLD)

    if gnn_found:
        return jsonify({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "source":                  "gnn_predicted",
            "found":                   True,
            "interaction_type":        "predicted",
            "interaction_description": (
                f"No documented interaction found in DrugBank for {name_a} and {name_b}. "
                f"The Graph Neural Network predicts a possible interaction "
                f"(probability {gnn_prob:.0%})."
            ),
            "retrieval_confidence":    retrieval_confidence,
            "evidence":   evidence_list,
            "gnn":        gnn_result,
            "disclaimer": (
                "⚠ This interaction is not documented in DrugBank but is predicted "
                "by the Graph Neural Network based on pharmacological graph patterns. "
                "To reduce risk, consider separating doses by at least 2–3 hours "
                "and consult a pharmacist or clinician before use. "
                "This is a clinical decision support signal — not a confirmed finding."
            ),
            "fallback":   fallback_method,
            "error":      None,
        })

    # ── No interaction found ──────────────────────────────────────────────
    return jsonify({
        "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
        "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
        "source":                  "not_found",
        "found":                   False,
        "interaction_type":        None,
        "interaction_description": (
            f"No interaction between {name_a} and {name_b} was found in DrugBank, "
            "and the GNN model did not predict one above the confidence threshold. "
            "Important: absence of a recorded interaction does not guarantee safety — "
            "always consult a pharmacist or clinician when in doubt."
        ),
        "retrieval_confidence":    retrieval_confidence,
        "evidence":   evidence_list,
        "gnn":        gnn_result,
        "disclaimer": None,
        "fallback":   fallback_method,
        "error":      None,
    })


@app.route("/api/check/batch", methods=["POST"])
def check_batch():
    """
    Check DDI for multiple pairs in one request.

    Request body (JSON):
        {
            "pairs": [
                {"drug_a": "Warfarin", "drug_b": "Aspirin"},
                {"drug_a": "Metformin", "drug_b": "Lisinopril"}
            ],
            "top_k": 3
        }

    Response:
        { "results": [ <same shape as /api/check for each pair> ] }
    """
    _init_rag()

    body  = request.get_json(silent=True) or {}
    pairs = body.get("pairs", [])
    top_k = int(body.get("top_k", 3))

    if not pairs:
        return jsonify({"error": "'pairs' list is required and must not be empty."}), 400
    if len(pairs) > 50:
        return jsonify({"error": "Maximum 50 pairs per batch request."}), 400

    import step7_rag_query as rag

    results = []
    for pair in pairs:
        drug_a = (pair.get("drug_a") or "").strip()
        drug_b = (pair.get("drug_b") or "").strip()

        if not drug_a or not drug_b:
            results.append({"error": "Both 'drug_a' and 'drug_b' are required."})
            continue

        try:
            id_a, name_a = rag.resolve_drug(drug_a)
            id_b, name_b = rag.resolve_drug(drug_b)
        except ValueError as e:
            results.append({"drug_a": {"query": drug_a}, "drug_b": {"query": drug_b},
                            "error": str(e)})
            continue

        retrieved = rag.retrieve(name_a, name_b, top_k=top_k)

        try:
            result = rag.call_llm(name_a, name_b, retrieved)
        except Exception as e:
            results.append({"drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
                            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
                            "error": f"LLM error: {e}"})
            continue

        confidence = round(float(sum(r["score"] for r in retrieved) / len(retrieved)), 4) if retrieved else 0.0

        results.append({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "found":                  result.get("found", False),
            "interaction_type":       result.get("interaction_type"),
            "interaction_description":result.get("interaction_description"),
            "confidence":             confidence,
            "evidence": [
                {"rank": i+1, "score": round(r["score"], 4), "text": r["text"]}
                for i, r in enumerate(retrieved)
            ],
            "error": None,
        })

    return jsonify({"results": results})


@app.route("/api/check/compare", methods=["POST"])
def check_compare():
    """
    Run all three methods side-by-side for a single drug pair.

    Request body (JSON):
        { "drug_a": "Warfarin", "drug_b": "Aspirin", "top_k": 3 }

    Response:
        {
            "drug_a": { "query": "...", "resolved": "...", "id": "..." },
            "drug_b": { ... },
            "methods": {
                "exact_lookup": { "found": true,  "note": "Pair in DrugBank database" },
                "tfidf":        { "found": true,  "score": 0.82, "best_match": "..." },
                "rag":          { "found": true,  "interaction_type": "...",
                                  "interaction_description": "...", "evidence": [...] }
            }
        }
    """
    _init_rag()
    _init_baselines()

    body   = request.get_json(silent=True) or {}
    drug_a = (body.get("drug_a") or "").strip()
    drug_b = (body.get("drug_b") or "").strip()
    top_k  = int(body.get("top_k", 3))

    if not drug_a or not drug_b:
        return jsonify({"error": "Both 'drug_a' and 'drug_b' are required."}), 400

    import step7_rag_query as rag
    from sklearn.metrics.pairwise import cosine_similarity

    try:
        id_a, name_a = rag.resolve_drug(drug_a)
        id_b, name_b = rag.resolve_drug(drug_b)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    results = {
        "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
        "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
        "methods": {}
    }

    # ── Method 1: Exact lookup ─────────────────────────────────────────────
    exact_found = ((id_a, id_b) in _lookup_set or (id_b, id_a) in _lookup_set)
    results["methods"]["exact_lookup"] = {
        "found": exact_found,
        "note": ("Pair found in DrugBank DDI database"
                 if exact_found else "Pair not in DrugBank DDI database"),
    }

    # ── Method 2: TF-IDF ──────────────────────────────────────────────────
    query   = f"{name_a} interaction with {name_b} is:"
    q_vec   = _tfidf_vec.transform([query])
    sims    = cosine_similarity(q_vec, _tfidf_mat).flatten()
    best_i  = int(sims.argmax())
    best_score = float(sims[best_i])
    tfidf_found = best_score >= TFIDF_THRESHOLD
    best_meta   = _tfidf_meta[best_i] if _tfidf_meta else ("", "", "", "")
    results["methods"]["tfidf"] = {
        "found":      tfidf_found,
        "score":      round(best_score, 4),
        "threshold":  TFIDF_THRESHOLD,
        "best_match": f"{best_meta[2]} x {best_meta[3]}",
    }

    # ── Method 3: RAG (PubMedBERT + LLM) ─────────────────────────────────
    retrieved = rag.retrieve(name_a, name_b, top_k=top_k)
    try:
        rag_result = rag.call_llm(name_a, name_b, retrieved)
        results["methods"]["rag"] = {
            "found":                  rag_result.get("found", False),
            "interaction_type":       rag_result.get("interaction_type"),
            "interaction_description":rag_result.get("interaction_description"),
            "evidence": [
                {"rank": i+1, "score": round(r["score"], 4), "text": r["text"]}
                for i, r in enumerate(retrieved)
            ],
            "error": None,
        }
    except Exception as e:
        results["methods"]["rag"] = {"found": None, "error": str(e)}

    # ── Method 4: GNN (Graph Neural Network) ──────────────────────────────
    try:
        import gnn_predictor
        gnn_result = gnn_predictor.predict(id_a, id_b)
        results["methods"]["gnn"] = gnn_result
    except Exception as e:
        results["methods"]["gnn"] = {
            "found": None, "probability": None,
            "note": str(e), "mock": True,
        }

    return jsonify(results)


@app.route("/api/drug/search", methods=["GET"])
def drug_search():
    """
    Drug name autocomplete / lookup.

    Query params:
        q     -- search string (min 2 chars)
        limit -- max results (default 10, max 50)

    Response:
        { "results": [ {"id": "DB00682", "name": "Warfarin"}, ... ] }
    """
    _init_rag()

    q     = (request.args.get("q") or "").strip().lower()
    limit = min(int(request.args.get("limit", 10)), 50)

    if len(q) < 2:
        return jsonify({"results": []})

    pop = _drug_pop or {}
    matches = [
        d for d in (_drug_names or [])
        if q in d["name"].lower()
    ]
    # starts-with first, then by interaction count descending
    matches.sort(key=lambda d: (
        not d["name"].lower().startswith(q),
        -pop.get(d["drugbank_id"], 0),
    ))

    return jsonify({"results": matches[:limit]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # threaded=False prevents multiple threads loading the 2.5 GB FAISS index
    # simultaneously which would crash low-RAM machines.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
