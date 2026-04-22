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

# Lazy-loaded components
_rag_ready     = False
_drug_names    = None   # list of approved drug names for search (sorted by popularity)
_drug_pop      = None   # dict drugbank_id -> interaction count
_ddi_lookup    = None   # primary: frozenset({id_a,id_b}) -> description — O(1) dict lookup
_rag_lock      = threading.Lock()

# FAISS is intentionally NOT loaded at startup.
# It is loaded on-demand, only when the caller passes use_rag=True in the request body.
# _faiss_lock ensures it is initialised at most once across concurrent requests.
_faiss_lock        = threading.Lock()
_faiss_loaded      = False
_faiss_unavailable = False   # True when the index file is missing (e.g. deployed without it)


def _ensure_faiss() -> bool:
    """
    Load PubMedBERT embed model + FAISS index on first use. Thread-safe.
    Returns True if FAISS is ready, False if the index file is missing.
    """
    global _faiss_loaded, _faiss_unavailable
    if _faiss_loaded:
        return True
    if _faiss_unavailable:
        return False
    with _faiss_lock:
        if _faiss_loaded:
            return True
        if _faiss_unavailable:
            return False
        try:
            import step7_rag_query as rag
            print("[DDI] Loading embed model + FAISS index (on-demand) …", flush=True)
            t0 = time.time()
            rag.get_embed_model()
            rag.get_index()
            _faiss_loaded = True
            print(f"[DDI] FAISS ready in {time.time()-t0:.1f}s", flush=True)
            return True
        except FileNotFoundError:
            _faiss_unavailable = True
            print("[DDI] FAISS index not found — RAG evidence unavailable on this deployment.",
                  flush=True)
            return False
        except Exception as e:
            _faiss_unavailable = True
            print(f"[DDI] FAISS load error: {e}", flush=True)
            return False

# Lazy-loaded baseline/compare components
_lookup_set    = None   # set of (id_a, id_b) — for compare endpoint exact-lookup method
_tfidf_vec     = None
_tfidf_mat     = None
_tfidf_meta    = None
_baseline_lock = threading.Lock()
TFIDF_THRESHOLD = 0.30


def _init_rag():
    """
    Load everything needed for /api/check:
      - drug name list (autocomplete + popularity ranking)
      - DDI lookup dict  frozenset({id_a, id_b}) -> description  (primary lookup)
      - FAISS index + embed model                                  (fallback / evidence)
    """
    global _rag_ready, _drug_names, _ddi_lookup
    if _rag_ready:
        return
    with _rag_lock:
        if _rag_ready:
            return
        print("[DDI] Loading pipeline ...", flush=True)
        t0 = time.time()

        approved  = os.path.join(BASE_DIR, "data", "step3_approved")
        ddi_path  = os.path.join(approved, "drug_interactions_dedup.csv")
        drug_path = os.path.join(approved, "drugs.csv")

        # ── Build primary DDI lookup dict ──────────────────────────────────
        # frozenset({id_a, id_b}) → description string
        # Order-independent, O(1) lookup, guaranteed to find any documented pair
        if os.path.exists(ddi_path):
            ddi = pd.read_csv(ddi_path)
            _ddi_lookup = {}
            for _, row in ddi.iterrows():
                key  = frozenset([row["drugbank_id_a"], row["drugbank_id_b"]])
                desc = str(row["description"]).strip() if pd.notna(row["description"]) else ""
                _ddi_lookup[key] = desc
            print(f"[DDI] Lookup dict: {len(_ddi_lookup):,} pairs", flush=True)
        else:
            _ddi_lookup = {}

        # ── Drug name list for autocomplete ────────────────────────────────
        if os.path.exists(drug_path):
            df = pd.read_csv(drug_path, usecols=["drugbank_id", "name"])
            if _ddi_lookup:
                counts = {}
                for key in _ddi_lookup:
                    for did in key:
                        counts[did] = counts.get(did, 0) + 1
                df["_pop"] = df["drugbank_id"].map(counts).fillna(0)
                df = df.sort_values("_pop", ascending=False).drop(columns="_pop")
            _drug_names = df.to_dict("records")
        else:
            _drug_names = []

        # ── Warm up step7's drug-resolution caches ────────────────────────
        # resolve_drug() lazily loads drugs.csv + drug_attributes.csv on first
        # call; pre-loading them here means the first /api/check request is fast.
        import step7_rag_query as rag
        rag.get_drugs_df()      # drugs.csv  (name ↔ id table)
        rag.get_synonym_map()   # drug_attributes.csv (synonym → canonical name)

        _rag_ready = True
        print(f"[DDI] Pipeline ready (dict lookup) in {time.time()-t0:.1f}s", flush=True)
        # FAISS is NOT loaded here — it is loaded on-demand via _ensure_faiss()
        # only when the caller explicitly requests RAG evidence (use_rag=True).


def _init_baselines():
    """Load TF-IDF index (lazy, used by /compare for method comparison only)."""
    global _lookup_set, _tfidf_vec, _tfidf_mat, _tfidf_meta
    if _lookup_set is not None:
        return
    with _baseline_lock:
        if _lookup_set is not None:
            return
        print("[DDI] Loading baselines ...", flush=True)
        t0 = time.time()

        approved = os.path.join(BASE_DIR, "data", "step3_approved")
        ddi   = pd.read_csv(os.path.join(approved, "drug_interactions_dedup.csv"))
        drugs = pd.read_csv(os.path.join(approved, "drugs.csv"),
                            usecols=["drugbank_id", "name"])
        nm = dict(zip(drugs["drugbank_id"], drugs["name"]))

        _lookup_set = set(zip(ddi["drugbank_id_a"], ddi["drugbank_id_b"]))

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
# Groq LLM helpers  (chatbot — NER + explanation)
# ---------------------------------------------------------------------------

_groq_client = None
_groq_lock   = threading.Lock()

GROQ_NER_MODEL  = "llama-3.1-8b-instant"    # fast, free — structured NER task
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"  # better quality for explanation


def _get_groq():
    """Lazy Groq client.  Returns None when GROQ_API_KEY is not set."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    with _groq_lock:
        if _groq_client is not None:
            return _groq_client
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            from groq import Groq
            _groq_client = Groq(api_key=api_key)
            print("[DDI] Groq client initialised.", flush=True)
        except Exception as e:
            print(f"[DDI] Groq init error: {e}", flush=True)
        return _groq_client


def extract_drugs_nlp(text: str) -> list:
    """
    Call Groq LLM to extract drug / medication names from free text (NER).
    Returns a list of drug name strings.  Falls back to [] on error / no key.
    """
    client = _get_groq()
    if not client:
        return []

    prompt = (
        "You are a medical named-entity recognition system. "
        "Extract every drug, medication, or supplement name mentioned in the text below. "
        "Return ONLY a valid JSON array of strings — no explanation, no markdown fences.\n"
        "Example output: [\"Warfarin\", \"Aspirin\"]\n"
        "If no drugs are found, return: []\n\n"
        f"Text: {text}"
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_NER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        drugs = json.loads(raw)
        if isinstance(drugs, list):
            return [str(d).strip() for d in drugs if str(d).strip()]
        return []
    except Exception as e:
        print(f"[DDI] NER error: {e}", flush=True)
        return []


def _fallback_reply(pair_results: list, drugs_not_found: list) -> str:
    """Template-based reply used when Groq is unavailable."""
    lines = []
    if drugs_not_found:
        lines.append(
            f"⚠ I couldn't find these in my database: **{', '.join(drugs_not_found)}**. "
            "Please check the spelling or use the Checker page for autocomplete suggestions."
        )
    for r in pair_results:
        src   = r.get("source", "")
        name_a = r.get("drug_a", {}).get("resolved", "Drug A")
        name_b = r.get("drug_b", {}).get("resolved", "Drug B")
        desc   = r.get("interaction_description", "")
        if src == "rag_documented":
            lines.append(
                f"⚠ **{name_a}** and **{name_b}** have a **documented interaction** in DrugBank. "
                "Please consult a pharmacist or clinician before combining these medications. "
                f"{desc}"
            )
        elif src == "gnn_predicted":
            prob = (r.get("gnn") or {}).get("probability") or 0
            lines.append(
                f"🔶 The GNN model predicts a **possible interaction** between **{name_a}** "
                f"and **{name_b}** (probability {prob:.0%}). "
                "This is not confirmed in DrugBank — consult a pharmacist before use."
            )
        else:
            lines.append(
                f"✓ No interaction between **{name_a}** and **{name_b}** was found in DrugBank "
                "or predicted by the GNN. Absence of data does not guarantee safety — always "
                "consult a healthcare professional when in doubt."
            )
    if not lines:
        return (
            "I couldn't identify any drug names in your message. "
            "Try mentioning specific medication names, e.g. \"Is warfarin safe with aspirin?\""
        )
    return "\n\n".join(lines)


def generate_chat_reply(
    user_message: str,
    drugs_resolved: list,
    pair_results: list,
    drugs_not_found: list,
) -> str:
    """
    Generate a plain-language conversational reply via Groq LLM.
    Falls back to _fallback_reply() when no API key is configured.
    """
    client = _get_groq()
    if not client:
        return _fallback_reply(pair_results, drugs_not_found)

    # Build structured context for the LLM
    ctx_lines = []
    for r in pair_results:
        src    = r.get("source", "")
        name_a = r.get("drug_a", {}).get("resolved", "")
        name_b = r.get("drug_b", {}).get("resolved", "")
        desc   = r.get("interaction_description", "")
        if src == "rag_documented":
            ctx_lines.append(
                f"- {name_a} + {name_b}: DOCUMENTED in DrugBank. {desc}"
            )
        elif src == "gnn_predicted":
            prob = (r.get("gnn") or {}).get("probability") or 0
            ctx_lines.append(
                f"- {name_a} + {name_b}: PREDICTED by GNN (probability {prob:.0%}). "
                "Not confirmed in DrugBank."
            )
        else:
            ctx_lines.append(f"- {name_a} + {name_b}: No interaction found.")

    if drugs_not_found:
        ctx_lines.append(
            f"- Drugs not found in database: {', '.join(drugs_not_found)}"
        )

    context = "\n".join(ctx_lines) if ctx_lines else "No pairs analysed."

    system_prompt = (
        "You are a clinical decision-support assistant that helps users understand "
        "drug-drug interaction (DDI) results from the DDI Checker system, which uses "
        "DrugBank data and a Graph Neural Network.\n"
        "Rules:\n"
        "1. Be concise (3-5 sentences), empathetic, and use plain language.\n"
        "2. For documented interactions: always recommend consulting a pharmacist or clinician.\n"
        "3. For GNN-predicted interactions: clearly state these are model predictions, not confirmed findings.\n"
        "4. Never diagnose or prescribe. Always remind users this is decision support, not medical advice.\n"
        "5. Mention specific drug names in your reply."
    )
    user_prompt = (
        f"User message: \"{user_message}\"\n\n"
        f"DDI analysis results:\n{context}\n\n"
        "Write a conversational reply explaining these results to the user."
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DDI] Groq chat error: {e}", flush=True)
        return _fallback_reply(pair_results, drugs_not_found)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness check."""
    return jsonify({
        "status":           "ok",
        "rag_loaded":       _rag_ready,         # dict lookup + drug caches ready
        "faiss_loaded":     _faiss_loaded,       # FAISS ready (only after first use_rag=True call)
        "faiss_available":  not _faiss_unavailable,  # False on deployments without the index file
    })


@app.route("/api/check", methods=["POST"])
def check_pair():
    """
    Two-stage DDI check (per project design):

      Stage 1 — Dict lookup (primary) + FAISS (fallback + evidence):
          If found → return as DOCUMENTED interaction (DrugBank-sourced, O(1) lookup).
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

    body    = request.get_json(silent=True) or {}
    drug_a  = (body.get("drug_a") or "").strip()
    drug_b  = (body.get("drug_b") or "").strip()
    top_k   = int(body.get("top_k", 3))
    use_rag = bool(body.get("use_rag", False))   # caller must opt-in to FAISS evidence

    if not drug_a or not drug_b:
        return jsonify({"error": "Both 'drug_a' and 'drug_b' are required."}), 400

    import step7_rag_query as rag

    try:
        id_a, name_a = rag.resolve_drug(drug_a)
    except ValueError:
        return jsonify({
            "source":         "drug_not_found",
            "drug_not_found": drug_a,
            "found":          False,
            "error":          None,
        })

    try:
        id_b, name_b = rag.resolve_drug(drug_b)
    except ValueError:
        return jsonify({
            "source":         "drug_not_found",
            "drug_not_found": drug_b,
            "found":          False,
            "error":          None,
        })

    # ── Stage 1: dict lookup (O(1), always runs) ─────────────────────────────
    t_rag_start = time.time()

    lookup_key = frozenset([id_a, id_b])
    dict_desc  = _ddi_lookup.get(lookup_key)
    dict_hit   = dict_desc is not None

    # ── FAISS evidence — only when user explicitly requests it ───────────────
    # FAISS is not loaded at startup; first use_rag=True call triggers load (~30 s).
    # Subsequent calls are fast (model + index already in memory).
    retrieved     = []
    evidence_list = []
    if use_rag and _ensure_faiss():
        try:
            retrieved = rag.retrieve(name_a, name_b, top_k=top_k)
        except Exception:
            retrieved = []
        evidence_list = [
            {"rank": i+1, "score": round(r["score"], 4), "text": r["text"]}
            for i, r in enumerate(retrieved)
        ]
        # FAISS safety net: catches the pair if dict somehow missed it
        if not dict_hit:
            faiss_match = rag.match_retrieved(id_a, id_b, retrieved)
            if faiss_match:
                dict_hit  = True
                dict_desc = faiss_match.get("interaction_description", "")

    t_rag_ms = round((time.time() - t_rag_start) * 1000)

    rag_result      = None
    fallback_method = None
    if dict_hit:
        rag_result = {
            "found":                   True,
            "interaction_type":        None,
            "interaction_description": dict_desc or (
                f"Documented interaction between {name_a} and {name_b} in DrugBank."
            ),
            "confidence":              1.0,
        }

    retrieval_confidence = 1.0 if dict_hit else (
        round(float(sum(r["score"] for r in retrieved) / len(retrieved)), 4)
        if retrieved else 0.0
    )

    # ── Stage 1 hit: documented interaction ──────────────────────────────
    if rag_result and rag_result.get("found"):
        return jsonify({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "source":                  "rag_documented",
            "found":                   True,
            "interaction_type":        rag_result.get("interaction_type"),
            "interaction_description": rag_result.get("interaction_description"),
            "retrieval_confidence":    retrieval_confidence,
            "retrieval_ms":            t_rag_ms,
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
            "retrieval_ms":            t_rag_ms,
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
        "retrieval_ms":            t_rag_ms,
        "evidence":   evidence_list,
        "gnn":        gnn_result,
        "disclaimer": None,
        "fallback":   fallback_method,
        "error":      None,
    })


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """
    Chatbot endpoint.

    Flow:
      1. Groq NER  — extract drug names from free-text message
      2. resolve_drug() — map raw names to DrugBank IDs
      3. Dict lookup + GNN — check every unique pair
      4. Groq LLM  — generate plain-language explanation

    Request body (JSON):
        { "message": "I'm taking warfarin and want to start aspirin, is that safe?" }

    Response:
        {
            "reply":           "...",     # LLM-generated explanation
            "drugs":           [...],     # resolved drug objects
            "drugs_not_found": [...],     # names not in DrugBank
            "pairs":           [...],     # one entry per unique pair
            "llm_available":   true,
            "error":           null
        }
    """
    _init_rag()

    body    = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 2000:
        return jsonify({"error": "message too long (max 2000 chars)"}), 400

    import step7_rag_query as rag

    llm_available   = _get_groq() is not None

    # ── Step 1: NER ──────────────────────────────────────────────────────
    drug_names_raw = extract_drugs_nlp(message)

    # ── Step 2: Resolve each name ────────────────────────────────────────
    drugs_resolved  = []  # [{"query": str, "name": str, "id": str}]
    drugs_not_found = []

    for raw in drug_names_raw:
        try:
            did, dname = rag.resolve_drug(raw)
            # Deduplicate by DrugBank ID
            if not any(d["id"] == did for d in drugs_resolved):
                drugs_resolved.append({"query": raw, "name": dname, "id": did})
        except ValueError:
            if raw not in drugs_not_found:
                drugs_not_found.append(raw)

    # ── Step 3: Check every unique pair ──────────────────────────────────
    GNN_THRESHOLD = 0.5
    pair_results  = []
    seen_pairs    = set()

    for i, da in enumerate(drugs_resolved):
        for db in drugs_resolved[i + 1:]:
            pair_key = frozenset([da["id"], db["id"]])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            id_a, name_a = da["id"], da["name"]
            id_b, name_b = db["id"], db["name"]

            # Stage 1 — dict lookup
            dict_desc = _ddi_lookup.get(frozenset([id_a, id_b]))
            if dict_desc is not None:
                pair_results.append({
                    "drug_a": {"query": da["query"], "resolved": name_a, "id": id_a},
                    "drug_b": {"query": db["query"], "resolved": name_b, "id": id_b},
                    "source":                  "rag_documented",
                    "found":                   True,
                    "interaction_description": dict_desc or (
                        f"Documented interaction between {name_a} and {name_b} in DrugBank."
                    ),
                    "gnn":       None,
                    "disclaimer": None,
                })
                continue

            # Stage 2 — GNN
            gnn_result = None
            try:
                import gnn_predictor
                gnn_result = gnn_predictor.predict(id_a, id_b)
            except Exception as e:
                gnn_result = {"found": None, "probability": None,
                              "note": str(e), "mock": True}

            gnn_prob  = (gnn_result or {}).get("probability")
            gnn_found = gnn_prob is not None and gnn_prob >= GNN_THRESHOLD

            if gnn_found:
                pair_results.append({
                    "drug_a": {"query": da["query"], "resolved": name_a, "id": id_a},
                    "drug_b": {"query": db["query"], "resolved": name_b, "id": id_b},
                    "source":                  "gnn_predicted",
                    "found":                   True,
                    "interaction_description": (
                        f"No documented interaction found in DrugBank for {name_a} and {name_b}. "
                        f"The GNN predicts a possible interaction (probability {gnn_prob:.0%})."
                    ),
                    "gnn":        gnn_result,
                    "disclaimer": (
                        "⚠ GNN prediction — not a confirmed DrugBank finding. "
                        "Consult a pharmacist or clinician before use."
                    ),
                })
            else:
                pair_results.append({
                    "drug_a": {"query": da["query"], "resolved": name_a, "id": id_a},
                    "drug_b": {"query": db["query"], "resolved": name_b, "id": id_b},
                    "source":                  "not_found",
                    "found":                   False,
                    "interaction_description": (
                        f"No interaction between {name_a} and {name_b} was found "
                        "in DrugBank or predicted by the GNN."
                    ),
                    "gnn":        gnn_result,
                    "disclaimer": None,
                })

    # ── Step 4: LLM explanation ──────────────────────────────────────────
    reply = generate_chat_reply(message, drugs_resolved, pair_results, drugs_not_found)

    return jsonify({
        "reply":           reply,
        "drugs":           drugs_resolved,
        "drugs_not_found": drugs_not_found,
        "pairs":           pair_results,
        "llm_available":   llm_available,
        "error":           None,
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
    pairs   = body.get("pairs", [])
    top_k   = int(body.get("top_k", 3))
    use_rag = bool(body.get("use_rag", False))

    if not pairs:
        return jsonify({"error": "'pairs' list is required and must not be empty."}), 400
    if len(pairs) > 50:
        return jsonify({"error": "Maximum 50 pairs per batch request."}), 400

    import step7_rag_query as rag

    faiss_ok = use_rag and _ensure_faiss()   # load once before the loop

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

        retrieved = []
        if faiss_ok:
            try:
                retrieved = rag.retrieve(name_a, name_b, top_k=top_k)
            except Exception:
                retrieved = []
        result = rag.match_retrieved(id_a, id_b, retrieved) if retrieved else None
        if result is None:
            result = {"found": False, "interaction_type": None,
                      "interaction_description": None, "confidence": 0.0}

        results.append({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "found":                   result.get("found", False),
            "interaction_type":        result.get("interaction_type"),
            "interaction_description": result.get("interaction_description"),
            "confidence":              result.get("confidence", 0.0),
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
    # compare endpoint: try direct match first, fall back to LLM for comparison value
    rag_direct = rag.match_retrieved(id_a, id_b, retrieved)
    if rag_direct:
        results["methods"]["rag"] = {
            "found":                   rag_direct["found"],
            "interaction_type":        rag_direct.get("interaction_type"),
            "interaction_description": rag_direct.get("interaction_description"),
            "confidence":              rag_direct.get("confidence"),
            "via_llm":                 False,
            "evidence": [
                {"rank": i+1, "score": round(r["score"], 4), "text": r["text"]}
                for i, r in enumerate(retrieved)
            ],
            "error": None,
        }
    else:
        try:
            rag_llm = rag.call_llm(name_a, name_b, retrieved)
            results["methods"]["rag"] = {
                "found":                   rag_llm.get("found", False),
                "interaction_type":        rag_llm.get("interaction_type"),
                "interaction_description": rag_llm.get("interaction_description"),
                "via_llm":                 True,
                "evidence": [
                    {"rank": i+1, "score": round(r["score"], 4), "text": r["text"]}
                    for i, r in enumerate(retrieved)
                ],
                "error": None,
            }
        except Exception as e:
            results["methods"]["rag"] = {"found": None, "via_llm": True, "error": str(e)}

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
# Eager startup — kick off dict/cache loading as soon as the module is imported.
# Works for both `python app.py` and Gunicorn worker processes.
# The thread is daemonised so it never blocks the process from exiting.
# ---------------------------------------------------------------------------
threading.Thread(target=_init_rag, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
