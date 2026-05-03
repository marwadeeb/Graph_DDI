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

import os, sys, time, json, threading, collections as _col, datetime as _dt, re
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
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024   # 512 KB max request body

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------
_CTRL = re.compile(r'[\x00-\x1f\x7f]')

def _sanitize(s: str, max_len: int = 120) -> str:
    if not isinstance(s, str):
        return ""
    return _CTRL.sub('', s.strip())[:max_len]


@app.after_request
def _security_headers(response):
    h = response.headers
    h['X-Content-Type-Options']  = 'nosniff'
    # X-Frame-Options intentionally omitted: HF Spaces embeds the app in an
    # iframe from huggingface.co — any DENY/SAMEORIGIN value breaks that.
    # Framing is instead governed by the CSP frame-ancestors directive below.
    h['X-XSS-Protection']        = '1; mode=block'
    h['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    # Inline scripts/styles needed for our vanilla templates.
    # frame-ancestors: allow HF Spaces embedding, block all other third parties.
    h['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space;"
    )
    return response

# Lazy-loaded components
_dict_ready    = False
_drug_names    = None   # list of approved drug names for search (sorted by popularity)
_drug_pop      = None   # dict drugbank_id -> interaction count
_ddi_lookup    = None   # primary: frozenset({id_a,id_b}) -> description — O(1) dict lookup
_dict_lock     = threading.Lock()

# ---------------------------------------------------------------------------
# Live stats  (in-memory — resets on server restart by design)
# ---------------------------------------------------------------------------
_stats = {
    "total": 0, "documented": 0, "gnn_predicted": 0,
    "not_found": 0, "drug_not_found": 0, "total_ms": 0.0,
}
_recent_queries = _col.deque(maxlen=200)
_pair_counts    = _col.Counter()
_server_start   = _dt.datetime.utcnow()


def _record_query(source: str, drug_a: str, drug_b: str, elapsed_ms: float):
    _stats["total"]    += 1
    _stats["total_ms"] += elapsed_ms
    if source in _stats:
        _stats[source] += 1
    _recent_queries.appendleft({
        "ts":     _dt.datetime.utcnow().strftime("%H:%M:%S"),
        "drug_a": drug_a[:35],
        "drug_b": drug_b[:35],
        "source": source,
        "ms":     round(elapsed_ms),
    })
    if source in ("documented", "gnn_predicted", "not_found"):
        key = " × ".join(sorted([drug_a.lower(), drug_b.lower()]))
        _pair_counts[key] += 1


def _init_dict():
    """
    Load everything needed for /api/check:
      - DDI lookup dict  frozenset({id_a, id_b}) -> description  (O(1) DrugBank lookup)
      - drug name list for autocomplete (sorted by interaction count)
      - drug name + synonym caches for resolve_drug()
    """
    global _dict_ready, _drug_names, _ddi_lookup
    if _dict_ready:
        return
    with _dict_lock:
        if _dict_ready:
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

        # ── Warm up drug-resolution caches ────────────────────────────────
        import rag_query as rag
        rag.get_drugs_df()      # drugs.csv  (name <-> id table)
        rag.get_synonym_map()   # drug_attributes.csv (synonym -> canonical name)

        _dict_ready = True
        print(f"[DDI] Pipeline ready in {time.time()-t0:.1f}s", flush=True)


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
        if src == "documented":
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
        if src == "documented":
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
        "You are a friendly medication safety assistant helping everyday people understand "
        "whether their medications might interact.\n"
        "Rules:\n"
        "1. Be concise (3-5 sentences), warm, and use plain everyday language. No technical terms.\n"
        "2. Never mention 'GNN', 'Graph Neural Network', 'model', 'algorithm', 'DrugBank', "
        "'probability', or any technical system details. Just say 'our system flagged' or "
        "'this combination may carry a risk'.\n"
        "3. For documented interactions: explain what the interaction actually means in plain terms "
        "(e.g. 'this can increase bleeding risk'), then recommend speaking to a pharmacist.\n"
        "4. For predicted interactions: say something like 'we found a possible concern' — "
        "do not explain how or why it was detected.\n"
        "5. Never diagnose or prescribe. Frame everything as 'worth checking with your pharmacist'.\n"
        "6. Mention specific drug names. Be reassuring but honest."
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
    return render_template("landing.html")


@app.route("/checker", methods=["GET"])
def checker_page():
    return render_template("checker.html")

@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


@app.route("/results", methods=["GET"])
def results_page():
    """Model performance comparison page (warm + cold-start evaluation)."""
    import json as _json
    results_path = os.path.join(BASE_DIR, "data", "evaluation", "baselines_results.json")

    results_data = {}
    try:
        with open(results_path) as f:
            results_data = _json.load(f)
    except Exception:
        pass

    return render_template("results.html",
                           results=results_data.get("results", {}),
                           cold_results=results_data.get("cold_results", {}),
                           config=results_data.get("config", {}))


@app.route("/responsible", methods=["GET"])
def responsible_page():
    """Responsible ML page — explainability, fairness, privacy, robustness."""
    import json as _json
    bias_path    = os.path.join(BASE_DIR, "data", "evaluation", "responsible_ml_bias.json")
    robust_path  = os.path.join(BASE_DIR, "data", "evaluation", "responsible_ml_robust.json")
    gnn_auc_path = os.path.join(BASE_DIR, "data", "evaluation", "responsible_ml_gnn_auc.json")

    bias_data, robust_data, gnn_auc_data = {}, {}, {}
    try:
        with open(bias_path) as f:
            bias_data = _json.load(f)
    except Exception:
        pass
    try:
        with open(robust_path) as f:
            robust_data = _json.load(f)
    except Exception:
        pass
    try:
        with open(gnn_auc_path) as f:
            gnn_auc_data = _json.load(f)
    except Exception:
        pass

    return render_template("responsible.html",
                           bias=bias_data,
                           robust=robust_data,
                           gnn_auc=gnn_auc_data)


@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness check."""
    return jsonify({
        "status":      "ok",
        "dict_loaded": _dict_ready,   # DDI lookup dict + drug caches ready
    })


@app.route("/api/check", methods=["POST"])
def check_pair():
    """
    Two-stage DDI check:

      Stage 1 — Dict lookup (O(1) against 824K documented DrugBank pairs).
      Stage 2 — GNN (only if dict lookup finds nothing).
      If both say no → return not_found.

    Request body (JSON):
        { "drug_a": "Warfarin", "drug_b": "Aspirin" }

    Response:
        {
            "drug_a":   { "query": "...", "resolved": "...", "id": "..." },
            "drug_b":   { ... },
            "source":   "documented" | "gnn_predicted" | "not_found",
            "found":    true | false,
            "interaction_type":        "...",
            "interaction_description": "...",
            "gnn":       { "probability": 0.87, "mock": true } | null,
            "disclaimer": "..." | null,
            "error":      null
        }
    """
    _init_dict()
    _t0 = time.time()

    GNN_THRESHOLD = 0.43

    body   = request.get_json(silent=True) or {}
    drug_a = _sanitize(body.get("drug_a") or "")
    drug_b = _sanitize(body.get("drug_b") or "")

    if not drug_a or not drug_b:
        return jsonify({"error": "Both 'drug_a' and 'drug_b' are required."}), 400

    import rag_query as rag

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

    # Stage 1: dict lookup (O(1))
    lookup_key = frozenset([id_a, id_b])
    dict_desc  = _ddi_lookup.get(lookup_key)
    dict_hit   = dict_desc is not None

    if dict_hit:
        _record_query("documented", name_a, name_b, (time.time() - _t0) * 1000)
        return jsonify({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "source":                  "documented",
            "found":                   True,
            "interaction_type":        None,
            "interaction_description": dict_desc or (
                f"Documented interaction between {name_a} and {name_b} in DrugBank."
            ),
            "gnn":        None,
            "disclaimer": None,
            "error":      None,
        })
    # Stage 2: GNN (dict lookup found nothing)
    gnn_result = None
    try:
        import gnn_predictor
        gnn_result = gnn_predictor.predict(id_a, id_b)
    except Exception as e:
        gnn_result = {"found": None, "probability": None, "note": str(e), "mock": True}

    gnn_prob  = gnn_result.get("probability") if gnn_result else None
    gnn_found = (gnn_prob is not None) and (gnn_prob >= GNN_THRESHOLD)

    if gnn_found:
        _record_query("gnn_predicted", name_a, name_b, (time.time() - _t0) * 1000)
        explanation = gnn_predictor.explain(id_a, id_b)
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
            "gnn":        gnn_result,
            "explanation": explanation,
            "disclaimer": (
                "This interaction is not documented in DrugBank but is predicted "
                "by the Graph Neural Network based on pharmacological graph patterns. "
                "Consult a pharmacist or clinician before use. "
                "This is a clinical decision support signal, not a confirmed finding."
            ),
            "error":      None,
        })

    # No interaction found
    _record_query("not_found", name_a, name_b, (time.time() - _t0) * 1000)
    return jsonify({
        "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
        "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
        "source":                  "not_found",
        "found":                   False,
        "interaction_type":        None,
        "interaction_description": (
            f"No interaction between {name_a} and {name_b} was found in DrugBank, "
            "and the GNN model did not predict one above the confidence threshold. "
            "Absence of a recorded interaction does not guarantee safety "
            "- always consult a pharmacist or clinician when in doubt."
        ),
        "gnn":        gnn_result,
        "disclaimer": None,
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
    _init_dict()

    body    = request.get_json(silent=True) or {}
    message = _sanitize(body.get("message") or "", max_len=2000)

    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 2000:
        return jsonify({"error": "message too long (max 2000 chars)"}), 400

    import rag_query as rag

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
                    "source":                  "documented",
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
                explanation = gnn_predictor.explain(id_a, id_b)
                pair_results.append({
                    "drug_a": {"query": da["query"], "resolved": name_a, "id": id_a},
                    "drug_b": {"query": db["query"], "resolved": name_b, "id": id_b},
                    "source":                  "gnn_predicted",
                    "explanation": explanation,
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
    _init_dict()

    body  = request.get_json(silent=True) or {}
    pairs = body.get("pairs", [])

    if not pairs:
        return jsonify({"error": "'pairs' list is required and must not be empty."}), 400
    if len(pairs) > 50:
        return jsonify({"error": "Maximum 50 pairs per batch request."}), 400

    import rag_query as rag

    results = []
    for pair in pairs:
        drug_a = _sanitize(pair.get("drug_a") or "")
        drug_b = _sanitize(pair.get("drug_b") or "")

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

        lookup_key = frozenset([id_a, id_b])
        dict_desc  = _ddi_lookup.get(lookup_key)
        dict_hit   = dict_desc is not None

        results.append({
            "drug_a": {"query": drug_a, "resolved": name_a, "id": id_a},
            "drug_b": {"query": drug_b, "resolved": name_b, "id": id_b},
            "found":                   dict_hit,
            "interaction_type":        None,
            "interaction_description": dict_desc if dict_hit else None,
            "error": None,
        })

    return jsonify({"results": results})


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
    _init_dict()

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


@app.route("/landing", methods=["GET"])
def landing_page():
    from flask import redirect
    return redirect("/")


@app.route("/about", methods=["GET"])
def about_page():
    return render_template("about.html")


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/api/stats", methods=["GET"])
def api_stats():
    uptime_s = int((_dt.datetime.utcnow() - _server_start).total_seconds())
    h, rem   = divmod(uptime_s, 3600)
    m, s     = divmod(rem, 60)
    avg_ms   = round(_stats["total_ms"] / max(_stats["total"], 1))
    hit_rate = round(_stats["documented"] / max(_stats["total"], 1) * 100, 1)

    try:
        import gnn_predictor
        gnn_ok = gnn_predictor.is_available()
    except Exception:
        gnn_ok = False

    return jsonify({
        "stats":  dict(_stats),
        "recent": list(_recent_queries)[:25],
        "top_pairs": [{"pair": p, "count": n} for p, n in _pair_counts.most_common(5)],
        "system": {
            "uptime":        f"{h}h {m:02d}m {s:02d}s",
            "avg_ms":        avg_ms,
            "hit_rate":      hit_rate,
            "dict_loaded":   _dict_ready,
            "gnn_available": gnn_ok,
            "dict_size":     len(_ddi_lookup) if _ddi_lookup else 0,
            "dict_loaded":   _dict_ready,
        },
    })


# ---------------------------------------------------------------------------
# Eager startup — kick off dict/cache loading as soon as the module is imported.
# Works for both `python app.py` and Gunicorn worker processes.
# The thread is daemonised so it never blocks the process from exiting.
# ---------------------------------------------------------------------------
threading.Thread(target=_init_dict, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
