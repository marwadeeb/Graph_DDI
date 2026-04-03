"""
step7_rag_query.py
------------------
RAG query pipeline for DDI detection, following the paper's three-stage design:

  Stage 1 -- Retrieve top-k interaction sentences from FAISS (k=3)
  Stage 2 -- Feed retrieved evidence to LLM with constrained JSON prompt
  Stage 3 -- Return structured result: found, interaction_type, description

LLM  : nvidia/nemotron-3-super-120b-a12b via OpenRouter (free)
Model: pritamdeka/S-PubMedBert-MS-MARCO for query embedding (same as index)

Usage (interactive):
    python pipeline/step7_rag_query.py
    python pipeline/step7_rag_query.py --drug-a Warfarin --drug-b Aspirin
    python pipeline/step7_rag_query.py --drug-a DB00682 --drug-b DB00945
    python pipeline/step7_rag_query.py --top-k 5   # retrieve 5 instead of 3

Requires:
    data/rag_index/faiss.index   (built by step6_rag_index.py)
    data/rag_index/metadata.pkl
    OPENROUTER_API_KEY in .env or environment
"""

import os, sys, json, pickle, argparse, textwrap
import numpy as np
import pandas as pd
import faiss

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# load .env
def load_env():
    # look in working dir root (parent of pipeline/)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        # fallback: current working directory
        env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()   # always override

load_env()

# ---------------------------------------------------------------------------
WORKING_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR    = os.path.join(WORKING_DIR, "data", "rag_index")
APPROVED_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")

MODEL_NAME   = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL    = "nvidia/nemotron-3-super-120b-a12b:free"
TOP_K        = 3
TEMPERATURE  = 0.0

# ---------------------------------------------------------------------------

_embed_model = None
_faiss_index = None
_metadata    = None
_drugs_df    = None
_synonym_map = None   # synonym (lowercase) -> (drugbank_id, canonical_name)

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(MODEL_NAME)
    return _embed_model

def get_index():
    global _faiss_index, _metadata
    if _faiss_index is None:
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        meta_path  = os.path.join(INDEX_DIR, "metadata.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}\n"
                "Run: python pipeline/step6_rag_index.py"
            )
        _faiss_index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            _metadata = pickle.load(f)
    return _faiss_index, _metadata

def get_drugs_df():
    global _drugs_df
    if _drugs_df is None:
        _drugs_df = pd.read_csv(
            os.path.join(APPROVED_DIR, "drugs.csv"),
            usecols=["drugbank_id", "name"]
        )
    return _drugs_df

def get_synonym_map():
    global _synonym_map
    if _synonym_map is None:
        df   = get_drugs_df()
        name_lookup = dict(zip(df["drugbank_id"], df["name"]))
        syn_df = pd.read_csv(
            os.path.join(APPROVED_DIR, "drug_attributes.csv"),
            usecols=["drugbank_id", "attr_type", "value"]
        )
        syn_df = syn_df[syn_df["attr_type"] == "synonym"]
        _synonym_map = {}
        for _, row in syn_df.iterrows():
            key = str(row["value"]).strip().lower()
            did = row["drugbank_id"]
            if did in name_lookup:
                _synonym_map[key] = (did, name_lookup[did])
    return _synonym_map

# ---------------------------------------------------------------------------

def resolve_drug(query: str) -> tuple[str, str]:
    """
    Resolve a drug name or DrugBank ID to (drugbank_id, name).
    Accepts: 'DB00682', 'Warfarin', 'warfarin' (case-insensitive).
    """
    df = get_drugs_df()
    q  = query.strip()

    # exact DrugBank ID
    if q.upper().startswith("DB"):
        row = df[df["drugbank_id"] == q.upper()]
        if not row.empty:
            return row.iloc[0]["drugbank_id"], row.iloc[0]["name"]

    # case-insensitive name match
    row = df[df["name"].str.lower() == q.lower()]
    if not row.empty:
        return row.iloc[0]["drugbank_id"], row.iloc[0]["name"]

    # partial name match
    row = df[df["name"].str.lower().str.contains(q.lower(), na=False)]
    if not row.empty:
        if len(row) > 1:
            print(f"  [warn] Multiple matches for '{q}', using first: {row.iloc[0]['name']}")
        return row.iloc[0]["drugbank_id"], row.iloc[0]["name"]

    # synonym lookup (e.g. "Aspirin" -> "Acetylsalicylic acid")
    syn_map = get_synonym_map()
    if q.lower() in syn_map:
        did, canonical = syn_map[q.lower()]
        print(f"  [info] '{q}' resolved via synonym -> {canonical} ({did})")
        return did, canonical

    raise ValueError(f"Drug not found in approved set: '{query}'")


def retrieve(name_a: str, name_b: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query '{name_a} interaction with {name_b}' and retrieve
    top-k most similar interaction sentences from the FAISS index.
    """
    model          = get_embed_model()
    index, metadata = get_index()

    query = f"{name_a} interaction with {name_b} is:"
    vec   = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)
    return results


def call_llm(name_a: str, name_b: str, retrieved: list[dict]) -> dict:
    """
    Call OpenRouter LLM with retrieved evidence.
    Returns structured dict: {found, interaction_type, interaction_description}.
    """
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set. Add it to .env or environment.")

    evidence = "\n".join(
        f"[{i+1}] (similarity={r['score']:.3f}) {r['text']}"
        for i, r in enumerate(retrieved)
    )

    prompt = textwrap.dedent(f"""
        You are a clinical pharmacology expert. Based ONLY on the evidence below,
        determine whether there is a clinically relevant drug-drug interaction (DDI)
        between {name_a} and {name_b}.

        Evidence:
        {evidence}

        Respond with ONLY a valid JSON object. No explanation, no markdown, no code block.
        Use exactly this schema:
        {{
          "found": true or false,
          "interaction_type": "<mechanism-based classification, e.g. pharmacokinetic/pharmacodynamic, or null if not found>",
          "interaction_description": "<concise evidence-grounded explanation, or null if not found>"
        }}
    """).strip()

    payload = json.dumps({
        "model":       LLM_MODEL,
        "temperature": TEMPERATURE,
        "messages":    [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/ddi-drugbank",
            "X-Title":       "DDI Detection System",
        },
        method="POST",
    )

    try:
        resp_obj = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter API error {e.code}: {err_body}") from e
    with resp_obj as resp:
        body = json.loads(resp.read().decode("utf-8"))

    content = body["choices"][0]["message"]["content"].strip()

    # strip markdown code block if LLM wraps it
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    return json.loads(content)


def detect(drug_a: str, drug_b: str, top_k: int = TOP_K, verbose: bool = True) -> dict:
    """
    Full pipeline for a single drug pair.
    drug_a / drug_b: DrugBank ID (DB#####) or drug name string.
    Returns: {drugbank_id_a, drugbank_id_b, name_a, name_b, found,
               interaction_type, interaction_description, retrieved}
    """
    id_a, name_a = resolve_drug(drug_a)
    id_b, name_b = resolve_drug(drug_b)

    if verbose:
        print(f"\n  Drug A : {name_a} ({id_a})")
        print(f"  Drug B : {name_b} ({id_b})")
        print(f"  Query  : '{name_a} interaction with {name_b} is:'")

    retrieved = retrieve(name_a, name_b, top_k)

    if verbose:
        print(f"\n  Top-{top_k} retrieved:")
        for i, r in enumerate(retrieved):
            print(f"    [{i+1}] score={r['score']:.3f}  {r['text'][:100]}...")

    result = call_llm(name_a, name_b, retrieved)

    output = {
        "drugbank_id_a":          id_a,
        "drugbank_id_b":          id_b,
        "name_a":                 name_a,
        "name_b":                 name_b,
        "found":                  result.get("found", False),
        "interaction_type":       result.get("interaction_type"),
        "interaction_description": result.get("interaction_description"),
        "retrieved":              retrieved,
    }

    if verbose:
        print(f"\n  Result:")
        print(f"    found               : {output['found']}")
        print(f"    interaction_type    : {output['interaction_type']}")
        print(f"    interaction_description:")
        desc = output["interaction_description"] or "(none)"
        for line in textwrap.wrap(desc, width=70):
            print(f"      {line}")

    return output


def detect_multiple(drug_list: list[str], top_k: int = TOP_K) -> list[dict]:
    """
    Generate all unique pairs from a list of drugs and run detect() on each.
    Matches the paper's endpoint behaviour (/drug with a list of drug names).
    """
    from itertools import combinations
    pairs   = list(combinations(drug_list, 2))
    results = []
    print(f"\n  Evaluating {len(pairs)} unique pair(s) from {len(drug_list)} drugs ...")
    for a, b in pairs:
        print(f"\n{'='*68}")
        results.append(detect(a, b, top_k=top_k))
    return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-a", type=str, help="Drug A (name or DB ID)")
    parser.add_argument("--drug-b", type=str, help="Drug B (name or DB ID)")
    parser.add_argument("--drugs",  type=str, nargs="+",
                        help="List of drugs — all pairs evaluated (paper mode)")
    parser.add_argument("--top-k",  type=int, default=TOP_K)
    args = parser.parse_args()

    print("=" * 68)
    print("  DDI RAG QUERY PIPELINE")
    print(f"  Embed model : {MODEL_NAME}")
    print(f"  LLM         : {LLM_MODEL}")
    print(f"  Top-k       : {args.top_k}")
    print("=" * 68)

    print("\n[1/3] Loading embed model ...")
    get_embed_model()
    print("[2/3] Loading FAISS index ...")
    get_index()
    print("[3/3] Ready.\n")

    if args.drugs:
        results = detect_multiple(args.drugs, top_k=args.top_k)
        print(f"\n{'='*68}")
        print(f"  SUMMARY: {sum(r['found'] for r in results)}/{len(results)} interactions found")

    elif args.drug_a and args.drug_b:
        detect(args.drug_a, args.drug_b, top_k=args.top_k)

    else:
        # interactive mode
        print("Interactive mode. Type drug names or DrugBank IDs.")
        print("Enter a single drug name per line, blank line to run, 'q' to quit.\n")
        while True:
            drugs = []
            while True:
                line = input("  Drug> ").strip()
                if line.lower() == "q":
                    sys.exit(0)
                if not line:
                    break
                drugs.append(line)

            if len(drugs) == 0:
                continue
            elif len(drugs) == 1:
                print("  Need at least 2 drugs.")
                continue
            elif len(drugs) == 2:
                detect(drugs[0], drugs[1], top_k=args.top_k)
            else:
                detect_multiple(drugs, top_k=args.top_k)
