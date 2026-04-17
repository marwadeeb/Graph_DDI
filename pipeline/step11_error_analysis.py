"""
step11_error_analysis.py
------------------------
TM6: Error analysis for the RAG DDI pipeline.

Loads rag_eval_results.csv and categorises every wrong prediction:
  - False Positives (FP): predicted interaction, no actual interaction
  - False Negatives (FN): missed actual interaction

For each error case, retrieves the top-3 FAISS evidence items and diagnoses
why the model failed using one of four error categories:

  WRONG_PAIR    -- evidence is for a different drug pair (retrieval failure)
  LOW_EVIDENCE  -- retrieval scores are low; model had weak signal
  LLM_HALLUC   -- retrieval scores are high but LLM still predicted wrong
                   (LLM hallucination / over-confidence)
  AMBIGUOUS     -- borderline case, scores near threshold

Outputs:
  data/evaluation/error_analysis.csv   -- per-case breakdown
  data/evaluation/error_analysis.json  -- summary statistics

Usage:
    python pipeline/step11_error_analysis.py
    python pipeline/step11_error_analysis.py --verbose   # print each case
"""

import os, sys, json, argparse
import pandas as pd
import numpy as np

WORKING_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROVED_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")
EVAL_DIR     = os.path.join(WORKING_DIR, "data", "evaluation")
RAG_RESULTS  = os.path.join(EVAL_DIR, "rag_eval_results.csv")
OUT_CSV      = os.path.join(EVAL_DIR, "error_analysis.csv")
OUT_JSON     = os.path.join(EVAL_DIR, "error_analysis.json")

# Thresholds for diagnosis
HIGH_SCORE   = 0.70   # retrieval score above this → evidence was relevant
LOW_SCORE    = 0.40   # retrieval score below this → weak evidence


def sep(label=""):
    w = 68
    if label:
        p = (w - len(label) - 2) // 2
        print("-" * p + " " + label + " " + "-" * (w - p - len(label) - 2))
    else:
        print("-" * w)


def build_tfidf_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    ddi   = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"))
    drugs = pd.read_csv(os.path.join(APPROVED_DIR, "drugs.csv"),
                        usecols=["drugbank_id", "name"])
    nm = dict(zip(drugs["drugbank_id"], drugs["name"]))
    ddi["name_a"] = ddi["drugbank_id_a"].map(nm)
    ddi["name_b"] = ddi["drugbank_id_b"].map(nm)
    ddi = ddi.dropna(subset=["name_a", "name_b", "description"])

    texts = [f"{r['name_a']} interaction with {r['name_b']} is: {r['description']}"
             for _, r in ddi.iterrows()]
    meta  = list(zip(ddi["drugbank_id_a"], ddi["drugbank_id_b"],
                     ddi["name_a"], ddi["name_b"]))

    print("  Building TF-IDF index ...", flush=True)
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(texts)
    print(f"  Index ready: {mat.shape[0]:,} documents", flush=True)
    return vec, mat, meta


def get_top_evidence(name_a, name_b, vec, mat, meta, top_k=3):
    from sklearn.metrics.pairwise import cosine_similarity
    q    = f"{name_a} interaction with {name_b} is:"
    qv   = vec.transform([q])
    sims = cosine_similarity(qv, mat).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        id_a, id_b, na, nb = meta[i]
        results.append({
            "score":      round(float(sims[i]), 4),
            "match_name_a": na,
            "match_name_b": nb,
            "match_id_a":   id_a,
            "match_id_b":   id_b,
        })
    return results


def diagnose(error_type, evidence, name_a, name_b):
    """Assign an error category based on retrieval evidence."""
    if not evidence:
        return "LOW_EVIDENCE", "No evidence retrieved"

    top_score = evidence[0]["score"]
    top_match_a = evidence[0]["match_name_a"].lower()
    top_match_b = evidence[0]["match_name_b"].lower()
    query_a = name_a.lower()
    query_b = name_b.lower()

    # Check if top evidence is for the correct pair
    pair_match = (
        (query_a in top_match_a or top_match_a in query_a) and
        (query_b in top_match_b or top_match_b in query_b)
    ) or (
        (query_b in top_match_a or top_match_a in query_b) and
        (query_a in top_match_b or top_match_b in query_a)
    )

    if top_score < LOW_SCORE:
        return "LOW_EVIDENCE", f"Max retrieval score {top_score:.3f} < {LOW_SCORE} — weak signal"

    if not pair_match:
        return "WRONG_PAIR", (
            f"Top evidence is for '{evidence[0]['match_name_a']} × "
            f"{evidence[0]['match_name_b']}' (score={top_score:.3f}), "
            f"not the queried pair"
        )

    if top_score >= HIGH_SCORE:
        if error_type == "FP":
            return "LLM_HALLUC", (
                f"Strong evidence (score={top_score:.3f}) for a related pair but "
                f"LLM over-generalised — predicted interaction where none exists"
            )
        else:  # FN
            return "LLM_HALLUC", (
                f"Strong evidence present (score={top_score:.3f}) but "
                f"LLM failed to confirm known interaction"
            )

    return "AMBIGUOUS", (
        f"Score {top_score:.3f} near threshold — borderline retrieval, "
        f"model prediction unreliable"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print full details for each error case")
    args = parser.parse_args()

    if not os.path.exists(RAG_RESULTS):
        print(f"ERROR: {RAG_RESULTS} not found. Run step8 first.")
        sys.exit(1)

    eval_df = pd.read_csv(RAG_RESULTS)
    clean   = eval_df[eval_df["error"].fillna("").str.strip() == ""].copy()
    clean["label_bool"] = clean["label"].astype(str).str.lower().isin(["true", "1"])
    clean["pred_bool"]  = clean["predicted"].astype(str).str.lower().isin(["true", "1"])

    errors  = clean[clean["label_bool"] != clean["pred_bool"]].copy()
    errors["error_type"] = errors.apply(
        lambda r: "FP" if (not r["label_bool"] and r["pred_bool"]) else "FN", axis=1
    )

    fp_df = errors[errors["error_type"] == "FP"]
    fn_df = errors[errors["error_type"] == "FN"]

    sep("STEP 11 — ERROR ANALYSIS")
    print(f"  Clean eval pairs : {len(clean)}")
    print(f"  Errors total     : {len(errors)}  (FP={len(fp_df)}, FN={len(fn_df)})")

    vec, mat, meta = build_tfidf_index()

    rows = []
    category_counts = {"FP": {}, "FN": {}}

    for _, row in errors.iterrows():
        etype    = row["error_type"]
        name_a   = row["name_a"]
        name_b   = row["name_b"]
        evidence = get_top_evidence(name_a, name_b, vec, mat, meta, top_k=3)
        cat, reason = diagnose(etype, evidence, name_a, name_b)

        category_counts[etype][cat] = category_counts[etype].get(cat, 0) + 1

        rows.append({
            "drugbank_id_a":  row["drugbank_id_a"],
            "name_a":         name_a,
            "drugbank_id_b":  row["drugbank_id_b"],
            "name_b":         name_b,
            "error_type":     etype,
            "error_category": cat,
            "reason":         reason,
            "top_score":      evidence[0]["score"] if evidence else 0.0,
            "top_match":      f"{evidence[0]['match_name_a']} × {evidence[0]['match_name_b']}" if evidence else "",
        })

        if args.verbose:
            sep(f"{etype} — {name_a} × {name_b}")
            print(f"  Category : {cat}")
            print(f"  Reason   : {reason}")
            for i, ev in enumerate(evidence):
                print(f"  Evidence {i+1} (score={ev['score']:.4f}): "
                      f"{ev['match_name_a']} × {ev['match_name_b']}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    sep("FALSE POSITIVES")
    print(f"  Total FP: {len(fp_df)}")
    for cat, cnt in sorted(category_counts["FP"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<16} {cnt:>4}  ({100*cnt/max(len(fp_df),1):.0f}%)")

    sep("FALSE NEGATIVES")
    print(f"  Total FN: {len(fn_df)}")
    for cat, cnt in sorted(category_counts["FN"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<16} {cnt:>4}  ({100*cnt/max(len(fn_df),1):.0f}%)")

    sep("INTERPRETATION")
    total_errors = len(errors)
    wrong_pair = sum(category_counts["FP"].get("WRONG_PAIR", 0) +
                     category_counts["FN"].get("WRONG_PAIR", 0) for _ in [1])
    low_ev     = sum(category_counts["FP"].get("LOW_EVIDENCE", 0) +
                     category_counts["FN"].get("LOW_EVIDENCE", 0) for _ in [1])
    halluc     = sum(category_counts["FP"].get("LLM_HALLUC", 0) +
                     category_counts["FN"].get("LLM_HALLUC", 0) for _ in [1])
    ambig      = sum(category_counts["FP"].get("AMBIGUOUS", 0) +
                     category_counts["FN"].get("AMBIGUOUS", 0) for _ in [1])

    wrong_pair = category_counts["FP"].get("WRONG_PAIR", 0) + category_counts["FN"].get("WRONG_PAIR", 0)
    low_ev     = category_counts["FP"].get("LOW_EVIDENCE", 0) + category_counts["FN"].get("LOW_EVIDENCE", 0)
    halluc     = category_counts["FP"].get("LLM_HALLUC", 0) + category_counts["FN"].get("LLM_HALLUC", 0)
    ambig      = category_counts["FP"].get("AMBIGUOUS", 0) + category_counts["FN"].get("AMBIGUOUS", 0)

    for label, cnt in [("WRONG_PAIR (retrieval)", wrong_pair),
                       ("LOW_EVIDENCE",           low_ev),
                       ("LLM_HALLUC",             halluc),
                       ("AMBIGUOUS",              ambig)]:
        pct = 100 * cnt / total_errors if total_errors else 0
        print(f"  {label:<28} {cnt:>4} / {total_errors}  ({pct:.0f}%)")

    summary = {
        "total_clean_pairs": len(clean),
        "total_errors": total_errors,
        "fp": len(fp_df),
        "fn": len(fn_df),
        "fp_categories": category_counts["FP"],
        "fn_categories": category_counts["FN"],
        "overall_categories": {
            "WRONG_PAIR":   wrong_pair,
            "LOW_EVIDENCE": low_ev,
            "LLM_HALLUC":  halluc,
            "AMBIGUOUS":   ambig,
        }
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    sep()
    print(f"  Per-case CSV : {OUT_CSV}")
    print(f"  Summary JSON : {OUT_JSON}")
    sep()
