"""
step9_baseline.py
-----------------
Non-AI baselines for DDI detection (TM2A requirement).

Baseline 1 — Exact lookup:
    Check if (drug_a, drug_b) exists in drug_interactions_dedup.csv.
    Pure database lookup — no ML, no similarity, no embeddings.

Baseline 2 — TF-IDF retrieval + threshold:
    Represent each DDI description as a TF-IDF vector.
    Query = "drug_a interaction with drug_b".
    If max cosine similarity > threshold → found.
    No neural networks, no LLM.

Both are evaluated on the SAME 500-pair test set as the RAG pipeline (seed=42)
so results are directly comparable.

Usage:
    python pipeline/step9_baseline.py                  # evaluate both baselines
    python pipeline/step9_baseline.py --results-only   # print saved comparison table
"""

import os, sys, json, time, random, argparse, csv
import pandas as pd
import numpy as np

WORKING_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROVED_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")
OUTPUT_DIR   = os.path.join(WORKING_DIR, "data", "evaluation")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "baseline_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "baseline_summary.json")
RAG_SUMMARY  = os.path.join(OUTPUT_DIR, "rag_eval_summary.json")

# TF-IDF threshold: cosine similarity above this → predicted found
TFIDF_THRESHOLD = 0.30


def sep(label=""):
    w = 68
    if label:
        p = (w - len(label) - 2) // 2
        print("-" * p + " " + label + " " + "-" * (w - p - len(label) - 2))
    else:
        print("-" * w)


def compute_metrics(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)
    tn = sum(1 for l, p in zip(labels, preds) if not l and not p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = (tp + tn) / len(labels) if labels else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4)}


def print_metrics(name, m):
    print(f"\n  {name}")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"    Precision : {m['precision']:.4f}")
    print(f"    Recall    : {m['recall']:.4f}")
    print(f"    F1-score  : {m['f1']:.4f}")
    print(f"    Accuracy  : {m['accuracy']:.4f}")


def build_test_set(n_pairs, pos_ratio, seed, n_drugs):
    """Identical to step8 — same seed produces same test set."""
    random.seed(seed)
    np.random.seed(seed)

    drugs = pd.read_csv(os.path.join(APPROVED_DIR, "drugs.csv"),
                        usecols=["drugbank_id", "name"])
    ddi   = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"),
                        usecols=["drugbank_id_a", "drugbank_id_b"])

    sampled_drugs = drugs.sample(n=min(n_drugs, len(drugs)), random_state=seed)
    sampled_ids   = set(sampled_drugs["drugbank_id"])
    name_map      = dict(zip(sampled_drugs["drugbank_id"], sampled_drugs["name"]))

    pos_mask  = (ddi["drugbank_id_a"].isin(sampled_ids) &
                 ddi["drugbank_id_b"].isin(sampled_ids))
    pos_pairs = ddi[pos_mask].copy()

    n_pos = min(int(n_pairs * pos_ratio), len(pos_pairs))
    n_neg = n_pairs - n_pos

    pos_sample = pos_pairs.sample(n=n_pos, random_state=seed)
    pos_set    = set(zip(pos_pairs["drugbank_id_a"], pos_pairs["drugbank_id_b"]))

    drug_list = list(sampled_ids)
    negatives, attempts = [], 0
    while len(negatives) < n_neg and attempts < n_neg * 100:
        a, b = random.sample(drug_list, 2)
        if a > b: a, b = b, a
        if (a, b) not in pos_set:
            negatives.append((a, b))
            pos_set.add((a, b))
        attempts += 1

    rows = []
    for _, r in pos_sample.iterrows():
        rows.append({"drugbank_id_a": r["drugbank_id_a"],
                     "name_a": name_map.get(r["drugbank_id_a"], r["drugbank_id_a"]),
                     "drugbank_id_b": r["drugbank_id_b"],
                     "name_b": name_map.get(r["drugbank_id_b"], r["drugbank_id_b"]),
                     "label": True})
    for a, b in negatives:
        rows.append({"drugbank_id_a": a, "name_a": name_map.get(a, a),
                     "drugbank_id_b": b, "name_b": name_map.get(b, b),
                     "label": False})

    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline 1: Exact lookup
# ---------------------------------------------------------------------------

def build_lookup_set():
    """Load all DDI pairs into a set of frozensets for O(1) lookup."""
    ddi = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"),
                      usecols=["drugbank_id_a", "drugbank_id_b"])
    return set(zip(ddi["drugbank_id_a"], ddi["drugbank_id_b"]))


def predict_exact(pair_key, lookup_set):
    """Return True if pair is in the DDI lookup table."""
    a, b = pair_key
    return (a, b) in lookup_set or (b, a) in lookup_set


# ---------------------------------------------------------------------------
# Baseline 2: TF-IDF retrieval + threshold
# ---------------------------------------------------------------------------

def build_tfidf_index(ddi_df):
    """
    Build a TF-IDF matrix over all DDI description texts.
    Returns (vectorizer, tfidf_matrix, metadata_list).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    sep("Building TF-IDF index")
    texts = []
    meta  = []
    for _, row in ddi_df.iterrows():
        text = f"{row['name_a']} interaction with {row['name_b']} is: {row['description']}"
        texts.append(text)
        meta.append((row["drugbank_id_a"], row["drugbank_id_b"],
                     row["name_a"], row["name_b"]))

    print(f"  Fitting TF-IDF on {len(texts):,} DDI descriptions ...", flush=True)
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    print(f"  Matrix shape: {matrix.shape}", flush=True)
    return vectorizer, matrix, meta


def predict_tfidf(name_a, name_b, vectorizer, matrix, threshold=TFIDF_THRESHOLD):
    """
    Query TF-IDF index for the drug pair. Returns (found, best_score).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query = f"{name_a} interaction with {name_b} is:"
    q_vec = vectorizer.transform([query])
    sims  = cosine_similarity(q_vec, matrix).flatten()
    best  = float(sims.max())
    return best >= threshold, round(best, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs",   type=int,   default=500)
    parser.add_argument("--pos-ratio", type=float, default=0.5)
    parser.add_argument("--n-drugs",   type=int,   default=1399)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--threshold", type=float, default=TFIDF_THRESHOLD,
                        help="TF-IDF cosine similarity threshold (default 0.30)")
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── results-only mode ──────────────────────────────────────────────────
    if args.results_only:
        if not os.path.exists(SUMMARY_FILE):
            print("No baseline summary found. Run without --results-only first.")
            sys.exit(1)
        with open(SUMMARY_FILE) as f:
            summary = json.load(f)

        sep("BASELINE COMPARISON")
        for name, m in summary["methods"].items():
            print_metrics(name, m)

        # also print RAG if available
        if os.path.exists(RAG_SUMMARY):
            with open(RAG_SUMMARY) as f:
                rag = json.load(f)
            print_metrics("RAG (PubMedBERT + LLM)", rag)

        sep()
        print("\n  Comparison table:")
        print(f"  {'Method':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
        for name, m in summary["methods"].items():
            print(f"  {name:<35} {m['precision']:>6.4f} {m['recall']:>6.4f} "
                  f"{m['f1']:>6.4f} {m['accuracy']:>6.4f}")
        if os.path.exists(RAG_SUMMARY):
            with open(RAG_SUMMARY) as f:
                rag = json.load(f)
            print(f"  {'RAG (PubMedBERT + LLM)':<35} {rag['precision']:>6.4f} "
                  f"{rag['recall']:>6.4f} {rag['f1']:>6.4f} {rag['accuracy']:>6.4f}")
        sep()
        sys.exit(0)

    # ── build test set (same seed as step8) ───────────────────────────────
    sep("STEP 9 - BASELINE EVALUATION")
    print(f"  Pairs     : {args.n_pairs}  |  seed={args.seed}  |  "
          f"TF-IDF threshold={args.threshold}")

    sep("BUILDING TEST SET")
    test_df = build_test_set(args.n_pairs, args.pos_ratio, args.seed, args.n_drugs)
    print(f"  Test set  : {len(test_df)} pairs  "
          f"({test_df['label'].sum()} pos, {(~test_df['label']).sum()} neg)")

    labels = list(test_df["label"])

    # ── Baseline 1: Exact lookup ───────────────────────────────────────────
    sep("Baseline 1 — Exact Lookup")
    print("  Loading DDI lookup table ...", flush=True)
    lookup_set = build_lookup_set()
    print(f"  Lookup set: {len(lookup_set):,} pairs", flush=True)

    t0 = time.time()
    exact_preds = [predict_exact((r["drugbank_id_a"], r["drugbank_id_b"]), lookup_set)
                   for _, r in test_df.iterrows()]
    exact_metrics = compute_metrics(labels, exact_preds)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_metrics("Exact Lookup", exact_metrics)

    # ── Baseline 2: TF-IDF ────────────────────────────────────────────────
    sep("Baseline 2 — TF-IDF + Threshold")
    ddi_full = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"))

    # need name columns — join from drugs.csv
    drugs_df = pd.read_csv(os.path.join(APPROVED_DIR, "drugs.csv"),
                           usecols=["drugbank_id", "name"])
    name_map = dict(zip(drugs_df["drugbank_id"], drugs_df["name"]))
    ddi_full["name_a"] = ddi_full["drugbank_id_a"].map(name_map)
    ddi_full["name_b"] = ddi_full["drugbank_id_b"].map(name_map)
    ddi_full = ddi_full.dropna(subset=["name_a", "name_b", "description"])

    vectorizer, matrix, _ = build_tfidf_index(ddi_full)

    print(f"\n  Scoring {len(test_df)} pairs (threshold={args.threshold}) ...",
          flush=True)
    t0 = time.time()
    tfidf_preds, tfidf_scores = [], []
    for i, (_, row) in enumerate(test_df.iterrows()):
        found, score = predict_tfidf(row["name_a"], row["name_b"],
                                     vectorizer, matrix, args.threshold)
        tfidf_preds.append(found)
        tfidf_scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(test_df)}]", flush=True)

    tfidf_metrics = compute_metrics(labels, tfidf_preds)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_metrics(f"TF-IDF (threshold={args.threshold})", tfidf_metrics)

    # ── Save per-pair results ──────────────────────────────────────────────
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["drugbank_id_a", "name_a", "drugbank_id_b", "name_b",
                         "label", "exact_pred", "tfidf_pred", "tfidf_score"])
        for i, (_, row) in enumerate(test_df.iterrows()):
            writer.writerow([row["drugbank_id_a"], row["name_a"],
                             row["drugbank_id_b"], row["name_b"],
                             row["label"], exact_preds[i],
                             tfidf_preds[i], tfidf_scores[i]])

    # ── Summary + comparison table ─────────────────────────────────────────
    summary = {
        "n_pairs": args.n_pairs, "seed": args.seed, "threshold": args.threshold,
        "methods": {
            "Exact Lookup (no ML)":            exact_metrics,
            f"TF-IDF + threshold={args.threshold}": tfidf_metrics,
        }
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    sep("COMPARISON TABLE")
    print(f"\n  {'Method':<38} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}")
    print(f"  {'-'*38} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for name, m in summary["methods"].items():
        print(f"  {name:<38} {m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['f1']:>6.4f} {m['accuracy']:>6.4f}")
    if os.path.exists(RAG_SUMMARY):
        with open(RAG_SUMMARY) as f:
            rag = json.load(f)
        print(f"  {'RAG (PubMedBERT + LLM)':<38} {rag['precision']:>6.4f} "
              f"{rag['recall']:>6.4f} {rag['f1']:>6.4f} {rag['accuracy']:>6.4f}")
    sep()
    print(f"  Results saved to : {RESULTS_FILE}")
    print(f"  Summary saved to : {SUMMARY_FILE}")
    sep()
