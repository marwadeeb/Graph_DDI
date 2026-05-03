"""
evaluate_rag.py
---------------------
Evaluate the RAG interaction detection pipeline on a sampled
ground-truth dataset drawn from DrugBank.

Methodology:
  1. Sample a drug universe of N drugs from the approved set
  2. Collect all positive pairs within that universe (from drug_interactions_dedup.csv)
  3. Sample an equal number of negative pairs (pairs NOT in the DDI list)
  4. Run the RAG pipeline on every pair -> found: true/false
  5. Compute per-pair and aggregate precision, recall, F1, accuracy

Results are saved incrementally -- if interrupted, re-run with --resume
to skip already-evaluated pairs.

Usage:
    python pipeline/evaluate_rag.py                     # 500 pairs default
    python pipeline/evaluate_rag.py --n-pairs 2000
    python pipeline/evaluate_rag.py --n-pairs 500 --seed 99
    python pipeline/evaluate_rag.py --resume            # continue interrupted run
    python pipeline/evaluate_rag.py --results-only      # print metrics from saved results
"""

import os, sys, json, time, random, argparse, csv
import pandas as pd
import numpy as np

# load .env
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")
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
WORKING_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROVED_DIR = os.path.join(WORKING_DIR, "data", "step3_approved")
OUTPUT_DIR   = os.path.join(WORKING_DIR, "data", "evaluation")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "rag_eval_results.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "rag_eval_summary.json")

# ---------------------------------------------------------------------------

def sep(label=""):
    w = 68
    if label:
        p = (w - len(label) - 2) // 2
        print("-" * p + " " + label + " " + "-" * (w - p - len(label) - 2))
    else:
        print("-" * w)


def compute_metrics(df):
    """Compute precision, recall, F1, accuracy from results dataframe."""
    tp = ((df["predicted"] == True) & (df["label"] == True)).sum()
    fp = ((df["predicted"] == True) & (df["label"] == False)).sum()
    fn = ((df["predicted"] == False) & (df["label"] == True)).sum()
    tn = ((df["predicted"] == False) & (df["label"] == False)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(df) if len(df) > 0 else 0.0

    return {
        "total":     int(len(df)),
        "tp":        int(tp),
        "fp":        int(fp),
        "fn":        int(fn),
        "tn":        int(tn),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
        "accuracy":  round(float(accuracy), 4),
    }


def print_metrics(m, label="RESULTS"):
    sep(label)
    print(f"  Total pairs evaluated : {m['total']}")
    print(f"  TP / FP / FN / TN     : {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}")
    print(f"  Precision             : {m['precision']:.4f}")
    print(f"  Recall                : {m['recall']:.4f}")
    print(f"  F1-score              : {m['f1']:.4f}")
    print(f"  Accuracy              : {m['accuracy']:.4f}")
    sep()


def build_test_set(n_pairs, pos_ratio, seed, n_drugs):
    """
    Sample drug universe, find positive pairs, sample negatives.
    Returns DataFrame with columns: drugbank_id_a, name_a, drugbank_id_b, name_b, label
    """
    sep("BUILDING TEST SET")
    random.seed(seed)
    np.random.seed(seed)

    drugs = pd.read_csv(os.path.join(APPROVED_DIR, "drugs.csv"),
                        usecols=["drugbank_id", "name"])
    ddi   = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"),
                        usecols=["drugbank_id_a", "drugbank_id_b"])

    # sample drug universe
    sampled_drugs = drugs.sample(n=min(n_drugs, len(drugs)), random_state=seed)
    sampled_ids   = set(sampled_drugs["drugbank_id"])
    name_map      = dict(zip(sampled_drugs["drugbank_id"], sampled_drugs["name"]))
    print(f"  Drug universe        : {len(sampled_ids):,} drugs")

    # find all positive pairs within the universe
    pos_mask = (ddi["drugbank_id_a"].isin(sampled_ids) &
                ddi["drugbank_id_b"].isin(sampled_ids))
    pos_pairs = ddi[pos_mask].copy()
    print(f"  Positive pairs found : {len(pos_pairs):,}")

    # how many positives and negatives to sample
    n_pos = min(int(n_pairs * pos_ratio), len(pos_pairs))
    n_neg = n_pairs - n_pos
    print(f"  Sampling             : {n_pos} positives + {n_neg} negatives = {n_pairs} total")

    pos_sample = pos_pairs.sample(n=n_pos, random_state=seed)
    pos_set    = set(zip(pos_pairs["drugbank_id_a"], pos_pairs["drugbank_id_b"]))

    # sample negatives: random pairs not in pos_set
    drug_list = list(sampled_ids)
    negatives = []
    attempts  = 0
    while len(negatives) < n_neg and attempts < n_neg * 100:
        a, b = random.sample(drug_list, 2)
        if a > b:
            a, b = b, a
        if (a, b) not in pos_set:
            negatives.append((a, b))
            pos_set.add((a, b))   # avoid duplicates
        attempts += 1

    if len(negatives) < n_neg:
        print(f"  [warn] Only found {len(negatives)} negatives (graph too dense?)")

    # assemble
    rows = []
    for _, r in pos_sample.iterrows():
        rows.append({
            "drugbank_id_a": r["drugbank_id_a"],
            "name_a":        name_map.get(r["drugbank_id_a"], r["drugbank_id_a"]),
            "drugbank_id_b": r["drugbank_id_b"],
            "name_b":        name_map.get(r["drugbank_id_b"], r["drugbank_id_b"]),
            "label":         True,
        })
    for a, b in negatives:
        rows.append({
            "drugbank_id_a": a,
            "name_a":        name_map.get(a, a),
            "drugbank_id_b": b,
            "name_b":        name_map.get(b, b),
            "label":         False,
        })

    test_df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Final test set       : {len(test_df)} pairs  "
          f"({test_df['label'].sum()} pos, {(~test_df['label']).sum()} neg)")
    return test_df


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs",    type=int,   default=500,
                        help="Total pairs to evaluate (default 500)")
    parser.add_argument("--pos-ratio",  type=float, default=0.5,
                        help="Fraction of positive pairs (default 0.5)")
    parser.add_argument("--n-drugs",    type=int,   default=1399,
                        help="Drug universe size (default 1399, matches paper)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--delay",      type=float, default=1.0,
                        help="Seconds to wait between API calls (default 1.0)")
    parser.add_argument("--top-k",      type=int,   default=3)
    parser.add_argument("--resume",     action="store_true",
                        help="Skip already-evaluated pairs from previous run")
    parser.add_argument("--results-only", action="store_true",
                        help="Print metrics from saved results without re-running")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # results-only mode
    if args.results_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"No results file found at {RESULTS_FILE}")
            sys.exit(1)
        rows_all, rows_clean = [], []
        with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                entry = {
                    "label":     r["label"].strip().lower() == "true",
                    "predicted": r["predicted"].strip().lower() == "true",
                }
                rows_all.append(entry)
                if not r.get("error", "").strip():
                    rows_clean.append(entry)
        df_all   = pd.DataFrame(rows_all)
        df_clean = pd.DataFrame(rows_clean)
        n_errors = len(rows_all) - len(rows_clean)
        print_metrics(compute_metrics(df_all),   "SAVED RESULTS (all, errors=False)")
        if n_errors:
            print(f"  [{n_errors} error rows counted as predicted=False above]")
            print()
            print_metrics(compute_metrics(df_clean), "CLEAN RESULTS (errors excluded)")
        sys.exit(0)

    sep("STEP 8 - RAG EVALUATION")
    print(f"  Pairs to evaluate : {args.n_pairs}")
    print(f"  Positive ratio    : {args.pos_ratio:.0%}")
    print(f"  Drug universe     : {args.n_drugs}")
    print(f"  Seed              : {args.seed}")
    print(f"  Top-k retrieval   : {args.top_k}")
    print(f"  API delay         : {args.delay}s")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_query import get_embed_model, get_index, get_synonym_map, call_llm, retrieve

    print("\n[1/3] Loading embed model ...")
    get_embed_model()
    print("[2/3] Loading FAISS index ...")
    get_index()
    print("[3/3] Ready.\n")

    # build test set
    test_df = build_test_set(args.n_pairs, args.pos_ratio, args.seed, args.n_drugs)

    # load existing results if resuming
    evaluated  = set()   # pairs successfully evaluated (no error) — skip these
    error_keys = set()   # pairs that errored — will be re-evaluated and overwritten
    if args.resume and os.path.exists(RESULTS_FILE):
        saved_rows = []
        with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["drugbank_id_a"], row["drugbank_id_b"])
                if row.get("error", "").strip():
                    error_keys.add(key)   # will be re-evaluated
                else:
                    evaluated.add(key)    # clean result — keep as-is
                saved_rows.append(row)
        # Rewrite CSV without the error rows (they'll be re-appended after re-eval)
        clean_rows = [r for r in saved_rows
                      if (r["drugbank_id_a"], r["drugbank_id_b"]) not in error_keys]
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["drugbank_id_a", "name_a", "drugbank_id_b", "name_b",
                             "label", "predicted", "interaction_type", "error"])
            for r in clean_rows:
                writer.writerow([r["drugbank_id_a"], r["name_a"],
                                 r["drugbank_id_b"], r["name_b"],
                                 r["label"], r["predicted"],
                                 r.get("interaction_type", ""), r.get("error", "")])
        n_redo = len(error_keys)
        n_new  = len(test_df) - len(evaluated) - n_redo
        print(f"  Resuming: {len(evaluated)} clean pairs kept, "
              f"{n_redo} errors will be re-evaluated, "
              f"~{max(n_new,0)} new pairs remaining\n")
    else:
        # start fresh with proper CSV header
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["drugbank_id_a", "name_a", "drugbank_id_b", "name_b",
                             "label", "predicted", "interaction_type", "error"])

    # evaluate
    sep("EVALUATING")
    t0       = time.time()
    n_done   = len(evaluated)
    n_total  = len(test_df)
    errors   = 0

    for i, row in test_df.iterrows():
        pair_key = (row["drugbank_id_a"], row["drugbank_id_b"])
        if pair_key in evaluated:
            continue

        # progress
        n_done += 1
        elapsed = time.time() - t0
        rate    = n_done / elapsed if elapsed > 0 else 0
        eta     = (n_total - n_done) / rate / 60 if rate > 0 else 0
        print(f"  [{n_done:>4}/{n_total}]  "
              f"{row['name_a'][:20]:<20} x {row['name_b'][:20]:<20}  "
              f"label={'Y' if row['label'] else 'N'}  "
              f"ETA~{eta:.0f}m", end="  ")

        predicted      = None
        interaction_type = ""
        error_msg      = ""

        try:
            retrieved = retrieve(row["name_a"], row["name_b"], top_k=args.top_k)
            result    = call_llm(row["name_a"], row["name_b"], retrieved)
            predicted = bool(result.get("found", False))
            interaction_type = result.get("interaction_type") or ""
            status    = "found" if predicted else "not found"
            correct   = (predicted == row["label"])
            print(f"-> {status:<10}  {'CORRECT' if correct else 'WRONG'}")
        except Exception as e:
            error_msg = str(e)[:120]
            predicted = False   # conservative: assume no interaction on error
            errors   += 1
            print(f"-> ERROR: {error_msg[:50]}")

        # save result immediately with proper CSV quoting
        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                row["drugbank_id_a"], row["name_a"],
                row["drugbank_id_b"], row["name_b"],
                row["label"], predicted,
                str(interaction_type).replace("\n", " "),
                error_msg.replace("\n", " "),
            ])

        evaluated.add(pair_key)

        # rate limit delay
        if args.delay > 0:
            time.sleep(args.delay)

    # final metrics
    sep("DONE")
    total_time = time.time() - t0
    print(f"  Evaluated {n_done} pairs in {total_time/60:.1f} min  ({errors} errors)")

    results_df = pd.read_csv(RESULTS_FILE)
    results_df["label"]     = results_df["label"].astype(bool)
    results_df["predicted"] = results_df["predicted"].astype(bool)

    m = compute_metrics(results_df)
    print_metrics(m)

    # save summary
    summary = {**m, "n_pairs": args.n_pairs, "n_drugs": args.n_drugs,
               "seed": args.seed, "top_k": args.top_k,
               "errors": errors, "total_time_min": round(total_time/60, 1)}
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to : {RESULTS_FILE}")
    print(f"  Summary saved to : {SUMMARY_FILE}")
    sep()
