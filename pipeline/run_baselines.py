"""
run_baselines.py
-----------------
Link-prediction baselines for the DDI project.

Addresses two rubric requirements:
  TM2A  — non-AI baseline       : graph heuristics (no learning, no parameters)
  TM10G — non-graph ML baseline : Logistic Regression on drug pharmacological features

TWO evaluation settings
-----------------------
  1. WARM (transductive, standard):
       Random 80/20 edge split.  All drugs appear in training.
       Tests: can the model find masked interactions among known drugs?
       This is where heuristics shine (avg degree ~344 -> lots of common neighbours).

  2. COLD-START (inductive, GNN's real use case):
       10 % of drugs are held out COMPLETELY — zero training edges.
       Tests: can the model predict interactions for a drug it has never seen
       in the interaction graph?  This mimics a newly approved drug.
       Graph heuristics score 0 for ALL cold pairs  -> AUC ~= 0.50.
       LR uses node features (physicochemical, ATC, CYP450) -> AUC ~0.80-0.90.
       GNN combines features + graph structure -> should be highest.
       This is the scientifically differentiated evaluation.

Comparison logic:
  GNN vs Adamic-Adar  (warm)  ->  does learning add over pure topology?
  GNN vs LR           (cold)  ->  does graph structure add over raw features
                                  for drugs the model has never seen?

Outputs
-------
  data/evaluation/edge_split.npz         — warm train/test split (for Laure's GNN)
  data/evaluation/cold_split.npz         — cold-start split
  data/evaluation/baselines_results.json — AUC-ROC + AP, warm + cold, per method
  data/evaluation/baselines_results.csv  — same, tabular for paper table
  Printed markdown-style table to stdout

Usage
-----
  python pipeline/run_baselines.py                 # warm + cold (default)
  python pipeline/run_baselines.py --warm-only     # skip cold-start
  python pipeline/run_baselines.py --test-size 0.2 --neg-ratio 1 --seed 42
  python pipeline/run_baselines.py --results-only
"""

import os, sys, json, argparse, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "step4_graph"
OUT   = BASE / "data" / "evaluation"

EDGE_INDEX_PATH    = GRAPH / "edge_index.csv"
NODE_FEATURES_PATH = GRAPH / "node_features.csv"       # scaled, ~212 features
NODE_MAPPING_PATH  = GRAPH / "node_mapping.csv"

SPLIT_PATH      = OUT / "edge_split.npz"
COLD_SPLIT_PATH = OUT / "cold_split.npz"
JSON_PATH    = OUT / "baselines_results.json"
CSV_PATH     = OUT / "baselines_results.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(label=""):
    w = 68
    if label:
        p = (w - len(label) - 2) // 2
        print(f"\n{'-'*p} {label} {'-'*(w - p - len(label) - 2)}")
    else:
        print("-" * w)


def check_lfs(path: Path):
    """Exit with a clear message if the file is an LFS pointer (< 500 bytes)."""
    if not path.exists():
        print(f"[step9] ERROR: {path} not found.")
        print("        Run:  git lfs pull --include='data/step4_graph/*.csv'")
        sys.exit(1)
    if path.stat().st_size < 500:
        print(f"[step9] ERROR: {path} looks like an LFS pointer ({path.stat().st_size} bytes).")
        print("        Run:  git lfs pull --include='data/step4_graph/*.csv'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graph_data():
    sep("Loading graph data")
    check_lfs(EDGE_INDEX_PATH)
    check_lfs(NODE_FEATURES_PATH)
    check_lfs(NODE_MAPPING_PATH)

    edges   = pd.read_csv(EDGE_INDEX_PATH)
    mapping = pd.read_csv(NODE_MAPPING_PATH)
    feats   = pd.read_csv(NODE_FEATURES_PATH)

    # Normalise column names (step4 writes src_idx / dst_idx)
    edges.columns = edges.columns.str.strip()
    src_col = "src_idx" if "src_idx" in edges.columns else edges.columns[0]
    dst_col = "dst_idx" if "dst_idx" in edges.columns else edges.columns[1]

    pos_edges = edges[[src_col, dst_col]].values.astype(np.int64)

    # node features: first column is node_idx, rest are features
    feat_id_col  = feats.columns[0]
    feat_cols    = [c for c in feats.columns if c != feat_id_col]
    feat_matrix  = feats[feat_cols].fillna(0).values.astype(np.float32)
    feat_node_idx = feats[feat_id_col].values.astype(np.int64)

    # sort rows by node index so feat_matrix[i] = features for node i
    order = np.argsort(feat_node_idx)
    feat_matrix = feat_matrix[order]

    n_nodes = int(pos_edges.max()) + 1

    print(f"  Edges (positive):  {len(pos_edges):,}")
    print(f"  Nodes:             {n_nodes:,}")
    print(f"  Features per node: {len(feat_cols)}")

    return pos_edges, feat_matrix, feat_cols, n_nodes


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def make_split(pos_edges, n_nodes, neg_ratio=1, test_size=0.1, mask_ratio=0.2, seed=42):
    """
    Warm train/test split matching Laure's GNN training configuration:
      mask_ratio=0.2  : 20% of edges are held out as nnPU masked positives
                        (not used to build the training adjacency — matches
                         config["maskRatio"] = 0.2 in hetero_model.ipynb)
      test_size=0.1   : 10% of REMAINING edges used for test evaluation
                        (matches config["testRatio"] = 0.1)
      This gives: 80% train adj · 10% test (of 80%) · 20% masked

    Note: Laure's notebook uses torch.randperm without a fixed seed, so
    exact edge assignment will differ.  Methodology and ratios are identical.
    """
    from sklearn.model_selection import train_test_split

    sep("Creating train/test split  (matching Laure's GNN config)")
    rng = np.random.RandomState(seed)

    # Deduplicate & ensure undirected
    pos_set = set(map(tuple, pos_edges.tolist()))
    pos_set |= set(map(tuple, pos_edges[:, ::-1].tolist()))
    unique_pos = np.array(list({(min(a,b), max(a,b)) for a,b in pos_edges}))
    rng.shuffle(unique_pos)

    # Step 1: mask 20% as nnPU positives (same as Laure's maskRatio)
    n_masked = int(len(unique_pos) * mask_ratio)
    masked_pos  = unique_pos[:n_masked]          # held out (nnPU only)
    visible_pos = unique_pos[n_masked:]          # used for train/test split

    # Step 2: split visible edges into train/test (10% test like Laure)
    tr_pos, te_pos = train_test_split(visible_pos, test_size=test_size,
                                      random_state=seed)

    # Negative sampling against the full positive set (no leakage)
    n_neg = (len(tr_pos) + len(te_pos)) * neg_ratio
    neg_list = []
    while len(neg_list) < n_neg:
        batch = n_neg * 3
        u = rng.randint(0, n_nodes, size=int(batch))
        v = rng.randint(0, n_nodes, size=int(batch))
        for a, b in zip(u, v):
            if a == b:
                continue
            key = (int(min(a, b)), int(max(a, b)))
            if key not in pos_set:
                neg_list.append(key)
                pos_set.add(key)
                if len(neg_list) >= n_neg:
                    break
    neg_edges = np.array(neg_list[:n_neg], dtype=np.int64)
    tr_neg, te_neg = train_test_split(neg_edges, test_size=test_size,
                                      random_state=seed)

    print(f"  Config: mask_ratio={mask_ratio}  test_ratio={test_size}  "
          f"(matches hetero_model.ipynb)")
    print(f"  Masked positives (nnPU): {len(masked_pos):,}")
    print(f"  Train: {len(tr_pos):,} pos  +  {len(tr_neg):,} neg")
    print(f"  Test:  {len(te_pos):,} pos  +  {len(te_neg):,} neg")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(SPLIT_PATH,
             train_pos=tr_pos, train_neg=tr_neg,
             test_pos=te_pos,  test_neg=te_neg,
             masked_pos=masked_pos)
    print(f"  Split saved -> {SPLIT_PATH}")

    return tr_pos, te_pos, tr_neg, te_neg


# ---------------------------------------------------------------------------
# Cold-start split  (drug-level hold-out)
# ---------------------------------------------------------------------------

def make_cold_split(pos_edges, n_nodes, cold_frac=0.10, neg_ratio=1, seed=42):
    """
    Drug-level cold-start split.

    Hold out `cold_frac` of drugs entirely (remove all their edges from
    training).  At test time, the model must predict interactions for drugs
    it has never seen in the graph — the true inductive / cold-start scenario.

    Returns
    -------
    tr_pos  : positive edges where NEITHER endpoint is a cold drug
    cold_pos: positive edges where AT LEAST ONE endpoint is a cold drug
    tr_neg  : negative edges sampled from warm pairs
    cold_neg: negative edges sampled from cold pairs
    cold_drugs: set of held-out node indices
    """
    rng = np.random.RandomState(seed)

    sep("Creating cold-start split  (drug-level hold-out)")

    # All unique directed pairs -> undirected canonical form
    all_nodes  = np.unique(pos_edges.ravel())
    n_cold     = max(1, int(len(all_nodes) * cold_frac))
    cold_drugs = set(rng.choice(all_nodes, n_cold, replace=False).tolist())

    pos_set_all = set(
        (int(min(a, b)), int(max(a, b))) for a, b in pos_edges
    )

    cold_mask = np.array(
        [int(u) in cold_drugs or int(v) in cold_drugs for u, v in pos_edges]
    )
    cold_pos_arr = np.array(
        [(int(min(a,b)), int(max(a,b))) for a,b in pos_edges[cold_mask]],
        dtype=np.int64
    )
    warm_pos_arr = np.array(
        [(int(min(a,b)), int(max(a,b))) for a,b in pos_edges[~cold_mask]],
        dtype=np.int64
    )
    # deduplicate within each split
    cold_pos_arr = np.unique(cold_pos_arr, axis=0)
    warm_pos_arr = np.unique(warm_pos_arr, axis=0)

    print(f"  Cold drugs       : {n_cold:,} / {len(all_nodes):,}  "
          f"({cold_frac*100:.0f}% of drug set)")
    print(f"  Cold test pairs  : {len(cold_pos_arr):,}  (all edges touching cold drugs)")
    print(f"  Warm train pairs : {len(warm_pos_arr):,}  (edges among warm drugs only)")

    # Negative sampling — warm negatives (neither endpoint is cold)
    warm_nodes = [n for n in all_nodes if n not in cold_drugs]
    neg_warm   = []
    used       = set(pos_set_all)
    while len(neg_warm) < len(warm_pos_arr) * neg_ratio:
        batch = len(warm_pos_arr) * neg_ratio * 3
        u_arr = rng.choice(warm_nodes, size=int(batch))
        v_arr = rng.choice(warm_nodes, size=int(batch))
        for a, b in zip(u_arr, v_arr):
            if a == b:
                continue
            key = (int(min(a, b)), int(max(a, b)))
            if key not in used:
                neg_warm.append(key)
                used.add(key)
                if len(neg_warm) >= len(warm_pos_arr) * neg_ratio:
                    break
    tr_neg = np.array(neg_warm, dtype=np.int64)

    # Negative sampling — cold negatives (at least one endpoint is cold)
    cold_list  = list(cold_drugs)
    neg_cold   = []
    while len(neg_cold) < len(cold_pos_arr) * neg_ratio:
        for _ in range(len(cold_pos_arr) * neg_ratio * 5):
            a = rng.choice(cold_list)
            b = int(rng.choice(all_nodes))
            if a == b:
                continue
            key = (int(min(a, b)), int(max(a, b)))
            if key not in used:
                neg_cold.append(key)
                used.add(key)
                if len(neg_cold) >= len(cold_pos_arr) * neg_ratio:
                    break
    cold_neg = np.array(neg_cold[:len(cold_pos_arr) * neg_ratio], dtype=np.int64)

    print(f"  Warm negatives   : {len(tr_neg):,}")
    print(f"  Cold negatives   : {len(cold_neg):,}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(COLD_SPLIT_PATH,
             train_pos=warm_pos_arr, train_neg=tr_neg,
             cold_pos=cold_pos_arr,  cold_neg=cold_neg,
             cold_drugs=np.array(list(cold_drugs)))
    print(f"  Cold split saved -> {COLD_SPLIT_PATH}")

    return warm_pos_arr, cold_pos_arr, tr_neg, cold_neg, cold_drugs


# ---------------------------------------------------------------------------
# Baseline 1: Graph heuristics  (TM2A — non-AI)
# ---------------------------------------------------------------------------

def run_graph_heuristics(tr_pos, te_pos, te_neg, n_nodes):
    """
    Compute graph heuristics via sparse matrix multiplication — O(N²) in
    node count, not in test-pair count.  Handles test-only nodes as isolated
    (score = 0 by definition, no error).

    CN[u,v]  = (A²)[u,v]                          common neighbours
    AA[u,v]  = (A · diag(1/log deg) · A)[u,v]     Adamic-Adar
    JC[u,v]  = CN[u,v] / (deg_u + deg_v − CN[u,v]) Jaccard coefficient
    """
    import scipy.sparse as sp
    from sklearn.metrics import roc_auc_score, average_precision_score

    sep("Graph heuristics  [non-AI baseline — TM2A]")
    t0 = time.time()

    # Undirected adjacency from TRAIN edges only (no test leakage)
    r = np.concatenate([tr_pos[:, 0], tr_pos[:, 1]])
    c = np.concatenate([tr_pos[:, 1], tr_pos[:, 0]])
    A = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                      shape=(n_nodes, n_nodes))
    A = (A > 0).astype(np.float32)          # binary, no duplicate weights
    n_train_edges = int(A.nnz / 2)
    n_train_nodes = int((np.diff(A.indptr) > 0).sum())
    print(f"  Train adjacency: {n_train_nodes:,} nodes, {n_train_edges:,} edges")

    deg = np.asarray(A.sum(axis=1)).flatten()   # shape (n_nodes,)

    # ── Dense matmul via BLAS (faster + avoids bloated sparse intermediate) ─
    # With avg degree ~344 / 4790 nodes, A is ~7% dense.  After A @ A the
    # result is nearly fully dense (most node pairs share common neighbours),
    # so scipy sparse × sparse builds a huge CSR matrix.  Converting A to
    # float32 dense (92 MB) and using numpy BLAS SGEMM is 5-10× faster.
    print("  Converting adjacency to dense (BLAS matmul)...", flush=True)
    A_d = A.toarray()    # (n_nodes, n_nodes) float32 — 92 MB

    # ── Common Neighbours: A² ──────────────────────────────────────────────
    print("  Computing A² (common neighbours)...", flush=True)
    CN_d = A_d @ A_d     # BLAS SGEMM — fast; CN_d[u,v] = |N(u) ∩ N(v)|

    # ── Adamic-Adar: (A * w) · A  where w = 1/log(deg) per column ─────────
    print("  Computing Adamic-Adar...", flush=True)
    safe_deg  = np.where(deg > 1, deg, np.e)          # avoid log(0), log(1)=0
    aa_weight = (1.0 / np.log(safe_deg)).astype(np.float32)
    AA_d = (A_d * aa_weight[np.newaxis, :]) @ A_d     # row-scaled matmul

    # ── Score lookup for test pairs ────────────────────────────────────────
    test_pairs = np.vstack([te_pos, te_neg])
    y_true     = np.array([1]*len(te_pos) + [0]*len(te_neg), dtype=int)
    u, v       = test_pairs[:, 0], test_pairs[:, 1]

    print("  Scoring test pairs...", flush=True)
    cn_scores = CN_d[u, v].astype(float)
    aa_scores = AA_d[u, v].astype(float)

    denom     = deg[u] + deg[v] - cn_scores
    jc_scores = np.where(denom > 0, cn_scores / denom, 0.0)

    results = {}
    for name, scores in [
        ("common_neighbors", cn_scores),
        ("adamic_adar",      aa_scores),
        ("jaccard",          jc_scores),
    ]:
        auc = roc_auc_score(y_true, scores)
        ap  = average_precision_score(y_true, scores)
        results[name] = {
            "auc_roc":       round(float(auc), 4),
            "avg_precision": round(float(ap),  4),
            "category":      "graph_heuristic",
            "label":         name.replace("_", " ").title(),
        }
        print(f"  {name:<22}  AUC-ROC={auc:.4f}   Avg-Prec={ap:.4f}")

    print(f"  Done in {time.time()-t0:.1f}s")
    return results, CN_d, AA_d, deg   # return matrices for cold-start reuse


def score_cold_heuristics(CN_d, AA_d, deg, cold_pos, cold_neg):
    """
    Score cold-start pairs using the WARM-trained adjacency matrices.

    Cold drugs have NO training edges, so:
      CN[u,v] = 0 for any cold node u -> common_neighbors = 0 always
      AA[u,v] = 0 for any cold node u -> adamic_adar     = 0 always
      JC[u,v] = 0                      -> jaccard         = 0 always

    All cold pairs receive score 0 -> heuristics cannot distinguish positive
    from negative cold pairs -> AUC ~= 0.50 (random).
    This is the expected and correct result — heuristics fundamentally
    cannot handle cold-start drugs.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    sep("Graph heuristics — COLD-START pairs")

    pairs  = np.vstack([cold_pos, cold_neg])
    y_true = np.array([1]*len(cold_pos) + [0]*len(cold_neg))
    u, v   = pairs[:, 0], pairs[:, 1]

    # Clip to matrix size (cold nodes may be out of range if n_nodes was tight)
    n = CN_d.shape[0]
    u_c = np.clip(u, 0, n - 1)
    v_c = np.clip(v, 0, n - 1)

    cn_scores = CN_d[u_c, v_c].astype(float)
    aa_scores = AA_d[u_c, v_c].astype(float)
    denom     = deg[u_c] + deg[v_c] - cn_scores
    jc_scores = np.where(denom > 0, cn_scores / denom, 0.0)

    cold_results = {}
    for name, scores in [
        ("common_neighbors", cn_scores),
        ("adamic_adar",      aa_scores),
        ("jaccard",          jc_scores),
    ]:
        # If all scores are 0 (expected for cold), roc_auc_score raises a
        # warning but still returns 0.5 — handle gracefully.
        if scores.max() == 0:
            auc, ap = 0.5, float(y_true.mean())
            print(f"  {name:<22}  AUC-ROC={auc:.4f}   Avg-Prec={ap:.4f}"
                  f"  [all scores=0 — cold drug has no training neighbours]")
        else:
            auc = roc_auc_score(y_true, scores)
            ap  = average_precision_score(y_true, scores)
            print(f"  {name:<22}  AUC-ROC={auc:.4f}   Avg-Prec={ap:.4f}")
        cold_results[name] = {"auc_roc": round(float(auc), 4),
                              "avg_precision": round(float(ap), 4)}

    print("  ^ Expected: AUC ~0.50 — heuristics are blind to cold-start drugs.")
    return cold_results


def score_cold_heuristics_direct(cold_pos, cold_neg):
    """
    Cold-start heuristic scoring without any matrix computation.

    Every pair in cold_pos/cold_neg contains at least one cold drug.
    Cold drugs have degree 0 in the training graph (their edges were withheld),
    so ALL graph heuristics score exactly 0 for every cold pair.

    When all scores are 0, no ranking is possible -> AUC = 0.50 exactly
    (any tie-breaking is random -> expected AUC is 0.5 by definition).
    Avg precision = fraction of positives = 0.5 (balanced labels).
    """
    sep("Graph heuristics — COLD-START pairs  (analytical, no matmul needed)")
    n_pos = len(cold_pos)
    n_neg = len(cold_neg)
    pos_frac = round(n_pos / (n_pos + n_neg), 4)
    print(f"  {n_pos:,} cold positive pairs + {n_neg:,} cold negative pairs")
    print(f"  All graph heuristic scores = 0 (cold drugs have no training neighbours)")
    print(f"  -> AUC = 0.5000  Avg-Prec = {pos_frac:.4f}  (random baseline, by design)")

    result = {"auc_roc": 0.5, "avg_precision": pos_frac}
    for name in ("common_neighbors", "adamic_adar", "jaccard"):
        print(f"  {name:<22}  AUC-ROC=0.5000   Avg-Prec={pos_frac:.4f}  [all scores=0]")
    return {k: dict(result) for k in ("common_neighbors", "adamic_adar", "jaccard")}


# ---------------------------------------------------------------------------
# Baseline 2: Logistic Regression  (TM10G — non-graph ML)
# ---------------------------------------------------------------------------

def run_logistic_regression(tr_pos, te_pos, tr_neg, te_neg, feat_matrix, feat_cols):
    """
    Pair feature vector: [|feat_A − feat_B|,  feat_A ⊙ feat_B]
    - element-wise product  captures shared pharmacological properties
      (e.g. both drugs are CYP3A4 substrates -> product feature = 1)
    - absolute difference   captures how dissimilar the drugs are
    Symmetric: prediction is order-independent (A,B) == (B,A).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score

    sep("Logistic Regression on drug features  [non-graph ML baseline — TM10G]")
    t0 = time.time()

    n_nodes = feat_matrix.shape[0]
    print(f"  Feature matrix:   {feat_matrix.shape[0]:,} nodes × {feat_matrix.shape[1]} features")

    def pair_features(pairs):
        # Clip indices to valid range (some negative-sampled nodes may be sparse)
        u = np.clip(pairs[:, 0], 0, n_nodes - 1)
        v = np.clip(pairs[:, 1], 0, n_nodes - 1)
        fa, fb = feat_matrix[u], feat_matrix[v]
        return np.hstack([np.abs(fa - fb), fa * fb])   # shape (N, 2F)

    X_tr = pair_features(np.vstack([tr_pos, tr_neg]))
    y_tr = np.array([1]*len(tr_pos) + [0]*len(tr_neg), dtype=int)

    X_te = pair_features(np.vstack([te_pos, te_neg]))
    y_te = np.array([1]*len(te_pos) + [0]*len(te_neg), dtype=int)

    print(f"  Pair feature dim: {X_tr.shape[1]}  ({feat_matrix.shape[1]} × 2)")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0,
                                      solver="lbfgs", random_state=42)),
    ])
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, probs)
    ap  = average_precision_score(y_te, probs)
    print(f"  {'logistic_regression':<22}  AUC-ROC={auc:.4f}   Avg-Prec={ap:.4f}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # Top features by |coefficient|  ->  feeds RM1 (explainability)
    lr_model   = clf.named_steps["lr"]
    pair_names = [f"diff_{c}" for c in feat_cols] + [f"prod_{c}" for c in feat_cols]
    coef       = lr_model.coef_[0]
    top_idx    = np.argsort(np.abs(coef))[-15:][::-1]
    top_feats  = [(pair_names[i], round(float(coef[i]), 4)) for i in top_idx]

    sep("Top-15 predictive features (LR coefficients)")
    for fname, c in top_feats:
        direction = "↑ interaction" if c > 0 else "↓ interaction"
        print(f"  {c:+.4f}   {fname:<55}  {direction}")

    return {
        "logistic_regression": {
            "auc_roc":        round(float(auc), 4),
            "avg_precision":  round(float(ap),  4),
            "category":       "non_graph_ml",
            "label":          "Logistic Regression (drug features)",
            "top_features":   top_feats,
        }
    }, clf   # return fitted clf for cold-start reuse


def score_cold_lr(clf, cold_pos, cold_neg, feat_matrix):
    """
    Evaluate the warm-trained LR on cold-start pairs.

    The LR was trained on warm pairs only (no cold drug edges in training).
    Cold drugs' NODE FEATURES (physicochemical, ATC, CYP450) ARE available —
    they were computed at graph-build time from DrugBank metadata, not from
    the interaction graph.

    LR should achieve meaningful AUC here because it relies on feature
    similarity, not graph topology.  This is its advantage over heuristics
    for cold-start drugs.  GNN should beat it by additionally using
    partial neighbourhood signals.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    sep("Logistic Regression — COLD-START pairs")

    n_nodes = feat_matrix.shape[0]

    def pair_features(pairs):
        u = np.clip(pairs[:, 0], 0, n_nodes - 1)
        v = np.clip(pairs[:, 1], 0, n_nodes - 1)
        fa, fb = feat_matrix[u], feat_matrix[v]
        return np.hstack([np.abs(fa - fb), fa * fb])

    X_cold = pair_features(np.vstack([cold_pos, cold_neg]))
    y_cold = np.array([1]*len(cold_pos) + [0]*len(cold_neg))

    probs = clf.predict_proba(X_cold)[:, 1]
    auc   = roc_auc_score(y_cold, probs)
    ap    = average_precision_score(y_cold, probs)

    print(f"  {'logistic_regression':<22}  AUC-ROC={auc:.4f}   Avg-Prec={ap:.4f}")
    print("  ^ Uses node features only — no graph topology needed for cold drugs.")
    return {"auc_roc": round(float(auc), 4), "avg_precision": round(float(ap), 4)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

DISPLAY_ORDER = [
    ("random_chance",        "Random (chance)",                   "0.5000", "—"),
    ("common_neighbors",     "Common Neighbors",                  None,     None),
    ("adamic_adar",          "Adamic-Adar  ★",                   None,     None),
    ("jaccard",              "Jaccard Coefficient",               None,     None),
    ("logistic_regression",  "Logistic Regression (drug feats)",  None,     None),
    ("gnn",                  "GNN — graph + features  ★  (ours)", "TBD",    "TBD"),
]


def save_and_print(results, cold_results, args):
    """
    results      : warm-split AUC-ROC per method
    cold_results : cold-start AUC-ROC per method (None if --warm-only)
    """
    OUT.mkdir(parents=True, exist_ok=True)

    # Inject GNN placeholders
    results["gnn"] = {
        "auc_roc":       "TBD",
        "avg_precision": "TBD",
        "category":      "gnn",
        "label":         "GNN (graph + features)",
        "note":          "Fill in from Laure's model evaluation on the same split.",
    }
    results["random_chance"] = {
        "auc_roc": 0.5,
        "avg_precision": "—",
        "category": "trivial",
        "label": "Random (chance)",
    }

    if cold_results:
        cold_results["gnn"] = {
            "auc_roc": "TBD", "avg_precision": "TBD"
        }
        cold_results["random_chance"] = {
            "auc_roc": 0.5, "avg_precision": "—"
        }

    payload = {
        "config": {
            "neg_ratio":  args.neg_ratio,
            "test_size":  args.test_size,
            "seed":       args.seed,
            "cold_frac":  getattr(args, "cold_frac", 0.10),
        },
        "results":      results,
        "cold_results": cold_results or {},
    }
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved -> {JSON_PATH}")

    # ── Warm table ────────────────────────────────────────────────────────────
    sep("WARM EVALUATION (standard transductive link prediction)")
    header = f"{'Method':<42} {'AUC-ROC':>8} {'Avg Prec':>10}"
    print(f"\n  {header}")
    print(f"  {'-'*42} {'-'*8} {'-'*10}")

    rows_csv = []
    for key, label, auc_default, ap_default in DISPLAY_ORDER:
        r   = results.get(key, {})
        auc = r.get("auc_roc",       auc_default) if auc_default is None else auc_default
        ap  = r.get("avg_precision", ap_default)  if ap_default  is None else ap_default
        print(f"  {label:<42} {str(auc):>8} {str(ap):>10}")
        rows_csv.append({"setting": "warm", "method": label,
                         "auc_roc": auc, "avg_precision": ap})

    # ── Cold-start table ──────────────────────────────────────────────────────
    if cold_results:
        sep("COLD-START EVALUATION (10% drugs held out — GNN's real use case)")
        print(f"\n  {'Method':<42} {'AUC-ROC':>8} {'Avg Prec':>10}")
        print(f"  {'-'*42} {'-'*8} {'-'*10}")
        for key, label, auc_default, ap_default in DISPLAY_ORDER:
            r   = cold_results.get(key, {})
            auc = r.get("auc_roc",       auc_default) if auc_default is None else auc_default
            ap  = r.get("avg_precision", ap_default)  if ap_default  is None else ap_default
            print(f"  {label:<42} {str(auc):>8} {str(ap):>10}")
            rows_csv.append({"setting": "cold", "method": label,
                             "auc_roc": auc, "avg_precision": ap})

        sep("INTERPRETATION")
        print()
        print("  WARM  — all methods score high because the dense interaction")
        print("          graph (avg degree 344) gives heuristics many common")
        print("          neighbours for any test pair.")
        print()
        print("  COLD  — heuristics score ~0.50 (random) because cold drugs")
        print("          have NO training neighbours.  LR uses node features")
        print("          (physicochemical, ATC, CYP450) and scores meaningfully.")
        print("          The GNN should outperform LR by combining features")
        print("          with partial graph context — this is its killer use case.")
        print()
        lr_warm  = results.get("logistic_regression", {}).get("auc_roc", "?")
        lr_cold  = cold_results.get("logistic_regression", {}).get("auc_roc", "?")
        aa_warm  = results.get("adamic_adar", {}).get("auc_roc", "?")
        aa_cold  = cold_results.get("adamic_adar", {}).get("auc_roc", "?")
        print(f"  Adamic-Adar  :  warm={aa_warm}  ->  cold={aa_cold}  (topology collapses)")
        print(f"  LR (features):  warm={lr_warm}  ->  cold={lr_cold}  (features survive)")
        print(f"  GNN           :  warm=TBD       ->  cold=TBD       (should beat both)")

    pd.DataFrame(rows_csv).to_csv(CSV_PATH, index=False)
    print(f"\n  Saved -> {CSV_PATH}")
    sep()
    print("  ★  Adamic-Adar is the primary non-AI baseline (TM2A).")
    print("  ★  GNN warm column -> fill from Laure's eval on edge_split.npz")
    print("  ★  GNN cold column -> fill from Laure's eval on cold_split.npz")
    sep()


def print_saved_results():
    if not JSON_PATH.exists():
        print(f"No results file at {JSON_PATH}. Run without --results-only first.")
        sys.exit(1)
    with open(JSON_PATH) as f:
        payload = json.load(f)
    results      = payload.get("results", {})
    cold_results = payload.get("cold_results", {})

    sep("WARM RESULTS")
    print(f"\n  {'Method':<42} {'AUC-ROC':>8} {'Avg Prec':>10}")
    print(f"  {'-'*42} {'-'*8} {'-'*10}")
    for key, label, auc_default, ap_default in DISPLAY_ORDER:
        r   = results.get(key, {})
        auc = r.get("auc_roc",       auc_default) if auc_default is None else auc_default
        ap  = r.get("avg_precision", ap_default)  if ap_default  is None else ap_default
        print(f"  {label:<42} {str(auc):>8} {str(ap):>10}")
    sep()

    if cold_results:
        sep("COLD-START RESULTS")
        print(f"\n  {'Method':<42} {'AUC-ROC':>8} {'Avg Prec':>10}")
        print(f"  {'-'*42} {'-'*8} {'-'*10}")
        for key, label, auc_default, ap_default in DISPLAY_ORDER:
            r   = cold_results.get(key, {})
            auc = r.get("auc_roc",       auc_default) if auc_default is None else auc_default
            ap  = r.get("avg_precision", ap_default)  if ap_default  is None else ap_default
            print(f"  {label:<42} {str(auc):>8} {str(ap):>10}")
        sep()

    lr = results.get("logistic_regression", {})
    if lr.get("top_features"):
        sep("Top features (LR coefficients — explainability)")
        for fname, c in lr["top_features"][:10]:
            direction = "↑ interaction" if c > 0 else "↓ interaction"
            print(f"  {c:+.4f}   {fname:<55}  {direction}")
        sep()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DDI link-prediction baselines (graph heuristics + LR)"
    )
    parser.add_argument("--neg-ratio",    type=int,   default=1,
                        help="Negative edges per positive (default 1)")
    parser.add_argument("--test-size",    type=float, default=0.1,
                        help="Test fraction of visible edges (default 0.1, matches Laure testRatio=0.1)")
    parser.add_argument("--mask-ratio",   type=float, default=0.2,
                        help="Fraction of edges masked as nnPU positives (default 0.2, matches Laure maskRatio=0.2)")
    parser.add_argument("--cold-frac",    type=float, default=0.10,
                        help="Fraction of drugs held out for cold-start eval (default 0.10)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--results-only", action="store_true",
                        help="Print previously saved results without re-running")
    parser.add_argument("--warm-only",    action="store_true",
                        help="Skip cold-start evaluation (faster)")
    parser.add_argument("--load-split",   type=str,   default=None,
                        help="Path to a .npz split file from Laure's GNN training "
                             "(keys: train_pos, train_neg, test_pos, test_neg). "
                             "If provided, uses hers for the warm evaluation.")
    args = parser.parse_args()

    if args.results_only:
        print_saved_results()
        return

    sep("STEP 9 — LINK PREDICTION BASELINES")
    print(f"  neg-ratio={args.neg_ratio}  mask-ratio={args.mask_ratio}  "
          f"test-size={args.test_size}  cold-frac={args.cold_frac}  seed={args.seed}")
    print(f"  Split config matches Laure's hetero_model.ipynb (maskRatio=0.2, testRatio=0.1)")

    pos_edges, feat_matrix, feat_cols, n_nodes = load_graph_data()

    # ── Warm split ─────────────────────────────────────────────────────────────
    if args.load_split:
        sep("Loading warm split from Laure's file")
        split  = np.load(args.load_split)
        tr_pos = split["train_pos"].astype(np.int64)
        tr_neg = split["train_neg"].astype(np.int64)
        te_pos = split["test_pos"].astype(np.int64)
        te_neg = split["test_neg"].astype(np.int64)
        print(f"  Loaded: {args.load_split}")
        print(f"  Train: {len(tr_pos):,} pos  +  {len(tr_neg):,} neg")
        print(f"  Test:  {len(te_pos):,} pos  +  {len(te_neg):,} neg")
    else:
        tr_pos, te_pos, tr_neg, te_neg = make_split(
            pos_edges, n_nodes,
            neg_ratio=args.neg_ratio,
            test_size=args.test_size,
            mask_ratio=args.mask_ratio,
            seed=args.seed,
        )

    # ── Warm evaluation ────────────────────────────────────────────────────────
    results = {}
    heuristic_res, CN_d, AA_d, deg = run_graph_heuristics(
        tr_pos, te_pos, te_neg, n_nodes)
    results.update(heuristic_res)

    lr_res, clf = run_logistic_regression(
        tr_pos, te_pos, tr_neg, te_neg, feat_matrix, feat_cols)
    results.update(lr_res)

    # ── Cold-start evaluation ──────────────────────────────────────────────────
    cold_results = None
    if not args.warm_only:
        sep("COLD-START EVALUATION SETUP")
        # Build cold split from the FULL edge set (independent of warm split)
        # so cold drugs are truly unseen in training
        cold_tr_pos, cold_pos, cold_tr_neg, cold_neg, cold_drugs = make_cold_split(
            pos_edges, n_nodes,
            cold_frac=args.cold_frac,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
        )
        # Heuristics on cold pairs — no matmul needed.
        # Every cold pair has at least one cold drug; cold drugs have degree 0
        # in the training graph (their edges are withheld), so their rows/cols
        # in A are all-zero -> CN[cold, :] = 0, AA[cold, :] = 0 for all
        # partners.  We assign score 0 directly and compute AUC analytically.
        cold_heuristic = score_cold_heuristics_direct(cold_pos, cold_neg)

        # LR: retrain on cold_tr_pos so no cold-drug edges leak into training
        lr_cold_res, clf_cold = run_logistic_regression(
            cold_tr_pos, cold_pos, cold_tr_neg, cold_neg, feat_matrix, feat_cols)
        cold_lr = score_cold_lr(clf_cold, cold_pos, cold_neg, feat_matrix)

        cold_results = {
            "common_neighbors":   cold_heuristic["common_neighbors"],
            "adamic_adar":        cold_heuristic["adamic_adar"],
            "jaccard":            cold_heuristic["jaccard"],
            "logistic_regression": {
                **cold_lr,
                "category": "non_graph_ml",
                "label":    "Logistic Regression (drug features)",
            },
        }

    save_and_print(results, cold_results, args)


if __name__ == "__main__":
    main()
