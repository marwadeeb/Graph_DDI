"""
gnn_predictor.py
----------------
Interface between the Flask API and Laure's GNN model.

This module exposes a single public function:

    predict(id_a, id_b)  ->  {"found": bool, "probability": float, "note": str}

Integration checklist for Laure:
  1. Train your GNN and save the model checkpoint, e.g.:
         torch.save(model.state_dict(), "data/step4_graph/gnn_model.pt")
  2. Replace _load_model() below with real model loading code.
  3. Replace _predict_real() with real inference code.
  4. The predict() function signature must NOT change — Flask calls it directly.

Until the model file exists, predict() returns mock data so the UI renders correctly.

Mock behaviour (DEMO ONLY):
    - If the pair is in the DrugBank DDI database → mock says "found", prob ~0.92
    - Otherwise                                   → mock says "not found", prob ~0.08
    The mock adds ±0.05 random noise per pair for visual realism.
"""

import os, hashlib
import numpy as np

WORKING_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR     = os.path.join(WORKING_DIR, "data", "step4_graph")
APPROVED_DIR  = os.path.join(WORKING_DIR, "data", "step3_approved")
MODEL_PATH    = os.path.join(GRAPH_DIR, "gnn_model.pt")

# ---------------------------------------------------------------------------
# Internal state (lazy-loaded)
# ---------------------------------------------------------------------------
_model       = None
_node_map    = None   # drugbank_id -> node_idx
_lookup_set  = None   # set of (id_a, id_b) known DDI pairs
_mock_mode   = None   # True until real model loaded


def _load_node_map():
    global _node_map
    if _node_map is not None:
        return _node_map
    import pandas as pd
    df = pd.read_csv(os.path.join(GRAPH_DIR, "node_mapping.csv"),
                     usecols=["node_idx", "drugbank_id"])
    _node_map = dict(zip(df["drugbank_id"], df["node_idx"]))
    return _node_map


def _load_lookup():
    global _lookup_set
    if _lookup_set is not None:
        return _lookup_set
    import pandas as pd
    ddi = pd.read_csv(os.path.join(APPROVED_DIR, "drug_interactions_dedup.csv"),
                      usecols=["drugbank_id_a", "drugbank_id_b"])
    _lookup_set = set(zip(ddi["drugbank_id_a"], ddi["drugbank_id_b"]))
    return _lookup_set


def _load_model():
    """
    LAURE: Replace this function with real model loading.

    Example:
        import torch
        from your_gnn_module import DDIGNNModel
        model = DDIGNNModel(in_channels=959, hidden_channels=256, out_channels=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    """
    global _model, _mock_mode
    if not os.path.exists(MODEL_PATH):
        _mock_mode = True
        return None
    try:
        import torch
        # Placeholder — replace with actual model class import
        _model     = torch.load(MODEL_PATH, map_location="cpu")
        _mock_mode = False
        print("[GNN] Model loaded from", MODEL_PATH, flush=True)
        return _model
    except Exception as e:
        print(f"[GNN] Failed to load model ({e}), using mock mode", flush=True)
        _mock_mode = True
        return None


def _predict_real(id_a: str, id_b: str) -> dict:
    """
    LAURE: Replace this function with real GNN inference.

    Example:
        node_map = _load_node_map()
        idx_a = node_map.get(id_a)
        idx_b = node_map.get(id_b)
        if idx_a is None or idx_b is None:
            return {"found": None, "probability": None,
                    "note": f"Drug not in graph: {id_a if idx_a is None else id_b}"}
        with torch.no_grad():
            # feed node pair through link-prediction head
            prob = model.predict_link(idx_a, idx_b).item()
        return {"found": prob >= 0.5, "probability": round(prob, 4), "note": "GNN prediction"}
    """
    raise NotImplementedError("Real GNN inference not yet implemented")


def _predict_mock(id_a: str, id_b: str) -> dict:
    """
    Deterministic mock using lookup set as ground truth.
    Uses a hash of the pair to generate stable noise so the same pair
    always returns the same probability (no randomness per request).

    DEMO_OVERRIDES: specific pairs given fixed high probability to demonstrate
    the GNN-predicted state in the UI. These represent clinically plausible
    interactions not explicitly documented in DrugBank — the kind of novel
    signal a real GNN would surface from graph structure.
    """
    # DrugBank IDs for demo pairs confirmed NOT in DrugBank DDI database.
    # These represent novel GNN predictions based on pharmacological graph patterns.
    #
    # Bicalutamide (DB01128) × Asenapine (DB06216):
    #   Clinical rationale: both drugs prolong the QT interval. Co-administration
    #   may increase risk of arrhythmia — a real clinical concern not captured
    #   as a formal DDI in DrugBank.
    DEMO_OVERRIDES = {
        frozenset(["DB01128", "DB06216"]): 0.82,   # Bicalutamide × Asenapine
    }

    pair_key = frozenset([id_a, id_b])
    if pair_key in DEMO_OVERRIDES:
        prob = DEMO_OVERRIDES[pair_key]
        return {
            "found":       True,
            "probability": prob,
            "note":        "DEMO — novel GNN prediction (not in DrugBank)",
        }

    lookup = _load_lookup()
    found  = (id_a, id_b) in lookup or (id_b, id_a) in lookup

    # Stable noise in [-0.05, +0.05] derived from pair hash
    h     = int(hashlib.md5(f"{id_a}:{id_b}".encode()).hexdigest(), 16)
    noise = ((h % 1000) / 1000 - 0.5) * 0.10

    base_prob = 0.91 if found else 0.08
    prob      = round(float(np.clip(base_prob + noise, 0.01, 0.99)), 4)

    return {
        "found":       found,
        "probability": prob,
        "note":        "DEMO — mock GNN (model not yet trained)",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Returns True when the real GNN model is loaded (not mock mode)."""
    if _mock_mode is None:
        _load_model()
    return not _mock_mode


def predict(id_a: str, id_b: str) -> dict:
    """
    Predict DDI for a drug pair.

    Args:
        id_a: DrugBank ID of drug A (e.g. "DB00682")
        id_b: DrugBank ID of drug B (e.g. "DB00945")

    Returns:
        {
            "found":       bool   — predicted interaction exists
            "probability": float  — model confidence [0, 1]
            "note":        str    — human-readable note
            "mock":        bool   — True if this is demo data
        }
    """
    if _mock_mode is None:
        _load_model()

    try:
        if _mock_mode:
            result = _predict_mock(id_a, id_b)
        else:
            result = _predict_real(id_a, id_b)
    except Exception as e:
        result = _predict_mock(id_a, id_b)
        result["note"] = f"GNN error ({e}) — falling back to mock"

    result["mock"] = _mock_mode
    return result
