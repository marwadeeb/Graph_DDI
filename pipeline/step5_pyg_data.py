"""
step5_pyg_data.py
-----------------
Build a PyTorch Geometric Data object from step4_graph CSVs and save as
data/step4_graph/ddi_graph.pt

Node features  : node_features_combined.csv  (4795 x 959 = 191 struct + 768 text)
Edge index     : edge_index.csv              (824,249 undirected DDI pairs)
Edge labels    : all 1 (positive edges only -- negative sampling done at train time)

Usage:
    python pipeline/step5_pyg_data.py
    python pipeline/step5_pyg_data.py --structural-only   # use 191-dim features only

Output files (data/step4_graph/):
    ddi_graph.pt               -- full PyG Data object (959-dim node features)
    ddi_graph_structural.pt    -- structural-only variant (191-dim)
"""

import os, sys, json, argparse, time
import pandas as pd
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DIR   = os.path.join(WORKING_DIR, "data", "step4_graph")

def sep(label=""):
    width = 68
    if label:
        pad = (width - len(label) - 2) // 2
        print("-" * pad + " " + label + " " + "-" * (width - pad - len(label) - 2))
    else:
        print("-" * width)

# ---------------------------------------------------------------------------

def build(structural_only: bool = False) -> Data:
    t0 = time.time()

    feat_file = "node_features.csv" if structural_only else "node_features_combined.csv"
    feat_path = os.path.join(GRAPH_DIR, feat_file)
    edge_path = os.path.join(GRAPH_DIR, "edge_index.csv")
    map_path  = os.path.join(GRAPH_DIR, "node_mapping.csv")
    fname_path = os.path.join(GRAPH_DIR, "feature_names.json")

    # --- node features ------------------------------------------------------
    sep("LOADING NODE FEATURES")
    print(f"  File : {feat_file}")
    feat_df = pd.read_csv(feat_path, index_col=0)       # drop node_idx col
    print(f"  Shape: {feat_df.shape[0]:,} nodes x {feat_df.shape[1]:,} features")

    # fill any remaining NaN (should be none after step4, but just in case)
    feat_df = feat_df.fillna(0.0)
    x = torch.tensor(feat_df.values, dtype=torch.float32)

    # --- edge index ---------------------------------------------------------
    sep("LOADING EDGE INDEX")
    edge_df = pd.read_csv(edge_path)
    print(f"  Shape: {len(edge_df):,} edges")

    src = torch.tensor(edge_df["src_idx"].values, dtype=torch.long)
    dst = torch.tensor(edge_df["dst_idx"].values, dtype=torch.long)

    # undirected: add both directions
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ], dim=0)                                            # [2, 2E]
    print(f"  edge_index shape (both dirs): {list(edge_index.shape)}")

    # keep interaction_id as edge attribute (one per original pair, duplicated)
    iid = torch.tensor(edge_df["interaction_id"].values, dtype=torch.long)
    edge_attr = torch.cat([iid, iid]).unsqueeze(1)      # [2E, 1]

    # --- node mapping -------------------------------------------------------
    mapping_df = pd.read_csv(map_path)
    drugbank_ids = mapping_df["drugbank_id"].tolist()
    names        = mapping_df["name"].tolist()

    # --- feature names ------------------------------------------------------
    feature_names = None
    if os.path.exists(fname_path):
        with open(fname_path, "r") as f:
            feature_names = json.load(f)

    # --- assemble Data object -----------------------------------------------
    sep("ASSEMBLING PyG DATA OBJECT")
    data = Data(
        x          = x,              # [N, F]  node feature matrix
        edge_index = edge_index,     # [2, 2E] COO format, both directions
        edge_attr  = edge_attr,      # [2E, 1] interaction_id
        num_nodes  = x.size(0),
    )

    # store metadata as plain attributes (not tensors)
    data.drugbank_ids   = drugbank_ids
    data.drug_names     = names
    data.feature_names  = feature_names
    data.feature_dim    = x.size(1)

    print(f"  num_nodes   : {data.num_nodes:,}")
    print(f"  num_edges   : {data.num_edges:,}  (= {data.num_edges // 2:,} undirected pairs x2)")
    print(f"  x shape     : {list(x.shape)}")
    print(f"  edge_index  : {list(edge_index.shape)}")
    print(f"  edge_attr   : {list(edge_attr.shape)}")
    print(f"  is_undirected: {data.is_undirected()}")

    # --- save ---------------------------------------------------------------
    sep("SAVING")
    suffix = "_structural" if structural_only else ""
    out_path = os.path.join(GRAPH_DIR, f"ddi_graph{suffix}.pt")
    torch.save(data, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {os.path.basename(out_path)}  ({size_mb:.1f} MB)")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return data


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-only", action="store_true",
                        help="Use 191-dim structural features only (skip text embeddings)")
    args = parser.parse_args()

    sep("STEP 5 - PyG DATA OBJECT")
    print(f"  Mode: {'structural only (191-dim)' if args.structural_only else 'combined (191 struct + 768 text = 959-dim)'}")

    data = build(structural_only=args.structural_only)

    # --- quick sanity checks ------------------------------------------------
    sep("SANITY CHECK")
    assert data.num_nodes == 4795,             f"Expected 4795 nodes, got {data.num_nodes}"
    assert data.num_edges == 824249 * 2,       f"Expected {824249*2} directed edges, got {data.num_edges}"
    assert not data.x.isnan().any(),           "NaNs found in node features!"
    assert data.edge_index.min() >= 0,         "Negative node index in edge_index"
    assert data.edge_index.max() < data.num_nodes, "Edge index out of bounds"
    print("  All checks passed.")
    sep()

    print()
    print("Load in your GNN training script with:")
    print("    import torch")
    print("    data = torch.load('data/step4_graph/ddi_graph.pt')")
    print("    x          = data.x           # [4795, 959]")
    print("    edge_index = data.edge_index  # [2, 1648498]")
