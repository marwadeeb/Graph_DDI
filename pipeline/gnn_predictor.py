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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_GRAPH_DIR = os.path.join(_REPO_ROOT, "data", "step4_graph")

_HETERO_MODEL_PATH = os.path.join(_GRAPH_DIR, "bestHeteroModel.pt")
_HETERO_GRAPH_PATH = os.path.join(_GRAPH_DIR, "hetero_ddi_graph.pt")
_HOMO_MODEL_PATH = os.path.join(_GRAPH_DIR, "bestModel.pt")
_HOMO_GRAPH_PATH = os.path.join(_GRAPH_DIR, "ddi_graph.pt")

GNN_THRESHOLD = 0.43


#Homo model (fallback)
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, hidden_ch)
        self.conv3 = SAGEConv(hidden_ch, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)


class LinkPredictor(nn.Module):
    def __init__(self, inChannels, hiddenChannels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inChannels * 2, hiddenChannels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hiddenChannels, 1),
        )

    def forward(self, z, edge_index):
        pair = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return torch.sigmoid(self.mlp(pair).squeeze(-1))


class HomoDDIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GraphSAGEEncoder(980, 256, 64, 0.3)
        self.predictor = LinkPredictor(64, 128)

    def forward(self, x, edge_index, pred_edge_index):
        return self.predictor(self.encoder(x, edge_index), pred_edge_index)


#Hetero model (preferred)
class HeteroGraphSAGEEncoder(nn.Module):
    def __init__(self, drug_in=980, protein_in=5, hidden=256, out=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        from torch_geometric.nn import HeteroConv, SAGEConv
        self.conv1 = HeteroConv({
            ("drug", "ddi", "drug"): SAGEConv(drug_in, hidden),
            ("drug", "targets", "protein"): SAGEConv((drug_in, protein_in), hidden),
            ("protein", "rev_targets", "drug"): SAGEConv((protein_in, drug_in), hidden),
        }, aggr="sum")
        self.conv2 = HeteroConv({
            ("drug", "ddi", "drug"): SAGEConv(hidden, hidden),
            ("drug", "targets", "protein"): SAGEConv(hidden, hidden),
            ("protein", "rev_targets", "drug"): SAGEConv(hidden, hidden),
        }, aggr="sum")
        self.conv3 = HeteroConv({
            ("drug", "ddi", "drug"): SAGEConv(hidden, out),
            ("drug", "targets", "protein"): SAGEConv(hidden, out),
            ("protein", "rev_targets", "drug"): SAGEConv(hidden, out),
        }, aggr="sum")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        return x_dict["drug"], x_dict["protein"]


class EnhancedLinkPredictor(nn.Module):
    #NCN-style decoder: pools shared DDI neighbours + shared protein targets per pair
    #Input = [z_u | z_v | cn_ddi_pool | cn_protein_pool] (256-dim)
    def __init__(self, drug_dim=64, protein_dim=64, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(drug_dim * 2 + drug_dim + protein_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, 1),
        )
        self._drug_dim = drug_dim
        self._protein_dim = protein_dim

    def predict_pair(self, z_drug, z_protein, ddi_adj, dp_adj, idx_a, idx_b):
        self.eval()
        dev = z_drug.device
        zero_d = torch.zeros(self._drug_dim, device=dev)
        zero_p = torch.zeros(self._protein_dim, device=dev)
        cn_mask = ddi_adj[idx_a] & ddi_adj[idx_b]
        cn_ddi = z_drug[cn_mask].mean(0) if cn_mask.any() else zero_d
        sp_mask = dp_adj[idx_a] & dp_adj[idx_b]
        cn_prot = z_protein[sp_mask].mean(0) if sp_mask.any() else zero_p
        vec = torch.cat([z_drug[idx_a], z_drug[idx_b], cn_ddi, cn_prot])
        return torch.sigmoid(self.mlp(vec.unsqueeze(0))).item()

    def forward(self, z_drug, z_protein, ddi_ei, dp_ei, pred_ei):
        src, dst = pred_ei[0], pred_ei[1]
        n_d, n_p = z_drug.shape[0], z_protein.shape[0]
        dev = z_drug.device
        adj_ddi = torch.zeros(n_d, n_d, dtype=torch.bool, device=dev)
        adj_ddi[ddi_ei[0], ddi_ei[1]] = True
        adj_ddi[ddi_ei[1], ddi_ei[0]] = True
        shared_ddi = (adj_ddi[src] & adj_ddi[dst]).float()
        cn_ddi = (shared_ddi @ z_drug) / shared_ddi.sum(1, keepdim=True).clamp(min=1)
        adj_dp = torch.zeros(n_d, n_p, dtype=torch.bool, device=dev)
        adj_dp[dp_ei[0], dp_ei[1]] = True
        shared_dp = (adj_dp[src] & adj_dp[dst]).float()
        cn_prot = (shared_dp @ z_protein) / shared_dp.sum(1, keepdim=True).clamp(min=1)
        pair = torch.cat([z_drug[src], z_drug[dst], cn_ddi, cn_prot], dim=-1)
        return torch.sigmoid(self.mlp(pair).squeeze(-1))


class HeteroDDIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HeteroGraphSAGEEncoder()
        self.predictor = EnhancedLinkPredictor(drug_dim=64, protein_dim=64, hidden=128)

    def forward(self, x_dict, edge_index_dict, pred_edge_index):
        z_drug, z_protein = self.encoder(x_dict, edge_index_dict)
        ddi_ei = edge_index_dict[("drug", "ddi", "drug")]
        dp_ei = edge_index_dict[("drug", "targets", "protein")]
        return self.predictor(z_drug, z_protein, ddi_ei, dp_ei, pred_edge_index)


#State
_state = {
    "loaded": False, "mock": False, "variant": None,
    "predictor": None, "embeddings": None, "protein_emb": None,
    "ddi_adj": None, "dp_adj": None, "id_to_idx": None, "device": None,
}


def _load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state["device"] = device

    #Try hetero model first
    if os.path.exists(_HETERO_MODEL_PATH) and os.path.exists(_HETERO_GRAPH_PATH):
        try:
            print("[GNN] Loading HETERO model ...")
            graph = torch.load(_HETERO_GRAPH_PATH, map_location=device, weights_only=False)
            model = HeteroDDIModel().to(device)
            model.load_state_dict(torch.load(_HETERO_MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                z_drug, z_protein = model.encoder(graph.x_dict, graph.edge_index_dict)
            n_d, n_p = graph["drug"].num_nodes, graph["protein"].num_nodes
            ddi_ei = graph["drug", "ddi", "drug"].edge_index
            ddi_adj = torch.zeros(n_d, n_d, dtype=torch.bool, device=device)
            ddi_adj[ddi_ei[0], ddi_ei[1]] = True
            ddi_adj[ddi_ei[1], ddi_ei[0]] = True
            dp_ei = graph["drug", "targets", "protein"].edge_index
            dp_adj = torch.zeros(n_d, n_p, dtype=torch.bool, device=device)
            dp_adj[dp_ei[0], dp_ei[1]] = True
            _state.update({
                "variant": "hetero", "predictor": model.predictor,
                "embeddings": z_drug, "protein_emb": z_protein,
                "ddi_adj": ddi_adj, "dp_adj": dp_adj,
                "id_to_idx": {db_id: i for i, db_id in enumerate(graph["drug"].drugbank_ids)},
                "loaded": True,
            })
            print(f"[GNN] Hetero model loaded. Drug emb: {z_drug.shape}, Protein emb: {z_protein.shape}. Device: {device}")
            return
        except Exception as e:
            print(f"[GNN] Hetero load failed ({e}), trying homo ...")

    #Fallback: homo model
    if os.path.exists(_HOMO_MODEL_PATH) and os.path.exists(_HOMO_GRAPH_PATH):
        try:
            print("[GNN] Loading HOMO model ...")
            graph = torch.load(_HOMO_GRAPH_PATH, map_location=device, weights_only=False)
            model = HomoDDIModel().to(device)
            model.load_state_dict(torch.load(_HOMO_MODEL_PATH, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                emb = model.encoder(graph.x, graph.edge_index)
            _state.update({
                "variant": "homo", "predictor": model.predictor, "embeddings": emb,
                "id_to_idx": {db_id: i for i, db_id in enumerate(graph.drugbank_ids)},
                "loaded": True,
            })
            print(f"[GNN] Homo model loaded. Embeddings: {emb.shape}. Device: {device}")
            return
        except Exception as e:
            print(f"[GNN] Homo load failed ({e}). Falling back to mock mode.")

    print("[GNN] WARNING: No model files found. Running in MOCK mode.")
    _state["mock"] = True
    _state["loaded"] = True


def _ensure_loaded():
    if not _state["loaded"]:
        _load()


def is_available():
    _ensure_loaded()
    return _state["loaded"] and not _state["mock"]


def predict(drugbank_id_a: str, drugbank_id_b: str) -> dict:
    _ensure_loaded()
    if _state["mock"]:
        return {"probability": 0.5, "found": False, "mock": True, "variant": "mock"}
    id_to_idx = _state["id_to_idx"]
    if drugbank_id_a not in id_to_idx or drugbank_id_b not in id_to_idx:
        return {"probability": 0.0, "found": False, "mock": False, "variant": _state["variant"]}
    idx_a, idx_b = id_to_idx[drugbank_id_a], id_to_idx[drugbank_id_b]
    with torch.no_grad():
        if _state["variant"] == "hetero":
            score = _state["predictor"].predict_pair(
                _state["embeddings"], _state["protein_emb"],
                _state["ddi_adj"], _state["dp_adj"], idx_a, idx_b)
        else:
            edge = torch.tensor([[idx_a], [idx_b]], dtype=torch.long, device=_state["device"])
            _state["predictor"].eval()
            score = _state["predictor"](_state["embeddings"], edge).item()
    return {"probability": round(score, 4), "found": True, "mock": False, "variant": _state["variant"]}


def get_model_info() -> dict:
    _ensure_loaded()
    notes = {
        "hetero": "HeteroGraphSAGE + EnhancedLinkPredictor (NCN-style CN pooling), nnPU. AUROC 0.9738 | AUPR 0.9589",
        "homo": "GraphSAGE (homogeneous), nnPU. AUROC 0.9615 | AUPR 0.9450",
        None: "Not loaded",
    }
    return {"loaded": _state["loaded"], "mock": _state["mock"],
            "variant": _state["variant"], "note": notes.get(_state["variant"], "Unknown")}