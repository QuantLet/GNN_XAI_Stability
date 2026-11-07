"""Heterogeneous GraphSAGE classifier (Sections 3.2-3.3).

Two SAGEConv layers, hidden dim 64, mean aggregation, two-class linear head.
Trained to convergence with Adam (lr 1e-3, weight decay 1e-4), cross-entropy on
labelled transaction nodes, early stopping patience 50 over <=200 epochs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

from . import config as C


class SAGEBackbone(nn.Module):
    """Homogeneous backbone; wrapped with to_hetero for the heterogeneous graph.

    Returns both layer-one and layer-two transaction representations so the
    hybrid can use either z^(1) or z^(2) (Eq. 1).
    """

    def __init__(self, hidden_dim: int = C.SAGE_PARAMS["hidden_dim"]):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim, aggr="mean")
        self.conv2 = SAGEConv((-1, -1), hidden_dim, aggr="mean")

    def forward(self, x, edge_index):
        z1 = F.relu(self.conv1(x, edge_index))
        z2 = F.relu(self.conv2(z1, edge_index))
        return z2  # to_hetero tracks intermediates separately; see GraphSAGE


class GraphSAGE(nn.Module):
    def __init__(self, metadata, hidden_dim: int = C.SAGE_PARAMS["hidden_dim"]):
        super().__init__()
        self.conv1 = to_hetero(_OneLayer(hidden_dim), metadata, aggr="sum")
        self.conv2 = to_hetero(_OneLayer(hidden_dim), metadata, aggr="sum")
        self.head = nn.Linear(hidden_dim, 2)

    def encode(self, x_dict, edge_index_dict):
        z1 = {k: F.relu(v) for k, v in self.conv1(x_dict, edge_index_dict).items()}
        z2 = {k: F.relu(v) for k, v in self.conv2(z1, edge_index_dict).items()}
        return z1, z2

    def forward(self, x_dict, edge_index_dict):
        z1, z2 = self.encode(x_dict, edge_index_dict)
        return self.head(z2["transaction"]), z1, z2


class _OneLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = SAGEConv((-1, -1), hidden_dim, aggr="mean")

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def train_graphsage(data, device: str = "cpu", params: dict = C.SAGE_PARAMS,
                    log_every: int = 10, *, seed: int = C.SEED):
    """Train GraphSAGE to convergence and return the fitted model.

    Early stopping monitors training cross-entropy on labelled nodes (a held-out
    GNN validation slice could be substituted; the paper uses patience on the
    training objective with the downstream tabular task as the real evaluator).
    Logs loss and per-epoch wall-time every `log_every` epochs.
    """
    import time

    from . import utils

    log = utils.get_logger()
    C.seed_everything(seed)
    model = GraphSAGE(data.metadata(), params["hidden_dim"]).to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])

    y = data["transaction"].y
    tm = data["transaction"].train_mask
    # class weights for the ~3.5% positive rate
    pos = float(y[tm].sum())
    neg = float((~y[tm].bool()).sum())
    w = torch.tensor([1.0, neg / max(pos, 1.0)], dtype=torch.float, device=device)

    best, best_state, wait = float("inf"), None, 0
    model.train()
    t0 = time.time()
    for epoch in range(params["max_epochs"]):
        opt.zero_grad()
        logits, _, _ = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(logits[tm], y[tm], weight=w)
        loss.backward()
        opt.step()
        val = loss.item()
        if val < best - 1e-5:
            best, best_state, wait = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= params["patience"]:
                log.info("    GNN early-stop at epoch %d (best loss %.4f)", epoch, best)
                break
        if epoch % log_every == 0:
            dt = (time.time() - t0) / (epoch + 1)
            log.info("    GNN epoch %d/%d  loss=%.4f  (%.1fs/epoch)",
                     epoch, params["max_epochs"], val, dt)
    if best_state is not None:
        model.load_state_dict(best_state)
    log.info("    GNN done in %.0fs", time.time() - t0)
    return model
