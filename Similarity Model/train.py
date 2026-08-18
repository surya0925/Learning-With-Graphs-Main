"""
Train EGNN on NCI1 graph classification.

The model uses attention pooling, Laplacian positional encodings, DropEdge,
and a supervised contrastive loss on graph embeddings.

Usage:
    python train.py
    python train.py --hidden 128 --layers 5 --epochs 400
"""

import argparse
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import degree as pyg_degree

from egnn import EGNN


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       type=str,   default="NCI1")
    p.add_argument("--max_nodes",     type=int,   default=150)
    p.add_argument("--epochs",        type=int,   default=400)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--layers",        type=int,   default=5)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--bs",            type=int,   default=64)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--drop_edge",     type=float, default=0.15)
    p.add_argument("--label_smooth",  type=float, default=0.05)
    p.add_argument("--wd",            type=float, default=5e-4)
    p.add_argument("--metric_weight", type=float, default=0.3)
    p.add_argument("--temperature",   type=float, default=0.5,
                   help="temperature for the contrastive loss")
    p.add_argument("--lpe_dim",       type=int,   default=8,
                   help="number of Laplacian eigenvectors")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# DropEdge
# ─────────────────────────────────────────────────────────────────────────────

def drop_edge(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or edge_index.size(1) == 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= p
    if keep.sum() == 0:
        keep[0] = True
    return edge_index[:, keep]


# ─────────────────────────────────────────────────────────────────────────────
# Structural features
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_pe(edge_index: torch.Tensor,
                 num_nodes:  int,
                 k:          int = 8,
                 device:     torch.device = None) -> torch.Tensor:
    """Return the first k non-trivial eigenvectors of the normalized Laplacian."""
    N      = num_nodes
    device = device or edge_index.device
    row, col = edge_index[0], edge_index[1]

    deg          = pyg_degree(row, N, dtype=torch.float).to(device)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), deg.new_zeros(()))

    A = torch.zeros(N, N, device=device)
    A[row, col] = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    L = torch.eye(N, device=device) - A
    L.diagonal().add_(1e-6)

    try:
        _, vecs = torch.linalg.eigh(L)
    except torch.linalg.LinAlgError:
        return torch.zeros(N, k, device=device)

    pe = vecs[:, 1 : k + 1]
    if pe.size(1) < k:
        pe = torch.cat([pe, torch.zeros(N, k - pe.size(1), device=device)], dim=1)

    signs = pe[pe.abs().argmax(dim=0), torch.arange(k, device=device)].sign()
    signs[signs == 0] = 1.0
    return pe * signs.unsqueeze(0)


def build_node_features(data, lpe_dim: int, edge_index_for_msg: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """
    Concatenate raw node features, normalized degree, and Laplacian PEs.
    """
    deg   = pyg_degree(edge_index_for_msg[0], data.num_nodes,
                       dtype=torch.float).to(device)
    deg_f = (deg / deg.max().clamp(min=1.0)).unsqueeze(-1)

    lpe = laplacian_pe(data.edge_index, data.num_nodes, k=lpe_dim, device=device)

    return torch.cat([data.x, deg_f, lpe], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(dataset_name, max_nodes, batch_size, seed):
    ds = TUDataset(
        root="./data/TUDataset",
        name=dataset_name,
        use_node_attr=True,
        pre_filter=lambda d: d.num_nodes <= max_nodes,
    )
    if len(ds) < 10:
        raise RuntimeError(f"Only {len(ds)} graphs after filtering.")
    if ds.num_features == 0:
        raise RuntimeError("Dataset has no node features.")

    g    = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds), generator=g)
    ds   = ds[perm]
    n    = len(ds)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)

    tl = DataLoader(ds[:n_tr],          batch_size=batch_size, shuffle=True)
    vl = DataLoader(ds[n_tr:n_tr+n_va], batch_size=batch_size)
    el = DataLoader(ds[n_tr+n_va:],     batch_size=batch_size)
    return tl, vl, el, ds.num_features, ds.num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GraphEGNN(nn.Module):
    """
    Graph-level EGNN classifier with attention pooling.
    """

    def __init__(self, in_node_dim, hidden_dim, n_layers, n_classes,
                 dropout=0.4, lpe_dim=8, drop_edge_p=0.15):
        super().__init__()
        self.lpe_dim     = lpe_dim
        self.drop_edge_p = drop_edge_p

        self.egnn = EGNN(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            edge_attr_dim=0,
            update_coords=False,   # invariant mode for graph classification
            infer_edges=True,      # soft edge weights (Section 3.3)
            vel_input=False,
            batch_norm=True,
        )

        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate_nn)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def encode_graph(self, data):
        device = data.x.device
        ei = data.edge_index

        if self.training and self.drop_edge_p > 0:
            ei = drop_edge(ei, self.drop_edge_p)

        h = build_node_features(data, self.lpe_dim, ei, device)

        x = torch.zeros(data.num_nodes, 3, device=device)

        h_out, _ = self.egnn(h, x, ei)

        # Attention pooling: one vector per graph
        return self.pool(h_out, data.batch)

    def forward(self, data, return_embedding=False):
        g      = self.encode_graph(data)
        logits = self.head(g)
        if return_embedding:
            return logits, g
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Supervised contrastive loss
# ─────────────────────────────────────────────────────────────────────────────

def supervised_contrastive_loss(embeddings, labels, temperature=0.5):
    """Supervised contrastive loss on graph embeddings."""
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    z = nn.functional.normalize(embeddings, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    labels    = labels.view(-1)
    pos_mask  = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
    pos_mask  = pos_mask & ~self_mask

    exp_sim  = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12))

    pos_count = pos_mask.sum(dim=1)
    valid     = pos_count > 0
    if not valid.any():
        return embeddings.new_zeros(())

    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, optimizer=None, label_smooth=0.0,
              metric_weight=0.0, temperature=0.5):
    is_train  = optimizer is not None
    model.train() if is_train else model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    total_loss = total_correct = total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for data in loader:
            data = data.to(device)
            logits, emb = model(data, return_embedding=True)
            y           = data.y.view(-1)
            cls_loss    = criterion(logits, y)
            metric_loss = supervised_contrastive_loss(emb, y, temperature=temperature)
            loss        = cls_loss + metric_weight * metric_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss    += loss.item() * y.size(0)
            total_correct += (logits.argmax(-1) == y).sum().item()
            total         += y.size(0)

    return total_loss / total, total_correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:        {device}")
    print(f"DropEdge p:    {args.drop_edge}")
    print(f"Label smooth:  {args.label_smooth}")
    print(f"Dropout:       {args.dropout}")
    print(f"Weight decay:  {args.wd}")
    print(f"Metric weight: {args.metric_weight}")
    print(f"Temperature:   {args.temperature}")
    print(f"LPE dim:       {args.lpe_dim} (now fed into h)")
    print(f"Pooling:       AttentionalAggregation (replaces mean+max)")

    tl, vl, el, raw_feat, n_classes = build_loaders(
        args.dataset, args.max_nodes, args.bs, args.seed)

    # in_node_dim = raw_feat + 1 (degree) + lpe_dim
    in_node_dim = raw_feat + 1 + args.lpe_dim
    print(f"Dataset: {args.dataset}  feat={raw_feat}→{in_node_dim}  classes={n_classes}")

    model = GraphEGNN(
        in_node_dim=in_node_dim,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_classes=n_classes,
        dropout=args.dropout,
        lpe_dim=args.lpe_dim,
        drop_edge_p=args.drop_edge,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, tl, device, optimizer,
                                    label_smooth=args.label_smooth,
                                    metric_weight=args.metric_weight,
                                    temperature=args.temperature)
        va_loss, va_acc = run_epoch(model, vl, device)
        scheduler.step()

        improved = va_acc > best_val
        if improved:
            best_val = va_acc
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            gap = tr_acc - va_acc
            print(f"Epoch {epoch:4d} | "
                  f"train {tr_loss:.4f}/{tr_acc:.4f} | "
                  f"val {va_loss:.4f}/{va_acc:.4f} | "
                  f"gap {gap:+.4f} | best {best_val:.4f}"
                  + (" ✓" if improved else ""))

    torch.save(model.state_dict(), "final_model.pt")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, el, device)
    print(f"\nSaved final_model.pt + best_model.pt")
    print(f"Test loss: {te_loss:.4f} | Test acc: {te_acc:.4f} (best val: {best_val:.4f})")


if __name__ == "__main__":
    main()
