"""
Graph similarity using trained EGNN embeddings.

The preprocessing matches train.py:
- Laplacian positional encodings
- degree feature appended to node features

Usage:
    python test_similarity.py
    python test_similarity.py --ckpt final_model.pt --seed 42
"""

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from train import GraphEGNN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=str, default="final_model.pt")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--lpe_dim", type=int, default=8)
    return p.parse_args()


def infer_arch(state_dict) -> Tuple[int, int, int, int]:
    in_node_dim = state_dict["egnn.embedding.weight"].shape[1]
    hidden_dim  = state_dict["egnn.embedding.weight"].shape[0]
    head_linear_keys = []
    for k, v in state_dict.items():
        if k.startswith("head.") and k.endswith(".weight") and v.ndim == 2:
            parts = k.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                head_linear_keys.append((int(parts[1]), k))
    if not head_linear_keys:
        raise RuntimeError("Could not infer classifier output size from checkpoint head.")
    last_linear_key = sorted(head_linear_keys, key=lambda x: x[0])[-1][1]
    n_classes = state_dict[last_linear_key].shape[0]
    layer_ids   = {int(k.split(".")[2]) for k in state_dict
                   if k.startswith("egnn.layers.") and k.split(".")[2].isdigit()}
    n_layers    = max(layer_ids) + 1 if layer_ids else 0
    return in_node_dim, hidden_dim, n_layers, n_classes


# ─────────────────────────────────────────────────────────────────────────────
# Graph generators
# ─────────────────────────────────────────────────────────────────────────────

def random_graph(n: int, raw_feat_dim: int, g: torch.Generator) -> Data:
    """Generate a random graph with one-hot node features."""
    idx = torch.randint(0, raw_feat_dim, (n,), generator=g)
    x   = torch.zeros(n, raw_feat_dim)
    x[torch.arange(n), idx] = 1.0
    p   = min(0.3, max(0.1, 4.0 / max(1, n)))
    s, d = torch.triu_indices(n, n, offset=1)
    keep = torch.rand(s.size(0), generator=g) < p
    s, d = s[keep], d[keep]
    if s.numel() == 0:
        s = torch.arange(n - 1); d = torch.arange(1, n)
    ei = torch.cat([torch.stack([s, d]), torch.stack([d, s])], dim=1).long()
    return Data(x=x, edge_index=ei)


def permute_graph(data: Data, g: torch.Generator) -> Data:
    n    = data.num_nodes
    perm = torch.randperm(n, generator=g)
    inv  = torch.empty_like(perm); inv[perm] = torch.arange(n)
    return Data(x=data.x[perm], edge_index=inv[data.edge_index])


def one_node_diff(data: Data, g: torch.Generator) -> Data:
    x  = data.x.clone()
    n, fd = x.shape
    i = torch.randint(0, n, (1,), generator=g).item()
    f = torch.randint(0, fd, (1,), generator=g).item()
    x[i].zero_(); x[i, f] = 1.0
    return Data(x=x, edge_index=data.edge_index.clone())


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────

def embed(model: GraphEGNN, graph: Data, device: torch.device,
          lpe_dim: int = 3) -> torch.Tensor:
    data       = graph.clone().to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        return model.encode_graph(data).squeeze(0)


def cosine_sim(model, a, b, device, lpe_dim):
    ea = embed(model, a, device, lpe_dim)
    eb = embed(model, b, device, lpe_dim)
    return F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    g      = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device)
    in_node_dim, hidden_dim, n_layers, n_classes = infer_arch(state)
    raw_feat_dim = in_node_dim - 1 - args.lpe_dim
    if raw_feat_dim <= 0:
        raise RuntimeError(
            f"Invalid raw feature dim inferred: {raw_feat_dim}. "
            f"Check --lpe_dim (currently {args.lpe_dim}) for this checkpoint."
        )

    model = GraphEGNN(in_node_dim, hidden_dim, n_layers, n_classes,
                      lpe_dim=args.lpe_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    base  = random_graph(n=20, raw_feat_dim=raw_feat_dim, g=g)
    perm  = permute_graph(base, g)
    diff1 = one_node_diff(base, g)
    other = random_graph(n=20, raw_feat_dim=raw_feat_dim, g=g)
    big   = random_graph(n=30, raw_feat_dim=raw_feat_dim, g=g)

    s1 = cosine_sim(model, base, perm,  device, args.lpe_dim)
    s2 = cosine_sim(model, base, diff1, device, args.lpe_dim)
    s3 = cosine_sim(model, base, other, device, args.lpe_dim)
    s4 = cosine_sim(model, base, big,   device, args.lpe_dim)

    print("Graph Similarity Scores (cosine)")
    print(f"1) Equivariant pair (same graph, permuted)    : {s1:.6f}  (expect ~1.000)")
    print(f"2) Similar pair (one node different)          : {s2:.6f}  (expect ~0.95-0.98)")
    print(f"3) Different pair (same size: 20 vs 20)       : {s3:.6f}  (expect ~0.80-0.94)")
    print(f"4) Different pair (diff size: 20 vs 30)       : {s4:.6f}  (expect ~0.80-0.94)")
    spread = s1 - min(s3, s4)
    print(f"\nSpread (case 1 vs worst): {spread:.4f}")
    print("OK — model discriminates graphs." if spread > 0.1
          else "Low spread — try more epochs or larger hidden dim.")


if __name__ == "__main__":
    main()
