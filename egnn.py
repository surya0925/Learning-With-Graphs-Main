"""
E(n) Equivariant Graph Neural Networks — full paper implementation
Paper: https://arxiv.org/abs/2102.09844

Implements every modelling section:

  Section 3   — Core EGCL
    mij     = φe(hi, hj, ||xi−xj||², aij)          [Eq. 3]
    xi_new  = xi + C·Σ_{j≠i} (xi−xj)·φx(mij)      [Eq. 4]
    mi      = Σ_{j≠i} mij                           [Eq. 5]
    hi_new  = φh(hi, mi)                            [Eq. 6]

  Section 3.1 — E(n) equivariance
    Architectural guarantee; verified in test_equivariance.py

  Section 3.2 — Velocity / momentum extension
    v_new   = φv(hi)·v_init + C·Σ_{j≠i} (xi−xj)·φx(mij)  [Eq. 7]
    xi_new  = xi + v_new

  Section 3.3 — Edge inference (soft edge weights)
    eij     = φinf(mij)   ∈ [0,1]                  [Eq. 8]
    mij     = eij · mij    (gate before aggregation)

MLP architectures from Appendix C:
  φe  : Linear → Swish → Linear → Swish   (two activations)
  φx  : Linear → Swish → Linear            (one activation, no final act)
  φv  : Linear → Swish → Linear            (one activation)
  φh  : Linear → Swish → Linear + skip(hi) (residual from hi, not concat)
  φinf: Linear → Sigmoid
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# MLP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mlp_two_act(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """φe style: Linear → Swish → Linear → Swish  (Appendix C)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.SiLU(),
    )


def _mlp_one_act(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """φx / φv style: Linear → Swish → Linear  (no final activation)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class _NodeMLP(nn.Module):
    """
    φh: [hi, mi] → hi_new   (Appendix C).

    Residual is from hi (size node_dim), NOT from the concatenated input.
    This matches:  hi_new = hi + MLP([hi, mi])
    """
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, h: torch.Tensor, m_i: torch.Tensor) -> torch.Tensor:
        return h + self.net(torch.cat([h, m_i], dim=-1))   # residual from h


# ─────────────────────────────────────────────────────────────────────────────
# EGCL — one equivariant graph convolutional layer
# ─────────────────────────────────────────────────────────────────────────────

class EGCL(nn.Module):
    """
    One EGNN layer.  Supports all paper configurations:

    Args
    ----
    node_dim      : dimension of node features h
    hidden_dim    : width of all internal MLPs
    edge_attr_dim : dimension of optional edge attributes aij
    update_coords : if True  → equivariant mode (Eq. 4 or 7)
                    if False → invariant mode (QM9 / classification)
    infer_edges   : if True  → add φinf gating (Section 3.3, Eq. 8)
    vel_input     : if True  → use velocity extension (Section 3.2, Eq. 7)
    """

    def __init__(self, node_dim: int, hidden_dim: int,
                 edge_attr_dim: int = 0,
                 update_coords: bool = True,
                 infer_edges:   bool = False,
                 vel_input:     bool = False,
                 batch_norm:    bool = False):
        super().__init__()
        self.update_coords = update_coords
        self.infer_edges   = infer_edges
        self.vel_input     = vel_input

        # ── φe: edge message  [Eq. 3, Appendix C — two Swish activations] ─
        edge_in = node_dim * 2 + 1 + edge_attr_dim
        self.phi_e = _mlp_two_act(edge_in, hidden_dim, hidden_dim)

        # ── φx: coordinate weight scalar  [Eq. 4, Appendix C] ──────────────
        if update_coords:
            self.phi_x = _mlp_one_act(hidden_dim, hidden_dim, 1)

        # ── φv: velocity scalar gate  [Eq. 7, Section 3.2] ─────────────────
        if vel_input:
            self.phi_v = _mlp_one_act(node_dim, hidden_dim, 1)

        # ── φinf: soft edge weight  [Eq. 8, Section 3.3] ───────────────────
        if infer_edges:
            self.phi_inf = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        # ── φh: node update  [Eq. 6, Appendix C — residual from hi] ────────
        self.phi_h = _NodeMLP(node_dim, hidden_dim)
        # Optional BatchNorm after node update
        self.bn = nn.BatchNorm1d(node_dim) if batch_norm else nn.Identity()

    # -------------------------------------------------------------------------
    def forward(self,
                h:          torch.Tensor,            # (N, node_dim)
                x:          torch.Tensor,            # (N, n_dims)
                edge_index: torch.Tensor,            # (2, E)
                edge_attr:  torch.Tensor | None = None,  # (E, edge_attr_dim)
                v_init:     torch.Tensor | None = None,  # (N, n_dims) Section 3.2
                ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        N = h.size(0)

        # ── Eq. 3: edge messages ─────────────────────────────────────────────
        diff    = x[src] - x[dst]                         # (E, n_dims)
        sq_dist = (diff ** 2).sum(dim=-1, keepdim=True)   # (E, 1)

        parts = [h[src], h[dst], sq_dist]
        if edge_attr is not None:
            parts.append(edge_attr)
        m_ij = self.phi_e(torch.cat(parts, dim=-1))       # (E, hidden_dim)

        # ── Eq. 8: edge inference  [Section 3.3] ─────────────────────────────
        if self.infer_edges:
            e_ij = self.phi_inf(m_ij)                     # (E, 1)  ∈ [0,1]
            m_ij = e_ij * m_ij                            # soft-gate messages

        # ── Eqs. 4 / 7: coordinate update ───────────────────────────────────
        if self.update_coords:
            weights       = self.phi_x(m_ij)              # (E, 1)
            weighted_diff = diff * weights                 # (E, n_dims)
            agg_diff      = torch.zeros_like(x)
            agg_diff.scatter_add_(
                0, dst.unsqueeze(-1).expand_as(weighted_diff), weighted_diff)
            C = 1.0 / (N - 1)

            if self.vel_input and v_init is not None:
                # Eq. 7 — velocity variant (Section 3.2):
                #   v_new   = φv(hi) · v_init + C·Σ (xi−xj)·φx(mij)
                #   xi_new  = xi + v_new
                v_new = self.phi_v(h) * v_init + C * agg_diff   # (N, n_dims)
                x = x + v_new
            else:
                # Eq. 4 — standard coordinate update:
                #   xi_new = xi + C·Σ (xi−xj)·φx(mij)
                x = x + C * agg_diff

        # ── Eq. 5: aggregate messages ─────────────────────────────────────────
        m_i = torch.zeros(N, m_ij.size(-1), device=h.device, dtype=h.dtype)
        m_i.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij)

        # ── Eq. 6: node update (with residual from hi) ────────────────────────
        h = self.bn(self.phi_h(h, m_i))

        return h, x


# ─────────────────────────────────────────────────────────────────────────────
# EGNN — stack of EGCLs
# ─────────────────────────────────────────────────────────────────────────────

class EGNN(nn.Module):
    """
    Full EGNN model.  All paper configurations:

      update_coords=True,  vel_input=False  → Eq. 4  (basic equivariant)
      update_coords=True,  vel_input=True   → Eq. 7  (momentum, Section 3.2)
      update_coords=False                   → invariant (QM9 / classification)
      infer_edges=True                      → soft edges (Section 3.3)
    """

    def __init__(self,
                 in_node_dim:  int,
                 hidden_dim:   int,
                 out_dim:      int,
                 n_layers:     int  = 4,
                 edge_attr_dim:int  = 0,
                 update_coords:bool = True,
                 infer_edges:  bool = False,
                 vel_input:    bool = False,
                 batch_norm:   bool = False):
        super().__init__()
        self.vel_input = vel_input

        self.embedding = nn.Linear(in_node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGCL(hidden_dim, hidden_dim,
                 edge_attr_dim=edge_attr_dim,
                 update_coords=update_coords,
                 infer_edges=infer_edges,
                 vel_input=vel_input,
                 batch_norm=batch_norm)
            for _ in range(n_layers)
        ])

        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self,
                h:          torch.Tensor,
                x:          torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor | None = None,
                v_init:     torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h          : (N, in_node_dim)
        x          : (N, n_dims)
        edge_index : (2, E)
        edge_attr  : (E, edge_attr_dim) or None
        v_init     : (N, n_dims) or None  — initial velocity (Section 3.2)

        Returns
        -------
        h_out : (N, out_dim)   E(n)-invariant node embeddings
        x_out : (N, n_dims)    E(n)-equivariant coordinates
        """
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index,
                         edge_attr=edge_attr,
                         v_init=v_init if self.vel_input else None)
        return self.output(h), x


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def fully_connected_edges(N: int, device=None) -> torch.Tensor:
    """Returns (2, N*(N-1)) edge_index for a complete graph (no self-loops)."""
    idx = torch.arange(N, device=device)
    src, dst = torch.meshgrid(idx, idx, indexing='ij')
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)