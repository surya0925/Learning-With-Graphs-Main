import torch
from torch import nn

from egnn_clean import unsorted_segment_sum, unsorted_segment_mean


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0,
                 act_fn=nn.SiLU(), attention=False, recurrent=True):
        super().__init__()
        self.attention = attention
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_in = torch.cat([x[row], x[col]], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        edge_feat = self.edge_mlp(edge_in)
        if self.attention:
            edge_feat = edge_feat * self.att_mlp(torch.abs(x[row] - x[col]))
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        out = self.node_mlp(torch.cat([x, agg], dim=1))
        if self.recurrent:
            out = out + x
        return out, edge_feat


class E_GCL_vel(nn.Module):
    """
    Velocity-aware equivariant layer following the original EGNN N-body codepath.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 act_fn=nn.SiLU(), recurrent=True, coords_weight=1.0,
                 attention=False, norm_diff=False, tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.norm_diff:
            coord_diff = coord_diff / (torch.sqrt(radial) + 1.0)
        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        return self.edge_mlp(out)

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100.0, max=100.0)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return coord + agg * self.coords_weight

    def node_model(self, h, edge_index, edge_feat):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        out = self.node_mlp(torch.cat([h, agg], dim=1))
        if self.recurrent:
            out = h + out
        return out

    def forward(self, h, edge_index, coord, vel, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        coord = coord + self.coord_mlp_vel(h) * vel
        h = self.node_model(h, edge_index, edge_feat)
        return h, coord, edge_attr


class GCL_rf_vel(nn.Module):
    def __init__(self, nf=64, edge_attr_nf=0, act_fn=nn.SiLU(), coords_weight=1.0):
        super().__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1),
        )
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(
            nn.Linear(1 + edge_attr_nf, nf),
            act_fn,
            layer,
            nn.Tanh(),
        )

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        x_diff = x[row] - x[col]
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_out = self.phi(torch.cat([radial, edge_attr], dim=1))
        edge_m = x_diff * e_out
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x = x + agg * self.coords_weight
        x = x + vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, device="cpu", act_fn=nn.SiLU(),
                 n_layers=4, attention=0, recurrent=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                    edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent),
            )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3),
        )
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)

    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        for i in range(self.n_layers):
            h, _ = self._modules[f"gcl_{i}"](h, edges, edge_attr=edge_attr)
        return self.decoder(h)


class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device="cpu",
                 act_fn=nn.SiLU(), n_layers=4, recurrent=False,
                 norm_diff=False, tanh=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL_vel(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
        self.to(self.device)

    def forward(self, h, x, edges, vel, edge_attr):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, edges, x, vel, edge_attr=edge_attr)
        return x


class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device="cpu",
                 act_fn=nn.SiLU(), n_layers=4):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn),
            )
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(self.n_layers):
            x, _ = self._modules[f"gcl_{i}"](x, vel_norm, vel, edges, edge_attr)
        return x


class Baseline(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.device = device
        self.to(self.device)

    def forward(self, loc):
        return loc


class Linear(nn.Module):
    def __init__(self, input_nf, output_nf, device="cpu"):
        super().__init__()
        self.linear = nn.Linear(input_nf, output_nf)
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return self.linear(input)


class Linear_dynamics(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.time = nn.Parameter(torch.ones(1) * 0.7)
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v * self.time
