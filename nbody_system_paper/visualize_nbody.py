import argparse
import os

import matplotlib.pyplot as plt
import torch

from dataset_nbody import NBodyDataset
from model import GNN, EGNN_vel, RF_vel, Baseline, Linear, Linear_dynamics


def get_args():
    p = argparse.ArgumentParser(description="Visualize N-body trajectories and predictions")
    p.add_argument("--model", type=str, default="egnn_vel",
                   choices=["gnn", "baseline", "linear", "linear_vel", "egnn_vel", "rf_vel"])
    p.add_argument("--ckpt", type=str, default=None,
                   help="Optional checkpoint to load. If omitted, only GT trajectory is shown.")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--partition", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--nf", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--norm_diff", type=eval, default=False)
    p.add_argument("--tanh", type=eval, default=False)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--frame_0", type=int, default=3000)
    p.add_argument("--frame_T", type=int, default=4000)
    p.add_argument("--n_particles", type=int, default=5)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--save", type=str, default=None,
                   help="Optional output path for the figure.")
    return p.parse_args()


def build_model(args, device):
    if args.model == "gnn":
        return GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    if args.model == "egnn_vel":
        return EGNN_vel(
            in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf,
            device=device, n_layers=args.n_layers, recurrent=True,
            norm_diff=args.norm_diff, tanh=args.tanh
        )
    if args.model == "baseline":
        return Baseline(device=device)
    if args.model == "linear_vel":
        return Linear_dynamics(device=device)
    if args.model == "linear":
        return Linear(6, 3, device=device)
    if args.model == "rf_vel":
        return RF_vel(hidden_nf=args.nf, edge_attr_nf=2, device=device, n_layers=args.n_layers)
    raise ValueError("Wrong model")


def get_edges(n_nodes, device):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]


def predict(model, sample, device, model_name):
    loc = sample["loc_0"].to(device)
    vel = sample["vel_0"].to(device)
    edge_attr = sample["edge_attr"].to(device)
    edges = get_edges(loc.size(0), device)

    with torch.no_grad():
        if model_name == "gnn":
            nodes = torch.cat([loc, vel], dim=1)
            return model(nodes, edges, edge_attr)
        if model_name == "egnn_vel":
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1)
            return model(nodes, loc, edges, vel, edge_attr)
        if model_name == "baseline":
            return model(loc)
        if model_name == "linear":
            return model(torch.cat([loc, vel], dim=1))
        if model_name == "linear_vel":
            return model(loc, vel)
        if model_name == "rf_vel":
            rows, cols = edges
            vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1)
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1)
            return model(vel_norm, loc, edges, vel, edge_attr)
    raise ValueError("Wrong model")


def plot_sample(sample, pred=None, title="N-body sample", save_path=None):
    traj = sample["loc_traj"].cpu()
    loc0 = sample["loc_0"].cpu()
    locT = sample["loc_T"].cpu()
    pred = pred.cpu() if pred is not None else None

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for i in range(traj.size(1)):
        ax1.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], linewidth=1.2)
        ax1.scatter(loc0[i, 0], loc0[i, 1], loc0[i, 2], marker="o", s=35)
        ax1.scatter(locT[i, 0], locT[i, 1], locT[i, 2], marker="x", s=45)

    ax1.set_title("Ground-truth trajectory")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    ax2.scatter(loc0[:, 0], loc0[:, 1], loc0[:, 2], marker="o", s=45, label="input t0")
    ax2.scatter(locT[:, 0], locT[:, 1], locT[:, 2], marker="x", s=55, label="target tT")
    if pred is not None:
        ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], marker="^", s=55, label="prediction")
        for i in range(pred.size(0)):
            ax2.plot(
                [locT[i, 0], pred[i, 0]],
                [locT[i, 1], pred[i, 1]],
                [locT[i, 2], pred[i, 2]],
                linestyle="--",
                linewidth=0.8,
            )
    ax2.set_title("Input / target / prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NBodyDataset(
        partition=args.partition,
        max_samples=args.max_samples,
        total_steps=args.total_steps,
        frame_0=args.frame_0,
        frame_T=args.frame_T,
        n_particles=args.n_particles,
        dt=args.dt,
        seed={"train": 0, "val": 1, "test": 2}[args.partition],
    )
    sample = dataset.get_full_sample(args.sample_idx)

    pred = None
    if args.ckpt is not None:
        model = build_model(args, device)
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()
        pred = predict(model, sample, device, args.model)

    plot_sample(
        sample,
        pred=pred,
        title=f"{args.model} | {args.partition} sample {args.sample_idx}",
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
