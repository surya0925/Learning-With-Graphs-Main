import argparse
import json
import os
import random

import torch
from torch import nn, optim

from dataset_nbody import NBodyDataset
from model import GNN, EGNN_vel, RF_vel, Baseline, Linear, Linear_dynamics


parser = argparse.ArgumentParser(description="Paper-style N-body training")
parser.add_argument("--exp_name", type=str, default="exp_1")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--test_interval", type=int, default=5)
parser.add_argument("--outf", type=str, default="logs")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--nf", type=int, default=64)
parser.add_argument(
    "--model",
    type=str,
    default="egnn_vel",
    choices=["gnn", "baseline", "linear", "linear_vel", "egnn_vel", "rf_vel"],
)
parser.add_argument("--attention", type=int, default=0)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--max_training_samples", type=int, default=3000)
parser.add_argument("--sweep_training", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=1e-12)
parser.add_argument("--norm_diff", type=eval, default=True)
parser.add_argument("--tanh", type=eval, default=True)
parser.add_argument("--clip_grad", type=float, default=1.0)
parser.add_argument("--total_steps", type=int, default=5000)
parser.add_argument("--frame_0", type=int, default=3000)
parser.add_argument("--frame_T", type=int, default=4000)
parser.add_argument("--n_particles", type=int, default=5)
parser.add_argument("--dt", type=float, default=1e-3)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_velocity_attr(loc, vel, rows, cols):
    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff / norm.clamp(min=1e-8)
    va = torch.sum(vel[rows] * u, dim=1).unsqueeze(1)
    return va


def build_model():
    if args.model == "gnn":
        return GNN(
            input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers,
            device=device, recurrent=True
        )
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
        return RF_vel(
            hidden_nf=args.nf, edge_attr_nf=2, device=device,
            act_fn=nn.SiLU(), n_layers=args.n_layers
        )
    raise ValueError("Wrong model specified")


def build_loaders():
    dataset_train = NBodyDataset(
        partition="train",
        max_samples=args.max_training_samples,
        total_steps=args.total_steps,
        frame_0=args.frame_0,
        frame_T=args.frame_T,
        n_particles=args.n_particles,
        dt=args.dt,
        seed=0,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyDataset(
        partition="val",
        max_samples=2000,
        total_steps=args.total_steps,
        frame_0=args.frame_0,
        frame_T=args.frame_T,
        n_particles=args.n_particles,
        dt=args.dt,
        seed=1,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    dataset_test = NBodyDataset(
        partition="test",
        max_samples=2000,
        total_steps=args.total_steps,
        frame_0=args.frame_0,
        frame_T=args.frame_T,
        n_particles=args.n_particles,
        dt=args.dt,
        seed=2,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    return loader_train, loader_val, loader_test


def train(model, optimizer, epoch, loader, backprop=True):
    model.train() if backprop else model.eval()
    res = {"loss": 0.0, "counter": 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]
        optimizer.zero_grad()

        if args.model == "gnn":
            nodes = torch.cat([loc, vel], dim=1)
            loc_pred = model(nodes, edges, edge_attr)
        elif args.model == "egnn_vel":
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
        elif args.model == "baseline":
            loc_pred = model(loc)
        elif args.model == "linear":
            loc_pred = model(torch.cat([loc, vel], dim=1))
        elif args.model == "linear_vel":
            loc_pred = model(loc, vel)
        elif args.model == "rf_vel":
            rows, cols = edges
            vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
            loc_pred = model(vel_norm, loc.detach(), edges, vel, edge_attr)
        else:
            raise ValueError("Wrong model")

        if not torch.isfinite(loc_pred).all():
            raise RuntimeError(
                f"Non-finite prediction detected in {loader.dataset.partition} "
                f"epoch {epoch} batch {batch_idx}."
            )
        loss = loss_mse(loc_pred, loc_end)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected in {loader.dataset.partition} "
                f"epoch {epoch} batch {batch_idx}."
            )
        if backprop:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size

        if batch_idx % args.log_interval == 0 and backprop and batch_idx == 0:
            print(
                f"{loader.dataset.partition} epoch {epoch} "
                f"batch {batch_idx} loss: {loss.item():.6f}"
            )

    avg = res["loss"] / res["counter"]
    prefix = "" if backprop else "==> "
    print(f"{prefix}{loader.dataset.partition} epoch {epoch} avg loss: {avg:.5f}")
    return avg


def main():
    set_seed(args.seed)
    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(os.path.join(args.outf, args.exp_name), exist_ok=True)

    loader_train, loader_val, loader_test = build_loaders()
    model = build_model()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {"epochs": [], "losses": []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0

    for epoch in range(args.epochs):
        train(model, optimizer, epoch, loader_train)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results["epochs"].append(epoch)
            results["losses"].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.outf, args.exp_name, "best_model.pt"))
            print(
                "*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
                % (best_val_loss, best_test_loss, best_epoch)
            )

        with open(os.path.join(args.outf, args.exp_name, "losses.json"), "w") as outfile:
            outfile.write(json.dumps(results, indent=4))

    torch.save(model.state_dict(), os.path.join(args.outf, args.exp_name, "final_model.pt"))
    return best_val_loss, best_test_loss, best_epoch


def main_sweep():
    training_samples = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000, 50000]
    n_epochs = [2000, 2000, 4000, 5000, 8000, 10000, 8000, 6000, 4000, 2000]
    if args.model == "egnn_vel":
        n_epochs = [4000, 4000, 2000, 2000, 2000, 1500, 1500, 1500, 1000, 1000]

    results = {"tr_samples": [], "test_loss": [], "best_epochs": []}
    for epochs, tr_samples in zip(n_epochs, training_samples):
        args.epochs = epochs
        args.max_training_samples = tr_samples
        args.test_interval = max(int(10000 / tr_samples), 1)
        best_val_loss, best_test_loss, best_epoch = main()
        results["tr_samples"].append(tr_samples)
        results["best_epochs"].append(best_epoch)
        results["test_loss"].append(best_test_loss)
        with open(os.path.join(args.outf, "sweep_results.json"), "w") as outfile:
            outfile.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    if args.sweep_training:
        main_sweep()
    else:
        main()
