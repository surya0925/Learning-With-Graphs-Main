import torch
from torch.utils.data import Dataset


def simulate_trajectory(pos, vel, charges, total_steps=5000, dt=1e-3,
                        interaction_strength=1.0, max_force=None):
    positions = []
    velocities = []
    n_nodes = pos.size(0)
    if max_force is None:
        max_force = 0.1 / dt

    for _ in range(total_steps):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist2 = (diff ** 2).sum(-1) + 1e-8
        dist = dist2.sqrt()
        dist3 = dist2 * dist
        qprod = charges.unsqueeze(1) * charges.unsqueeze(0)
        force = interaction_strength * (qprod.unsqueeze(-1) / dist3.unsqueeze(-1)) * diff
        force = torch.clamp(force, min=-max_force, max=max_force)
        mask = 1 - torch.eye(n_nodes, device=pos.device, dtype=pos.dtype)
        acc = (force * mask.unsqueeze(-1)).sum(1)

        vel = vel + dt * acc
        pos = pos + dt * vel
        positions.append(pos.clone())
        velocities.append(vel.clone())

    return torch.stack(positions), torch.stack(velocities)


class NBodyDataset(Dataset):
    """
    Charged-particle N-body dataset for the dynamical systems experiment.

    Defaults:
    - 5 particles in 3D
    - 5000 simulated steps
    - input frame 3000
    - target frame 4000
    - train/val/test sizes 3000/2000/2000
    """

    def __init__(
        self,
        partition="train",
        max_samples=3000,
        total_steps=5000,
        frame_0=3000,
        frame_T=4000,
        n_particles=5,
        dt=1e-3,
        loc_std=1.0,
        vel_norm=0.5,
        interaction_strength=1.0,
        seed=0,
    ):
        self.partition = partition
        self.max_samples = int(max_samples)
        self.total_steps = total_steps
        self.frame_0 = frame_0
        self.frame_T = frame_T
        self.n_particles = n_particles
        self.dt = dt
        self.loc_std = loc_std * (float(n_particles) / 5.0) ** (1.0 / 3.0)
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.seed = seed
        self.data, self.edges = self.load()

    def load(self):
        rng = torch.Generator().manual_seed(self.seed)
        init_pos = torch.randn(self.max_samples, self.n_particles, 3, generator=rng) * self.loc_std
        init_vel = torch.randn(self.max_samples, self.n_particles, 3, generator=rng)
        init_vel_norm = init_vel.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        init_vel = init_vel * (self.vel_norm / init_vel_norm)
        charges = (
            torch.randint(0, 2, (self.max_samples, self.n_particles), generator=rng) * 2 - 1
        ).float()

        loc_all = []
        vel_all = []
        edge_attr_all = []
        rows, cols = [], []
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j:
                    rows.append(i)
                    cols.append(j)

        print(
            f"[{self.partition}] Simulating {self.max_samples} trajectories "
            f"(steps={self.total_steps}, frame_0={self.frame_0}, frame_T={self.frame_T})...",
            flush=True,
        )
        for idx in range(self.max_samples):
            loc, vel = simulate_trajectory(
                init_pos[idx], init_vel[idx], charges[idx],
                total_steps=self.total_steps,
                dt=self.dt,
                interaction_strength=self.interaction_strength,
            )
            loc_all.append(loc)
            vel_all.append(vel)
            edge_attr = []
            for i in range(self.n_particles):
                for j in range(self.n_particles):
                    if i != j:
                        edge_attr.append(charges[idx, i] * charges[idx, j])
            edge_attr_all.append(torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1))
            if (idx + 1) % max(1, self.max_samples // 10) == 0:
                print(f"  {idx + 1}/{self.max_samples}", flush=True)

        loc_all = torch.stack(loc_all)          # (B, T, N, 3)
        vel_all = torch.stack(vel_all)          # (B, T, N, 3)
        edge_attr_all = torch.stack(edge_attr_all)
        charges = charges.unsqueeze(-1)
        print(f"[{self.partition}] Done.", flush=True)
        return (loc_all, vel_all, edge_attr_all, charges), [rows, cols]

    def __getitem__(self, idx):
        loc, vel, edge_attr, charges = self.data
        return (
            loc[idx, self.frame_0],
            vel[idx, self.frame_0],
            edge_attr[idx],
            charges[idx],
            loc[idx, self.frame_T],
        )

    def get_full_sample(self, idx):
        """Return the full trajectory together with the supervised frames."""
        loc, vel, edge_attr, charges = self.data
        return {
            "loc_traj": loc[idx],
            "vel_traj": vel[idx],
            "edge_attr": edge_attr[idx],
            "charges": charges[idx],
            "loc_0": loc[idx, self.frame_0],
            "vel_0": vel[idx, self.frame_0],
            "loc_T": loc[idx, self.frame_T],
            "frame_0": self.frame_0,
            "frame_T": self.frame_T,
        }

    def __len__(self):
        return self.data[0].size(0)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        return [torch.cat(rows), torch.cat(cols)]
