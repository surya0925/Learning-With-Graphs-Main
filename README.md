# Learning With Graphs — Project Report

> A research-focused implementation of E(n) Equivariant Graph Neural Networks (EGNN) and supporting experiments for graph-level tasks and N-body dynamical systems. This repository collects code, notebooks, model checkpoints, and paper PDFs used to reproduce and explore EGNN variants from the literature.

Table of contents
- What this project is
- Background & motivation
- Highlights and contributions
- How the repository is organized (annotated)
- How the pieces fit together (runtime / data flow)
- Quickstart — run training and evaluation
- Reproducing experiments (Similarity Model & N-Body)
- Data, model checkpoints, and tests
- Recommended environment & dependencies
- Development notes, tips & best practices
- Citation and license
- Contributing and contact

## What this project is
Learning-With-Graphs-Main implements the E(n) Equivariant Graph Neural Network family (EGNN) and demonstrates applications to: (1) graph-level classification (TUDatasets like NCI1) using a contrastive + classification training recipe, and (2) N-body dynamical system modelling and visualization. The code is research-oriented and intended for reproducing experiments, extending EGNN layers, and benchmarking variants.

### Background & motivation
EGNNs are a class of graph neural networks with built-in equivariance to E(n) geometric transformations (rotations, translations, reflections in n dimensions). Equivariance is especially important when node coordinates (3D positions, spatial embeddings) are meaningful: models that respect these symmetries generalize better and reduce the burden on the model to learn invariances.

This repo is organized to make it easy to:
- inspect and reuse a clear implementation of the EGCL (equivariant graph convolutional layer) and EGNN stacks,
- run a supervised contrastive + classification experiment on graph datasets using PyTorch Geometric,
- reproduce N-body simulation experiments and visualize predicted trajectories.

Key academic references included in the repository:
- The EGNN paper: `2102.09844v3.pdf` (full implementation mirrors equations in the paper and Appendix C MLP choices).
- A second PDF `E_n__Equivariant_Graph_Neural_Networks.pdf` is included for quick reading.

## Highlights and contributions
- Faithful, well-documented implementation of EGCL (edge message MLP, coordinate update, edge inference gating, velocity extensions).
- Graph-level model (GraphEGNN) that combines Laplacian positional encodings (LPE), DropEdge, attention pooling, and a supervised contrastive loss for better graph embeddings.
- Reproducible N-body dynamical systems code to evaluate velocity and coordinate-update variants of EGNNs.
- Ready-to-run training scripts and saved model checkpoints (best_model.pt and final_model.pt) for the similarity model experiments.

## How the repository is organized
Top-level annotated tree:

```
README.md                    — this report (updated)
2102.09844v3.pdf             — EGNN paper PDF
E_n__Equivariant_Graph_Neural_Networks.pdf — another copy / related PDF
egnn.py                      — core EGNN implementation (single-file reference)
Similarity Model/             — experiments and training for graph classification
  ├─ egnn.py                 — project-local EGNN build (used by train.py)
  ├─ train.py                — training script for NCI1 / TUDataset experiments
  ├─ test_similarity.py      — evaluation / test utilities
  ├─ best_model.pt           — saved checkpoint (binary, ~2.3MB)
  └─ final_model.pt          — final checkpoint
nbody_system_paper/          — N-body dynamical systems experiment code
  ├─ README.md               — notes and run examples for N-body experiments
  ├─ dataset_nbody.py        — dataset generator for simulated charged particles
  ├─ egnn_clean.py           — EGNN core variant (clean reference used in paper experiments)
  ├─ main_nbody.py           — training entry point for the N-body experiments
  ├─ model.py                — model definitions for dynamical tasks
  └─ visualize_nbody.py      — visualization utilities (save / display trajectories)
data/                        — datasets (TUDataset cache path referenced by train.py)
.vscode/                     — editor settings (ignore for runtime)
__pycache__/                 — Python bytecode caches (ignore)
```

Files of special interest:
- `egnn.py` and `Similarity Model/egnn.py`: both contain the EGNN implementation. The top-level `egnn.py` is a general, fully-documented single-file implementation; the `Similarity Model/egnn.py` mirrors this within the experiment folder.
- `Similarity Model/train.py`: full training pipeline that assembles the GraphEGNN model, data loaders, Laplacian PEs, DropEdge, supervised contrastive loss, training loop, and checkpointing.
- `nbody_system_paper/*.py`: full experiment code for simulating, training, and visualizing N-body trajectories.

## How it fits together
- For graph classification experiments, `Similarity Model/train.py` loads a TUDataset (default: NCI1) from `./data/TUDataset`. Node features are augmented with normalized degree and Laplacian positional encodings (LPE). The GraphEGNN encoder uses an EGNN stack in invariant mode (update_coords=False) with edge inference enabled (soft edge weights). Encoded graph vectors are aggregated using an attentional pooling layer and fed to a classification head; the loss combines cross-entropy and a supervised contrastive term.

- For N-body experiments, `nbody_system_paper` contains dataset generation, model definitions (including velocity-enabled variants), training, and visualization scripts. The default recipes follow Appendix C of the EGNN paper and the repo README in `nbody_system_paper/README.md` explains run parameters.

## Quickstart — run training and evaluation
Below are the minimal commands to run the main experiments. GPU is strongly recommended.

1) Similarity model (graph classification on NCI1)

```bash
# from the repository root
cd "Similarity Model"
# Basic run with defaults
python train.py

# Example: increase model size and run fewer epochs for quick checks
python train.py --hidden 128 --layers 5 --epochs 50 --bs 32
```

2) N-body dynamical system (small smoke test)

```bash
cd nbody_system_paper
# run the EGNN velocity model with safer defaults (see folder README)
python main_nbody.py --model egnn_vel

# smoke test quick run
python main_nbody.py --model egnn_vel --epochs 1 --batch_size 2 --max_training_samples 4 --test_interval 1 --total_steps 30 --frame_0 10 --frame_T 20
```

3) Visualize an N-body sample

```bash
# ground-truth trajectory only
python visualize_nbody.py --partition test --sample_idx 0 --max_samples 8

# ground-truth + model prediction
python visualize_nbody.py --model egnn_vel --ckpt logs/exp_1/best_model.pt --partition test --sample_idx 0 --max_samples 8
```

## Reproducing experiments and checkpoints
- The `Similarity Model` folder contains `best_model.pt` and `final_model.pt`. These are Torch state dictionaries saved by `train.py`.
- To reproduce published results, inspect `train.py` for the scheduler, optimizer (Adam), learning rate, weight decay, label smoothing, and contrastive loss temperature. The script checkpoints the best validation accuracy to `best_model.pt`.
- N-body experiments use the `nbody_system_paper/main_nbody.py` training entry point and save logs/checkpoints under `nbody_system_paper/logs`.

## Data and dependencies
There is no repository-level requirements file. The main dependencies (tested / expected) are:
- Python 3.8+ (3.9/3.10 recommended)
- PyTorch (1.13+ or compatible with your CUDA)
- torch-geometric (PyG) and its associated dependencies (torch-scatter, torch-sparse, torch-cluster, torch-spline-conv) compatible with your PyTorch/CUDA setup
- numpy
- matplotlib (for visualization)
- optional: tqdm, scikit-learn

Install example (CPU, simple):

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric
pip install numpy matplotlib
```

For a proper GPU setup, follow PyG installation instructions at https://pytorch-geometric.readthedocs.io/.

## Tests and verification
- `Similarity Model/test_similarity.py` contains evaluation code and smaller tests for the model pipeline. Use this to sanity check that the model loads and runs on a subset of the dataset.
- The repository includes unit-like checks and smoke-test commands in the nbody README for quick verification.

## Development notes and tips
- The EGNN code includes two complementary implementations (root `egnn.py` and `Similarity Model/egnn.py`). Use the one inside an experiment folder if you want a copy tightly coupled to that experiment. The top-level file is a canonical reference implementation.
- When training on larger datasets or models, monitor GPU memory: EGNN uses edge-wise message computations that can grow with node count and edge density (fully-connected graphs blow up as O(N^2) edges). Use `--max_nodes` and `--drop_edge` to manage memory.
- For graph classification, an LPE dimension of 8 is used by default. You can disable LPE by adjusting the script and removing the related concatenation in `build_node_features`.

## Known limitations and suggested next steps
- There is no cross-platform install script (e.g., requirements.txt or conda environment). Adding a pinned requirements file will improve reproducibility.
- Checkpoint naming and logging are minimal; consider adding structured experiment logging (tensorboard or MLFlow) for long runs.
- Some larger models / full reproductions require the appropriate datasets (TUDataset will be downloaded automatically by PyG if not present).

## Citation
If you use this code in research, please cite the EGNN paper (included in the repository) and mention this implementation as a reproduction/reference.

## License
No explicit LICENSE file is present in the repository. Before reuse, please ensure you add an appropriate license (MIT, Apache-2.0, etc.) or ask the repository owner for permission if you intend to use the code beyond educational purposes.

## Contributing and contact
This repository is organized for research and experimentation. If you want to contribute:
- Open an issue describing the feature / bug.
- Submit a PR that includes tests or a reproducible example.
- For questions about how to run particular experiments, check the corresponding folder README (e.g., `nbody_system_paper/README.md`) or open an issue.

---

This README was generated after reading the repository structure and the main experiment scripts. If you want, I can:
- add a pinned requirements.txt with pinned package versions appropriate to your CUDA/PyTorch setup,
- add a small reproducible example that runs quickly on CPU (unit test style), or
- split the README into a short top-level README and longer docs/DEVELOPER.md.
