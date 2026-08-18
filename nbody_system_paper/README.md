# N-Body Dynamical Systems

This folder implements the Appendix C.1 dynamical systems setup around
`egnn_clean.py` and the original `main_nbody.py` structure.

Implemented here:
- charged-particle N-body dataset in 3D with `N=5`
- no virtual boxes
- `5000` simulated steps
- input at timestep `3000`
- target at timestep `4000`
- default split:
  - train `3000`
  - val `2000`
  - test `2000`
- sweep support from `100` to `50000` training samples
- models:
  - `egnn_vel`
  - `gnn`
  - `rf_vel`
  - `baseline`
  - `linear`
  - `linear_vel`

Not implemented here:
- QM9 task
- SE(3)-Transformer
- TFN

## Main files

- `egnn_clean.py`: core equivariant layer and EGNN reference implementation
- `dataset_nbody.py`: local paper-style dataset generator
- `model.py`: dynamical-system models
- `main_nbody.py`: training entry point

## Run

```bash
cd /Users/surya/EDU/PRoject/nbody_system_paper
python main_nbody.py --model egnn_vel
```

This uses the safer defaults we added for stability:
- `--lr 1e-4`
- `--norm_diff True`
- `--tanh True`
- gradient clipping in training

Quick smoke test:

```bash
python main_nbody.py --model egnn_vel --epochs 1 --batch_size 2 --max_training_samples 4 --test_interval 1 --total_steps 30 --frame_0 10 --frame_T 20
```

## Visualize

Ground-truth trajectory only:

```bash
python visualize_nbody.py --partition test --sample_idx 0 --max_samples 8
```

Ground-truth plus model prediction:

```bash
python visualize_nbody.py --model egnn_vel --ckpt logs/exp_1/best_model.pt --partition test --sample_idx 0 --max_samples 8
```

Save the figure instead of opening a window:

```bash
python visualize_nbody.py --model egnn_vel --ckpt logs/exp_1/best_model.pt --save outputs/sample0.png
```
