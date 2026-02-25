# Quick Start

Here is a simple example of how to use AI4Plasma.

## Running an Example

All example scripts live under `app/`.

From the repository root, run one of the examples:

```bash
cd app/operator/deeponet
python solve_1d_poisson.py
```

Other commonly used scripts:

```bash
# 2D Poisson with DeepONet
python app/operator/deeponet/solve_2d_poisson.py

# 1D steady arc discharge (CS-PINN)
python app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py

# 1D corona discharge (RK-PINN)
python app/piml/rk_pinn/solve_1d_corona_rk_pinn.py
```

## Monitor training with TensorBoard

Many training scripts write TensorBoard logs under `app/**/runs/`.

```bash
tensorboard --logdir app
```
