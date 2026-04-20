# Acoustic Co-Design

Joint optimization of a thin-plate acoustic resonator's geometry, piezoelectric
actuator placement, and modal feedback gains to match a target frequency
response. Course project for 18-848 *Computational Design of Cyber-Physical
Systems*, Spring 2026.

<!-- Replace YOURUSER with your GitHub handle after pushing. -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOURUSER/acoustic-codesign/blob/main/demo.ipynb)

## Problem

Designing the acoustic cavity of a stringed instrument to hit a target timbre
is traditionally an iterative craft process. Purely passive shape optimization
is one way to automate it, but the reachable frequency responses are bounded by
what the shape alone can do. This project asks whether jointly optimizing the
shape together with a small number of piezo actuators and a modal feedback law
can reach target spectra that neither subsystem can reach on its own.

The framing is **co-design**: the plant and the controller are optimized as
coupled variables rather than sequentially. The best actuator positions depend
on the plate's mode shapes, which depend on its thickness field. The best
geometry depends on which resonances a controller can correct for free.
Searching both at once beats freezing one and then designing the other.

## Method in one paragraph

A thin rectangular plate with simply-supported edges is parameterized by a 2D
sine expansion of its thickness field. Finite differences on a regular grid
give a Kirchhoff plate stiffness operator with spatially varying bending
stiffness; the resulting generalized eigenproblem yields natural frequencies
and mode shapes. Piezo actuator influences are computed as values of the mode
shapes at their normalized positions. For each mode, a 2x2 LQR Riccati is
solved analytically in closed form (isotropic control-effort weight makes
$BB^\top$ scalar), giving per-actuator feedback gains. The closed-loop
frequency response to a fixed excitation point is computed and compared to a
target spectrum. Everything runs in JAX, so a single `jax.grad` call produces
gradients with respect to all design blocks, and Adam drives the joint
optimization.

## Quickstart

**CPU (plain pip):**
```bash
git clone https://github.com/YOURUSER/acoustic-codesign.git
cd acoustic-codesign
pip install -e .
jupyter notebook demo.ipynb
```

**GPU (conda + CUDA 12 on Linux or WSL2-Ubuntu):**
```bash
conda env create -f environment.yml
conda activate acoustic-codesign
python verify_gpu.py           # confirms JAX sees the GPU
python run_batched.py          # 8 parallel searches; saves multistart_result.npz
```

Or from Python directly:

```python
import codesign
params, history, freqs, target = codesign.run(num_steps=400)
```

On a laptop CPU, a single run takes about a minute at the default (medium)
profile. On a modern NVIDIA GPU, the batched 8-seed run finishes in comparable
wall time.

## Reproducing the hero figure

```bash
jupyter nbconvert --to notebook --execute demo.ipynb --output demo_executed.ipynb
```

This populates `hero.png` and `modes.png` in the repo root. The hero figure
overlays target spectrum, passive-only FRF, and co-designed FRF, plus the
optimized thickness field and final actuator layout.

## Project layout

```
acoustic-codesign/
├── codesign.py          # physics, control, optimization
├── plots.py             # visualization helpers
├── run_batched.py       # N-seed multi-start runner (uses jax.vmap)
├── verify_gpu.py        # JAX GPU sanity check + benchmark
├── demo.ipynb           # end-to-end demo notebook
├── environment.yml      # conda env with GPU JAX
├── requirements.txt     # pip fallback
├── pyproject.toml
├── README.md
└── .gitignore
```

## Design-block summary

| Block | Symbol | Shape | What it controls |
| --- | --- | --- | --- |
| Thickness coefficients | `c` | `(M, M)` | Plate geometry via sine expansion |
| Actuator positions | `positions` | `(n_actuators, 2)` | Where piezos are bonded, in [0,1]^2 |
| LQR state weight | `log_q` | scalar | Aggression on modal displacement |
| LQR control weight | `log_r` | scalar | Penalty on actuator effort |

All four enter the same differentiable pipeline and are updated together by
Adam.

## Key knobs in `codesign.Config`

* `Lx, Ly`, `Nx, Ny`: plate dimensions and grid resolution
* `h0`: nominal thickness
* `M`: thickness basis order (design-space size is $M^2$)
* `n_modes`: how many modes to keep
* `n_actuators`: number of piezos
* `disturb_xy`: normalized (x, y) of the string/bridge excitation
* `freq_lo, freq_hi, n_freqs`: frequency grid for the target match

## Acknowledgements

Built on suggestions from the course instructor, in particular the pointer
toward parameterized thin-shell surfaces with analytic bases rather than full
voxel-level topology optimization.
