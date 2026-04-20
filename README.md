# Acoustic Co-Design: Joint Optimization of Plate Geometry, Actuator Placement, and Modal Feedback

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOURUSER/acoustic-codesign/blob/main/demo.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/framework-JAX-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Course project for 18-848: Computational Design of Cyber-Physical Systems, Spring 2026.**
> Joint optimization of a thin-plate acoustic resonator's geometry, piezoelectric actuator placement,
> and modal feedback gains to match a target frequency response.

---

## Overview

Designing the acoustic body of a stringed instrument to produce a target timbre is traditionally an
iterative craft process guided by experience rather than computation. Purely passive shape optimization
can automate part of this, but the reachable set of frequency responses is bounded by what geometry
alone can achieve. This project asks whether jointly optimizing the plate **and** a small number of
piezo actuators with a modal feedback law can reach target spectra that neither subsystem can reach
independently.

The framing is **co-design**: the plant (plate geometry) and the controller (actuator positions and
LQR gains) are optimized as coupled variables rather than sequentially. The best actuator placement
depends on the plate's mode shapes, which in turn depend on the thickness field. The best geometry
depends on which resonances a controller can correct without added hardware cost. Searching both
simultaneously consistently outperforms freezing one and then optimizing the other.

A secondary mode targets a different physical problem: tuning a plate so that its resonant
frequencies match a prescribed set of string pitches, with string tensions treated as additional
design variables.

---

## The Hero Figure

<p align="center">
  <img src="outputs/hero.png" alt="Hero figure: target spectrum overlay, thickness field, and actuator layout" width="820"/>
</p>

**Figure 1.** The hero figure produced by `demo.ipynb`. Left: frequency response functions (FRF) for
the passive baseline (grey), the co-designed closed-loop system (blue), and the target spectrum
(orange dashes). The co-designed system closely tracks the target across the full frequency band,
while the passive plate alone cannot reach the target peaks and troughs. Right: optimized thickness
field (colormap, lighter = thicker) overlaid with final actuator positions (crosses). The optimizer
discovers a non-uniform stiffness distribution that moves mode shapes toward actuator-controllable
configurations.

---

## Mode Shape Visualization

<p align="center">
  <img src="outputs/modes.png" alt="Mode shape visualization for the optimized plate" width="700"/>
</p>

**Figure 2.** The first eight mode shapes of the optimized plate. The simply-supported boundary
conditions enforce zero displacement at all edges; the interior shape is determined by the
optimized thickness field. Thickened ridges concentrate modal energy and shift natural frequencies
relative to a uniform plate, giving the LQR controller more leverage over the closed-loop spectrum.

---

## Method

### Physical model

The plate is a thin rectangular panel with simply-supported edges, parameterized by a 2D sine
expansion of its thickness field:

$$h(x, y) = h_0 + \sum_{i=1}^{M}\sum_{j=1}^{M} c_{ij}
\sin\!\left(\frac{i\pi x}{L_x}\right)
\sin\!\left(\frac{j\pi y}{L_y}\right)$$

The $M^2$ coefficients $\mathbf{c}$ form the geometric design block. Finite differences on a
regular grid give a Kirchhoff plate stiffness operator with spatially varying bending stiffness
$D(x,y) = Eh(x,y)^3 / 12(1-\nu^2)$. The resulting generalized eigenproblem

$$\mathbf{K}(\mathbf{c})\,\mathbf{u}_k = \omega_k^2\,\mathbf{M}(\mathbf{c})\,\mathbf{u}_k$$

yields natural frequencies $\omega_k$ and mode shapes $\mathbf{u}_k$ for $k = 1, \ldots, n_\text{modes}$.

### Actuator model

Each piezoelectric actuator at normalized position $(p_x, p_y) \in [0,1]^2$ contributes a modal
force proportional to the mode shape evaluated at its location. The influence vector for actuator
$a$ on mode $k$ is

$$b_{k,a} = u_k(p_{x,a},\, p_{y,a})$$

giving a per-mode input matrix $B_k \in \mathbb{R}^{1 \times n_\text{act}}$.

### LQR feedback

For each mode $k$ the dynamics reduce to a 2D linear system (displacement and velocity). A 2x2
LQR Riccati equation is solved analytically in closed form; the isotropic control-effort weighting
makes $B_k B_k^\top$ scalar, so the Riccati solution has a clean expression without a matrix
square root. The resulting per-mode feedback gains $K_k$ close the loop and move the effective
resonant frequency and damping of each mode, extending the reachable set of frequency responses
beyond what geometry alone can provide.

### Differentiable pipeline

The closed-loop frequency response to a fixed excitation point is computed from the per-mode
transfer functions and compared to a target spectrum via a log-magnitude loss. All four design
blocks enter the same differentiable pipeline:

| Block | Symbol | Shape | Controls |
|---|---|---|---|
| Thickness coefficients | $\mathbf{c}$ | $(M, M)$ | Plate geometry via sine expansion |
| Actuator positions | $\mathbf{p}$ | $(n_\text{act},\, 2)$ | Where piezos are bonded, in $[0,1]^2$ |
| LQR state weight | $\log q$ | scalar | Aggressiveness on modal displacement |
| LQR control weight | $\log r$ | scalar | Penalty on actuator effort |

Everything runs in JAX, so a single `jax.grad` call produces gradients with respect to all four
blocks simultaneously, and Adam drives the joint optimization. Multi-start runs with `jax.vmap`
search eight initializations in parallel at roughly the same wall time as a single run.

### Strings mode

A separate optimization mode replaces the LQR controller with string tension variables. Given a
target set of pitches (e.g. G3, D4, A4, E5), the optimizer jointly adjusts the plate thickness
field and the string tensions so that the plate's resonant frequencies align with the string
harmonics. This is particularly useful for designing a top plate whose resonances reinforce the
target timbre without any electronic actuation.

---

## Results

### LQR co-design vs. passive baseline

<p align="center">
  <img src="outputs/fig_convergence.png" alt="Convergence curves for co-design and passive baseline" width="680"/>
</p>

**Figure 3.** Loss curves over 600 Adam steps for the 8-seed multi-start (thin lines) and the
single best run (thick line), for both the co-designed system (blue) and the passive-only baseline
(grey). The co-designed system reaches a loss roughly 40--60% below the passive optimum, depending
on the target spectrum, confirming that the controller contributes non-trivially beyond what
geometry rearrangement alone can do.

<p align="center">
  <img src="outputs/fig_actuator_sweep.png" alt="Loss vs. number of actuators" width="560"/>
</p>

**Figure 4.** Final loss after 400 steps as a function of actuator count (sweep over $n_\text{act}
\in \{2, 4, 6, 8\}$). Returns diminish beyond four actuators for the medium-complexity target
spectrum used here, suggesting that hardware cost can be traded against achievable spectral
accuracy in a principled way.

### Strings mode: pitch alignment

<p align="center">
  <img src="outputs/fig_strings_spectrum.png" alt="String-mode target pitch alignment" width="680"/>
</p>

**Figure 5.** Frequency response in strings mode, showing the target pitches (dashed vertical
lines) and the actual resonant peaks of the co-designed plate-plus-strings system. The optimizer
places plate resonances within a few Hz of each target pitch across the full range tested
(196--880 Hz).

---

## Repository Layout

```
acoustic-codesign/
├── demo.ipynb              # End-to-end demo and results notebook
├── codesign.py             # Physics, control, and optimization (LQR path)
├── codesign_core.py        # Shared Config dataclass and grid helpers
├── codesign_strings.py     # Strings variant (tension optimization)
├── plots.py                # Visualization helpers (imported by codesign.py)
├── environment.yml         # Conda environment with GPU JAX
├── requirements.txt        # Pip fallback
├── pyproject.toml
├── README.md
├── .gitignore
│
├── scripts/
│   ├── run_batched.py      # N-seed multi-start runner (jax.vmap)
│   ├── run_sweep.py        # Sensitivity sweep over n_actuators / n_strings
│   ├── export_geometry.py  # Write thickness-field CSVs from best_params.npz
│   ├── verify_gpu.py       # JAX GPU sanity check and benchmark
│   └── analysis.py         # Post-hoc figure generation (fig_*.pdf/png)
│
└── outputs/                # All generated files land here
    ├── results.json        # Summary metrics from the last notebook run
    ├── best_params.npz     # Optimized parameters from demo.ipynb
    ├── multistart_lqr.npz  # Batched run results (LQR)
    ├── hero.png            # Overlay figure (tracked in git)
    ├── modes.png           # Mode-shape grid (tracked in git)
    └── fig_*.png / fig_*.pdf  # Analysis figures (tracked in git)
```

---

## Quickstart

**CPU (plain pip, tested on macOS and Linux):**

```bash
git clone https://github.com/YOURUSER/acoustic-codesign.git
cd acoustic-codesign
pip install -e .
jupyter notebook demo.ipynb
```

On a laptop CPU, one optimization run (600 steps, medium grid) takes about 60--90 seconds.

**GPU (conda + CUDA 12, Linux or WSL2):**

```bash
conda env create -f environment.yml
conda activate acoustic-codesign
python scripts/verify_gpu.py           # Confirm JAX sees the GPU
python scripts/run_batched.py          # 8 seeds in parallel -> outputs/multistart_lqr.npz
```

**From Python:**

```python
import codesign
params, history, freqs, target = codesign.run(num_steps=400)
```

**Reproducing all figures:**

```bash
jupyter nbconvert --to notebook --execute demo.ipynb --output demo_executed.ipynb
python scripts/analysis.py
```

This writes `outputs/hero.png`, `outputs/modes.png`, and all `outputs/fig_*.png` figures.

---

## Key Configuration Knobs

All knobs live in `codesign_core.Config` and `codesign_strings.StringConfig`. Edit either
dataclass directly or override fields at call sites.

### `codesign_core.Config`

| Field | Default | Effect |
|---|---|---|
| `Lx`, `Ly` | `0.35`, `0.25` m | Plate dimensions |
| `Nx`, `Ny` | `32`, `24` | Finite-difference grid resolution |
| `h0` | `3e-3` m | Nominal plate thickness |
| `M` | `6` | Sine-basis order; design space is $M^2$ dimensional |
| `n_modes` | `12` | Number of modes retained |
| `n_actuators` | `4` | Number of piezo actuators |
| `disturb_xy` | `(0.3, 0.4)` | Normalized excitation point |
| `freq_lo`, `freq_hi` | `50`, `600` Hz | Frequency grid bounds |
| `n_freqs` | `512` | Points on the frequency grid |

### `codesign_strings.StringConfig`

| Field | Default | Effect |
|---|---|---|
| `target_pitches_hz` | `(196, 293.7, 440, 659.3)` | Target string pitches (G3 D4 A4 E5) |
| `string_mass_per_length` | `3e-4` kg/m | Linear mass density of strings |
| `string_length` | `0.65` m | Vibrating string length |

---

## Background: Why Co-Design?

Standard acoustic design approaches treat the instrument body as a purely passive resonator and
optimize its shape to get as close as possible to a target spectrum. This works well for
high-Q resonances but faces hard limits: once the plate's natural frequencies are fixed, no
geometric change can move them continuously without altering the entire mode shape basis.

Piezoelectric actuators bonded to the plate surface provide a second handle: modal feedback can
shift the apparent resonant frequency and damping of each mode independently, without changing
the plate's passive properties. The catch is that the effectiveness of each actuator depends
entirely on the mode shapes at its bonding location, which in turn depend on the plate's
geometry. Optimizing the two subsystems separately misses this coupling.

Co-design resolves the coupling by treating geometry and control as a single differentiable
program. JAX makes this tractable: the same `jax.grad` call that differentiates through the
finite-difference eigensolver also differentiates through the LQR Riccati solution and the
closed-loop transfer function, with no manual derivations or finite-difference gradient
approximations.

---

## Related Work

This project sits at the intersection of computational acoustics and co-design of cyber-physical
systems.

**Acoustic shape optimization.** Li et al. (SIGGRAPH 2016) introduced *Acoustic Voxels*, a
method for designing modular acoustic filters from 3D-printed voxel assemblies by optimizing
transmission loss over a frequency band. The approach is purely passive and voxel-discrete; this
project instead uses a continuous sine-basis parameterization and adds an active feedback layer.

**Aerophone design.** Umetani et al. (SIGGRAPH 2016) demonstrated computational design of
playable wind instruments by optimizing bore profiles to align resonances with target pitch
ladders. Their system targets tone holes and bore shapes rather than a coupled structural-control
problem, and does not include any active components.

**Metallophone contact optimization.** Bharaj et al. (ACM TOG 2015) computed bar and
mallet geometries for percussive instruments by matching mode frequencies to target pitches via
gradient-based shape optimization. This project shares the modal frequency-matching objective
but adds a thin-plate structural model and a closed-loop LQR layer rather than working with
free bars.

**Co-design of cyber-physical systems.** The co-design framing follows the course treatment:
plant and controller are coupled optimization variables rather than a sequential design pipeline.
Classical sequential approaches (shape-then-control or control-then-shape) are provably
suboptimal when the plant's controllability depends on the design parameters, as it does here
through the mode-shape dependence of actuator influence.

---

## Limitations and Future Work

**Model fidelity.** The Kirchhoff plate model assumes thin-plate bending with no in-plane
coupling and no acoustic radiation load. A more accurate model would include fluid-structure
interaction (the plate radiates into the air inside the instrument cavity) and geometric
nonlinearity for large deflections. Coupling to a 3D acoustic cavity model (e.g. via a boundary
element method) would allow targeting the radiated pressure spectrum rather than the plate's
structural FRF.

**Fabrication.** The optimized thickness field is a smooth function by construction (sine basis),
so it is compatible with CNC milling or multi-material 3D printing. The `scripts/export_geometry.py`
script writes the thickness field as a point-cloud CSV. A natural next step is importing this
into a CAD tool for fabrication and measuring the actual acoustic response for sim-to-real
validation.

**Actuator dynamics.** The current model treats piezo actuators as ideal point-force inputs.
A more complete model would account for the finite patch size, the capacitive dynamics of the
piezo element, and the amplifier's bandwidth and saturation.

**Broader co-design.** The current design space couples two subsystems (geometry and control).
Adding string properties (length, linear mass density, material) and bridge placement as
additional co-design variables would make the system a more complete model of a full stringed
instrument.

---

## Acknowledgements

Built on feedback from the course instructor, particularly the suggestion to use a continuous
sine-basis parameterization of the thickness field rather than a voxel-level topology
optimization, and the pointer to address the co-design framing by jointly optimizing the plant
and a feedback controller. Related work pointers from the professor's proposal feedback were
instrumental in contextualizing the method against the existing computational acoustics
literature.

Related papers consulted:
- Li et al., *Acoustic Voxels*, SIGGRAPH 2016.
- Umetani et al., *Aerophones in Flatland*, SIGGRAPH 2016.
- Bharaj et al., *Computational Design of Metallophone Contact Geometry*, ACM TOG 2015.