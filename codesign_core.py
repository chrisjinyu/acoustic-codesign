"""
codesign_core.py -- shared physics and infrastructure.

Contains everything that is independent of the specific co-design formulation:
  * Config
  * Thickness parameterization (bounded sigmoid)
  * 5-point Laplacian, biharmonic plate assembly
  * Custom VJP for `eigh` that nullifies degenerate-mode gradients
  * Generalized eigensolve via diagonal mass scaling
  * Bilinear sampling helper
  * FRF magnitude via closed-loop state-space
  * Fixed-target Gaussian spectrum generator

Downstream modules (codesign.py for LQR variant, codesign_strings.py for string
variant) import from here and only add their own loss/init/run functions.
"""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp

# fp32 is not viable for this problem; eigh gradients and FRF condition
# number demand fp64 precision.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Scale profiles (swap the defaults below to change):

        Small  (CPU, quick iter):  Nx=25, Ny=20, M=3, n_modes=10, n_actuators=3
        Medium (single GPU):       Nx=50, Ny=40, M=5, n_modes=20, n_actuators=6   <- default
        Large  (GPU stress):       Nx=80, Ny=60, M=7, n_modes=30, n_actuators=8

    The `material` knob switches between steel (default) and spruce, which is
    the physically appropriate choice for a violin-scale plate driven by strings.
    """
    # Plate geometry
    Lx: float = 0.30
    Ly: float = 0.20
    Nx: int = 50
    Ny: int = 40
    h0: float = 3.0e-3

    # Material ("steel" or "spruce")
    material: str = "spruce"

    # Design space
    M: int = 5
    n_modes: int = 20
    n_actuators: int = 6

    # Excitation point (normalized coords) representing string/bridge contact
    disturb_xy: tuple = (0.20, 0.50)

    # Frequency grid for the FRF objective
    n_freqs: int = 300
    freq_lo: float = 40.0
    freq_hi: float = 1500.0
    zeta: float = 0.02

    @property
    def E(self) -> float:
        return {"steel": 2.1e11, "spruce": 1.0e10}[self.material]

    @property
    def nu(self) -> float:
        return {"steel": 0.30, "spruce": 0.30}[self.material]

    @property
    def rho(self) -> float:
        return {"steel": 7800.0, "spruce": 450.0}[self.material]


cfg = Config()


# =============================================================================
# Spatial grid (module-level globals, rebuilt by reconfigure())
# =============================================================================
xs = jnp.linspace(0.0, cfg.Lx, cfg.Nx)
ys = jnp.linspace(0.0, cfg.Ly, cfg.Ny)
X, Y = jnp.meshgrid(xs, ys, indexing="ij")
DX = cfg.Lx / (cfg.Nx - 1)
DY = cfg.Ly / (cfg.Ny - 1)


def reconfigure(**kwargs):
    """Update module-level config and recompute all dependent constants.

    Call this from your notebook BEFORE calling any physics functions.
    Any Config field can be overridden, for example::

        import codesign_core as core
        core.reconfigure(material="spruce", n_actuators=4)

    Returns the new cfg so you can inspect it.

    IMPORTANT: In a notebook, call this before the downstream module imports
    since codesign.py and codesign_strings.py both read `cfg` at their own
    import time. The demo notebook handles this automatically by calling
    reconfigure() right after importing codesign_core, then reloading the
    downstream modules.
    """
    import dataclasses
    global cfg, xs, ys, X, Y, DX, DY, L_LAP

    cfg = dataclasses.replace(cfg, **kwargs)

    xs = jnp.linspace(0.0, cfg.Lx, cfg.Nx)
    ys = jnp.linspace(0.0, cfg.Ly, cfg.Ny)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    DX = cfg.Lx / (cfg.Nx - 1)
    DY = cfg.Ly / (cfg.Ny - 1)
    L_LAP = _laplacian_matrix()

    print(f"Reconfigured: material={cfg.material}, "
          f"E={cfg.E:.2e} Pa, rho={cfg.rho:.0f} kg/m^3, "
          f"n_actuators={cfg.n_actuators}, "
          f"grid={cfg.Nx}x{cfg.Ny}, M={cfg.M}")
    return cfg


# =============================================================================
# Geometry: thickness parameterization
# =============================================================================
def thickness(c: jnp.ndarray) -> jnp.ndarray:
    """h(x, y) as a sigmoid-bounded sine expansion.
    Keeps the plate within the thin-plate regime at all optimization steps.
    """
    ms = jnp.arange(1, cfg.M + 1)[:, None, None, None]
    ns = jnp.arange(1, cfg.M + 1)[None, :, None, None]
    basis = (jnp.sin(ms * jnp.pi * X[None, None] / cfg.Lx)
             * jnp.sin(ns * jnp.pi * Y[None, None] / cfg.Ly))
    perturb = jnp.sum(c[:, :, None, None] * basis, axis=(0, 1))
    h_min = 0.5 * cfg.h0
    h_max = 3.0 * cfg.h0
    return h_min + (h_max - h_min) * jax.nn.sigmoid(perturb / cfg.h0)


def bending_stiffness(h: jnp.ndarray) -> jnp.ndarray:
    return cfg.E * h**3 / (12.0 * (1.0 - cfg.nu**2))


# =============================================================================
# Physics: stiffness and mass matrices
# =============================================================================
def _laplacian_matrix() -> jnp.ndarray:
    """5-point Laplacian, zero Dirichlet BCs, row-major (C-contiguous) order."""
    e_x = jnp.ones(cfg.Nx)
    Dxx = (jnp.diag(-2 * e_x)
           + jnp.diag(e_x[:-1], 1)
           + jnp.diag(e_x[:-1], -1)) / DX**2
    Ix = jnp.eye(cfg.Nx)
    e_y = jnp.ones(cfg.Ny)
    Dyy = (jnp.diag(-2 * e_y)
           + jnp.diag(e_y[:-1], 1)
           + jnp.diag(e_y[:-1], -1)) / DY**2
    Iy = jnp.eye(cfg.Ny)
    return jnp.kron(Dxx, Iy) + jnp.kron(Ix, Dyy)


L_LAP = _laplacian_matrix()


def assemble_system(c: jnp.ndarray):
    """Return (K, M_mat) for the Kirchhoff plate generalized eigenproblem."""
    h = thickness(c)
    D = bending_stiffness(h)
    D_flat = D.flatten()
    K = (L_LAP * D_flat[None, :]) @ L_LAP * (DX * DY)
    K = 0.5 * (K + K.T)
    mass = cfg.rho * h.flatten() * DX * DY
    M_mat = jnp.diag(mass)
    return K, M_mat


# =============================================================================
# Custom VJP: masks degenerate-mode 1/(lambda_i - lambda_j) singularity
# =============================================================================
@jax.custom_vjp
def safe_eigh(A):
    return jnp.linalg.eigh(A)


def _safe_eigh_fwd(A):
    w, v = safe_eigh(A)
    return (w, v), (w, v)


def _safe_eigh_bwd(res, g):
    w, v = res
    g_w, g_v = g
    vt_gv = jnp.matmul(v.T, g_v)
    diffs = w[None, :] - w[:, None]
    mask = jnp.abs(diffs) < 1e-8
    safe_diffs = jnp.where(mask, 1.0, diffs)
    F = jnp.where(mask, 0.0, 1.0 / safe_diffs)
    mid = jnp.diag(g_w) + F * vt_gv
    mid = 0.5 * (mid + mid.T)
    dA = jnp.matmul(v, jnp.matmul(mid, v.T))
    return (dA,)


safe_eigh.defvjp(_safe_eigh_fwd, _safe_eigh_bwd)


def solve_modes(c: jnp.ndarray):
    """First n_modes natural frequencies (rad/s) and M-orthonormal mode shapes.
    Reduces generalized eigenproblem K phi = lambda M phi to standard form via
    diagonal mass scaling, since JAX eigh does not support the b parameter.
    """
    K, M_mat = assemble_system(c)
    m_diag = jnp.diag(M_mat)
    sqrt_m_inv = 1.0 / jnp.sqrt(m_diag + 1e-30)
    K_tilde = K * sqrt_m_inv[:, None] * sqrt_m_inv[None, :]
    K_tilde = 0.5 * (K_tilde + K_tilde.T)
    reg = 1e-8 * jnp.eye(K_tilde.shape[0])
    w, V_tilde = safe_eigh(K_tilde + reg)
    w = jnp.clip(w[: cfg.n_modes], min=0.0)
    omega = jnp.sqrt(w + 1e-12)
    Phi = sqrt_m_inv[:, None] * V_tilde[:, : cfg.n_modes]
    return omega, Phi


# =============================================================================
# Bilinear sampling (used by actuators, bridges, strings)
# =============================================================================
def bilinear_sample(field_grid: jnp.ndarray, positions_norm: jnp.ndarray) -> jnp.ndarray:
    """field_grid: (Nx, Ny, K). positions_norm in [0,1]^2, shape (P, 2).
    Returns sampled values of shape (P, K)."""
    x = positions_norm[:, 0] * (cfg.Nx - 1)
    y = positions_norm[:, 1] * (cfg.Ny - 1)
    x0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, cfg.Nx - 2)
    y0 = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, cfg.Ny - 2)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    f00 = field_grid[x0, y0]
    f10 = field_grid[x0 + 1, y0]
    f01 = field_grid[x0, y0 + 1]
    f11 = field_grid[x0 + 1, y0 + 1]
    return ((1 - fx) * (1 - fy) * f00
            + fx * (1 - fy) * f10
            + (1 - fx) * fy * f01
            + fx * fy * f11)


def modal_values_at_points(Phi: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Return array of shape (n_modes, n_points) containing phi_i(p_j)."""
    Phi_grid = Phi.reshape(cfg.Nx, cfg.Ny, cfg.n_modes)
    return bilinear_sample(Phi_grid, positions).T


# =============================================================================
# Closed-loop state-space and FRF magnitude
# =============================================================================
def closed_loop(omega: jnp.ndarray,
                B: jnp.ndarray,
                K_gain: jnp.ndarray,
                b_dist: jnp.ndarray,
                zeta: float = cfg.zeta):
    """Assemble A_cl, B_d, C_out in stacked modal state space.
    Pass K_gain = zeros((n_act, 2*n_modes)) for the passive case.
    """
    n = cfg.n_modes
    idx_p = jnp.arange(n) * 2
    idx_d = idx_p + 1
    A = jnp.zeros((2 * n, 2 * n))
    A = A.at[idx_p, idx_d].set(1.0)
    A = A.at[idx_d, idx_p].set(-omega**2)
    A = A.at[idx_d, idx_d].set(-2 * zeta * omega)
    n_act = B.shape[1]
    B_u = jnp.zeros((2 * n, n_act))
    B_u = B_u.at[idx_d, :].set(B)
    B_d = jnp.zeros((2 * n, 1))
    B_d = B_d.at[idx_d, 0].set(b_dist)
    C_out = jnp.zeros((1, 2 * n))
    C_out = C_out.at[0, idx_d].set(1.0)
    A_cl = A - B_u @ K_gain
    return A_cl, B_d, C_out


def frf_magnitude(A_cl: jnp.ndarray,
                  B_d: jnp.ndarray,
                  C_out: jnp.ndarray,
                  freqs_hz: jnp.ndarray) -> jnp.ndarray:
    """|C (jw I - A_cl)^{-1} B_d| at each frequency (Hz)."""
    omega_rad = 2.0 * jnp.pi * freqs_hz
    I = jnp.eye(A_cl.shape[0])

    def _one(w):
        resp = jnp.linalg.solve(1j * w * I - A_cl, B_d[:, 0])
        return jnp.abs(C_out[0] @ resp + 1e-12 + 1e-12j)

    return jax.vmap(_one)(omega_rad)


def frf_passive(c: jnp.ndarray, freqs_hz: jnp.ndarray) -> jnp.ndarray:
    """Convenience: FRF with no control (pure passive plate response)."""
    omega, Phi = solve_modes(c)
    bridge = jnp.array([cfg.disturb_xy])
    b_dist = modal_values_at_points(Phi, bridge)[:, 0]
    # Dummy B and zero gain
    B_zero = jnp.zeros((cfg.n_modes, 1))
    K_zero = jnp.zeros((1, 2 * cfg.n_modes))
    A_cl, B_d, C_out = closed_loop(omega, B_zero, K_zero, b_dist)
    return frf_magnitude(A_cl, B_d, C_out, freqs_hz)


# =============================================================================
# Fixed-target Gaussian spectrum (used by LQR mode)
# =============================================================================
def target_spectrum_fixed(centers_hz=(120.0, 300.0, 620.0),
                          widths_hz=(15.0, 20.0, 30.0)):
    """Return (freqs, spectrum) with Gaussian peaks at fixed frequencies."""
    freqs = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)
    centers = jnp.array(centers_hz)
    widths = jnp.array(widths_hz)
    spec = sum(jnp.exp(-((freqs - ci) / wi)**2) for ci, wi in zip(centers, widths))
    return freqs, spec
