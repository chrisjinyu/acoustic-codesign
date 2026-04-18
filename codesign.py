"""
Acoustic Co-Design: joint optimization of thin-plate geometry,
piezoelectric actuator placement, and modal feedback gains.

18-848 Computational Design of Cyber-Physical Systems
Christian Yu, Spring 2026.

Pipeline:
  thickness field c  -->  K(c), M(c)  -->  (omega_i, phi_i)
                                              |
           actuator positions p  -->  B_modal = phi_i(p_j)
                                              |
                   LQR weights q, r  -->  K_gain via analytic modal Riccati
                                              |
           disturbance at bridge pt  -->  closed-loop FRF |H(jw)|
                                              |
                     target spectrum  -->  L(c, p, q, r)
                                              |
                                          jax.grad -> Adam
"""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.scipy.linalg import eigh as jeigh
import optax

# On consumer NVIDIA GPUs (e.g. RTX 4070 Ti Super) fp64 is ~60x slower than fp32.
# Default to fp32 for speed; flip to True if you see eigendecomp instability
# or near-degenerate modes giving NaN gradients.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Scale profiles (swap the defaults below to change):

        Small  (CPU, quick iter):  Nx=25, Ny=20, M=3, n_modes=10, n_actuators=3
        Medium (single GPU):       Nx=50, Ny=40, M=5, n_modes=20, n_actuators=5   <- default
        Large  (GPU stress):       Nx=80, Ny=60, M=7, n_modes=30, n_actuators=8
    """
    # Plate geometry
    Lx: float = 0.30
    Ly: float = 0.20
    Nx: int = 50          # grid resolution x  (was 25)
    Ny: int = 40          # grid resolution y  (was 20)
    h0: float = 3.0e-3
    # Material (steel by default; swap for spruce if you like)
    E: float = 2.1e11
    nu: float = 0.30
    rho: float = 7800.0
    # Design space
    M: int = 5            # thickness basis resolution  (was 3)
    n_modes: int = 20     # modes to keep  (was 10)
    n_actuators: int = 5  # was 3
    # Excitation point (normalized coords) representing string/bridge contact
    disturb_xy: tuple = (0.20, 0.50)
    # Frequency grid for the FRF objective
    n_freqs: int = 300    # was 200
    freq_lo: float = 40.0
    freq_hi: float = 1500.0  # was 1200
    zeta: float = 0.02    # modal damping ratio


cfg = Config()


# =============================================================================
# Spatial grid (constant)
# =============================================================================
xs = jnp.linspace(0.0, cfg.Lx, cfg.Nx)
ys = jnp.linspace(0.0, cfg.Ly, cfg.Ny)
X, Y = jnp.meshgrid(xs, ys, indexing="ij")
DX = cfg.Lx / (cfg.Nx - 1)
DY = cfg.Ly / (cfg.Ny - 1)


# =============================================================================
# Geometry: thickness parameterization
# =============================================================================
def thickness(c: jnp.ndarray) -> jnp.ndarray:
    """h(x,y) = h0 + sum_{m,n} c_{mn} sin(m pi x/Lx) sin(n pi y/Ly).
    A scaled softplus ensures h >= h_min > 0 smoothly.
    """
    ms = jnp.arange(1, cfg.M + 1)[:, None, None, None]
    ns = jnp.arange(1, cfg.M + 1)[None, :, None, None]
    basis = jnp.sin(ms * jnp.pi * X[None, None] / cfg.Lx) \
          * jnp.sin(ns * jnp.pi * Y[None, None] / cfg.Ly)
    perturb = jnp.sum(c[:, :, None, None] * basis, axis=(0, 1))
    h_raw = cfg.h0 + perturb
    h_min = 0.3 * cfg.h0
    # Smooth max(h_raw, h_min) with transition width ~h0
    h = h_min + cfg.h0 * jax.nn.softplus((h_raw - h_min) / cfg.h0)
    return h


def bending_stiffness(h: jnp.ndarray) -> jnp.ndarray:
    return cfg.E * h**3 / (12.0 * (1.0 - cfg.nu**2))


# =============================================================================
# Physics: stiffness and mass matrices
# =============================================================================
def _laplacian_matrix() -> jnp.ndarray:
    """5-point Laplacian with zero Dirichlet BCs. Approximates simply-supported
    edges (exact for w=0; del^2 w = 0 only enforced in the weak limit)."""
    e_x = jnp.ones(cfg.Nx)
    Dxx = (jnp.diag(-2 * e_x) + jnp.diag(e_x[:-1], 1) + jnp.diag(e_x[:-1], -1)) / DX**2
    Ix = jnp.eye(cfg.Nx)
    e_y = jnp.ones(cfg.Ny)
    Dyy = (jnp.diag(-2 * e_y) + jnp.diag(e_y[:-1], 1) + jnp.diag(e_y[:-1], -1)) / DY**2
    Iy = jnp.eye(cfg.Ny)
    
    # FIX: Correct Kronecker order for row-major (C-contiguous) flattening
    return jnp.kron(Dxx, Iy) + jnp.kron(Ix, Dyy)

L_LAP = _laplacian_matrix()  # (Nx*Ny, Nx*Ny), constant in c


def assemble_system(c: jnp.ndarray):
    """Return (K, M_mat) for the Kirchhoff plate generalized eigenproblem."""
    h = thickness(c)
    D = bending_stiffness(h)
    D_flat = D.flatten()
    
    # FIX: Multiply by the differential volume element (DX * DY) to properly integrate
    K = (L_LAP * D_flat[None, :]) @ L_LAP * (DX * DY)
    K = 0.5 * (K + K.T)  # numerical symmetrization
    
    mass = cfg.rho * h.flatten() * DX * DY
    M_mat = jnp.diag(mass)
    return K, M_mat


def solve_modes(c: jnp.ndarray):
    """First n_modes natural frequencies (rad/s) and M-orthonormal mode shapes.

    JAX's `eigh` only implements the standard eigenvalue problem (b=None), so
    we convert  K x = lambda M x  to standard form using the diagonal structure
    of M:  let  D = diag(sqrt(m)), then  K_tilde = D^-1 K D^-1  satisfies
    K_tilde y = lambda y  with  y = D x.  Back-transform: x = D^-1 y.
    The resulting x_i are automatically M-orthonormal (x_i^T M x_j = delta_ij).
    """
    K, M_mat = assemble_system(c)
    m_diag = jnp.diag(M_mat)
    sqrt_m_inv = 1.0 / jnp.sqrt(m_diag + 1e-30)
    K_tilde = K * sqrt_m_inv[:, None] * sqrt_m_inv[None, :]
    K_tilde = 0.5 * (K_tilde + K_tilde.T)              # numerical symmetrization
    reg = 1e-10 * jnp.eye(K_tilde.shape[0])
    w, V_tilde = jeigh(K_tilde + reg)
    w = jnp.clip(w[: cfg.n_modes], min=0.0)
    omega = jnp.sqrt(w + 1e-12)
    Phi = sqrt_m_inv[:, None] * V_tilde[:, : cfg.n_modes]  # back-transform: x = D^-1 y
    return omega, Phi


# =============================================================================
# Actuators: modal input matrix via bilinear interpolation
# =============================================================================
def _bilinear_sample(field_grid: jnp.ndarray, positions_norm: jnp.ndarray) -> jnp.ndarray:
    """field_grid: (Nx, Ny, K); positions_norm in [0,1]^2 shape (P, 2).
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


def modal_input_matrix(Phi: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Return B of shape (n_modes, n_actuators), where B[i,j] = phi_i(p_j)."""
    Phi_grid = Phi.reshape(cfg.Nx, cfg.Ny, cfg.n_modes)
    vals = _bilinear_sample(Phi_grid, positions)   # (n_act, n_modes)
    return vals.T                                  # (n_modes, n_act)


# =============================================================================
# Modal LQR: analytic 2x2 Riccati per mode (MIMO with R = r*I)
# =============================================================================
def modal_lqr_gains(omega: jnp.ndarray,
                    B: jnp.ndarray,
                    q_weight: jnp.ndarray,
                    r_weight: jnp.ndarray,
                    zeta: float = cfg.zeta) -> jnp.ndarray:
    """Closed-form LQR for each modal 2x2 subsystem with isotropic R = r*I.
    Because BB^T collapses to scalar ||B_i||^2 per mode, the Riccati has
    an analytic solution.

    Returns K_gain of shape (n_actuators, 2*n_modes), arranged so that the
    control signal is  u = -K_gain @ x  with x stacked as
        [q_0, qdot_0, q_1, qdot_1, ..., q_{n-1}, qdot_{n-1}].
    """
    b_norm_sq = jnp.sum(B**2, axis=1) + 1e-12          # (n_modes,)
    gamma = b_norm_sq / r_weight
    p12 = q_weight / (omega**2 + jnp.sqrt(omega**4 + gamma * q_weight))
    p22 = (2 * p12 + q_weight) / (2 * zeta * omega + jnp.sqrt(4 * zeta**2 * omega**2 + gamma * (2 * p12 + q_weight)))
    # Per-actuator gains: k_{p,ij} = B_{ij} * p12 / r ; k_{d,ij} = B_{ij} * p22 / r
    kp = B * (p12 / r_weight)[:, None]    # (n_modes, n_act)
    kd = B * (p22 / r_weight)[:, None]
    n = cfg.n_modes
    K_gain = jnp.zeros((cfg.n_actuators, 2 * n))
    idx_p = jnp.arange(n) * 2
    idx_d = idx_p + 1
    K_gain = K_gain.at[:, idx_p].set(kp.T)
    K_gain = K_gain.at[:, idx_d].set(kd.T)
    return K_gain


# =============================================================================
# Closed-loop assembly and frequency response
# =============================================================================
def closed_loop(omega: jnp.ndarray,
                B: jnp.ndarray,
                K_gain: jnp.ndarray,
                b_dist: jnp.ndarray,
                zeta: float = cfg.zeta):
    """Assemble A_cl, B_d, C_out in stacked modal state space."""
    n = cfg.n_modes
    idx_p = jnp.arange(n) * 2
    idx_d = idx_p + 1
    A = jnp.zeros((2 * n, 2 * n))
    A = A.at[idx_p, idx_d].set(1.0)
    A = A.at[idx_d, idx_p].set(-omega**2)
    A = A.at[idx_d, idx_d].set(-2 * zeta * omega)
    B_u = jnp.zeros((2 * n, cfg.n_actuators))
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
        # Prevent NaN gradients at anti-resonance zeros
        return jnp.abs(C_out[0] @ resp + 1e-12 + 1e-12j)

    return jax.vmap(_one)(omega_rad)


# =============================================================================
# Objective
# =============================================================================
def loss_fn(params, target_spectrum, target_freqs):
    c, positions, log_q, log_r = params
    q_weight = jnp.exp(log_q)
    r_weight = jnp.exp(log_r)

    omega, Phi = solve_modes(c)
    B = modal_input_matrix(Phi, positions)                       # (n_modes, n_act)

    disturb_pos = jnp.array([cfg.disturb_xy])
    b_dist = modal_input_matrix(Phi, disturb_pos)[:, 0]          # (n_modes,)

    K_gain = modal_lqr_gains(omega, B, q_weight, r_weight)
    A_cl, B_d, C_out = closed_loop(omega, B, K_gain, b_dist)

    Hmag = frf_magnitude(A_cl, B_d, C_out, target_freqs)
    Hn = Hmag / (jnp.max(Hmag) + 1e-12)
    Tn = target_spectrum / (jnp.max(target_spectrum) + 1e-12)
    spec_loss = jnp.mean((Hn - Tn) ** 2)

    # Keep actuators inside the plate with a margin
    pos_pen = jnp.sum(jax.nn.relu(0.05 - positions)
                      + jax.nn.relu(positions - 0.95))
    return spec_loss + 10.0 * pos_pen


# =============================================================================
# Optimization
# =============================================================================
def target_spectrum_example():
    freqs = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)
    centers = jnp.array([120.0, 300.0, 620.0])
    widths = jnp.array([15.0, 20.0, 30.0])
    spec = sum(jnp.exp(-((freqs - ci) / wi) ** 2) for ci, wi in zip(centers, widths))
    return freqs, spec


def run(num_steps: int = 400, seed: int = 0, verbose: bool = True):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    c0 = 1e-4 * jax.random.normal(k1, (cfg.M, cfg.M))
    pos0 = jnp.clip(0.5 + 0.15 * jax.random.normal(k2, (cfg.n_actuators, 2)), 0.1, 0.9)
    log_q0 = jnp.log(1.0)
    log_r0 = jnp.log(1.0)
    params = (c0, pos0, log_q0, log_r0)

    freqs, target = target_spectrum_example()

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Prevents resonance gradient spikes
        optax.adam(1e-2)
    )
    state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    history = []
    for step in range(num_steps):
        loss, grads = value_and_grad(params, target, freqs)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if verbose and step % 25 == 0:
            print(f"step {step:4d}   loss={float(loss):.4e}")
    return params, history, freqs, target


if __name__ == "__main__":
    final_params, hist, freqs, target = run()
    print(f"Done. Final loss: {hist[-1]:.4e}")