"""
codesign.py -- LQR + piezo actuator variant.

Co-design: plate geometry (thickness field) + actuator positions + LQR weights.
Target: fixed Gaussian spectrum.

Re-exports commonly used symbols from codesign_core so existing callers that
do `import codesign; codesign.solve_modes(...)` continue to work.

Ablation helpers (added for the report's co-design validation):
  * run_fixed_qr            -- optimize (c, positions) with Q,R held constant
  * run_sequential_stage2   -- optimize (positions, log_q, log_r) with c frozen

These support the three-way ladder used in the report:
    Passive < Fixed-QR <= Sequential <= Joint
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import optax

from codesign_core import (
    cfg,
    solve_modes,
    modal_values_at_points,
    closed_loop,
    frf_magnitude,
    frf_passive,
    thickness,
    target_spectrum_fixed,
)

# Alias for the LQR variant; the old code called this "modal_input_matrix"
modal_input_matrix = modal_values_at_points


# =============================================================================
# Modal LQR: analytic 2x2 Riccati per mode, MIMO with R = r*I
# =============================================================================
def modal_lqr_gains(omega: jnp.ndarray,
                    B: jnp.ndarray,
                    q_weight: jnp.ndarray,
                    r_weight: jnp.ndarray,
                    zeta: float = cfg.zeta) -> jnp.ndarray:
    """Closed-form LQR gains stacked over all modes."""
    b_norm_sq = jnp.sum(B**2, axis=1) + 1e-12
    gamma = b_norm_sq / r_weight
    inner_1 = omega**4 + gamma * q_weight + 1e-12
    p12 = q_weight / (omega**2 + jnp.sqrt(inner_1))
    inner_2 = 4 * zeta**2 * omega**2 + gamma * (2 * p12 + q_weight) + 1e-12
    p22 = (2 * p12 + q_weight) / (2 * zeta * omega + jnp.sqrt(inner_2))
    kp = B * (p12 / r_weight)[:, None]
    kd = B * (p22 / r_weight)[:, None]
    n = cfg.n_modes
    K_gain = jnp.zeros((cfg.n_actuators, 2 * n))
    idx_p = jnp.arange(n) * 2
    idx_d = idx_p + 1
    K_gain = K_gain.at[:, idx_p].set(kp.T)
    K_gain = K_gain.at[:, idx_d].set(kd.T)
    return K_gain


# =============================================================================
# Actuator Repulsion Incentive
# =============================================================================
def actuator_repulsion(positions, min_sep: float = 0.15):
    """Penalize actuator pairs closer than min_sep (in normalized coords).
    min_sep=0.15 means actuators must be at least 15% of the plate apart.
    """
    n = positions.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = jnp.sum((positions[i] - positions[j])**2)
            total += jax.nn.relu(min_sep**2 - dist_sq)
    return total


# =============================================================================
# Loss
# =============================================================================
def loss_fn(params, target_spectrum, target_freqs):
    c, positions, log_q, log_r = params
    q_weight = jnp.exp(log_q)
    r_weight = jnp.exp(log_r)

    omega, Phi = solve_modes(c)
    B = modal_input_matrix(Phi, positions)
    disturb_pos = jnp.array([cfg.disturb_xy])
    b_dist = modal_input_matrix(Phi, disturb_pos)[:, 0]

    K_gain = modal_lqr_gains(omega, B, q_weight, r_weight)
    A_cl, B_d, C_out = closed_loop(omega, B, K_gain, b_dist)
    Hmag = frf_magnitude(A_cl, B_d, C_out, target_freqs)

    Hn = Hmag / (jnp.max(Hmag) + 1e-12)
    off_target_mask = (target_spectrum < 0.05)
    suppression_loss = jnp.mean(Hn**2 * off_target_mask)
    Tn = target_spectrum / (jnp.max(target_spectrum) + 1e-12)
    spec_loss = jnp.mean((Hn - Tn)**2) + 0.5 * suppression_loss

    pos_pen = jnp.sum(jax.nn.relu(0.05 - positions)
                      + jax.nn.relu(positions - 0.95))
    repulsion = actuator_repulsion(positions, min_sep=0.15)
    return spec_loss + 10.0 * pos_pen + 5.0 * repulsion


# =============================================================================
# Init and run (joint co-design)
# =============================================================================
def init_params_one(key):
    """Build one initial params pytree from a single PRNG key."""
    k1, k2 = jax.random.split(key)
    c = 1e-4 * jax.random.normal(k1, (cfg.M, cfg.M))
    pos = jnp.clip(0.5 + 0.15 * jax.random.normal(k2, (cfg.n_actuators, 2)),
                   0.1, 0.9)
    log_q = jnp.log(1.0)
    log_r = jnp.log(1.0)
    return (c, pos, log_q, log_r)


def run(num_steps: int = 400,
        seed: int = 0,
        lr: float = 5e-3,
        centers_hz: tuple = (280.0, 470.0, 530.0),
        widths_hz: tuple = (40.0, 40.0, 40.0),
        verbose: bool = True):
    """Joint co-design of plate geometry, actuator positions, and LQR weights.

    The target spectrum is built directly from centers_hz and widths_hz and
    passed explicitly into every loss evaluation. There is no module-level
    state to override; changing the target is done by passing different values
    here.

    Parameters
    ----------
    centers_hz : Gaussian peak centers for the target spectrum (Hz).
                 Defaults match the violin body resonance targets in the notebook.
    widths_hz  : Gaussian half-widths corresponding to each center (Hz).
    """
    params = init_params_one(jax.random.PRNGKey(seed))
    freqs, target = target_spectrum_fixed(centers_hz=centers_hz,
                                          widths_hz=widths_hz)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    history = []
    best_loss = float('inf')
    best_params = params

    for step in range(num_steps):
        loss, grads = value_and_grad(params, target, freqs)
        if loss < best_loss:
            best_loss = float(loss)
            best_params = params
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if verbose and step % 25 == 0:
            print(f"step {step:4d}   loss={float(loss):.4e}")

    return best_params, best_loss, history, freqs, target


# =============================================================================
# Ablation A: Fixed Q/R (co-design of geometry + actuators only)
# ---------------------------------------------------------------------------
# Isolates the value of treating LQR weights as design variables. If
# run_fixed_qr converges to a loss close to the joint run, the Q/R
# optimization is providing little marginal benefit. If the joint run is
# clearly better, the Q/R design variables matter.
# =============================================================================
def loss_fn_fixed_qr(params_2, target_spectrum, target_freqs,
                     log_q_const: float = 0.0, log_r_const: float = 0.0):
    """Loss with (c, positions) optimized and Q,R held at fixed log values."""
    c, positions = params_2
    lq = jnp.asarray(log_q_const, dtype=c.dtype)
    lr = jnp.asarray(log_r_const, dtype=c.dtype)
    return loss_fn((c, positions, lq, lr), target_spectrum, target_freqs)


def run_fixed_qr(target, freqs,
                 num_steps: int = 600,
                 seed: int = 0,
                 lr: float = 5e-3,
                 log_q_const: float = 0.0,
                 log_r_const: float = 0.0,
                 verbose: bool = True):
    """Optimize (c, positions) with Q and R held fixed.

    Returns
    -------
    c_best, positions_best, best_loss, history
    """
    full_init = init_params_one(jax.random.PRNGKey(seed))
    params_2 = (full_init[0], full_init[1])

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = optimizer.init(params_2)

    def _loss(p2, tgt, frq):
        return loss_fn_fixed_qr(p2, tgt, frq,
                                log_q_const=log_q_const,
                                log_r_const=log_r_const)

    value_and_grad = jax.jit(jax.value_and_grad(_loss))

    history = []
    best_loss = float('inf')
    best_params = params_2

    for step in range(num_steps):
        loss, grads = value_and_grad(params_2, target, freqs)
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_params = params_2
        updates, state = optimizer.update(grads, state, params_2)
        params_2 = optax.apply_updates(params_2, updates)
        history.append(float(loss))
        if verbose and step % 50 == 0:
            print(f"  [fixed-QR]  step {step:4d}   loss={float(loss):.4e}")

    c_best, pos_best = best_params
    return c_best, pos_best, best_loss, history


# =============================================================================
# Ablation B: Sequential co-design (Stage 2 only)
# ---------------------------------------------------------------------------
# Stage 1 is the passive geometry optimization run elsewhere; its output
# c_passive is passed in as c_fixed. Stage 2 then optimizes
# (positions, log_q, log_r) on that frozen geometry. This is the classical
# "plant then controller" workflow that co-design claims to beat.
# =============================================================================
def loss_fn_controller_only(params_3, c_fixed, target_spectrum, target_freqs):
    """Loss with plate geometry frozen. Optimize (positions, log_q, log_r)."""
    positions, log_q, log_r = params_3
    return loss_fn((c_fixed, positions, log_q, log_r),
                   target_spectrum, target_freqs)


def run_sequential_stage2(c_fixed,
                          target, freqs,
                          num_steps: int = 600,
                          seed: int = 0,
                          lr: float = 5e-3,
                          verbose: bool = True):
    """Stage 2 of sequential co-design: freeze c, optimize the controller.

    Parameters
    ----------
    c_fixed : (M, M) array
        Thickness coefficients from Stage 1 (passive geometry optimization).
    target, freqs : target spectrum and frequency grid, passed explicitly.

    Returns
    -------
    positions_best, log_q_best, log_r_best, best_loss, history
    """
    full_init = init_params_one(jax.random.PRNGKey(seed))
    params_3 = (full_init[1], full_init[2], full_init[3])  # pos, log_q, log_r

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = optimizer.init(params_3)

    c_fixed_j = jnp.asarray(c_fixed)

    def _loss(p3, tgt, frq):
        return loss_fn_controller_only(p3, c_fixed_j, tgt, frq)

    value_and_grad = jax.jit(jax.value_and_grad(_loss))

    history = []
    best_loss = float('inf')
    best_params = params_3

    for step in range(num_steps):
        loss, grads = value_and_grad(params_3, target, freqs)
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_params = params_3
        updates, state = optimizer.update(grads, state, params_3)
        params_3 = optax.apply_updates(params_3, updates)
        history.append(float(loss))
        if verbose and step % 50 == 0:
            print(f"  [seq stage2]  step {step:4d}   loss={float(loss):.4e}")

    pos_best, log_q_best, log_r_best = best_params
    return pos_best, log_q_best, log_r_best, best_loss, history


if __name__ == "__main__":
    best_params, best_loss, hist, freqs, target = run()
    print(f"Done. Best loss: {best_loss:.4e}")