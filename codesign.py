"""
codesign.py -- LQR + piezo actuator variant.

Co-design: plate geometry (thickness field) + actuator positions + LQR weights.
Target: fixed Gaussian spectrum.

Re-exports commonly used symbols from codesign_core so existing callers that
do `import codesign; codesign.solve_modes(...)` continue to work.
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

# Backwards-compat re-export of the fixed target generator
target_spectrum_example = target_spectrum_fixed

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
    return spec_loss + 10.0 * pos_pen


# =============================================================================
# Init and run
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


def run(num_steps: int = 400, seed: int = 0, verbose: bool = True):
    params = init_params_one(jax.random.PRNGKey(seed))
    freqs, target = target_spectrum_fixed()

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2))
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


if __name__ == "__main__":
    best_params, best_loss, hist, freqs, target = run()
    print(f"Done. Best loss: {best_loss:.4e}")
