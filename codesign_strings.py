"""
codesign_strings.py -- plate geometry + string tensions co-design.

Physical framing:
  * The spruce plate is the passive resonator (physical plant).
  * The strings are the active excitation mechanism. Their tensions set which
    harmonic frequencies get injected into the plate at the bridge. Under an
    automated-tuning framing (motorized pegs), string tensions are a control
    input that can be co-optimized with the plate.

Co-design variables:
  * c            -- thickness coefficients  (shape (M, M))
  * log_tensions -- per-string tension in log space  (shape (n_strings,))

The target spectrum is derived from the current string fundamentals; as the
optimizer adjusts tensions, the target moves. A separate pitch penalty keeps
the strings near the user-requested nominal pitches (e.g., C#4/A4/C#5).
"""
from __future__ import annotations
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax

from codesign_core import (
    cfg,
    solve_modes,
    modal_values_at_points,
    closed_loop,
    frf_magnitude,
)


# =============================================================================
# String physics
# =============================================================================
@dataclass(frozen=True)
class StringConfig:
    """Parameters of the vibrating string set.

    Defaults: violin-scale instrument with steel-core strings at 0.3 mm
    diameter. `target_pitches_hz` defines the nominal notes the strings are
    supposed to produce; the optimizer may deviate slightly in exchange for
    better plate coupling, subject to `pitch_penalty_weight`.
    """
    L: float = 0.328                   # vibrating length (m)
    rho_string: float = 7800.0         # kg/m^3
    diameter: float = 3.0e-4           # m
    target_pitches_hz: tuple = (275.0, 440.0, 540.0)  # C#4, A4, C#5
    target_width_hz: float = 40.0      # Gaussian width of each peak
    n_harmonics: int = 3               # how many harmonics to include per string
    harmonic_falloff: float = 0.5      # amplitude factor for each overtone

    @property
    def mu(self) -> float:
        """Linear mass density (kg/m) from diameter and string density."""
        return self.rho_string * jnp.pi * (self.diameter / 2) ** 2

    @property
    def n_strings(self) -> int:
        return len(self.target_pitches_hz)


scfg = StringConfig()


def string_fundamentals(log_tensions: jnp.ndarray,
                        sc: StringConfig = scfg) -> jnp.ndarray:
    """Fundamental frequency f_0 = (1/2L) sqrt(T/mu) for each string."""
    T = jnp.exp(log_tensions)
    return (1.0 / (2 * sc.L)) * jnp.sqrt(T / sc.mu)


def initial_log_tensions(sc: StringConfig = scfg) -> jnp.ndarray:
    """Tensions that produce exactly the target pitches (optimizer start)."""
    pitches = jnp.array(sc.target_pitches_hz)
    T = sc.mu * (2 * sc.L * pitches) ** 2
    return jnp.log(T)


def target_spectrum_from_strings(log_tensions: jnp.ndarray,
                                 freqs: jnp.ndarray,
                                 sc: StringConfig = scfg) -> jnp.ndarray:
    """Gaussian peaks at the fundamental of each string plus harmonics."""
    f0 = string_fundamentals(log_tensions, sc)   # (n_strings,)
    width = sc.target_width_hz

    spec = jnp.zeros_like(freqs)
    for k in range(1, sc.n_harmonics + 1):
        amp = sc.harmonic_falloff ** (k - 1)
        for i in range(sc.n_strings):
            center = k * f0[i]
            in_range = center < cfg.freq_hi  # ignore harmonics above the grid
            contribution = amp * jnp.exp(-((freqs - center) / width) ** 2)
            spec = spec + jnp.where(in_range, contribution, 0.0)
    return spec


def pitch_penalty(log_tensions: jnp.ndarray,
                  sc: StringConfig = scfg) -> jnp.ndarray:
    """Relative squared error between string fundamentals and target pitches."""
    f0 = string_fundamentals(log_tensions, sc)
    target = jnp.array(sc.target_pitches_hz)
    return jnp.mean(((f0 - target) / target) ** 2)


# =============================================================================
# Loss: plate + strings, no LQR controller
# =============================================================================
def loss_fn(params, freqs,
            pitch_penalty_weight: float = 5.0,
            suppression_weight: float = 0.5,
            sc: StringConfig = scfg):
    """Plate + strings co-design loss. Note the signature differs from the
    LQR loss: the target spectrum is derived internally from log_tensions
    rather than passed in.
    """
    c, log_tensions = params

    # Target spectrum moves with the strings
    target = target_spectrum_from_strings(log_tensions, freqs, sc)

    # Passive plate FRF at the bridge point
    omega, Phi = solve_modes(c)
    bridge = jnp.array([cfg.disturb_xy])
    b_dist = modal_values_at_points(Phi, bridge)[:, 0]
    B_zero = jnp.zeros((cfg.n_modes, 1))
    K_zero = jnp.zeros((1, 2 * cfg.n_modes))
    A_cl, B_d, C_out = closed_loop(omega, B_zero, K_zero, b_dist)
    Hmag = frf_magnitude(A_cl, B_d, C_out, freqs)

    # Normalized spectrum matching
    Hn = Hmag / (jnp.max(Hmag) + 1e-12)
    Tn = target / (jnp.max(target) + 1e-12)
    off_target_mask = (target < 0.05)
    suppression_loss = jnp.mean(Hn ** 2 * off_target_mask)
    spec_loss = jnp.mean((Hn - Tn) ** 2) + suppression_weight * suppression_loss

    # Strings should land near the requested pitches
    pitch_pen = pitch_penalty(log_tensions, sc)

    return spec_loss + pitch_penalty_weight * pitch_pen


# =============================================================================
# Init and run
# =============================================================================
def init_params_one(key, sc: StringConfig = scfg):
    """Initialize (c, log_tensions) from a single PRNG key."""
    c = 1e-4 * jax.random.normal(key, (cfg.M, cfg.M))
    log_T = initial_log_tensions(sc)
    # Small perturbation so multi-start seeds differ in tension as well
    k2, _ = jax.random.split(key)
    log_T = log_T + 0.05 * jax.random.normal(k2, log_T.shape)
    return (c, log_T)


def run(num_steps: int = 400,
        seed: int = 0,
        verbose: bool = True,
        sc: StringConfig = scfg,
        lr: float = 5e-3):
    params = init_params_one(jax.random.PRNGKey(seed), sc)
    freqs = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)

    # Bind sc into a closure rather than relying on loss_fn's default argument.
    # This is critical: loss_fn's default sc is captured at function definition
    # time and can become stale if the module-level scfg is replaced (e.g.
    # after importlib.reload). Binding here ensures init_params_one and
    # loss_fn always see exactly the same sc object.
    def _loss(params, freqs):
        return loss_fn(params, freqs, sc=sc)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(_loss))

    history = []
    best_loss = float('inf')
    best_params = params

    for step in range(num_steps):
        loss, grads = value_and_grad(params, freqs)
        if loss < best_loss:
            best_loss = float(loss)
            best_params = params
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if verbose and step % 25 == 0:
            f0 = string_fundamentals(params[1], sc)
            f0_str = ", ".join(f"{float(f):.1f}" for f in f0)
            print(f"step {step:4d}   loss={float(loss):.4e}   "
                  f"pitches=[{f0_str}] Hz")

    final_target = target_spectrum_from_strings(best_params[1], freqs, sc)
    return best_params, best_loss, history, freqs, final_target


if __name__ == "__main__":
    best_params, best_loss, hist, freqs, target = run()
    print(f"Done. Best loss: {best_loss:.4e}")
    print(f"Final pitches: {string_fundamentals(best_params[1])}")