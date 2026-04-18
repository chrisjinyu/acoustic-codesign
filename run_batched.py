"""Batched multi-start co-design.

Runs N parallel optimization instances from different random initializations
using `jax.vmap`. On a GPU, N runs cost roughly the same wall-clock as 1 up to
the point where memory fills, so you get:

  * variance statistics (mean, std of final loss) for your report
  * a better best-of-N outcome than any single run
  * a workload that actually saturates the card

Usage:
    python run_batched.py                  # defaults: N=8 seeds, 400 steps
    python run_batched.py --seeds 16       # more parallelism
    python run_batched.py --steps 800      # longer optimization
"""
from __future__ import annotations
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

import codesign
from codesign import cfg, loss_fn, target_spectrum_example


# -----------------------------------------------------------------------------
def init_params_one(key):
    """Build one initial params pytree from a single PRNG key."""
    k1, k2 = jax.random.split(key)
    c = 1e-4 * jax.random.normal(k1, (cfg.M, cfg.M))
    pos = jnp.clip(0.5 + 0.15 * jax.random.normal(k2, (cfg.n_actuators, 2)),
                   0.1, 0.9)
    log_q = jnp.log(1.0)
    log_r = jnp.log(1.0)
    return (c, pos, log_q, log_r)


# -----------------------------------------------------------------------------
def run_multistart(n_seeds: int = 8,
                   num_steps: int = 400,
                   lr: float = 1e-2,
                   verbose: bool = True):
    """Vmapped optimization over `n_seeds` independent initializations."""
    keys = jax.random.split(jax.random.PRNGKey(42), n_seeds)
    params = jax.vmap(init_params_one)(keys)

    optimizer = optax.adam(lr)
    opt_state = jax.vmap(optimizer.init)(params)

    freqs, target = target_spectrum_example()

    def one_step(params_i, opt_state_i, target, freqs):
        loss, grads = jax.value_and_grad(loss_fn)(params_i, target, freqs)
        updates, new_opt = optimizer.update(grads, opt_state_i, params_i)
        new_params = optax.apply_updates(params_i, updates)
        return new_params, new_opt, loss

    batched_step = jax.jit(jax.vmap(one_step, in_axes=(0, 0, None, None)))

    history = []
    t0 = time.time()
    for step in range(num_steps):
        params, opt_state, losses = batched_step(params, opt_state, target, freqs)
        history.append(np.asarray(losses))
        if verbose and step % 25 == 0:
            loss_min = float(jnp.min(losses))
            loss_mean = float(jnp.mean(losses))
            loss_std = float(jnp.std(losses))
            print(f"step {step:4d}  min={loss_min:.3e}  mean={loss_mean:.3e}  std={loss_std:.3e}")
    elapsed = time.time() - t0
    history = np.stack(history)  # (num_steps, n_seeds)

    final = history[-1]
    best_seed = int(np.argmin(final))
    best_params = jax.tree.map(lambda x: x[best_seed], params)

    print()
    print(f"Wall time   : {elapsed:.1f}s for {n_seeds} runs x {num_steps} steps")
    print(f"Per-run time: {elapsed / n_seeds:.2f}s effective")
    print(f"Final loss  : min={final.min():.3e}  mean={final.mean():.3e}  std={final.std():.3e}")
    print(f"Best seed   : {best_seed}")

    return best_params, params, history, freqs, target


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Profile    : Nx={cfg.Nx} Ny={cfg.Ny} M={cfg.M} n_modes={cfg.n_modes} "
          f"n_actuators={cfg.n_actuators}")
    print(f"Running {args.seeds} parallel searches for {args.steps} steps...")
    print()

    best_params, all_params, history, freqs, target = run_multistart(
        n_seeds=args.seeds, num_steps=args.steps, lr=args.lr
    )

    # Save the best result and trajectory for downstream plotting
    np.savez("multistart_result.npz",
             c=np.asarray(best_params[0]),
             positions=np.asarray(best_params[1]),
             log_q=np.asarray(best_params[2]),
             log_r=np.asarray(best_params[3]),
             history=history,
             freqs=np.asarray(freqs),
             target=np.asarray(target))
    print("Saved best params and history to multistart_result.npz")


if __name__ == "__main__":
    main()
