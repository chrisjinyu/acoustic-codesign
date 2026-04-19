"""Batched multi-start co-design, mode-aware.

Runs N parallel optimization instances from different random initializations
using jax.vmap. Supports both variants:

  --mode lqr      (default): plate + actuator placement + LQR gains
  --mode strings            : plate + string tensions

Usage:
    python run_batched.py                            # LQR, 8 seeds, 400 steps
    python run_batched.py --mode strings
    python run_batched.py --mode strings --seeds 16 --steps 600
"""
from __future__ import annotations
import argparse
import importlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from codesign_core import cfg


def load_mode(mode: str):
    """Return (module, loss_call, target_fn) for the requested variant.

    loss_call(params, freqs, target) -> scalar loss. The strings loss ignores
    `target` (it derives its own internally) but we keep a uniform signature
    so the vmapped step function is shared across modes.
    """
    if mode == "lqr":
        mod = importlib.import_module("codesign")
        def loss_call(params, freqs, target):
            return mod.loss_fn(params, target, freqs)
        def target_fn():
            return mod.target_spectrum_fixed()
        return mod, loss_call, target_fn
    elif mode == "strings":
        mod = importlib.import_module("codesign_strings")
        sc = mod.scfg  # snapshot current scfg so loss and init always agree
        def loss_call(params, freqs, target):
            return mod.loss_fn(params, freqs, sc=sc)
        def target_fn():
            freqs = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)
            log_T0 = mod.initial_log_tensions(sc)
            target = mod.target_spectrum_from_strings(log_T0, freqs, sc)
            return freqs, target
        # Wrap init so vmap can call it without the sc argument
        import functools
        mod.init_params_one = functools.partial(mod.init_params_one, sc=sc)
        return mod, loss_call, target_fn
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'lqr' or 'strings'.")


def run_multistart(mode: str = "lqr",
                   n_seeds: int = 8,
                   num_steps: int = 400,
                   lr: float = 1e-2,
                   verbose: bool = True):
    mod, loss_call, target_fn = load_mode(mode)
    keys = jax.random.split(jax.random.PRNGKey(42), n_seeds)
    params = jax.vmap(mod.init_params_one)(keys)

    optimizer = optax.adam(lr)
    opt_state = jax.vmap(optimizer.init)(params)
    freqs, target = target_fn()

    def one_step(params_i, opt_state_i, target, freqs):
        loss, grads = jax.value_and_grad(loss_call)(params_i, freqs, target)
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
            print(f"step {step:4d}  "
                  f"min={float(jnp.min(losses)):.3e}  "
                  f"mean={float(jnp.mean(losses)):.3e}  "
                  f"std={float(jnp.std(losses)):.3e}")

    elapsed = time.time() - t0
    history = np.stack(history)
    final = history[-1]
    best_seed = int(np.argmin(final))
    best_params = jax.tree.map(lambda x: x[best_seed], params)

    print()
    print(f"Mode        : {mode}")
    print(f"Wall time   : {elapsed:.1f}s for {n_seeds} runs x {num_steps} steps")
    print(f"Per-run time: {elapsed / n_seeds:.2f}s effective")
    print(f"Final loss  : min={final.min():.3e}  "
          f"mean={final.mean():.3e}  std={final.std():.3e}")
    print(f"Best seed   : {best_seed}")

    return best_params, params, history, freqs, target, best_seed


def save_outputs(mode: str, best_params, history, freqs, target, best_seed,
                 outfile: str = None):
    """Save results to a mode-specific .npz."""
    if outfile is None:
        outfile = f"multistart_{mode}.npz"

    if mode == "lqr":
        c, positions, log_q, log_r = best_params
        np.savez(outfile,
                 mode=mode,
                 c=np.asarray(c),
                 positions=np.asarray(positions),
                 log_q=np.asarray(log_q),
                 log_r=np.asarray(log_r),
                 history=history,
                 freqs=np.asarray(freqs),
                 target=np.asarray(target),
                 best_seed=best_seed)
    elif mode == "strings":
        c, log_tensions = best_params
        np.savez(outfile,
                 mode=mode,
                 c=np.asarray(c),
                 log_tensions=np.asarray(log_tensions),
                 history=history,
                 freqs=np.asarray(freqs),
                 target=np.asarray(target),
                 best_seed=best_seed)
    print(f"Saved to {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Profile    : Nx={cfg.Nx} Ny={cfg.Ny} M={cfg.M} "
          f"n_modes={cfg.n_modes} material={cfg.material}")
    print(f"Mode       : {args.mode}")
    print(f"Running {args.seeds} parallel searches for {args.steps} steps...")
    print()

    best_params, _, history, freqs, target, best_seed = run_multistart(
        mode=args.mode, n_seeds=args.seeds,
        num_steps=args.steps, lr=args.lr,
    )
    save_outputs(args.mode, best_params, history, freqs, target, best_seed,
                 outfile=args.outfile)


if __name__ == "__main__":
    main()