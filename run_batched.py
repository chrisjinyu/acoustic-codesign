"""Batched multi-start co-design, mode-aware.

Runs N parallel optimization instances from different random initializations
using jax.vmap. Supports both variants:

  --mode lqr      (default): plate + actuator placement + LQR gains
  --mode strings            : plate + string tensions

Usage:
    python run_batched.py                              # LQR, 8 seeds, 600 steps
    python run_batched.py --mode strings
    python run_batched.py --mode strings --seeds 16 --steps 800
    python run_batched.py --mode lqr --centers 105 190 245 --widths 30 35 35
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


def load_mode(mode: str,
              lqr_centers: tuple = (105.0, 190.0, 245.0),
              lqr_widths:   tuple = (30.0,  35.0,  35.0)):
    """Return (module, loss_call, target_fn) for the requested variant."""
    if mode == "lqr":
        mod = importlib.import_module("codesign")
        def loss_call(params, freqs, target):
            return mod.loss_fn(params, target, freqs)
        def target_fn():
            return mod.target_spectrum_fixed(
                centers_hz=lqr_centers,
                widths_hz=lqr_widths,
            )
        return mod, loss_call, target_fn

    elif mode == "strings":
        mod = importlib.import_module("codesign_strings")
        sc = mod.scfg  # snapshot so loss and init always agree
        def loss_call(params, freqs, target):
            return mod.loss_fn(params, freqs, sc=sc)
        def target_fn():
            freqs  = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)
            log_T0 = mod.initial_log_tensions(sc)
            target = mod.target_spectrum_from_strings(log_T0, freqs, sc)
            return freqs, target
        import functools
        mod.init_params_one = functools.partial(mod.init_params_one, sc=sc)
        return mod, loss_call, target_fn

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'lqr' or 'strings'.")


def run_multistart(mode: str = "lqr",
                   n_seeds: int = 8,
                   num_steps: int = 600,
                   lr: float = 5e-3,
                   lqr_centers: tuple = (105.0, 190.0, 245.0),
                   lqr_widths:   tuple = (30.0,  35.0,  35.0),
                   verbose: bool = True):
    """Vmapped optimization over n_seeds independent initializations.

    Matches the demo notebook exactly:
      - Adam with gradient norm clipping at 1.0
      - lr=5e-3 default
      - Best-loss tracking (not final-step loss)
    """
    mod, loss_call, target_fn = load_mode(mode, lqr_centers, lqr_widths)
    keys = jax.random.split(jax.random.PRNGKey(42), n_seeds)
    params = jax.vmap(mod.init_params_one)(keys)

    # Match the notebook optimizer exactly
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    opt_state = jax.vmap(optimizer.init)(params)
    freqs, target = target_fn()

    def one_step(params_i, opt_state_i, target, freqs):
        loss, grads = jax.value_and_grad(loss_call)(params_i, freqs, target)
        updates, new_opt = optimizer.update(grads, opt_state_i, params_i)
        new_params = optax.apply_updates(params_i, updates)
        return new_params, new_opt, loss

    batched_step = jax.jit(jax.vmap(one_step, in_axes=(0, 0, None, None)))

    history = []
    # Best-loss tracking per seed (matches notebook behaviour)
    best_losses = np.full(n_seeds, np.inf)
    best_params = params

    t0 = time.time()
    for step in range(num_steps):
        params, opt_state, losses = batched_step(params, opt_state, target, freqs)
        losses_np = np.asarray(losses)
        history.append(losses_np)

        # Update best params for each seed independently
        improved = losses_np < best_losses
        if improved.any():
            best_losses = np.where(improved, losses_np, best_losses)
            best_params = jax.tree.map(
                lambda b, p: jnp.where(
                    jnp.array(improved).reshape((-1,) + (1,) * (b.ndim - 1)),
                    p, b
                ),
                best_params, params
            )

        if verbose and step % 25 == 0:
            print(f"step {step:4d}  "
                  f"min={losses_np.min():.3e}  "
                  f"mean={losses_np.mean():.3e}  "
                  f"std={losses_np.std():.3e}")

    elapsed = time.time() - t0
    history = np.stack(history)   # (num_steps, n_seeds)

    # Best seed by best-loss, not final-step loss
    best_seed       = int(np.argmin(best_losses))
    best_params_one = jax.tree.map(lambda x: x[best_seed], best_params)

    print()
    print(f"Mode           : {mode}")
    print(f"Wall time      : {elapsed:.1f}s for {n_seeds} runs x {num_steps} steps")
    print(f"Per-run time   : {elapsed / n_seeds:.2f}s effective")
    print(f"Best loss/seed : min={best_losses.min():.3e}  "
          f"mean={best_losses.mean():.3e}  std={best_losses.std():.3e}")
    print(f"Best seed      : {best_seed}")

    return best_params_one, best_params, history, best_losses, freqs, target, best_seed


def save_outputs(mode: str, best_params, all_params,
                 history, best_losses,
                 freqs, target, best_seed,
                 outfile: str = None):
    """Save results to a mode-specific .npz."""
    if outfile is None:
        outfile = f"multistart_{mode}.npz"

    common = dict(
        history=history,           # (steps, seeds) -- full trajectory
        best_losses=best_losses,   # (seeds,)        -- best per seed
        freqs=np.asarray(freqs),
        target=np.asarray(target),
        best_seed=best_seed,
    )

    if mode == "lqr":
        c, positions, log_q, log_r = best_params
        np.savez(outfile, mode=mode,
                 c=np.asarray(c),
                 positions=np.asarray(positions),
                 log_q=np.asarray(log_q),
                 log_r=np.asarray(log_r),
                 **common)
    elif mode == "strings":
        c, log_tensions = best_params
        np.savez(outfile, mode=mode,
                 c=np.asarray(c),
                 log_tensions=np.asarray(log_tensions),
                 **common)

    print(f"Saved to {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--seeds",   type=int,   default=8)
    parser.add_argument("--steps",   type=int,   default=600)
    parser.add_argument("--lr",      type=float, default=5e-3)
    parser.add_argument("--outfile", type=str,   default=None)
    parser.add_argument("--centers", type=float, nargs="+",
                        default=[105.0, 190.0, 245.0],
                        help="LQR target center frequencies in Hz")
    parser.add_argument("--widths",  type=float, nargs="+",
                        default=[30.0, 35.0, 35.0],
                        help="LQR target Gaussian widths in Hz")
    args = parser.parse_args()

    assert len(args.centers) == len(args.widths), \
        "Number of --centers and --widths must match."

    print(f"JAX backend  : {jax.default_backend()}")
    print(f"Profile      : Nx={cfg.Nx} Ny={cfg.Ny} M={cfg.M} "
          f"n_modes={cfg.n_modes} material={cfg.material}")
    print(f"Mode         : {args.mode}")
    if args.mode == "lqr":
        print(f"LQR targets  : {args.centers} Hz  (widths {args.widths} Hz)")
    print(f"lr           : {args.lr}")
    print(f"Running {args.seeds} parallel searches for {args.steps} steps...")
    print()

    (best_params, all_params, history, best_losses,
     freqs, target, best_seed) = run_multistart(
        mode=args.mode,
        n_seeds=args.seeds,
        num_steps=args.steps,
        lr=args.lr,
        lqr_centers=tuple(args.centers),
        lqr_widths=tuple(args.widths),
    )
    save_outputs(args.mode, best_params, all_params,
                 history, best_losses,
                 freqs, target, best_seed,
                 outfile=args.outfile)


if __name__ == "__main__":
    main()