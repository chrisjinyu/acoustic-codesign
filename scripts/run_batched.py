"""Batched multi-start co-design, mode-aware.

Runs N parallel optimization instances from different random initializations
using jax.vmap. Supports both variants:

  --mode lqr      (default): plate + actuator placement + LQR gains
  --mode strings            : plate + string tensions

Usage (from repo root):
    python scripts/run_batched.py                              # LQR, 8 seeds, 600 steps
    python scripts/run_batched.py --mode strings
    python scripts/run_batched.py --mode strings --seeds 16 --steps 800
    python scripts/run_batched.py --mode lqr --centers 105 190 245 --widths 30 35 35

All output files land in outputs/ by default.
"""
from __future__ import annotations
import sys
import argparse
import importlib
import time
from pathlib import Path

# Allow imports of codesign*, plots, etc. from the repo root regardless of
# where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from codesign_core import cfg

# Output directory sits at the repo root, next to demo.ipynb.
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


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
        sc = mod.scfg
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
    """Vmapped optimization over n_seeds independent initializations."""
    mod, loss_call, target_fn = load_mode(mode, lqr_centers, lqr_widths)

    freqs, target = target_fn()
    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
    params_batch = jax.vmap(mod.init_params_one)(keys)

    optimizer = optax.adam(lr)
    opt_state = jax.vmap(optimizer.init)(params_batch)

    @jax.jit
    def step(params, state):
        loss, grads = jax.vmap(
            lambda p: jax.value_and_grad(loss_call)(p, freqs, target)
        )(params)
        updates, new_state = jax.vmap(optimizer.update)(grads, state)
        new_params = jax.vmap(optax.apply_updates)(params, updates)
        return new_params, new_state, loss

    history = []
    t0 = time.time()
    for i in range(num_steps):
        params_batch, opt_state, losses = step(params_batch, opt_state)
        history.append(losses)
        if verbose and (i % 100 == 0 or i == num_steps - 1):
            print(f"  step {i:4d}  best_loss={float(losses.min()):.4f}  "
                  f"mean_loss={float(losses.mean()):.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    return params_batch, jnp.stack(history), freqs, target


def main():
    parser = argparse.ArgumentParser(description="Multi-start co-design runner.")
    parser.add_argument("--mode", choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--centers", type=float, nargs=3,
                        default=[105.0, 190.0, 245.0],
                        metavar=("C1", "C2", "C3"))
    parser.add_argument("--widths", type=float, nargs=3,
                        default=[30.0, 35.0, 35.0],
                        metavar=("W1", "W2", "W3"))
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Path for the output .npz file. Defaults to outputs/multistart_{mode}.npz",
    )
    args = parser.parse_args()

    outfile = Path(args.outfile) if args.outfile else OUTPUTS_DIR / f"multistart_{args.mode}.npz"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}  seeds: {args.seeds}  steps: {args.steps}")
    print(f"Output: {outfile}")

    params_batch, history, freqs, target = run_multistart(
        mode=args.mode,
        n_seeds=args.seeds,
        num_steps=args.steps,
        lr=args.lr,
        lqr_centers=tuple(args.centers),
        lqr_widths=tuple(args.widths),
    )

    np.savez(outfile,
             params=np.array(params_batch),
             history=np.array(history),
             freqs=np.array(freqs),
             target=np.array(target))
    print(f"Saved -> {outfile}")


if __name__ == "__main__":
    main()