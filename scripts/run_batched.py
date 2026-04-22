"""Batched multi-start co-design, mode-aware.

Runs N parallel optimization instances from different random initializations
using jax.vmap. Supports both variants:

  --mode lqr      (default): plate + actuator placement + LQR gains
  --mode strings            : plate + string tensions

All defaults exactly mirror the demo.ipynb configuration cell so that
running this script without arguments produces multi-start results that
are directly comparable to the notebook's single-seed run.

Usage (from repo root):
    python scripts/run_batched.py                        # LQR, notebook defaults
    python scripts/run_batched.py --mode strings         # strings, notebook defaults
    python scripts/run_batched.py --seeds 16 --steps 800
    python scripts/run_batched.py --mode lqr --centers 300 500 600 --widths 30 30 30

All output files land in outputs/ by default.
"""
from __future__ import annotations
import sys
import argparse
import dataclasses
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

import codesign_core as core
from codesign_core import cfg

# Output directory sits at the repo root, next to demo.ipynb.
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_mode(mode: str,
              lqr_centers: tuple,
              lqr_widths: tuple,
              str_pitches: tuple,
              str_width: float,
              str_harmonics: int):
    """Return (module, loss_call, target_fn) for the requested variant.

    All string and LQR configuration is passed explicitly so that every
    parameter matches whatever was set in main() rather than relying on
    module-level defaults.
    """
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

        # Override the module-level scfg with the notebook-matching parameters
        # rather than using the dataclass defaults directly.
        sc = dataclasses.replace(
            mod.scfg,
            target_pitches_hz=str_pitches,
            target_width_hz=str_width,
            n_harmonics=str_harmonics,
        )

        def loss_call(params, freqs, target):
            return mod.loss_fn(params, freqs, sc=sc)

        def target_fn():
            freqs_ = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)
            log_T0 = mod.initial_log_tensions(sc)
            target_ = mod.target_spectrum_from_strings(log_T0, freqs_, sc)
            return freqs_, target_

        import functools
        mod.init_params_one = functools.partial(mod.init_params_one, sc=sc)
        return mod, loss_call, target_fn

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'lqr' or 'strings'.")


def run_multistart(mode: str,
                   n_seeds: int,
                   num_steps: int,
                   lr: float,
                   lqr_centers: tuple,
                   lqr_widths: tuple,
                   str_pitches: tuple,
                   str_width: float,
                   str_harmonics: int,
                   verbose: bool = True):
    """Vmapped optimization over n_seeds independent initializations."""
    mod, loss_call, target_fn = load_mode(
        mode, lqr_centers, lqr_widths,
        str_pitches, str_width, str_harmonics,
    )

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
    parser = argparse.ArgumentParser(
        description="Multi-start co-design runner. "
                    "All defaults mirror the demo.ipynb configuration exactly."
    )

    # --- Mode and optimizer ---
    parser.add_argument("--mode", choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-3)

    # --- LQR target spectrum (notebook: LQR_TARGET_CENTERS / LQR_TARGET_WIDTHS) ---
    parser.add_argument("--centers", type=float, nargs=3,
                        default=[280.0, 470.0, 530.0],
                        metavar=("C1", "C2", "C3"),
                        help="LQR target Gaussian centers in Hz (default: 280 470 530)")
    parser.add_argument("--widths", type=float, nargs=3,
                        default=[40.0, 20.0, 20.0],
                        metavar=("W1", "W2", "W3"),
                        help="LQR target Gaussian widths in Hz (default: 40 40 40)")

    # --- String target configuration (notebook: STRING_TARGET_PITCHES etc.) ---
    parser.add_argument("--pitches", type=float, nargs="+",
                        default=[196.0, 293.7, 440.0, 659.3],
                        metavar="HZ",
                        help="String target fundamental frequencies in Hz "
                             "(default: 196.0 293.7 440.0 659.3  -- violin G3 D4 A4 E5)")
    parser.add_argument("--string-width", type=float, default=40.0,
                        help="Gaussian width per string peak in Hz (default: 40.0)")
    parser.add_argument("--harmonics", type=int, default=6,
                        help="Number of harmonics per string (default: 6)")

    # --- Output ---
    parser.add_argument("--outfile", type=str, default=None,
                        help="Path for the output .npz file. "
                             "Defaults to outputs/multistart_{mode}.npz")

    args = parser.parse_args()

    outfile = (Path(args.outfile) if args.outfile
               else OUTPUTS_DIR / f"multistart_{args.mode}.npz")
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Print the full configuration so it is auditable in logs.
    print(f"Mode     : {args.mode}")
    print(f"Seeds    : {args.seeds}   Steps: {args.steps}   LR: {args.lr}")
    if args.mode == "lqr":
        print(f"Centers  : {args.centers} Hz")
        print(f"Widths   : {args.widths} Hz")
    else:
        print(f"Pitches  : {args.pitches} Hz")
        print(f"Str width: {args.string_width} Hz   Harmonics: {args.harmonics}")
    print(f"Material : {cfg.material}   Grid: {cfg.Nx}x{cfg.Ny}   "
          f"M={cfg.M}   n_actuators={cfg.n_actuators}")
    print(f"Output   : {outfile}")
    print()

    params_batch, history, freqs, target = run_multistart(
        mode=args.mode,
        n_seeds=args.seeds,
        num_steps=args.steps,
        lr=args.lr,
        lqr_centers=tuple(args.centers),
        lqr_widths=tuple(args.widths),
        str_pitches=tuple(args.pitches),
        str_width=args.string_width,
        str_harmonics=args.harmonics,
    )

    # Save each component under its own key so numpy does not try to
    # coerce the inhomogeneous pytree into a single array.
    if args.mode == "lqr":
        c_b, pos_b, logq_b, logr_b = params_batch
        np.savez(outfile,
                 c=np.array(c_b),
                 positions=np.array(pos_b),
                 log_q=np.array(logq_b),
                 log_r=np.array(logr_b),
                 history=np.array(history),
                 freqs=np.array(freqs),
                 target=np.array(target))
    else:
        c_b, logt_b = params_batch
        np.savez(outfile,
                 c=np.array(c_b),
                 log_tensions=np.array(logt_b),
                 history=np.array(history),
                 freqs=np.array(freqs),
                 target=np.array(target))

    print(f"Saved -> {outfile}")


if __name__ == "__main__":
    main()