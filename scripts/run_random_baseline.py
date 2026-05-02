"""run_random_baseline.py -- pure random search baseline, isolated from the
main optimization runs.

Evaluates the same loss function used by the optimizer at a large batch of
random parameter samples (no gradient information). Reports the best, mean,
median, and percentile losses across the batch. The best random sample is
saved along with its FRF so analysis.py can plot it next to the optimized
results.

This exists for two reasons:

  1. The course rubric calls out a random-guessing baseline as the canonical
     way to validate a new optimization formulation.

  2. Without a random baseline, we cannot tell whether the multi-start
     gradient descent is actually exploiting structure in the loss landscape
     or merely benefiting from the wide initialization noise. If random
     sampling at matched compute budget gets close to the optimizer, the
     optimization story is weak.

Two sampling distributions are supported:

  --distribution init   : sample exactly from init_params_one(). This shows
                          the loss at the same starting points the optimizer
                          sees, which is the lower bound the optimizer
                          MUST beat before declaring success.

  --distribution wide   : sample uniformly across a wider design space.
                          This is the bona-fide pure-random-search baseline.
                          Default.

Usage (from repo root):
    python scripts/run_random_baseline.py                                  # LQR, wide, 4096
    python scripts/run_random_baseline.py --mode strings
    python scripts/run_random_baseline.py --samples 16384 --distribution init
    python scripts/run_random_baseline.py --mode lqr --centers 300 500 600 --widths 30 30 30

Outputs (in outputs/, alongside multistart_*.npz):
    random_baseline_{mode}.npz   -- best_params, best_frf, all_losses, freqs, target
    random_baseline_{mode}.json  -- summary stats (best, mean, std, percentiles)

The script does not depend on demo.ipynb having run; it builds the target
spectrum from CLI arguments. To exactly match the demo's target, pass the
same --centers/--widths the notebook used.
"""
from __future__ import annotations
import sys
import json
import time
import argparse
import dataclasses
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax
import jax.numpy as jnp

import codesign_core as core
from codesign_core import cfg

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# LQR samplers
# ---------------------------------------------------------------------------
def sample_lqr_init(key):
    """Sample one (c, positions, log_q, log_r) from the same distribution as
    codesign.init_params_one. Use for the 'init' baseline."""
    import codesign  # local import so reconfigure() takes effect first
    return codesign.init_params_one(key)


def sample_lqr_wide(key):
    """Sample one (c, positions, log_q, log_r) from a wide design-space prior.

    * c ~ N(0, 1) over (M, M). The thickness sigmoid bounds c effectively to
      the active region without further clipping; std=1 covers the bulk of
      the reachable thickness fields.

    * positions ~ Uniform(0.1, 0.9) over (n_actuators, 2). Same box the
      optimizer's penalty enforces.

    * log_q, log_r ~ Uniform(-3, 3). Q and R from ~0.05 to ~20 covers the
      regime the optimizer typically converges into.
    """
    k_c, k_pos, k_q, k_r = jax.random.split(key, 4)
    c = jax.random.normal(k_c, (cfg.M, cfg.M))
    pos = jax.random.uniform(k_pos, (cfg.n_actuators, 2),
                             minval=0.1, maxval=0.9)
    log_q = jax.random.uniform(k_q, (), minval=-3.0, maxval=3.0)
    log_r = jax.random.uniform(k_r, (), minval=-3.0, maxval=3.0)
    return (c, pos, log_q, log_r)


# ---------------------------------------------------------------------------
# Strings samplers
# ---------------------------------------------------------------------------
def sample_strings_init(key, sc):
    """Sample one (c, log_tensions) from codesign_strings.init_params_one."""
    import codesign_strings
    return codesign_strings.init_params_one(key, sc=sc)


def sample_strings_wide(key, sc):
    """Sample one (c, log_tensions) from a wide design-space prior.

    * c ~ N(0, 1).

    * log_tensions ~ initial_log_tensions + N(0, 0.3). A std of 0.3 in log
      space corresponds to ~30% pitch deviation per string; the pitch_penalty
      will heavily weight off-target samples but the plate has freedom to
      explore. Going wider would saturate the loss with pitch_penalty alone
      and stop being informative about plate geometry.
    """
    import codesign_strings
    k_c, k_t = jax.random.split(key)
    c = jax.random.normal(k_c, (cfg.M, cfg.M))
    log_T0 = codesign_strings.initial_log_tensions(sc)
    log_T = log_T0 + 0.3 * jax.random.normal(k_t, log_T0.shape)
    return (c, log_T)


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
def build_lqr(distribution, centers_hz, widths_hz):
    """Return (sampler, loss_call, target_fn, frf_fn, freqs, target) for LQR mode."""
    import codesign

    sampler = sample_lqr_init if distribution == "init" else sample_lqr_wide

    freqs, target = codesign.target_spectrum_fixed(
        centers_hz=centers_hz, widths_hz=widths_hz)

    def loss_call(params):
        return codesign.loss_fn(params, target, freqs)

    def frf_call(params):
        c, positions, log_q, log_r = params
        omega, Phi = core.solve_modes(c)
        B = core.modal_values_at_points(Phi, positions)
        b_dist = core.modal_values_at_points(
            Phi, jnp.array([cfg.disturb_xy]))[:, 0]
        K_gain = codesign.modal_lqr_gains(
            omega, B, jnp.exp(log_q), jnp.exp(log_r))
        A_cl, B_d, C_out = core.closed_loop(omega, B, K_gain, b_dist)
        return core.frf_magnitude(A_cl, B_d, C_out, freqs)

    return sampler, loss_call, frf_call, freqs, target


def build_strings(distribution, pitches_hz, width_hz, n_harmonics):
    """Return (sampler, loss_call, frf_fn, freqs, target) for strings mode."""
    import codesign_strings
    sc = dataclasses.replace(
        codesign_strings.scfg,
        target_pitches_hz=tuple(pitches_hz),
        target_width_hz=width_hz,
        n_harmonics=n_harmonics,
    )

    if distribution == "init":
        def sampler(key):
            return sample_strings_init(key, sc)
    else:
        def sampler(key):
            return sample_strings_wide(key, sc)

    freqs = jnp.linspace(cfg.freq_lo, cfg.freq_hi, cfg.n_freqs)
    log_T_nominal = codesign_strings.initial_log_tensions(sc)
    target_nominal = codesign_strings.target_spectrum_from_strings(
        log_T_nominal, freqs, sc)

    def loss_call(params):
        return codesign_strings.loss_fn(params, freqs, sc=sc)

    def frf_call(params):
        return core.frf_passive(params[0], freqs)

    return sampler, loss_call, frf_call, freqs, target_nominal


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_random_baseline(mode, n_samples, distribution,
                        lqr_centers, lqr_widths,
                        str_pitches, str_width, str_harmonics,
                        seed=0, batch=32, verbose=True):
    """Evaluate `loss_call` at n_samples random samples, vmapped in chunks.

    Chunking keeps peak memory bounded when n_samples is very large.
    """
    if mode == "lqr":
        sampler, loss_call, frf_call, freqs, target = build_lqr(
            distribution, lqr_centers, lqr_widths)
    else:
        sampler, loss_call, frf_call, freqs, target = build_strings(
            distribution, str_pitches, str_width, str_harmonics)

    # vmapped sampling and loss evaluation
    sample_batch = jax.jit(jax.vmap(sampler))
    loss_batch = jax.jit(jax.vmap(loss_call))

    base_key = jax.random.PRNGKey(seed)
    all_losses = []
    best_loss_so_far = float("inf")
    best_params = None

    n_chunks = (n_samples + batch - 1) // batch
    t0 = time.time()
    for chunk_idx in range(n_chunks):
        chunk_size = min(batch, n_samples - chunk_idx * batch)
        chunk_key = jax.random.fold_in(base_key, chunk_idx)
        keys = jax.random.split(chunk_key, chunk_size)

        params_chunk = sample_batch(keys)
        losses_chunk = loss_batch(params_chunk)
        losses_np = np.asarray(losses_chunk)
        all_losses.append(losses_np)

        # Track the best we have seen across chunks
        local_min_idx = int(np.argmin(losses_np))
        local_min = float(losses_np[local_min_idx])
        if local_min < best_loss_so_far:
            best_loss_so_far = local_min
            # Extract the single sample at local_min_idx from the pytree
            best_params = jax.tree_util.tree_map(
                lambda x: np.asarray(x[local_min_idx]), params_chunk)

        if verbose and (chunk_idx % max(1, n_chunks // 16) == 0
                         or chunk_idx == n_chunks - 1):
            done = (chunk_idx + 1) * batch
            done = min(done, n_samples)
            print(f"  chunk {chunk_idx + 1:4d}/{n_chunks}  "
                  f"({done:6d}/{n_samples} samples)   "
                  f"best_so_far={best_loss_so_far:.4e}   "
                  f"elapsed={time.time() - t0:.1f}s")

    losses_all = np.concatenate(all_losses)

    # Compute the FRF of the best random sample once at the end
    best_params_jnp = jax.tree_util.tree_map(jnp.asarray, best_params)
    best_frf = np.asarray(frf_call(best_params_jnp))

    return {
        "losses": losses_all,
        "best_loss": float(losses_all.min()),
        "best_params": best_params,
        "best_frf": best_frf,
        "freqs": np.asarray(freqs),
        "target": np.asarray(target),
        "elapsed_s": time.time() - t0,
    }


def summarize(losses):
    """Return dict of percentiles and moments suitable for json.dump."""
    return {
        "n_samples": int(losses.size),
        "best": float(np.min(losses)),
        "mean": float(np.mean(losses)),
        "median": float(np.median(losses)),
        "std": float(np.std(losses)),
        "p10": float(np.percentile(losses, 10)),
        "p90": float(np.percentile(losses, 90)),
        "worst": float(np.max(losses)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Random-search baseline. Forward-only, no gradients. "
                    "All defaults mirror demo.ipynb so results are directly "
                    "comparable to the optimized runs."
    )
    parser.add_argument("--mode", choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--samples", type=int, default=4096,
                        help="Total random samples to evaluate (default 4096).")
    parser.add_argument("--distribution", choices=["init", "wide"], default="wide",
                        help="'init' = sample from init_params_one() like the "
                             "optimizer's starting points; 'wide' = sample "
                             "uniformly over a broader design space (default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32,
                        help="vmap batch size. Each sample requires an "
                             "eigendecomposition of an Nx*Ny by Nx*Ny matrix, "
                             "so memory scales as batch * (Nx*Ny)**2. At the "
                             "default 50x40 grid, batch=32 needs ~1 GB; "
                             "batch=64 needs ~2 GB. Lower this if you hit OOM.")

    # LQR target (mirror notebook defaults exactly)
    parser.add_argument("--centers", type=float, nargs=3,
                        default=[280.0, 470.0, 530.0])
    parser.add_argument("--widths", type=float, nargs=3,
                        default=[40.0, 20.0, 20.0])

    # Strings target (mirror notebook defaults exactly)
    parser.add_argument("--pitches", type=float, nargs="+",
                        default=[196.0, 293.7, 440.0, 659.3])
    parser.add_argument("--string-width", type=float, default=40.0)
    parser.add_argument("--harmonics", type=int, default=6)

    parser.add_argument("--outfile-npz", type=str, default=None)
    parser.add_argument("--outfile-json", type=str, default=None)

    args = parser.parse_args()

    npz_path = (Path(args.outfile_npz) if args.outfile_npz
                else OUTPUTS_DIR / f"random_baseline_{args.mode}.npz")
    json_path = (Path(args.outfile_json) if args.outfile_json
                 else OUTPUTS_DIR / f"random_baseline_{args.mode}.json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Random baseline")
    print("=" * 60)
    print(f"Mode         : {args.mode}")
    print(f"Distribution : {args.distribution}")
    print(f"Samples      : {args.samples}   batch={args.batch}   seed={args.seed}")
    if args.mode == "lqr":
        print(f"Centers      : {args.centers} Hz")
        print(f"Widths       : {args.widths} Hz")
    else:
        print(f"Pitches      : {args.pitches} Hz")
        print(f"Width        : {args.string_width} Hz   harmonics={args.harmonics}")
    print(f"Material     : {cfg.material}   Grid: {cfg.Nx}x{cfg.Ny}   "
          f"M={cfg.M}   n_actuators={cfg.n_actuators}")
    print(f"Output (npz) : {npz_path}")
    print(f"Output (json): {json_path}")
    print()

    result = run_random_baseline(
        mode=args.mode,
        n_samples=args.samples,
        distribution=args.distribution,
        lqr_centers=tuple(args.centers),
        lqr_widths=tuple(args.widths),
        str_pitches=tuple(args.pitches),
        str_width=args.string_width,
        str_harmonics=args.harmonics,
        seed=args.seed,
        batch=args.batch,
    )

    # Write npz: arrays only
    save_dict = {
        "losses": result["losses"],
        "best_frf": result["best_frf"],
        "freqs": result["freqs"],
        "target": result["target"],
    }
    if args.mode == "lqr":
        c, pos, log_q, log_r = result["best_params"]
        save_dict.update({
            "best_c": c, "best_positions": pos,
            "best_log_q": log_q, "best_log_r": log_r,
        })
    else:
        c, log_T = result["best_params"]
        save_dict.update({"best_c": c, "best_log_tensions": log_T})

    np.savez(npz_path, **save_dict)
    print(f"Saved arrays -> {npz_path}")

    # Write json: scalar summary
    summary = summarize(result["losses"])
    summary["mode"] = args.mode
    summary["distribution"] = args.distribution
    summary["seed"] = args.seed
    summary["elapsed_s"] = result["elapsed_s"]
    if args.mode == "lqr":
        summary["target_centers_hz"] = list(args.centers)
        summary["target_widths_hz"] = list(args.widths)
    else:
        summary["target_pitches_hz"] = list(args.pitches)
        summary["target_width_hz"] = args.string_width
        summary["n_harmonics"] = args.harmonics

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {json_path}")

    print()
    print("Summary:")
    print(f"  best   = {summary['best']:.4e}")
    print(f"  p10    = {summary['p10']:.4e}")
    print(f"  median = {summary['median']:.4e}")
    print(f"  mean   = {summary['mean']:.4e}  (std={summary['std']:.3e})")
    print(f"  p90    = {summary['p90']:.4e}")
    print(f"  worst  = {summary['worst']:.4e}")
    print(f"  elapsed: {result['elapsed_s']:.1f}s")
    print()
    print("Compare against the optimizer's best loss in outputs/results.json.")


if __name__ == "__main__":
    main()