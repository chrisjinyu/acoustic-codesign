"""run_ablations.py -- co-design validation ablations for the LQR variant.

Runs two ablations that isolate specific contributions of the co-design
formulation, and writes their results for analysis.py to plot:

  1. Fixed-QR : optimize (c, positions) with Q=R=1 (log_q=log_r=0).
                Isolates the value of tuning LQR weights as design variables.
  2. Sequential : Stage 1 = passive geometry optimization (reused from
                demo.ipynb's passive baseline);
                Stage 2 = freeze that geometry, optimize (positions, Q, R).
                This is the classical "plant first, controller second"
                pipeline that co-design claims to beat.

The joint (all four variables at once) result is produced by demo.ipynb and is
read back here for the summary, not re-run.

Prerequisites (all produced by demo.ipynb):
    outputs/results.json        -- config, seed, target spectrum, joint loss
    outputs/best_params.npz     -- must contain 'c_passive_lqr' and related arrays

Outputs:
    outputs/ablations_lqr.npz   -- arrays (FRFs, histories, geometries)
    outputs/ablations_lqr.json  -- scalar metrics (losses, deltas)

Usage (from repo root):
    python scripts/run_ablations.py
    python scripts/run_ablations.py --steps 400 --seed 18848
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import jax
import jax.numpy as jnp

import codesign_core as core
import codesign

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _frf_with_gains(c, positions, log_q, log_r, freqs):
    """Closed-loop FRF magnitude given (c, positions, log_q, log_r)."""
    omega, Phi = core.solve_modes(c)
    B = core.modal_values_at_points(Phi, jnp.asarray(positions))
    b_dist = core.modal_values_at_points(
        Phi, jnp.array([core.cfg.disturb_xy]))[:, 0]
    K_gain = codesign.modal_lqr_gains(
        omega, B, jnp.exp(log_q), jnp.exp(log_r))
    A_cl, B_d, C_out = core.closed_loop(omega, B, K_gain, b_dist)
    return np.asarray(core.frf_magnitude(A_cl, B_d, C_out, jnp.asarray(freqs)))


def _apply_config_from_results(res: dict):
    """Re-apply the material + grid config recorded by demo.ipynb.

    demo.ipynb calls core.reconfigure() once at the top of its run. We mirror
    that here so the loss landscape and eigenproblem resolution match exactly.
    """
    c = res["config"]
    # scale_kwargs were applied in the notebook too; the key dimensions are
    # Nx, Ny, M, n_modes. n_actuators is a top-level field in results['config']
    # only if it was stored there; otherwise the current cfg default is used.
    kwargs = {
        "material": c["material"],
        "Lx": c["Lx"], "Ly": c["Ly"],
        "Nx": c["Nx"], "Ny": c["Ny"],
        "M": c["M"], "n_modes": c["n_modes"],
    }
    # Only override n_actuators if we have it; otherwise infer from
    # best_params.npz positions shape.
    return core.reconfigure(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run Fixed-QR and Sequential LQR ablations.")
    parser.add_argument("--results", type=str,
                        default=str(OUTPUTS_DIR / "results.json"))
    parser.add_argument("--npz", type=str,
                        default=str(OUTPUTS_DIR / "best_params.npz"))
    parser.add_argument("--seed", type=int, default=18848,
                        help="Seed for ablation runs (match demo.ipynb SEED).")
    parser.add_argument("--steps", type=int, default=600,
                        help="Optimization steps (match demo.ipynb CODESIGN_STEPS).")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--log-q-fixed", type=float, default=0.0,
                        help="Fixed log_q for the Fixed-QR ablation (default 0).")
    parser.add_argument("--log-r-fixed", type=float, default=0.0,
                        help="Fixed log_r for the Fixed-QR ablation (default 0).")
    args = parser.parse_args()

    results_path = Path(args.results)
    npz_path = Path(args.npz)

    if not results_path.exists() or not npz_path.exists():
        print(f"ERROR: expected {results_path} and {npz_path} to exist.")
        print("Run demo.ipynb end-to-end first.")
        sys.exit(1)

    with open(results_path) as f:
        res = json.load(f)

    demo = np.load(npz_path, allow_pickle=True)

    if res.get("mode") not in ("lqr", "both"):
        print(f"ERROR: demo.ipynb was run with mode={res.get('mode')!r}.")
        print("Re-run with MODE='lqr' or MODE='both' so LQR outputs exist.")
        sys.exit(1)

    if "c_passive_lqr" not in demo.files:
        print("ERROR: 'c_passive_lqr' not found in best_params.npz.")
        print("You must add the line")
        print("    'c_passive_lqr': np.asarray(c_passive_lqr),")
        print("to the save_args dict in demo.ipynb's save cell, then re-run")
        print("that cell. See scripts/run_ablations.py docstring for the one-")
        print("line patch.")
        sys.exit(1)

    # --- Apply matching config so physics lines up with the joint run ------
    cfg = _apply_config_from_results(res)

    # --- Rebuild the LQR target spectrum exactly as demo.ipynb had it ------
    centers = tuple(res["config"]["lqr_target_centers"])
    widths = tuple(res["config"]["lqr_target_widths"])
    freqs = jnp.asarray(demo["freqs_lqr"])
    target = jnp.asarray(demo["target_lqr"])
    freqs_np = np.asarray(freqs)

    # --- Pull previously computed quantities we need -----------------------
    c_passive = jnp.asarray(demo["c_passive_lqr"])
    c_joint = jnp.asarray(demo["c_lqr"])
    pos_joint = jnp.asarray(demo["positions_lqr"])
    H_passive = np.asarray(demo["H_passive_lqr"])
    H_joint = np.asarray(demo["H_lqr"])
    best_loss_passive = float(res["passive_lqr_best_loss"])
    best_loss_joint = float(res["lqr_best_loss"])
    log_q_joint = float(res["log_q_final"])
    log_r_joint = float(res["log_r_final"])

    print("=" * 60)
    print("LQR co-design ablations")
    print("=" * 60)
    print(f"Target centers (Hz) : {centers}")
    print(f"Target widths  (Hz) : {widths}")
    print(f"Material            : {cfg.material}  grid {cfg.Nx}x{cfg.Ny}  M={cfg.M}")
    print(f"Seed                : {args.seed}")
    print(f"Steps               : {args.steps}")
    print()
    print(f"Passive (reference) : loss = {best_loss_passive:.4e}")
    print(f"Joint   (reference) : loss = {best_loss_joint:.4e}")
    print()

    # -------------------------------------------------------------------
    # Ablation 1: Fixed Q/R  (optimize c + positions, Q=R=1 constants)
    # -------------------------------------------------------------------
    print("-" * 60)
    print(f"Ablation 1: Fixed Q=R=1 "
          f"(log_q={args.log_q_fixed}, log_r={args.log_r_fixed})")
    print("-" * 60)
    c_fqr, pos_fqr, loss_fqr, hist_fqr = codesign.run_fixed_qr(
        target, freqs,
        num_steps=args.steps, seed=args.seed, lr=args.lr,
        log_q_const=args.log_q_fixed, log_r_const=args.log_r_fixed,
    )
    H_fqr = _frf_with_gains(c_fqr, pos_fqr,
                            args.log_q_fixed, args.log_r_fixed, freqs)
    print(f"  Fixed-QR best loss : {loss_fqr:.4e}")

    # -------------------------------------------------------------------
    # Ablation 2: Sequential  (passive c -> optimize positions + Q + R)
    # -------------------------------------------------------------------
    print()
    print("-" * 60)
    print("Ablation 2: Sequential co-design (passive c then controller)")
    print("-" * 60)
    pos_seq, log_q_seq, log_r_seq, loss_seq, hist_seq = \
        codesign.run_sequential_stage2(
            c_passive, target, freqs,
            num_steps=args.steps, seed=args.seed, lr=args.lr,
        )
    H_seq = _frf_with_gains(c_passive, pos_seq, log_q_seq, log_r_seq, freqs)
    print(f"  Stage-1 (reused passive) : {best_loss_passive:.4e}")
    print(f"  Stage-2 best loss        : {loss_seq:.4e}")
    print(f"  Tuned gains              : log_q={float(log_q_seq):.3f}  "
          f"log_r={float(log_r_seq):.3f}  "
          f"(Q={float(jnp.exp(log_q_seq)):.3f}, R={float(jnp.exp(log_r_seq)):.3f})")

    # -------------------------------------------------------------------
    # Ladder summary
    # -------------------------------------------------------------------
    def _pct(a, b):
        return 100.0 * (a - b) / a if a != 0 else 0.0

    pct_fqr_vs_pas = _pct(best_loss_passive, loss_fqr)
    pct_seq_vs_pas = _pct(best_loss_passive, loss_seq)
    pct_jnt_vs_pas = _pct(best_loss_passive, best_loss_joint)
    pct_jnt_vs_seq = _pct(loss_seq, best_loss_joint)
    pct_jnt_vs_fqr = _pct(loss_fqr, best_loss_joint)

    print()
    print("=" * 60)
    print("Co-design ladder  (lower loss = better)")
    print("=" * 60)
    print(f"  Passive only        : {best_loss_passive:.4e}   (reference)")
    print(f"  Fixed Q/R co-design : {loss_fqr:.4e}   "
          f"({pct_fqr_vs_pas:+.1f}% vs passive)")
    print(f"  Sequential          : {loss_seq:.4e}   "
          f"({pct_seq_vs_pas:+.1f}% vs passive)")
    print(f"  Joint               : {best_loss_joint:.4e}   "
          f"({pct_jnt_vs_pas:+.1f}% vs passive)")
    print()
    print(f"  Value of tuning Q/R   : joint vs fixed-QR = {pct_jnt_vs_fqr:+.1f}%")
    print(f"  Value of joint opt    : joint vs sequential = {pct_jnt_vs_seq:+.1f}%")

    # -------------------------------------------------------------------
    # Write outputs
    # -------------------------------------------------------------------
    arr_out = OUTPUTS_DIR / "ablations_lqr.npz"
    np.savez(
        arr_out,
        # shared
        freqs=freqs_np,
        target=np.asarray(target),
        # passive (reference, re-stored for single-file analysis convenience)
        c_passive=np.asarray(c_passive),
        H_passive=H_passive,
        # fixed-QR ablation
        c_fixedqr=np.asarray(c_fqr),
        positions_fixedqr=np.asarray(pos_fqr),
        H_fixedqr=H_fqr,
        history_fixedqr=np.asarray(hist_fqr),
        log_q_fixed=np.asarray(args.log_q_fixed),
        log_r_fixed=np.asarray(args.log_r_fixed),
        # sequential ablation
        c_sequential=np.asarray(c_passive),   # same as passive (Stage 1)
        positions_sequential=np.asarray(pos_seq),
        log_q_sequential=np.asarray(log_q_seq),
        log_r_sequential=np.asarray(log_r_seq),
        H_sequential=H_seq,
        history_sequential_stage2=np.asarray(hist_seq),
        # joint (reference, for single-file analysis)
        c_joint=np.asarray(c_joint),
        positions_joint=np.asarray(pos_joint),
        H_joint=H_joint,
        log_q_joint=np.asarray(log_q_joint),
        log_r_joint=np.asarray(log_r_joint),
    )
    print(f"\nSaved arrays  -> {arr_out}")

    json_out = OUTPUTS_DIR / "ablations_lqr.json"
    with open(json_out, "w") as f:
        json.dump({
            "seed": args.seed,
            "steps": args.steps,
            "lr": args.lr,
            "target_centers_hz": list(centers),
            "target_widths_hz": list(widths),
            "passive_loss": best_loss_passive,
            "fixedqr_loss": float(loss_fqr),
            "fixedqr_log_q": float(args.log_q_fixed),
            "fixedqr_log_r": float(args.log_r_fixed),
            "sequential_loss": float(loss_seq),
            "sequential_log_q": float(log_q_seq),
            "sequential_log_r": float(log_r_seq),
            "joint_loss": best_loss_joint,
            "joint_log_q": log_q_joint,
            "joint_log_r": log_r_joint,
            "pct_fixedqr_vs_passive": float(pct_fqr_vs_pas),
            "pct_sequential_vs_passive": float(pct_seq_vs_pas),
            "pct_joint_vs_passive": float(pct_jnt_vs_pas),
            "pct_joint_vs_sequential": float(pct_jnt_vs_seq),
            "pct_joint_vs_fixedqr": float(pct_jnt_vs_fqr),
        }, f, indent=2)
    print(f"Saved scalars -> {json_out}")


if __name__ == "__main__":
    main()
