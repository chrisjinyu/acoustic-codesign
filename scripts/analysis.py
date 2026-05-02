"""
analysis.py  --  regenerate all report figures from saved outputs.

Run after demo.ipynb and run_batched.py have both completed:

    python scripts/analysis.py

And, if the ablation and random baseline scripts have been run:

    python scripts/run_ablations.py
    python scripts/run_random_baseline.py --mode lqr
    python scripts/run_random_baseline.py --mode strings
    python scripts/analysis.py

Reads (from outputs/):
    results.json                  -- scalar results and metadata from demo.ipynb
    best_params.npz               -- FRF arrays and histories from demo.ipynb
                                     (must contain c_passive_lqr; see patch note
                                      in scripts/run_ablations.py)
    multistart_lqr.npz            -- multi-start trajectories from run_batched.py --mode lqr
    multistart_strings.npz        -- multi-start trajectories from run_batched.py --mode strings
    ablations_lqr.npz             -- ablation arrays from run_ablations.py (optional)
    ablations_lqr.json            -- ablation scalars from run_ablations.py (optional)
    random_baseline_lqr.json      -- random search summary from run_random_baseline.py (optional)
    random_baseline_strings.json  -- random search summary from run_random_baseline.py (optional)

Writes (to outputs/):
    fig_frf_lqr.png              -- LQR FRF comparison (passive vs co-designed)
    fig_frf_strings.png          -- strings FRF comparison (passive vs co-designed)
    fig_multistart_lqr.png       -- multi-start variance, LQR
    fig_multistart_strings.png   -- multi-start variance, strings
    fig_summary_table.png        -- one-page results summary
    fig_geometry_comparison.png  -- passive-optimal vs joint thickness fields
    fig_ablation_ladder.png      -- Random | Passive | Fixed-QR | Sequential | Joint
    fig_ablation_frf.png         -- FRF overlay for the four designs
"""
from __future__ import annotations
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load demo outputs ─────────────────────────────────────────────────────────
with open(OUTPUTS_DIR / "results.json") as f:
    res = json.load(f)

demo = np.load(OUTPUTS_DIR / "best_params.npz", allow_pickle=True)
MODE = res["mode"]
cfg  = res["config"]

print(f"Mode     : {MODE}")
print(f"Material : {cfg['material']}  grid {cfg['Nx']}x{cfg['Ny']}  M={cfg['M']}")
print()

# Strings improvement computed lazily the first time it is needed
imp_str = None


# ── Helper: normalise an FRF array to max=1 ───────────────────────────────────
def norm(H):
    return np.asarray(H) / (np.asarray(H).max() + 1e-12)


# ── Load optional outputs (all guarded so the script runs with partial data) ──
abl_npz_path  = OUTPUTS_DIR / "ablations_lqr.npz"
abl_json_path = OUTPUTS_DIR / "ablations_lqr.json"
rand_lqr_path = OUTPUTS_DIR / "random_baseline_lqr.json"
rand_str_path = OUTPUTS_DIR / "random_baseline_strings.json"

abl     = np.load(abl_npz_path,  allow_pickle=True) if abl_npz_path.exists()  else None
abl_res = json.load(open(abl_json_path))             if abl_json_path.exists() else None
rand_lqr = json.load(open(rand_lqr_path))            if rand_lqr_path.exists() else None
rand_str = json.load(open(rand_str_path))             if rand_str_path.exists() else None

if rand_lqr:
    print(f"Random baseline (LQR)     : best={rand_lqr['best']:.4e}  "
          f"N={rand_lqr['n_samples']}  dist={rand_lqr['distribution']}")
if rand_str:
    print(f"Random baseline (strings) : best={rand_str['best']:.4e}  "
          f"N={rand_str['n_samples']}  dist={rand_str['distribution']}")
if rand_lqr or rand_str:
    print()


# =============================================================================
# Figure 1: LQR FRF comparison
# =============================================================================
if MODE in ("lqr", "both") and "H_lqr" in demo:
    freqs_lqr  = demo["freqs_lqr"]
    target_lqr = demo["target_lqr"]
    H_pass_lqr = demo["H_passive_lqr"]
    H_lqr      = demo["H_lqr"]

    passive_loss = res["passive_lqr_best_loss"]
    lqr_loss     = res["lqr_best_loss"]
    imp_lqr      = res["lqr_improvement_pct"]

    fig, ax = plt.subplots(figsize=(9, 4))
    T = norm(target_lqr)
    ax.fill_between(freqs_lqr, 0, T, color="gray", alpha=0.25, label="target")
    ax.plot(freqs_lqr, T, "k--", lw=1.2, alpha=0.6)
    ax.plot(freqs_lqr, norm(H_pass_lqr), lw=1.8,
            label=f"passive only  (loss={passive_loss:.3e})")
    ax.plot(freqs_lqr, norm(H_lqr), lw=1.8,
            label=f"LQR co-designed  (loss={lqr_loss:.3e})")

    # Overlay the best random sample FRF if present
    rand_lqr_npz_path = OUTPUTS_DIR / "random_baseline_lqr.npz"
    if rand_lqr_npz_path.exists():
        rnpz = np.load(rand_lqr_npz_path, allow_pickle=True)
        if "best_frf" in rnpz.files:
            H_rand = np.asarray(rnpz["best_frf"])
            ax.plot(freqs_lqr, norm(H_rand), lw=1.4, ls=":", color="#888888",
                    label=f"random best  (loss={rand_lqr['best']:.3e})")

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H(jw)|  normalized")
    ax.set_title(f"LQR co-design: {imp_lqr:.1f}% improvement over passive")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_frf_lqr.png", dpi=160)
    print("Saved fig_frf_lqr.png")
    plt.show()


# =============================================================================
# Figure 2: Strings FRF comparison
# =============================================================================
if MODE in ("strings", "both") and "H_str" in demo:
    freqs_str  = demo["freqs_str"]
    target_str = demo["target_str_nominal"]
    H_pass_str = demo["H_passive_str"]
    H_str      = demo["H_str"]

    passive_loss = res["passive_str_best_loss"]
    str_loss     = res["strings_best_loss"]
    imp_str      = 100 * (passive_loss - str_loss) / passive_loss

    fig, ax = plt.subplots(figsize=(9, 4))
    T = norm(target_str)
    ax.fill_between(freqs_str, 0, T, color="gray", alpha=0.25, label="target")
    ax.plot(freqs_str, T, "k--", lw=1.2, alpha=0.6)
    ax.plot(freqs_str, norm(H_pass_str), lw=1.8,
            label=f"passive only  (loss={passive_loss:.3e})")
    ax.plot(freqs_str, norm(H_str), lw=1.8,
            label=f"strings co-designed  (loss={str_loss:.3e})")

    # Overlay best random sample FRF if present
    rand_str_npz_path = OUTPUTS_DIR / "random_baseline_strings.npz"
    if rand_str_npz_path.exists():
        rnpz = np.load(rand_str_npz_path, allow_pickle=True)
        if "best_frf" in rnpz.files:
            H_rand_s = np.asarray(rnpz["best_frf"])
            ax.plot(freqs_str, norm(H_rand_s), lw=1.4, ls=":", color="#888888",
                    label=f"random best  (loss={rand_str['best']:.3e})")

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H(jw)|  normalized")
    ax.set_title(f"Strings co-design: {imp_str:.1f}% improvement over passive")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_frf_strings.png", dpi=160)
    print("Saved fig_frf_strings.png")
    plt.show()


# =============================================================================
# Figure 3: Multi-start variance -- LQR
# =============================================================================
lqr_batch_path = OUTPUTS_DIR / "multistart_lqr.npz"
if lqr_batch_path.exists():
    b = np.load(lqr_batch_path, allow_pickle=True)
    history_all = b["history"]          # (n_steps, n_seeds)
    best_losses = history_all[-1]       # final-step loss per seed
    n_seeds = history_all.shape[1]

    steps = np.arange(history_all.shape[0])
    mean  = history_all.mean(axis=1)
    std   = history_all.std(axis=1)
    best  = history_all.min(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                    label="mean +/- 1 std")
    ax.semilogy(steps, mean, lw=2, label="mean across seeds")
    ax.semilogy(steps, best, lw=2, ls="--", label="best-step trajectory")

    # Draw a horizontal reference line at the random baseline best loss
    if rand_lqr is not None:
        ax.axhline(rand_lqr["best"], color="#888888", lw=1.2, ls=":",
                   label=f"random best (N={rand_lqr['n_samples']})")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"LQR multi-start ({n_seeds} seeds)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_multistart_lqr.png", dpi=160)
    print("Saved fig_multistart_lqr.png")
    print(f"  best-loss across seeds: "
          f"min={best_losses.min():.3e}  "
          f"mean={best_losses.mean():.3e}  "
          f"std={best_losses.std():.3e}")
    plt.show()
else:
    print(f"Skipping LQR multi-start figure ({lqr_batch_path} not found)")


# =============================================================================
# Figure 4: Multi-start variance -- strings
# =============================================================================
str_batch_path = OUTPUTS_DIR / "multistart_strings.npz"
if str_batch_path.exists():
    b = np.load(str_batch_path, allow_pickle=True)
    history_all = b["history"]
    best_losses = history_all[-1]
    n_seeds = history_all.shape[1]

    steps = np.arange(history_all.shape[0])
    mean  = history_all.mean(axis=1)
    std   = history_all.std(axis=1)
    best  = history_all.min(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                    label="mean +/- 1 std")
    ax.semilogy(steps, mean, lw=2, label="mean across seeds")
    ax.semilogy(steps, best, lw=2, ls="--", label="best-step trajectory")

    if rand_str is not None:
        ax.axhline(rand_str["best"], color="#888888", lw=1.2, ls=":",
                   label=f"random best (N={rand_str['n_samples']})")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Strings multi-start ({n_seeds} seeds)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_multistart_strings.png", dpi=160)
    print("Saved fig_multistart_strings.png")
    print(f"  best-loss across seeds: "
          f"min={best_losses.min():.3e}  "
          f"mean={best_losses.mean():.3e}  "
          f"std={best_losses.std():.3e}")
    plt.show()
else:
    print(f"Skipping strings multi-start figure ({str_batch_path} not found)")


# =============================================================================
# Figure 5: Summary table (one page, report-ready)
# =============================================================================
if imp_str is None and MODE in ("strings", "both"):
    imp_str = (100 * (res["passive_str_best_loss"] - res["strings_best_loss"])
               / res["passive_str_best_loss"])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.axis("off")

rows = [["Metric", "LQR co-design", "Strings co-design"]]

if MODE in ("lqr", "both"):
    lqr_modes = [f"{f:.0f}" for f in res["omega_hz_lqr"][:6]]
    rows.append(["Passive best loss",
                 f"{res['passive_lqr_best_loss']:.4e}", ""])
    rows.append(["Co-design best loss",
                 f"{res['lqr_best_loss']:.4e}",
                 f"{res['strings_best_loss']:.4e}" if MODE == "both" else ""])
    rows.append(["Improvement vs passive",
                 f"{res['lqr_improvement_pct']:.1f}%",
                 f"{imp_str:.1f}%" if (MODE == "both" and imp_str is not None) else ""])
    rows.append(["First 6 modes (Hz)", ", ".join(lqr_modes), ""])

if MODE in ("strings", "both"):
    str_modes = [f"{f:.0f}" for f in res["omega_hz_strings"][:6]]
    if MODE == "both":
        rows[2][2] = f"{res['strings_best_loss']:.4e}"
        rows[3][2] = (f"{imp_str:.1f}%" if imp_str is not None else "")
        rows[4][2] = ", ".join(str_modes)
        rows.append(["", "", ""])
        rows.append(["String tuning", "target Hz -> actual Hz", "detune"])
        for i, (ft, fa) in enumerate(zip(
                res["target_pitches_hz"], res["actual_pitches_hz"])):
            rows.append([f"  String {i+1}",
                         f"{ft:.1f} -> {fa:.1f} Hz",
                         f"{100*(fa-ft)/ft:+.2f}%"])
    else:
        rows.append(["Passive best loss (strings)",
                     "", f"{res['passive_str_best_loss']:.4e}"])
        rows.append(["Co-design best loss",
                     "", f"{res['strings_best_loss']:.4e}"])
        rows.append(["Improvement",
                     "", f"{imp_str:.1f}%"])
        rows.append(["First 6 modes (Hz)", "", ", ".join(str_modes)])
        rows.append(["", "", ""])
        rows.append(["String tuning", "target Hz -> actual Hz", "detune"])
        for i, (ft, fa) in enumerate(zip(
                res["target_pitches_hz"], res["actual_pitches_hz"])):
            rows.append([f"  String {i+1}",
                         f"{ft:.1f} -> {fa:.1f} Hz",
                         f"{100*(fa-ft)/ft:+.2f}%"])

# Random baseline rows
if rand_lqr is not None or rand_str is not None:
    rows.append(["", "", ""])
    rows.append(["Random baseline (best of N)", "LQR", "Strings"])
    lqr_rand_cell = (f"{rand_lqr['best']:.4e}  (N={rand_lqr['n_samples']})"
                     if rand_lqr else "")
    str_rand_cell = (f"{rand_str['best']:.4e}  (N={rand_str['n_samples']})"
                     if rand_str else "")
    rows.append(["  Best random loss", lqr_rand_cell, str_rand_cell])
    if rand_lqr and MODE in ("lqr", "both"):
        gap = 100 * (rand_lqr["best"] - res["lqr_best_loss"]) / rand_lqr["best"]
        rows.append(["  Joint better than random by", f"{gap:.1f}%", ""])
    if rand_str and MODE in ("strings", "both"):
        gap = 100 * (rand_str["best"] - res["strings_best_loss"]) / rand_str["best"]
        rows.append(["  Joint better than random by", "", f"{gap:.1f}%"])

# Ablation ladder section (LQR only)
if abl_res is not None and MODE in ("lqr", "both"):
    rows.append(["", "", ""])
    rows.append(["Co-design ablation ladder", "LQR loss", "vs passive"])
    rows.append(["  Passive",
                 f"{abl_res['passive_loss']:.4e}", "(reference)"])
    rows.append(["  Fixed Q/R",
                 f"{abl_res['fixedqr_loss']:.4e}",
                 f"{abl_res['pct_fixedqr_vs_passive']:+.1f}%"])
    rows.append(["  Sequential",
                 f"{abl_res['sequential_loss']:.4e}",
                 f"{abl_res['pct_sequential_vs_passive']:+.1f}%"])
    rows.append(["  Joint",
                 f"{abl_res['joint_loss']:.4e}",
                 f"{abl_res['pct_joint_vs_passive']:+.1f}%"])
    rows.append(["", "", ""])
    rows.append(["Value of tuning Q/R (joint vs fixed-QR)", "",
                 f"{abl_res['pct_joint_vs_fixedqr']:+.1f}%"])
    rows.append(["Value of joint vs sequential", "",
                 f"{abl_res['pct_joint_vs_sequential']:+.1f}%"])

tbl = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
tbl.auto_set_column_width(col=list(range(len(rows[0]))))
ax.set_title(
    f"Results summary  |  material={cfg['material']}  "
    f"grid={cfg['Nx']}x{cfg['Ny']}  M={cfg['M']}",
    pad=60, fontsize=11, 
)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "fig_summary_table.png", dpi=160, bbox_inches="tight")
print("Saved fig_summary_table.png")
plt.show()


# =============================================================================
# Figure 6: Geometry comparison -- passive-optimal vs joint-optimal thickness
# =============================================================================
def _reconstruct_thickness(c_coef, cfg_dict):
    """JAX-free re-implementation of codesign_core.thickness.

    Kept self-contained so analysis.py does not require a JAX install.
    """
    c    = np.asarray(c_coef)
    Nx   = cfg_dict["Nx"]
    Ny   = cfg_dict["Ny"]
    Lx   = cfg_dict["Lx"]
    Ly   = cfg_dict["Ly"]
    M    = c.shape[0]
    h0   = 3.0e-3
    xs   = np.linspace(0, Lx, Nx)
    ys   = np.linspace(0, Ly, Ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    perturb = np.zeros_like(XX)
    for i in range(M):
        for j in range(M):
            perturb += (c[i, j]
                        * np.sin((i + 1) * np.pi * XX / Lx)
                        * np.sin((j + 1) * np.pi * YY / Ly))
    sig   = 1.0 / (1.0 + np.exp(-perturb / h0))
    h_min = 0.5 * h0
    h_max = 3.0 * h0
    return XX, YY, h_min + (h_max - h_min) * sig


if MODE in ("lqr", "both") and "c_lqr" in demo.files:
    if "c_passive_lqr" not in demo.files:
        print("Skipping fig_geometry_comparison.png: c_passive_lqr not in "
              "best_params.npz. Add it to save_args in demo.ipynb (one-liner).")
    else:
        XX, YY, h_passive = _reconstruct_thickness(demo["c_passive_lqr"], cfg)
        XX, YY, h_joint   = _reconstruct_thickness(demo["c_lqr"],         cfg)

        diff       = (h_joint - h_passive) * 1e3
        diff_abs_max = max(abs(diff.min()), abs(diff.max())) + 1e-9

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        im0 = axes[0].pcolormesh(XX * 1e3, YY * 1e3, h_passive * 1e3,
                                 cmap="viridis", shading="auto")
        axes[0].set_title("Passive-only geometry")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        axes[0].set_aspect("equal")
        plt.colorbar(im0, ax=axes[0], label="h (mm)")

        im1 = axes[1].pcolormesh(XX * 1e3, YY * 1e3, h_joint * 1e3,
                                 cmap="viridis", shading="auto")
        axes[1].set_title("Joint co-design geometry")
        axes[1].set_xlabel("x (mm)")
        axes[1].set_aspect("equal")
        plt.colorbar(im1, ax=axes[1], label="h (mm)")

        im2 = axes[2].pcolormesh(XX * 1e3, YY * 1e3, diff,
                                 cmap="RdBu_r",
                                 vmin=-diff_abs_max, vmax=diff_abs_max,
                                 shading="auto")
        axes[2].set_title("Difference  (joint minus passive)")
        axes[2].set_xlabel("x (mm)")
        axes[2].set_aspect("equal")
        plt.colorbar(im2, ax=axes[2], label="dh (mm)")

        fig.suptitle(
            "How geometry changes when a controller is available  "
            f"(material={cfg['material']})", fontsize=12
        )
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "fig_geometry_comparison.png", dpi=160)
        print("Saved fig_geometry_comparison.png")
        plt.show()


# =============================================================================
# Figures 7 + 8: Ablation ladder and FRF overlay
# =============================================================================
if abl is not None and abl_res is not None:

    # -------------------------------------------------------------------------
    # Figure 7: Loss ladder  (Random | Passive | Fixed-QR | Sequential | Joint)
    # -------------------------------------------------------------------------
    passive_loss = abl_res["passive_loss"]

    losses = [
        abl_res["passive_loss"],
        abl_res["fixedqr_loss"],
        abl_res["sequential_loss"],
        abl_res["joint_loss"],
    ]
    labels = [
        "Passive\n(no controller)",
        "Fixed Q/R\n(c, positions)",
        "Sequential\n(passive c then controller)",
        "Joint\n(all four variables)",
    ]
    colors = ["#999999", "#4c9fcb", "#e29657", "#c6423f"]

    # Prepend random baseline bar if available
    if rand_lqr is not None:
        losses.insert(0, rand_lqr["best"])
        labels.insert(0, f"Random\n({rand_lqr['n_samples']} samples)")
        colors.insert(0, "#d4d4d4")

    # Percentages always relative to the passive (uncontrolled) reference
    pcts = [100.0 * (passive_loss - L) / passive_loss for L in losses]

    fig, ax = plt.subplots(figsize=(9 if rand_lqr else 8, 4.5))
    bars = ax.bar(labels, losses, color=colors, edgecolor="black", linewidth=0.6)
    for bar, L, pct in zip(bars, losses, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{L:.3e}\n({pct:+.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Final loss  (log scale)")
    ax.set_title("Co-design validation ladder  (lower = closer to target)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_ylim(top=max(losses) * 3.0)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_ablation_ladder.png", dpi=160)
    print("Saved fig_ablation_ladder.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Figure 8: FRF overlay for all four designs
    # -------------------------------------------------------------------------
    freqs_abl  = abl["freqs"]
    target_abl = abl["target"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    T = norm(target_abl)
    ax.fill_between(freqs_abl, 0, T, color="gray", alpha=0.25, label="target")
    ax.plot(freqs_abl, T, "k--", lw=1.2, alpha=0.6)

    curves = [
        ("Passive",    abl["H_passive"],    "#999999", "-"),
        ("Fixed Q/R",  abl["H_fixedqr"],    "#4c9fcb", "-"),
        ("Sequential", abl["H_sequential"], "#e29657", "-"),
        ("Joint",      abl["H_joint"],      "#c6423f", "-"),
    ]
    for lbl, H, color, ls in curves:
        ax.plot(freqs_abl, norm(H), lw=1.9, ls=ls, color=color, label=lbl)

    # Also overlay the best random FRF if the npz was saved
    rand_lqr_npz_path = OUTPUTS_DIR / "random_baseline_lqr.npz"
    if rand_lqr is not None and rand_lqr_npz_path.exists():
        rnpz = np.load(rand_lqr_npz_path, allow_pickle=True)
        if "best_frf" in rnpz.files:
            ax.plot(freqs_abl, norm(np.asarray(rnpz["best_frf"])),
                    lw=1.4, ls=":", color="#aaaaaa",
                    label=f"Random best (N={rand_lqr['n_samples']})")

    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H(jw)|  normalized")
    ax.set_title("FRF across the co-design ladder")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_ablation_frf.png", dpi=160)
    print("Saved fig_ablation_frf.png")
    plt.show()

else:
    print(f"Skipping ablation figures "
          f"({abl_npz_path.name} and/or {abl_json_path.name} not found; "
          f"run scripts/run_ablations.py first).")


# =============================================================================
# Console summary
# =============================================================================
print()
print("=" * 60)
print("Results summary")
print("=" * 60)

if MODE in ("lqr", "both"):
    print(f"LQR passive best loss  : {res['passive_lqr_best_loss']:.4e}")
    print(f"LQR co-design best     : {res['lqr_best_loss']:.4e}"
          f"  ({res['lqr_improvement_pct']:+.1f}%)")
    if rand_lqr:
        gap = 100 * (rand_lqr["best"] - res["lqr_best_loss"]) / rand_lqr["best"]
        print(f"LQR random best        : {rand_lqr['best']:.4e}"
              f"  (joint is {gap:.1f}% better)")
    print(f"LQR modes (Hz)         : "
          f"{[f'{f:.0f}' for f in res['omega_hz_lqr'][:6]]}")
    print()

if MODE in ("strings", "both"):
    if imp_str is None:
        imp_str = (100 * (res["passive_str_best_loss"] - res["strings_best_loss"])
                   / res["passive_str_best_loss"])
    print(f"Strings passive best   : {res['passive_str_best_loss']:.4e}")
    print(f"Strings co-design best : {res['strings_best_loss']:.4e}"
          f"  ({imp_str:+.1f}%)")
    if rand_str:
        gap = 100 * (rand_str["best"] - res["strings_best_loss"]) / rand_str["best"]
        print(f"Strings random best    : {rand_str['best']:.4e}"
              f"  (joint is {gap:.1f}% better)")
    print(f"Strings modes (Hz)     : "
          f"{[f'{f:.0f}' for f in res['omega_hz_strings'][:6]]}")
    print()
    print("String tuning:")
    for i, (ft, fa, T) in enumerate(zip(
            res["target_pitches_hz"],
            res["actual_pitches_hz"],
            res["tensions_N"])):
        print(f"  String {i+1}: {ft:.1f} -> {fa:.1f} Hz "
              f"({100*(fa-ft)/ft:+.2f}%)  tension {T:.1f} N")

if abl_res is not None:
    print()
    print("Co-design ladder:")
    if rand_lqr:
        print(f"  Random        : {rand_lqr['best']:.4e}")
    print(f"  Passive       : {abl_res['passive_loss']:.4e}   (reference)")
    print(f"  Fixed Q/R     : {abl_res['fixedqr_loss']:.4e}   "
          f"({abl_res['pct_fixedqr_vs_passive']:+.1f}% vs passive)")
    print(f"  Sequential    : {abl_res['sequential_loss']:.4e}   "
          f"({abl_res['pct_sequential_vs_passive']:+.1f}% vs passive)")
    print(f"  Joint         : {abl_res['joint_loss']:.4e}   "
          f"({abl_res['pct_joint_vs_passive']:+.1f}% vs passive)")
    print(f"  joint vs fixed-QR  : {abl_res['pct_joint_vs_fixedqr']:+.1f}%")
    print(f"  joint vs sequential: {abl_res['pct_joint_vs_sequential']:+.1f}%")