"""
analysis.py  --  regenerate all report figures from saved outputs.

Run after demo.ipynb and run_batched.py have both completed:

    python scripts/analysis.py

Reads (from outputs/):
    results.json               -- scalar results and metadata from demo.ipynb
    best_params.npz            -- FRF arrays and histories from demo.ipynb
    multistart_lqr.npz         -- multi-start trajectories from run_batched.py --mode lqr
    multistart_strings.npz     -- multi-start trajectories from run_batched.py --mode strings

Writes (to outputs/):
    fig_frf_lqr.pdf            -- LQR FRF comparison (passive vs co-designed)
    fig_frf_strings.pdf        -- strings FRF comparison (passive vs co-designed)
    fig_multistart_lqr.pdf     -- multi-start variance, LQR
    fig_multistart_strings.pdf -- multi-start variance, strings
    fig_summary_table.pdf      -- one-page results summary
"""
from __future__ import annotations
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Allow imports of codesign* modules from the repo root regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Load demo outputs ────────────────────────────────────────────────────────
with open(OUTPUTS_DIR / "results.json") as f:
    res = json.load(f)

demo  = np.load(OUTPUTS_DIR / "best_params.npz", allow_pickle=True)
MODE  = res["mode"]
cfg   = res["config"]

print(f"Mode     : {MODE}")
print(f"Material : {cfg['material']}  grid {cfg['Nx']}x{cfg['Ny']}  M={cfg['M']}")
print()


# ── Helper: normalise an FRF array to max=1 ──────────────────────────────────
def norm(H):
    return np.asarray(H) / (np.asarray(H).max() + 1e-12)


# =============================================================================
# Figure 1: LQR FRF comparison
# =============================================================================
if MODE in ("lqr", "both") and "H_lqr" in demo:
    freqs_lqr   = demo["freqs_lqr"]
    target_lqr  = demo["target_lqr"]
    H_pass_lqr  = demo["H_passive_lqr"]
    H_lqr       = demo["H_lqr"]

    passive_loss = res["passive_lqr_best_loss"]
    lqr_loss     = res["lqr_best_loss"]
    imp_lqr      = res["lqr_improvement_pct"]

    fig, ax = plt.subplots(figsize=(9, 4))
    T = norm(target_lqr)
    ax.fill_between(freqs_lqr, 0, T, color="gray", alpha=0.25, label="target")
    ax.plot(freqs_lqr, T, "k--", lw=1.2, alpha=0.6)
    ax.plot(freqs_lqr, norm(H_pass_lqr), lw=1.8,
            label=f"passive only  (loss={passive_loss:.3e})")
    ax.plot(freqs_lqr, norm(H_lqr),      lw=1.8,
            label=f"LQR co-designed  (loss={lqr_loss:.3e})")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H(jω)|  normalized")
    ax.set_title(f"LQR co-design: {imp_lqr:.1f}% improvement over passive")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_frf_lqr.pdf", dpi=160)
    print("Saved fig_frf_lqr.pdf")
    plt.show()


# =============================================================================
# Figure 2: Strings FRF comparison
# =============================================================================
if MODE in ("strings", "both") and "H_str" in demo:
    freqs_str       = demo["freqs_str"]
    target_str_nom  = demo["target_str_nominal"]
    H_pass_str      = demo["H_passive_str"]
    H_str           = demo["H_str"]

    passive_loss_str = res["passive_str_best_loss"]
    str_loss         = res["strings_best_loss"]
    imp_str          = 100 * (passive_loss_str - str_loss) / passive_loss_str

    target_pitches = np.array(res["target_pitches_hz"])
    actual_pitches = np.array(res["actual_pitches_hz"])

    fig, ax = plt.subplots(figsize=(9, 4))
    T = norm(target_str_nom)
    ax.fill_between(freqs_str, 0, T, color="gray", alpha=0.25,
                    label="target (nominal tensions)")
    ax.plot(freqs_str, T, "k--", lw=1.2, alpha=0.6)
    ax.plot(freqs_str, norm(H_pass_str), lw=1.8,
            label=f"passive only  (loss={passive_loss_str:.3e})")
    ax.plot(freqs_str, norm(H_str),      lw=1.8,
            label=f"strings co-designed  (loss={str_loss:.3e})")
    # Mark target pitches
    for ft in target_pitches:
        ax.axvline(ft, color="red", ls=":", alpha=0.4, lw=1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H(jω)|  normalized")
    ax.set_title(f"Strings co-design: {imp_str:.1f}% improvement over passive")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_frf_strings.pdf", dpi=160)
    print("Saved fig_frf_strings.pdf")
    plt.show()


# =============================================================================
# Figure 3: Multi-start variance -- LQR
# =============================================================================
lqr_batch_path = OUTPUTS_DIR / "multistart_lqr.npz"
if lqr_batch_path.exists():
    b = np.load(lqr_batch_path)
    history_all  = b["history"]      # (steps, seeds)
    best_losses  = b["best_losses"]  # (seeds,)
    n_seeds      = history_all.shape[1]

    steps  = np.arange(history_all.shape[0])
    mean   = history_all.mean(axis=1)
    std    = history_all.std(axis=1)
    best   = history_all.min(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                    label="mean ± 1 std")
    ax.semilogy(steps, mean, lw=2,  label="mean across seeds")
    ax.semilogy(steps, best, lw=2, ls="--", label="best-step trajectory")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"LQR multi-start ({n_seeds} seeds)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_multistart_lqr.pdf", dpi=160)
    print("Saved fig_multistart_lqr.pdf")
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
    b = np.load(str_batch_path)
    history_all  = b["history"]
    best_losses  = b["best_losses"]
    n_seeds      = history_all.shape[1]

    steps = np.arange(history_all.shape[0])
    mean  = history_all.mean(axis=1)
    std   = history_all.std(axis=1)
    best  = history_all.min(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                    label="mean ± 1 std")
    ax.semilogy(steps, mean, lw=2,  label="mean across seeds")
    ax.semilogy(steps, best, lw=2, ls="--", label="best-step trajectory")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Strings multi-start ({n_seeds} seeds)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fig_multistart_strings.pdf", dpi=160)
    print("Saved fig_multistart_strings.pdf")
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
fig = plt.figure(figsize=(9, 5))
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
    rows.append(["Improvement",
                 f"{res['lqr_improvement_pct']:.1f}%",
                 f"{imp_str:.1f}%" if MODE == "both" else ""])
    rows.append(["First 6 modes (Hz)",
                 ", ".join(lqr_modes), ""])

if MODE in ("strings", "both"):
    str_modes = [f"{f:.0f}" for f in res["omega_hz_strings"][:6]]
    if MODE == "both":
        # patch rows already inserted
        rows[2][2] = f"{res['strings_best_loss']:.4e}"
        rows[3][2] = f"{imp_str:.1f}%"
        rows[4][2] = ", ".join(str_modes)
        # string tuning rows
        rows.append(["", "", ""])
        rows.append(["String tuning", "target Hz → actual Hz", "detune"])
        for i, (ft, fa) in enumerate(zip(
                res["target_pitches_hz"], res["actual_pitches_hz"])):
            rows.append([f"  String {i+1}",
                         f"{ft:.1f} → {fa:.1f} Hz",
                         f"{100*(fa-ft)/ft:+.2f}%"])
    else:
        rows.append(["Passive best loss (strings)",
                     "", f"{res['passive_str_best_loss']:.4e}"])
        rows.append(["Co-design best loss",
                     "", f"{res['strings_best_loss']:.4e}"])
        rows.append(["Improvement", "", f"{imp_str:.1f}%"])
        rows.append(["First 6 modes (Hz)", "", ", ".join(str_modes)])
        rows.append(["", "", ""])
        rows.append(["String tuning", "target Hz → actual Hz", "detune"])
        for i, (ft, fa) in enumerate(zip(
                res["target_pitches_hz"], res["actual_pitches_hz"])):
            rows.append([f"  String {i+1}",
                         f"{ft:.1f} → {fa:.1f} Hz",
                         f"{100*(fa-ft)/ft:+.2f}%"])

tbl = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
ax.set_title(
    f"Results summary  |  material={cfg['material']}  "
    f"grid={cfg['Nx']}x{cfg['Ny']}  M={cfg['M']}",
    pad=20, fontsize=11,
)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "fig_summary_table.pdf", dpi=160, bbox_inches="tight")
print("Saved fig_summary_table.pdf")
plt.show()


# =============================================================================
# Console summary
# =============================================================================
print()
print("=" * 55)
print("Results summary")
print("=" * 55)
if MODE in ("lqr", "both"):
    print(f"LQR passive best loss  : {res['passive_lqr_best_loss']:.4e}")
    print(f"LQR co-design best     : {res['lqr_best_loss']:.4e}"
          f"  ({res['lqr_improvement_pct']:+.1f}%)")
    print(f"LQR modes (Hz)         : "
          f"{[f'{f:.0f}' for f in res['omega_hz_lqr'][:6]]}")
    print()
if MODE in ("strings", "both"):
    print(f"Strings passive best   : {res['passive_str_best_loss']:.4e}")
    print(f"Strings co-design best : {res['strings_best_loss']:.4e}"
          f"  ({imp_str:+.1f}%)")
    print(f"Strings modes (Hz)     : "
          f"{[f'{f:.0f}' for f in res['omega_hz_strings'][:6]]}")
    print()
    print("String tuning:")
    for i, (ft, fa, T) in enumerate(zip(
            res["target_pitches_hz"],
            res["actual_pitches_hz"],
            res["tensions_N"])):
        print(f"  String {i+1}: {ft:.1f} → {fa:.1f} Hz "
              f"({100*(fa-ft)/ft:+.2f}%)  tension {T:.1f} N")