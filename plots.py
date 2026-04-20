"""Visualization helpers for acoustic co-design results.

All functions return the Axes or Figure they draw on, so you can compose them
into larger grids for your report.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from codesign_core import cfg, X, Y, thickness, solve_modes


# -----------------------------------------------------------------------------
def plot_thickness(c, ax=None, title="Thickness  h(x, y)"):
    """Heatmap of the plate thickness field (in mm)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    h = np.asarray(thickness(c))
    im = ax.pcolormesh(np.asarray(X) * 1e3,
                       np.asarray(Y) * 1e3,
                       h * 1e3,
                       cmap="viridis", shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="h (mm)")
    return ax


# -----------------------------------------------------------------------------
def plot_mode_shapes(c, positions=None, n_show=6, figsize=(12, 6)):
    """Grid of mode shapes. If actuator positions are passed, overlay them."""
    omega, Phi = solve_modes(c)
    Phi_np = np.asarray(Phi).reshape(cfg.Nx, cfg.Ny, cfg.n_modes)
    freqs_hz = np.asarray(omega) / (2 * np.pi)

    ncols = (n_show + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=figsize)
    for i, ax in enumerate(axes.ravel()):
        if i >= n_show:
            ax.axis("off")
            continue
        field = Phi_np[:, :, i]
        vmax = np.abs(field).max() + 1e-12
        ax.pcolormesh(np.asarray(X) * 1e3,
                      np.asarray(Y) * 1e3,
                      field,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        if positions is not None:
            px = np.asarray(positions)[:, 0] * cfg.Lx * 1e3
            py = np.asarray(positions)[:, 1] * cfg.Ly * 1e3
            ax.scatter(px, py, s=80, marker="o",
                       facecolors="none", edgecolors="black", linewidths=1.6)
        ax.set_aspect("equal")
        ax.set_title(f"mode {i + 1}:  {freqs_hz[i]:.0f} Hz", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
def plot_frf_comparison(freqs, target, frfs_dict, ax=None, figsize=(9, 4)):
    """Overlay target spectrum with one or more FRF curves.

    Parameters
    ----------
    freqs : array of frequencies in Hz
    target : array, the target spectrum (will be normalized to max=1)
    frfs_dict : dict label -> FRF magnitude array
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    freqs_np = np.asarray(freqs)
    T = np.asarray(target)
    T = T / (T.max() + 1e-12)
    ax.fill_between(freqs_np, 0, T, color="gray", alpha=0.25, label="target")
    ax.plot(freqs_np, T, "k--", lw=1.5, alpha=0.7)
    for label, H in frfs_dict.items():
        H = np.asarray(H)
        H = H / (H.max() + 1e-12)
        ax.plot(freqs_np, H, lw=1.8, label=label)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H(jw)|   (normalized)")
    ax.set_title("Frequency response: target vs designs")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


# -----------------------------------------------------------------------------
def plot_loss_history(history, ax=None, figsize=(6, 3.5), label=None):
    """Semilog plot of optimization loss across iterations."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.semilogy(history, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Optimization history")
    ax.grid(alpha=0.3, which="both")
    if label is not None:
        ax.legend()
    return ax


# -----------------------------------------------------------------------------
def dashboard(params, history, freqs, target, baseline_frf=None, optimized_frf=None, string_table_data=None):
    """One-call figure that puts the money shot on a single page.

    Handles both co-design variants:
      LQR mode:     params = (c, positions, log_q, log_r)
      Strings mode: params = (c, log_tensions)
    """
    is_lqr = len(params) == 4
    c = params[0]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Top Left: Thickness
    ax1 = fig.add_subplot(gs[0, 0])
    plot_thickness(c, ax=ax1, title="Optimized thickness")

    # Top Middle/Right: FRF Comparison
    ax2 = fig.add_subplot(gs[0, 1:])
    frfs = {}
    if baseline_frf is not None:
        frfs["passive only"] = baseline_frf
    if optimized_frf is not None:
        frfs["co-designed"] = optimized_frf
    plot_frf_comparison(freqs, target, frfs, ax=ax2)

    # Bottom Left: Loss History
    ax3 = fig.add_subplot(gs[1, 0])
    plot_loss_history(history, ax=ax3)

    # Bottom Middle/Right: Mode-Specific Data (Actuators OR Table)
    ax4 = fig.add_subplot(gs[1, 1:])

    if is_lqr:
        # --- LQR Mode: Plot Actuators ---
        ax4.add_patch(plt.Rectangle((0, 0), cfg.Lx * 1e3, cfg.Ly * 1e3,
                                    fill=False, edgecolor="black", lw=1.5))
        _, positions, _, _ = params
        px = np.asarray(positions)[:, 0] * cfg.Lx * 1e3
        py = np.asarray(positions)[:, 1] * cfg.Ly * 1e3
        ax4.scatter(px, py, s=120, c="tab:red", edgecolors="black",
                    label="piezo actuators", zorder=3)
        ax4.set_title("Actuator layout")
        
        dx, dy = cfg.disturb_xy
        ax4.scatter([dx * cfg.Lx * 1e3], [dy * cfg.Ly * 1e3],
                    s=120, c="tab:blue", marker="s", edgecolors="black",
                    label="excitation / bridge", zorder=3)
        ax4.set_aspect("equal")
        ax4.set_xlim(-5, cfg.Lx * 1e3 + 5)
        ax4.set_ylim(-5, cfg.Ly * 1e3 + 5)
        ax4.set_xlabel("x (mm)")
        ax4.set_ylabel("y (mm)")
        ax4.legend(loc="upper right")
        ax4.grid(alpha=0.3)
        
    elif string_table_data is not None:
        # --- Strings Mode: Plot Tuning Table ---
        ax4.axis('off')
        rows, target_pitches, f0_str, tensions_N = string_table_data
        for i, (ft, fa, T) in enumerate(zip(target_pitches, f0_str, tensions_N)):
            rows.append((f'String {i+1}', f'{ft:.1f}', f'{fa:.1f}',
                         f'{100*(fa-ft)/ft:+.2f}%', f'{T:.1f}'))
        tbl = ax4.table(cellText=rows[1:], colLabels=rows[0],
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.8)
        ax4.set_title('Optimized String Tuning')
        
    else:
        # --- Fallback: Passive Plate Only ---
        ax4.add_patch(plt.Rectangle((0, 0), cfg.Lx * 1e3, cfg.Ly * 1e3,
                                    fill=False, edgecolor="black", lw=1.5))
        ax4.set_title("Plate layout (strings mode: no actuators)")
        dx, dy = cfg.disturb_xy
        ax4.scatter([dx * cfg.Lx * 1e3], [dy * cfg.Ly * 1e3],
                    s=120, c="tab:blue", marker="s", edgecolors="black",
                    label="excitation / bridge", zorder=3)
        ax4.set_aspect("equal")
        ax4.set_xlim(-5, cfg.Lx * 1e3 + 5)
        ax4.set_ylim(-5, cfg.Ly * 1e3 + 5)
        ax4.set_xlabel("x (mm)")
        ax4.set_ylabel("y (mm)")
        ax4.legend(loc="upper right")
        ax4.grid(alpha=0.3)

    return fig
