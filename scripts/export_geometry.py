"""Export optimized plate geometry to a mesh-friendly format.

Reads best_params.npz from outputs/ (or a custom path) and writes per-mode
geometry CSVs there as well.

Usage (from repo root):
    python scripts/export_geometry.py
    python scripts/export_geometry.py --npz path/to/custom.npz
"""
from __future__ import annotations
import sys
import argparse
import numpy as np
from pathlib import Path

# Resolve repo root so codesign* modules are importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def export_geometry(c: np.ndarray, label: str, out_dir: Path) -> None:
    """Write a thickness-field CSV for the given coefficient array."""
    out_path = out_dir / f"geometry_{label}.csv"

    import codesign_core as core
    cfg = core.cfg

    xs = np.linspace(0, cfg.Lx, cfg.Nx)
    ys = np.linspace(0, cfg.Ly, cfg.Ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    # Reconstruct thickness field from sine coefficients.
    M = c.shape[0]
    h = np.zeros_like(XX)
    for i in range(M):
        for j in range(M):
            h += c[i, j] * np.sin((i + 1) * np.pi * XX / cfg.Lx) \
                          * np.sin((j + 1) * np.pi * YY / cfg.Ly)
    h = cfg.h0 + h

    rows = [["x_m", "y_m", "thickness_m"]]
    for ix in range(cfg.Nx):
        for iy in range(cfg.Ny):
            rows.append([f"{XX[ix,iy]:.6f}", f"{YY[ix,iy]:.6f}", f"{h[ix,iy]:.6f}"])

    with open(out_path, "w") as f:
        f.write("\n".join(",".join(r) for r in rows) + "\n")

    print(f"  Wrote {out_path}  ({cfg.Nx * cfg.Ny} grid points)")


def main():
    parser = argparse.ArgumentParser(description="Export geometry from best_params.npz.")
    parser.add_argument(
        "--npz",
        type=str,
        default=str(OUTPUTS_DIR / "best_params.npz"),
        help="Path to best_params.npz (default: outputs/best_params.npz)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OUTPUTS_DIR),
        help="Directory for output CSVs (default: outputs/)",
    )
    args = parser.parse_args()

    npz_file = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_file.exists():
        print(f"Error: {npz_file} not found.")
        print("Run demo.ipynb first (it saves outputs/best_params.npz).")
        sys.exit(1)

    data = np.load(npz_file)
    found_any = False

    if "c_lqr" in data:
        print("Processing LQR mode geometry...")
        export_geometry(data["c_lqr"], "lqr", out_dir)
        found_any = True

    if "c_str" in data:
        print("Processing strings mode geometry...")
        export_geometry(data["c_str"], "strings", out_dir)
        found_any = True

    if not found_any:
        print(f"Warning: neither 'c_lqr' nor 'c_str' found in {npz_file}.")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()