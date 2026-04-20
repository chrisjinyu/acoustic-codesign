"""Sensitivity sweep, mode-aware.

In LQR mode, sweeps over n_actuators (the Config field). In strings mode,
sweeps over the number of strings (length of StringConfig.target_pitches_hz,
with pitches chosen from a default ladder).

Both modes monkeypatch the source file in place and subprocess out to
scripts/run_batched.py so each run gets a fresh Python interpreter with a
fresh JIT cache.

Usage (from repo root):
    python scripts/run_sweep.py                        # LQR actuator sweep
    python scripts/run_sweep.py --mode strings         # strings count sweep
    python scripts/run_sweep.py --values 3 5 7         # custom sweep values

All per-sweep .npz files land in outputs/ by default.
"""
from __future__ import annotations
import sys
import argparse
import re
import subprocess
import shutil
from pathlib import Path

# Resolve repo root so we can locate source files regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Pitches spaced roughly like violin tuning (G3 D4 A4 E5 + extras)
DEFAULT_PITCH_LADDER = [196.0, 293.7, 440.0, 659.3, 523.3, 349.2, 783.99, 880.0]


def patch_n_actuators(n: int, path: Path = REPO_ROOT / "codesign_core.py"):
    content = path.read_text()
    patched = re.sub(r'n_actuators:\s*int\s*=\s*\d+',
                     f'n_actuators: int = {n}', content)
    path.write_text(patched)


def patch_n_strings(n: int, path: Path = REPO_ROOT / "codesign_strings.py"):
    content = path.read_text()
    pitches = DEFAULT_PITCH_LADDER[:n]
    tup = "(" + ", ".join(f"{p}" for p in pitches) + ("," if n == 1 else "") + ")"
    patched = re.sub(
        r'target_pitches_hz:\s*tuple\s*=\s*\([^)]*\)',
        f'target_pitches_hz: tuple = {tup}',
        content,
    )
    path.write_text(patched)


def run_one(mode: str, value: int, seeds: int, steps: int, out_dir: Path):
    if mode == "lqr":
        print(f"\n=== LQR sweep: n_actuators = {value} ===")
        patch_n_actuators(value)
    else:
        print(f"\n=== Strings sweep: n_strings = {value} ===")
        patch_n_strings(value)

    out = out_dir / f"multistart_{mode}_{value}.npz"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_batched.py"),
         "--mode", mode,
         "--seeds", str(seeds),
         "--steps", str(steps),
         "--outfile", str(out)],
        check=True,
        cwd=str(REPO_ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description="Sensitivity sweep runner.")
    parser.add_argument("--mode", choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--values", type=int, nargs="+", default=None,
                        help="Values to sweep (default: [2,4,6,8] for lqr, [2,3,4,5] for strings)")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", type=str, default=str(OUTPUTS_DIR),
                        help="Directory for output .npz files (default: outputs/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.values is None:
        args.values = [2, 4, 6, 8] if args.mode == "lqr" else [2, 3, 4, 5]

    print(f"Sweep mode: {args.mode}  values: {args.values}  out_dir: {out_dir}")

    # Back up source files before patching so we can restore them afterward.
    src_lqr = REPO_ROOT / "codesign_core.py"
    src_str = REPO_ROOT / "codesign_strings.py"
    backup_lqr = src_lqr.with_suffix(".py.bak")
    backup_str = src_str.with_suffix(".py.bak")
    shutil.copy(src_lqr, backup_lqr)
    shutil.copy(src_str, backup_str)

    try:
        for v in args.values:
            run_one(args.mode, v, args.seeds, args.steps, out_dir)
    finally:
        # Always restore originals, even if a run fails.
        shutil.copy(backup_lqr, src_lqr)
        shutil.copy(backup_str, src_str)
        backup_lqr.unlink(missing_ok=True)
        backup_str.unlink(missing_ok=True)
        print("\nSource files restored.")

    print(f"\nSweep complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()