"""Sensitivity sweep, mode-aware.

In LQR mode, sweeps over n_actuators (the Config field). In strings mode,
sweeps over the number of strings (length of StringConfig.target_pitches_hz,
with pitches chosen from a default ladder).

Both modes use the same monkeypatch trick as before to avoid JAX retracing
across different shapes: we edit the source file in place, then subprocess
out to run_batched.py so each run gets a fresh Python interpreter with a
fresh JIT cache.

Usage:
    python run_sweep.py                       # LQR actuator sweep
    python run_sweep.py --mode strings        # strings count sweep
    python run_sweep.py --values 3 5 7        # custom sweep values
"""
from __future__ import annotations
import argparse
import re
import subprocess
import shutil
from pathlib import Path


# Pitches spaced roughly like violin tuning (G3 D4 A4 E5 + extras)
DEFAULT_PITCH_LADDER = [196.0, 293.7, 440.0, 659.3, 523.3, 349.2, 783.99, 880.0]


def patch_n_actuators(n: int, path: Path = Path("codesign_core.py")):
    content = path.read_text()
    patched = re.sub(r'n_actuators:\s*int\s*=\s*\d+',
                     f'n_actuators: int = {n}', content)
    path.write_text(patched)


def patch_n_strings(n: int, path: Path = Path("codesign_strings.py")):
    content = path.read_text()
    pitches = DEFAULT_PITCH_LADDER[:n]
    tup = "(" + ", ".join(f"{p}" for p in pitches) + ("," if n == 1 else "") + ")"
    patched = re.sub(
        r'target_pitches_hz:\s*tuple\s*=\s*\([^)]*\)',
        f'target_pitches_hz: tuple = {tup}',
        content,
    )
    path.write_text(patched)


def run_one(mode: str, value: int, seeds: int, steps: int, backup_dir: Path):
    if mode == "lqr":
        print(f"\n=== LQR sweep: n_actuators = {value} ===")
        patch_n_actuators(value)
    else:
        print(f"\n=== Strings sweep: n_strings = {value} ===")
        patch_n_strings(value)

    out = backup_dir / f"multistart_{mode}_{value}.npz"
    subprocess.run(
        ["python", "run_batched.py",
         "--mode", mode,
         "--seeds", str(seeds),
         "--steps", str(steps),
         "--outfile", str(out)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lqr", "strings"], default="lqr")
    parser.add_argument("--values", type=int, nargs="+",
                        default=None,
                        help="Values to sweep over (default: [2,4,6,8] for lqr, [2,3,4,5] for strings)")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    values = args.values
    if values is None:
        values = [2, 4, 6, 8] if args.mode == "lqr" else [2, 3, 4, 5]

    # Back up the source file we are going to mutate
    source_file = Path("codesign_core.py" if args.mode == "lqr"
                       else "codesign_strings.py")
    backup = source_file.with_suffix(source_file.suffix + ".bak")
    shutil.copy(source_file, backup)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    try:
        for v in values:
            run_one(args.mode, v, args.seeds, args.steps, out_dir)
    finally:
        # Always restore the source file, even if a run fails
        shutil.copy(backup, source_file)
        backup.unlink()
        print(f"\nRestored {source_file}")

    print(f"\nSweep complete. Files: multistart_{args.mode}_N.npz for N in {values}")


if __name__ == "__main__":
    main()
