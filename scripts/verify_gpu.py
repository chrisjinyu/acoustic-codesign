"""JAX GPU sanity check and benchmark.

Usage (from repo root):
    python scripts/verify_gpu.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp


def check_devices() -> None:
    devices = jax.devices()
    print(f"JAX version  : {jax.__version__}")
    print(f"Devices      : {devices}")
    backend = jax.default_backend()
    print(f"Default backend: {backend}")
    if backend == "gpu":
        print("GPU detected — JAX will use CUDA acceleration.")
    else:
        print("No GPU detected — running on CPU (expected on a laptop).")


def benchmark(size: int = 4096, repeats: int = 5) -> None:
    """Matrix-multiply benchmark to confirm XLA compilation and throughput."""
    print(f"\nBenchmark: {size}x{size} matmul, {repeats} repeats")
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (size, size))
    B = jax.random.normal(key, (size, size))

    # Warm up.
    _ = (A @ B).block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = (A @ B).block_until_ready()
        times.append(time.perf_counter() - t0)

    mean_ms = 1e3 * sum(times) / repeats
    print(f"  Mean wall time: {mean_ms:.1f} ms")


if __name__ == "__main__":
    check_devices()
    benchmark()