"""Quick GPU sanity check for JAX.

Run:
    python verify_gpu.py

Expected on a 4070 Ti Super:
    Default backend: gpu
    Devices: [CudaDevice(id=0)]
    4096x4096 fp32 matmul x 20: ~0.3s  (~40 TFLOPS)
"""
import time
import jax
import jax.numpy as jnp


def main():
    print(f"JAX version     : {jax.__version__}")
    print(f"Default backend : {jax.default_backend()}")
    print(f"Devices         : {jax.devices()}")
    print()

    if jax.default_backend() != "gpu":
        print("WARNING: JAX is not using the GPU.")
        print("Diagnostics to try:")
        print("  1. Is `nvidia-smi` working inside your shell/WSL?")
        print("  2. Did you install jax[cuda12] (not plain jax)?")
        print("  3. Restart the shell so it picks up driver changes.")
        return

    # fp32 matmul benchmark (what you'll actually use for medium/large)
    N = 4096
    x = jax.random.normal(jax.random.PRNGKey(0), (N, N), dtype=jnp.float32)
    x.block_until_ready()
    # Warmup
    for _ in range(3):
        y = x @ x
    y.block_until_ready()

    n_iters = 20
    t0 = time.time()
    for _ in range(n_iters):
        y = x @ x
    y.block_until_ready()
    elapsed = time.time() - t0
    tflops = (2 * N ** 3 * n_iters) / elapsed / 1e12
    print(f"fp32 matmul  {N}x{N} x {n_iters}: {elapsed:.3f}s  (~{tflops:.1f} TFLOPS)")

    # Eigh benchmark at the size the medium profile uses
    import jax.scipy.linalg as jsl
    A = jax.random.normal(jax.random.PRNGKey(1), (2000, 2000), dtype=jnp.float32)
    A = 0.5 * (A + A.T)
    A.block_until_ready()
    t0 = time.time()
    w, V = jsl.eigh(A)
    V.block_until_ready()
    print(f"fp32 eigh    2000x2000:       {time.time() - t0:.3f}s")

    print()
    print("GPU is working. You can run run_batched.py next.")


if __name__ == "__main__":
    main()
