"""
analysis.py  --  load saved outputs and regenerate all report figures.
Run after demo.ipynb and run_batched.py have completed.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# ── Load demo outputs ────────────────────────────────────────────────────────
with open("results.json") as f:
    res = json.load(f)

demo = np.load("best_params.npz")
freqs  = demo["freqs"]
target = demo["target"]
H_codesigned = demo["H_codesigned"]  # add this to your save cell (see below)
H_passive    = demo["H_passive"]

history_passive    = demo["history_passive"]
history_codesigned = demo["history_codesigned"]

# ── Load batched outputs ─────────────────────────────────────────────────────
batched = np.load("multistart_result.npz")
history_all = batched["history"]   # shape (steps, seeds)
best_seed   = int(batched["best_seed"]) if "best_seed" in batched else None


# ── Figure 1: FRF comparison (hero) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
T = target / target.max()
ax.fill_between(freqs, 0, T, color="gray", alpha=0.25, label="target")
ax.plot(freqs, T, "k--", lw=1.2, alpha=0.6)
H_p = H_passive / H_passive.max()
H_c = H_codesigned / H_codesigned.max()
ax.plot(freqs, H_p, lw=1.8, label=f"passive only  (loss={res['passive_best_loss']:.3e})")
ax.plot(freqs, H_c, lw=1.8, label=f"co-designed   (loss={res['codesign_best_loss']:.3e})")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("|H(jw)|  normalized")
ax.set_title(f"Frequency response: {res['improvement_pct']:.1f}% improvement from co-design")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_frf.pdf", dpi=160)
plt.show()


# ── Figure 2: Multi-start variance ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
steps = np.arange(history_all.shape[0])
mean  = history_all.mean(axis=1)
std   = history_all.std(axis=1)
best  = history_all.min(axis=1)

ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="mean ± 1 std")
ax.semilogy(steps, mean, lw=2, label="mean across seeds")
ax.semilogy(steps, best, lw=2, linestyle="--", label="best seed")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title(f"Multi-start optimization ({history_all.shape[1]} seeds)")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig_multistart.pdf", dpi=160)
plt.show()


# ── Print summary table ──────────────────────────────────────────────────────
print("=" * 50)
print("Results summary")
print("=" * 50)
print(f"Passive best loss   : {res['passive_best_loss']:.4e}")
print(f"Co-design best loss : {res['codesign_best_loss']:.4e}")
print(f"Improvement         : {res['improvement_pct']:.1f}%")
print()
print(f"Multi-start ({history_all.shape[1]} seeds)")
final = history_all[-1]
print(f"  Final loss  min={final.min():.3e}  mean={final.mean():.3e}  std={final.std():.3e}")
print(f"  Best loss   min={history_all.min():.3e}")
print()
print("Co-designed modal frequencies (Hz):")
for i, f in enumerate(res["omega_hz_codesigned"]):
    print(f"  mode {i+1:2d}: {f:.1f} Hz")