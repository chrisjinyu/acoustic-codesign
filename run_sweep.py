import subprocess
import jax

# We will literally rewrite the config dynamically for the sweep to avoid JAX graph recompilation issues across different shapes
sweep_counts = [2, 4, 6, 8]

for n in sweep_counts:
    print(f"=== Running sweep for n_actuators = {n} ===")
    
    # Read the codesign.py file
    with open("codesign.py", "r") as f:
        content = f.read()
        
    # Quick and dirty monkeypatch of the config for the sweep
    import re
    patched = re.sub(r'n_actuators:\s*int\s*=\s*\d+', f'n_actuators: int = {n}', content)
    
    with open("codesign.py", "w") as f:
        f.write(patched)
        
    # Execute the batched run
    subprocess.run(["python", "run_batched.py", "--seeds", "8", "--steps", "400"])
    
    # Rename the output file so it isn't overwritten
    subprocess.run(["mv", "multistart_result.npz", f"multistart_actuators_{n}.npz"])

print("Sweep complete. Data saved to multistart_actuators_N.npz")