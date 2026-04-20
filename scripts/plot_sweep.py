import numpy as np
import matplotlib.pyplot as plt

def plot_actuator_sweep():
    actuator_counts = [2, 4, 6, 8]
    best_losses = []

    # Set up a professional plot style
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: Convergence Trajectories ---
    for n in actuator_counts:
        try:
            data = np.load(f"multistart_actuators_{n}.npz")
            # History is shape (steps, n_seeds). Find the best seed.
            history = data['history']
            best_seed_idx = np.argmin(history[-1, :])
            best_history = history[:, best_seed_idx]
            
            ax1.plot(best_history, label=f'{n} Actuators')
            
            # Save the final best loss for the Pareto plot
            best_losses.append(best_history[-1])
        except FileNotFoundError:
            print(f"Warning: multistart_actuators_{n}.npz not found.")
            best_losses.append(np.nan)

    ax1.set_yscale('log')
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Loss (Log Scale)')
    ax1.set_title('Best Convergence per Count')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.5)

    # --- Plot 2: Pareto Frontier ---
    valid_counts = [n for n, l in zip(actuator_counts, best_losses) if not np.isnan(l)]
    valid_losses = [l for l in best_losses if not np.isnan(l)]

    ax2.plot(valid_counts, valid_losses, marker='o', linestyle='-', color='indigo', markersize=8, linewidth=2)
    ax2.set_xlabel('Number of Actuators')
    ax2.set_ylabel('Best Achieved Loss')
    ax2.set_title('Performance vs. Hardware Cost')
    ax2.set_xticks(valid_counts)
    ax2.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig_actuator_sweep.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("fig_actuator_sweep.png", dpi=300, bbox_inches='tight')
    print("Saved fig_actuator_sweep.png and .pdf")
    plt.show()

if __name__ == "__main__":
    plot_actuator_sweep()