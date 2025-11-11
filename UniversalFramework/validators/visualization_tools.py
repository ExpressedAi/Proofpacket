"""
Visualization Tools for Phase-Locking Analysis
==============================================

Tools to visualize χ (criticality), hazard scores, phase coherence,
and other key metrics over time.

Requires: matplotlib (optional - gracefully degrades if unavailable)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available - visualization disabled")


# =============================================================================
# CHI (CRITICALITY) VISUALIZATION
# =============================================================================

def plot_chi_over_time(chi_history: List[float],
                       title: str = "Phase-Lock Criticality Over Time",
                       save_path: Optional[str] = None):
    """
    Plot χ(t) with critical threshold at χ=1

    Args:
        chi_history: List of χ values over time
        title: Plot title
        save_path: If provided, save figure to this path
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.arange(len(chi_history))
    chi = np.array(chi_history)

    # Plot χ(t)
    ax.plot(t, chi, 'b-', linewidth=2, label='χ(t)')

    # Critical line
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Critical (χ=1)')

    # Shade regions
    ax.fill_between(t, 0, 1, alpha=0.1, color='green', label='Subcritical (stable)')
    ax.fill_between(t, 1, max(chi.max(), 1.5), alpha=0.1, color='red', label='Supercritical (unstable)')

    ax.set_xlabel('Time / Tick', fontsize=12)
    ax.set_ylabel('χ = flux / dissipation', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Statistics
    mean_chi = np.mean(chi)
    max_chi = np.max(chi)
    time_supercritical = np.sum(chi > 1) / len(chi) * 100

    stats_text = f"Mean χ: {mean_chi:.3f}\nMax χ: {max_chi:.3f}\nSupercritical: {time_supercritical:.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_chi_phase_space(chi_history: List[float],
                         energy_history: List[float],
                         save_path: Optional[str] = None):
    """
    Plot (χ, E) phase space trajectory

    Shows how system moves through stability regions.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    chi = np.array(chi_history)
    E = np.array(energy_history)

    # Trajectory
    ax.plot(chi, E, 'b-', alpha=0.6, linewidth=1)
    ax.scatter(chi, E, c=np.arange(len(chi)), cmap='viridis', s=30, zorder=5)

    # Start and end points
    ax.scatter(chi[0], E[0], c='green', s=200, marker='o', edgecolors='black',
              linewidths=2, label='Start', zorder=10)
    ax.scatter(chi[-1], E[-1], c='red', s=200, marker='s', edgecolors='black',
              linewidths=2, label='End', zorder=10)

    # Critical line
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='χ=1 (critical)')

    # Regions
    ax.axvspan(0, 1, alpha=0.1, color='green', label='Stable')
    ax.axvspan(1, max(chi.max(), 1.5), alpha=0.1, color='red', label='Unstable')

    ax.set_xlabel('χ (criticality)', fontsize=12)
    ax.set_ylabel('E (energy)', fontsize=12)
    ax.set_title('Phase Space: (χ, E) Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# =============================================================================
# HAZARD FUNCTION VISUALIZATION
# =============================================================================

def plot_hazard_components(epsilon: List[float],
                           g_phi: List[float],
                           brittleness: List[float],
                           alignment: List[float],
                           prior: List[float],
                           save_path: Optional[str] = None):
    """
    Plot all 5 components of hazard function

    h = κ · ε · g(e_φ) · (1-ζ/ζ*) · u · p
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    t = np.arange(len(epsilon))

    # Component 1: ε (capture window)
    axes[0].plot(t, epsilon, 'b-', linewidth=2)
    axes[0].set_ylabel('ε (capture)', fontsize=11)
    axes[0].set_title('Hazard Function Components', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Component 2: g (phase coherence)
    axes[1].plot(t, g_phi, 'g-', linewidth=2)
    axes[1].set_ylabel('g(e_φ) (coherence)', fontsize=11)
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3)

    # Component 3: 1-ζ/ζ* (brittleness margin)
    brittleness_term = [1 - z/0.9 for z in brittleness]  # Assuming ζ*=0.9
    axes[2].plot(t, brittleness_term, 'r-', linewidth=2)
    axes[2].set_ylabel('(1-ζ/ζ*) (margin)', fontsize=11)
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].axhline(y=0, color='r', linestyle='--', linewidth=1, label='Frozen (ζ→ζ*)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')

    # Component 4: u (alignment)
    axes[3].plot(t, alignment, 'purple', linewidth=2)
    axes[3].set_ylabel('u (alignment)', fontsize=11)
    axes[3].set_ylim([0, 1.1])
    axes[3].grid(True, alpha=0.3)

    # Component 5: p (prior)
    axes[4].plot(t, prior, 'orange', linewidth=2)
    axes[4].set_ylabel('p (prior)', fontsize=11)
    axes[4].set_ylim([0, 1.1])
    axes[4].set_xlabel('Time / Tick', fontsize=12)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_hazard_evolution(hazard_history: List[float],
                          threshold: float = 0.5,
                          commit_times: Optional[List[int]] = None,
                          save_path: Optional[str] = None):
    """
    Plot hazard h(t) with commit threshold

    Shows when system commits (h > h*)
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    t = np.arange(len(hazard_history))
    h = np.array(hazard_history)

    # Plot hazard
    ax.plot(t, h, 'b-', linewidth=2, label='Hazard h(t)')

    # Threshold
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold h*={threshold}')

    # Commit times
    if commit_times:
        for ct in commit_times:
            ax.axvline(x=ct, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax.scatter([ct], [h[ct]], c='green', s=200, marker='*', edgecolors='black',
                      linewidths=2, zorder=10, label='Commit' if ct == commit_times[0] else '')

    # Shade regions
    ax.fill_between(t, 0, threshold, alpha=0.1, color='orange', label='Deliberation (h<h*)')
    ax.fill_between(t, threshold, max(h.max(), threshold*1.2), alpha=0.1, color='green',
                    label='Commit zone (h>h*)')

    ax.set_xlabel('Time / Tick', fontsize=12)
    ax.set_ylabel('Hazard h', fontsize=12)
    ax.set_title('Hazard Evolution and Commit Events', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# =============================================================================
# PHASE COHERENCE VISUALIZATION
# =============================================================================

def plot_phase_coherence_circular(phases: List[float],
                                 title: str = "Phase Distribution",
                                 save_path: Optional[str] = None):
    """
    Circular plot of phase distribution

    Shows phase-locking visually.
    """
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')

    phases_array = np.array(phases)

    # Plot individual phases
    ax.scatter(phases_array, np.ones_like(phases_array), c='blue', s=100, alpha=0.6)

    # Mean resultant vector
    z = np.mean(np.exp(1j * phases_array))
    R = np.abs(z)
    mean_phase = np.angle(z)

    ax.plot([0, mean_phase], [0, R], 'r-', linewidth=3, label=f'Mean R={R:.3f}')
    ax.scatter([mean_phase], [R], c='red', s=200, marker='*', edgecolors='black',
              linewidths=2, zorder=10)

    ax.set_ylim([0, 1.2])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_phase_locking_strength(phases_over_time: List[List[float]],
                                save_path: Optional[str] = None):
    """
    Plot R(t) showing phase-locking strength over time

    R → 1 means strong phase-locking
    R → 0 means no coherence
    """
    if not HAS_MATPLOTLIB:
        return

    R_history = []
    for phases in phases_over_time:
        z = np.mean(np.exp(1j * np.array(phases)))
        R = np.abs(z)
        R_history.append(R)

    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.arange(len(R_history))
    R = np.array(R_history)

    ax.plot(t, R, 'b-', linewidth=2, label='Phase coherence R(t)')

    # Thresholds
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1, label='Strong locking (R>0.8)')
    ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, label='Weak locking (R<0.3)')

    # Regions
    ax.fill_between(t, 0.8, 1, alpha=0.1, color='green', label='Locked')
    ax.fill_between(t, 0, 0.3, alpha=0.1, color='red', label='Unlocked')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('R (mean resultant length)', fontsize=12)
    ax.set_title('Phase-Locking Strength Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# =============================================================================
# COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_vbc_dashboard(chi_history: List[float],
                       hazard_history: List[float],
                       epsilon_history: List[float],
                       coherence_history: List[float],
                       threshold: float = 0.5,
                       commit_times: Optional[List[int]] = None,
                       save_path: Optional[str] = None):
    """
    Complete VBC monitoring dashboard

    Shows χ, h, ε, g all in one view
    """
    if not HAS_MATPLOTLIB:
        return

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    t = np.arange(len(chi_history))

    # 1. Chi (criticality)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, chi_history, 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2)
    ax1.fill_between(t, 0, 1, alpha=0.1, color='green')
    ax1.fill_between(t, 1, max(max(chi_history), 1.5), alpha=0.1, color='red')
    ax1.set_ylabel('χ (criticality)', fontsize=11)
    ax1.set_title('A. Phase-Lock Criticality', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Hazard
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, hazard_history, 'g-', linewidth=2)
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    if commit_times:
        for ct in commit_times:
            ax2.axvline(x=ct, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    ax2.fill_between(t, 0, threshold, alpha=0.1, color='orange')
    ax2.set_ylabel('h (hazard)', fontsize=11)
    ax2.set_title('B. Hazard Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Epsilon (capture window)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, epsilon_history, 'purple', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('ε (capture)', fontsize=11)
    ax3.set_title('C. Capture Window', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Phase coherence
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, coherence_history, 'orange', linewidth=2)
    ax4.axhline(y=0.8, color='green', linestyle='--', linewidth=1)
    ax4.set_ylabel('g(e_φ) (coherence)', fontsize=11)
    ax4.set_title('D. Phase Coherence', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3)

    # 5. Chi vs Hazard scatter
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(chi_history, hazard_history, c=t, cmap='viridis', s=30, alpha=0.6)
    ax5.axvline(x=1.0, color='r', linestyle='--', linewidth=2)
    ax5.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('χ (criticality)', fontsize=11)
    ax5.set_ylabel('h (hazard)', fontsize=11)
    ax5.set_title('E. Phase Space: (χ, h)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Time')

    # 6. Statistics panel
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    stats = f"""
    VBC STATISTICS
    {'='*30}

    Criticality (χ):
      Mean: {np.mean(chi_history):.3f}
      Max:  {np.max(chi_history):.3f}
      >1:   {np.sum(np.array(chi_history) > 1)/len(chi_history)*100:.1f}%

    Hazard (h):
      Mean: {np.mean(hazard_history):.3f}
      Max:  {np.max(hazard_history):.3f}
      >h*:  {np.sum(np.array(hazard_history) > threshold)/len(hazard_history)*100:.1f}%

    Commits: {len(commit_times) if commit_times else 0}

    Phase Coherence:
      Mean R: {np.mean(coherence_history):.3f}
      Max R:  {np.max(coherence_history):.3f}

    Capture Window:
      Mean ε: {np.mean(epsilon_history):.3f}
    """

    ax6.text(0.1, 0.9, stats, fontsize=11, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('VBC Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# =============================================================================
# EXPORT DATA FOR EXTERNAL PLOTTING
# =============================================================================

def export_to_csv(data_dict: Dict[str, List[float]], filename: str):
    """
    Export time series data to CSV for external analysis

    Args:
        data_dict: Dictionary of name -> values
        filename: Output CSV path
    """
    import csv

    # Get length from first array
    length = len(next(iter(data_dict.values())))

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['time'] + list(data_dict.keys()))

        # Data rows
        for i in range(length):
            row = [i] + [data_dict[key][i] for key in data_dict.keys()]
            writer.writerow(row)

    print(f"Exported {length} rows to {filename}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - install with: pip install matplotlib")
        exit(1)

    print("Generating example visualizations...")

    # Generate synthetic data
    np.random.seed(42)
    n_steps = 100

    # Chi increasing toward critical point
    chi = 0.5 + 0.6 * (np.arange(n_steps) / n_steps) + 0.1 * np.random.randn(n_steps)
    chi = np.clip(chi, 0.3, 1.5)

    # Energy decaying
    energy = np.exp(-0.01 * np.arange(n_steps)) + 0.1 * np.random.randn(n_steps)

    # Hazard building up
    hazard = 0.2 + 0.4 * (np.arange(n_steps) / n_steps) + 0.1 * np.random.randn(n_steps)
    hazard = np.clip(hazard, 0, 1)

    # Epsilon varying
    epsilon = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(n_steps) / 20) + 0.05 * np.random.randn(n_steps)
    epsilon = np.clip(epsilon, 0, 1)

    # Coherence increasing
    coherence = 0.3 + 0.5 * (np.arange(n_steps) / n_steps) + 0.1 * np.random.randn(n_steps)
    coherence = np.clip(coherence, 0, 1)

    # Commits happen when hazard > 0.5
    commit_times = [i for i in range(n_steps) if hazard[i] > 0.5][:3]  # First 3

    print("\n1. Plotting χ over time...")
    plot_chi_over_time(chi.tolist(), save_path="chi_over_time.png")

    print("2. Plotting (χ, E) phase space...")
    plot_chi_phase_space(chi.tolist(), energy.tolist(), save_path="chi_phase_space.png")

    print("3. Plotting hazard evolution...")
    plot_hazard_evolution(hazard.tolist(), threshold=0.5, commit_times=commit_times,
                         save_path="hazard_evolution.png")

    print("4. Plotting phase coherence...")
    phases = np.random.uniform(-np.pi, np.pi, 20)
    plot_phase_coherence_circular(phases.tolist(), save_path="phase_coherence.png")

    print("5. Generating VBC dashboard...")
    plot_vbc_dashboard(chi.tolist(), hazard.tolist(), epsilon.tolist(),
                      coherence.tolist(), threshold=0.5, commit_times=commit_times,
                      save_path="vbc_dashboard.png")

    print("\n6. Exporting data to CSV...")
    export_to_csv({
        'chi': chi.tolist(),
        'hazard': hazard.tolist(),
        'epsilon': epsilon.tolist(),
        'coherence': coherence.tolist()
    }, "vbc_data.csv")

    print("\n✓ All visualizations generated successfully!")
    print("Files created: chi_over_time.png, chi_phase_space.png, hazard_evolution.png,")
    print("               phase_coherence.png, vbc_dashboard.png, vbc_data.csv")
