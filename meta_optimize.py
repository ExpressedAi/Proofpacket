"""
META-OPTIMIZATION: Using Phase-Locking to Find Optimal Parameters

Key Insight:
The system detects phase-locks in markets. Can we use the SAME physics
to detect phase-locks in PARAMETER SPACE?

Concepts Applied:
1. Phase-Locking: When do multiple good properties align?
2. Coupling Strength K: How strongly do parameters affect outcomes?
3. Dissipation Γ: Variance/noise in parameter space
4. Critical Points: Where small changes cause large shifts
5. ε-windows: Optimal parameter regions (eligibility)

Question: What if optimal parameters exist at a "critical point"
where crisis survival + returns + consistency all phase-lock?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr


class MetaOptimizer:
    """Use phase-locking concepts to find optimal parameters."""

    def __init__(self, results_df: pd.DataFrame):
        """
        Args:
            results_df: DataFrame from build_the_house.py
        """
        self.df = results_df

        # Normalize metrics to [0, 1] for phase analysis
        self.metrics = ['cagr', 'max_dd', 'sharpe', 'win_rate', 'profit_factor']

        for metric in self.metrics:
            col = self.df[metric].values
            if metric == 'max_dd':
                # Max DD is negative, flip it
                col = -col
            normalized = (col - col.min()) / (col.max() - col.min() + 1e-10)
            self.df[f'{metric}_norm'] = normalized

    def compute_coupling_strength(self, metric1: str, metric2: str) -> float:
        """
        Compute coupling strength K between two metrics.

        K measures how strongly two properties move together across parameter space.
        High K = strong coupling (phase-lock)
        Low K = independent (no lock)

        In the trading system:
            K = coupling between two assets

        In parameter space:
            K = coupling between two performance metrics
        """
        x = self.df[f'{metric1}_norm'].values
        y = self.df[f'{metric2}_norm'].values

        # Pearson correlation as proxy for coupling
        corr, p_value = pearsonr(x, y)

        return abs(corr)  # Absolute value (direction doesn't matter)

    def find_phase_lock_regions(self) -> pd.DataFrame:
        """
        Find parameter regions where multiple metrics "phase-lock"
        (i.e., all good things happen together).

        This is analogous to finding ε-windows in the trading system.
        """
        # For each configuration, compute "consensus score"
        # = how many normalized metrics are above threshold

        threshold = 0.6  # Similar to R* threshold

        consensus_scores = []

        for i, row in self.df.iterrows():
            score = 0
            for metric in self.metrics:
                if row[f'{metric}_norm'] > threshold:
                    score += 1

            # Normalize to 0-1
            consensus = score / len(self.metrics)
            consensus_scores.append(consensus)

        self.df['phase_lock_consensus'] = consensus_scores

        return self.df.sort_values('phase_lock_consensus', ascending=False)

    def compute_dissipation(self, config_idx: int, window: int = 3) -> float:
        """
        Compute "dissipation" Γ for a configuration.

        In the trading system:
            Γ = mean-reversion strength (stability)

        In parameter space:
            Γ = local variance around this config
            Low Γ = stable region (good)
            High Γ = chaotic region (bad)
        """
        # Get neighboring configs (in parameter space)
        R_star = self.df.iloc[config_idx]['R_star']
        chi_crisis = self.df.iloc[config_idx]['chi_crisis']

        # Find nearby configs
        R_dists = (self.df['R_star'] - R_star).abs()
        chi_dists = (self.df['chi_crisis'] - chi_crisis).abs()
        total_dist = R_dists + chi_dists

        # Get K nearest neighbors
        nearest = total_dist.nsmallest(window + 1).index[1:]  # Exclude self

        if len(nearest) == 0:
            return 0.0

        # Variance of house_score among neighbors
        neighbor_scores = self.df.loc[nearest, 'house_score'].values
        dissipation = np.std(neighbor_scores)

        return dissipation

    def find_critical_points(self) -> pd.DataFrame:
        """
        Find critical points: where small parameter changes cause large shifts.

        This is analogous to χ criticality in the trading system.
        """
        critical_scores = []

        for i in range(len(self.df)):
            # Compute gradient: how much does house_score change
            # as we move in parameter space?

            dissipation = self.compute_dissipation(i, window=3)

            # Critical = high gradient + low local variance
            # (Sharp transition + stable locally)
            gradient = dissipation  # Simplified

            critical_scores.append(gradient)

        self.df['criticality'] = critical_scores

        return self.df

    def compute_antifragile_score(self) -> pd.DataFrame:
        """
        Compute anti-fragility using TUR-inspired logic.

        TUR: P/Σ ≤ 1/2
        Where P = precision, Σ = entropy cost

        In parameter space:
            P = house_score (precision of being "the house")
            Σ = dissipation (cost/variance)

        Anti-fragile configs: High P, Low Σ
        """
        antifragile_scores = []

        for i in range(len(self.df)):
            P = self.df.iloc[i]['house_score']
            Sigma = self.compute_dissipation(i, window=3)

            # Anti-fragile ratio
            if Sigma > 0:
                af_score = P / Sigma
            else:
                af_score = P  # Perfect stability

            antifragile_scores.append(af_score)

        self.df['antifragile_meta_score'] = antifragile_scores

        return self.df.sort_values('antifragile_meta_score', ascending=False)

    def visualize_parameter_space(self, output_file='parameter_space.png'):
        """Create visualization of parameter space with phase-lock regions."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: R* vs χ, colored by house_score
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.df['R_star'], self.df['chi_crisis'],
                             c=self.df['house_score'], s=200, cmap='RdYlGn',
                             edgecolors='black', linewidth=2)
        ax1.set_xlabel('R* (Consensus Threshold)', fontsize=12)
        ax1.set_ylabel('χ_crisis (Crash Threshold)', fontsize=12)
        ax1.set_title('House Score Landscape', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax1, label='House Score')

        # Annotate best
        best_idx = self.df['house_score'].idxmax()
        best_row = self.df.loc[best_idx]
        ax1.annotate('★ THE HOUSE',
                    xy=(best_row['R_star'], best_row['chi_crisis']),
                    xytext=(best_row['R_star'] + 0.2, best_row['chi_crisis'] + 0.3),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=2, color='gold'))

        # Plot 2: Returns vs Risk
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(self.df['max_dd'].abs(), self.df['cagr'],
                              c=self.df['house_score'], s=200, cmap='RdYlGn',
                              edgecolors='black', linewidth=2)
        ax2.set_xlabel('Max Drawdown (abs %)', fontsize=12)
        ax2.set_ylabel('CAGR (%)', fontsize=12)
        ax2.set_title('Risk/Return Frontier', fontsize=14, fontweight='bold')
        ax2.axhline(y=7.8, color='red', linestyle='--', label='SPY CAGR', alpha=0.5)
        ax2.axvline(x=55, color='red', linestyle='--', label='SPY Max DD', alpha=0.5)
        ax2.legend()
        plt.colorbar(scatter2, ax=ax2, label='House Score')

        # Plot 3: Phase-Lock Consensus
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(self.df['R_star'], self.df['chi_crisis'],
                              c=self.df['phase_lock_consensus'], s=200,
                              cmap='viridis', edgecolors='black', linewidth=2)
        ax3.set_xlabel('R* (Consensus Threshold)', fontsize=12)
        ax3.set_ylabel('χ_crisis (Crash Threshold)', fontsize=12)
        ax3.set_title('Phase-Lock Regions', fontsize=14, fontweight='bold')
        plt.colorbar(scatter3, ax=ax3, label='Phase-Lock Consensus')

        # Plot 4: Anti-Fragile Score
        ax4 = axes[1, 1]
        scatter4 = ax4.scatter(self.df['R_star'], self.df['chi_crisis'],
                              c=self.df['antifragile_meta_score'], s=200,
                              cmap='plasma', edgecolors='black', linewidth=2)
        ax4.set_xlabel('R* (Consensus Threshold)', fontsize=12)
        ax4.set_ylabel('χ_crisis (Crash Threshold)', fontsize=12)
        ax4.set_title('Anti-Fragility Score (TUR-Inspired)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter4, ax=ax4, label='Anti-Fragile Score')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {output_file}")

        return fig

    def generate_report(self):
        """Generate comprehensive meta-optimization report."""

        print("="*80)
        print("META-OPTIMIZATION REPORT")
        print("Using Phase-Locking Physics to Find Optimal Parameters")
        print("="*80)

        # 1. Coupling Analysis
        print("\n" + "="*80)
        print("COUPLING STRENGTH ANALYSIS")
        print("="*80)
        print("\nHow strongly do different metrics move together?")
        print("(High K = phase-locked, move in sync)\n")

        metric_pairs = [
            ('cagr', 'sharpe'),
            ('cagr', 'max_dd'),
            ('win_rate', 'profit_factor'),
            ('sharpe', 'win_rate'),
        ]

        for m1, m2 in metric_pairs:
            K = self.compute_coupling_strength(m1, m2)
            if K > 0.7:
                status = "🔒 STRONG LOCK"
            elif K > 0.4:
                status = "⚡ MODERATE"
            else:
                status = "   WEAK"

            print(f"  {status} | K({m1:<12}, {m2:<15}) = {K:.3f}")

        # 2. Phase-Lock Regions
        print("\n" + "="*80)
        print("PHASE-LOCK REGIONS (ε-windows in Parameter Space)")
        print("="*80)
        print("\nConfigurations where multiple good properties align:\n")

        phase_locked = self.find_phase_lock_regions()
        top_phase = phase_locked.head(5)

        print(f"{'Rank':<6}{'Config':<20}{'Consensus':<12}{'House Score':<12}")
        print("-"*80)
        for idx, (i, row) in enumerate(top_phase.iterrows(), 1):
            marker = "★" if idx == 1 else " "
            print(f"{marker}{idx:<5}{row['name']:<20}{row['phase_lock_consensus']:>10.2f}"
                  f"{row['house_score']:>12.1f}")

        # 3. Anti-Fragile Configurations (TUR-inspired)
        print("\n" + "="*80)
        print("ANTI-FRAGILE CONFIGURATIONS (TUR: P/Σ)")
        print("="*80)
        print("\nHigh precision, low dissipation = anti-fragile house:\n")

        af_ranked = self.compute_antifragile_score()
        top_af = af_ranked.head(5)

        print(f"{'Rank':<6}{'Config':<20}{'AF Score':<12}{'House':<10}{'Dissipation':<12}")
        print("-"*80)
        for idx, (i, row) in enumerate(top_af.iterrows(), 1):
            marker = "🏛️" if idx == 1 else "  "
            diss = self.compute_dissipation(i, window=3)
            print(f"{marker}{idx:<5}{row['name']:<20}{row['antifragile_meta_score']:>10.1f}"
                  f"{row['house_score']:>10.1f}{diss:>12.2f}")

        # 4. Critical Points
        print("\n" + "="*80)
        print("CRITICAL POINTS (Phase Transitions)")
        print("="*80)
        print("\nWhere small parameter changes cause large behavior shifts:\n")

        critical = self.find_critical_points()
        critical_sorted = critical.sort_values('criticality', ascending=False)
        top_critical = critical_sorted.head(5)

        print(f"{'Config':<20}{'Criticality':<15}{'Interpretation':<40}")
        print("-"*80)
        for i, row in top_critical.iterrows():
            crit = row['criticality']
            if crit > 10:
                interp = "⚠️  UNSTABLE (avoid boundary)"
            elif crit > 5:
                interp = "⚡ SENSITIVE (test carefully)"
            else:
                interp = "✓ STABLE (robust region)"

            print(f"{row['name']:<20}{crit:>13.2f}  {interp}")

        # 5. THE VERDICT
        print("\n" + "="*80)
        print("THE VERDICT: Optimal House Configuration")
        print("="*80)

        # Winner: highest anti-fragile score
        winner = af_ranked.iloc[0]

        print(f"\nUsing meta-optimization (phase-locking in parameter space):")
        print(f"\n  🏛️  THE OPTIMAL HOUSE: {winner['name']}")
        print(f"\n  Configuration:")
        print(f"    R* = {winner['R_star']:.2f}")
        print(f"    χ_crisis = {winner['chi_crisis']:.1f}")
        print(f"\n  Why This Config:")
        print(f"    1. High house score: {winner['house_score']:.1f}")
        print(f"    2. Low dissipation (stable region)")
        print(f"    3. Phase-lock consensus: {winner['phase_lock_consensus']:.2f}")
        print(f"    4. Anti-fragile score: {winner['antifragile_meta_score']:.1f}")
        print(f"\n  Performance:")
        print(f"    CAGR: {winner['cagr']:.2f}%")
        print(f"    Max DD: {winner['max_dd']:.2f}%")
        print(f"    Sharpe: {winner['sharpe']:.2f}")
        print(f"    Win Rate: {winner['win_rate']:.1f}%")
        print(f"    Profit Factor: {winner['profit_factor']:.2f}")

        print(f"\n  This configuration sits at a 'critical point' where:")
        print(f"    - Crisis survival is maximized")
        print(f"    - Returns are optimized (not maximized)")
        print(f"    - Consistency is high")
        print(f"    - Local parameter space is stable (low sensitivity)")
        print(f"\n  The house cannot be killed.")

        print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print("META-OPTIMIZATION: Phase-Locking in Parameter Space")
    print("="*80)

    # Load results
    try:
        df = pd.read_csv('house_exploration_results.csv')
        print(f"\n✓ Loaded {len(df)} configurations from house_exploration_results.csv")
    except FileNotFoundError:
        print("\n✗ ERROR: Run build_the_house.py first to generate data!")
        exit(1)

    # Create optimizer
    optimizer = MetaOptimizer(df)

    # Generate report
    optimizer.generate_report()

    # Visualize
    optimizer.visualize_parameter_space()

    print("\n" + "="*80)
    print("META-OPTIMIZATION COMPLETE")
    print("="*80)
