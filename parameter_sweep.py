"""
Parameter Sweep: Find Optimal Thresholds for Δ-Trading System

Tests combinations of:
- R* (consensus threshold)
- χ_crisis (crash threshold)

Goal: Maximize Sharpe ratio while maintaining drawdown < 10%
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import itertools

from delta_trading_system import DeltaTradingSystem
from tur_optimizer import TradeFrequency
from historical_backtest import HistoricalBacktest


class ParameterSweep:
    """Run systematic parameter sweep across key thresholds."""

    def __init__(self):
        """Initialize parameter sweep."""
        self.results: List[Dict] = []

    def run_single_backtest(
        self,
        R_star: float,
        chi_crisis: float,
        h_threshold: float,
        eps_threshold: float,
        verbose: bool = False
    ) -> Dict:
        """
        Run backtest with specific parameters.

        Returns:
            Dict with performance metrics
        """
        if verbose:
            print(f"\nTesting: R*={R_star}, χ_crisis={chi_crisis}, h={h_threshold}, eps={eps_threshold}")

        # Create custom backtest with modified system
        backtest = HistoricalBacktest(
            start_date="2000-01-01",
            end_date="2024-12-31",
            initial_capital=100000,
            universe_size=50
        )

        # Override system parameters
        from consensus_detector import ConsensusDetector
        from chi_crash_detector import ChiCrashDetector

        backtest.system.layer1_consensus = ConsensusDetector(
            R_star=R_star,
            h_threshold=h_threshold,
            eps_threshold=eps_threshold
        )

        backtest.system.layer2_chi = ChiCrashDetector(
            flux_window=5,
            dissipation_window=20,
            regime_lag=3,
            crisis_threshold=chi_crisis
        )

        # Run backtest (suppress output)
        import sys
        import io

        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            result = backtest.run_backtest()
        finally:
            if not verbose:
                sys.stdout = old_stdout

        # Return key metrics
        return {
            'R_star': R_star,
            'chi_crisis': chi_crisis,
            'h_threshold': h_threshold,
            'eps_threshold': eps_threshold,
            'total_return': result.total_return,
            'cagr': result.cagr,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'consensus_signals': result.consensus_signals,
            'final_value': result.equity_curve[-1]
        }

    def sweep_grid(self, verbose: bool = True):
        """
        Run grid search over parameter space.

        Focus on key parameters:
        - R_star: [1.0, 1.25, 1.5, 1.75, 2.0]
        - chi_crisis: [1.5, 2.0, 2.5, 3.0]
        """
        print("=" * 70)
        print("PARAMETER SWEEP: GRID SEARCH")
        print("=" * 70)

        # Define parameter grid
        R_star_values = [1.0, 1.25, 1.5, 1.75, 2.0]
        chi_crisis_values = [1.5, 2.0, 2.5, 3.0]

        # h and eps scale with R_star
        # Lower R_star → lower thresholds needed

        total_runs = len(R_star_values) * len(chi_crisis_values)
        current_run = 0

        print(f"\nTotal combinations to test: {total_runs}")
        print(f"Estimated time: {total_runs * 0.5:.0f} seconds (~{total_runs * 0.5 / 60:.1f} minutes)\n")

        for R_star in R_star_values:
            for chi_crisis in chi_crisis_values:
                current_run += 1

                # Scale h and eps with R_star
                # Lower R_star → need lower individual signal thresholds
                h_threshold = 0.15 + (R_star - 1.0) * 0.05  # 0.15-0.20
                eps_threshold = 0.03 + (R_star - 1.0) * 0.02  # 0.03-0.05

                print(f"[{current_run}/{total_runs}] R*={R_star:.2f}, χ={chi_crisis:.1f}, h={h_threshold:.2f}, ε={eps_threshold:.2f}...", end=" ")

                result = self.run_single_backtest(
                    R_star=R_star,
                    chi_crisis=chi_crisis,
                    h_threshold=h_threshold,
                    eps_threshold=eps_threshold,
                    verbose=False
                )

                self.results.append(result)

                # Print quick summary
                print(f"CAGR={result['cagr']:.1f}%, Sharpe={result['sharpe']:.2f}, DD={result['max_dd']:.1f}%")

        print("\n" + "=" * 70)
        print("SWEEP COMPLETE")
        print("=" * 70)

    def analyze_results(self):
        """Analyze sweep results and find optimal parameters."""

        if not self.results:
            print("No results to analyze!")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS BY SHARPE RATIO")
        print("=" * 70)

        top_sharpe = df.nlargest(10, 'sharpe')

        print("\n{:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
            "Rank", "R*", "χ_cris", "CAGR%", "Sharpe", "MaxDD%", "Trades"
        ))
        print("-" * 70)

        for idx, (i, row) in enumerate(top_sharpe.iterrows(), 1):
            print("{:<6} {:<8.2f} {:<8.1f} {:<8.1f} {:<8.2f} {:<8.1f} {:<8}".format(
                idx,
                row['R_star'],
                row['chi_crisis'],
                row['cagr'],
                row['sharpe'],
                row['max_dd'],
                row['total_trades']
            ))

        print("\n" + "=" * 70)
        print("TOP 10 CONFIGURATIONS BY CAGR")
        print("=" * 70)

        top_cagr = df.nlargest(10, 'cagr')

        print("\n{:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
            "Rank", "R*", "χ_cris", "CAGR%", "Sharpe", "MaxDD%", "Trades"
        ))
        print("-" * 70)

        for idx, (i, row) in enumerate(top_cagr.iterrows(), 1):
            print("{:<6} {:<8.2f} {:<8.1f} {:<8.1f} {:<8.2f} {:<8.1f} {:<8}".format(
                idx,
                row['R_star'],
                row['chi_crisis'],
                row['cagr'],
                row['sharpe'],
                row['max_dd'],
                row['total_trades']
            ))

        print("\n" + "=" * 70)
        print("BEST RISK-ADJUSTED (Sharpe > 0.6, DD < 10%)")
        print("=" * 70)

        # Filter for good risk-adjusted performance
        filtered = df[(df['sharpe'] > 0.6) & (df['max_dd'] > -10)]

        if len(filtered) > 0:
            best = filtered.nlargest(5, 'sharpe')

            print("\n{:<6} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
                "Rank", "R*", "χ_cris", "CAGR%", "Sharpe", "MaxDD%", "Trades"
            ))
            print("-" * 70)

            for idx, (i, row) in enumerate(best.iterrows(), 1):
                print("{:<6} {:<8.2f} {:<8.1f} {:<8.1f} {:<8.2f} {:<8.1f} {:<8}".format(
                    idx,
                    row['R_star'],
                    row['chi_crisis'],
                    row['cagr'],
                    row['sharpe'],
                    row['max_dd'],
                    row['total_trades']
                ))

            # Recommend best overall
            print("\n" + "=" * 70)
            print("RECOMMENDED CONFIGURATION")
            print("=" * 70)

            best_row = best.iloc[0]

            print(f"""
Configuration:
  R* (consensus threshold):    {best_row['R_star']:.2f}
  χ_crisis (crash threshold):  {best_row['chi_crisis']:.1f}
  h_threshold:                 {best_row['h_threshold']:.3f}
  eps_threshold:               {best_row['eps_threshold']:.3f}

Performance:
  Total Return:                {best_row['total_return']:.2f}%
  CAGR:                        {best_row['cagr']:.2f}%
  Sharpe Ratio:                {best_row['sharpe']:.2f}
  Max Drawdown:                {best_row['max_dd']:.2f}%
  Win Rate:                    {best_row['win_rate']:.2f}%
  Profit Factor:               {best_row['profit_factor']:.2f}
  Total Trades:                {best_row['total_trades']}

Final Portfolio Value:         ${best_row['final_value']:,.2f}
            """)
        else:
            print("\nNo configurations met criteria (Sharpe > 0.6, DD < 10%)")

        # Save results to CSV
        output_file = 'parameter_sweep_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Full results saved to {output_file}")

        return df


if __name__ == "__main__":
    print("=" * 70)
    print("Δ-TRADING SYSTEM: PARAMETER OPTIMIZATION")
    print("=" * 70)

    sweep = ParameterSweep()

    # Run grid search
    sweep.sweep_grid(verbose=True)

    # Analyze results
    results_df = sweep.analyze_results()

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
