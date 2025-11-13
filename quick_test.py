"""
Quick test of most promising configurations
"""

from delta_trading_system import DeltaTradingSystem
from consensus_detector import ConsensusDetector
from chi_crash_detector import ChiCrashDetector
from historical_backtest import HistoricalBacktest
import sys
import io

def test_config(R_star, chi_crisis, h_threshold, eps_threshold, name):
    """Test a single configuration."""

    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print(f"  R* = {R_star}, χ_crisis = {chi_crisis}")
    print(f"  h = {h_threshold}, eps = {eps_threshold}")
    print(f"{'='*70}")

    # Create backtest
    backtest = HistoricalBacktest(
        start_date="2000-01-01",
        end_date="2024-12-31",
        initial_capital=100000,
        universe_size=50
    )

    # Override parameters
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

    # Suppress verbose output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        result = backtest.run_backtest()
    finally:
        sys.stdout = old_stdout

    # Print results
    print(f"\nRESULTS:")
    print(f"  Total Return:    {result.total_return:7.2f}%")
    print(f"  CAGR:            {result.cagr:7.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:7.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown:7.2f}%")
    print(f"  Win Rate:        {result.win_rate:7.2f}%")
    print(f"  Profit Factor:   {result.profit_factor:7.2f}")
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Final Value:     ${result.equity_curve[-1]:,.2f}")

    return result


if __name__ == "__main__":
    print("="*70)
    print("QUICK PARAMETER TEST: Most Promising Configurations")
    print("="*70)

    print("\nCurrent Best: R*=1.5, χ=2.0 → 11.2% CAGR, 0.64 Sharpe")
    print("Testing variations to find improvement...\n")

    results = []

    # Test 1: Slightly more aggressive
    r1 = test_config(
        R_star=1.25,
        chi_crisis=2.0,
        h_threshold=0.175,  # 0.15 + (1.25-1.0)*0.05
        eps_threshold=0.035,  # 0.03 + (1.25-1.0)*0.02
        name="Slightly More Aggressive"
    )
    results.append(("R*=1.25, χ=2.0", r1))

    # Test 2: Very aggressive
    r2 = test_config(
        R_star=1.0,
        chi_crisis=2.5,
        h_threshold=0.15,
        eps_threshold=0.03,
        name="Very Aggressive"
    )
    results.append(("R*=1.0, χ=2.5", r2))

    # Test 3: Current best (for comparison)
    r3 = test_config(
        R_star=1.5,
        chi_crisis=2.0,
        h_threshold=0.20,
        eps_threshold=0.05,
        name="Current Best (Baseline)"
    )
    results.append(("R*=1.5, χ=2.0 (current)", r3))

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Configuration':<25} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8} {'Final$':>12}")
    print("-"*70)

    for name, r in results:
        print(f"{name:<25} {r.cagr:>8.2f} {r.sharpe_ratio:>8.2f} {r.max_drawdown:>8.2f} {r.total_trades:>8} ${r.equity_curve[-1]:>11,.0f}")

    # Find best
    best_sharpe = max(results, key=lambda x: x[1].sharpe_ratio)
    best_cagr = max(results, key=lambda x: x[1].cagr)

    print("\n" + "="*70)
    print("WINNER")
    print("="*70)

    if best_sharpe == best_cagr:
        print(f"\n🏆 BEST OVERALL: {best_sharpe[0]}")
        print(f"   CAGR: {best_sharpe[1].cagr:.2f}%")
        print(f"   Sharpe: {best_sharpe[1].sharpe_ratio:.2f}")
        print(f"   Max DD: {best_sharpe[1].max_drawdown:.2f}%")
        print(f"   $100K → ${best_sharpe[1].equity_curve[-1]:,.0f}")
    else:
        print(f"\n🏆 BEST SHARPE: {best_sharpe[0]} (Sharpe={best_sharpe[1].sharpe_ratio:.2f})")
        print(f"🚀 BEST CAGR: {best_cagr[0]} (CAGR={best_cagr[1].cagr:.2f}%)")

    print("\n" + "="*70)
