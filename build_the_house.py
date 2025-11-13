"""
BUILD THE HOUSE: Comprehensive Parameter Exploration

Testing philosophy:
- Explore WIDE range of configurations (conservative → aggressive)
- Measure what matters: crisis survival, consistency, anti-fragility
- Find the sweet spot: maximum returns WITHOUT compromising the house

North Star: "We are the house, not the gambler"
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import io

from delta_trading_system import DeltaTradingSystem
from consensus_detector import ConsensusDetector
from chi_crash_detector import ChiCrashDetector
from historical_backtest import HistoricalBacktest


def test_configuration(R_star, chi_crisis, name="Config"):
    """Test a single configuration and return house metrics."""

    # Derived thresholds
    h_threshold = 0.10 + (R_star - 0.5) * 0.06
    eps_threshold = 0.02 + (R_star - 0.5) * 0.02

    # Create backtest
    backtest = HistoricalBacktest(
        start_date="2000-01-01",
        end_date="2024-12-31",
        initial_capital=100000,
        universe_size=50
    )

    # Set parameters
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

    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        result = backtest.run_backtest()
    finally:
        sys.stdout = old_stdout

    # Calculate house metrics
    equity = np.array(result.equity_curve)

    # Recovery speed: how fast do we bounce back from drawdowns?
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    underwater_days = np.sum(drawdown < -0.01)  # Days below -1%

    # Consistency: standard deviation of monthly returns
    # (Lower is better - house should be steady)
    if len(result.returns) > 12:
        monthly_chunks = [result.returns[i:i+4] for i in range(0, len(result.returns), 4)]
        monthly_returns = [np.sum(chunk) for chunk in monthly_chunks if len(chunk) == 4]
        consistency_score = 1.0 / (np.std(monthly_returns) + 0.001)  # Higher is better
    else:
        consistency_score = 0.0

    # Anti-fragility score: ratio of upside to downside
    # House should have limited downside, unlimited upside potential
    upside_capture = (equity[-1] - equity[0]) / equity[0]
    downside_risk = abs(result.max_drawdown)
    antifragile_ratio = upside_capture / (downside_risk + 0.01)

    # Ruin probability (simplified): chance of catastrophic loss
    # If max drawdown > 20%, risk of ruin increases exponentially
    if result.max_drawdown < -0.20:
        ruin_risk = 1.0 - (0.8 ** (abs(result.max_drawdown) * 100))
    else:
        ruin_risk = 0.0

    # House score: composite metric
    # Weights:
    # - Crisis survival (50%): max DD, ruin risk
    # - Consistency (25%): win rate, profit factor, std dev
    # - Returns (25%): CAGR, Sharpe

    crisis_survival = (1.0 - min(1.0, abs(result.max_drawdown) / 0.55)) * 50  # vs SPY -55%
    crisis_survival -= ruin_risk * 25  # Penalty for ruin risk

    consistency = (result.win_rate * 15 +
                   min(5.0, result.profit_factor) * 2 +
                   consistency_score * 8)

    returns_score = (min(result.cagr * 100 / 7.8, 2.0) * 15 +  # vs SPY 7.8%
                     min(result.sharpe_ratio / 2.0, 1.0) * 10)  # vs target 2.0

    house_score = crisis_survival + consistency + returns_score

    return {
        'name': name,
        'R_star': R_star,
        'chi_crisis': chi_crisis,
        'h_threshold': h_threshold,
        'eps_threshold': eps_threshold,

        # Returns
        'cagr': result.cagr * 100,
        'total_return': result.total_return * 100,
        'sharpe': result.sharpe_ratio,
        'final_value': result.equity_curve[-1],

        # House metrics
        'max_dd': result.max_drawdown * 100,
        'win_rate': result.win_rate * 100,
        'profit_factor': result.profit_factor,
        'consistency': consistency_score,
        'antifragile_ratio': antifragile_ratio,
        'ruin_risk': ruin_risk * 100,
        'underwater_days': underwater_days,

        # Trading
        'total_trades': result.total_trades,
        'avg_win': result.avg_win,
        'avg_loss': result.avg_loss,

        # Score
        'house_score': house_score
    }


if __name__ == "__main__":
    print("="*80)
    print("BUILD THE HOUSE: Comprehensive Parameter Exploration")
    print("="*80)

    print("\nPhilosophy:")
    print("  We are not trying to maximize returns.")
    print("  We are trying to build a system that CANNOT BE KILLED.")
    print("  The house always wins because the house can't go bankrupt.\n")

    # Test configurations
    configs = [
        # Conservative (Fortress)
        (2.0, 2.5, "Fortress"),
        (2.0, 2.0, "Conservative"),
        (1.75, 2.5, "Cautious"),

        # Current baseline
        (1.5, 2.0, "Baseline"),
        (1.5, 2.5, "Baseline+"),

        # Balanced (Seeking sweet spot)
        (1.25, 2.0, "Balanced-A"),
        (1.25, 2.5, "Balanced-B"),
        (1.0, 2.5, "Balanced-C"),

        # Aggressive (More exposure)
        (1.0, 2.0, "Aggressive-A"),
        (0.75, 2.5, "Aggressive-B"),
        (0.5, 3.0, "Very Aggressive"),

        # Gambler (Maximum exposure - for comparison)
        (0.5, 1.5, "Gambler"),
    ]

    print(f"Testing {len(configs)} configurations...\n")

    results = []
    for i, (R, chi, name) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing {name:<20} (R*={R:.2f}, χ={chi:.1f})... ", end="", flush=True)

        try:
            result = test_configuration(R, chi, name)
            results.append(result)
            print(f"✓ House Score: {result['house_score']:.1f}")
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:50]}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    df.to_csv('house_exploration_results.csv', index=False)
    print(f"\n✓ Saved detailed results to house_exploration_results.csv")

    print("\n" + "="*80)
    print("RANKINGS: BY HOUSE SCORE (What Actually Matters)")
    print("="*80)

    df_sorted = df.sort_values('house_score', ascending=False)

    print(f"\n{'Rank':<6}{'Config':<20}{'House':<8}{'CAGR%':<8}{'MaxDD%':<9}{'Sharpe':<8}{'Win%':<7}{'PF':<6}")
    print("-"*80)

    for idx, (i, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{idx:<6}{row['name']:<20}{row['house_score']:>6.1f}"
              f"{row['cagr']:>8.2f}{row['max_dd']:>9.2f}{row['sharpe']:>8.2f}"
              f"{row['win_rate']:>7.1f}{row['profit_factor']:>6.2f}")

    print("\n" + "="*80)
    print("THE HOUSE: Top 3 Configurations")
    print("="*80)

    top3 = df_sorted.head(3)

    for idx, (i, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"#{idx}: {row['name']}")
        print(f"{'='*80}")
        print(f"  Configuration:")
        print(f"    R* = {row['R_star']:.2f}, χ_crisis = {row['chi_crisis']:.1f}")
        print(f"    h = {row['h_threshold']:.3f}, eps = {row['eps_threshold']:.3f}")
        print(f"\n  Performance:")
        print(f"    CAGR:            {row['cagr']:>7.2f}% (vs SPY 7.8%)")
        print(f"    Total Return:    {row['total_return']:>7.2f}%")
        print(f"    Sharpe Ratio:    {row['sharpe']:>7.2f}")
        print(f"    Final Value:     ${row['final_value']:>11,.0f}")
        print(f"\n  House Metrics:")
        print(f"    House Score:     {row['house_score']:>7.1f} / 100")
        print(f"    Max Drawdown:    {row['max_dd']:>7.2f}% (vs SPY -55%)")
        print(f"    Win Rate:        {row['win_rate']:>7.1f}%")
        print(f"    Profit Factor:   {row['profit_factor']:>7.2f}")
        print(f"    Ruin Risk:       {row['ruin_risk']:>7.2f}%")
        print(f"    Antifragile:     {row['antifragile_ratio']:>7.2f}")
        print(f"\n  Trading:")
        print(f"    Total Trades:    {row['total_trades']:>7.0f}")
        print(f"    Avg Win:         ${row['avg_win']:>7.0f}")
        print(f"    Avg Loss:        ${row['avg_loss']:>7.0f}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS: What Makes a Strong House?")
    print("="*80)

    # Best crisis survival
    best_crisis = df.loc[df['max_dd'].abs().idxmin()]
    print(f"\n🛡️  BEST CRISIS SURVIVAL: {best_crisis['name']}")
    print(f"   Max Drawdown: {best_crisis['max_dd']:.2f}% (vs SPY -55%)")
    print(f"   CAGR: {best_crisis['cagr']:.2f}%")

    # Best returns
    best_returns = df.loc[df['cagr'].idxmax()]
    print(f"\n💰 BEST RETURNS: {best_returns['name']}")
    print(f"   CAGR: {best_returns['cagr']:.2f}%")
    print(f"   Max Drawdown: {best_returns['max_dd']:.2f}%")

    # Best consistency
    best_consistency = df.loc[df['profit_factor'].idxmax()]
    print(f"\n⚖️  BEST CONSISTENCY: {best_consistency['name']}")
    print(f"   Profit Factor: {best_consistency['profit_factor']:.2f}")
    print(f"   Win Rate: {best_consistency['win_rate']:.1f}%")

    # The house winner
    winner = df_sorted.iloc[0]
    print(f"\n🏆 THE HOUSE (Best Overall): {winner['name']}")
    print(f"   House Score: {winner['house_score']:.1f}")
    print(f"   Perfect Balance: Crisis survival + Consistency + Returns")

    print("\n" + "="*80)
    print("INSIGHT: Risk/Return Frontier")
    print("="*80)

    # Plot returns vs drawdown
    print(f"\n{'Config':<20}{'CAGR%':<10}{'MaxDD%':<10}{'Return/Risk':<12}")
    print("-"*80)

    df['return_per_risk'] = df['cagr'] / df['max_dd'].abs()
    df_frontier = df.sort_values('return_per_risk', ascending=False)

    for i, row in df_frontier.iterrows():
        marker = "🏆" if row['name'] == winner['name'] else "  "
        print(f"{marker} {row['name']:<18}{row['cagr']:>8.2f}{row['max_dd']:>10.2f}{row['return_per_risk']:>12.2f}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    print(f"\nBased on comprehensive testing, the optimal 'house' configuration is:\n")
    print(f"  🏛️  {winner['name']}")
    print(f"     R* = {winner['R_star']:.2f}")
    print(f"     χ_crisis = {winner['chi_crisis']:.1f}")
    print(f"\nThis configuration achieves:")
    print(f"  - {winner['cagr']:.2f}% CAGR (vs SPY 7.8%)")
    print(f"  - {winner['max_dd']:.2f}% max drawdown (vs SPY -55%)")
    print(f"  - {winner['win_rate']:.1f}% win rate")
    print(f"  - {winner['profit_factor']:.2f} profit factor")
    print(f"  - {winner['ruin_risk']:.2f}% ruin risk")
    print(f"\n  The house cannot be killed. The house always survives.")

    print("\n" + "="*80)
