#!/usr/bin/env python3
"""
Sensitivity Analysis for χ-Threshold Strategy

Tests robustness of the strategy to parameter changes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chi_backtest_with_fallback import ChiThresholdBacktest, generate_synthetic_market_data

def sensitivity_test_thresholds():
    """Test different χ threshold values."""
    print("="*70)
    print("SENSITIVITY ANALYSIS: χ THRESHOLDS")
    print("="*70)

    # Generate data once
    stock_data, etf_data, stock_returns = generate_synthetic_market_data(
        '2000-01-01', '2024-10-31', 50
    )

    results = []

    # Test different threshold multipliers
    multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    for mult in multipliers:
        backtest = ChiThresholdBacktest(
            start_date='2000-01-01',
            end_date='2024-10-31',
            n_stocks=50,
            window=20,
            transaction_cost=0.001
        )

        # Adjust thresholds
        backtest.thresholds = {
            'optimal': 0.382 * mult,
            'rising': 0.618 * mult,
            'critical': 1.0 * mult
        }

        # Calculate chi
        chi_series = backtest.calculate_chi(stock_returns)

        # Backtest
        strategy_df, rebalances = backtest.backtest_strategy(chi_series, etf_data)

        final_value = strategy_df['portfolio_value'].iloc[-1]
        total_return = (final_value / 100000 - 1) * 100

        daily_returns = strategy_df['portfolio_value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe = ((final_value / 100000) ** (1/24.8) - 1 - 0.02) / (volatility/100)

        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100

        results.append({
            'Threshold Mult': f"{mult:.1f}x",
            'Optimal': f"{0.382*mult:.3f}",
            'Rising': f"{0.618*mult:.3f}",
            'Critical': f"{1.0*mult:.3f}",
            'Total Return (%)': f"{total_return:.1f}",
            'Sharpe': f"{sharpe:.2f}",
            'Max DD (%)': f"{max_dd:.1f}",
            'Rebalances': len(rebalances)
        })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    return results_df

def sensitivity_test_windows():
    """Test different rolling window sizes."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: ROLLING WINDOW SIZE")
    print("="*70)

    stock_data, etf_data, stock_returns = generate_synthetic_market_data(
        '2000-01-01', '2024-10-31', 50
    )

    results = []

    windows = [10, 15, 20, 30, 40, 60]

    for window in windows:
        backtest = ChiThresholdBacktest(
            start_date='2000-01-01',
            end_date='2024-10-31',
            n_stocks=50,
            window=window,
            transaction_cost=0.001
        )

        chi_series = backtest.calculate_chi(stock_returns)
        strategy_df, rebalances = backtest.backtest_strategy(chi_series, etf_data)

        final_value = strategy_df['portfolio_value'].iloc[-1]
        total_return = (final_value / 100000 - 1) * 100

        daily_returns = strategy_df['portfolio_value'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe = ((final_value / 100000) ** (1/24.8) - 1 - 0.02) / (volatility/100)

        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100

        results.append({
            'Window (days)': window,
            'Total Return (%)': f"{total_return:.1f}",
            'Sharpe': f"{sharpe:.2f}",
            'Max DD (%)': f"{max_dd:.1f}",
            'Rebalances': len(rebalances),
            'Avg Days Between': f"{len(strategy_df)/len(rebalances):.0f}" if len(rebalances) > 0 else "N/A"
        })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    return results_df

def sensitivity_test_transaction_costs():
    """Test impact of transaction costs."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: TRANSACTION COSTS")
    print("="*70)

    stock_data, etf_data, stock_returns = generate_synthetic_market_data(
        '2000-01-01', '2024-10-31', 50
    )

    results = []

    costs = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.010]

    for cost in costs:
        backtest = ChiThresholdBacktest(
            start_date='2000-01-01',
            end_date='2024-10-31',
            n_stocks=50,
            window=20,
            transaction_cost=cost
        )

        chi_series = backtest.calculate_chi(stock_returns)
        strategy_df, rebalances = backtest.backtest_strategy(chi_series, etf_data)

        final_value = strategy_df['portfolio_value'].iloc[-1]
        total_return = (final_value / 100000 - 1) * 100

        results.append({
            'TX Cost (bps)': int(cost * 10000),
            'TX Cost (%)': f"{cost*100:.2f}",
            'Total Return (%)': f"{total_return:.1f}",
            'Rebalances': len(rebalances),
            'Total TX Cost': f"${(100000 * cost * len(rebalances)):,.0f}"
        })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    return results_df

def test_monte_carlo_stability():
    """Run multiple simulations to test stability."""
    print("\n" + "="*70)
    print("MONTE CARLO STABILITY TEST")
    print("="*70)
    print("Running 10 simulations with different random seeds...")

    results = []

    for seed in range(10):
        np.random.seed(seed)

        stock_data, etf_data, stock_returns = generate_synthetic_market_data(
            '2000-01-01', '2024-10-31', 50
        )

        backtest = ChiThresholdBacktest(
            start_date='2000-01-01',
            end_date='2024-10-31',
            n_stocks=50,
            window=20,
            transaction_cost=0.001
        )

        chi_series = backtest.calculate_chi(stock_returns)
        strategy_df, rebalances = backtest.backtest_strategy(chi_series, etf_data)

        # Also run 60/40 benchmark
        benchmark = backtest.backtest_benchmark(etf_data, {'SPY': 0.6, 'TLT': 0.4}, rebalance_freq='Q')

        chi_final = strategy_df['portfolio_value'].iloc[-1]
        chi_return = (chi_final / 100000 - 1) * 100

        bench_final = benchmark.iloc[-1]
        bench_return = (bench_final / 100000 - 1) * 100

        outperformance = chi_return - bench_return

        results.append({
            'Sim': seed + 1,
            'χ-Strategy (%)': f"{chi_return:.1f}",
            '60/40 (%)': f"{bench_return:.1f}",
            'Outperformance (%)': f"{outperformance:+.1f}",
            'Rebalances': len(rebalances)
        })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    # Summary statistics
    chi_returns = [float(r['χ-Strategy (%)']) for r in results]
    bench_returns = [float(r['60/40 (%)']) for r in results]
    outperf = [float(r['Outperformance (%)']) for r in results]

    print(f"\n{'-'*70}")
    print("SUMMARY STATISTICS (10 simulations):")
    print(f"{'-'*70}")
    print(f"χ-Strategy:     Mean = {np.mean(chi_returns):.1f}%, StdDev = {np.std(chi_returns):.1f}%")
    print(f"60/40:          Mean = {np.mean(bench_returns):.1f}%, StdDev = {np.std(bench_returns):.1f}%")
    print(f"Outperformance: Mean = {np.mean(outperf):+.1f}%, StdDev = {np.std(outperf):.1f}%")
    print(f"Win rate:       {sum(1 for x in outperf if x > 0)}/10 ({sum(1 for x in outperf if x > 0)*10}%)")

    return results_df

def main():
    """Run all sensitivity analyses."""
    print("\n")
    print("="*70)
    print("χ-THRESHOLD STRATEGY: COMPREHENSIVE SENSITIVITY ANALYSIS")
    print("="*70)
    print("\n")

    # Test 1: Threshold sensitivity
    threshold_results = sensitivity_test_thresholds()
    threshold_results.to_csv('/home/user/Proofpacket/sensitivity_thresholds.csv', index=False)

    # Test 2: Window size sensitivity
    window_results = sensitivity_test_windows()
    window_results.to_csv('/home/user/Proofpacket/sensitivity_windows.csv', index=False)

    # Test 3: Transaction cost impact
    cost_results = sensitivity_test_transaction_costs()
    cost_results.to_csv('/home/user/Proofpacket/sensitivity_costs.csv', index=False)

    # Test 4: Monte Carlo stability
    mc_results = test_monte_carlo_stability()
    mc_results.to_csv('/home/user/Proofpacket/sensitivity_montecarlo.csv', index=False)

    print("\n" + "="*70)
    print("✓ SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print("\nSaved files:")
    print("  - sensitivity_thresholds.csv")
    print("  - sensitivity_windows.csv")
    print("  - sensitivity_costs.csv")
    print("  - sensitivity_montecarlo.csv")

if __name__ == "__main__":
    main()
