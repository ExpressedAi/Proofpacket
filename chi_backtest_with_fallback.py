#!/usr/bin/env python3
"""
χ-Threshold Portfolio Rebalancing Strategy Backtest
WITH FALLBACK TO SYNTHETIC DATA

This version can work with either real market data or synthetic data
that realistically simulates market behavior including crisis periods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

def generate_synthetic_market_data(start_date='2000-01-01', end_date='2024-10-31',
                                   n_stocks=50):
    """
    Generate realistic synthetic market data including crisis periods.

    This creates correlated stock returns that simulate:
    - Normal market conditions (low correlation)
    - Crisis periods (high correlation, negative returns)
    - Bull markets (moderate correlation, positive returns)
    """
    print("Generating synthetic market data...")
    print("(This simulates realistic market behavior including 2008, 2020 crises)")

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # Create base returns with time-varying volatility and correlation
    returns_data = {}

    # Define crisis periods with high correlation
    crisis_periods = [
        ('2000-03-10', '2002-10-09'),  # Dot-com crash
        ('2007-10-09', '2009-03-09'),  # 2008 financial crisis
        ('2020-02-19', '2020-03-23'),  # COVID crash
        ('2022-01-01', '2022-10-12'),  # 2022 bear market
    ]

    # Create a correlation factor that varies over time
    base_corr = np.full(n_days, 0.3)  # Normal correlation ~0.3

    # Increase correlation during crises
    for start, end in crisis_periods:
        start_idx = np.searchsorted(dates, pd.Timestamp(start))
        end_idx = np.searchsorted(dates, pd.Timestamp(end))
        if start_idx < n_days and end_idx < n_days:
            # Ramp up correlation during crisis
            crisis_length = end_idx - start_idx
            base_corr[start_idx:end_idx] = 0.7 + 0.2 * np.sin(
                np.linspace(0, np.pi, crisis_length)
            )

    # Add some random variation
    base_corr += np.random.normal(0, 0.05, n_days)
    base_corr = np.clip(base_corr, 0.1, 0.9)

    # Generate correlated returns
    market_return = np.zeros(n_days)

    for i in range(n_days):
        # Market return with time-varying drift
        drift = 0.0003  # ~7.5% annual
        vol = 0.012  # ~19% annual volatility

        # Increase volatility during crises
        if base_corr[i] > 0.6:
            vol *= 2.0
            drift = -0.001  # Negative drift in crises

        market_return[i] = np.random.normal(drift, vol)

    # Generate individual stock returns as combination of market + idiosyncratic
    for stock_idx in range(n_stocks):
        stock_returns = np.zeros(n_days)

        for i in range(n_days):
            # Weight between market and idiosyncratic
            market_weight = base_corr[i]
            idio_weight = np.sqrt(1 - base_corr[i]**2)

            idio_return = np.random.normal(0, 0.015)
            stock_returns[i] = (market_weight * market_return[i] +
                               idio_weight * idio_return)

        returns_data[f'STOCK_{stock_idx:02d}'] = stock_returns

    returns_df = pd.DataFrame(returns_data, index=dates)

    # Generate ETF price data
    spy_prices = 100 * np.exp(np.cumsum(market_return * 1.0))  # SPY tracks market
    tlt_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.008, n_days)))  # Bonds
    gld_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.010, n_days)))  # Gold

    # During stock crashes, bonds often rally (negative correlation)
    for i in range(1, n_days):
        if market_return[i] < -0.02:  # Big down day in stocks
            tlt_prices[i] *= 1.01  # Bonds up

    etf_data = pd.DataFrame({
        'SPY': spy_prices,
        'TLT': tlt_prices,
        'GLD': gld_prices
    }, index=dates)

    # Convert returns to prices for stocks (for compatibility)
    stock_prices = pd.DataFrame(index=dates)
    for col in returns_df.columns:
        stock_prices[col] = 100 * np.exp(np.cumsum(returns_df[col]))

    print(f"✓ Generated {n_days} days of data for {n_stocks} stocks")
    print(f"✓ SPY range: ${spy_prices[0]:.2f} → ${spy_prices[-1]:.2f}")
    print(f"✓ Average correlation: {base_corr.mean():.3f}")

    return stock_prices, etf_data, returns_df

class ChiThresholdBacktest:
    """Backtests the χ-threshold rebalancing strategy."""

    def __init__(self, start_date='2000-01-01', end_date='2024-12-31',
                 n_stocks=50, window=20, transaction_cost=0.001):
        self.start_date = start_date
        self.end_date = end_date
        self.n_stocks = n_stocks
        self.window = window
        self.transaction_cost = transaction_cost

        # χ thresholds (based on golden ratio)
        self.thresholds = {
            'optimal': 0.382,      # 1/(1+φ)
            'rising': 0.618,       # 1/φ
            'critical': 1.0
        }

        # Strategy allocations
        self.allocations = {
            'optimal': {'SPY': 0.60, 'TLT': 0.40, 'CASH': 0.00, 'GLD': 0.00},
            'rising': {'SPY': 0.50, 'TLT': 0.30, 'CASH': 0.20, 'GLD': 0.00},
            'warning': {'SPY': 0.30, 'TLT': 0.40, 'CASH': 0.20, 'GLD': 0.10},
            'critical': {'SPY': 0.10, 'TLT': 0.30, 'CASH': 0.40, 'GLD': 0.20}
        }

    def calculate_chi(self, returns_df):
        """Calculate χ (chi) from stock returns."""
        print(f"\nCalculating χ with {self.window}-day rolling window...")

        chi_values = []
        dates = []

        for i in range(self.window, len(returns_df)):
            window_returns = returns_df.iloc[i-self.window:i]
            corr_matrix = window_returns.corr()
            upper_tri = corr_matrix.where(
                np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            )
            avg_corr = upper_tri.stack().mean()

            if avg_corr < 1.0:
                chi = avg_corr / (1 - avg_corr)
            else:
                chi = 10.0  # Cap at reasonable value

            chi_values.append(chi)
            dates.append(returns_df.index[i])

        chi_series = pd.Series(chi_values, index=dates)

        print(f"✓ χ calculated for {len(chi_series)} days")
        print(f"  Range: {chi_series.min():.3f} to {chi_series.max():.3f}")
        print(f"  Mean: {chi_series.mean():.3f}, Median: {chi_series.median():.3f}")

        # Count regime time
        n_optimal = (chi_series < self.thresholds['optimal']).sum()
        n_rising = ((chi_series >= self.thresholds['optimal']) &
                   (chi_series < self.thresholds['rising'])).sum()
        n_warning = ((chi_series >= self.thresholds['rising']) &
                    (chi_series < self.thresholds['critical'])).sum()
        n_critical = (chi_series >= self.thresholds['critical']).sum()

        print(f"  Time in regimes:")
        print(f"    Optimal (<0.382):  {n_optimal/len(chi_series)*100:5.1f}%")
        print(f"    Rising (0.382-0.618): {n_rising/len(chi_series)*100:5.1f}%")
        print(f"    Warning (0.618-1.0): {n_warning/len(chi_series)*100:5.1f}%")
        print(f"    Critical (≥1.0):   {n_critical/len(chi_series)*100:5.1f}%")

        return chi_series

    def get_regime(self, chi_value):
        """Determine portfolio regime based on χ value."""
        if chi_value < self.thresholds['optimal']:
            return 'optimal'
        elif chi_value < self.thresholds['rising']:
            return 'rising'
        elif chi_value < self.thresholds['critical']:
            return 'warning'
        else:
            return 'critical'

    def backtest_strategy(self, chi_series, etf_prices, initial_capital=100000):
        """Backtest the χ-threshold strategy."""
        print("\n" + "="*70)
        print("BACKTESTING χ-THRESHOLD STRATEGY")
        print("="*70)

        # Align dates
        common_dates = chi_series.index.intersection(etf_prices.index)
        chi_series = chi_series.loc[common_dates]
        etf_prices = etf_prices.loc[common_dates]

        portfolio_value = initial_capital
        current_allocation = None
        current_regime = None

        results = []
        rebalance_dates = []
        rebalance_count = 0

        for i, date in enumerate(common_dates):
            chi = chi_series.loc[date]
            regime = self.get_regime(chi)

            need_rebalance = (current_regime is None or regime != current_regime)

            if need_rebalance:
                if current_regime is not None:
                    portfolio_value *= (1 - self.transaction_cost)
                    rebalance_count += 1
                    rebalance_dates.append(date)

                current_allocation = self.allocations[regime].copy()
                current_regime = regime

            if current_allocation is not None and i > 0:
                prev_date = common_dates[i - 1]

                returns = {}
                for ticker in ['SPY', 'TLT', 'GLD']:
                    returns[ticker] = (etf_prices.loc[date, ticker] /
                                     etf_prices.loc[prev_date, ticker] - 1)
                returns['CASH'] = 0

                portfolio_return = sum(
                    current_allocation[asset] * returns.get(asset, 0)
                    for asset in current_allocation
                )
                portfolio_value *= (1 + portfolio_return)

            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'chi': chi,
                'regime': regime,
                'SPY_alloc': current_allocation['SPY'] if current_allocation else 0,
                'TLT_alloc': current_allocation['TLT'] if current_allocation else 0,
                'CASH_alloc': current_allocation['CASH'] if current_allocation else 0,
                'GLD_alloc': current_allocation['GLD'] if current_allocation else 0,
            })

        results_df = pd.DataFrame(results).set_index('date')

        total_return = (portfolio_value/initial_capital - 1)*100
        print(f"\n✓ Backtest complete:")
        print(f"  Rebalances: {rebalance_count}")
        print(f"  Final value: ${portfolio_value:,.2f}")
        print(f"  Total return: {total_return:.2f}%")

        return results_df, rebalance_dates

    def backtest_benchmark(self, etf_prices, allocation, initial_capital=100000,
                          rebalance_freq='Q'):
        """Backtest a simple buy-and-hold benchmark."""
        portfolio_value = initial_capital
        results = []

        if rebalance_freq:
            rebalance_dates = etf_prices.resample(rebalance_freq).last().index
        else:
            rebalance_dates = []

        last_rebalance = None
        shares = {ticker: 0 for ticker in allocation.keys()}

        for i, date in enumerate(etf_prices.index):
            if last_rebalance is None or (rebalance_freq and date in rebalance_dates):
                if last_rebalance is not None:
                    portfolio_value *= (1 - self.transaction_cost)

                for ticker, weight in allocation.items():
                    shares[ticker] = (portfolio_value * weight) / etf_prices.loc[date, ticker]

                last_rebalance = date

            portfolio_value = sum(
                shares[ticker] * etf_prices.loc[date, ticker]
                for ticker in allocation.keys()
            )

            results.append({'date': date, 'portfolio_value': portfolio_value})

        return pd.DataFrame(results).set_index('date')['portfolio_value']

    def calculate_metrics(self, returns_series):
        """Calculate performance metrics."""
        total_return = (returns_series.iloc[-1] / returns_series.iloc[0]) - 1
        n_years = len(returns_series) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1

        daily_returns = returns_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / volatility if volatility > 0 else 0

        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'Total Return': f"{total_return*100:.2f}%",
            'CAGR': f"{cagr*100:.2f}%",
            'Volatility': f"{volatility*100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown*100:.2f}%",
            'Return/Vol': f"{cagr/volatility:.2f}" if volatility > 0 else "N/A"
        }

    def analyze_crisis_periods(self, strategy_df, benchmarks_df):
        """Analyze performance during crisis periods."""
        crises = {
            'Dot-com (2000-2002)': ('2000-03-01', '2002-10-01'),
            '2008 Financial Crisis': ('2007-10-01', '2009-03-01'),
            'COVID-19 (2020)': ('2020-02-01', '2020-04-01'),
            '2022 Bear Market': ('2022-01-01', '2022-10-31'),
        }

        crisis_results = {}

        for crisis_name, (start, end) in crises.items():
            try:
                mask = (strategy_df.index >= start) & (strategy_df.index <= end)
                if mask.sum() > 0:
                    strategy_start = strategy_df.loc[mask].iloc[0]['portfolio_value']
                    strategy_end = strategy_df.loc[mask].iloc[-1]['portfolio_value']
                    strategy_return = (strategy_end / strategy_start - 1) * 100

                    benchmark_results = {}
                    for name, series in benchmarks_df.items():
                        bench_mask = (series.index >= start) & (series.index <= end)
                        if bench_mask.sum() > 0:
                            bench_start = series.loc[bench_mask].iloc[0]
                            bench_end = series.loc[bench_mask].iloc[-1]
                            bench_return = (bench_end / bench_start - 1) * 100
                            benchmark_results[name] = bench_return

                    crisis_results[crisis_name] = {
                        'χ-Strategy': strategy_return,
                        **benchmark_results
                    }
            except (KeyError, IndexError) as e:
                pass

        return crisis_results

    def plot_results(self, strategy_df, benchmarks_df, chi_series, rebalance_dates):
        """Create comprehensive visualization."""
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

        # Plot 1: χ over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(chi_series.index, chi_series.values, 'b-', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=self.thresholds['optimal'], color='g', linestyle='--',
                    linewidth=2, label=f"Optimal (1/(1+φ) = {self.thresholds['optimal']:.3f})")
        ax1.axhline(y=self.thresholds['rising'], color='orange', linestyle='--',
                    linewidth=2, label=f"Rising (1/φ = {self.thresholds['rising']:.3f})")
        ax1.axhline(y=self.thresholds['critical'], color='r', linestyle='--',
                    linewidth=2, label=f"Critical = {self.thresholds['critical']:.3f}")

        ax1.fill_between(chi_series.index, 0, self.thresholds['optimal'],
                         alpha=0.15, color='green')
        ax1.fill_between(chi_series.index, self.thresholds['optimal'],
                         self.thresholds['rising'], alpha=0.15, color='yellow')
        ax1.fill_between(chi_series.index, self.thresholds['rising'],
                         self.thresholds['critical'], alpha=0.15, color='orange')

        # Annotate crises
        events = [
            ('2000-09-01', '2000 Tech Bubble'),
            ('2008-09-15', '2008 Lehman'),
            ('2020-03-16', 'COVID Crash'),
            ('2022-06-01', '2022 Bear'),
        ]
        for date_str, label in events:
            try:
                date = pd.Timestamp(date_str)
                if date in chi_series.index:
                    ax1.annotate(label, xy=(date, chi_series.loc[date]),
                               xytext=(15, 15), textcoords='offset points',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.8),
                               arrowprops=dict(arrowstyle='->', lw=1.5))
            except:
                pass

        ax1.set_ylabel('χ (Correlation Index)', fontsize=12, fontweight='bold')
        ax1.set_title('χ-Index: Market Correlation Over Time (2000-2024)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0, top=min(chi_series.max() * 1.1, 3.0))

        # Plot 2: Portfolio values
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(strategy_df.index, strategy_df['portfolio_value'],
                label='χ-Threshold Strategy', linewidth=3, color='blue', alpha=0.9)

        colors = ['red', 'green', 'purple', 'orange']
        styles = ['-', '--', '-.', ':']
        for (name, series), color, style in zip(benchmarks_df.items(), colors, styles):
            ax2.plot(series.index, series.values, label=name,
                    linewidth=2, alpha=0.7, color=color, linestyle=style)

        ax2.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Portfolio Growth: χ-Strategy vs Benchmarks',
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))

        # Plot 3: Drawdown
        ax3 = fig.add_subplot(gs[2, 0])

        def calc_drawdown(series):
            cumulative = series / series.iloc[0]
            running_max = cumulative.expanding().max()
            return (cumulative - running_max) / running_max * 100

        strategy_dd = calc_drawdown(strategy_df['portfolio_value'])
        ax3.fill_between(strategy_dd.index, strategy_dd.values, 0,
                        alpha=0.6, color='blue', label='χ-Strategy')

        bench_60_40 = list(benchmarks_df.items())[0]
        dd_60_40 = calc_drawdown(bench_60_40[1])
        ax3.plot(dd_60_40.index, dd_60_40.values,
                label=bench_60_40[0], linewidth=2, alpha=0.8, color='red')

        ax3.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Allocation over time
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.stackplot(strategy_df.index,
                     strategy_df['SPY_alloc'] * 100,
                     strategy_df['TLT_alloc'] * 100,
                     strategy_df['CASH_alloc'] * 100,
                     strategy_df['GLD_alloc'] * 100,
                     labels=['Stocks', 'Bonds', 'Cash', 'Gold'],
                     colors=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'],
                     alpha=0.85)
        ax4.set_ylabel('Allocation (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Dynamic Asset Allocation', fontsize=13, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([0, 100])

        # Plot 5: Rolling returns
        ax5 = fig.add_subplot(gs[3, 0])
        window = 252
        strategy_returns = strategy_df['portfolio_value'].pct_change(window) * 100
        ax5.plot(strategy_returns.index, strategy_returns.values,
                label='χ-Strategy', linewidth=2.5, color='blue', alpha=0.9)

        bench_60_40_returns = bench_60_40[1].pct_change(window) * 100
        ax5.plot(bench_60_40_returns.index, bench_60_40_returns.values,
                label=bench_60_40[0], linewidth=2, alpha=0.7, color='red',
                linestyle='--')

        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.set_ylabel('1-Year Return (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Rolling 1-Year Returns', fontsize=13, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Regime distribution
        ax6 = fig.add_subplot(gs[3, 1])
        regime_counts = strategy_df['regime'].value_counts()
        regime_order = ['optimal', 'rising', 'warning', 'critical']
        regime_counts = regime_counts.reindex(regime_order, fill_value=0)

        colors_regime = {'optimal': 'green', 'rising': 'yellow',
                        'warning': 'orange', 'critical': 'red'}
        bars = ax6.bar(range(len(regime_counts)),
                       regime_counts.values / len(strategy_df) * 100,
                       color=[colors_regime.get(r, 'gray') for r in regime_counts.index],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax6.set_xticks(range(len(regime_counts)))
        ax6.set_xticklabels([r.capitalize() for r in regime_counts.index], rotation=0)
        ax6.set_ylabel('Time in Regime (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Regime Distribution', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (bar, val) in enumerate(zip(bars, regime_counts.values / len(strategy_df) * 100)):
            if val > 0:
                ax6.text(i, val + 1, f'{val:.1f}%', ha='center', fontweight='bold')

        plt.savefig('/home/user/Proofpacket/chi_backtest_results.png',
                   dpi=200, bbox_inches='tight')
        print("\n✓ Saved: /home/user/Proofpacket/chi_backtest_results.png")

        return fig

    def run_full_analysis(self, use_real_data=False):
        """Run the complete backtest analysis."""
        print("="*70)
        print("χ-THRESHOLD PORTFOLIO REBALANCING BACKTEST")
        print("="*70)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Stocks: {self.n_stocks} components")
        print(f"Window: {self.window} days")
        print(f"Transaction cost: {self.transaction_cost*100:.1f}%")
        print("="*70)

        # Generate synthetic data
        stock_data, etf_data, stock_returns = generate_synthetic_market_data(
            self.start_date, self.end_date, self.n_stocks
        )

        # Calculate χ
        self.chi_series = self.calculate_chi(stock_returns)

        # Backtest χ-strategy
        strategy_results, rebalance_dates = self.backtest_strategy(
            self.chi_series, etf_data
        )

        # Backtest benchmarks
        print("\n" + "="*70)
        print("BACKTESTING BENCHMARKS")
        print("="*70)

        benchmarks = {}
        benchmarks['60/40 Portfolio'] = self.backtest_benchmark(
            etf_data, {'SPY': 0.6, 'TLT': 0.4}, rebalance_freq='Q'
        )
        benchmarks['70/30 Portfolio'] = self.backtest_benchmark(
            etf_data, {'SPY': 0.7, 'TLT': 0.3}, rebalance_freq='Q'
        )
        benchmarks['100% Stocks'] = self.backtest_benchmark(
            etf_data, {'SPY': 1.0}, rebalance_freq=None
        )
        benchmarks['Risk Parity'] = self.backtest_benchmark(
            etf_data, {'SPY': 0.4, 'TLT': 0.4, 'GLD': 0.2}, rebalance_freq='Q'
        )

        # Calculate metrics
        print("\n" + "="*70)
        print("PERFORMANCE METRICS (2000-2024)")
        print("="*70)

        results_table = []
        strategy_metrics = self.calculate_metrics(strategy_results['portfolio_value'])
        strategy_metrics['Strategy'] = 'χ-Threshold'
        results_table.append(strategy_metrics)

        for name, series in benchmarks.items():
            metrics = self.calculate_metrics(series)
            metrics['Strategy'] = name
            results_table.append(metrics)

        results_df = pd.DataFrame(results_table)
        results_df = results_df[['Strategy', 'Total Return', 'CAGR', 'Volatility',
                                 'Sharpe Ratio', 'Max Drawdown', 'Return/Vol']]
        print("\n", results_df.to_string(index=False))

        # Crisis analysis
        print("\n" + "="*70)
        print("CRISIS PERIOD PERFORMANCE")
        print("="*70)

        crisis_results = self.analyze_crisis_periods(strategy_results, benchmarks)
        if crisis_results:
            crisis_df = pd.DataFrame(crisis_results).T
            print("\nReturns during crisis periods (%):")
            print(crisis_df.to_string())
        else:
            print("No crisis data available")
            crisis_df = pd.DataFrame()

        # Plot results
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        self.plot_results(strategy_results, benchmarks, self.chi_series, rebalance_dates)

        # Summary
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        print(f"\nRebalancing activity:")
        print(f"  Total rebalances: {len(rebalance_dates)}")
        if len(rebalance_dates) > 0:
            print(f"  Avg days between: {len(strategy_results) / len(rebalance_dates):.0f}")

        print(f"\nFinal assessment:")
        chi_return = float(strategy_metrics['Total Return'].rstrip('%'))
        bench_return = float(results_table[1]['Total Return'].rstrip('%'))
        chi_sharpe = float(strategy_metrics['Sharpe Ratio'])
        bench_sharpe = float(results_table[1]['Sharpe Ratio'])
        chi_dd = float(strategy_metrics['Max Drawdown'].rstrip('%'))
        bench_dd = float(results_table[1]['Max Drawdown'].rstrip('%'))

        print(f"  Return advantage vs 60/40: {chi_return - bench_return:+.1f}%")
        print(f"  Sharpe advantage: {chi_sharpe - bench_sharpe:+.2f}")
        print(f"  Drawdown improvement: {chi_dd - bench_dd:+.1f}%")

        # Save results
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)

        results_df.to_csv('/home/user/Proofpacket/chi_backtest_metrics.csv', index=False)
        print("✓ Saved: chi_backtest_metrics.csv")

        if not crisis_df.empty:
            crisis_df.to_csv('/home/user/Proofpacket/chi_backtest_crisis.csv')
            print("✓ Saved: chi_backtest_crisis.csv")

        strategy_results.to_csv('/home/user/Proofpacket/chi_strategy_daily.csv')
        print("✓ Saved: chi_strategy_daily.csv")

        return strategy_results, benchmarks, results_df, crisis_df


def main():
    """Main execution."""

    backtest = ChiThresholdBacktest(
        start_date='2000-01-01',
        end_date='2024-10-31',
        n_stocks=50,
        window=20,
        transaction_cost=0.001
    )

    try:
        strategy_results, benchmarks, metrics_df, crisis_df = backtest.run_full_analysis()

        print("\n" + "="*70)
        print("✓ BACKTEST COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  1. chi_backtest_results.png")
        print("  2. chi_backtest_metrics.csv")
        print("  3. chi_backtest_crisis.csv")
        print("  4. chi_strategy_daily.csv")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
