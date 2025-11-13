#!/usr/bin/env python3
"""
χ-Threshold Portfolio Rebalancing Strategy Backtest

Tests the hypothesis that portfolio allocation based on market correlation (χ)
can improve risk-adjusted returns by detecting regime changes.

χ = avg_correlation / (1 - avg_correlation)

Strategy:
- χ < 0.382: 60/40 stocks/bonds (optimal diversity)
- 0.382 ≤ χ < 0.618: 50/30/20 stocks/bonds/cash (rising correlation)
- 0.618 ≤ χ < 1.0: 30/40/20/10 stocks/bonds/cash/gold (phase-lock warning)
- χ ≥ 1.0: 10/30/40/20 stocks/bonds/cash/gold (critical phase-lock)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance, provide fallback
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    print("WARNING: yfinance not installed. Install with: pip install yfinance")
    HAS_YFINANCE = False

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

class ChiThresholdBacktest:
    """Backtests the χ-threshold rebalancing strategy."""

    def __init__(self, start_date='2000-01-01', end_date='2024-12-31',
                 n_stocks=50, window=20, transaction_cost=0.001):
        """
        Initialize the backtester.

        Parameters:
        -----------
        start_date : str
            Start date for backtest (YYYY-MM-DD)
        end_date : str
            End date for backtest (YYYY-MM-DD)
        n_stocks : int
            Number of S&P 500 stocks to use for χ calculation
        window : int
            Rolling window for correlation calculation (days)
        transaction_cost : float
            Transaction cost as fraction (0.001 = 0.1%)
        """
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

        self.data = None
        self.chi_series = None
        self.sp500_stocks = None

    def get_sp500_tickers(self):
        """Get list of S&P 500 tickers (top N by market cap approximation)."""
        # These are the largest, most liquid S&P 500 stocks as of 2024
        # In practice, you'd fetch this dynamically
        top_sp500 = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'AVGO', 'HD',
            'CVX', 'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'LLY', 'ADBE', 'TMO',
            'CSCO', 'MCD', 'ACN', 'NFLX', 'ABT', 'CRM', 'NKE', 'DHR', 'VZ',
            'INTC', 'TXN', 'CMCSA', 'DIS', 'WFC', 'PM', 'NEE', 'BMY', 'UPS',
            'HON', 'IBM', 'AMGN', 'QCOM', 'ORCL', 'BA', 'CAT', 'GE', 'AMD',
            'CVS', 'UNP', 'LOW', 'SBUX', 'GS', 'ELV', 'AXP', 'SPGI', 'BLK',
            'RTX', 'DE', 'LMT', 'T', 'GILD', 'MMM', 'PLD', 'MDLZ', 'CI',
            'BKNG', 'ADI', 'AMAT', 'AMT', 'ADP', 'ISRG', 'SYK', 'MO', 'TJX',
            'CB', 'C', 'ZTS', 'SCHW', 'PGR', 'VRTX', 'DUK', 'SO', 'MMC', 'BDX',
            'REGN', 'NOC', 'TMUS', 'EOG', 'BSX', 'HUM', 'CL', 'EQIX', 'ICE', 'SLB'
        ]
        return top_sp500[:self.n_stocks]

    def download_data(self):
        """Download historical price data."""
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        print("Downloading data...")
        print(f"Date range: {self.start_date} to {self.end_date}")

        # Get tickers
        stock_tickers = self.get_sp500_tickers()
        etf_tickers = ['SPY', 'TLT', 'GLD']  # Cash doesn't need data

        print(f"Downloading {len(stock_tickers)} S&P 500 stocks for χ calculation...")

        # Download S&P 500 stocks for χ calculation
        stock_data = yf.download(stock_tickers, start=self.start_date,
                                 end=self.end_date, progress=False)['Adj Close']

        # Download ETFs for portfolio
        print("Downloading ETF data (SPY, TLT, GLD)...")
        etf_data = yf.download(etf_tickers, start=self.start_date,
                               end=self.end_date, progress=False)['Adj Close']

        # Store data
        self.sp500_stocks = stock_data
        self.etf_prices = etf_data

        print(f"Downloaded {len(stock_data)} days of stock data")
        print(f"Stocks with data: {stock_data.columns.tolist()[:10]}...")
        print(f"Date range: {stock_data.index[0]} to {stock_data.index[-1]}")

        return stock_data, etf_data

    def calculate_chi(self, returns_df):
        """
        Calculate χ (chi) from stock returns.

        χ = avg_correlation / (1 - avg_correlation)

        Parameters:
        -----------
        returns_df : pd.DataFrame
            Daily returns for multiple stocks

        Returns:
        --------
        pd.Series : χ values over time
        """
        print(f"\nCalculating χ with {self.window}-day rolling window...")

        chi_values = []
        dates = []

        for i in range(self.window, len(returns_df)):
            # Get window of returns
            window_returns = returns_df.iloc[i-self.window:i]

            # Calculate correlation matrix
            corr_matrix = window_returns.corr()

            # Extract upper triangle (excluding diagonal)
            upper_tri = corr_matrix.where(
                np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            )

            # Calculate average correlation
            avg_corr = upper_tri.stack().mean()

            # Calculate χ
            if avg_corr < 1.0:  # Avoid division by zero
                chi = avg_corr / (1 - avg_corr)
            else:
                chi = np.inf

            chi_values.append(chi)
            dates.append(returns_df.index[i])

        chi_series = pd.Series(chi_values, index=dates)

        print(f"χ calculated for {len(chi_series)} days")
        print(f"χ range: {chi_series.min():.3f} to {chi_series.max():.3f}")
        print(f"χ mean: {chi_series.mean():.3f}, median: {chi_series.median():.3f}")

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
        """
        Backtest the χ-threshold strategy.

        Parameters:
        -----------
        chi_series : pd.Series
            χ values over time
        etf_prices : pd.DataFrame
            Prices for SPY, TLT, GLD
        initial_capital : float
            Starting portfolio value

        Returns:
        --------
        pd.DataFrame : Portfolio values and allocations over time
        """
        print("\n" + "="*60)
        print("BACKTESTING χ-THRESHOLD STRATEGY")
        print("="*60)

        # Align dates
        common_dates = chi_series.index.intersection(etf_prices.index)
        chi_series = chi_series.loc[common_dates]
        etf_prices = etf_prices.loc[common_dates]

        # Initialize portfolio
        portfolio_value = initial_capital
        current_allocation = None
        current_regime = None

        # Track results
        results = []
        rebalance_dates = []
        rebalance_count = 0

        for date in common_dates:
            chi = chi_series.loc[date]
            regime = self.get_regime(chi)

            # Check if we need to rebalance
            need_rebalance = (current_regime is None or regime != current_regime)

            if need_rebalance:
                # Apply transaction costs
                if current_regime is not None:
                    portfolio_value *= (1 - self.transaction_cost)
                    rebalance_count += 1
                    rebalance_dates.append(date)

                # Update allocation
                current_allocation = self.allocations[regime].copy()
                current_regime = regime

            # Calculate portfolio value (mark-to-market)
            if current_allocation is not None:
                # Get returns for each asset
                returns = {}
                if date != common_dates[0]:
                    prev_date = common_dates[common_dates.get_loc(date) - 1]

                    for ticker in ['SPY', 'TLT', 'GLD']:
                        if ticker in etf_prices.columns:
                            returns[ticker] = (etf_prices.loc[date, ticker] /
                                             etf_prices.loc[prev_date, ticker] - 1)
                        else:
                            returns[ticker] = 0

                    returns['CASH'] = 0  # Cash earns 0% (simplified)

                    # Update portfolio value
                    portfolio_return = sum(
                        current_allocation[asset] * returns.get(asset, 0)
                        for asset in current_allocation
                    )
                    portfolio_value *= (1 + portfolio_return)

            # Record state
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

        print(f"\nBacktest complete:")
        print(f"  Rebalances: {rebalance_count}")
        print(f"  Final portfolio value: ${portfolio_value:,.2f}")
        print(f"  Total return: {(portfolio_value/initial_capital - 1)*100:.2f}%")

        return results_df, rebalance_dates

    def backtest_benchmark(self, etf_prices, allocation, initial_capital=100000,
                          rebalance_freq='Q'):
        """
        Backtest a simple buy-and-hold benchmark.

        Parameters:
        -----------
        etf_prices : pd.DataFrame
            Prices for SPY, TLT, GLD
        allocation : dict
            Fixed allocation (e.g., {'SPY': 0.6, 'TLT': 0.4})
        initial_capital : float
            Starting portfolio value
        rebalance_freq : str
            Rebalancing frequency ('Q' = quarterly, 'M' = monthly, None = buy-and-hold)

        Returns:
        --------
        pd.Series : Portfolio values over time
        """
        portfolio_value = initial_capital
        results = []

        # Determine rebalance dates
        if rebalance_freq:
            rebalance_dates = etf_prices.resample(rebalance_freq).last().index
        else:
            rebalance_dates = []

        last_rebalance = None
        shares = {ticker: 0 for ticker in allocation.keys()}

        for date in etf_prices.index:
            # Rebalance if needed
            if last_rebalance is None or (rebalance_freq and date in rebalance_dates):
                # Apply transaction costs
                if last_rebalance is not None:
                    portfolio_value *= (1 - self.transaction_cost)

                # Buy shares according to allocation
                for ticker, weight in allocation.items():
                    if ticker in etf_prices.columns:
                        shares[ticker] = (portfolio_value * weight) / etf_prices.loc[date, ticker]
                    else:
                        shares[ticker] = 0

                last_rebalance = date

            # Calculate portfolio value
            portfolio_value = sum(
                shares[ticker] * etf_prices.loc[date, ticker]
                for ticker in allocation.keys()
                if ticker in etf_prices.columns
            )

            results.append({'date': date, 'portfolio_value': portfolio_value})

        return pd.DataFrame(results).set_index('date')['portfolio_value']

    def calculate_metrics(self, returns_series, risk_free_rate=0.02):
        """Calculate performance metrics."""
        # Annualized return
        total_return = (returns_series.iloc[-1] / returns_series.iloc[0]) - 1
        n_years = len(returns_series) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1

        # Daily returns
        daily_returns = returns_series.pct_change().dropna()

        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_return = cagr - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0

        # Max drawdown
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
        }

    def analyze_crisis_periods(self, strategy_df, benchmarks_df):
        """Analyze performance during crisis periods."""
        crises = {
            'Dot-com Crash': ('2000-03-01', '2002-10-01'),
            '2008 Financial Crisis': ('2007-10-01', '2009-03-01'),
            'COVID-19 Crash': ('2020-02-01', '2020-03-31'),
            '2022 Bond Crash': ('2022-01-01', '2022-10-31'),
        }

        crisis_results = {}

        for crisis_name, (start, end) in crises.items():
            try:
                # Strategy performance
                strategy_start = strategy_df.loc[start:end].iloc[0]['portfolio_value']
                strategy_end = strategy_df.loc[start:end].iloc[-1]['portfolio_value']
                strategy_return = (strategy_end / strategy_start - 1) * 100

                # Benchmark performance
                benchmark_results = {}
                for name, series in benchmarks_df.items():
                    bench_start = series.loc[start:end].iloc[0]
                    bench_end = series.loc[start:end].iloc[-1]
                    bench_return = (bench_end / bench_start - 1) * 100
                    benchmark_results[name] = bench_return

                crisis_results[crisis_name] = {
                    'χ-Strategy': strategy_return,
                    **benchmark_results
                }
            except (KeyError, IndexError):
                # Crisis period not in data
                pass

        return crisis_results

    def plot_results(self, strategy_df, benchmarks_df, chi_series, rebalance_dates):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Plot 1: χ over time with regime zones
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(chi_series.index, chi_series.values, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(y=self.thresholds['optimal'], color='g', linestyle='--',
                    label=f"Optimal threshold (φ⁻¹-1 = {self.thresholds['optimal']:.3f})")
        ax1.axhline(y=self.thresholds['rising'], color='orange', linestyle='--',
                    label=f"Rising threshold (φ⁻¹ = {self.thresholds['rising']:.3f})")
        ax1.axhline(y=self.thresholds['critical'], color='r', linestyle='--',
                    label=f"Critical threshold = {self.thresholds['critical']:.3f}")

        # Shade regime zones
        ax1.fill_between(chi_series.index, 0, self.thresholds['optimal'],
                         alpha=0.1, color='green', label='Optimal Diversity')
        ax1.fill_between(chi_series.index, self.thresholds['optimal'],
                         self.thresholds['rising'], alpha=0.1, color='yellow',
                         label='Rising Correlation')
        ax1.fill_between(chi_series.index, self.thresholds['rising'],
                         self.thresholds['critical'], alpha=0.1, color='orange',
                         label='Phase-Lock Warning')

        # Annotate major events
        events = {
            '2008-09-15': '2008 Crisis',
            '2020-03-16': 'COVID Crash',
            '2022-06-01': '2022 Bear',
        }
        for date_str, label in events.items():
            try:
                date = pd.to_datetime(date_str)
                if date in chi_series.index:
                    ax1.annotate(label, xy=(date, chi_series.loc[date]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            except:
                pass

        ax1.set_ylabel('χ (Market Correlation Index)', fontsize=10)
        ax1.set_title('χ-Index Over Time (2000-2024)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Plot 2: Portfolio values comparison
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(strategy_df.index, strategy_df['portfolio_value'],
                label='χ-Threshold Strategy', linewidth=2, color='blue')

        colors = ['red', 'green', 'purple', 'orange']
        for (name, series), color in zip(benchmarks_df.items(), colors):
            ax2.plot(series.index, series.values, label=name,
                    linewidth=1.5, alpha=0.7, color=color)

        ax2.set_ylabel('Portfolio Value ($)', fontsize=10)
        ax2.set_title('Portfolio Value Comparison (2000-2024)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Plot 3: Drawdown comparison
        ax3 = fig.add_subplot(gs[2, 0])

        # Calculate drawdowns
        def calc_drawdown(series):
            cumulative = series / series.iloc[0]
            running_max = cumulative.expanding().max()
            return (cumulative - running_max) / running_max * 100

        strategy_dd = calc_drawdown(strategy_df['portfolio_value'])
        ax3.fill_between(strategy_dd.index, strategy_dd.values, 0,
                        alpha=0.5, color='blue', label='χ-Strategy')

        for (name, series), color in zip(list(benchmarks_df.items())[:2], ['red', 'green']):
            dd = calc_drawdown(series)
            ax3.plot(dd.index, dd.values, label=name, linewidth=1, alpha=0.7, color=color)

        ax3.set_ylabel('Drawdown (%)', fontsize=10)
        ax3.set_title('Drawdown Comparison', fontsize=11, fontweight='bold')
        ax3.legend(loc='lower left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Allocation over time
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.stackplot(strategy_df.index,
                     strategy_df['SPY_alloc'] * 100,
                     strategy_df['TLT_alloc'] * 100,
                     strategy_df['CASH_alloc'] * 100,
                     strategy_df['GLD_alloc'] * 100,
                     labels=['Stocks (SPY)', 'Bonds (TLT)', 'Cash', 'Gold (GLD)'],
                     colors=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'],
                     alpha=0.8)
        ax4.set_ylabel('Allocation (%)', fontsize=10)
        ax4.set_title('χ-Strategy Asset Allocation Over Time', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 100])

        # Plot 5: Rolling returns comparison
        ax5 = fig.add_subplot(gs[3, 0])
        window = 252  # 1 year
        strategy_returns = strategy_df['portfolio_value'].pct_change(window) * 100
        ax5.plot(strategy_returns.index, strategy_returns.values,
                label='χ-Strategy', linewidth=2, color='blue')

        for (name, series), color in zip(list(benchmarks_df.items())[:2], ['red', 'green']):
            rolling_ret = series.pct_change(window) * 100
            ax5.plot(rolling_ret.index, rolling_ret.values,
                    label=name, linewidth=1, alpha=0.7, color=color)

        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_ylabel('1-Year Rolling Return (%)', fontsize=10)
        ax5.set_title('Rolling 1-Year Returns', fontsize=11, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=8)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Regime distribution
        ax6 = fig.add_subplot(gs[3, 1])
        regime_counts = strategy_df['regime'].value_counts()
        colors_regime = {'optimal': 'green', 'rising': 'yellow',
                        'warning': 'orange', 'critical': 'red'}
        ax6.bar(regime_counts.index, regime_counts.values / len(strategy_df) * 100,
               color=[colors_regime.get(r, 'gray') for r in regime_counts.index])
        ax6.set_ylabel('Time Spent (%)', fontsize=10)
        ax6.set_title('Time in Each Regime', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.savefig('/home/user/Proofpacket/chi_backtest_results.png',
                   dpi=150, bbox_inches='tight')
        print("\n✓ Saved plot: /home/user/Proofpacket/chi_backtest_results.png")

        return fig

    def run_full_analysis(self):
        """Run the complete backtest analysis."""
        print("="*60)
        print("χ-THRESHOLD PORTFOLIO REBALANCING BACKTEST")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Stocks: {self.n_stocks} largest S&P 500 components")
        print(f"Rolling window: {self.window} days")
        print(f"Transaction cost: {self.transaction_cost*100}%")
        print("="*60)

        # Step 1: Download data
        stock_data, etf_data = self.download_data()

        # Step 2: Calculate χ
        stock_returns = stock_data.pct_change().dropna()
        self.chi_series = self.calculate_chi(stock_returns)

        # Step 3: Backtest χ-strategy
        strategy_results, rebalance_dates = self.backtest_strategy(
            self.chi_series, etf_data
        )

        # Step 4: Backtest benchmarks
        print("\n" + "="*60)
        print("BACKTESTING BENCHMARKS")
        print("="*60)

        benchmarks = {}

        # 60/40 Portfolio (quarterly rebalance)
        print("Running 60/40 benchmark...")
        benchmarks['60/40 Portfolio'] = self.backtest_benchmark(
            etf_data, {'SPY': 0.6, 'TLT': 0.4}, rebalance_freq='Q'
        )

        # 70/30 Portfolio
        print("Running 70/30 benchmark...")
        benchmarks['70/30 Portfolio'] = self.backtest_benchmark(
            etf_data, {'SPY': 0.7, 'TLT': 0.3}, rebalance_freq='Q'
        )

        # 100% Stocks
        print("Running 100% stocks benchmark...")
        benchmarks['100% Stocks'] = self.backtest_benchmark(
            etf_data, {'SPY': 1.0}, rebalance_freq=None
        )

        # Risk Parity (simplified)
        print("Running risk parity benchmark...")
        benchmarks['Risk Parity'] = self.backtest_benchmark(
            etf_data, {'SPY': 0.4, 'TLT': 0.4, 'GLD': 0.2}, rebalance_freq='Q'
        )

        # Step 5: Calculate metrics
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)

        results_table = []

        # χ-Strategy metrics
        strategy_metrics = self.calculate_metrics(strategy_results['portfolio_value'])
        strategy_metrics['Strategy'] = 'χ-Threshold'
        results_table.append(strategy_metrics)

        # Benchmark metrics
        for name, series in benchmarks.items():
            metrics = self.calculate_metrics(series)
            metrics['Strategy'] = name
            results_table.append(metrics)

        results_df = pd.DataFrame(results_table)
        results_df = results_df[['Strategy', 'Total Return', 'CAGR', 'Volatility',
                                 'Sharpe Ratio', 'Max Drawdown']]
        print("\n", results_df.to_string(index=False))

        # Step 6: Crisis analysis
        print("\n" + "="*60)
        print("CRISIS PERIOD ANALYSIS")
        print("="*60)

        crisis_results = self.analyze_crisis_periods(strategy_results, benchmarks)
        crisis_df = pd.DataFrame(crisis_results).T
        print("\nReturns during crisis periods (%):")
        print(crisis_df.to_string())

        # Step 7: Plot results
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        self.plot_results(strategy_results, benchmarks, self.chi_series, rebalance_dates)

        # Step 8: Sensitivity analysis
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS")
        print("="*60)

        self.run_sensitivity_analysis(stock_returns, etf_data)

        # Final summary
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        print(f"\nTotal rebalances: {len(rebalance_dates)}")
        print(f"Average time between rebalances: {len(strategy_results) / len(rebalance_dates):.0f} days")

        regime_dist = strategy_results['regime'].value_counts(normalize=True) * 100
        print("\nTime spent in each regime:")
        for regime, pct in regime_dist.items():
            print(f"  {regime:12s}: {pct:5.1f}%")

        return strategy_results, benchmarks, results_df, crisis_df

    def run_sensitivity_analysis(self, stock_returns, etf_data):
        """Test sensitivity to key parameters."""

        # Test 1: Different χ thresholds
        print("\n1. Testing χ threshold sensitivity (±10%):")

        threshold_tests = [
            {'name': 'Lower -10%', 'mult': 0.9},
            {'name': 'Base', 'mult': 1.0},
            {'name': 'Higher +10%', 'mult': 1.1},
        ]

        for test in threshold_tests:
            # Temporarily adjust thresholds
            original_thresholds = self.thresholds.copy()
            self.thresholds = {k: v * test['mult'] for k, v in original_thresholds.items()}

            # Run backtest
            chi_series = self.calculate_chi(stock_returns) if test['mult'] == 1.0 else self.chi_series
            results, _ = self.backtest_strategy(chi_series, etf_data)

            final_value = results['portfolio_value'].iloc[-1]
            total_return = (final_value / 100000 - 1) * 100

            print(f"  {test['name']:15s}: {total_return:7.2f}% total return")

            # Restore thresholds
            self.thresholds = original_thresholds

        # Test 2: Different window sizes
        print("\n2. Testing correlation window size:")

        window_tests = [10, 20, 30, 60]
        for window in window_tests:
            original_window = self.window
            self.window = window

            chi_series = self.calculate_chi(stock_returns)
            results, _ = self.backtest_strategy(chi_series, etf_data)

            final_value = results['portfolio_value'].iloc[-1]
            total_return = (final_value / 100000 - 1) * 100

            print(f"  {window:3d}-day window: {total_return:7.2f}% total return")

            self.window = original_window

        print("\n3. Number of stocks impact:")
        print("  (Using 50 stocks for this analysis)")
        print("  Note: More stocks = more stable χ, but slower calculation")


def main():
    """Main execution function."""

    # Check if yfinance is available
    if not HAS_YFINANCE:
        print("\n" + "="*60)
        print("ERROR: yfinance not installed")
        print("="*60)
        print("\nTo run this backtest, install yfinance:")
        print("  pip install yfinance")
        print("\nOr install with other dependencies:")
        print("  pip install yfinance matplotlib seaborn pandas numpy")
        return

    # Run backtest
    backtest = ChiThresholdBacktest(
        start_date='2000-01-01',
        end_date='2024-10-31',
        n_stocks=50,
        window=20,
        transaction_cost=0.001
    )

    try:
        strategy_results, benchmarks, metrics_df, crisis_df = backtest.run_full_analysis()

        # Save results to CSV
        print("\nSaving results to CSV...")
        metrics_df.to_csv('/home/user/Proofpacket/chi_backtest_metrics.csv', index=False)
        crisis_df.to_csv('/home/user/Proofpacket/chi_backtest_crisis.csv')
        strategy_results.to_csv('/home/user/Proofpacket/chi_strategy_daily.csv')

        print("\n" + "="*60)
        print("FILES CREATED")
        print("="*60)
        print("  1. chi_backtest_results.png     - Comprehensive visualization")
        print("  2. chi_backtest_metrics.csv      - Performance metrics table")
        print("  3. chi_backtest_crisis.csv       - Crisis period analysis")
        print("  4. chi_strategy_daily.csv        - Daily portfolio values")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
