"""
Diagnose why consensus detector isn't generating signals
"""

import numpy as np
import pandas as pd
from datetime import datetime
from consensus_detector import ConsensusDetector, MarketState

print("=" * 70)
print("CONSENSUS DETECTOR DIAGNOSTIC")
print("=" * 70)

# Load SPY data
spy_df = pd.read_csv('spy_historical.csv', parse_dates=['Date'])
spy_df.set_index('Date', inplace=True)
spy_df = spy_df[spy_df.index >= datetime(2000, 1, 1)]
spy_df = spy_df[spy_df.index <= datetime(2024, 12, 31)]

print(f"\nLoaded {len(spy_df)} days of SPY data")
print(f"Date range: {spy_df.index[0]} to {spy_df.index[-1]}")

# Create synthetic universe (same as backtest)
universe_size = 50
tickers = [f"STOCK{i:02d}" for i in range(universe_size)]

spy_returns = spy_df['Close'].pct_change().fillna(0).values

stocks_data = {}
for i, ticker in enumerate(tickers):
    beta = 0.5 + np.random.random() * 1.5
    idio_vol = 0.005 + np.random.random() * 0.015

    stock_returns = np.zeros(len(spy_returns))
    for day in range(len(spy_returns)):
        market_component = beta * spy_returns[day]
        idio_component = np.random.normal(0, idio_vol)
        stock_returns[day] = market_component + idio_component

    start_price = 50 + np.random.random() * 150
    price = start_price * np.exp(np.cumsum(stock_returns))

    stocks_data[ticker] = {
        'returns': stock_returns,
        'price': price,
        'beta': beta
    }

print(f"\nGenerated {len(stocks_data)} stocks")

# Initialize consensus detector
detector = ConsensusDetector(R_star=3.5)

# Test on different windows
windows = [20, 50, 100, 250]  # 1 month, ~2 months, ~4 months, 1 year

for window in windows:
    print(f"\n" + "=" * 70)
    print(f"Testing window = {window} days")
    print("=" * 70)

    signals_found = 0

    # Test multiple periods
    test_periods = [
        (250, "2001 Q1"),
        (1250, "2005 Q1"),
        (2500, "2010 Q1"),
        (3750, "2015 Q1"),
        (5000, "2020 Q1"),
    ]

    for start_idx, label in test_periods:
        if start_idx + window >= len(spy_returns):
            continue

        # Create market state for one stock
        ticker = "STOCK00"
        stock = stocks_data[ticker]

        returns = stock['returns'][start_idx:start_idx + window]
        prices = stock['price'][start_idx:start_idx + window]

        # Create correlation matrix (simplified - just use returns)
        # In real backtest, this would be cross-stock correlations
        correlation = np.corrcoef(
            returns[:-10],
            returns[10:]
        )[0, 1] if len(returns) > 10 else 0.5

        state = MarketState(
            ticker=ticker,
            price=prices[-1],
            returns=returns,
            volume=np.ones(len(returns)) * 1e6,  # Dummy volume
            correlation_matrix=np.eye(universe_size) * 0.5 + 0.5,  # Dummy correlation
            volatility=np.std(returns),
            momentum=np.mean(returns[-20:]) if len(returns) >= 20 else 0,
        )

        # Check consensus
        result = detector.check_consensus(state)

        if result.consensus:
            signals_found += 1
            print(f"  ✓ {label}: SIGNAL! R={result.redundancy:.2f}, "
                  f"dR/dt={result.redundancy_rate:.4f}, strength={result.strength:.1%}")
        else:
            print(f"    {label}: No signal. R={result.redundancy:.2f}, "
                  f"dR/dt={result.redundancy_rate:.4f}")

    print(f"\nSignals found with window={window}: {signals_found}")

print("\n" + "=" * 70)
print("TESTING WITH LOWER THRESHOLDS")
print("=" * 70)

# Test with progressively lower R thresholds
thresholds = [3.5, 3.0, 2.5, 2.0, 1.5, 1.0]

for R_thresh in thresholds:
    detector_test = ConsensusDetector(R_star=R_thresh)
    signals = 0

    # Test across all test periods with window=100
    for start_idx, label in [(250, "2001"), (2500, "2010"), (5000, "2020")]:
        if start_idx + 100 >= len(spy_returns):
            continue

        ticker = "STOCK00"
        stock = stocks_data[ticker]
        returns = stock['returns'][start_idx:start_idx + 100]
        prices = stock['price'][start_idx:start_idx + 100]

        state = MarketState(
            ticker=ticker,
            price=prices[-1],
            returns=returns,
            volume=np.ones(len(returns)) * 1e6,
            correlation_matrix=np.eye(universe_size) * 0.5 + 0.5,
            volatility=np.std(returns),
            momentum=np.mean(returns[-20:]) if len(returns) >= 20 else 0,
        )

        result = detector_test.check_consensus(state)
        if result.consensus:
            signals += 1

    print(f"  R_threshold={R_thresh}: {signals}/3 signals found")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
