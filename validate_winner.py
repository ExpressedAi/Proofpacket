"""
Validate WINNER configuration with corrected metrics
"""

from delta_trading_system import DeltaTradingSystem
from consensus_detector import ConsensusDetector
from chi_crash_detector import ChiCrashDetector
from historical_backtest import HistoricalBacktest

print("="*70)
print("VALIDATING WINNER CONFIGURATION")
print("="*70)

print("\nConfiguration: R*=1.0, χ_crisis=2.5")
print("  - Very aggressive consensus threshold")
print("  - Stay invested longer during volatility")
print("  - Expected: ~$223K final value, ~13-15% CAGR\n")

# Create backtest
backtest = HistoricalBacktest(
    start_date="2000-01-01",
    end_date="2024-12-31",
    initial_capital=100000,
    universe_size=50
)

# Set winner parameters
backtest.system.layer1_consensus = ConsensusDetector(
    R_star=1.0,
    h_threshold=0.15,
    eps_threshold=0.03
)

backtest.system.layer2_chi = ChiCrashDetector(
    flux_window=5,
    dissipation_window=20,
    regime_lag=3,
    crisis_threshold=2.5
)

# Run backtest
result = backtest.run_backtest()

# Print results
backtest.print_results(result)

print("\n" + "="*70)
print("COMPARISON TO SPY")
print("="*70)

spy_cagr = 7.8
spy_total = 549.8
spy_dd = -55.0

print(f"\nSPY (Buy & Hold):")
print(f"  CAGR:         {spy_cagr:7.2f}%")
print(f"  Total Return: {spy_total:7.2f}%")
print(f"  Max Drawdown: {spy_dd:7.2f}%")
print(f"  $100K → ${100000 * (1 + spy_total/100):,.0f}")

print(f"\nOur System:")
print(f"  CAGR:         {result.cagr*100:7.2f}%")
print(f"  Total Return: {result.total_return*100:7.2f}%")
print(f"  Max Drawdown: {result.max_drawdown*100:7.2f}%")
print(f"  $100K → ${result.equity_curve[-1]:,.0f}")

advantage = result.cagr * 100 - spy_cagr
print(f"\n🏆 CAGR Advantage: {advantage:+.2f} percentage points!")

if advantage > 0:
    print(f"✓ BEATING THE MARKET!")
else:
    print(f"⚠ Underperforming (but better risk-adjusted?)")

print("\n" + "="*70)
