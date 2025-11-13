"""
TUR (Thermodynamic Uncertainty Relation) Execution Optimizer
Layer 4: Optimal Rebalancing Frequency for Δ-Trading System

This module finds the optimal trade frequency by maximizing precision-per-entropy.

Key Insight from Statistical Physics:
    TUR: P / Σ ≤ 1/2

    where:
        P = precision = (signal)² / (noise)²
        Σ = entropy production = total transaction costs

Most quant funds OVERTRADE (maximize signal but ignore costs).
TUR finds the sweet spot: maximize information extracted per unit cost.

Example:
    1-minute rebalancing:
        P = 100 (very precise signal)
        Σ = 10,000 (high costs due to frequent trading)
        P/Σ = 0.01 (inefficient!)

    1-day rebalancing:
        P = 80 (still good signal)
        Σ = 100 (low costs)
        P/Σ = 0.80 (much better!)

    Optimal frequency maximizes P/Σ while respecting TUR bound.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum


class TradeFrequency(Enum):
    """Standard rebalancing frequencies."""
    TICK = "tick"           # Every tick (seconds) - very high frequency
    MINUTE_1 = "1min"       # Every minute
    MINUTE_5 = "5min"       # Every 5 minutes
    MINUTE_15 = "15min"     # Every 15 minutes
    HOUR_1 = "1hour"        # Hourly
    HOUR_4 = "4hour"        # Every 4 hours
    DAILY = "daily"         # Daily
    WEEKLY = "weekly"       # Weekly
    MONTHLY = "monthly"     # Monthly


@dataclass
class FrequencyMetrics:
    """Metrics for a specific rebalancing frequency."""
    frequency: TradeFrequency
    trades_per_year: int
    signal_strength: float      # Signal magnitude
    signal_noise: float         # Noise magnitude
    precision: float            # P = signal² / noise²
    entropy: float              # Σ = transaction costs
    efficiency: float           # P / Σ (what we maximize)
    tur_ratio: float            # P / Σ / 0.5 (should be ≤ 1.0)
    is_tur_valid: bool          # True if TUR bound satisfied

    def __str__(self):
        return (f"{self.frequency.value:8s}: P={self.precision:6.2f}, "
                f"Σ={self.entropy:8.0f}, P/Σ={self.efficiency:.4f}, "
                f"TUR={self.tur_ratio:.2f}")


@dataclass
class OptimalFrequency:
    """Result of TUR optimization."""
    optimal: FrequencyMetrics
    all_frequencies: List[FrequencyMetrics]
    improvement_vs_daily: float  # % improvement over daily rebalancing


class TUROptimizer:
    """
    Find optimal rebalancing frequency using TUR principle.

    The optimizer:
        1. Tests multiple frequencies (1min to monthly)
        2. Computes signal persistence at each frequency
        3. Computes transaction costs at each frequency
        4. Finds frequency that maximizes P/Σ
        5. Validates TUR bound (P/Σ ≤ 0.5)
    """

    def __init__(
        self,
        cost_per_trade_bps: float = 10.0,  # Transaction cost (basis points)
        slippage_bps: float = 5.0,          # Slippage (basis points)
        position_size: float = 100000,      # Typical position size ($)
    ):
        """
        Args:
            cost_per_trade_bps: Transaction cost in basis points (10 bps = 0.1%)
            slippage_bps: Slippage in basis points
            position_size: Typical position size for cost calculation
        """
        self.cost_per_trade_bps = cost_per_trade_bps
        self.slippage_bps = slippage_bps
        self.position_size = position_size

        # Total cost per trade (bps)
        self.total_cost_bps = cost_per_trade_bps + slippage_bps

        # Convert to dollar cost
        self.cost_per_trade = (self.total_cost_bps / 10000) * position_size

    def get_trades_per_year(self, frequency: TradeFrequency) -> int:
        """
        Get number of trades per year for a given frequency.

        Assumptions:
            - 252 trading days per year
            - 6.5 hours per trading day
        """
        trades_map = {
            TradeFrequency.TICK: 252 * 6.5 * 3600,     # ~5.9M trades/year
            TradeFrequency.MINUTE_1: 252 * 6.5 * 60,   # 98,280 trades/year
            TradeFrequency.MINUTE_5: 252 * 6.5 * 12,   # 19,656 trades/year
            TradeFrequency.MINUTE_15: 252 * 6.5 * 4,   # 6,552 trades/year
            TradeFrequency.HOUR_1: 252 * 6.5,          # 1,638 trades/year
            TradeFrequency.HOUR_4: 252 * 6.5 / 4,      # 410 trades/year
            TradeFrequency.DAILY: 252,                 # 252 trades/year
            TradeFrequency.WEEKLY: 52,                 # 52 trades/year
            TradeFrequency.MONTHLY: 12,                # 12 trades/year
        }
        return int(trades_map[frequency])

    def compute_signal_at_frequency(
        self,
        frequency: TradeFrequency,
        base_signal: float = 1.0,
        decay_rate: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute signal strength and noise at a given frequency.

        Signal decays with time (phase-locks drift).
        Noise increases with sampling frequency (more observations = more noise).

        Args:
            frequency: Rebalancing frequency
            base_signal: Base signal strength (at daily frequency)
            decay_rate: Signal decay rate (per day)

        Returns:
            (signal_strength, noise_strength)
        """
        # Time between trades (in days)
        trades_per_year = self.get_trades_per_year(frequency)
        days_between_trades = 252 / trades_per_year

        # Signal decays exponentially: S(t) = S₀ · exp(-λt)
        signal = base_signal * np.exp(-decay_rate * days_between_trades)

        # Noise increases with frequency (more samples = more noise)
        # N(f) ∝ sqrt(f)
        noise_scaling = np.sqrt(trades_per_year / 252)  # Normalized to daily
        base_noise = 0.2  # Base noise level
        noise = base_noise * noise_scaling

        return signal, noise

    def compute_precision(self, signal: float, noise: float) -> float:
        """
        Compute precision P = (signal)² / (noise)².

        High precision = strong signal, low noise.
        """
        if noise == 0:
            return np.inf

        P = (signal ** 2) / (noise ** 2)

        return P

    def compute_entropy(self, frequency: TradeFrequency) -> float:
        """
        Compute entropy production Σ = total transaction costs per year.

        Σ = (cost per trade) × (trades per year)
        """
        trades_per_year = self.get_trades_per_year(frequency)
        entropy = self.cost_per_trade * trades_per_year

        return entropy

    def compute_tur_ratio(self, precision: float, entropy: float) -> float:
        """
        Compute TUR ratio: (P/Σ) / 0.5

        TUR bound: P/Σ ≤ 0.5
        So TUR ratio should be ≤ 1.0

        Returns:
            ratio ∈ [0, ∞) where:
                < 1.0 = respects TUR bound
                ≥ 1.0 = violates TUR bound (theoretically impossible)
        """
        if entropy == 0:
            return np.inf

        efficiency = precision / entropy
        tur_ratio = efficiency / 0.5

        return tur_ratio

    def evaluate_frequency(
        self,
        frequency: TradeFrequency,
        signal_fn: Optional[Callable[[TradeFrequency], Tuple[float, float]]] = None
    ) -> FrequencyMetrics:
        """
        Evaluate metrics for a single frequency.

        Args:
            frequency: Rebalancing frequency to evaluate
            signal_fn: Optional custom signal function (freq -> (signal, noise))

        Returns:
            FrequencyMetrics with all computed values
        """
        trades_per_year = self.get_trades_per_year(frequency)

        # Compute signal and noise
        if signal_fn is not None:
            signal, noise = signal_fn(frequency)
        else:
            signal, noise = self.compute_signal_at_frequency(frequency)

        # Precision
        precision = self.compute_precision(signal, noise)

        # Entropy
        entropy = self.compute_entropy(frequency)

        # Efficiency (what we maximize)
        efficiency = precision / entropy if entropy > 0 else 0

        # TUR ratio (should be ≤ 1.0)
        tur_ratio = self.compute_tur_ratio(precision, entropy)
        is_tur_valid = tur_ratio <= 1.0

        return FrequencyMetrics(
            frequency=frequency,
            trades_per_year=trades_per_year,
            signal_strength=signal,
            signal_noise=noise,
            precision=precision,
            entropy=entropy,
            efficiency=efficiency,
            tur_ratio=tur_ratio,
            is_tur_valid=is_tur_valid
        )

    def find_optimal_frequency(
        self,
        frequencies: Optional[List[TradeFrequency]] = None,
        signal_fn: Optional[Callable[[TradeFrequency], Tuple[float, float]]] = None
    ) -> OptimalFrequency:
        """
        Find optimal rebalancing frequency that maximizes P/Σ.

        Args:
            frequencies: List of frequencies to test (default: all standard frequencies)
            signal_fn: Optional custom signal function

        Returns:
            OptimalFrequency with best choice and all results
        """
        if frequencies is None:
            # Test all standard frequencies
            frequencies = [
                TradeFrequency.MINUTE_1,
                TradeFrequency.MINUTE_5,
                TradeFrequency.MINUTE_15,
                TradeFrequency.HOUR_1,
                TradeFrequency.HOUR_4,
                TradeFrequency.DAILY,
                TradeFrequency.WEEKLY,
                TradeFrequency.MONTHLY,
            ]

        # Evaluate all frequencies
        results = []
        for freq in frequencies:
            metrics = self.evaluate_frequency(freq, signal_fn=signal_fn)
            results.append(metrics)

        # Find optimal (max P/Σ)
        optimal = max(results, key=lambda x: x.efficiency)

        # Compute improvement vs daily
        daily_metrics = [r for r in results if r.frequency == TradeFrequency.DAILY]
        if daily_metrics:
            daily_efficiency = daily_metrics[0].efficiency
            if daily_efficiency > 0:
                improvement = ((optimal.efficiency - daily_efficiency) / daily_efficiency) * 100
            else:
                improvement = 0.0
        else:
            improvement = 0.0

        return OptimalFrequency(
            optimal=optimal,
            all_frequencies=results,
            improvement_vs_daily=improvement
        )

    def get_recommendation(
        self,
        strategy_type: str = "phase_lock"
    ) -> OptimalFrequency:
        """
        Get frequency recommendation for a specific strategy type.

        Args:
            strategy_type: Type of strategy:
                - "phase_lock": Phase-lock based trading (medium persistence)
                - "momentum": Momentum strategy (high persistence)
                - "mean_reversion": Mean reversion (low persistence)
                - "high_freq": High frequency (very low persistence)

        Returns:
            OptimalFrequency recommendation
        """
        # Strategy-specific signal functions
        def phase_lock_signal(freq: TradeFrequency) -> Tuple[float, float]:
            """Phase-locks persist for hours to days."""
            return self.compute_signal_at_frequency(
                freq, base_signal=1.0, decay_rate=0.1
            )

        def momentum_signal(freq: TradeFrequency) -> Tuple[float, float]:
            """Momentum persists for days to weeks."""
            return self.compute_signal_at_frequency(
                freq, base_signal=1.2, decay_rate=0.05
            )

        def mean_reversion_signal(freq: TradeFrequency) -> Tuple[float, float]:
            """Mean reversion is fast (minutes to hours)."""
            return self.compute_signal_at_frequency(
                freq, base_signal=0.8, decay_rate=0.5
            )

        def high_freq_signal(freq: TradeFrequency) -> Tuple[float, float]:
            """HFT signals decay very fast (seconds)."""
            return self.compute_signal_at_frequency(
                freq, base_signal=0.5, decay_rate=2.0
            )

        signal_fns = {
            "phase_lock": phase_lock_signal,
            "momentum": momentum_signal,
            "mean_reversion": mean_reversion_signal,
            "high_freq": high_freq_signal,
        }

        signal_fn = signal_fns.get(strategy_type, phase_lock_signal)

        return self.find_optimal_frequency(signal_fn=signal_fn)


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("TUR Execution Optimizer - Example Usage")
    print("=" * 70)

    # Create optimizer
    optimizer = TUROptimizer(
        cost_per_trade_bps=10.0,  # 10 bps = 0.1% round-trip
        slippage_bps=5.0,
        position_size=100000
    )

    print(f"\nConfiguration:")
    print(f"  Transaction cost: {optimizer.cost_per_trade_bps} bps")
    print(f"  Slippage: {optimizer.slippage_bps} bps")
    print(f"  Total cost: {optimizer.total_cost_bps} bps = ${optimizer.cost_per_trade:.2f}/trade")
    print(f"  Position size: ${optimizer.position_size:,.0f}")

    # Test different strategy types
    strategies = [
        ("phase_lock", "Phase-Lock Trading (Δ-system)"),
        ("momentum", "Momentum Strategy"),
        ("mean_reversion", "Mean Reversion"),
    ]

    print("\n" + "=" * 70)
    print("Strategy-Specific Optimization")
    print("=" * 70)

    for strategy_type, strategy_name in strategies:
        print(f"\n{strategy_name}:")
        print("-" * 70)

        result = optimizer.get_recommendation(strategy_type=strategy_type)

        # Show all frequencies
        print(f"\nAll Frequencies (sorted by efficiency):")
        sorted_results = sorted(
            result.all_frequencies,
            key=lambda x: x.efficiency,
            reverse=True
        )

        for i, metrics in enumerate(sorted_results[:5], 1):
            marker = "★" if metrics == result.optimal else " "
            print(f"  {marker} {i}. {metrics}")

        # Optimal recommendation
        print(f"\n  Optimal Frequency: {result.optimal.frequency.value.upper()}")
        print(f"  Trades/Year: {result.optimal.trades_per_year:,}")
        print(f"  Efficiency P/Σ: {result.optimal.efficiency:.6f}")
        print(f"  TUR Ratio: {result.optimal.tur_ratio:.3f} "
              f"({'VALID' if result.optimal.is_tur_valid else 'INVALID'})")
        print(f"  Improvement vs Daily: {result.improvement_vs_daily:+.1f}%")

    # Deep dive: Phase-lock strategy
    print("\n" + "=" * 70)
    print("Deep Dive: Phase-Lock Strategy (Our System)")
    print("=" * 70)

    result = optimizer.get_recommendation("phase_lock")

    print(f"\nComplete Frequency Analysis:")
    print(f"{'Frequency':<12} {'Trades/Yr':<12} {'Signal':<8} {'Noise':<8} "
          f"{'P':<8} {'Σ ($)':<10} {'P/Σ':<10} {'TUR':<6}")
    print("-" * 70)

    for metrics in sorted(result.all_frequencies, key=lambda x: x.trades_per_year, reverse=True):
        print(f"{metrics.frequency.value:<12} "
              f"{metrics.trades_per_year:<12,} "
              f"{metrics.signal_strength:<8.3f} "
              f"{metrics.signal_noise:<8.3f} "
              f"{metrics.precision:<8.2f} "
              f"{metrics.entropy:<10,.0f} "
              f"{metrics.efficiency:<10.6f} "
              f"{metrics.tur_ratio:<6.3f}")

    # Cost analysis
    print(f"\nCost Analysis (Optimal: {result.optimal.frequency.value}):")
    print(f"  Trades per year: {result.optimal.trades_per_year:,}")
    print(f"  Cost per trade: ${result.optimal.entropy / result.optimal.trades_per_year:.2f}")
    print(f"  Total annual cost: ${result.optimal.entropy:,.0f}")
    print(f"  % of capital (per $100k): {(result.optimal.entropy / 100000) * 100:.2f}%")

    # Comparison to common mistakes
    print(f"\n" + "=" * 70)
    print("Common Mistakes: Why Most Traders Overtrade")
    print("=" * 70)

    minute_1 = [m for m in result.all_frequencies if m.frequency == TradeFrequency.MINUTE_1][0]
    optimal = result.optimal

    print(f"\n1-Minute Rebalancing (Overtrading):")
    print(f"  Signal: {minute_1.signal_strength:.3f}")
    print(f"  Precision: {minute_1.precision:.2f}")
    print(f"  Annual cost: ${minute_1.entropy:,.0f}")
    print(f"  Efficiency P/Σ: {minute_1.efficiency:.6f}")

    print(f"\nOptimal ({optimal.frequency.value}):")
    print(f"  Signal: {optimal.signal_strength:.3f} (only {((1 - optimal.signal_strength/minute_1.signal_strength)*100):.1f}% weaker)")
    print(f"  Precision: {optimal.precision:.2f}")
    print(f"  Annual cost: ${optimal.entropy:,.0f} ({((1 - optimal.entropy/minute_1.entropy)*100):.1f}% cheaper!)")
    print(f"  Efficiency P/Σ: {optimal.efficiency:.6f} ({(optimal.efficiency/minute_1.efficiency):.0f}x better!)")

    print(f"\nKey Insight:")
    print(f"  By reducing frequency from 1-min to {optimal.frequency.value}, you:")
    print(f"  • Save {((1 - optimal.entropy/minute_1.entropy)*100):.1f}% on transaction costs")
    print(f"  • Only lose {((1 - optimal.signal_strength/minute_1.signal_strength)*100):.1f}% of signal strength")
    print(f"  • Increase efficiency by {((optimal.efficiency/minute_1.efficiency - 1)*100):.0f}%")

    print("\n" + "=" * 70)
    print("✓ Day 4 implementation complete")
    print("  TUR optimizer ready for integration")
    print(f"  Recommendation: {result.optimal.frequency.value.upper()} rebalancing")
    print("  Next: Day 5 - Four-layer system integration")
    print("=" * 70)
