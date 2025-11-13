"""
ΔH* Calculator: Evidence Scoring for Locks and Strategies

ΔH* = Evidence gain from a lock/strategy over baseline nulls.

Replaces arbitrary Sharpe ratios with physics-grounded metric:
- How much does this lock improve predictive power?
- Does it beat domain-specific nulls?
- Is the gain stable across RG transforms?

Formula (conceptual):
    ΔH* = log(P(data|lock)) - log(P(data|null))

Practical implementations:
- Per-trade: Realized P&L vs expected from null
- Per-window: Information gain from lock existence
- Aggregate: Running sum of ΔH* over time
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime
import numpy as np
from scipy import stats

from delta_state_v2 import LockState, StrategyState, DeltaState


# ============================================================================
# ΔH* Result Structure
# ============================================================================

@dataclass
class DeltaHResult:
    """Result of ΔH* calculation."""
    lock_id: str
    strategy_id: Optional[str] = None

    # Evidence gain
    delta_h_star: float = 0.0  # Main metric

    # Breakdown
    signal_strength: float = 0.0  # How strong is the lock?
    null_baseline: float = 0.0    # What would null achieve?
    predictive_gain: float = 0.0  # Signal - null

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    window_size: int = 0
    n_trades: int = 0
    notes: str = ""


# ============================================================================
# Per-Trade ΔH* (Realized)
# ============================================================================

def compute_trade_delta_h(
    actual_pnl: float,
    expected_pnl_null: float,
    trade_cost: float = 0.0
) -> float:
    """
    Per-trade ΔH*: Realized evidence gain.

    ΔH* = (actual - null) / |null|

    Positive if trade beats null expectation.
    Negative if trade underperforms null.

    Args:
        actual_pnl: Actual P&L from trade
        expected_pnl_null: Expected P&L from null model (e.g., random)
        trade_cost: Transaction costs (spreads, fees)

    Returns:
        ΔH* ∈ [-∞, +∞], typically [-2, +2]
    """
    net_pnl = actual_pnl - trade_cost

    # Avoid division by zero
    if abs(expected_pnl_null) < 1e-6:
        expected_pnl_null = 1e-6

    delta_h = (net_pnl - expected_pnl_null) / abs(expected_pnl_null)

    return delta_h


def compute_batch_trade_delta_h(
    actual_pnls: np.ndarray,
    null_pnls: np.ndarray
) -> Tuple[float, float]:
    """
    Batch ΔH* for multiple trades.

    Returns (mean_delta_h, std_delta_h).
    """
    delta_hs = []

    for actual, null in zip(actual_pnls, null_pnls):
        dh = compute_trade_delta_h(actual, null)
        delta_hs.append(dh)

    delta_hs = np.array(delta_hs)

    return np.mean(delta_hs), np.std(delta_hs)


# ============================================================================
# Per-Window ΔH* (Expected)
# ============================================================================

def compute_window_delta_h(
    lock_signal: np.ndarray,
    null_signals: List[np.ndarray],
    returns: np.ndarray
) -> DeltaHResult:
    """
    Per-window ΔH*: Expected evidence gain from lock.

    Measures how much lock signal improves return prediction
    vs null signals (phase-shuffled, vol-matched, etc.).

    Args:
        lock_signal: (N,) Lock strength over window
        null_signals: List of (N,) null surrogate signals
        returns: (N,) Actual returns

    Returns:
        DeltaHResult with ΔH* and breakdown
    """
    n = len(lock_signal)

    # 1. Signal strength: Correlation between lock and returns
    if np.std(lock_signal) > 1e-6 and np.std(returns) > 1e-6:
        signal_corr = np.corrcoef(lock_signal, returns)[0, 1]
    else:
        signal_corr = 0.0

    signal_strength = abs(signal_corr)

    # 2. Null baseline: Average correlation of nulls with returns
    null_corrs = []
    for null_signal in null_signals:
        if np.std(null_signal) > 1e-6:
            null_corr = np.corrcoef(null_signal, returns)[0, 1]
            null_corrs.append(abs(null_corr))

    if null_corrs:
        null_baseline = np.mean(null_corrs)
    else:
        null_baseline = 0.0

    # 3. Predictive gain
    predictive_gain = signal_strength - null_baseline

    # 4. ΔH* = log-odds improvement
    # If signal strength >> null, ΔH* is large and positive
    # If signal strength ≤ null, ΔH* is negative

    # Use log-ratio to get scale-invariant metric
    if null_baseline > 1e-6:
        delta_h_star = np.log(max(signal_strength, 1e-6) / null_baseline)
    else:
        # No null baseline, just use signal strength
        delta_h_star = signal_strength

    return DeltaHResult(
        lock_id="",  # Fill in by caller
        delta_h_star=delta_h_star,
        signal_strength=signal_strength,
        null_baseline=null_baseline,
        predictive_gain=predictive_gain,
        window_size=n,
        notes=f"Signal corr={signal_corr:.3f}, Null mean={null_baseline:.3f}"
    )


# ============================================================================
# Information-Theoretic ΔH* (KL Divergence)
# ============================================================================

def compute_kl_delta_h(
    lock_predictions: np.ndarray,
    null_predictions: np.ndarray,
    actual_outcomes: np.ndarray
) -> float:
    """
    ΔH* via KL divergence.

    ΔH* = KL(actual || null) - KL(actual || lock)

    Positive if lock predictions are closer to reality than null.

    Args:
        lock_predictions: (N,) Probabilities from lock model
        null_predictions: (N,) Probabilities from null model
        actual_outcomes: (N,) Binary outcomes (0 or 1)

    Returns:
        ΔH* ∈ [-∞, +∞]
    """
    # Clamp probabilities to avoid log(0)
    eps = 1e-10
    lock_predictions = np.clip(lock_predictions, eps, 1 - eps)
    null_predictions = np.clip(null_predictions, eps, 1 - eps)

    # KL divergence: sum of actual * log(actual / predicted)
    # For binary, this simplifies to cross-entropy
    kl_null = -np.sum(
        actual_outcomes * np.log(null_predictions) +
        (1 - actual_outcomes) * np.log(1 - null_predictions)
    )

    kl_lock = -np.sum(
        actual_outcomes * np.log(lock_predictions) +
        (1 - actual_outcomes) * np.log(1 - lock_predictions)
    )

    # ΔH* = reduction in KL
    delta_h = kl_null - kl_lock

    return delta_h


# ============================================================================
# Aggregate ΔH* (Running Sum)
# ============================================================================

def update_aggregate_delta_h(
    current_aggregate: float,
    new_delta_h: float,
    decay: float = 0.95
) -> float:
    """
    Update aggregate ΔH* with exponential decay.

    ΔH*_t = decay * ΔH*_{t-1} + (1 - decay) * ΔH*_new

    This gives more weight to recent evidence.

    Args:
        current_aggregate: Current ΔH* aggregate
        new_delta_h: New ΔH* measurement
        decay: Decay factor (0-1), higher = longer memory

    Returns:
        Updated ΔH*
    """
    return decay * current_aggregate + (1 - decay) * new_delta_h


# ============================================================================
# ΔH* for Lock (Over Historical Window)
# ============================================================================

def compute_lock_delta_h_historical(
    lock: LockState,
    asset_a_prices: np.ndarray,
    asset_b_prices: np.ndarray,
    n_null_surrogates: int = 50
) -> DeltaHResult:
    """
    Compute ΔH* for a lock over historical data.

    Process:
    1. Compute lock signal (e.g., phase difference)
    2. Generate null signals (phase-shuffled, etc.)
    3. Measure how well lock predicts relative returns
    4. Compare to null baseline

    Args:
        lock: LockState with p, q, K
        asset_a_prices: Historical prices for asset A
        asset_b_prices: Historical prices for asset B
        n_null_surrogates: Number of null surrogates

    Returns:
        DeltaHResult
    """
    # 1. Compute lock signal (simplified: just use K as proxy)
    # In real implementation, compute actual phase difference
    returns_a = np.diff(np.log(asset_a_prices))
    returns_b = np.diff(np.log(asset_b_prices))

    # Lock signal: Coupling strength modulated by relative returns
    lock_signal = lock.K * (returns_a[:-1] - returns_b[:-1])

    # Target: Predict next-period relative returns
    target_returns = returns_a[1:] - returns_b[1:]

    # 2. Generate null signals (phase-shuffled)
    null_signals = []
    for _ in range(n_null_surrogates):
        # Phase shuffle returns_a
        fft_a = np.fft.fft(returns_a)
        amplitudes_a = np.abs(fft_a)
        random_phases = np.random.uniform(0, 2 * np.pi, len(fft_a))
        null_fft = amplitudes_a * np.exp(1j * random_phases)
        null_returns_a = np.real(np.fft.ifft(null_fft))

        null_signal = lock.K * (null_returns_a[:-1] - returns_b[:-1])
        null_signals.append(null_signal)

    # 3. Compute ΔH*
    result = compute_window_delta_h(
        lock_signal=lock_signal,
        null_signals=null_signals,
        returns=target_returns
    )

    result.lock_id = lock.id
    result.n_trades = len(lock_signal)

    return result


# ============================================================================
# ΔH* for Strategy (Over Trade History)
# ============================================================================

def compute_strategy_delta_h(
    strategy: StrategyState,
    trade_pnls: np.ndarray,
    null_pnls: np.ndarray
) -> float:
    """
    Compute aggregate ΔH* for a strategy.

    Args:
        strategy: StrategyState
        trade_pnls: (N,) Realized P&Ls from strategy trades
        null_pnls: (N,) Expected P&Ls from null model

    Returns:
        Aggregate ΔH*
    """
    mean_delta_h, _ = compute_batch_trade_delta_h(trade_pnls, null_pnls)

    return mean_delta_h


# ============================================================================
# ΔH* Decay and Promotion Logic
# ============================================================================

def check_delta_h_promotion(
    current_delta_h: float,
    threshold_e1: float = 0.05,
    threshold_e2: float = 0.10,
    threshold_e3: float = 0.15
) -> int:
    """
    Determine E-level based on ΔH* threshold.

    E1: ΔH* > 0.05 (beats nulls)
    E2: ΔH* > 0.10 (RG-stable gain)
    E3: ΔH* > 0.15 (live-validated gain)

    Args:
        current_delta_h: Current ΔH*
        threshold_e1, threshold_e2, threshold_e3: Thresholds

    Returns:
        Max E-level (0-3) based on ΔH*
    """
    if current_delta_h >= threshold_e3:
        return 3
    elif current_delta_h >= threshold_e2:
        return 2
    elif current_delta_h >= threshold_e1:
        return 1
    else:
        return 0


def check_delta_h_demotion(
    current_delta_h: float,
    historical_delta_h: List[float],
    degradation_threshold: float = 0.5
) -> bool:
    """
    Check if ΔH* has degraded significantly.

    Returns True if current ΔH* is < degradation_threshold * historical mean.

    Args:
        current_delta_h: Current ΔH*
        historical_delta_h: Past ΔH* measurements
        degradation_threshold: Fraction of historical mean

    Returns:
        True if degraded, False otherwise
    """
    if not historical_delta_h:
        return False

    historical_mean = np.mean(historical_delta_h)

    if historical_mean <= 0:
        return False  # Can't degrade from zero

    degraded = current_delta_h < degradation_threshold * historical_mean

    return degraded


# ============================================================================
# Integration with DeltaState
# ============================================================================

def update_lock_delta_h(
    state: DeltaState,
    lock_id: str,
    new_delta_h: float,
    decay: float = 0.95
):
    """
    Update lock's evidence_score with new ΔH* measurement.

    Args:
        state: DeltaState
        lock_id: Lock ID
        new_delta_h: New ΔH* measurement
        decay: Decay factor for exponential averaging
    """
    if lock_id not in state.locks:
        state.add_log(f"⚠ Lock {lock_id} not found in state")
        return

    lock = state.locks[lock_id]

    # Update aggregate ΔH*
    lock.evidence_score = update_aggregate_delta_h(
        lock.evidence_score,
        new_delta_h,
        decay
    )

    # Update last_updated
    lock.last_updated = datetime.utcnow()

    state.add_log(f"Updated {lock_id}: ΔH*={lock.evidence_score:.3f} (new={new_delta_h:.3f})")


def update_strategy_delta_h(
    state: DeltaState,
    strategy_name: str,
    new_delta_h: float,
    decay: float = 0.95
):
    """
    Update strategy's evidence_score with new ΔH* measurement.

    Args:
        state: DeltaState
        strategy_name: Strategy name
        new_delta_h: New ΔH* measurement
        decay: Decay factor
    """
    if strategy_name not in state.strategies:
        state.add_log(f"⚠ Strategy {strategy_name} not found in state")
        return

    strategy = state.strategies[strategy_name]

    # Update aggregate ΔH*
    strategy.evidence_score = update_aggregate_delta_h(
        strategy.evidence_score,
        new_delta_h,
        decay
    )

    state.add_log(f"Updated {strategy_name}: ΔH*={strategy.evidence_score:.3f} (new={new_delta_h:.3f})")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ΔH* CALCULATOR: Evidence Scoring for Locks and Strategies")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Per-trade ΔH*
    print("\n[TEST 1: PER-TRADE ΔH*]")
    actual_pnl = 150.0
    expected_null = 50.0  # Null would have made $50
    trade_cost = 5.0

    delta_h_trade = compute_trade_delta_h(actual_pnl, expected_null, trade_cost)
    print(f"  Actual P&L: ${actual_pnl:.2f}")
    print(f"  Null baseline: ${expected_null:.2f}")
    print(f"  Trade cost: ${trade_cost:.2f}")
    print(f"  → ΔH* = {delta_h_trade:.3f}")

    # Test 2: Per-window ΔH*
    print("\n[TEST 2: PER-WINDOW ΔH*]")
    n = 100
    returns = np.random.randn(n) * 0.01

    # Lock signal: correlated with returns
    lock_signal = 0.5 * returns + 0.5 * np.random.randn(n) * 0.01

    # Null signals: phase-shuffled
    null_signals = []
    for _ in range(20):
        fft = np.fft.fft(returns)
        amplitudes = np.abs(fft)
        random_phases = np.random.uniform(0, 2 * np.pi, len(fft))
        null_fft = amplitudes * np.exp(1j * random_phases)
        null_signal = np.real(np.fft.ifft(null_fft))
        null_signals.append(null_signal)

    result = compute_window_delta_h(lock_signal, null_signals, returns)
    print(f"  Signal strength: {result.signal_strength:.3f}")
    print(f"  Null baseline: {result.null_baseline:.3f}")
    print(f"  Predictive gain: {result.predictive_gain:.3f}")
    print(f"  → ΔH* = {result.delta_h_star:.3f}")

    # Test 3: Aggregate ΔH*
    print("\n[TEST 3: AGGREGATE ΔH* WITH DECAY]")
    aggregate = 0.0
    measurements = [0.08, 0.12, 0.10, 0.15, 0.11]

    for i, measurement in enumerate(measurements):
        aggregate = update_aggregate_delta_h(aggregate, measurement, decay=0.9)
        print(f"  t={i}: new={measurement:.3f}, aggregate={aggregate:.3f}")

    # Test 4: Promotion logic
    print("\n[TEST 4: E-LEVEL PROMOTION FROM ΔH*]")
    test_values = [0.03, 0.08, 0.12, 0.18]
    for dh in test_values:
        e_level = check_delta_h_promotion(dh)
        print(f"  ΔH*={dh:.3f} → E{e_level}")

    # Test 5: Degradation check
    print("\n[TEST 5: DEGRADATION CHECK]")
    historical = [0.15, 0.18, 0.16, 0.14, 0.17]
    current = 0.08  # Degraded

    degraded = check_delta_h_demotion(current, historical, degradation_threshold=0.5)
    print(f"  Historical mean: {np.mean(historical):.3f}")
    print(f"  Current: {current:.3f}")
    print(f"  → Degraded: {degraded}")

    print("\n" + "=" * 70)
    print("ΔH* = Evidence gain that replaces arbitrary Sharpe")
    print("=" * 70)
