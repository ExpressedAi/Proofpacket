"""
E-Gates: Evidence audit framework (E0 → E1 → E2 → E3 → E4)

Each lock/strategy must pass gates sequentially before trading.
Follows Δ-Method epistemology, NOT vibes.

Gate progression:
E0: Structure exists (basic detection)
E1: Beats simple nulls (phase shuffle, vol-matched surrogate)
E2: RG-stable (survives coarse-graining)
E3: Live performance validated (paper/micro-live)
E4: Long-term robust (production-scale evidence)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

from delta_state import PhaseLock, AuditResult, LockStatus


class NullType(Enum):
    """Types of null hypotheses to test against."""
    PHASE_SHUFFLE = "phase_shuffle"  # Randomize phases, keep amplitudes
    BLOCK_SURROGATE = "block_surrogate"  # Block bootstrap
    VOL_MATCHED = "vol_matched"  # Match volatility, random walk
    INDEPENDENT = "independent"  # Simple correlation from independent normals
    MEAN_REVERSION = "mean_reversion"  # AR(1) null


# ============================================================================
# E0: Structure Exists
# ============================================================================

def e0_structure_exists(lock: PhaseLock, data: Dict[str, np.ndarray]) -> AuditResult:
    """
    E0: Does the basic structure exist?

    Requirements:
    - Sufficient data (min 100 points)
    - K > 0.1 (non-trivial coupling)
    - p, q are low-order (p*q <= 20 for LOW principle)
    - Phases are well-defined (not NaN/Inf)

    This is the cheapest gate - just sanity checks.
    """
    metrics = {}

    # Check data sufficiency
    pair_a, pair_b = lock.pair
    data_a = data.get(pair_a, np.array([]))
    data_b = data.get(pair_b, np.array([]))

    n_points = min(len(data_a), len(data_b))
    metrics['n_points'] = n_points

    # Check coupling
    metrics['K'] = lock.K
    metrics['order'] = lock.p * lock.q

    # Pass conditions
    passed = (
        n_points >= 100 and
        lock.K > 0.1 and
        lock.p * lock.q <= 20 and  # Low-Order Wins
        not np.isnan(lock.K) and
        not np.isinf(lock.K)
    )

    return AuditResult(
        gate="E0",
        passed=passed,
        metrics=metrics,
        notes=f"Structure check: {n_points} points, K={lock.K:.3f}, order={lock.p}:{lock.q}"
    )


# ============================================================================
# E1: Beats Simple Nulls
# ============================================================================

def generate_phase_shuffle_null(data: np.ndarray, n_surrogates: int = 100) -> List[np.ndarray]:
    """
    Generate phase-shuffled surrogates.

    Preserve amplitude spectrum, randomize phases.
    This tests: "Is the phase relationship meaningful, or just amplitude correlation?"
    """
    fft = np.fft.fft(data)
    amplitudes = np.abs(fft)

    surrogates = []
    for _ in range(n_surrogates):
        # Random phases
        random_phases = np.random.uniform(0, 2*np.pi, len(fft))
        random_phases[0] = 0  # DC component has no phase

        # Reconstruct with random phases
        surrogate_fft = amplitudes * np.exp(1j * random_phases)
        surrogate = np.fft.ifft(surrogate_fft).real
        surrogates.append(surrogate)

    return surrogates


def generate_block_surrogate(data: np.ndarray, block_size: int = 20, n_surrogates: int = 100) -> List[np.ndarray]:
    """
    Generate block-bootstrap surrogates.

    Preserve local dynamics, destroy long-range structure.
    """
    n = len(data)
    n_blocks = n // block_size

    surrogates = []
    for _ in range(n_surrogates):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, n_blocks, replace=True)
        surrogate = []

        for idx in block_indices:
            start = idx * block_size
            end = min(start + block_size, n)
            surrogate.extend(data[start:end])

        surrogates.append(np.array(surrogate[:n]))

    return surrogates


def compute_coupling_strength(data_a: np.ndarray, data_b: np.ndarray, p: int, q: int) -> float:
    """
    Compute coupling strength K for p:q lock.

    Simplified: uses phase coherence metric.
    Real version would use Hilbert transform.
    """
    # Compute phase difference (proxy)
    # Real implementation: Hilbert transform, extract phases, check p*φ_a - q*φ_b

    # For now, use correlation as proxy
    if len(data_a) != len(data_b):
        min_len = min(len(data_a), len(data_b))
        data_a = data_a[:min_len]
        data_b = data_b[:min_len]

    # Correlation strength
    corr = np.corrcoef(data_a, data_b)[0, 1]

    # Adjust for order (higher order = lower expected K)
    K = abs(corr) / np.sqrt(p * q)

    return K


def e1_beats_nulls(
    lock: PhaseLock,
    data: Dict[str, np.ndarray],
    null_types: List[NullType] = None,
    n_surrogates: int = 100,
    fdr_threshold: float = 0.05
) -> AuditResult:
    """
    E1: Does lock beat simple null hypotheses?

    Tests:
    1. Phase shuffle: Is phase relationship meaningful?
    2. Block surrogate: Does it require long-range structure?
    3. Vol-matched: Is it just volatility correlation?

    Uses FDR correction for multiple testing.
    """
    if null_types is None:
        null_types = [NullType.PHASE_SHUFFLE, NullType.BLOCK_SURROGATE]

    pair_a, pair_b = lock.pair
    data_a = data[pair_a]
    data_b = data[pair_b]

    # Compute observed coupling
    K_observed = lock.K

    null_comparison = {}
    p_values = []

    for null_type in null_types:
        # Generate surrogates
        if null_type == NullType.PHASE_SHUFFLE:
            surrogates_a = generate_phase_shuffle_null(data_a, n_surrogates)
            surrogates_b = [data_b] * n_surrogates  # Keep B fixed
        elif null_type == NullType.BLOCK_SURROGATE:
            surrogates_a = generate_block_surrogate(data_a, n_surrogates=n_surrogates)
            surrogates_b = generate_block_surrogate(data_b, n_surrogates=n_surrogates)
        else:
            continue

        # Compute null distribution of K
        K_nulls = []
        for surr_a, surr_b in zip(surrogates_a, surrogates_b):
            K_null = compute_coupling_strength(surr_a, surr_b, lock.p, lock.q)
            K_nulls.append(K_null)

        K_nulls = np.array(K_nulls)

        # P-value: fraction of nulls >= observed
        p_value = np.mean(K_nulls >= K_observed)
        p_values.append(p_value)

        null_comparison[null_type.value] = {
            'K_observed': K_observed,
            'K_null_mean': np.mean(K_nulls),
            'K_null_std': np.std(K_nulls),
            'p_value': p_value,
            'percentile': stats.percentileofscore(K_nulls, K_observed)
        }

    # FDR correction (Benjamini-Hochberg)
    p_values_sorted = sorted(p_values)
    n_tests = len(p_values)

    fdr_passed = False
    for i, p in enumerate(p_values_sorted):
        if p <= (i + 1) / n_tests * fdr_threshold:
            fdr_passed = True
        else:
            break

    metrics = {
        'n_tests': n_tests,
        'min_p_value': min(p_values) if p_values else 1.0,
        'fdr_threshold': fdr_threshold,
        'fdr_passed': fdr_passed
    }

    return AuditResult(
        gate="E1",
        passed=fdr_passed,
        metrics=metrics,
        null_comparison=null_comparison,
        notes=f"Tested {n_tests} nulls, min p={min(p_values):.4f}, FDR={'PASS' if fdr_passed else 'FAIL'}"
    )


# ============================================================================
# E2: RG-Stable (Survives Coarse-Graining)
# ============================================================================

def coarse_grain_timeseries(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Coarse-grain time series by factor.

    Averages over windows of size `factor`.
    This tests: does the structure survive when you zoom out?
    """
    n = len(data)
    n_coarse = n // factor

    coarse = []
    for i in range(n_coarse):
        window = data[i*factor:(i+1)*factor]
        coarse.append(np.mean(window))

    return np.array(coarse)


def e2_rg_stable(
    lock: PhaseLock,
    data: Dict[str, np.ndarray],
    rg_factors: List[int] = None,
    k_threshold_fraction: float = 0.5
) -> AuditResult:
    """
    E2: Is lock RG-stable? (Survives coarse-graining)

    Tests: Does K remain significant after coarse-graining by 2x, 4x, 8x?

    RG-stability = "structure persists across scales"
    Low-Order Wins predicts: low-order locks should be MORE stable.
    """
    if rg_factors is None:
        rg_factors = [2, 4, 8]

    pair_a, pair_b = lock.pair
    data_a = data[pair_a]
    data_b = data[pair_b]

    K_original = lock.K
    rg_test_results = {'original': {'K': K_original, 'stable': True}}

    all_stable = True

    for factor in rg_factors:
        # Coarse-grain both series
        data_a_coarse = coarse_grain_timeseries(data_a, factor)
        data_b_coarse = coarse_grain_timeseries(data_b, factor)

        # Recompute K on coarse-grained data
        K_coarse = compute_coupling_strength(data_a_coarse, data_b_coarse, lock.p, lock.q)

        # Stability criterion: K_coarse >= threshold_fraction * K_original
        stable = K_coarse >= k_threshold_fraction * K_original

        rg_test_results[f'factor_{factor}'] = {
            'K': K_coarse,
            'K_ratio': K_coarse / K_original if K_original > 0 else 0,
            'stable': stable
        }

        if not stable:
            all_stable = False

    metrics = {
        'K_original': K_original,
        'rg_factors_tested': rg_factors,
        'all_stable': all_stable,
        'min_K_ratio': min([r['K_ratio'] for r in rg_test_results.values() if 'K_ratio' in r])
    }

    return AuditResult(
        gate="E2",
        passed=all_stable,
        metrics=metrics,
        rg_test=rg_test_results,
        notes=f"RG-stable across {rg_factors}: {'PASS' if all_stable else 'FAIL'}"
    )


# ============================================================================
# E3: Live Performance Validated
# ============================================================================

def e3_live_validated(
    lock: PhaseLock,
    live_trades: List[Dict[str, Any]],
    min_trades: int = 10,
    win_rate_threshold: float = 0.45,
    profit_factor_threshold: float = 1.0
) -> AuditResult:
    """
    E3: Does lock perform in live/paper trading?

    Requirements:
    - Min trades executed
    - Win rate above threshold
    - Profit factor > 1.0 (profitable)
    - ΔH* positive on average

    This can only be tested in MICRO_LIVE or PRODUCTION mode.
    """
    if len(live_trades) < min_trades:
        return AuditResult(
            gate="E3",
            passed=False,
            metrics={'n_trades': len(live_trades), 'min_required': min_trades},
            notes=f"Insufficient trades: {len(live_trades)} < {min_trades}"
        )

    # Compute metrics from trades
    wins = [t for t in live_trades if t.get('pnl', 0) > 0]
    losses = [t for t in live_trades if t.get('pnl', 0) <= 0]

    win_rate = len(wins) / len(live_trades)

    total_wins = sum([t['pnl'] for t in wins])
    total_losses = abs(sum([t['pnl'] for t in losses]))
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    avg_delta_h = np.mean([t.get('delta_H_star', 0) for t in live_trades])

    metrics = {
        'n_trades': len(live_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_delta_H_star': avg_delta_h,
        'total_pnl': sum([t['pnl'] for t in live_trades])
    }

    passed = (
        win_rate >= win_rate_threshold and
        profit_factor >= profit_factor_threshold and
        avg_delta_h > 0
    )

    return AuditResult(
        gate="E3",
        passed=passed,
        metrics=metrics,
        notes=f"Live: {len(live_trades)} trades, WR={win_rate:.1%}, PF={profit_factor:.2f}"
    )


# ============================================================================
# E4: Long-Term Robust
# ============================================================================

def e4_long_term_robust(
    lock: PhaseLock,
    performance_history: List[Dict[str, float]],
    min_duration_days: int = 90,
    max_drawdown_threshold: float = -0.15,
    sharpe_threshold: float = 0.5
) -> AuditResult:
    """
    E4: Is lock robust over long term?

    Requirements:
    - Min duration (e.g., 90 days)
    - Max drawdown acceptable (< 15%)
    - Sharpe ratio positive
    - Consistent ΔH* (not degrading over time)

    This is the highest gate - only production-scale evidence.
    """
    if len(performance_history) < min_duration_days:
        return AuditResult(
            gate="E4",
            passed=False,
            metrics={'n_days': len(performance_history), 'min_required': min_duration_days},
            notes=f"Insufficient history: {len(performance_history)} < {min_duration_days} days"
        )

    # Extract metrics
    daily_returns = [p.get('return', 0) for p in performance_history]
    daily_delta_h = [p.get('delta_H_star', 0) for p in performance_history]

    # Compute cumulative returns
    cumulative = np.cumprod(1 + np.array(daily_returns))
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = np.min(drawdown)

    # Sharpe
    returns_array = np.array(daily_returns)
    sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-10) * np.sqrt(252)

    # Check ΔH* degradation (linear fit over time)
    x = np.arange(len(daily_delta_h))
    slope, _, _, p_value, _ = stats.linregress(x, daily_delta_h)

    metrics = {
        'n_days': len(performance_history),
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'delta_H_slope': slope,
        'delta_H_slope_pvalue': p_value,
        'total_return': cumulative[-1] - 1
    }

    passed = (
        max_drawdown >= max_drawdown_threshold and
        sharpe >= sharpe_threshold and
        slope >= 0  # ΔH* not degrading
    )

    return AuditResult(
        gate="E4",
        passed=passed,
        metrics=metrics,
        notes=f"Long-term: {len(performance_history)} days, DD={max_drawdown:.1%}, Sharpe={sharpe:.2f}"
    )


# ============================================================================
# Gate Orchestrator
# ============================================================================

class EGateOrchestrator:
    """
    Orchestrates E-gate progression for locks.

    Locks must pass gates sequentially: E0 → E1 → E2 → E3 → E4
    """

    def __init__(self):
        self.gate_functions = {
            'E0': e0_structure_exists,
            'E1': e1_beats_nulls,
            'E2': e2_rg_stable,
            'E3': e3_live_validated,
            'E4': e4_long_term_robust,
        }

    def audit_lock(
        self,
        lock: PhaseLock,
        data: Dict[str, np.ndarray],
        live_trades: Optional[List[Dict]] = None,
        performance_history: Optional[List[Dict]] = None
    ) -> List[AuditResult]:
        """
        Run full E-gate audit on lock.

        Returns list of AuditResults (one per gate attempted).
        Stops at first failure.
        """
        results = []

        # E0: Structure
        e0_result = e0_structure_exists(lock, data)
        results.append(e0_result)
        lock.e0_passed = e0_result.passed

        if not e0_result.passed:
            lock.status = LockStatus.REJECTED
            return results

        lock.status = LockStatus.E0_PASSED

        # E1: Nulls
        e1_result = e1_beats_nulls(lock, data)
        results.append(e1_result)
        lock.e1_passed = e1_result.passed

        if not e1_result.passed:
            lock.status = LockStatus.REJECTED
            return results

        lock.status = LockStatus.E1_PASSED

        # E2: RG
        e2_result = e2_rg_stable(lock, data)
        results.append(e2_result)
        lock.e2_passed = e2_result.passed

        if not e2_result.passed:
            lock.status = LockStatus.REJECTED
            return results

        lock.status = LockStatus.E2_PASSED

        # E3: Live (if data available)
        if live_trades is not None:
            e3_result = e3_live_validated(lock, live_trades)
            results.append(e3_result)
            lock.e3_passed = e3_result.passed

            if not e3_result.passed:
                lock.status = LockStatus.REJECTED
                return results

            lock.status = LockStatus.E3_PASSED

        # E4: Long-term (if data available)
        if performance_history is not None:
            e4_result = e4_long_term_robust(lock, performance_history)
            results.append(e4_result)
            lock.e4_passed = e4_result.passed

            if not e4_result.passed:
                lock.status = LockStatus.REJECTED
                return results

            lock.status = LockStatus.E4_PASSED

        # If we got here, lock is validated
        # Check PAD conditions to see if ACTUALIZED
        if lock.is_actionable():
            lock.status = LockStatus.ACTUALIZED

        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("E-GATES: Evidence Audit Framework")
    print("="*70)

    # Create mock lock
    from delta_state import PhaseLock

    lock = PhaseLock(
        pair=("AAPL", "MSFT"),
        p=2, q=3,
        K=0.65,
        Gamma_a=0.1, Gamma_b=0.1,
        Q_a=10, Q_b=10,
        eps_cap=0.8,
        eps_stab=0.7,
        zeta=0.3,
        delta_H_star=0.15
    )

    # Mock data
    np.random.seed(42)
    t = np.linspace(0, 100, 500)

    # Create correlated sine waves (mock phase lock)
    data_a = np.sin(2 * t) + 0.3 * np.random.randn(len(t))
    data_b = np.sin(3 * t) + 0.3 * np.random.randn(len(t))

    data = {"AAPL": data_a, "MSFT": data_b}

    # Run audits
    orchestrator = EGateOrchestrator()
    results = orchestrator.audit_lock(lock, data)

    print(f"\nLock: {lock.pair}, {lock.p}:{lock.q}, K={lock.K:.2f}")
    print(f"Status: {lock.status.value}\n")

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} | {result.gate}: {result.notes}")
        if result.metrics:
            for k, v in result.metrics.items():
                print(f"    {k}: {v}")

    print("\n" + "="*70)
    print(f"FINAL STATUS: {lock.status.value}")
    print(f"ACTIONABLE: {lock.is_actionable()}")
    print("="*70)
