"""
E-Gates V2: Oracle-Compliant Evidence Audit Framework (E0 → E1 → E2 → E3 → E4)

Each lock/strategy must pass gates sequentially before trading.
Follows Δ-Method epistemology with rigorous null tests.

Gate progression:
E0: Structure exists (basic detection)
E1: Beats domain-specific nulls (from null_tests.py)
E2: RG-stable (survives coarse-graining)
E3: Live performance validated (paper/micro-live)
E4: Long-term robust (production-scale evidence)

Integrates with:
- delta_state_v2.py (LockState, AuditStats)
- null_tests.py (domain-specific nulls)
- delta_h_calculator.py (evidence scoring)
- pad_checker.py (promotion logic)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime

from delta_state_v2 import (
    DeltaState, LockState, AuditStats, EStatus,
    OperatingMode, StrategyState
)
from null_tests import (
    run_all_nulls_for_layer,
    NullTestResult,
    benjamini_hochberg_correction
)
from delta_h_calculator import (
    compute_window_delta_h,
    update_lock_delta_h,
    DeltaHResult
)


# ============================================================================
# E0: Structure Exists
# ============================================================================

def e0_structure_exists(
    lock: LockState,
    data: Dict[str, np.ndarray]
) -> Tuple[bool, Dict[str, Any]]:
    """
    E0: Does the basic structure exist?

    Requirements:
    - Sufficient data (min 100 points)
    - K > 0.1 (non-trivial coupling)
    - p, q are low-order (p+q <= 7 for LOW principle)
    - Quality factors Q > 5

    This is the cheapest gate - just sanity checks.

    Returns:
        (passed, diagnostics)
    """
    diagnostics = {}

    # Check data sufficiency
    data_a = data.get(lock.a, np.array([]))
    data_b = data.get(lock.b, np.array([]))

    n_points = min(len(data_a), len(data_b))
    diagnostics['n_points'] = n_points

    # Check coupling
    diagnostics['K'] = lock.K
    diagnostics['order'] = lock.order

    # Check quality factors
    diagnostics['Q_a'] = lock.Q_a
    diagnostics['Q_b'] = lock.Q_b

    # Pass conditions
    checks = {
        'data_sufficient': n_points >= 100,
        'coupling_strong': abs(lock.K) > 0.1,
        'low_order': lock.order <= 7,
        'quality_factors': lock.Q_a > 5.0 and lock.Q_b > 5.0,
        'no_nan': not np.isnan(lock.K) and not np.isinf(lock.K)
    }

    diagnostics.update(checks)
    passed = all(checks.values())

    return passed, diagnostics


# ============================================================================
# E1: Beats Domain-Specific Nulls
# ============================================================================

def compute_coupling_strength(
    data_a: np.ndarray,
    data_b: np.ndarray,
    p: int,
    q: int
) -> float:
    """
    Compute coupling strength K for p:q lock.

    Simplified: uses phase coherence metric.
    Real version would use Hilbert transform.
    """
    if len(data_a) != len(data_b):
        min_len = min(len(data_a), len(data_b))
        data_a = data_a[:min_len]
        data_b = data_b[:min_len]

    # Correlation strength (proxy for phase coherence)
    corr = np.corrcoef(data_a, data_b)[0, 1]

    # Adjust for order (higher order = lower expected K)
    K = abs(corr) / np.sqrt(p * q)

    return K


def e1_beats_nulls(
    lock: LockState,
    data: Dict[str, np.ndarray],
    layer: str = "consensus",
    n_surrogates: int = 100
) -> Tuple[bool, Dict[str, Any], List[NullTestResult]]:
    """
    E1: Does lock beat domain-specific null hypotheses?

    Uses null_tests.py for domain-specific nulls.

    Args:
        lock: LockState to test
        data: Market data dict
        layer: Which layer's nulls to use ("consensus", "chi_crash", "fraud", "tur")
        n_surrogates: Number of null surrogates

    Returns:
        (passed, diagnostics, null_results)
    """
    data_a = data[lock.a]
    data_b = data[lock.b]

    # Compute observed coupling
    K_observed = lock.K

    # Prepare data for null tests
    # For now, use generic phase-lock nulls
    # In production, would use layer-specific nulls

    from null_tests import (
        generate_phase_shuffle_null,
        generate_block_surrogate
    )

    # Phase shuffle null
    surrogates_a = generate_phase_shuffle_null(data_a, n_surrogates)
    K_nulls_phase = []
    for surr_a in surrogates_a:
        K_null = compute_coupling_strength(surr_a, data_b, lock.p, lock.q)
        K_nulls_phase.append(K_null)

    p_value_phase = np.mean(np.array(K_nulls_phase) >= K_observed)

    # Block surrogate null
    surrogates_a_block = generate_block_surrogate(data_a, n_surrogates=n_surrogates)
    surrogates_b_block = generate_block_surrogate(data_b, n_surrogates=n_surrogates)

    K_nulls_block = []
    for surr_a, surr_b in zip(surrogates_a_block, surrogates_b_block):
        K_null = compute_coupling_strength(surr_a, surr_b, lock.p, lock.q)
        K_nulls_block.append(K_null)

    p_value_block = np.mean(np.array(K_nulls_block) >= K_observed)

    # FDR correction
    p_values = [p_value_phase, p_value_block]
    fdr_results = benjamini_hochberg_correction(p_values, alpha=0.05)

    passed = any(fdr_results)  # At least one null beaten

    diagnostics = {
        'K_observed': K_observed,
        'p_value_phase': p_value_phase,
        'p_value_block': p_value_block,
        'fdr_corrected': fdr_results,
        'min_p_value': min(p_values)
    }

    # Create NullTestResult objects for tracking
    null_results = []

    return passed, diagnostics, null_results


# ============================================================================
# E2: RG-Stable (Survives Coarse-Graining)
# ============================================================================

def coarse_grain_timeseries(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Coarse-grain time series by averaging over blocks.

    RG transform: reduce resolution by factor.
    """
    n = len(data)
    n_coarse = n // factor

    coarse_data = []
    for i in range(n_coarse):
        start = i * factor
        end = start + factor
        coarse_data.append(np.mean(data[start:end]))

    return np.array(coarse_data)


def e2_rg_stable(
    lock: LockState,
    data: Dict[str, np.ndarray],
    rg_factors: List[int] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    E2: Does lock survive RG (coarse-graining)?

    Tests if coupling persists when:
    - Data resolution reduced by 2x, 4x, 8x
    - High-frequency noise removed
    - Only robust low-frequency structure remains

    Returns:
        (passed, diagnostics)
    """
    if rg_factors is None:
        rg_factors = [2, 4, 8]

    data_a = data[lock.a]
    data_b = data[lock.b]

    K_original = lock.K
    K_coarse = {}

    for factor in rg_factors:
        # Coarse-grain both series
        data_a_coarse = coarse_grain_timeseries(data_a, factor)
        data_b_coarse = coarse_grain_timeseries(data_b, factor)

        # Recompute coupling
        K_coarse[factor] = compute_coupling_strength(
            data_a_coarse, data_b_coarse, lock.p, lock.q
        )

    # Pass if coupling stays above 50% of original for all factors
    threshold = 0.5
    stability_checks = {
        f'RG_{factor}x': (K_coarse[factor] >= threshold * abs(K_original))
        for factor in rg_factors
    }

    passed = all(stability_checks.values())

    diagnostics = {
        'K_original': K_original,
        'K_coarse': K_coarse,
        'threshold': threshold,
        'stability_checks': stability_checks
    }

    return passed, diagnostics


# ============================================================================
# E3: Live Performance Validated
# ============================================================================

def e3_live_validated(
    lock: LockState,
    live_trades: List[Dict[str, Any]],
    min_trades: int = 10
) -> Tuple[bool, Dict[str, Any]]:
    """
    E3: Does lock perform well in live/paper trading?

    Requirements:
    - Min 10 trades executed
    - Win rate > 45%
    - Profit factor > 1.0
    - ΔH* > 0 (evidence gain positive)

    Returns:
        (passed, diagnostics)
    """
    if len(live_trades) < min_trades:
        diagnostics = {
            'n_trades': len(live_trades),
            'min_trades': min_trades,
            'sufficient_trades': False
        }
        return False, diagnostics

    # Extract P&Ls
    pnls = [trade['pnl'] for trade in live_trades]
    pnls = np.array(pnls)

    # Win rate
    wins = np.sum(pnls > 0)
    losses = np.sum(pnls < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Average P&L
    avg_pnl = np.mean(pnls)

    # ΔH* (assuming null baseline is break-even)
    # Real implementation would compute proper ΔH* vs nulls
    delta_h_star = avg_pnl / (np.std(pnls) + 1e-6)

    # Pass conditions
    checks = {
        'sufficient_trades': len(live_trades) >= min_trades,
        'win_rate': win_rate > 0.45,
        'profit_factor': profit_factor > 1.0,
        'positive_delta_h': delta_h_star > 0
    }

    passed = all(checks.values())

    diagnostics = {
        'n_trades': len(live_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_pnl': avg_pnl,
        'delta_h_star': delta_h_star,
        **checks
    }

    return passed, diagnostics


# ============================================================================
# E4: Long-Term Robust
# ============================================================================

def e4_long_term_robust(
    lock: LockState,
    performance_history: List[Dict[str, Any]],
    min_days: int = 90
) -> Tuple[bool, Dict[str, Any]]:
    """
    E4: Is lock robust over long-term production?

    Requirements:
    - Min 90 days of live data
    - Max drawdown < 15%
    - Sharpe > 0.5
    - ΔH* not degrading (stays above 50% of peak)

    Returns:
        (passed, diagnostics)
    """
    if len(performance_history) < min_days:
        diagnostics = {
            'n_days': len(performance_history),
            'min_days': min_days,
            'sufficient_history': False
        }
        return False, diagnostics

    # Extract metrics
    daily_pnls = [day['pnl'] for day in performance_history]
    daily_pnls = np.array(daily_pnls)

    # Cumulative returns
    cumulative = np.cumsum(daily_pnls)

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / (peak + 1e-6)
    max_drawdown = np.min(drawdown)

    # Sharpe
    sharpe = np.mean(daily_pnls) / (np.std(daily_pnls) + 1e-6) * np.sqrt(252)

    # ΔH* trend
    delta_h_history = [day.get('delta_h_star', 0) for day in performance_history]
    delta_h_peak = max(delta_h_history) if delta_h_history else 0
    delta_h_current = delta_h_history[-1] if delta_h_history else 0

    delta_h_stable = (delta_h_current >= 0.5 * delta_h_peak) if delta_h_peak > 0 else False

    # Pass conditions
    checks = {
        'sufficient_history': len(performance_history) >= min_days,
        'max_drawdown': abs(max_drawdown) < 0.15,
        'sharpe': sharpe > 0.5,
        'delta_h_stable': delta_h_stable
    }

    passed = all(checks.values())

    diagnostics = {
        'n_days': len(performance_history),
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'delta_h_peak': delta_h_peak,
        'delta_h_current': delta_h_current,
        **checks
    }

    return passed, diagnostics


# ============================================================================
# E-Gate Orchestrator
# ============================================================================

class EGateOrchestrator:
    """
    Orchestrates E-gate audits for locks and strategies.

    Updates DeltaState with audit results.
    Enforces sequential gate passing (E0 → E1 → E2 → E3 → E4).
    """

    def __init__(self):
        pass

    def audit_lock(
        self,
        state: DeltaState,
        lock_id: str,
        target_level: int = 4
    ) -> bool:
        """
        Run E-gate audits for a lock up to target_level.

        Returns True if all gates up to target passed.

        Args:
            state: DeltaState
            lock_id: Lock ID to audit
            target_level: Target E-level (0-4)

        Returns:
            All gates passed up to target
        """
        if lock_id not in state.locks:
            state.add_log(f"⚠ Lock {lock_id} not found")
            return False

        lock = state.locks[lock_id]

        # Get or create AuditStats
        if lock_id not in state.audits:
            state.audits[lock_id] = AuditStats(id=lock_id)

        audit = state.audits[lock_id]

        # E0: Structure exists
        if target_level >= 0 and audit.E0 != EStatus.PASS:
            passed, diag = e0_structure_exists(lock, state.markets)
            audit.E0 = EStatus.PASS if passed else EStatus.FAIL
            audit.notes['E0'] = diag
            state.add_log(f"E0 {lock_id}: {'PASS' if passed else 'FAIL'}")

            if not passed:
                lock.e0_status = EStatus.FAIL
                return False

            lock.e0_status = EStatus.PASS

        # E1: Beats nulls
        if target_level >= 1 and audit.E1 != EStatus.PASS:
            if audit.E0 != EStatus.PASS:
                state.add_log(f"⚠ E1 {lock_id}: E0 not passed yet")
                return False

            passed, diag, null_results = e1_beats_nulls(lock, state.markets)
            audit.E1 = EStatus.PASS if passed else EStatus.FAIL
            audit.notes['E1'] = diag
            state.add_log(f"E1 {lock_id}: {'PASS' if passed else 'FAIL'} (p={diag.get('min_p_value', 1.0):.3f})")

            if not passed:
                lock.e1_status = EStatus.FAIL
                return False

            lock.e1_status = EStatus.PASS

        # E2: RG-stable
        if target_level >= 2 and audit.E2 != EStatus.PASS:
            if audit.E1 != EStatus.PASS:
                state.add_log(f"⚠ E2 {lock_id}: E1 not passed yet")
                return False

            passed, diag = e2_rg_stable(lock, state.markets)
            audit.E2 = EStatus.PASS if passed else EStatus.FAIL
            audit.notes['E2'] = diag
            state.add_log(f"E2 {lock_id}: {'PASS' if passed else 'FAIL'}")

            if not passed:
                lock.e2_status = EStatus.FAIL
                return False

            lock.e2_status = EStatus.PASS

        # E3: Live validated (requires external data)
        if target_level >= 3 and audit.E3 != EStatus.PASS:
            if audit.E2 != EStatus.PASS:
                state.add_log(f"⚠ E3 {lock_id}: E2 not passed yet")
                return False

            # Would get live trades from state
            live_trades = lock.live_performance.get('trades', [])

            passed, diag = e3_live_validated(lock, live_trades)
            audit.E3 = EStatus.PASS if passed else EStatus.FAIL
            audit.notes['E3'] = diag
            state.add_log(f"E3 {lock_id}: {'PASS' if passed else 'FAIL'} (n={len(live_trades)})")

            if not passed:
                lock.e3_status = EStatus.FAIL
                return False

            lock.e3_status = EStatus.PASS

        # E4: Long-term robust (requires extended history)
        if target_level >= 4 and audit.E4 != EStatus.PASS:
            if audit.E3 != EStatus.PASS:
                state.add_log(f"⚠ E4 {lock_id}: E3 not passed yet")
                return False

            # Would get long-term performance from state
            performance_history = lock.live_performance.get('history', [])

            passed, diag = e4_long_term_robust(lock, performance_history)
            audit.E4 = EStatus.PASS if passed else EStatus.FAIL
            audit.notes['E4'] = diag
            state.add_log(f"E4 {lock_id}: {'PASS' if passed else 'FAIL'} (n={len(performance_history)} days)")

            if not passed:
                lock.e4_status = EStatus.FAIL
                return False

            lock.e4_status = EStatus.PASS

        # Update lock's e_level_passed
        lock.e_level_passed = audit.max_level_passed()
        audit.last_updated = datetime.utcnow()

        return audit.all_passed_up_to(target_level)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from delta_state_v2 import create_research_state

    print("=" * 70)
    print("E-GATES V2: Oracle-Compliant Evidence Audit Framework")
    print("=" * 70)

    # Create state with synthetic data
    np.random.seed(42)
    state = create_research_state()

    # Add synthetic market data
    n = 500
    dates = np.arange(n)
    prices_a = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices_b = 100 + np.cumsum(0.7 * np.diff(np.concatenate([[0], prices_a])) + 0.3 * np.random.randn(n))

    state.markets['AAPL'] = prices_a
    state.markets['MSFT'] = prices_b

    # Create test lock
    lock = LockState(
        id="AAPL-MSFT-2:3",
        a="AAPL", b="MSFT",
        p=2, q=3,
        K=0.65,
        Gamma_a=0.1, Gamma_b=0.1,
        Q_a=10, Q_b=10,
        eps_cap=0.5, eps_stab=0.6, zeta=0.4
    )

    state.locks[lock.id] = lock

    # Run E-gate audits
    orchestrator = EGateOrchestrator()

    print(f"\n[AUDITING LOCK: {lock.id}]")
    print(f"Initial status: E{lock.e_level_passed}")

    # Run E0-E2 (research mode gates)
    success = orchestrator.audit_lock(state, lock.id, target_level=2)

    print(f"\nFinal status: E{lock.e_level_passed}")
    print(f"Audit complete: {success}")

    # Print audit stats
    if lock.id in state.audits:
        audit = state.audits[lock.id]
        print(f"\nE-Gate Status:")
        print(f"  E0: {audit.E0.value}")
        print(f"  E1: {audit.E1.value}")
        print(f"  E2: {audit.E2.value}")
        print(f"  E3: {audit.E3.value}")
        print(f"  E4: {audit.E4.value}")

    print("\n" + "=" * 70)
    print("E-gates ensure rigorous epistemology before capital deployment")
    print("=" * 70)
