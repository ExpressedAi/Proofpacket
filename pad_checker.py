"""
PAD Checker: Potential → Actualized → Deployed

Three-stage gate for phase-locks:
1. POTENTIAL (P): Basic structure exists, worth investigating
2. ACTUALIZED (A): Evidence strong, passes E2, PAD conditions met
3. DEPLOYED (D): Live validated (E3+), ready for capital

Follows oracle Δ-spec for rigorous lock promotion.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from delta_state_v2 import LockState, EStatus, DeltaState


# ============================================================================
# PAD Conditions (From Oracle Spec)
# ============================================================================

@dataclass
class PADConditions:
    """PAD thresholds for lock promotion."""

    # POTENTIAL conditions
    max_order: int = 7  # p + q <= 7 (low-order wins)
    min_coupling: float = 0.1  # K > 0.1 (non-trivial)
    min_quality: float = 5.0  # Q > 5 (low dissipation)

    # ACTUALIZED conditions
    min_eps_cap: float = 0.3  # ε_cap > 0.3 (capture eligibility)
    min_eps_stab: float = 0.3  # ε_stab > 0.3 (stability window)
    max_zeta: float = 0.7  # ζ <= 0.7 (brittleness threshold)
    min_delta_h: float = 0.0  # ΔH* > 0 (evidence gain)

    # DEPLOYED conditions
    min_e_level: int = 3  # Must pass E3 (live validation)
    min_evidence_score: float = 0.0  # Aggregate ΔH* > 0
    min_live_trades: int = 10  # Minimum live trades for E3


# ============================================================================
# PAD Checker
# ============================================================================

class PADChecker:
    """
    Check PAD conditions for phase-locks.

    Three gates:
    1. is_potential(): Basic sanity, E0 passed
    2. is_actualized(): PAD conditions + E1/E2 passed
    3. is_deployable(): E3+ passed, ready for capital
    """

    def __init__(self, conditions: Optional[PADConditions] = None):
        self.conditions = conditions or PADConditions()

    def check_potential(self, lock: LockState) -> Tuple[bool, Dict[str, str]]:
        """
        Check POTENTIAL conditions.

        Returns (passed, diagnostics).
        """
        diagnostics = {}

        # Check 1: Low-order wins (p + q <= max_order)
        if lock.order > self.conditions.max_order:
            diagnostics["order"] = f"FAIL: order={lock.order} > {self.conditions.max_order}"
            return False, diagnostics
        else:
            diagnostics["order"] = f"PASS: order={lock.order} <= {self.conditions.max_order}"

        # Check 2: Non-trivial coupling (|K| > min_coupling)
        if abs(lock.K) <= self.conditions.min_coupling:
            diagnostics["coupling"] = f"FAIL: |K|={abs(lock.K):.3f} <= {self.conditions.min_coupling}"
            return False, diagnostics
        else:
            diagnostics["coupling"] = f"PASS: |K|={abs(lock.K):.3f} > {self.conditions.min_coupling}"

        # Check 3: Quality factors (Q_a, Q_b > min_quality)
        if lock.Q_a < self.conditions.min_quality or lock.Q_b < self.conditions.min_quality:
            diagnostics["quality"] = f"FAIL: Q_a={lock.Q_a:.1f}, Q_b={lock.Q_b:.1f} < {self.conditions.min_quality}"
            return False, diagnostics
        else:
            diagnostics["quality"] = f"PASS: Q_a={lock.Q_a:.1f}, Q_b={lock.Q_b:.1f} > {self.conditions.min_quality}"

        # Check 4: E0 passed (structure exists)
        if lock.e0_status != EStatus.PASS:
            diagnostics["e0"] = f"FAIL: E0 status = {lock.e0_status.value}"
            return False, diagnostics
        else:
            diagnostics["e0"] = f"PASS: E0 passed"

        return True, diagnostics

    def check_actualized(self, lock: LockState) -> Tuple[bool, Dict[str, str]]:
        """
        Check ACTUALIZED conditions.

        Must pass POTENTIAL first, then check:
        - ε_cap, ε_stab, ζ (PAD conditions)
        - ΔH* > 0 (evidence)
        - E1, E2 passed
        """
        diagnostics = {}

        # Prerequisite: Must be potential
        is_pot, pot_diag = self.check_potential(lock)
        if not is_pot:
            diagnostics["potential"] = "FAIL: Not potential"
            diagnostics.update(pot_diag)
            return False, diagnostics
        else:
            diagnostics["potential"] = "PASS: Potential conditions met"

        # Check 1: Capture eligibility (ε_cap > min)
        if lock.eps_cap <= self.conditions.min_eps_cap:
            diagnostics["eps_cap"] = f"FAIL: ε_cap={lock.eps_cap:.3f} <= {self.conditions.min_eps_cap}"
            return False, diagnostics
        else:
            diagnostics["eps_cap"] = f"PASS: ε_cap={lock.eps_cap:.3f} > {self.conditions.min_eps_cap}"

        # Check 2: Stability eligibility (ε_stab > min)
        if lock.eps_stab <= self.conditions.min_eps_stab:
            diagnostics["eps_stab"] = f"FAIL: ε_stab={lock.eps_stab:.3f} <= {self.conditions.min_eps_stab}"
            return False, diagnostics
        else:
            diagnostics["eps_stab"] = f"PASS: ε_stab={lock.eps_stab:.3f} > {self.conditions.min_eps_stab}"

        # Check 3: Brittleness (ζ <= max)
        if lock.zeta > self.conditions.max_zeta:
            diagnostics["zeta"] = f"FAIL: ζ={lock.zeta:.3f} > {self.conditions.max_zeta}"
            return False, diagnostics
        else:
            diagnostics["zeta"] = f"PASS: ζ={lock.zeta:.3f} <= {self.conditions.max_zeta}"

        # Check 4: Evidence gain (ΔH* > 0)
        if lock.evidence_score <= self.conditions.min_delta_h:
            diagnostics["delta_h"] = f"FAIL: ΔH*={lock.evidence_score:.3f} <= {self.conditions.min_delta_h}"
            return False, diagnostics
        else:
            diagnostics["delta_h"] = f"PASS: ΔH*={lock.evidence_score:.3f} > {self.conditions.min_delta_h}"

        # Check 5: E1 passed (beats nulls)
        if lock.e1_status != EStatus.PASS:
            diagnostics["e1"] = f"FAIL: E1 status = {lock.e1_status.value}"
            return False, diagnostics
        else:
            diagnostics["e1"] = f"PASS: E1 passed (beats nulls)"

        # Check 6: E2 passed (RG-stable)
        if lock.e2_status != EStatus.PASS:
            diagnostics["e2"] = f"FAIL: E2 status = {lock.e2_status.value}"
            return False, diagnostics
        else:
            diagnostics["e2"] = f"PASS: E2 passed (RG-stable)"

        return True, diagnostics

    def check_deployable(self, lock: LockState) -> Tuple[bool, Dict[str, str]]:
        """
        Check DEPLOYABLE conditions.

        Must pass ACTUALIZED first, then check:
        - E3 passed (live validation)
        - Evidence score > 0
        """
        diagnostics = {}

        # Prerequisite: Must be actualized
        is_act, act_diag = self.check_actualized(lock)
        if not is_act:
            diagnostics["actualized"] = "FAIL: Not actualized"
            diagnostics.update(act_diag)
            return False, diagnostics
        else:
            diagnostics["actualized"] = "PASS: Actualized conditions met"

        # Check 1: E3 passed (live validation)
        if lock.e3_status != EStatus.PASS:
            diagnostics["e3"] = f"FAIL: E3 status = {lock.e3_status.value}"
            return False, diagnostics
        else:
            diagnostics["e3"] = f"PASS: E3 passed (live validated)"

        # Check 2: E-level at least 3
        if lock.e_level_passed < self.conditions.min_e_level:
            diagnostics["e_level"] = f"FAIL: E-level={lock.e_level_passed} < {self.conditions.min_e_level}"
            return False, diagnostics
        else:
            diagnostics["e_level"] = f"PASS: E-level={lock.e_level_passed} >= {self.conditions.min_e_level}"

        # Check 3: Aggregate evidence score > 0
        if lock.evidence_score <= self.conditions.min_evidence_score:
            diagnostics["evidence"] = f"FAIL: Evidence={lock.evidence_score:.3f} <= {self.conditions.min_evidence_score}"
            return False, diagnostics
        else:
            diagnostics["evidence"] = f"PASS: Evidence={lock.evidence_score:.3f} > {self.conditions.min_evidence_score}"

        return True, diagnostics

    def promote_lock(self, lock: LockState) -> str:
        """
        Determine highest PAD level for lock.

        Returns: "potential", "actualized", "deployable", or "none"
        """
        # Check deployable (highest level)
        is_deploy, _ = self.check_deployable(lock)
        if is_deploy:
            return "deployable"

        # Check actualized
        is_act, _ = self.check_actualized(lock)
        if is_act:
            return "actualized"

        # Check potential
        is_pot, _ = self.check_potential(lock)
        if is_pot:
            return "potential"

        return "none"

    def get_deployable_locks(self, state: DeltaState) -> List[LockState]:
        """Get all locks ready for deployment."""
        deployable = []
        for lock in state.locks.values():
            is_deploy, _ = self.check_deployable(lock)
            if is_deploy:
                deployable.append(lock)
        return deployable

    def get_actualized_locks(self, state: DeltaState) -> List[LockState]:
        """Get all actualized locks (E2 passed, PAD met)."""
        actualized = []
        for lock in state.locks.values():
            is_act, _ = self.check_actualized(lock)
            if is_act:
                actualized.append(lock)
        return actualized

    def get_potential_locks(self, state: DeltaState) -> List[LockState]:
        """Get all potential locks (E0 passed, basic sanity)."""
        potential = []
        for lock in state.locks.values():
            is_pot, _ = self.check_potential(lock)
            if is_pot:
                potential.append(lock)
        return potential

    def generate_report(self, lock: LockState) -> str:
        """Generate detailed PAD report for a lock."""
        report = []
        report.append(f"=" * 70)
        report.append(f"PAD REPORT: {lock.id}")
        report.append(f"=" * 70)

        # Potential check
        report.append(f"\n[1. POTENTIAL CHECK]")
        is_pot, pot_diag = self.check_potential(lock)
        for key, msg in pot_diag.items():
            report.append(f"  {key}: {msg}")
        report.append(f"  → Result: {'PASS' if is_pot else 'FAIL'}")

        if not is_pot:
            report.append(f"\n⚠ Lock failed POTENTIAL gate. Investigation not recommended.")
            return "\n".join(report)

        # Actualized check
        report.append(f"\n[2. ACTUALIZED CHECK]")
        is_act, act_diag = self.check_actualized(lock)
        for key, msg in act_diag.items():
            if key != "potential":  # Skip duplicate potential diagnostics
                report.append(f"  {key}: {msg}")
        report.append(f"  → Result: {'PASS' if is_act else 'FAIL'}")

        if not is_act:
            report.append(f"\n⚠ Lock is POTENTIAL but not ACTUALIZED. Needs more evidence (E1/E2).")
            return "\n".join(report)

        # Deployable check
        report.append(f"\n[3. DEPLOYABLE CHECK]")
        is_deploy, deploy_diag = self.check_deployable(lock)
        for key, msg in deploy_diag.items():
            if key != "actualized":  # Skip duplicate actualized diagnostics
                report.append(f"  {key}: {msg}")
        report.append(f"  → Result: {'PASS' if is_deploy else 'FAIL'}")

        if not is_deploy:
            report.append(f"\n⚠ Lock is ACTUALIZED but not DEPLOYABLE. Needs live validation (E3).")
            return "\n".join(report)

        # Success
        report.append(f"\n✓ Lock is DEPLOYABLE. Ready for capital allocation.")
        report.append(f"=" * 70)

        return "\n".join(report)


# ============================================================================
# MDL Penalty (Low-Order Wins)
# ============================================================================

def compute_mdl_penalty(p: int, q: int) -> float:
    """
    MDL penalty for lock order.

    Low-order wins: prefer small p, q.
    Penalty = 1 / (p * q)
    """
    return 1.0 / (p * q)


def rank_locks_by_low_order(locks: List[LockState]) -> List[LockState]:
    """
    Rank locks by low-order wins principle.

    Lower order (p + q) ranks higher.
    Tie-break by evidence score.
    """
    def score_fn(lock):
        mdl = compute_mdl_penalty(lock.p, lock.q)
        return (lock.order, -lock.evidence_score)  # Ascending order, descending evidence

    return sorted(locks, key=score_fn)


# ============================================================================
# Brittleness Calculations
# ============================================================================

def compute_brittleness(
    concentration: float,
    overfit_risk: float,
    leverage: float = 1.0
) -> float:
    """
    Compute brittleness ζ from multiple factors.

    ζ = w_conc * concentration + w_overfit * overfit + w_lev * leverage

    Args:
        concentration: Position concentration (0-1)
        overfit_risk: Overfitting indicator (0-1)
        leverage: Leverage multiplier (1.0 = no leverage)

    Returns:
        ζ ∈ [0, 1], higher = more brittle
    """
    w_conc = 0.4
    w_overfit = 0.4
    w_lev = 0.2

    leverage_factor = min((leverage - 1.0) / 4.0, 1.0)  # Cap at 5x leverage

    zeta = (
        w_conc * concentration +
        w_overfit * overfit_risk +
        w_lev * leverage_factor
    )

    return np.clip(zeta, 0.0, 1.0)


def estimate_overfit_risk(
    in_sample_sharpe: float,
    out_sample_sharpe: float,
    n_parameters: int
) -> float:
    """
    Estimate overfitting risk from IS/OOS degradation.

    Higher risk if:
    - Large IS/OOS gap
    - Many parameters
    """
    sharpe_degradation = max(0, in_sample_sharpe - out_sample_sharpe) / (in_sample_sharpe + 1e-6)
    parameter_penalty = min(n_parameters / 20.0, 1.0)  # Cap at 20 params

    overfit_risk = 0.6 * sharpe_degradation + 0.4 * parameter_penalty

    return np.clip(overfit_risk, 0.0, 1.0)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from delta_state_v2 import create_research_state

    print("=" * 70)
    print("PAD CHECKER: Potential → Actualized → Deployed")
    print("=" * 70)

    # Create state with test locks
    state = create_research_state()
    checker = PADChecker()

    # Lock 1: POTENTIAL only (E0 passed, but no E1/E2)
    lock1 = LockState(
        id="AAPL-MSFT-2:3",
        a="AAPL", b="MSFT",
        p=2, q=3,
        K=0.65,
        Gamma_a=0.1, Gamma_b=0.1,
        Q_a=10, Q_b=10,
        eps_cap=0.5, eps_stab=0.6, zeta=0.4,
        evidence_score=0.0  # No evidence yet
    )
    lock1.e0_status = EStatus.PASS
    lock1.e1_status = EStatus.PENDING
    lock1.e2_status = EStatus.PENDING

    state.locks[lock1.id] = lock1

    # Lock 2: ACTUALIZED (E0, E1, E2 passed, PAD met)
    lock2 = LockState(
        id="TSLA-NVDA-1:1",
        a="TSLA", b="NVDA",
        p=1, q=1,  # Low-order!
        K=0.85,
        Gamma_a=0.05, Gamma_b=0.05,
        Q_a=15, Q_b=15,
        eps_cap=0.8, eps_stab=0.7, zeta=0.3,
        evidence_score=0.15  # Positive evidence
    )
    lock2.e0_status = EStatus.PASS
    lock2.e1_status = EStatus.PASS
    lock2.e2_status = EStatus.PASS
    lock2.e3_status = EStatus.PENDING
    lock2.e_level_passed = 2

    state.locks[lock2.id] = lock2

    # Lock 3: DEPLOYABLE (all gates passed)
    lock3 = LockState(
        id="SPY-QQQ-1:2",
        a="SPY", b="QQQ",
        p=1, q=2,  # Low-order
        K=0.92,
        Gamma_a=0.03, Gamma_b=0.03,
        Q_a=20, Q_b=20,
        eps_cap=0.9, eps_stab=0.85, zeta=0.2,
        evidence_score=0.25  # Strong evidence
    )
    lock3.e0_status = EStatus.PASS
    lock3.e1_status = EStatus.PASS
    lock3.e2_status = EStatus.PASS
    lock3.e3_status = EStatus.PASS
    lock3.e_level_passed = 3

    state.locks[lock3.id] = lock3

    # Check each lock
    print(f"\n[LOCK 1: {lock1.id}]")
    level1 = checker.promote_lock(lock1)
    print(f"PAD Level: {level1.upper()}")

    print(f"\n[LOCK 2: {lock2.id}]")
    level2 = checker.promote_lock(lock2)
    print(f"PAD Level: {level2.upper()}")

    print(f"\n[LOCK 3: {lock3.id}]")
    level3 = checker.promote_lock(lock3)
    print(f"PAD Level: {level3.upper()}")

    # Summary
    print(f"\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Potential locks: {len(checker.get_potential_locks(state))}")
    print(f"  Actualized locks: {len(checker.get_actualized_locks(state))}")
    print(f"  Deployable locks: {len(checker.get_deployable_locks(state))}")
    print(f"=" * 70)

    # Detailed report for deployable lock
    print(f"\n{checker.generate_report(lock3)}")

    # Test low-order ranking
    print(f"\n[LOW-ORDER RANKING]")
    ranked = rank_locks_by_low_order(list(state.locks.values()))
    for i, lock in enumerate(ranked, 1):
        print(f"  {i}. {lock.id}: order={lock.order}, ΔH*={lock.evidence_score:.3f}")
