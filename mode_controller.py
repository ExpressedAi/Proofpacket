"""
Mode Controller: Gates system capabilities by evidence level

Three modes with increasing evidence requirements:
- RESEARCH: E0-E2 only, no capital, exploratory
- MICRO_LIVE: Tiny capital, E3 validation, high logging
- PRODUCTION: Full capital, E4 validated, all gates active

Promotion/demotion based on E-test passes, ΔH* vs nulls, hazard scores.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

from delta_state import DeltaState, OperatingMode, PhaseLock, LockStatus
from e_gates import EGateOrchestrator


@dataclass
class ModeTransitionCriteria:
    """Criteria for mode transitions."""
    # Promotion thresholds
    min_locks_validated: int
    min_e_gate_level: str  # "E0", "E1", "E2", etc.
    min_avg_delta_h: float
    max_failure_rate: float
    min_duration_days: int

    # Demotion triggers
    max_consecutive_failures: int
    max_drawdown_breach: float
    min_sharpe_threshold: float


class ModeController:
    """
    Controls system operating mode and enforces evidence gates.

    Prevents premature deployment:
    - RESEARCH → MICRO_LIVE requires E2-passing locks
    - MICRO_LIVE → PRODUCTION requires E3-passing locks + live performance
    - AUTO-DEMOTE if production performance degrades
    """

    def __init__(self):
        self.orchestrator = EGateOrchestrator()

        # Define transition criteria
        self.criteria = {
            OperatingMode.RESEARCH: ModeTransitionCriteria(
                # Promotion to MICRO_LIVE
                min_locks_validated=3,
                min_e_gate_level="E2",
                min_avg_delta_h=0.05,
                max_failure_rate=0.3,
                min_duration_days=0,
                # Demotion (N/A - already at bottom)
                max_consecutive_failures=999,
                max_drawdown_breach=-1.0,
                min_sharpe_threshold=-999
            ),
            OperatingMode.MICRO_LIVE: ModeTransitionCriteria(
                # Promotion to PRODUCTION
                min_locks_validated=5,
                min_e_gate_level="E3",
                min_avg_delta_h=0.10,
                max_failure_rate=0.2,
                min_duration_days=30,
                # Demotion to RESEARCH
                max_consecutive_failures=10,
                max_drawdown_breach=-0.10,
                min_sharpe_threshold=0.0
            ),
            OperatingMode.PRODUCTION: ModeTransitionCriteria(
                # Promotion (N/A - already at top)
                min_locks_validated=999,
                min_e_gate_level="E4",
                min_avg_delta_h=999,
                max_failure_rate=0.0,
                min_duration_days=999,
                # Demotion to MICRO_LIVE
                max_consecutive_failures=5,
                max_drawdown_breach=-0.15,
                min_sharpe_threshold=0.3
            )
        }

    def check_promotion_eligibility(self, state: DeltaState) -> bool:
        """
        Check if current mode can be promoted.

        Returns True if criteria met for next level.
        """
        current_mode = state.mode
        criteria = self.criteria[current_mode]

        # Count locks at required E-gate level
        validated_locks = []

        for lock in state.locks:
            if criteria.min_e_gate_level == "E0" and lock.e0_passed:
                validated_locks.append(lock)
            elif criteria.min_e_gate_level == "E1" and lock.e1_passed:
                validated_locks.append(lock)
            elif criteria.min_e_gate_level == "E2" and lock.e2_passed:
                validated_locks.append(lock)
            elif criteria.min_e_gate_level == "E3" and lock.e3_passed:
                validated_locks.append(lock)
            elif criteria.min_e_gate_level == "E4" and lock.e4_passed:
                validated_locks.append(lock)

        # Check criteria
        has_enough_locks = len(validated_locks) >= criteria.min_locks_validated

        # Average ΔH*
        if validated_locks:
            avg_delta_h = sum([l.delta_H_star for l in validated_locks]) / len(validated_locks)
        else:
            avg_delta_h = 0.0

        has_positive_delta_h = avg_delta_h >= criteria.min_avg_delta_h

        # Duration check (for MICRO_LIVE → PRODUCTION)
        if current_mode == OperatingMode.MICRO_LIVE:
            if state.mode_history:
                # Find when we entered MICRO_LIVE
                micro_live_start = None
                for timestamp, mode, _ in reversed(state.mode_history):
                    if mode == OperatingMode.MICRO_LIVE:
                        micro_live_start = timestamp
                        break

                if micro_live_start:
                    duration = (state.time - micro_live_start).days
                    has_sufficient_duration = duration >= criteria.min_duration_days
                else:
                    has_sufficient_duration = False
            else:
                has_sufficient_duration = False
        else:
            has_sufficient_duration = True

        eligible = (
            has_enough_locks and
            has_positive_delta_h and
            has_sufficient_duration
        )

        state.add_log(
            f"Promotion check: locks={len(validated_locks)}/{criteria.min_locks_validated}, "
            f"ΔH*={avg_delta_h:.3f}/{criteria.min_avg_delta_h:.3f}, "
            f"eligible={eligible}"
        )

        return eligible

    def check_demotion_triggers(self, state: DeltaState) -> bool:
        """
        Check if current mode should be demoted.

        Returns True if safety triggers are breached.
        """
        current_mode = state.mode
        criteria = self.criteria[current_mode]

        # Can't demote from RESEARCH (already at bottom)
        if current_mode == OperatingMode.RESEARCH:
            return False

        # Check drawdown breach
        drawdown_breach = state.portfolio.max_drawdown < criteria.max_drawdown_breach

        # Check Sharpe degradation
        sharpe_fail = state.portfolio.sharpe < criteria.min_sharpe_threshold

        # Check consecutive failures (simplified - would need trade history)
        # For now, use house score as proxy
        performance_fail = state.portfolio.house_score < 30  # Arbitrary threshold

        should_demote = drawdown_breach or sharpe_fail or performance_fail

        if should_demote:
            reason = []
            if drawdown_breach:
                reason.append(f"DD={state.portfolio.max_drawdown:.2%}")
            if sharpe_fail:
                reason.append(f"Sharpe={state.portfolio.sharpe:.2f}")
            if performance_fail:
                reason.append(f"HouseScore={state.portfolio.house_score:.1f}")

            state.add_log(f"DEMOTION TRIGGER: {', '.join(reason)}")

        return should_demote

    def attempt_promotion(self, state: DeltaState) -> bool:
        """
        Attempt to promote to next mode.

        Returns True if promoted, False otherwise.
        """
        if not self.check_promotion_eligibility(state):
            return False

        # Determine next mode
        if state.mode == OperatingMode.RESEARCH:
            next_mode = OperatingMode.MICRO_LIVE
            state.portfolio.cash = 1000.0  # Start with $1K
            state.portfolio.total_value = 1000.0
        elif state.mode == OperatingMode.MICRO_LIVE:
            next_mode = OperatingMode.PRODUCTION
            state.portfolio.cash = 100000.0  # Scale up to $100K
            state.portfolio.total_value = 100000.0
        else:
            return False  # Already at max

        # Record transition
        state.mode_history.append((state.time, next_mode, "PROMOTION"))
        state.mode = next_mode

        state.add_log(f"✓ PROMOTED to {next_mode.value}")

        return True

    def force_demotion(self, state: DeltaState, reason: str = "Performance degradation"):
        """
        Force demotion to safer mode.
        """
        if state.mode == OperatingMode.PRODUCTION:
            next_mode = OperatingMode.MICRO_LIVE
            state.portfolio.cash = 1000.0
            state.portfolio.total_value = 1000.0
        elif state.mode == OperatingMode.MICRO_LIVE:
            next_mode = OperatingMode.RESEARCH
            state.portfolio.cash = 0.0
            state.portfolio.total_value = 0.0
        else:
            return  # Already at minimum

        # Record transition
        state.mode_history.append((state.time, next_mode, f"DEMOTION: {reason}"))
        state.mode = next_mode

        state.add_log(f"⚠ DEMOTED to {next_mode.value}: {reason}")

    def enforce_mode_constraints(self, state: DeltaState, proposed_action: str) -> bool:
        """
        Check if proposed action is allowed in current mode.

        Returns True if allowed, False otherwise.
        """
        current_mode = state.mode

        # RESEARCH: No real capital, only data analysis
        if current_mode == OperatingMode.RESEARCH:
            forbidden = ["EXECUTE_TRADE", "DEPLOY_CAPITAL", "MODIFY_PRODUCTION"]
            if any(f in proposed_action for f in forbidden):
                state.add_log(f"✗ BLOCKED: {proposed_action} not allowed in RESEARCH mode")
                return False

        # MICRO_LIVE: Limited capital, high logging
        elif current_mode == OperatingMode.MICRO_LIVE:
            # Check position size limits
            if "EXECUTE_TRADE" in proposed_action:
                max_position = 0.1 * state.portfolio.total_value  # 10% max

                # Would need to parse position size from action
                # For now, just warn
                state.add_log(f"⚠ MICRO_LIVE: Max position ${max_position:.2f}")

        # PRODUCTION: Full capability, but with audit trail
        elif current_mode == OperatingMode.PRODUCTION:
            state.add_log(f"✓ PRODUCTION: {proposed_action} authorized")

        return True

    def update(self, state: DeltaState):
        """
        Update mode controller: check for promotions or demotions.

        Should be called periodically (e.g., daily).
        """
        # Check demotion triggers first (safety)
        if self.check_demotion_triggers(state):
            self.force_demotion(state)

        # Then check promotion eligibility
        elif self.check_promotion_eligibility(state):
            self.attempt_promotion(state)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from delta_state import create_research_state, PhaseLock

    print("="*70)
    print("MODE CONTROLLER: Evidence-Gated Deployment")
    print("="*70)

    # Start in RESEARCH
    state = create_research_state()
    controller = ModeController()

    print(f"\n[INITIAL] Mode: {state.mode.value}")
    print(f"Capital: ${state.portfolio.cash:.2f}")

    # Add some E2-validated locks
    for i in range(5):
        lock = PhaseLock(
            pair=(f"ASSET_{i}", "SPY"),
            p=2, q=3,
            K=0.6 + i*0.05,
            Gamma_a=0.1, Gamma_b=0.1,
            Q_a=10, Q_b=10,
            eps_cap=0.8,
            eps_stab=0.7,
            zeta=0.3,
            delta_H_star=0.08 + i*0.02
        )
        lock.e0_passed = True
        lock.e1_passed = True
        lock.e2_passed = True
        lock.status = LockStatus.E2_PASSED

        state.locks.append(lock)

    # Try to promote
    print(f"\n[ATTEMPT PROMOTION]")
    controller.update(state)

    print(f"\n[AFTER UPDATE] Mode: {state.mode.value}")
    print(f"Capital: ${state.portfolio.cash:.2f}")

    # Check if action is allowed
    print(f"\n[ACTION GATE]")
    allowed = controller.enforce_mode_constraints(state, "EXECUTE_TRADE BUY AAPL 100")
    print(f"Action allowed: {allowed}")

    # Simulate performance degradation
    print(f"\n[SIMULATE BAD PERFORMANCE]")
    state.portfolio.max_drawdown = -0.12  # Breach threshold
    state.portfolio.sharpe = -0.2

    controller.update(state)

    print(f"\n[AFTER DEMOTION] Mode: {state.mode.value}")
    print(f"Capital: ${state.portfolio.cash:.2f}")

    print("\n" + "="*70)
    print("Mode transition history:")
    for timestamp, mode, reason in state.mode_history:
        print(f"  {timestamp}: {mode.value} - {reason}")
    print("="*70)
