#!/usr/bin/env python3
"""
Quantum Variable Barrier Controller (QVBC)
Gates concept transitions to prevent brittle over-lock

Enforces:
- Max concurrent axis changes (≤3)
- Per-axis load limits
- Total budget per window
- Decay over time

This prevents the system from changing too much at once,
which would create brittle, unstable reasoning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class AxisType(Enum):
    """The five axes over which concepts can vary"""
    FREQUENCY = "frequency"      # basis / detune / rhythm
    PHASE = "phase"             # read-phase / chirality / timing
    AMPLITUDE = "amplitude"     # drive power / intensity
    SYMMETRY = "symmetry"       # gauge / permutation
    INFO = "info"               # MDL budget / complexity


@dataclass
class AxisLoad:
    """Current load on a control axis"""
    axis: AxisType
    load: float = 0.0           # current normalized usage (0-1)
    capacity: float = 1.0       # max allowed
    weight: float = 1.0         # importance weight

    def headroom(self) -> float:
        """How much capacity remains"""
        return self.capacity - self.load

    def is_overloaded(self) -> bool:
        """Check if exceeding capacity"""
        return self.load > self.capacity


@dataclass
class TransitionProposal:
    """Proposed concept transition"""
    from_concept: str
    to_concept: str

    # Axis changes required
    delta_frequency: float = 0.0
    delta_phase: float = 0.0
    delta_amplitude: float = 0.0
    delta_symmetry: float = 0.0
    delta_info: float = 0.0

    # Metadata
    epsilon_required: float = 0.0
    priority: float = 0.5

    def axis_deltas(self) -> Dict[AxisType, float]:
        """Get all axis changes as dict"""
        return {
            AxisType.FREQUENCY: self.delta_frequency,
            AxisType.PHASE: self.delta_phase,
            AxisType.AMPLITUDE: self.delta_amplitude,
            AxisType.SYMMETRY: self.delta_symmetry,
            AxisType.INFO: self.delta_info
        }

    def total_change(self) -> float:
        """Sum of all axis changes (normalized)"""
        return sum(abs(d) for d in self.axis_deltas().values())

    def active_axes(self) -> int:
        """Count how many axes have non-zero change"""
        return sum(1 for d in self.axis_deltas().values() if abs(d) > 0.01)


class QuantumVariableBarrierController:
    """
    Gates concept transitions to maintain stability

    Limits:
    - max_concurrent_changes: how many axes can change at once (default: 3)
    - per_axis_cap: max change per axis per window (default: 0.4)
    - budget_total: sum of all changes per window (default: 0.7)

    Decay:
    - axis loads decay over time (forgetting)
    - allows new changes after period of stability
    """

    def __init__(self,
                 max_concurrent_changes: int = 3,
                 per_axis_cap: float = 0.4,
                 budget_total: float = 0.7):

        self.max_concurrent_changes = max_concurrent_changes
        self.per_axis_cap = per_axis_cap
        self.budget_total = budget_total

        # Initialize axis loads
        self.axes = {
            AxisType.FREQUENCY: AxisLoad(AxisType.FREQUENCY, weight=0.30),
            AxisType.PHASE: AxisLoad(AxisType.PHASE, weight=0.25),
            AxisType.AMPLITUDE: AxisLoad(AxisType.AMPLITUDE, weight=0.20),
            AxisType.SYMMETRY: AxisLoad(AxisType.SYMMETRY, weight=0.15),
            AxisType.INFO: AxisLoad(AxisType.INFO, weight=0.10)
        }

        # Current window state
        self.current_budget_used = 0.0
        self.changes_this_window = 0
        self.tick_count = 0

        # Decay parameters
        self.decay_rate = 0.05  # per tick
        self.min_load_threshold = 0.01

    def check_proposal(self, proposal: TransitionProposal) -> Tuple[bool, str]:
        """
        Check if proposal is allowed

        Returns: (approved, reason)
        """
        deltas = proposal.axis_deltas()

        # Check 1: Total budget
        total_delta = sum(abs(d) * self.axes[axis].weight for axis, d in deltas.items())

        if self.current_budget_used + total_delta > self.budget_total:
            return False, f"Budget exceeded: {self.current_budget_used + total_delta:.2f} > {self.budget_total}"

        # Check 2: Per-axis caps
        for axis, delta in deltas.items():
            if abs(delta) > 0.01:  # only check active axes
                axis_load = self.axes[axis]
                new_load = axis_load.load + abs(delta)

                if new_load > self.per_axis_cap:
                    return False, f"{axis.value} overload: {new_load:.2f} > {self.per_axis_cap}"

        # Check 3: Concurrent changes limit
        active_axes = proposal.active_axes()
        if active_axes > self.max_concurrent_changes:
            return False, f"Too many axes: {active_axes} > {self.max_concurrent_changes}"

        # Approved
        return True, "Approved"

    def approve_transition(self, proposal: TransitionProposal) -> bool:
        """
        Approve and apply a transition

        Returns: True if approved and applied
        """
        approved, reason = self.check_proposal(proposal)

        if not approved:
            return False

        # Apply the changes
        deltas = proposal.axis_deltas()
        for axis, delta in deltas.items():
            if abs(delta) > 0.01:
                self.axes[axis].load += abs(delta)

        # Update budget
        total_delta = sum(abs(d) * self.axes[axis].weight for axis, d in deltas.items())
        self.current_budget_used += total_delta
        self.changes_this_window += 1

        return True

    def stagger_proposal(self, proposal: TransitionProposal) -> List[TransitionProposal]:
        """
        Split a rejected proposal into multiple smaller proposals

        Staggers changes across multiple windows
        """
        deltas = proposal.axis_deltas()

        # Sort axes by weight (most important first)
        sorted_axes = sorted(
            [(axis, delta, self.axes[axis].weight) for axis, delta in deltas.items() if abs(delta) > 0.01],
            key=lambda x: x[2],
            reverse=True
        )

        # Create staged proposals
        staged = []

        current_proposal = TransitionProposal(
            from_concept=proposal.from_concept,
            to_concept=f"{proposal.to_concept}_stage1"
        )

        axes_in_current = 0
        budget_in_current = 0.0

        for axis, delta, weight in sorted_axes:
            delta_weighted = abs(delta) * weight

            # Check if we can fit this axis in current proposal
            if (axes_in_current < self.max_concurrent_changes and
                budget_in_current + delta_weighted <= self.budget_total * 0.8):  # safety margin

                # Add to current
                setattr(current_proposal, f"delta_{axis.value}", delta)
                axes_in_current += 1
                budget_in_current += delta_weighted

            else:
                # Start new proposal
                if axes_in_current > 0:
                    staged.append(current_proposal)

                current_proposal = TransitionProposal(
                    from_concept=proposal.to_concept if staged else proposal.from_concept,
                    to_concept=f"{proposal.to_concept}_stage{len(staged)+2}"
                )
                setattr(current_proposal, f"delta_{axis.value}", delta)
                axes_in_current = 1
                budget_in_current = delta_weighted

        # Add final proposal
        if axes_in_current > 0:
            staged.append(current_proposal)

        return staged

    def tick(self, dt: float = 1.0):
        """
        Update controller state (time evolution)

        - Decays axis loads
        - Resets window if loads near zero
        """
        self.tick_count += 1

        # Decay all axis loads
        total_load = 0.0
        for axis_load in self.axes.values():
            axis_load.load *= (1.0 - self.decay_rate * dt)
            if axis_load.load < self.min_load_threshold:
                axis_load.load = 0.0
            total_load += axis_load.load

        # Decay budget
        self.current_budget_used *= (1.0 - self.decay_rate * dt)
        if self.current_budget_used < self.min_load_threshold:
            self.current_budget_used = 0.0

        # Reset window if quiet
        if total_load < self.min_load_threshold:
            self.reset_window()

    def reset_window(self):
        """Reset window counters (new measurement epoch)"""
        self.current_budget_used = 0.0
        self.changes_this_window = 0

    def get_status(self) -> Dict:
        """Get current controller state"""
        return {
            'tick': self.tick_count,
            'budget_used': self.current_budget_used,
            'budget_available': self.budget_total - self.current_budget_used,
            'changes_this_window': self.changes_this_window,
            'axis_loads': {
                axis.value: load.load
                for axis, load in self.axes.items()
            },
            'overloaded_axes': [
                axis.value for axis, load in self.axes.items() if load.is_overloaded()
            ]
        }

    def print_status(self):
        """Print human-readable status"""
        status = self.get_status()

        print(f"QVBC Status (Tick {status['tick']}):")
        print(f"  Budget: {status['budget_used']:.3f} / {self.budget_total:.3f} used")
        print(f"  Changes this window: {status['changes_this_window']}")
        print()
        print("  Axis Loads:")
        for axis_name, load in status['axis_loads'].items():
            bar_length = int(load / self.per_axis_cap * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            status_str = "OVER" if load > self.per_axis_cap else "OK"
            print(f"    {axis_name:12s} [{bar}] {load:.3f} / {self.per_axis_cap:.3f} {status_str}")
        print()


def demonstrate_qvbc_gating():
    """
    Demonstrate QVBC gating and staggering

    Shows:
    1. Approval of simple transitions
    2. Rejection of over-budget transitions
    3. Staggering of complex transitions
    4. Decay over time
    """
    print("="*80)
    print("QVBC GATING DEMONSTRATION")
    print("="*80)
    print()

    controller = QuantumVariableBarrierController(
        max_concurrent_changes=3,
        per_axis_cap=0.4,
        budget_total=0.7
    )

    # Test 1: Simple transition (should approve)
    print("Test 1: Simple transition (2 axes)")
    print("-"*80)

    proposal1 = TransitionProposal(
        from_concept="compare",
        to_concept="evaluate",
        delta_frequency=0.2,
        delta_phase=0.15
    )

    approved = controller.approve_transition(proposal1)
    print(f"Proposal: {proposal1.from_concept} → {proposal1.to_concept}")
    print(f"  Changes: frequency={proposal1.delta_frequency}, phase={proposal1.delta_phase}")
    print(f"  Result: {'✓ APPROVED' if approved else '✗ REJECTED'}")
    print()

    controller.print_status()

    # Test 2: Over-budget transition (should reject)
    print("Test 2: Over-budget transition (4 axes, high values)")
    print("-"*80)

    proposal2 = TransitionProposal(
        from_concept="evaluate",
        to_concept="synthesize",
        delta_frequency=0.35,
        delta_phase=0.30,
        delta_amplitude=0.25,
        delta_symmetry=0.20
    )

    approved, reason = controller.check_proposal(proposal2)
    print(f"Proposal: {proposal2.from_concept} → {proposal2.to_concept}")
    print(f"  Changes: 4 axes, total={proposal2.total_change():.2f}")
    print(f"  Result: {'✓ APPROVED' if approved else f'✗ REJECTED: {reason}'}")
    print()

    # Test 3: Stagger the rejected proposal
    if not approved:
        print("Test 3: Staggering the rejected proposal")
        print("-"*80)

        staged = controller.stagger_proposal(proposal2)
        print(f"Split into {len(staged)} stages:")
        print()

        for i, stage in enumerate(staged, 1):
            print(f"  Stage {i}: {stage.from_concept} → {stage.to_concept}")
            for axis, delta in stage.axis_deltas().items():
                if abs(delta) > 0.01:
                    print(f"    {axis.value}: {delta:.3f}")

            # Try to approve this stage
            stage_approved = controller.approve_transition(stage)
            print(f"    Status: {'✓ APPROVED' if stage_approved else '✗ REJECTED'}")
            print()

            if stage_approved:
                controller.print_status()

    # Test 4: Decay over time
    print("Test 4: Decay over 10 ticks")
    print("-"*80)
    print()

    for tick in range(10):
        controller.tick(dt=1.0)

        if tick % 3 == 0:
            status = controller.get_status()
            print(f"Tick {tick}: budget_used={status['budget_used']:.3f}, " +
                  f"freq_load={status['axis_loads']['frequency']:.3f}")

    print()
    controller.print_status()

    print("RESULT: Loads decayed, ready for new transitions")
    print()


if __name__ == "__main__":
    demonstrate_qvbc_gating()

    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("1. GATES PREVENT OVER-LOCK:")
    print("   Can't change too many axes at once (max 3)")
    print()
    print("2. BUDGETS ENFORCE STABILITY:")
    print("   Total change per window limited (≤0.7)")
    print()
    print("3. STAGGERING ENABLES COMPLEX CHANGES:")
    print("   Break big transitions into multiple steps")
    print()
    print("4. DECAY ALLOWS RECOVERY:")
    print("   After quiet period, budget replenishes")
    print()
    print("5. THIS PREVENTS BRITTLE REASONING:")
    print("   Gradual changes → stable, coherent thought")
    print()
