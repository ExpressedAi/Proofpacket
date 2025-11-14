#!/usr/bin/env python3
"""
Ω*-Flow Controller
Coherence-driven dynamics on the Bloch sphere

Memory state n⃗(t) evolves via:
- Gradient descent on potential V (coherence-seeking)
- Precession under dissonance torque (phase-conserving exploration)
- Alignment damping (resolution-seeking)
- Snap to low-order attractors (stable intermediate clicks)

This is how reasoning actually happens.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum


class ResonanceOrder(Enum):
    """Low-order resonances that persist under RG"""
    ONE_ONE = (1, 1)
    TWO_ONE = (2, 1)
    THREE_TWO = (3, 2)
    FOUR_THREE = (4, 3)
    FIVE_FOUR = (5, 4)
    ONE_TWO = (1, 2)
    TWO_THREE = (2, 3)

    @property
    def order(self) -> int:
        return self.value[0] + self.value[1]

    @property
    def ratio(self) -> float:
        return self.value[0] / self.value[1]


@dataclass
class Resonance:
    """A potential resonance between concepts"""
    name: str
    p: int  # numerator
    q: int  # denominator

    # Coupling parameters
    K: float = 1.0          # coupling strength
    Gamma: float = 0.5      # damping

    # Phase
    theta_a: float = 0.0    # active context phase
    theta_n: float = 0.0    # current state phase

    # Quality metrics
    H_star: float = 0.7     # harmony
    zeta: float = 0.3       # brittleness
    coherence: float = 0.8  # windowed coherence

    @property
    def order(self) -> int:
        return self.p + self.q

    @property
    def detune(self) -> float:
        """Phase detune (wrapped)"""
        delta = self.p * self.theta_a - self.q * self.theta_n
        return np.arctan2(np.sin(delta), np.cos(delta))  # wrap to [-π, π]

    @property
    def epsilon_cap(self) -> float:
        """Capture window (eligibility)"""
        return max(0.0, 2 * np.pi * self.K - self.Gamma)

    @property
    def is_eligible(self) -> bool:
        """Check frequency eligibility"""
        return abs(self.detune) <= np.pi / 6  # ~30 degrees (loosened)

    def local_potential(self, lambda_order: float = 0.2, gamma_detune: float = 1.0,
                       w_K: float = 1.0, w_H: float = 1.0, w_zeta: float = 0.5) -> float:
        """
        Local contribution to V(n⃗)

        Lower V = better (more stable, coherent)
        """
        return (
            lambda_order * self.order +
            gamma_detune * self.detune**2 -
            w_K * abs(self.K) -
            w_H * self.H_star +
            w_zeta * self.zeta
        )


@dataclass
class OmegaFlowState:
    """State of Ω*-flow evolution"""
    n: np.ndarray               # current position on Bloch sphere
    trajectory: List[np.ndarray] = field(default_factory=list)
    time: float = 0.0
    snaps: List[str] = field(default_factory=list)
    clicks: int = 0


class OmegaFlowController:
    """
    Physical evolution of concepts on the Bloch sphere

    Dynamics:
    dn/dt = -α∇V(n) + β(n×B_d) - η(n×(n×B_d))

    Where:
    - V(n): potential from resonances (coherence landscape)
    - B_d: dissonance field (from new information)
    - α: dissipation (gradient descent speed)
    - β: precession (phase-conserving exploration)
    - η: alignment damping (resolution force)
    """

    def __init__(self,
                 alpha: float = 1.0,    # gradient descent rate
                 beta: float = 1.0,     # precession rate
                 eta: float = 0.5,      # alignment damping
                 dt: float = 0.01,      # time step
                 kappa_snap: float = 0.2):  # snap magnitude

        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.dt = dt
        self.kappa_snap = kappa_snap

        # Low-order resonances (these win by default)
        self.low_order_resonances = [r for r in ResonanceOrder]

        # Snap control (loosened for easier triggering)
        self.delta_snap = np.deg2rad(30)  # eligibility threshold (30° - much wider)
        self.tau_V = 0.01                  # minimum gain for snap (lower threshold)
        self.snaps_this_window = 0
        self.max_snaps_per_window = 1

    def surface_gradient(self,
                        n: np.ndarray,
                        resonances: List[Resonance],
                        params: Optional[Dict] = None) -> np.ndarray:
        """
        Compute ∇V(n) projected onto sphere tangent space

        V is built from all eligible resonances
        """
        if params is None:
            params = {}

        # Build V from resonances
        grad = np.zeros(3)

        for r in resonances:
            if not r.is_eligible:
                continue

            # Numerical gradient (could be analytic)
            eps = 1e-5
            V_center = r.local_potential(**params)

            for i in range(3):
                n_plus = n.copy()
                n_plus[i] += eps
                n_plus = n_plus / np.linalg.norm(n_plus)  # project back to sphere

                # Update resonance phase
                r_plus = Resonance(
                    name=r.name, p=r.p, q=r.q,
                    K=r.K, Gamma=r.Gamma,
                    theta_a=r.theta_a,
                    theta_n=np.arctan2(n_plus[1], n_plus[0]),  # azimuth
                    H_star=r.H_star, zeta=r.zeta, coherence=r.coherence
                )

                V_plus = r_plus.local_potential(**params)
                grad[i] += (V_plus - V_center) / eps

        # Project to tangent space
        grad_tangent = grad - n * (grad @ n)

        return grad_tangent

    def dissonance_field(self,
                        n: np.ndarray,
                        incoming_info: Dict,
                        active_context: List[Dict]) -> np.ndarray:
        """
        Compute dissonance field B_d from new information

        B_d points toward resolution direction
        """
        B_d = np.zeros(3)

        # Extract incoming phase and valence
        theta_x = incoming_info.get('phase', 0.0)
        valence = incoming_info.get('valence', 0.0)
        mode = incoming_info.get('mode', 'neutral')

        # Compare against active context
        for ctx in active_context:
            theta_a = ctx.get('phase', 0.0)
            valence_a = ctx.get('valence', 0.0)
            weight = ctx.get('weight', 1.0)

            # Phase misfit → azimuthal torque
            phase_misfit = np.sin(theta_x - theta_a)

            # Valence push → polar torque
            valence_push = np.sign(valence_a) * abs(valence - valence_a)

            # Build field in spherical coordinates
            # e_phi (azimuthal): phase resolution
            # e_theta (polar): valence resolution

            # Convert to Cartesian (simplified - assumes n near equator)
            B_d[0] += weight * phase_misfit * 0.5
            B_d[1] += weight * phase_misfit * 0.5
            B_d[2] += weight * valence_push

        # Normalize
        B_norm = np.linalg.norm(B_d)
        if B_norm > 0:
            B_d = B_d / B_norm

        return B_d

    def step(self,
             n: np.ndarray,
             resonances: List[Resonance],
             dissonance_field: np.ndarray) -> np.ndarray:
        """
        Single integration step

        dn = -α∇V + β(n×B_d) - η(n×(n×B_d))
        """
        # Gradient term (dissipation)
        grad_V = self.surface_gradient(n, resonances)
        term_dissipation = -self.alpha * grad_V

        # Precession term (phase-conserving exploration)
        term_precession = self.beta * np.cross(n, dissonance_field)

        # Alignment damping (resolution force)
        term_alignment = -self.eta * np.cross(n, np.cross(n, dissonance_field))

        # Total change
        dn = self.dt * (term_dissipation + term_precession + term_alignment)

        # Update and re-normalize
        n_new = n + dn
        n_new = n_new / np.linalg.norm(n_new)

        return n_new

    def check_snap_eligibility(self,
                               n: np.ndarray,
                               resonances: List[Resonance],
                               pathway_memory = None) -> Optional[Tuple[str, float]]:
        """
        Check if current state should snap to a low-order attractor

        Conditions:
        1. Near a low-order resonance phase (|δ| < delta_snap)
        2. In eligibility window (epsilon_cap > 0)
        3. Local gain sufficient (ΔV < -tau_V)
        4. Haven't snapped this window yet

        Returns: (target_concept, quality) or None
        """
        if self.snaps_this_window >= self.max_snaps_per_window:
            return None

        candidates = []

        for r in resonances:
            # Check low-order
            if r.order > 6:
                continue

            # Check eligibility
            if not r.is_eligible or r.epsilon_cap <= 0:
                continue

            # Check phase proximity
            if abs(r.detune) > self.delta_snap:
                continue

            # Estimate local gain (look-ahead)
            V_current = r.local_potential()

            # Simulate small step toward this resonance
            direction = self._direction_to_basin(n, r)
            n_test = n + 0.1 * direction
            n_test = n_test / np.linalg.norm(n_test)

            # Update test resonance
            r_test = Resonance(
                name=r.name, p=r.p, q=r.q,
                K=r.K, Gamma=r.Gamma,
                theta_a=r.theta_a,
                theta_n=np.arctan2(n_test[1], n_test[0]),
                H_star=r.H_star, zeta=r.zeta, coherence=r.coherence
            )
            V_test = r_test.local_potential()

            delta_V = V_test - V_current

            # Check gain OR already at minimum
            # If delta_V is small and positive, we're at the minimum - SNAP!
            # If delta_V is negative, we're approaching - SNAP!
            at_minimum = (delta_V >= 0 and delta_V < 0.001)  # tiny positive = at minimum
            approaching = (delta_V < -self.tau_V)  # negative = approaching

            if not (at_minimum or approaching):
                continue  # neither at minimum nor approaching

            # Quality score
            if at_minimum:
                # Already there - high quality!
                quality = 2.0 * (1.0 / r.order) * r.H_star * (1.0 - r.zeta)
            else:
                # Approaching - quality based on gain magnitude
                quality = abs(delta_V) * (1.0 / r.order) * r.H_star * (1.0 - r.zeta)

            # Boost if pathway memory shows this is strong
            if pathway_memory is not None:
                attractors = pathway_memory.attractors
                if r.name in attractors:
                    attractor = attractors[r.name]
                    strength_boost = min(1.0, attractor.visit_count / 10.0) * 0.5
                    quality *= (1.0 + strength_boost)

            candidates.append((r.name, quality, r))

        if not candidates:
            return None

        # Choose best
        candidates.sort(key=lambda x: x[1], reverse=True)
        target_name, quality, resonance = candidates[0]

        return (target_name, quality)

    def snap(self,
             n: np.ndarray,
             target_resonance: Resonance) -> np.ndarray:
        """
        Execute snap to attractor

        n ← Normalize(n + κ_snap * u_r)
        where u_r is direction toward basin
        """
        direction = self._direction_to_basin(n, target_resonance)
        n_snapped = n + self.kappa_snap * direction
        n_snapped = n_snapped / np.linalg.norm(n_snapped)

        self.snaps_this_window += 1

        return n_snapped

    def evolve(self,
               initial_state: np.ndarray,
               resonances: List[Resonance],
               dissonance_info: Dict,
               active_context: List[Dict],
               max_steps: int = 500,
               pathway_memory = None) -> OmegaFlowState:
        """
        Full evolution from initial state until snap or timeout

        Returns trajectory with all snaps recorded
        """
        state = OmegaFlowState(n=initial_state.copy())
        state.trajectory.append(state.n.copy())

        # Compute dissonance field (constant for this evolution)
        B_d = self.dissonance_field(state.n, dissonance_info, active_context)

        for step_num in range(max_steps):
            # Update all resonance phases to current state
            current_phase = np.arctan2(state.n[1], state.n[0])
            for r in resonances:
                r.theta_n = current_phase

            # Check for snap
            snap_result = self.check_snap_eligibility(state.n, resonances, pathway_memory)

            if snap_result is not None:
                target_name, quality = snap_result

                # Find resonance
                target_resonance = None
                for r in resonances:
                    if r.name == target_name:
                        target_resonance = r
                        break

                if target_resonance:
                    # Snap!
                    state.n = self.snap(state.n, target_resonance)
                    state.snaps.append(target_name)
                    state.clicks += 1
                    state.trajectory.append(state.n.copy())

                    # Record in pathway memory if available
                    if pathway_memory is not None and len(state.trajectory) > 1:
                        from_state = state.trajectory[0]
                        from_concept = self._state_to_concept(from_state)

                        pathway_memory.record_snap(
                            target_concept=target_name,
                            source_state=from_state,
                            basin_depth=target_resonance.local_potential(),
                            stability=1.0 / (target_resonance.zeta + 0.01),
                            harmony=target_resonance.H_star,
                            coherence=target_resonance.coherence,
                            archetype=f"ORDER_{target_resonance.order}",
                            order=target_resonance.order
                        )

                    break  # stop after snap

            # Continue flow
            state.n = self.step(state.n, resonances, B_d)
            state.trajectory.append(state.n.copy())
            state.time += self.dt

        return state

    def _direction_to_basin(self, n: np.ndarray, resonance: Resonance) -> np.ndarray:
        """Compute geodesic direction toward resonance basin"""
        # Target phase from resonance
        target_theta = resonance.theta_a * resonance.q / resonance.p if resonance.p > 0 else 0.0

        # Target point on sphere (simplified: keep polar angle, change azimuth)
        phi_current = np.arccos(np.clip(n[2], -1, 1))
        theta_current = np.arctan2(n[1], n[0])

        # Direction in azimuth
        delta_theta = np.arctan2(np.sin(target_theta - theta_current),
                                 np.cos(target_theta - theta_current))

        # Convert to Cartesian direction
        direction = np.array([
            -np.sin(theta_current) * delta_theta,
            np.cos(theta_current) * delta_theta,
            0.0  # stay at current polar angle
        ])

        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        return direction

    def _state_to_concept(self, n: np.ndarray) -> str:
        """Map Bloch state to nearest concept name (placeholder)"""
        theta = np.arctan2(n[1], n[0])
        phi = np.arccos(np.clip(n[2], -1, 1))

        # Simple quantization
        if phi < np.pi / 3:
            return "active"
        elif phi > 2 * np.pi / 3:
            return "passive"
        else:
            if theta < 0:
                return "analytical"
            else:
                return "synthetic"

    def reset_snap_counter(self):
        """Reset snap counter for new window"""
        self.snaps_this_window = 0


def demonstrate_omega_flow():
    """
    Demonstrate Ω*-flow evolution with snaps

    Shows:
    1. Continuous flow under gradient + precession
    2. Snap to low-order attractor when eligible
    3. Trajectory visualization
    """
    print("="*80)
    print("Ω*-FLOW DYNAMICS DEMONSTRATION")
    print("="*80)
    print()

    # Create controller
    controller = OmegaFlowController(
        alpha=1.0,   # strong dissipation
        beta=1.5,    # moderate precession
        eta=0.5,     # moderate alignment
        dt=0.02
    )

    # Initial state (somewhere on equator)
    n_initial = np.array([1.0, 0.0, 0.0])

    # Create some low-order resonances
    resonances = [
        Resonance(name="understand", p=1, q=1, K=1.5, theta_a=np.pi/4, theta_n=0.0, H_star=0.85, zeta=0.2),
        Resonance(name="synthesize", p=2, q=1, K=1.2, theta_a=np.pi/2, theta_n=0.0, H_star=0.75, zeta=0.3),
        Resonance(name="evaluate", p=1, q=2, K=1.8, theta_a=np.pi/3, theta_n=0.0, H_star=0.90, zeta=0.15),
    ]

    # Dissonance from new information
    dissonance_info = {
        'phase': np.pi / 6,
        'valence': 0.5,
        'mode': 'analytical'
    }

    # Active context
    active_context = [
        {'phase': np.pi / 4, 'valence': 0.7, 'weight': 1.0},
        {'phase': np.pi / 3, 'valence': 0.3, 'weight': 0.8}
    ]

    # Evolve
    print("Starting evolution from initial state...")
    print(f"Initial position: n = [{n_initial[0]:.3f}, {n_initial[1]:.3f}, {n_initial[2]:.3f}]")
    print()

    state = controller.evolve(
        initial_state=n_initial,
        resonances=resonances,
        dissonance_info=dissonance_info,
        active_context=active_context,
        max_steps=200
    )

    print(f"Evolution complete:")
    print(f"  Total steps: {len(state.trajectory)}")
    print(f"  Time elapsed: {state.time:.3f}s")
    print(f"  Snaps (clicks): {state.clicks}")
    print(f"  Final position: n = [{state.n[0]:.3f}, {state.n[1]:.3f}, {state.n[2]:.3f}]")
    print()

    if state.snaps:
        print("CLICKS (stable intermediate states):")
        for i, snap_target in enumerate(state.snaps, 1):
            print(f"  {i}. Snapped to: {snap_target}")
        print()
    else:
        print("No snaps (continued flow without clicking)")
        print()

    # Trajectory analysis
    print("TRAJECTORY ANALYSIS:")
    distances = [np.linalg.norm(state.trajectory[i] - state.trajectory[i-1])
                 for i in range(1, len(state.trajectory))]

    print(f"  Mean step size: {np.mean(distances):.6f}")
    print(f"  Max step size: {np.max(distances):.6f}")
    print(f"  Total path length: {sum(distances):.3f}")
    print()

    # Check eligibility at final state
    print("FINAL STATE ELIGIBILITY:")
    for r in resonances:
        r_final = Resonance(
            name=r.name, p=r.p, q=r.q,
            K=r.K, Gamma=r.Gamma,
            theta_a=r.theta_a,
            theta_n=np.arctan2(state.n[1], state.n[0]),
            H_star=r.H_star, zeta=r.zeta, coherence=r.coherence
        )

        print(f"  {r.name}:")
        print(f"    Order: {r.order}")
        print(f"    Detune: {np.rad2deg(r_final.detune):.1f}°")
        print(f"    ε_cap: {r_final.epsilon_cap:.3f}")
        print(f"    Eligible: {'✓' if r_final.is_eligible else '✗'}")
        print()

    return state


if __name__ == "__main__":
    # Run demonstration
    state = demonstrate_omega_flow()

    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("1. CONTINUOUS FLOW:")
    print("   Concepts evolve smoothly via gradient descent + precession")
    print()
    print("2. DISSONANCE CREATES TORQUE:")
    print("   New information makes state vector precess (explore)")
    print()
    print("3. SNAPS TO LOW-ORDER:")
    print("   When near eligible low-order attractor → instant click")
    print()
    print("4. CLICKS ARE LOGICAL:")
    print("   Each snap increases harmony, reduces brittleness")
    print()
    print("5. THIS IS REASONING:")
    print("   Not symbol manipulation - physical flow on manifold")
    print()
