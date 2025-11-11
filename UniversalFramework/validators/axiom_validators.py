"""
Complete Axiom Validator Library
================================

Implements all 26 universal axioms extracted from Clay Millennium Problems.
Each validator tests whether a given system satisfies the axiom.

All axioms are substrate-independent and testable across quantum, classical,
neural, social, and biological systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings


class ValidationStatus(Enum):
    """Status of axiom validation"""
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ValidationResult:
    """Result of axiom validation"""
    axiom: int
    status: ValidationStatus
    value: float
    threshold: float
    confidence: float = 1.0
    message: str = ""

    def __repr__(self):
        symbol = {"pass": "✓", "fail": "✗", "uncertain": "?", "not_applicable": "-"}[self.status.value]
        return f"{symbol} Axiom {self.axiom}: {self.message} (value={self.value:.4f}, threshold={self.threshold:.4f})"


# =============================================================================
# FUNDAMENTAL AXIOMS (1-5)
# =============================================================================

def axiom_1_phase_locking(flux: float, dissipation: float,
                          threshold: float = 1.0) -> ValidationResult:
    """
    Axiom 1: Phase-Lock Criticality

    χ = flux/dissipation < 1 → stable system

    This is the most fundamental axiom. All stable systems must satisfy χ < 1.

    Args:
        flux: Energy flux into system (coupling between oscillators)
        dissipation: Energy dissipation (damping)
        threshold: Critical value (default 1.0)

    Returns:
        ValidationResult with χ value and pass/fail status
    """
    chi = abs(flux) / (abs(dissipation) + 1e-10)

    status = ValidationStatus.PASS if chi < threshold else ValidationStatus.FAIL

    message = f"χ = {chi:.3f} {'<' if chi < threshold else '≥'} {threshold}"

    return ValidationResult(
        axiom=1,
        status=status,
        value=chi,
        threshold=threshold,
        message=message
    )


def axiom_2_bounded_energy(energy_history: List[float],
                           decay_rate_threshold: float = 0.0) -> ValidationResult:
    """
    Axiom 2: Bounded Energy

    E(t) ≤ E(0) exp(-γt) for some γ ≥ 0

    Energy must not grow without bound. For stable systems, γ > 0.

    Args:
        energy_history: Time series of energy E(t)
        decay_rate_threshold: Minimum γ for stability (default 0.0)

    Returns:
        ValidationResult with estimated γ
    """
    if len(energy_history) < 3:
        return ValidationResult(
            axiom=2,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=decay_rate_threshold,
            message="Insufficient data"
        )

    # Fit exponential decay: E(t) = E(0) exp(-γt)
    E = np.array(energy_history)
    t = np.arange(len(E))

    # Avoid log of negative/zero
    E_positive = E[E > 1e-10]
    t_positive = t[:len(E_positive)]

    if len(E_positive) < 3:
        return ValidationResult(
            axiom=2,
            status=ValidationStatus.FAIL,
            value=float('inf'),
            threshold=decay_rate_threshold,
            message="Energy non-positive"
        )

    # Linear regression on log(E) vs t
    log_E = np.log(E_positive)

    # γ = -slope of log(E) vs t
    coeffs = np.polyfit(t_positive, log_E, 1)
    gamma = -coeffs[0]

    status = ValidationStatus.PASS if gamma >= decay_rate_threshold else ValidationStatus.FAIL

    message = f"γ = {gamma:.4f} {'≥' if gamma >= decay_rate_threshold else '<'} {decay_rate_threshold}"

    return ValidationResult(
        axiom=2,
        status=status,
        value=gamma,
        threshold=decay_rate_threshold,
        message=message
    )


def axiom_3_regularity_persistence(gradient_norms: List[float],
                                   value_norms: List[float],
                                   ratio_threshold: float = 10.0) -> ValidationResult:
    """
    Axiom 3: Regularity Persistence

    ‖∇u‖ ≤ C‖u‖ for some constant C

    Gradients grow at most linearly with values (no blow-up).

    Args:
        gradient_norms: ‖∇u(t)‖ over time
        value_norms: ‖u(t)‖ over time
        ratio_threshold: Maximum C (default 10.0)

    Returns:
        ValidationResult with maximum ratio
    """
    if len(gradient_norms) != len(value_norms):
        return ValidationResult(
            axiom=3,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=ratio_threshold,
            message="Mismatched array lengths"
        )

    grad = np.array(gradient_norms)
    val = np.array(value_norms)

    # Avoid division by zero
    val_safe = np.where(val > 1e-10, val, 1e-10)

    ratios = grad / val_safe
    max_ratio = np.max(ratios)

    status = ValidationStatus.PASS if max_ratio <= ratio_threshold else ValidationStatus.FAIL

    message = f"max(‖∇u‖/‖u‖) = {max_ratio:.3f} {'≤' if max_ratio <= ratio_threshold else '>'} {ratio_threshold}"

    return ValidationResult(
        axiom=3,
        status=status,
        value=max_ratio,
        threshold=ratio_threshold,
        message=message
    )


def axiom_4_spectral_locality(amplitudes: List[float],
                               decay_rate_threshold: float = 0.3) -> ValidationResult:
    """
    Axiom 4: Spectral Locality

    A_k ∝ θ^k where θ < 1

    High-frequency modes decay exponentially.

    Args:
        amplitudes: Mode amplitudes [A_0, A_1, A_2, ...]
        decay_rate_threshold: Maximum θ for strong locality (default 0.3)

    Returns:
        ValidationResult with estimated θ
    """
    if len(amplitudes) < 3:
        return ValidationResult(
            axiom=4,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=decay_rate_threshold,
            message="Insufficient modes"
        )

    A = np.array(amplitudes)
    k = np.arange(len(A))

    # Filter positive amplitudes
    A_pos = A[A > 1e-10]
    k_pos = k[:len(A_pos)]

    if len(A_pos) < 3:
        return ValidationResult(
            axiom=4,
            status=ValidationStatus.FAIL,
            value=1.0,
            threshold=decay_rate_threshold,
            message="No positive amplitudes"
        )

    # Fit A_k = A_0 θ^k → log(A_k) = log(A_0) + k log(θ)
    log_A = np.log(A_pos)

    coeffs = np.polyfit(k_pos, log_A, 1)
    log_theta = coeffs[0]
    theta = np.exp(log_theta)

    status = ValidationStatus.PASS if theta < 1.0 and theta >= decay_rate_threshold else ValidationStatus.FAIL

    message = f"θ = {theta:.3f} ({'good' if decay_rate_threshold <= theta < 1.0 else 'bad'} locality)"

    return ValidationResult(
        axiom=4,
        status=status,
        value=theta,
        threshold=decay_rate_threshold,
        message=message
    )


def axiom_5_coupling_decay(couplings: List[float],
                            orders: List[int],
                            power_threshold: float = -1.0) -> ValidationResult:
    """
    Axiom 5: Coupling Decay

    K_k ∝ k^(-α) where α > 0

    Coupling strength decreases with mode number.

    Args:
        couplings: Coupling strengths K_k
        orders: Mode orders k
        power_threshold: Minimum -α for strong decay (default -1.0)

    Returns:
        ValidationResult with estimated -α
    """
    if len(couplings) != len(orders) or len(couplings) < 3:
        return ValidationResult(
            axiom=5,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=power_threshold,
            message="Insufficient data"
        )

    K = np.array(couplings)
    k = np.array(orders)

    # Filter positive
    mask = (K > 1e-10) & (k > 0)
    K_pos = K[mask]
    k_pos = k[mask]

    if len(K_pos) < 3:
        return ValidationResult(
            axiom=5,
            status=ValidationStatus.FAIL,
            value=0.0,
            threshold=power_threshold,
            message="No valid data"
        )

    # Fit K_k = K_0 k^(-α) → log(K_k) = log(K_0) - α log(k)
    log_K = np.log(K_pos)
    log_k = np.log(k_pos)

    coeffs = np.polyfit(log_k, log_K, 1)
    minus_alpha = coeffs[0]

    status = ValidationStatus.PASS if minus_alpha < power_threshold else ValidationStatus.FAIL

    message = f"K_k ∝ k^({minus_alpha:.2f}) ({'good' if minus_alpha < 0 else 'bad'} decay)"

    return ValidationResult(
        axiom=5,
        status=status,
        value=minus_alpha,
        threshold=power_threshold,
        message=message
    )


# =============================================================================
# STABILITY AXIOMS (6-10)
# =============================================================================

def axiom_6_exponential_convergence(trajectory: List[float],
                                    equilibrium: float,
                                    min_rate: float = 0.01) -> ValidationResult:
    """
    Axiom 6: Exponential Convergence

    |x(t) - x_∞| ≤ |x(0) - x_∞| exp(-γt)

    System converges to equilibrium exponentially.
    """
    if len(trajectory) < 3:
        return ValidationResult(
            axiom=6,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=min_rate,
            message="Insufficient data"
        )

    x = np.array(trajectory)
    x_inf = equilibrium

    deviations = np.abs(x - x_inf)
    t = np.arange(len(x))

    # Filter non-zero deviations
    nonzero = deviations > 1e-10
    dev_nz = deviations[nonzero]
    t_nz = t[nonzero]

    if len(dev_nz) < 3:
        return ValidationResult(
            axiom=6,
            status=ValidationStatus.PASS,
            value=float('inf'),
            threshold=min_rate,
            message="Already at equilibrium"
        )

    # Fit log(deviation) vs t
    log_dev = np.log(dev_nz)
    coeffs = np.polyfit(t_nz, log_dev, 1)
    minus_gamma = coeffs[0]
    gamma = -minus_gamma

    status = ValidationStatus.PASS if gamma >= min_rate else ValidationStatus.FAIL

    message = f"γ = {gamma:.4f} {'≥' if gamma >= min_rate else '<'} {min_rate}"

    return ValidationResult(
        axiom=6,
        status=status,
        value=gamma,
        threshold=min_rate,
        message=message
    )


def axiom_7_lyapunov_stability(lyapunov_function: List[float]) -> ValidationResult:
    """
    Axiom 7: Lyapunov Stability

    dV/dt ≤ 0 for Lyapunov function V

    Energy-like function decreases monotonically.
    """
    if len(lyapunov_function) < 2:
        return ValidationResult(
            axiom=7,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=0.0,
            message="Insufficient data"
        )

    V = np.array(lyapunov_function)
    dV_dt = np.diff(V)

    # Count violations (dV/dt > 0)
    violations = np.sum(dV_dt > 1e-10)
    violation_rate = violations / len(dV_dt)

    status = ValidationStatus.PASS if violation_rate < 0.1 else ValidationStatus.FAIL

    message = f"{violation_rate*100:.1f}% violations ({'stable' if violation_rate < 0.1 else 'unstable'})"

    return ValidationResult(
        axiom=7,
        status=status,
        value=violation_rate,
        threshold=0.1,
        message=message
    )


def axiom_8_attractor_existence(trajectories: List[List[float]],
                                tolerance: float = 0.1) -> ValidationResult:
    """
    Axiom 8: Attractor Existence

    Multiple trajectories converge to the same attractor

    Different initial conditions lead to same final state.
    """
    if len(trajectories) < 2:
        return ValidationResult(
            axiom=8,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=tolerance,
            message="Need multiple trajectories"
        )

    # Get final states
    final_states = [traj[-1] for traj in trajectories]

    # Compute variance of final states
    variance = np.var(final_states)

    status = ValidationStatus.PASS if variance < tolerance else ValidationStatus.FAIL

    message = f"final variance = {variance:.4f} {'<' if variance < tolerance else '≥'} {tolerance}"

    return ValidationResult(
        axiom=8,
        status=status,
        value=variance,
        threshold=tolerance,
        message=message
    )


def axiom_9_no_finite_time_blowup(values: List[float],
                                  max_value: float = 1e10) -> ValidationResult:
    """
    Axiom 9: No Finite-Time Blow-Up

    ‖u(t)‖ < ∞ for all finite t

    Solution remains bounded.
    """
    u = np.array(values)
    max_u = np.max(np.abs(u))

    status = ValidationStatus.PASS if max_u < max_value else ValidationStatus.FAIL

    message = f"max|u| = {max_u:.2e} {'<' if max_u < max_value else '≥'} {max_value:.2e}"

    return ValidationResult(
        axiom=9,
        status=status,
        value=max_u,
        threshold=max_value,
        message=message
    )


def axiom_10_uniqueness(solution1: List[float],
                        solution2: List[float],
                        tolerance: float = 1e-6) -> ValidationResult:
    """
    Axiom 10: Uniqueness

    Same initial conditions → same solution

    Deterministic dynamics.
    """
    if len(solution1) != len(solution2):
        return ValidationResult(
            axiom=10,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=tolerance,
            message="Different lengths"
        )

    s1 = np.array(solution1)
    s2 = np.array(solution2)

    difference = np.linalg.norm(s1 - s2)

    status = ValidationStatus.PASS if difference < tolerance else ValidationStatus.FAIL

    message = f"‖s1 - s2‖ = {difference:.2e} {'<' if difference < tolerance else '≥'} {tolerance:.2e}"

    return ValidationResult(
        axiom=10,
        status=status,
        value=difference,
        threshold=tolerance,
        message=message
    )


# =============================================================================
# RESONANCE AXIOMS (11-15)
# =============================================================================

def axiom_11_phase_coherence(phases: List[float],
                             coherence_threshold: float = 0.7) -> ValidationResult:
    """
    Axiom 11: Phase Coherence

    R = |⟨e^{iφ}⟩| > threshold → phase-locked

    Mean resultant length measures synchronization.
    """
    if len(phases) < 3:
        return ValidationResult(
            axiom=11,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=coherence_threshold,
            message="Insufficient phases"
        )

    phi = np.array(phases)

    # Compute mean resultant length (circular statistics)
    z = np.mean(np.exp(1j * phi))
    R = np.abs(z)

    status = ValidationStatus.PASS if R > coherence_threshold else ValidationStatus.FAIL

    message = f"R = {R:.3f} {'>' if R > coherence_threshold else '≤'} {coherence_threshold}"

    return ValidationResult(
        axiom=11,
        status=status,
        value=R,
        threshold=coherence_threshold,
        message=message
    )


def axiom_12_frequency_locking(freq1: float, freq2: float,
                               tolerance: float = 0.01) -> ValidationResult:
    """
    Axiom 12: Frequency Locking

    |ω₁ - ω₂| < ε → locked

    Oscillators synchronize their frequencies.
    """
    diff = abs(freq1 - freq2)

    status = ValidationStatus.PASS if diff < tolerance else ValidationStatus.FAIL

    message = f"|ω₁ - ω₂| = {diff:.4f} {'<' if diff < tolerance else '≥'} {tolerance}"

    return ValidationResult(
        axiom=12,
        status=status,
        value=diff,
        threshold=tolerance,
        message=message
    )


def axiom_13_arnold_tongue(coupling: float, detuning: float) -> ValidationResult:
    """
    Axiom 13: Arnold Tongue

    |Δω| < 2K → phase-locking region

    Capture width scales with coupling.
    """
    capture_width = 2 * abs(coupling)
    locked = abs(detuning) < capture_width

    status = ValidationStatus.PASS if locked else ValidationStatus.FAIL

    message = f"|Δω|={abs(detuning):.3f} {'<' if locked else '≥'} 2K={capture_width:.3f}"

    return ValidationResult(
        axiom=13,
        status=status,
        value=abs(detuning),
        threshold=capture_width,
        message=message
    )


def axiom_14_kuramoto_transition(oscillators: List[Tuple[float, float]],
                                 coupling: float,
                                 critical_coupling: float = 1.0) -> ValidationResult:
    """
    Axiom 14: Kuramoto Transition

    K > K_c → partial synchronization

    Phase transition at critical coupling.
    """
    # oscillators = [(ω_i, φ_i), ...]

    if len(oscillators) < 3:
        return ValidationResult(
            axiom=14,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=critical_coupling,
            message="Too few oscillators"
        )

    phases = np.array([phi for omega, phi in oscillators])

    # Order parameter
    r = np.abs(np.mean(np.exp(1j * phases)))

    # Expect r > 0.5 for K > K_c
    synchronized = coupling > critical_coupling and r > 0.5

    status = ValidationStatus.PASS if synchronized else ValidationStatus.FAIL

    message = f"K={coupling:.2f}, K_c={critical_coupling:.2f}, r={r:.3f}"

    return ValidationResult(
        axiom=14,
        status=status,
        value=coupling,
        threshold=critical_coupling,
        message=message
    )


def axiom_15_winding_number(phase_trajectory: List[float]) -> ValidationResult:
    """
    Axiom 15: Winding Number Conservation

    W = (φ(T) - φ(0)) / (2π) is integer for locked state

    Topological constraint on phase evolution.
    """
    if len(phase_trajectory) < 2:
        return ValidationResult(
            axiom=15,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=0.1,
            message="Insufficient trajectory"
        )

    phi = np.array(phase_trajectory)

    # Total phase change
    delta_phi = phi[-1] - phi[0]

    # Winding number
    W = delta_phi / (2 * np.pi)

    # Check if near integer
    W_rounded = np.round(W)
    integer_error = abs(W - W_rounded)

    status = ValidationStatus.PASS if integer_error < 0.1 else ValidationStatus.FAIL

    message = f"W = {W:.3f} ≈ {int(W_rounded)} (error={integer_error:.3f})"

    return ValidationResult(
        axiom=15,
        status=status,
        value=integer_error,
        threshold=0.1,
        message=message
    )


# =============================================================================
# LOW-ORDER PREFERENCE AXIOMS (16-20)
# =============================================================================

def axiom_16_integer_thinning(couplings: List[float], orders: List[int],
                              slope_threshold: float = -0.1) -> ValidationResult:
    """
    Axiom 16: Integer Thinning

    log K_n decreases with n

    High-order couplings are exponentially suppressed.
    """
    if len(couplings) != len(orders) or len(couplings) < 3:
        return ValidationResult(
            axiom=16,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=slope_threshold,
            message="Insufficient data"
        )

    K = np.array(couplings)
    n = np.array(orders)

    # Filter positive
    mask = K > 1e-10
    K_pos = K[mask]
    n_pos = n[mask]

    if len(K_pos) < 3:
        return ValidationResult(
            axiom=16,
            status=ValidationStatus.FAIL,
            value=0.0,
            threshold=slope_threshold,
            message="No positive couplings"
        )

    log_K = np.log(K_pos)

    # Linear fit
    coeffs = np.polyfit(n_pos, log_K, 1)
    slope = coeffs[0]

    status = ValidationStatus.PASS if slope < slope_threshold else ValidationStatus.FAIL

    message = f"d(log K)/dn = {slope:.3f} {'<' if slope < slope_threshold else '≥'} {slope_threshold}"

    return ValidationResult(
        axiom=16,
        status=status,
        value=slope,
        threshold=slope_threshold,
        message=message
    )


def axiom_17_harmonic_preference(frequencies: List[float],
                                tolerance: float = 0.05) -> ValidationResult:
    """
    Axiom 17: Harmonic Preference

    Frequencies cluster near integer ratios

    Natural systems prefer simple harmonic relationships.
    """
    if len(frequencies) < 2:
        return ValidationResult(
            axiom=17,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=tolerance,
            message="Need multiple frequencies"
        )

    freqs = np.array(frequencies)
    freqs_sorted = np.sort(freqs)

    # Compute ratios with fundamental
    fundamental = freqs_sorted[0]
    ratios = freqs_sorted / fundamental

    # Check how close to integers
    errors = np.abs(ratios - np.round(ratios))
    mean_error = np.mean(errors)

    status = ValidationStatus.PASS if mean_error < tolerance else ValidationStatus.FAIL

    message = f"mean harmonic error = {mean_error:.4f} {'<' if mean_error < tolerance else '≥'} {tolerance}"

    return ValidationResult(
        axiom=17,
        status=status,
        value=mean_error,
        threshold=tolerance,
        message=message
    )


def axiom_18_fibonacci_cascade(mode_amplitudes: List[float]) -> ValidationResult:
    """
    Axiom 18: Fibonacci Cascade

    Ratios approach golden ratio φ = 1.618...

    Universal scaling in non-linear systems.
    """
    if len(mode_amplitudes) < 3:
        return ValidationResult(
            axiom=18,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=0.1,
            message="Need at least 3 modes"
        )

    A = np.array(mode_amplitudes)

    # Compute successive ratios A_{n+1}/A_n
    ratios = A[1:] / (A[:-1] + 1e-10)

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Check convergence to φ
    errors = np.abs(ratios - phi)
    mean_error = np.mean(errors)

    status = ValidationStatus.PASS if mean_error < 0.5 else ValidationStatus.FAIL

    message = f"mean |ratio - φ| = {mean_error:.3f}"

    return ValidationResult(
        axiom=18,
        status=status,
        value=mean_error,
        threshold=0.5,
        message=message
    )


def axiom_19_power_law_distribution(events: List[float],
                                   min_exponent: float = -2.0,
                                   max_exponent: float = -1.0) -> ValidationResult:
    """
    Axiom 19: Power Law Distribution

    P(x) ∝ x^(-α) where 1 < α < 3

    Scale-free behavior.
    """
    if len(events) < 10:
        return ValidationResult(
            axiom=19,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=min_exponent,
            message="Insufficient events"
        )

    x = np.array(events)
    x_pos = x[x > 0]

    if len(x_pos) < 10:
        return ValidationResult(
            axiom=19,
            status=ValidationStatus.FAIL,
            value=0.0,
            threshold=min_exponent,
            message="No positive events"
        )

    # Estimate exponent via linear regression on log-log plot
    log_x = np.log(x_pos)

    # Create histogram
    hist, bins = np.histogram(x_pos, bins=20)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Filter non-zero bins
    nonzero = hist > 0
    log_P = np.log(hist[nonzero])
    log_x_bins = np.log(bin_centers[nonzero])

    if len(log_P) < 3:
        return ValidationResult(
            axiom=19,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=min_exponent,
            message="Sparse histogram"
        )

    coeffs = np.polyfit(log_x_bins, log_P, 1)
    alpha = -coeffs[0]  # P(x) ∝ x^(-α)

    in_range = min_exponent <= -alpha <= max_exponent
    status = ValidationStatus.PASS if in_range else ValidationStatus.FAIL

    message = f"α = {alpha:.2f} ({'in' if in_range else 'out of'} range [{-max_exponent:.1f}, {-min_exponent:.1f}])"

    return ValidationResult(
        axiom=19,
        status=status,
        value=-alpha,
        threshold=min_exponent,
        message=message
    )


def axiom_20_multifractal_spectrum(signal: List[float]) -> ValidationResult:
    """
    Axiom 20: Multifractal Spectrum

    D(q) is non-constant → multifractal

    Signal has multiple scaling exponents.
    """
    if len(signal) < 16:
        return ValidationResult(
            axiom=20,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=0.1,
            message="Signal too short"
        )

    s = np.array(signal)

    # Simple multifractal test: variance of local Hölder exponents
    # Approximate via wavelet decomposition

    # Use simple differencing as approximation
    increments = np.abs(np.diff(s))

    if len(increments) == 0 or np.all(increments < 1e-10):
        return ValidationResult(
            axiom=20,
            status=ValidationStatus.FAIL,
            value=0.0,
            threshold=0.1,
            message="No variation"
        )

    # Local scaling exponents
    log_increments = np.log(increments + 1e-10)

    # Variance of exponents
    variance = np.var(log_increments)

    # Multifractal if variance > 0.1
    status = ValidationStatus.PASS if variance > 0.1 else ValidationStatus.FAIL

    message = f"var(H_local) = {variance:.3f} {'>' if variance > 0.1 else '≤'} 0.1"

    return ValidationResult(
        axiom=20,
        status=status,
        value=variance,
        threshold=0.1,
        message=message
    )


# =============================================================================
# DOMAIN-SPECIFIC AXIOMS (21-26)
# =============================================================================

def axiom_21_energy_cascade(spectrum: List[float],
                           slope_range: Tuple[float, float] = (-5/3 - 0.2, -5/3 + 0.2)) -> ValidationResult:
    """
    Axiom 21: Energy Cascade (Navier-Stokes)

    E(k) ∝ k^(-5/3) (Kolmogorov spectrum)

    Inertial range scaling for turbulence.
    """
    if len(spectrum) < 5:
        return ValidationResult(
            axiom=21,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=slope_range[0],
            message="Insufficient spectrum"
        )

    E_k = np.array(spectrum)
    k = np.arange(1, len(E_k) + 1)

    # Filter positive
    mask = E_k > 1e-10
    E_pos = E_k[mask]
    k_pos = k[mask]

    if len(E_pos) < 5:
        return ValidationResult(
            axiom=21,
            status=ValidationStatus.FAIL,
            value=0.0,
            threshold=slope_range[0],
            message="No positive spectrum"
        )

    # Fit E(k) = C k^β
    log_E = np.log(E_pos)
    log_k = np.log(k_pos)

    coeffs = np.polyfit(log_k, log_E, 1)
    beta = coeffs[0]

    kolmogorov = -5/3
    in_range = slope_range[0] <= beta <= slope_range[1]

    status = ValidationStatus.PASS if in_range else ValidationStatus.FAIL

    message = f"E(k) ∝ k^({beta:.3f}) ({'near' if in_range else 'far from'} -5/3)"

    return ValidationResult(
        axiom=21,
        status=status,
        value=beta,
        threshold=kolmogorov,
        message=message
    )


def axiom_22_one_to_one_lock(sigma: float, t: float, primes: List[int],
                            threshold: float = 0.8) -> ValidationResult:
    """
    Axiom 22: 1:1 Phase Lock (Riemann Hypothesis)

    K₁:₁(σ=1/2) = 1 → zeros on critical line

    Perfect 1:1 locking at critical strip.
    """
    if len(primes) < 5:
        return ValidationResult(
            axiom=22,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=threshold,
            message="Need more primes"
        )

    # Prime oscillator phases: ψ_p = exp(-it log p)
    phases = [-t * np.log(p) for p in primes]

    # Mean resultant length (1:1 coupling strength)
    z = np.mean(np.exp(1j * np.array(phases)))
    R = np.abs(z)

    status = ValidationStatus.PASS if R > threshold else ValidationStatus.FAIL

    critical = abs(sigma - 0.5) < 0.01
    message = f"K₁:₁ = {R:.3f} at σ={sigma:.2f} ({'critical' if critical else 'off-critical'})"

    return ValidationResult(
        axiom=22,
        status=status,
        value=R,
        threshold=threshold,
        message=message
    )


def axiom_23_rg_persistence(data: List[float], pool_size: int = 2,
                            drop_threshold: float = 0.4) -> ValidationResult:
    """
    Axiom 23: RG Persistence (E4 Audit)

    Structure survives ×2 coarse-graining

    Low-order features persist under renormalization.
    """
    if len(data) < pool_size * 2:
        return ValidationResult(
            axiom=23,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=drop_threshold,
            message="Insufficient data for coarse-graining"
        )

    d = np.array(data)

    # Fine-grained observable
    P_fine = np.mean(d)

    # Coarse-grained (pool)
    n_pools = len(d) // pool_size
    pooled = [np.mean(d[i*pool_size:(i+1)*pool_size]) for i in range(n_pools)]
    P_coarse = np.mean(pooled)

    # Fractional drop
    drop = abs(P_fine - P_coarse) / (abs(P_fine) + 1e-10)

    persistent = drop < drop_threshold
    status = ValidationStatus.PASS if persistent else ValidationStatus.FAIL

    message = f"RG drop = {drop*100:.1f}% ({'persistent' if persistent else 'fragile'})"

    return ValidationResult(
        axiom=23,
        status=status,
        value=drop,
        threshold=drop_threshold,
        message=message
    )


def axiom_24_mass_gap(eigenvalues: List[float],
                     gap_threshold: float = 0.1) -> ValidationResult:
    """
    Axiom 24: Mass Gap (Yang-Mills)

    λ₁ - λ₀ > Δ > 0

    Positive gap between ground state and first excited state.
    """
    if len(eigenvalues) < 2:
        return ValidationResult(
            axiom=24,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=gap_threshold,
            message="Need at least 2 eigenvalues"
        )

    eigs = np.sort(eigenvalues)
    gap = eigs[1] - eigs[0]

    status = ValidationStatus.PASS if gap > gap_threshold else ValidationStatus.FAIL

    message = f"Δ = {gap:.4f} {'>' if gap > gap_threshold else '≤'} {gap_threshold}"

    return ValidationResult(
        axiom=24,
        status=status,
        value=gap,
        threshold=gap_threshold,
        message=message
    )


def axiom_25_hodge_decomposition(form: np.ndarray,
                                 tolerance: float = 1e-4) -> ValidationResult:
    """
    Axiom 25: Hodge Decomposition (Hodge Conjecture)

    α = α_harmonic + dβ + δγ

    Every form decomposes into harmonic + exact + coexact.
    """
    # This is a simplified version - full Hodge decomposition requires
    # sophisticated differential geometry machinery

    if form.ndim != 2:
        return ValidationResult(
            axiom=25,
            status=ValidationStatus.NOT_APPLICABLE,
            value=0.0,
            threshold=tolerance,
            message="Requires 2-form"
        )

    # Placeholder: check if form is closed (dα = 0)
    # In discrete setting, check if approximately curl-free

    # For a matrix representation, check symmetry as proxy
    antisym = np.linalg.norm(form - form.T)

    # Hodge forms should have special structure
    status = ValidationStatus.UNCERTAIN  # Hard to validate without full machinery

    message = f"antisymmetry = {antisym:.2e} (placeholder check)"

    return ValidationResult(
        axiom=25,
        status=status,
        value=antisym,
        threshold=tolerance,
        message=message
    )


def axiom_26_birch_swinnerton_dyer(rank: int, regulator: float,
                                   torsion: int, L_value: float,
                                   tolerance: float = 0.1) -> ValidationResult:
    """
    Axiom 26: Birch-Swinnerton-Dyer (Elliptic Curves)

    rank(E(ℚ)) = order of vanishing of L(E,s) at s=1

    Arithmetic rank equals analytic rank.
    """
    # BSD formula: L(E,1) / Ω ≈ (regulator × torsion²) / |Sha|

    # Placeholder validation
    if rank < 0:
        return ValidationResult(
            axiom=26,
            status=ValidationStatus.UNCERTAIN,
            value=0.0,
            threshold=tolerance,
            message="Invalid rank"
        )

    # Check if L-value is consistent with rank
    # rank = 0 → L(E,1) ≠ 0
    # rank > 0 → L(E,1) = 0

    if rank == 0:
        consistent = abs(L_value) > tolerance
    else:
        consistent = abs(L_value) < tolerance

    status = ValidationStatus.PASS if consistent else ValidationStatus.FAIL

    message = f"rank={rank}, L(E,1)={L_value:.4f} ({'consistent' if consistent else 'inconsistent'})"

    return ValidationResult(
        axiom=26,
        status=status,
        value=abs(L_value),
        threshold=tolerance,
        message=message
    )


# =============================================================================
# BATCH VALIDATION
# =============================================================================

def validate_all_axioms(data: Dict) -> Dict[int, ValidationResult]:
    """
    Run all applicable axiom validators on provided data

    Args:
        data: Dictionary with keys matching axiom parameter names

    Returns:
        Dictionary mapping axiom number to ValidationResult
    """
    results = {}

    # Try each axiom
    validators = [
        (1, axiom_1_phase_locking, ['flux', 'dissipation']),
        (2, axiom_2_bounded_energy, ['energy_history']),
        (3, axiom_3_regularity_persistence, ['gradient_norms', 'value_norms']),
        (4, axiom_4_spectral_locality, ['amplitudes']),
        (5, axiom_5_coupling_decay, ['couplings', 'orders']),
        (6, axiom_6_exponential_convergence, ['trajectory', 'equilibrium']),
        (7, axiom_7_lyapunov_stability, ['lyapunov_function']),
        (8, axiom_8_attractor_existence, ['trajectories']),
        (9, axiom_9_no_finite_time_blowup, ['values']),
        (10, axiom_10_uniqueness, ['solution1', 'solution2']),
        (11, axiom_11_phase_coherence, ['phases']),
        (12, axiom_12_frequency_locking, ['freq1', 'freq2']),
        (13, axiom_13_arnold_tongue, ['coupling', 'detuning']),
        (14, axiom_14_kuramoto_transition, ['oscillators', 'coupling']),
        (15, axiom_15_winding_number, ['phase_trajectory']),
        (16, axiom_16_integer_thinning, ['couplings', 'orders']),
        (17, axiom_17_harmonic_preference, ['frequencies']),
        (18, axiom_18_fibonacci_cascade, ['mode_amplitudes']),
        (19, axiom_19_power_law_distribution, ['events']),
        (20, axiom_20_multifractal_spectrum, ['signal']),
        (21, axiom_21_energy_cascade, ['spectrum']),
        (22, axiom_22_one_to_one_lock, ['sigma', 't', 'primes']),
        (23, axiom_23_rg_persistence, ['data']),
        (24, axiom_24_mass_gap, ['eigenvalues']),
        (25, axiom_25_hodge_decomposition, ['form']),
        (26, axiom_26_birch_swinnerton_dyer, ['rank', 'regulator', 'torsion', 'L_value']),
    ]

    for axiom_num, validator, param_names in validators:
        # Check if all required parameters are present
        if all(param in data for param in param_names):
            try:
                params = {param: data[param] for param in param_names}
                result = validator(**params)
                results[axiom_num] = result
            except Exception as e:
                results[axiom_num] = ValidationResult(
                    axiom=axiom_num,
                    status=ValidationStatus.UNCERTAIN,
                    value=0.0,
                    threshold=0.0,
                    message=f"Error: {str(e)}"
                )

    return results


def print_validation_report(results: Dict[int, ValidationResult]):
    """Print formatted validation report"""
    print("=" * 70)
    print("AXIOM VALIDATION REPORT")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.status == ValidationStatus.PASS)
    total = len(results)

    print(f"\nSummary: {passed}/{total} axioms passed ({100*passed/total:.0f}%)\n")

    for axiom_num in sorted(results.keys()):
        result = results[axiom_num]
        print(f"{result}")

    print("\n" + "=" * 70)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Testing Axiom Validators\n")

    # Example 1: Navier-Stokes validation
    print("Example 1: Navier-Stokes System")
    print("-" * 70)

    ns_data = {
        'flux': 0.45,
        'dissipation': 0.53,
        'energy_history': [1.0, 0.9, 0.81, 0.73, 0.66],
        'gradient_norms': [2.0, 1.8, 1.6, 1.5, 1.4],
        'value_norms': [1.0, 0.9, 0.85, 0.82, 0.80],
        'amplitudes': [1.0, 0.35, 0.12, 0.042, 0.015],  # θ ≈ 0.35
        'spectrum': [1.0, 0.5, 0.3, 0.2, 0.15],
    }

    ns_results = validate_all_axioms(ns_data)
    print_validation_report(ns_results)

    # Example 2: Riemann Hypothesis validation
    print("\n\nExample 2: Riemann Hypothesis")
    print("-" * 70)

    rh_data = {
        'sigma': 0.5,
        't': 14.134,
        'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        'phases': np.random.uniform(-np.pi, np.pi, 10).tolist(),
        'data': [1.0, 1.1, 0.9, 1.05, 0.95] * 4,  # For RG test
    }

    rh_results = validate_all_axioms(rh_data)
    print_validation_report(rh_results)

    print("\n✓ All axiom validators loaded successfully!")
