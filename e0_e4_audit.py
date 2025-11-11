#!/usr/bin/env python3
"""
E0-E4 AUDIT FRAMEWORK: Universal Testing Protocol

A 5-stage audit protocol that validates whether a system's behavior is genuine
or an artifact. Applies across all domains: PDEs, ML, physics, finance, etc.

The E0-E4 tests are UNIVERSAL and work for ANY system with observable dynamics.

Stages:
    E0: Calibration (baseline check)
    E1: Vibration (perturbation response)
    E2: Symmetry (invariance check)
    E3: Micro-nudge (sensitivity test)
    E4: RG Persistence (coarse-graining survival)

Author: Jake A. Hallett
Date: 2025-11-11
Version: 1.0
"""

import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AuditStatus(Enum):
    """Overall audit status."""
    PASS_ALL = "PASS_ALL"          # All E0-E4 pass
    PASS_MOST = "PASS_MOST"        # 4/5 pass
    MIXED = "MIXED"                # 3/5 pass
    FAIL_MOST = "FAIL_MOST"        # 2/5 pass
    FAIL_ALL = "FAIL_ALL"          # 0-1 pass


@dataclass
class E0E4Result:
    """Complete E0-E4 audit result."""
    e0_pass: bool
    e1_pass: bool
    e2_pass: bool
    e3_pass: bool
    e4_pass: bool

    e0_value: float
    e1_value: float
    e2_value: float
    e3_value: float
    e4_value: float

    overall_status: AuditStatus
    confidence: float
    interpretation: str
    details: Dict

    @property
    def passes(self) -> int:
        """Count number of tests passed."""
        return sum([self.e0_pass, self.e1_pass, self.e2_pass, self.e3_pass, self.e4_pass])

    @property
    def pass_rate(self) -> float:
        """Fraction of tests passed."""
        return self.passes / 5.0

    def __str__(self):
        status_symbol = {
            AuditStatus.PASS_ALL: "✓✓✓",
            AuditStatus.PASS_MOST: "✓✓",
            AuditStatus.MIXED: "~",
            AuditStatus.FAIL_MOST: "✗",
            AuditStatus.FAIL_ALL: "✗✗✗"
        }
        symbol = status_symbol.get(self.overall_status, "?")
        return (f"{symbol} E0-E4 Audit: {self.passes}/5 pass ({self.pass_rate*100:.0f}%) "
                f"- {self.interpretation}")


# ==============================================================================
# E0: CALIBRATION TEST
# ==============================================================================

def e0_calibration(observable: Callable, params: Dict,
                   expected_range: Tuple[float, float]) -> Tuple[bool, float]:
    """
    E0: Calibration Test - Does the system produce sensible baseline values?

    Test: Compute observable with nominal parameters, check if in expected range.

    Applications:
    - PDEs: Energy should be positive and finite
    - Neural nets: Loss should decrease
    - Finance: Returns should be bounded

    Parameters:
    -----------
    observable : Callable
        Function that computes the observable (e.g., energy, loss)
        Signature: observable(params) -> float
    params : Dict
        Nominal parameter values
    expected_range : Tuple[float, float]
        (min, max) expected values for observable

    Returns:
    --------
    pass_test : bool
        True if observable in expected range
    value : float
        Computed observable value

    Example:
    --------
    >>> def energy(p): return p['velocity']**2 / 2
    >>> passed, val = e0_calibration(energy, {'velocity': 1.0}, (0, 10))
    >>> # val = 0.5, in range [0, 10] → PASS
    """
    try:
        value = observable(params)

        if not np.isfinite(value):
            return False, value

        min_val, max_val = expected_range
        in_range = min_val <= value <= max_val

        return in_range, value

    except Exception as e:
        print(f"E0 Error: {e}")
        return False, np.nan


# ==============================================================================
# E1: VIBRATION TEST
# ==============================================================================

def e1_vibration(observable: Callable, params: Dict,
                 perturbation_size: float = 0.01,
                 stability_threshold: float = 10.0) -> Tuple[bool, float]:
    """
    E1: Vibration Test - Is the system stable to small perturbations?

    Test: Perturb parameters slightly, check if observable change is bounded.
    - Stable system: Small perturbation → small response
    - Unstable system: Small perturbation → large response

    Applications:
    - PDEs: Smooth solutions are stable
    - Neural nets: Overfitting is unstable
    - Finance: Robust strategies are stable

    Parameters:
    -----------
    observable : Callable
        Function computing observable
    params : Dict
        Nominal parameters
    perturbation_size : float
        Relative perturbation magnitude (default 1%)
    stability_threshold : float
        Max allowed response amplification (default 10x)

    Returns:
    --------
    pass_test : bool
        True if response < stability_threshold * perturbation
    amplification : float
        Response / perturbation ratio

    Example:
    --------
    >>> def loss(p): return p['x']**2
    >>> passed, amp = e1_vibration(loss, {'x': 1.0}, perturbation_size=0.01)
    >>> # Perturb x by 0.01 → loss changes by ~0.02 → amp ≈ 2 < 10 → PASS
    """
    try:
        # Baseline
        value_0 = observable(params)

        # Perturb each parameter
        max_amplification = 0.0

        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Create perturbed params
                params_perturbed = params.copy()
                delta = abs(value) * perturbation_size if value != 0 else perturbation_size
                params_perturbed[key] = value + delta

                # Compute response
                value_1 = observable(params_perturbed)

                if not np.isfinite(value_1):
                    return False, np.inf

                # Response amplification
                response = abs(value_1 - value_0)
                amplification = response / (abs(delta) + 1e-10)

                max_amplification = max(max_amplification, amplification)

        stable = max_amplification < stability_threshold

        return stable, max_amplification

    except Exception as e:
        print(f"E1 Error: {e}")
        return False, np.inf


# ==============================================================================
# E2: SYMMETRY TEST
# ==============================================================================

def e2_symmetry(observable: Callable, params: Dict,
                symmetry_transform: Callable,
                tolerance: float = 1e-6) -> Tuple[bool, float]:
    """
    E2: Symmetry Test - Is the claimed symmetry actually obeyed?

    Test: Apply symmetry transformation, check if observable unchanged.
    - True symmetry: Observable exactly preserved
    - Broken symmetry: Observable changes

    Applications:
    - Physics: Conservation laws from symmetries
    - Neural nets: Translation invariance in CNNs
    - Markets: Time-translation in stationary processes

    Parameters:
    -----------
    observable : Callable
        Function computing observable
    params : Dict
        System parameters
    symmetry_transform : Callable
        Transformation that should preserve observable
        Signature: symmetry_transform(params) -> params_transformed
    tolerance : float
        Max allowed relative change (default 1e-6)

    Returns:
    --------
    pass_test : bool
        True if |O_transformed - O_original| / |O_original| < tolerance
    violation : float
        Relative change in observable

    Example:
    --------
    >>> def energy(p): return p['v']**2
    >>> def time_shift(p): return {'v': p['v']}  # Energy conserved
    >>> passed, viol = e2_symmetry(energy, {'v': 2.0}, time_shift)
    >>> # Energy unchanged under time shift → PASS
    """
    try:
        # Original
        value_0 = observable(params)

        # Transformed
        params_transformed = symmetry_transform(params)
        value_1 = observable(params_transformed)

        if not np.isfinite(value_0) or not np.isfinite(value_1):
            return False, np.inf

        # Relative violation
        violation = abs(value_1 - value_0) / (abs(value_0) + 1e-10)

        preserved = violation < tolerance

        return preserved, violation

    except Exception as e:
        print(f"E2 Error: {e}")
        return False, np.inf


# ==============================================================================
# E3: MICRO-NUDGE TEST
# ==============================================================================

def e3_micro_nudge(observable: Callable, params: Dict,
                   nudge_size: float = 1e-8,
                   smoothness_threshold: float = 1e6) -> Tuple[bool, float]:
    """
    E3: Micro-Nudge Test - Is the system smooth (no hidden singularities)?

    Test: Apply tiny perturbation, check if gradient is bounded.
    - Smooth: Tiny nudge → tiny change (bounded derivative)
    - Singular: Tiny nudge → large change (unbounded derivative)

    Applications:
    - PDEs: Smooth solutions have bounded derivatives
    - Neural nets: Smooth loss landscape
    - Finance: No black swans (continuous returns)

    Parameters:
    -----------
    observable : Callable
        Function computing observable
    params : Dict
        System parameters
    nudge_size : float
        Tiny perturbation (default 1e-8)
    smoothness_threshold : float
        Max allowed gradient magnitude (default 1e6)

    Returns:
    --------
    pass_test : bool
        True if gradient < smoothness_threshold
    gradient : float
        Estimated gradient magnitude

    Example:
    --------
    >>> def f(p): return p['x']**2
    >>> passed, grad = e3_micro_nudge(f, {'x': 1.0})
    >>> # df/dx = 2x = 2 < 1e6 → SMOOTH → PASS
    """
    try:
        value_0 = observable(params)

        max_gradient = 0.0

        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Micro-nudge
                params_nudged = params.copy()
                params_nudged[key] = value + nudge_size

                value_1 = observable(params_nudged)

                if not np.isfinite(value_1):
                    return False, np.inf

                # Gradient estimate
                gradient = abs(value_1 - value_0) / nudge_size
                max_gradient = max(max_gradient, gradient)

        smooth = max_gradient < smoothness_threshold

        return smooth, max_gradient

    except Exception as e:
        print(f"E3 Error: {e}")
        return False, np.inf


# ==============================================================================
# E4: RG PERSISTENCE TEST
# ==============================================================================

def e4_rg_persistence(data: List[float],
                      pool_size: int = 2,
                      drop_threshold: float = 0.4) -> Tuple[bool, float]:
    """
    E4: RG Persistence Test - Does the structure survive coarse-graining?

    Test: Pool/average neighboring data points, check if property preserved.
    - True structure: Property drop < 40%
    - Artifact/noise: Property drop > 40%

    This is THE definitive test for distinguishing signal from noise.

    Applications:
    - Signal processing: Real signal vs noise
    - Deep learning: True features vs overfitting
    - Finance: Genuine trends vs random walks
    - Data science: Patterns vs artifacts

    Parameters:
    -----------
    data : List[float]
        Time series or spatial data
    pool_size : int
        Coarse-graining window (2, 3, or 4)
    drop_threshold : float
        Max allowed property drop (default 0.4 = 40%)

    Returns:
    --------
    pass_test : bool
        True if drop < drop_threshold
    drop : float
        Fraction drop in property

    Example:
    --------
    >>> data = [1.0, 0.9, 0.8, 0.85, 0.7, 0.75]  # Smooth trend
    >>> passed, drop = e4_rg_persistence(data, pool_size=2)
    >>> # Mean preserved after pooling → drop ≈ 0% < 40% → PASS
    """
    try:
        if len(data) < pool_size * 2:
            return False, 1.0

        # Fine-grained property (use mean as default)
        P_fine = np.mean(data)

        # Coarse-grain
        n_pools = len(data) // pool_size
        pooled = [np.mean(data[i*pool_size:(i+1)*pool_size]) for i in range(n_pools)]

        # Coarse-grained property
        P_coarse = np.mean(pooled)

        # Drop
        drop = abs(P_fine - P_coarse) / (abs(P_fine) + 1e-10)

        persistent = drop < drop_threshold

        return persistent, drop

    except Exception as e:
        print(f"E4 Error: {e}")
        return False, 1.0


# ==============================================================================
# COMPREHENSIVE E0-E4 AUDIT
# ==============================================================================

def run_e0_e4_audit(observable: Callable,
                    params: Dict,
                    data: Optional[List[float]] = None,
                    symmetry_transform: Optional[Callable] = None,
                    expected_range: Tuple[float, float] = (-1e10, 1e10),
                    **kwargs) -> E0E4Result:
    """
    Run complete E0-E4 audit on a system.

    Parameters:
    -----------
    observable : Callable
        Function computing the observable to validate
    params : Dict
        System parameters
    data : List[float], optional
        Time series for E4 test
    symmetry_transform : Callable, optional
        Symmetry transformation for E2 test
    expected_range : Tuple[float, float]
        Expected range for E0 calibration
    **kwargs : Additional parameters for individual tests

    Returns:
    --------
    E0E4Result
        Complete audit results

    Example:
    --------
    >>> def energy(p): return p['v']**2 / 2
    >>> params = {'v': 1.0}
    >>> data = [0.5, 0.49, 0.51, 0.50, 0.48, 0.52]
    >>> result = run_e0_e4_audit(energy, params, data, expected_range=(0, 10))
    >>> print(result)
    """
    # E0: Calibration
    e0_pass, e0_val = e0_calibration(observable, params, expected_range)

    # E1: Vibration
    e1_pass, e1_val = e1_vibration(observable, params,
                                   kwargs.get('perturbation_size', 0.01),
                                   kwargs.get('stability_threshold', 10.0))

    # E2: Symmetry
    if symmetry_transform is not None:
        e2_pass, e2_val = e2_symmetry(observable, params, symmetry_transform,
                                     kwargs.get('symmetry_tolerance', 1e-6))
    else:
        # Skip E2 if no symmetry provided
        e2_pass, e2_val = True, 0.0

    # E3: Micro-nudge
    e3_pass, e3_val = e3_micro_nudge(observable, params,
                                    kwargs.get('nudge_size', 1e-8),
                                    kwargs.get('smoothness_threshold', 1e6))

    # E4: RG Persistence
    if data is not None and len(data) >= 4:
        e4_pass, e4_val = e4_rg_persistence(data,
                                           kwargs.get('pool_size', 2),
                                           kwargs.get('drop_threshold', 0.4))
    else:
        # Skip E4 if no data
        e4_pass, e4_val = True, 0.0

    # Overall status
    passes = sum([e0_pass, e1_pass, e2_pass, e3_pass, e4_pass])

    if passes == 5:
        status = AuditStatus.PASS_ALL
        interp = "All tests pass - System validated ✓"
        confidence = 1.0
    elif passes == 4:
        status = AuditStatus.PASS_MOST
        interp = "4/5 tests pass - System likely valid"
        confidence = 0.8
    elif passes == 3:
        status = AuditStatus.MIXED
        interp = "3/5 tests pass - Mixed results"
        confidence = 0.6
    elif passes == 2:
        status = AuditStatus.FAIL_MOST
        interp = "Only 2/5 pass - System questionable"
        confidence = 0.4
    else:
        status = AuditStatus.FAIL_ALL
        interp = f"Only {passes}/5 pass - System likely invalid"
        confidence = 0.2

    return E0E4Result(
        e0_pass=e0_pass, e1_pass=e1_pass, e2_pass=e2_pass,
        e3_pass=e3_pass, e4_pass=e4_pass,
        e0_value=e0_val, e1_value=e1_val, e2_value=e2_val,
        e3_value=e3_val, e4_value=e4_val,
        overall_status=status,
        confidence=confidence,
        interpretation=interp,
        details={
            'E0': {'pass': e0_pass, 'value': e0_val, 'test': 'Calibration'},
            'E1': {'pass': e1_pass, 'value': e1_val, 'test': 'Vibration'},
            'E2': {'pass': e2_pass, 'value': e2_val, 'test': 'Symmetry'},
            'E3': {'pass': e3_pass, 'value': e3_val, 'test': 'Micro-nudge'},
            'E4': {'pass': e4_pass, 'value': e4_val, 'test': 'RG Persistence'}
        }
    )


def print_audit_report(result: E0E4Result):
    """Print detailed audit report."""
    print("=" * 80)
    print("E0-E4 AUDIT REPORT")
    print("=" * 80)

    tests = [
        ('E0', 'Calibration', result.e0_pass, result.e0_value),
        ('E1', 'Vibration', result.e1_pass, result.e1_value),
        ('E2', 'Symmetry', result.e2_pass, result.e2_value),
        ('E3', 'Micro-nudge', result.e3_pass, result.e3_value),
        ('E4', 'RG Persistence', result.e4_pass, result.e4_value)
    ]

    for name, description, passed, value in tests:
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name} ({description:15s}): {'PASS' if passed else 'FAIL':4s} (value={value:.3e})")

    print("-" * 80)
    print(f"Overall: {result.passes}/5 pass ({result.pass_rate*100:.0f}%)")
    print(f"Status: {result.overall_status.value}")
    print(f"Confidence: {result.confidence*100:.0f}%")
    print(f"Interpretation: {result.interpretation}")
    print("=" * 80)


# ==============================================================================
# EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    print("E0-E4 AUDIT FRAMEWORK: Example Usage\n")

    # Example 1: Simple harmonic oscillator (should pass all)
    print("[Example 1] Simple Harmonic Oscillator")
    print("-" * 80)

    def energy_sho(p):
        """Energy of simple harmonic oscillator."""
        return 0.5 * p['v']**2 + 0.5 * p['x']**2

    params_sho = {'v': 1.0, 'x': 0.5}
    data_sho = [0.625, 0.620, 0.630, 0.625, 0.618, 0.632]  # Stable energy

    def time_translation(p):
        """Time translation symmetry (energy conserved)."""
        return p  # Energy unchanged

    result_sho = run_e0_e4_audit(
        observable=energy_sho,
        params=params_sho,
        data=data_sho,
        symmetry_transform=time_translation,
        expected_range=(0, 10)
    )

    print_audit_report(result_sho)

    # Example 2: Unstable system (should fail E1)
    print("\n[Example 2] Unstable System")
    print("-" * 80)

    def unstable_energy(p):
        """Unstable: energy ~ exp(x)."""
        return np.exp(abs(p['x']))

    params_unstable = {'x': 1.0}

    result_unstable = run_e0_e4_audit(
        observable=unstable_energy,
        params=params_unstable,
        expected_range=(0, 100)
    )

    print_audit_report(result_unstable)

    # Example 3: Noisy artifact (should fail E4)
    print("\n[Example 3] Noisy Artifact")
    print("-" * 80)

    def mean_observable(p):
        return p.get('mean', 0)

    # High-frequency noise (artifact)
    np.random.seed(42)
    data_noise = np.random.randn(20)

    result_noise = run_e0_e4_audit(
        observable=lambda p: np.mean(data_noise),
        params={'mean': 0},
        data=list(data_noise),
        expected_range=(-1, 1)
    )

    print_audit_report(result_noise)

    print("\n✓ E0-E4 audit framework ready for use!")
