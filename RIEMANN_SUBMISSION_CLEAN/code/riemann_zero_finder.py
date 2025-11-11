#!/usr/bin/env python3
"""
Riemann Zero Finder: Actually compute zeros, don't hardcode them

Finds zeros of ζ(s) on critical line by searching for |ζ(0.5+it)| = 0
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import mpmath
import json
from datetime import datetime

mpmath.mp.dps = 50  # High precision

def zeta_abs(t, sigma=0.5):
    """Compute |ζ(σ+it)|"""
    s = mpmath.mpc(sigma, t)
    z = mpmath.zeta(s)
    return float(abs(z))

def find_zero_bracket(t_start, t_end, sigma=0.5, n_points=100):
    """
    Find brackets [a,b] where zeta changes sign
    Returns list of (t_a, t_b) brackets
    """
    t_values = np.linspace(t_start, t_end, n_points)
    brackets = []

    for i in range(len(t_values)-1):
        t_a = t_values[i]
        t_b = t_values[i+1]

        # Check for sign change in real part
        s_a = mpmath.mpc(sigma, t_a)
        s_b = mpmath.mpc(sigma, t_b)

        z_a = mpmath.zeta(s_a)
        z_b = mpmath.zeta(s_b)

        # Look for sign change in real or imaginary part
        real_sign_change = (z_a.real * z_b.real < 0)
        imag_sign_change = (z_a.imag * z_b.imag < 0)

        if real_sign_change or imag_sign_change:
            brackets.append((float(t_a), float(t_b)))

    return brackets

def refine_zero(t_bracket, sigma=0.5, tol=1e-10):
    """
    Refine zero location using |ζ(σ+it)| minimization
    """
    t_a, t_b = t_bracket

    # Minimize |ζ(σ+it)|
    result = minimize_scalar(
        lambda t: zeta_abs(t, sigma),
        bounds=(t_a, t_b),
        method='bounded',
        options={'xatol': tol}
    )

    t_zero = result.x
    zeta_value = zeta_abs(t_zero, sigma)

    return t_zero, zeta_value

def find_zeros_in_range(t_min=0, t_max=100, sigma=0.5):
    """
    Find all zeros in range [t_min, t_max]

    Returns list of (t, |ζ(σ+it)|) tuples
    """
    print(f"Searching for zeros in t ∈ [{t_min}, {t_max}] at σ={sigma}")

    # Find brackets
    print("Finding brackets...")
    brackets = find_zero_bracket(t_min, t_max, sigma, n_points=1000)
    print(f"Found {len(brackets)} potential zeros")

    # Refine each
    zeros = []
    for i, bracket in enumerate(brackets):
        t_zero, abs_zeta = refine_zero(bracket, sigma)

        # Only keep if |ζ| is small enough
        if abs_zeta < 1e-6:
            zeros.append((t_zero, abs_zeta))
            print(f"  Zero {i+1}: t = {t_zero:.10f}, |ζ| = {abs_zeta:.2e}")

    return zeros

def compute_spacing_statistics(zeros):
    """
    Compute spacing statistics and compare to RMT predictions

    RMT predicts:
    - Mean spacing: grows like log(t)/(2π)
    - Normalized spacing distribution matches GUE
    """
    t_values = np.array([z[0] for z in zeros])

    if len(t_values) < 2:
        return {}

    # Raw spacings
    spacings = np.diff(t_values)

    # Normalized spacings (divide by mean)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing

    # Statistics
    stats = {
        'n_zeros': len(zeros),
        't_min': float(t_values[0]),
        't_max': float(t_values[-1]),
        'mean_spacing': float(mean_spacing),
        'std_spacing': float(np.std(spacings)),
        'min_spacing': float(np.min(spacings)),
        'max_spacing': float(np.max(spacings)),
        'normalized_spacings': normalized_spacings.tolist()[:20],  # First 20
    }

    # Compare to RMT prediction: mean ~ log(t)/(2π)
    t_mid = (t_values[0] + t_values[-1]) / 2
    rmt_prediction = np.log(t_mid) / (2 * np.pi)

    stats['rmt_predicted_spacing'] = float(rmt_prediction)
    stats['ratio_observed_to_rmt'] = float(mean_spacing / rmt_prediction)

    return stats

def verify_zero_on_critical_line(t_zero, delta_sigma=0.3):
    """
    Verify zero is on critical line by checking σ ≠ 0.5

    If RH true: |ζ(0.5+it)| = 0 but |ζ(0.8+it)| > 0
    """
    sigma_on = 0.5
    sigma_off = 0.5 + delta_sigma

    # On line
    z_on = zeta_abs(t_zero, sigma_on)

    # Off line
    z_off = zeta_abs(t_zero, sigma_off)

    # Zero should exist on-line but not off-line
    on_line_zero = z_on < 1e-6
    off_line_nonzero = z_off > 1e-3

    return {
        't': float(t_zero),
        'z_on_line': float(z_on),
        'z_off_line': float(z_off),
        'on_line_zero': bool(on_line_zero),
        'off_line_nonzero': bool(off_line_nonzero),
        'passes_test': bool(on_line_zero and off_line_nonzero)
    }

def main():
    print("="*80)
    print("RIEMANN ZERO FINDER: Actually Computing Zeros")
    print("="*80)
    print(f"Started: {datetime.now()}\n")

    # Find first 50 zeros
    t_max = 150  # Should contain ~50 zeros
    zeros = find_zeros_in_range(0, t_max, sigma=0.5)

    print(f"\n✓ Found {len(zeros)} zeros")

    # Spacing statistics
    print("\nComputing spacing statistics...")
    stats = compute_spacing_statistics(zeros)

    print(f"\nSpacing Statistics:")
    print(f"  Mean spacing: {stats['mean_spacing']:.4f}")
    print(f"  RMT prediction: {stats['rmt_predicted_spacing']:.4f}")
    print(f"  Ratio: {stats['ratio_observed_to_rmt']:.4f}")

    # Verify first 10 are on critical line
    print("\nVerifying zeros are on critical line...")
    verifications = []
    for i, (t, _) in enumerate(zeros[:10]):
        verification = verify_zero_on_critical_line(t, delta_sigma=0.3)
        verifications.append(verification)
        status = "✓" if verification['passes_test'] else "✗"
        print(f"  {status} t={verification['t']:.6f}: "
              f"|ζ(0.5+it)|={verification['z_on_line']:.2e}, "
              f"|ζ(0.8+it)|={verification['z_off_line']:.2e}")

    # Verdict
    n_passed = sum(1 for v in verifications if v['passes_test'])
    print(f"\n{'='*80}")
    print(f"VERDICT: {n_passed}/{len(verifications)} zeros verified on critical line")
    print(f"{'='*80}")

    # Save results
    result = {
        'timestamp': str(datetime.now()),
        'n_zeros_found': len(zeros),
        't_range': [0, t_max],
        'zeros': [{'t': float(t), 'abs_zeta': float(z)} for t, z in zeros],
        'spacing_statistics': stats,
        'verifications': verifications,
        'verdict': 'SUPPORTED' if n_passed == len(verifications) else 'INCONCLUSIVE'
    }

    with open('../results/riemann_zeros_computed.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to: ../results/riemann_zeros_computed.json")
    print(f"Completed: {datetime.now()}")

if __name__ == "__main__":
    main()
