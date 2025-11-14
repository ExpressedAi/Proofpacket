#!/usr/bin/env python3
"""
Test snap mechanics directly

Debug why snaps aren't happening
"""

import numpy as np
from omega_flow_controller import OmegaFlowController, Resonance
from pathway_memory import PathwayMemory

# Create controller
controller = OmegaFlowController()
memory = PathwayMemory()

print("="*80)
print("SNAP MECHANICS DEBUG")
print("="*80)
print()

# Create a strong resonance at a specific phase
target_phase = np.pi / 4  # 45 degrees
resonance = Resonance(
    name="test_target",
    p=1,
    q=1,
    K=5.0,  # very strong coupling
    Gamma=0.2,
    theta_a=target_phase,
    theta_n=0.0,
    H_star=0.9,
    zeta=0.1,
    coherence=0.9
)

print(f"Created resonance '{resonance.name}':")
print(f"  K = {resonance.K}")
print(f"  epsilon_cap = {resonance.epsilon_cap:.3f}")
print(f"  target phase (theta_a) = {np.rad2deg(target_phase):.1f}°")
print()

# Test 1: State very close to resonance
print("Test 1: State VERY close to resonance phase")
print("-"*80)

# Put state at almost exactly the target phase
n_close = np.array([np.cos(target_phase + 0.05),  # 3° off
                     np.sin(target_phase + 0.05),
                     0.2])
n_close = n_close / np.linalg.norm(n_close)

# Update resonance with current state phase
resonance.theta_n = np.arctan2(n_close[1], n_close[0])

print(f"State phase: {np.rad2deg(resonance.theta_n):.1f}°")
print(f"Detune: {np.rad2deg(resonance.detune):.1f}°")
print(f"Is eligible? {resonance.is_eligible}")
print(f"Epsilon cap: {resonance.epsilon_cap:.3f}")
print()

# Check snap eligibility
snap_result = controller.check_snap_eligibility(n_close, [resonance], memory)

if snap_result:
    target_name, quality = snap_result
    print(f"✓ SNAP ELIGIBLE!")
    print(f"  Target: {target_name}")
    print(f"  Quality: {quality:.3f}")
else:
    print("✗ No snap detected")
    print()
    print("Debugging:")
    print(f"  Detune check: {abs(resonance.detune)} <= {controller.delta_snap} ? {abs(resonance.detune) <= controller.delta_snap}")
    print(f"  Eligibility: {resonance.is_eligible}")
    print(f"  Epsilon > 0: {resonance.epsilon_cap > 0}")

    # Check gain
    V_current = resonance.local_potential()
    direction = controller._direction_to_basin(n_close, resonance)
    n_test = n_close + 0.1 * direction
    n_test = n_test / np.linalg.norm(n_test)

    r_test = Resonance(
        name=resonance.name, p=resonance.p, q=resonance.q,
        K=resonance.K, Gamma=resonance.Gamma,
        theta_a=resonance.theta_a,
        theta_n=np.arctan2(n_test[1], n_test[0]),
        H_star=resonance.H_star, zeta=resonance.zeta, coherence=resonance.coherence
    )
    V_test = r_test.local_potential()
    delta_V = V_test - V_current

    print(f"  Delta V: {delta_V:.6f} (need < {-controller.tau_V})")
    print(f"  V_current: {V_current:.6f}")
    print(f"  V_test: {V_test:.6f}")

print()
print()

# Test 2: State farther away
print("Test 2: State farther away (20° off)")
print("-"*80)

n_far = np.array([np.cos(target_phase + 0.35),  # 20° off
                   np.sin(target_phase + 0.35),
                   0.2])
n_far = n_far / np.linalg.norm(n_far)

resonance.theta_n = np.arctan2(n_far[1], n_far[0])

print(f"State phase: {np.rad2deg(resonance.theta_n):.1f}°")
print(f"Detune: {np.rad2deg(resonance.detune):.1f}°")
print(f"Is eligible? {resonance.is_eligible}")
print()

snap_result = controller.check_snap_eligibility(n_far, [resonance], memory)

if snap_result:
    target_name, quality = snap_result
    print(f"✓ SNAP ELIGIBLE!")
    print(f"  Target: {target_name}")
    print(f"  Quality: {quality:.3f}")
else:
    print("✗ No snap detected (expected - too far)")

print()
print()

# Test 3: Multiple resonances, varying distances
print("Test 3: Multiple resonances at different phases")
print("-"*80)

resonances = [
    Resonance("close_target", p=1, q=1, K=4.0, Gamma=0.2,
              theta_a=0.5, theta_n=0.0, H_star=0.9, zeta=0.1, coherence=0.9),
    Resonance("medium_target", p=1, q=1, K=3.5, Gamma=0.2,
              theta_a=1.2, theta_n=0.0, H_star=0.85, zeta=0.15, coherence=0.85),
    Resonance("far_target", p=1, q=1, K=3.0, Gamma=0.2,
              theta_a=2.5, theta_n=0.0, H_star=0.8, zeta=0.2, coherence=0.8),
]

# Put state near the first resonance
test_phase = 0.55  # close to 0.5
n_test = np.array([np.cos(test_phase), np.sin(test_phase), 0.2])
n_test = n_test / np.linalg.norm(n_test)

# Update all resonances
for r in resonances:
    r.theta_n = test_phase

print(f"State phase: {np.rad2deg(test_phase):.1f}°")
print()

for r in resonances:
    detune_deg = np.rad2deg(r.detune)
    print(f"{r.name}:")
    print(f"  Target phase: {np.rad2deg(r.theta_a):.1f}°")
    print(f"  Detune: {detune_deg:.1f}°")
    print(f"  Eligible: {r.is_eligible}")
    print(f"  Epsilon: {r.epsilon_cap:.3f}")
    print()

snap_result = controller.check_snap_eligibility(n_test, resonances, memory)

if snap_result:
    target_name, quality = snap_result
    print(f"✓ SNAPS TO: {target_name} (quality: {quality:.3f})")
else:
    print("✗ No snap detected")

print()
