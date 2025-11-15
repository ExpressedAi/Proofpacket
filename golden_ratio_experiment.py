#!/usr/bin/env python3
"""
GOLDEN RATIO EXPERIMENT
Compare 0.4 vs 0.382 (exact golden ratio constant) in pathway dynamics

Testing hypothesis: Using exact φ-derived constants improves performance
"""

import numpy as np
import time
from copy import deepcopy
from generative_learning_engine import GenerativeLearningEngine, ReasoningRequest
from pathway_memory import PathwayMemory, TransitionStats
from quantum_vbc import QuantumVariableBarrierController
import json

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
GOLDEN_INVERSE = 1 / PHI    # 0.618033988749895
GOLDEN_WEIGHT = 1 / (1 + PHI)  # 0.381966011250105  ← THIS IS THE KEY!


def run_exploration_experiment(use_golden: bool, num_sessions: int = 100):
    """
    Run exploration with either 0.4 (rounded) or 0.382 (exact golden ratio).

    Args:
        use_golden: If True, use 0.382; if False, use 0.4
        num_sessions: Number of reasoning sessions to run

    Returns:
        dict with results
    """

    constant = GOLDEN_WEIGHT if use_golden else 0.4
    name = "GOLDEN (0.382)" if use_golden else "ROUNDED (0.4)"

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*80}")
    print(f"Weight constant: {constant:.9f}")
    print()

    # Monkey-patch the constants
    import pathway_memory
    import quantum_vbc

    # Store originals
    orig_vbc_cap = quantum_vbc.QuantumVariableBarrierController.__init__.__defaults__[1]

    # Apply new constant
    if use_golden:
        # Patch QVBC
        quantum_vbc.QuantumVariableBarrierController.__init__.__defaults__ = (
            3,  # max_concurrent_changes
            GOLDEN_WEIGHT,  # per_axis_cap = 0.382
            0.7  # budget_total
        )

        # Patch pathway_memory strength calculation
        original_strength = TransitionStats.strength.fget

        def golden_strength(self):
            usage_factor = min(1.0, self.usage_count / 50.0)
            success_factor = self.success_rate
            speed_factor = 1.0 / (1.0 + self.average_convergence_time)
            validation_factor = self.rg_persistence_score

            # Use golden ratio weights instead of 0.4, 0.3, 0.2, 0.1
            # Normalized to sum to 1.0:
            w1 = GOLDEN_WEIGHT  # 0.382 (was 0.4)
            w2 = 0.3
            w3 = 0.2
            w4 = 0.1
            total = w1 + w2 + w3 + w4  # normalize

            return (w1/total) * usage_factor + \
                   (w2/total) * success_factor + \
                   (w3/total) * speed_factor + \
                   (w4/total) * validation_factor

        TransitionStats.strength = property(golden_strength)

    # Initialize engine
    engine = GenerativeLearningEngine()
    all_concepts = [c.name for c in engine.semantic_sphere.concepts]

    # Track stats
    stats = {
        'snaps': 0,
        'total_time': 0.0,
        'convergence_times': [],
        'pathway_strengths': [],
        'snap_sessions': []
    }

    start_time = time.time()

    for session in range(1, num_sessions + 1):
        # Pick random concepts
        from_c = np.random.choice(all_concepts)
        to_c = np.random.choice([c for c in all_concepts if c != from_c])

        # Run reasoning
        request = ReasoningRequest(
            problem=f"Transition from {from_c} to {to_c}",
            context={'experiment': name, 'session': session},
            initial_concepts=[from_c],
            target_concepts=[to_c],
            max_time=3.0,
            session_id=session
        )

        session_start = time.time()
        result = engine.reason(request)
        session_time = time.time() - session_start

        # Track results
        stats['total_time'] += session_time
        stats['convergence_times'].append(session_time)

        if len(result.snaps) > 0:
            stats['snaps'] += 1
            stats['snap_sessions'].append(session)

        # Track pathway strengths
        memory = engine.pathway_memory
        if memory.transitions:
            avg_strength = np.mean([s.strength for s in memory.transitions.values()])
            stats['pathway_strengths'].append(avg_strength)

        if session % 20 == 0:
            num_pathways = len(memory.transitions)
            snap_rate = 100 * stats['snaps'] / session
            avg_time = stats['total_time'] / session
            avg_strength = stats['pathway_strengths'][-1] if stats['pathway_strengths'] else 0

            print(f"  Session {session:3d} | "
                  f"Pathways: {num_pathways:3d} | "
                  f"Snaps: {snap_rate:4.1f}% | "
                  f"Avg time: {avg_time:.3f}s | "
                  f"Avg strength: {avg_strength:.4f}")

    elapsed = time.time() - start_time

    # Restore originals
    if use_golden:
        quantum_vbc.QuantumVariableBarrierController.__init__.__defaults__ = (
            3, orig_vbc_cap, 0.7
        )
        TransitionStats.strength = property(original_strength)

    # Final statistics
    final_stats = {
        'name': name,
        'constant': constant,
        'sessions': num_sessions,
        'total_time': elapsed,
        'snaps': stats['snaps'],
        'snap_rate': stats['snaps'] / num_sessions,
        'avg_convergence_time': np.mean(stats['convergence_times']),
        'std_convergence_time': np.std(stats['convergence_times']),
        'pathways_built': len(engine.pathway_memory.transitions),
        'avg_pathway_strength': np.mean(stats['pathway_strengths']) if stats['pathway_strengths'] else 0,
        'final_pathway_strength': stats['pathway_strengths'][-1] if stats['pathway_strengths'] else 0,
        'strength_growth': stats['pathway_strengths'][-1] - stats['pathway_strengths'][0] if len(stats['pathway_strengths']) > 1 else 0
    }

    return final_stats


def main():
    """Run side-by-side comparison"""

    print("\n" + "="*80)
    print("GOLDEN RATIO vs ROUNDED CONSTANT EXPERIMENT")
    print("="*80)
    print()
    print(f"φ (phi) = {PHI:.15f}")
    print(f"1/φ = {GOLDEN_INVERSE:.15f}")
    print(f"1/(1+φ) = {GOLDEN_WEIGHT:.15f}  ← Using this for golden experiment")
    print()
    print("Hypothesis: Exact golden ratio constants improve pathway dynamics")
    print()

    num_sessions = 100

    # Experiment 1: Traditional 0.4
    print("\n" + "="*80)
    print("CONTROL: Traditional 0.4 constant")
    print("="*80)
    control_results = run_exploration_experiment(use_golden=False, num_sessions=num_sessions)

    # Small delay
    time.sleep(2)

    # Experiment 2: Golden ratio 0.382
    print("\n" + "="*80)
    print("TREATMENT: Golden ratio 0.382 constant")
    print("="*80)
    golden_results = run_exploration_experiment(use_golden=True, num_sessions=num_sessions)

    # Comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()

    print(f"{'Metric':<30} {'0.4 (Control)':<20} {'0.382 (Golden)':<20} {'Δ %':<10}")
    print("-" * 80)

    def compare(name, control_val, golden_val, format_str="{:.3f}", higher_is_better=True):
        delta_pct = 100 * (golden_val - control_val) / control_val if control_val != 0 else 0
        symbol = "✓" if (delta_pct > 0 and higher_is_better) or (delta_pct < 0 and not higher_is_better) else "✗"

        print(f"{name:<30} {format_str.format(control_val):<20} "
              f"{format_str.format(golden_val):<20} "
              f"{symbol} {delta_pct:+6.2f}%")

    compare("Snap rate", control_results['snap_rate'], golden_results['snap_rate'],
            format_str="{:.1%}", higher_is_better=True)

    compare("Pathways built", control_results['pathways_built'], golden_results['pathways_built'],
            format_str="{:.0f}", higher_is_better=True)

    compare("Avg convergence time", control_results['avg_convergence_time'],
            golden_results['avg_convergence_time'], format_str="{:.3f}s", higher_is_better=False)

    compare("Final pathway strength", control_results['final_pathway_strength'],
            golden_results['final_pathway_strength'], format_str="{:.4f}", higher_is_better=True)

    compare("Strength growth", control_results['strength_growth'],
            golden_results['strength_growth'], format_str="{:.4f}", higher_is_better=True)

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)

    # Determine winner
    golden_score = 0
    control_score = 0

    if golden_results['snap_rate'] > control_results['snap_rate']:
        golden_score += 1
    else:
        control_score += 1

    if golden_results['pathways_built'] > control_results['pathways_built']:
        golden_score += 1
    else:
        control_score += 1

    if golden_results['avg_convergence_time'] < control_results['avg_convergence_time']:
        golden_score += 1
    else:
        control_score += 1

    if golden_results['final_pathway_strength'] > control_results['final_pathway_strength']:
        golden_score += 1
    else:
        control_score += 1

    print()
    print(f"Golden ratio (0.382): {golden_score}/4 metrics improved")
    print(f"Traditional (0.4):    {control_score}/4 metrics improved")
    print()

    if golden_score > control_score:
        print("✓ GOLDEN RATIO WINS!")
        print("  The exact φ-derived constant (0.382) outperforms the rounded value.")
        print("  Recommendation: Update all instances of 0.4 → 0.382")
    elif control_score > golden_score:
        print("✗ Traditional constant performs better")
        print("  The rounded value (0.4) is adequate for this system.")
    else:
        print("≈ TIE - No significant difference")
        print("  Both constants perform similarly.")

    print()

    # Save results
    results = {
        'phi': PHI,
        'golden_weight': GOLDEN_WEIGHT,
        'control': control_results,
        'golden': golden_results,
        'winner': 'golden' if golden_score > control_score else 'control' if control_score > golden_score else 'tie'
    }

    with open('golden_ratio_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved to: golden_ratio_experiment_results.json")
    print()


if __name__ == '__main__':
    main()
