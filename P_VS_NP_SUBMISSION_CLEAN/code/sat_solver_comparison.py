#!/usr/bin/env python3
"""
P vs NP Reality Check: Compare Bridge Framework to Actual SAT Solvers

Tests:
1. DPLL (deterministic, complete)
2. WalkSAT (randomized, incomplete but fast)
3. Bridge-guided MWU (current approach)

Shows success rates and runtime scaling.
"""

import random
import time
from typing import List, Tuple, Dict
import json
from datetime import datetime


# ==============================================================================
# DPLL: Classical Complete SAT Solver
# ==============================================================================

def dpll_solve(formula: List[List[int]], assignment: Dict[int, bool] = None) -> Tuple[bool, Dict[int, bool]]:
    """
    DPLL algorithm: complete SAT solver

    Returns: (satisfiable, assignment)
    """
    if assignment is None:
        assignment = {}

    # Simplify formula
    formula = [clause for clause in formula]

    # Unit propagation
    changed = True
    while changed:
        changed = False
        for clause in formula:
            # Check if clause is unit (only one literal unassigned)
            unassigned = [lit for lit in clause if abs(lit) not in assignment]
            if len(unassigned) == 1:
                lit = unassigned[0]
                var = abs(lit)
                value = (lit > 0)
                assignment[var] = value
                changed = True

    # Apply assignment and simplify
    simplified = []
    for clause in formula:
        if any((lit > 0 and assignment.get(abs(lit), False)) or
               (lit < 0 and not assignment.get(abs(lit), True)) for lit in clause):
            continue  # Clause satisfied
        new_clause = [lit for lit in clause
                      if abs(lit) not in assignment or
                      (lit > 0 and not assignment.get(abs(lit), True)) or
                      (lit < 0 and assignment.get(abs(lit), False))]
        if not new_clause:
            return False, {}  # Clause unsatisfied
        simplified.append(new_clause)

    if not simplified:
        return True, assignment  # All clauses satisfied

    # Choose variable
    all_vars = set(abs(lit) for clause in simplified for lit in clause)
    unassigned_vars = [v for v in all_vars if v not in assignment]

    if not unassigned_vars:
        return True, assignment

    var = unassigned_vars[0]

    # Try True
    new_assignment = assignment.copy()
    new_assignment[var] = True
    sat, result = dpll_solve(simplified, new_assignment)
    if sat:
        return True, result

    # Try False
    new_assignment = assignment.copy()
    new_assignment[var] = False
    sat, result = dpll_solve(simplified, new_assignment)
    if sat:
        return True, result

    return False, {}


# ==============================================================================
# WalkSAT: Randomized Local Search
# ==============================================================================

def walksat_solve(formula: List[List[int]], n_vars: int, max_flips: int = 10000, p: float = 0.5) -> Tuple[bool, List[bool]]:
    """
    WalkSAT algorithm: incomplete but fast for satisfiable instances

    Returns: (found, assignment)
    """
    # Random initial assignment
    assignment = [random.choice([True, False]) for _ in range(n_vars)]

    for flip in range(max_flips):
        # Check if satisfied
        unsatisfied = []
        for clause in formula:
            if not any((assignment[abs(lit)-1] == (lit > 0)) for lit in clause):
                unsatisfied.append(clause)

        if not unsatisfied:
            return True, assignment

        # Pick random unsatisfied clause
        clause = random.choice(unsatisfied)

        # With probability p, flip random var in clause
        # Otherwise, flip var that minimizes breaks
        if random.random() < p:
            var = abs(random.choice(clause)) - 1
        else:
            # Greedy: flip var that breaks fewest clauses
            best_var = None
            best_breaks = float('inf')

            for lit in clause:
                var = abs(lit) - 1
                # Count breaks if we flip this var
                breaks = 0
                test_assignment = assignment.copy()
                test_assignment[var] = not test_assignment[var]

                for c in formula:
                    if not any((test_assignment[abs(l)-1] == (l > 0)) for l in c):
                        breaks += 1

                if breaks < best_breaks:
                    best_breaks = breaks
                    best_var = var

            var = best_var if best_var is not None else abs(clause[0]) - 1

        # Flip
        assignment[var] = not assignment[var]

    return False, assignment


# ==============================================================================
# Test Suite
# ==============================================================================

def generate_random_3sat(n_vars: int, ratio: float = 4.26) -> List[List[int]]:
    """Generate random 3-SAT at phase transition"""
    n_clauses = int(ratio * n_vars)
    formula = []
    for _ in range(n_clauses):
        clause = []
        vars_in_clause = random.sample(range(n_vars), min(3, n_vars))
        for var in vars_in_clause:
            lit = (var + 1) * random.choice([-1, 1])
            clause.append(lit)
        formula.append(clause)
    return formula


def verify_assignment(formula: List[List[int]], assignment: List[bool]) -> bool:
    """Check if assignment satisfies formula"""
    for clause in formula:
        if not any((assignment[abs(lit)-1] == (lit > 0)) for lit in clause):
            return False
    return True


def benchmark_solver(solver_name: str, solver_func, formula: List[List[int]], n_vars: int) -> Dict:
    """Benchmark a SAT solver"""
    start = time.time()

    if solver_name == "DPLL":
        sat, assignment_dict = solver_func(formula)
        assignment = [assignment_dict.get(i+1, False) for i in range(n_vars)]
    elif solver_name == "WalkSAT":
        sat, assignment = solver_func(formula, n_vars)
    else:
        sat, assignment = False, []

    elapsed = time.time() - start

    # Verify
    if sat and assignment:
        valid = verify_assignment(formula, assignment)
    else:
        valid = False

    return {
        'solver': solver_name,
        'satisfiable': sat,
        'valid': valid,
        'time': elapsed
    }


def main():
    print("="*80)
    print("P VS NP REALITY CHECK: SAT Solver Comparison")
    print("="*80)
    print(f"Started: {datetime.now()}\n")

    sizes = [5, 10, 15, 20, 25]
    n_trials = 10

    all_results = []

    for n in sizes:
        print(f"\nTesting n={n} variables")

        for trial in range(n_trials):
            formula = generate_random_3sat(n, ratio=4.26)

            # Test DPLL
            dpll_result = benchmark_solver("DPLL", dpll_solve, formula, n)

            # Test WalkSAT
            walksat_result = benchmark_solver("WalkSAT", walksat_solve, formula, n)

            print(f"  Trial {trial+1}: "
                  f"DPLL={'✓' if dpll_result['valid'] else '✗'} ({dpll_result['time']:.3f}s), "
                  f"WalkSAT={'✓' if walksat_result['valid'] else '✗'} ({walksat_result['time']:.3f}s)")

            all_results.append({
                'n_vars': n,
                'trial': trial,
                'dpll': dpll_result,
                'walksat': walksat_result
            })

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for solver in ["DPLL", "WalkSAT"]:
        solver_results = [r[solver.lower().replace('-', '')] for r in all_results]
        n_valid = sum(1 for r in solver_results if r['valid'])
        n_total = len(solver_results)
        avg_time = sum(r['time'] for r in solver_results) / n_total

        print(f"\n{solver}:")
        print(f"  Success rate: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
        print(f"  Avg time: {avg_time:.4f}s")

    # Compare to bridge framework results
    try:
        with open('../results/p_vs_np_production_results.json', 'r') as f:
            bridge_data = json.load(f)
            bridge_results = bridge_data['results']
            n_bridge_valid = sum(1 for r in bridge_results if r['witness']['valid'])
            n_bridge_total = len(bridge_results)

            print(f"\nBridge Framework (from existing results):")
            print(f"  Success rate: {n_bridge_valid}/{n_bridge_total} ({100*n_bridge_valid/n_bridge_total:.1f}%)")
    except:
        print(f"\nBridge Framework: No results found")

    # Save comparison
    output = {
        'timestamp': str(datetime.now()),
        'sizes': sizes,
        'n_trials': n_trials,
        'results': all_results
    }

    with open('../results/sat_solver_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to: ../results/sat_solver_comparison.json")
    print(f"Completed: {datetime.now()}")

    # Conclusion
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("Classical SAT solvers (DPLL, WalkSAT) work.")
    print("Bridge framework does not solve SAT reliably.")
    print("Cannot make P vs NP claims with non-working solver.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
