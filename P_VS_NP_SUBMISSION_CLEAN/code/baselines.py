#!/usr/bin/env python3
"""
Baseline SAT Solvers for Comparison
WalkSAT, GSAT, Random Restart
"""

import random
import math
from typing import List, Tuple, Dict, Optional

class WalkSAT:
    """WalkSAT algorithm (baseline)"""
    
    def __init__(self, max_flips: int = 10000, noise: float = 0.57):
        self.max_flips = max_flips
        self.noise = noise
    
    def solve(self, formula: List[List[int]], n_vars: int) -> Tuple[List[bool], bool, Dict]:
        """Solve SAT instance using WalkSAT"""
        # Random initial assignment
        assignment = [random.choice([False, True]) for _ in range(n_vars)]
        
        for flip in range(self.max_flips):
            # Check if satisfied
            if self.verify(assignment, formula):
                return assignment, True, {'flips': flip}
            
            # Find unsatisfied clauses
            unsatisfied = []
            for i, clause in enumerate(formula):
                if not any(assignment[abs(lit) - 1] == (lit > 0) for lit in clause):
                    unsatisfied.append(i)
            
            if not unsatisfied:
                return assignment, True, {'flips': flip}
            
            # Pick random unsatisfied clause
            clause_idx = random.choice(unsatisfied)
            clause = formula[clause_idx]
            
            # With probability noise, flip random variable in clause
            # Otherwise, flip variable that minimizes broken clauses
            if random.random() < self.noise:
                lit = random.choice(clause)
                var_idx = abs(lit) - 1
            else:
                # Greedy: flip variable that minimizes broken clauses
                best_var = None
                best_broken = float('inf')
                for lit in clause:
                    var_idx = abs(lit) - 1
                    test_assignment = assignment.copy()
                    test_assignment[var_idx] = not test_assignment[var_idx]
                    broken = sum(1 for c in formula 
                               if not any(test_assignment[abs(l) - 1] == (l > 0) for l in c))
                    if broken < best_broken:
                        best_broken = broken
                        best_var = var_idx
                var_idx = best_var if best_var is not None else abs(clause[0]) - 1
            
            assignment[var_idx] = not assignment[var_idx]
        
        return assignment, False, {'flips': self.max_flips}
    
    @staticmethod
    def verify(assignment: List[bool], formula: List[List[int]]) -> bool:
        """Verify if assignment satisfies formula"""
        for clause in formula:
            if not any(assignment[abs(lit) - 1] == (lit > 0) for lit in clause):
                return False
        return True


class GSAT:
    """GSAT algorithm (baseline)"""
    
    def __init__(self, max_flips: int = 10000, max_tries: int = 10):
        self.max_flips = max_flips
        self.max_tries = max_tries
    
    def solve(self, formula: List[List[int]], n_vars: int) -> Tuple[List[bool], bool, Dict]:
        """Solve SAT instance using GSAT"""
        for try_num in range(self.max_tries):
            # Random initial assignment
            assignment = [random.choice([False, True]) for _ in range(n_vars)]
            
            for flip in range(self.max_flips):
                # Check if satisfied
                if WalkSAT.verify(assignment, formula):
                    return assignment, True, {'flips': flip, 'tries': try_num + 1}
                
                # Find variable that maximizes satisfied clauses
                best_var = None
                best_satisfied = -1
                
                for var_idx in range(n_vars):
                    test_assignment = assignment.copy()
                    test_assignment[var_idx] = not test_assignment[var_idx]
                    satisfied = sum(1 for clause in formula 
                                  if any(test_assignment[abs(lit) - 1] == (lit > 0) for lit in clause))
                    if satisfied > best_satisfied:
                        best_satisfied = satisfied
                        best_var = var_idx
                
                if best_var is not None:
                    assignment[best_var] = not assignment[best_var]
            
        return assignment, False, {'flips': self.max_flips, 'tries': self.max_tries}


class RandomRestart:
    """Random restart baseline"""
    
    def __init__(self, max_restarts: int = 1000, max_flips_per_restart: int = 100):
        self.max_restarts = max_restarts
        self.max_flips_per_restart = max_flips_per_restart
    
    def solve(self, formula: List[List[int]], n_vars: int) -> Tuple[List[bool], bool, Dict]:
        """Solve SAT instance using random restart"""
        total_flips = 0
        
        for restart in range(self.max_restarts):
            # Random initial assignment
            assignment = [random.choice([False, True]) for _ in range(n_vars)]
            
            # Check if satisfied
            if WalkSAT.verify(assignment, formula):
                return assignment, True, {'restarts': restart + 1, 'flips': total_flips}
            
            # Random flips
            for _ in range(self.max_flips_per_restart):
                var_idx = random.randint(0, n_vars - 1)
                assignment[var_idx] = not assignment[var_idx]
                total_flips += 1
                
                if WalkSAT.verify(assignment, formula):
                    return assignment, True, {'restarts': restart + 1, 'flips': total_flips}
        
        return assignment, False, {'restarts': self.max_restarts, 'flips': total_flips}


def run_baselines(formula: List[List[int]], n_vars: int, 
                  budget: int = 10000) -> Dict:
    """
    Run all baselines with matched budget
    Returns comparison results
    """
    results = {}
    
    # WalkSAT
    walksat = WalkSAT(max_flips=budget)
    assignment, valid, stats = walksat.solve(formula, n_vars)
    results['WalkSAT'] = {
        'valid': valid,
        'flips': stats.get('flips', budget),
        'time': stats.get('flips', budget) * 0.001  # Rough time estimate
    }
    
    # GSAT
    gsat = GSAT(max_flips=budget // 10, max_tries=10)
    assignment, valid, stats = gsat.solve(formula, n_vars)
    results['GSAT'] = {
        'valid': valid,
        'flips': stats.get('flips', budget),
        'tries': stats.get('tries', 10),
        'time': stats.get('flips', budget) * 0.001
    }
    
    # Random Restart
    rr = RandomRestart(max_restarts=budget // 100, max_flips_per_restart=100)
    assignment, valid, stats = rr.solve(formula, n_vars)
    results['RandomRestart'] = {
        'valid': valid,
        'restarts': stats.get('restarts', budget // 100),
        'flips': stats.get('flips', budget),
        'time': stats.get('flips', budget) * 0.001
    }
    
    return results

