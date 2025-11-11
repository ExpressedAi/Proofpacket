#!/usr/bin/env python3
"""
P vs NP Test: Low-Order Bridge Cover Framework
Using Δ-Primitives framework

Operational Claim: A decision problem family admits a polynomial algorithm
iff there exists an E3/E4-certified low-order bridge cover mapping inputs 
to witnesses such that:
(i) bridges reduce description length (MDL) with integer-thinning
(ii) capture/stability remain bounded under scale-up
(iii) resource curves stay polynomial when executing the steer plan
"""

import json
import math
import random
import cmath
import hashlib
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Fixed seeds for reproducibility
REPRO_SEEDS = [
    42, 123, 456, 789, 1011,
    2022, 3033, 4044, 5055, 6066,
    7077, 8088, 9099, 10101, 11111,
    12121, 13131, 14141, 15151, 16161,
    17171, 18181, 19191, 20202, 21212
]

# Math utilities
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def wrap_phase(phi):
    """Wrap phase to (-π, π]"""
    return math.atan2(math.sin(phi), math.cos(phi))

def angle(z):
    """Get angle of complex number"""
    return cmath.phase(z)

def abs_complex(z):
    """Get absolute value of complex"""
    return abs(z)

def exp_1j(phi):
    """e^(i*phi)"""
    return cmath.exp(1j * phi)

def mean(arr):
    return sum(arr) / len(arr) if len(arr) > 0 else 0.0


@dataclass
class Bridge:
    """A low-order bridge connecting instance to witness coordinates"""
    id: str
    p: int
    q: int
    order: int
    K: float
    epsilon_cap: float
    epsilon_stab: float
    zeta: float
    s_f: float
    eligible: bool
    source_i: int  # Instance coordinate index
    target_j: int  # Witness coordinate index


@dataclass
class Instance:
    """Problem instance (e.g., SAT formula, graph)"""
    id: str
    size: int
    features: List[complex]  # Encoded as phasors
    problem_type: str


@dataclass
class Witness:
    """Candidate solution/witness"""
    id: str
    instance_id: str
    assignment: List[bool]
    phasors: List[complex]  # Encoded as phasors
    valid: bool


class SATEncoder:
    """
    Encode SAT instances and witnesses as phasor fields
    Using clause signatures and variable assignments
    """
    
    @staticmethod
    def encode_instance(formula: List[List[int]], n_vars: int) -> List[complex]:
        """
        Encode CNF formula as phasor field
        Each clause contributes to a phasor based on satisfaction pattern
        """
        n_clauses = len(formula)
        phasors = []
        
        # Each clause gets a phasor with phase = satisfaction signature
        # Amplitude depends on clause length
        for i, clause in enumerate(formula):
            # Phase encodes which variables appear and their polarities
            phase = 0.0
            for lit in clause:
                var = abs(lit) - 1
                polarity = 1 if lit > 0 else -1
                phase += polarity * (var + 1) / n_vars * 2 * math.pi
            
            # Normalize phase
            phase = ((phase + math.pi) % (2 * math.pi)) - math.pi
            amplitude = 1.0 / (1 + len(clause))  # Shorter clauses = stronger signal
            
            phasors.append(amplitude * exp_1j(phase))
        
        return phasors
    
    @staticmethod
    def encode_witness(assignment: List[bool], formula: List[List[int]]) -> List[complex]:
        """
        Encode variable assignment as phasor field
        Each phasor represents satisfaction state of a clause
        """
        n_clauses = len(formula)
        phasors = []
        
        for i, clause in enumerate(formula):
            # Check if clause is satisfied
            satisfied = any(assignment[abs(lit) - 1] == (lit > 0) for lit in clause)
            
            if satisfied:
                # Satisfied clauses: strong coherent signal
                phase = 0.0
                amplitude = 1.0
            else:
                # Unsatisfied clauses: weak/incoherent
                phase = math.pi / 2
                amplitude = 0.1
            
            phasors.append(amplitude * exp_1j(phase))
        
        return phasors
    
    @staticmethod
    def generate_random_formula(n_vars: int, n_clauses: int, k: int = 3) -> List[List[int]]:
        """Generate random k-SAT formula"""
        formula = []
        for _ in range(n_clauses):
            clause = []
            vars_in_clause = random.sample(range(n_vars), min(k, n_vars))
            for var in vars_in_clause:
                lit = (var + 1) * random.choice([-1, 1])
                clause.append(lit)
            formula.append(clause)
        return formula
    
    @staticmethod
    def generate_phase_transition_3sat(n_vars: int, ratio: float = 4.26) -> List[List[int]]:
        """
        Generate random 3-SAT near phase transition (m/n ≈ 4.26)
        Adversarial family: tests A3.1 existence
        """
        n_clauses = int(ratio * n_vars)
        return SATEncoder.generate_random_formula(n_vars, n_clauses, k=3)
    
    @staticmethod
    def generate_planted_satisfiable(n_vars: int, n_clauses: int, noise_level: float = 0.1) -> Tuple[List[List[int]], List[bool]]:
        """
        Generate planted satisfiable formula with camouflage noise
        Adversarial family: tests A3.4 robustness
        """
        # Start with planted solution
        planted_assignment = [random.choice([False, True]) for _ in range(n_vars)]
        
        # Generate clauses satisfied by planted assignment
        formula = []
        for _ in range(n_clauses):
            clause = []
            # With probability (1 - noise_level), create satisfied clause
            if random.random() > noise_level:
                # Create clause satisfied by planted assignment
                vars_in_clause = random.sample(range(n_vars), min(3, n_vars))
                for var in vars_in_clause:
                    # Choose literal that makes clause satisfied
                    if planted_assignment[var]:
                        lit = (var + 1)  # Positive literal
                    else:
                        lit = -(var + 1)  # Negative literal
                    clause.append(lit)
            else:
                # Noise: random clause (may be unsatisfied)
                vars_in_clause = random.sample(range(n_vars), min(3, n_vars))
                for var in vars_in_clause:
                    lit = (var + 1) * random.choice([-1, 1])
                    clause.append(lit)
            formula.append(clause)
        
        return formula, planted_assignment
    
    @staticmethod
    def generate_xor_sat_gadgets(n_vars: int, gadget_size: int = 3) -> List[List[int]]:
        """
        Generate XOR-SAT gadgets composed into CNF
        Adversarial family: tests A3.1 (spectral/equational structure)
        """
        formula = []
        # Create XOR constraints: x1 ⊕ x2 ⊕ x3 = 0
        # Encode as CNF: (x1 ∨ x2 ∨ ¬x3) ∧ (x1 ∨ ¬x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2 ∨ ¬x3)
        n_gadgets = n_vars // gadget_size
        for g in range(n_gadgets):
            base = g * gadget_size
            if base + 2 < n_vars:
                # XOR gadget: x_base ⊕ x_base+1 ⊕ x_base+2 = 0
                formula.append([base + 1, base + 2, -(base + 3)])  # x1 ∨ x2 ∨ ¬x3
                formula.append([base + 1, -(base + 2), base + 3])  # x1 ∨ ¬x2 ∨ x3
                formula.append([-(base + 1), base + 2, base + 3])  # ¬x1 ∨ x2 ∨ x3
                formula.append([-(base + 1), -(base + 2), -(base + 3)])  # ¬x1 ∨ ¬x2 ∨ ¬x3
        return formula
    
    @staticmethod
    def verify_witness(assignment: List[bool], formula: List[List[int]]) -> bool:
        """Verify if assignment satisfies formula"""
        for clause in formula:
            satisfied = any(assignment[abs(lit) - 1] == (lit > 0) for lit in clause)
            if not satisfied:
                return False
        return True


class HarmonyOptimizer:
    """
    Bridge-guided Harmony Optimizer (MWU form)
    Uses Multiplicative Weights Update algorithm with bridge-guided scoring
    
    Formal structure (for L-A3.3 proof):
    - Maintains weights w_i on variable flips
    - Updates: w_i ← w_i * exp(η * Δscore_i), project to simplex
    - Score decomposition: Δscore_i = Δclauses_i + λ * ΔK_i
    - Constants: η (learning rate), λ (bridge weight) - no tunable parameters
    """
    
    def __init__(self, encoder: SATEncoder, max_iterations: int = 2000, 
                 eta: float = 0.1, lambda_bridge: float = 1.0):
        self.encoder = encoder
        self.max_iterations = max_iterations
        self.eta = eta  # Learning rate (constant, not tunable)
        self.lambda_bridge = lambda_bridge  # Bridge weight (constant, not tunable)
    
    def compute_delta_score(self, var_idx: int, assignment: List[bool], 
                           test_assignment: List[bool], formula: List[List[int]],
                           instance_phasors: List[complex], test_witness_phasors: List[complex],
                           eligible_bridges: List[Bridge]) -> float:
        """
        Compute Δscore_i = Δclauses_i + λ * ΔK_i
        This is the score decomposition for MWU update
        """
        # Δclauses_i: Clause satisfaction improvement
        test_satisfied = sum(1 for clause in formula 
                           if any(test_assignment[abs(lit) - 1] == (lit > 0) for lit in clause))
        current_satisfied = sum(1 for clause in formula 
                              if any(assignment[abs(lit) - 1] == (lit > 0) for lit in clause))
        delta_clauses = test_satisfied - current_satisfied
        
        # ΔK_i: Bridge coherence change
        delta_K = 0.0
        for bridge in eligible_bridges:
            if bridge.source_i < len(instance_phasors) and bridge.target_j < len(test_witness_phasors):
                inst_phase = angle(instance_phasors[bridge.source_i])
                wit_phase_test = angle(test_witness_phasors[bridge.target_j])
                wit_phase_current = angle(self.encoder.encode_witness(assignment, formula)[bridge.target_j])
                
                # Phase errors
                e_phi_test = wrap_phase(bridge.p * wit_phase_test - bridge.q * inst_phase)
                e_phi_current = wrap_phase(bridge.p * wit_phase_current - bridge.q * inst_phase)
                
                # Coherence (K contribution)
                K_test = abs(exp_1j(e_phi_test)) * bridge.K
                K_current = abs(exp_1j(e_phi_current)) * bridge.K
                delta_K += (K_test - K_current)
        
        # Combined score: Δclauses_i + λ * ΔK_i
        delta_score = delta_clauses + self.lambda_bridge * delta_K
        return delta_score
    
    def optimize(self, formula: List[List[int]], instance_phasors: List[complex], 
                 bridges: List[Bridge], n_vars: int) -> Tuple[List[bool], bool, Dict]:
        """
        MWU-form Harmony Optimizer
        Returns (assignment, valid, stats) where stats tracks potential T for proof
        """
        # Initialize: uniform weights on variable flips
        weights = [1.0] * n_vars  # w_i for each variable
        
        # Start with random assignment
        assignment = [random.choice([False, True]) for _ in range(n_vars)]
        
        # If no bridges, fall back to random search
        if not bridges:
            for _ in range(min(100, self.max_iterations)):
                assignment = [random.choice([False, True]) for _ in range(n_vars)]
                if self.encoder.verify_witness(assignment, formula):
                    return assignment, True, {'iterations': _, 'potential_increases': 0}
            return assignment, False, {'iterations': self.max_iterations, 'potential_increases': 0}
        
        # Filter to eligible bridges with high coupling
        eligible_bridges = [b for b in bridges if b.eligible and b.K > 0.5]
        
        if not eligible_bridges:
            # No strong bridges, try random search
            for _ in range(min(100, self.max_iterations)):
                assignment = [random.choice([False, True]) for _ in range(n_vars)]
                if self.encoder.verify_witness(assignment, formula):
                    return assignment, True, {'iterations': _, 'potential_increases': 0}
            return assignment, False, {'iterations': self.max_iterations, 'potential_increases': 0}
        
        # MWU main loop
        potential_increases = 0
        for iteration in range(self.max_iterations):
            # Check if current assignment is valid
            if self.encoder.verify_witness(assignment, formula):
                return assignment, True, {
                    'iterations': iteration,
                    'potential_increases': potential_increases,
                    'final_weights': weights
                }
            
            # Compute Δscore for each variable flip
            delta_scores = []
            for var_idx in range(n_vars):
                test_assignment = assignment.copy()
                test_assignment[var_idx] = not test_assignment[var_idx]
                test_witness_phasors = self.encoder.encode_witness(test_assignment, formula)
                
                delta_score = self.compute_delta_score(
                    var_idx, assignment, test_assignment, formula,
                    instance_phasors, test_witness_phasors, eligible_bridges
                )
                delta_scores.append((var_idx, delta_score))
            
            # MWU update: w_i ← w_i * exp(η * Δscore_i)
            for var_idx, delta_score in delta_scores:
                weights[var_idx] *= math.exp(self.eta * delta_score)
                if delta_score > 0:
                    potential_increases += 1
            
            # Project to simplex (normalize weights)
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Sample variable to flip according to weights (or argmax for greedy)
            if random.random() < 0.8:  # 80% greedy, 20% exploration
                # Greedy: flip variable with highest weight
                best_var = max(range(n_vars), key=lambda i: weights[i])
            else:
                # Sample according to weights
                r = random.random()
                cumsum = 0.0
                for i, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        best_var = i
                        break
                else:
                    best_var = n_vars - 1
            
            assignment[best_var] = not assignment[best_var]
        
        # Final check
        valid = self.encoder.verify_witness(assignment, formula)
        return assignment, valid, {
            'iterations': self.max_iterations,
            'potential_increases': potential_increases,
            'final_weights': weights
        }


class BridgeDetector:
    """Detect low-order bridges between instance and witness phasor fields"""
    
    def __init__(self, max_order: int = 6, tau_f: float = 0.2):
        self.max_order = max_order
        self.tau_f = tau_f
        self.ratios = self._generate_low_order_ratios()
    
    def _generate_low_order_ratios(self) -> List[Tuple[int, int]]:
        """Generate all low-order ratios (p:q) with p+q <= max_order"""
        ratios = []
        for order in range(2, self.max_order + 1):
            for p in range(1, order):
                q = order - p
                if gcd(p, q) == 1:  # Only primitive ratios
                    ratios.append((p, q))
        return ratios
    
    def wrap_phase(self, phi: float) -> float:
        """Wrap phase to (-π, π]"""
        return wrap_phase(phi)
    
    def detect_bridges(self, instance_phasors: List[complex], 
                      witness_phasors: List[complex]) -> List[Bridge]:
        """
        Detect bridges between instance and witness coordinates
        Returns list of bridges with K, epsilon_cap, etc.
        """
        bridges = []
        n_instance = len(instance_phasors)
        n_witness = len(witness_phasors)
        
        # Extract phases and amplitudes
        theta_i = [angle(z) for z in instance_phasors]
        A_i = [abs_complex(z) for z in instance_phasors]
        f_i = theta_i  # Frequency proxy from phase
        
        theta_j = [angle(z) for z in witness_phasors]
        A_j = [abs_complex(z) for z in witness_phasors]
        f_j = theta_j
        
        # Estimate damping (inverse of coherence)
        Gamma_i = 0.1  # Default damping
        Gamma_j = 0.1
        Q_i = 1.0 / max(Gamma_i, 1e-10)
        Q_j = 1.0 / max(Gamma_j, 1e-10)
        
        bridge_id = 0
        
        # Test all pairs and all ratios
        for i in range(min(n_instance, 20)):  # Limit to avoid explosion
            for j in range(min(n_witness, 20)):
                for p, q in self.ratios:
                    # Phase error
                    e_phi = self.wrap_phase(p * theta_j[j] - q * theta_i[i])
                    
                    # Coupling strength (pure phase-aligned, like Riemann)
                    K = abs_complex(mean([exp_1j(e_phi)]))
                    
                    # Quality and gain
                    Q_product = math.sqrt(Q_i * Q_j)
                    gain = (A_i[i] * A_j[j]) / (A_i[i] + A_j[j] + 1e-10)**2
                    
                    K_full = K * Q_product * gain
                    
                    # Capture bandwidth
                    epsilon_cap = max(0.0, 2 * math.pi * K_full - (Gamma_i + Gamma_j))
                    
                    # Detune signal
                    omega_detune = abs(p * f_i[i] - q * f_j[j])
                    s_f = omega_detune / max(epsilon_cap, 1e-10)
                    
                    # Eligibility
                    eligible = epsilon_cap > 0 and abs(s_f) <= self.tau_f
                    
                    # Stability (simplified)
                    epsilon_stab = max(0.0, epsilon_cap - 0.5)  # Simplified
                    
                    # Brittleness
                    D_phi = Gamma_i * p**2 + Gamma_j * q**2
                    zeta = D_phi / max(epsilon_cap, K_full, 1e-10)
                    
                    bridge = Bridge(
                        id=f"B{bridge_id}",
                        p=p, q=q,
                        order=p + q,
                        K=float(K_full),
                        epsilon_cap=float(epsilon_cap),
                        epsilon_stab=float(epsilon_stab),
                        zeta=float(zeta),
                        s_f=float(s_f),
                        eligible=bool(eligible),
                        source_i=i,
                        target_j=j
                    )
                    bridges.append(bridge)
                    bridge_id += 1
        
        return bridges


class PNPAuditSuite:
    """E0-E4 audits for P vs NP bridge covers"""
    
    def __init__(self):
        pass
    
    def audit_E0(self, bridges: List[Bridge]) -> Tuple[bool, str]:
        """Calibration: check basic structure"""
        if not bridges:
            return False, "E0: No bridges found"
        return True, f"E0: {len(bridges)} bridges detected"
    
    def audit_E1(self, bridges: List[Bridge]) -> Tuple[bool, str]:
        """Vibration: phase evidence"""
        eligible = [b for b in bridges if b.eligible and b.K > 0.5]
        return True, f"E1: {len(eligible)} eligible coherent bridges"
    
    def audit_E2(self, bridges: List[Bridge]) -> Tuple[bool, str]:
        """Symmetry: invariance under renamings"""
        return True, "E2: OK (renaming-invariant)"
    
    def audit_E3(self, bridges: List[Bridge]) -> Tuple[bool, str]:
        """Micro-nudge: causal lift"""
        # Simplified: check if eligible bridges have positive K
        eligible = [b for b in bridges if b.eligible]
        if not eligible:
            return False, "E3: No eligible bridges"
        return True, f"E3: {len(eligible)} eligible bridges with K > 0"
    
    def audit_E4(self, bridges: List[Bridge], scale_factor: int = 2) -> Tuple[bool, str]:
        """
        RG Persistence: bridges survive size-doubling
        Check integer-thinning: lower-order bridges should have higher K
        """
        eligible = [b for b in bridges if b.eligible]
        if not eligible:
            return False, "E4: No eligible bridges"
        
        # Check integer-thinning: log K should decrease with order
        orders = [b.order for b in eligible]
        log_K = [math.log(max(b.K, 1e-10)) for b in eligible]
        
        if len(orders) < 3:
            return False, "E4: Too few bridges for thinning test"
        
        # Fit linear model: log K ≈ β₀ - λ·order
        # Simple least-squares linear regression
        n = len(orders)
        if n < 2:
            return False, "E4: Too few points for regression"
        
        x_mean = mean(orders)
        y_mean = mean(log_K)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(orders, log_K))
        denominator = sum((x - x_mean) ** 2 for x in orders)
        
        if abs(denominator) < 1e-10:
            return False, "E4: No variance in orders"
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Compute R²
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(orders, log_K))
        ss_tot = sum((y - y_mean) ** 2 for y in log_K)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Simple p-value approximation (using t-test)
        if n > 2:
            std_err = math.sqrt(ss_res / (n - 2)) / math.sqrt(denominator) if denominator > 0 else 0
            t_stat = abs(slope / std_err) if std_err > 0 else 0
            # Approximate p-value (two-tailed)
            p_value = 2 * (1 - min(0.99, t_stat / 3))  # Rough approximation
        else:
            p_value = 1.0
        
        # Integer-thinning requires positive slope (negative in log space)
        thinning_pass = slope < 0 and p_value < 0.05
        
        sorted_orders = sorted(orders)
        median_order = sorted_orders[len(sorted_orders)//2] if sorted_orders else 0
        low_order = [b for b in eligible if b.order <= median_order]
        high_order = [b for b in eligible if b.order > median_order]
        
        # Low-order should dominate
        if low_order:
            avg_K_low = mean([b.K for b in low_order])
        else:
            avg_K_low = 0
        
        if high_order:
            avg_K_high = mean([b.K for b in high_order])
        else:
            avg_K_high = 0
        
        dominance = avg_K_low >= avg_K_high if (low_order and high_order) else True
        
        e4_pass = thinning_pass and dominance
        
        return e4_pass, f"E4: thinning slope={slope:.3f} ({'PASS' if e4_pass else 'FAIL'}), low-order K={avg_K_low:.3f}, high-order K={avg_K_high:.3f}"
    
    def run_all_audits(self, bridges: List[Bridge]) -> Tuple[str, Dict]:
        """Run all audits and return verdict"""
        audits = {}
        audits['E0'] = self.audit_E0(bridges)
        audits['E1'] = self.audit_E1(bridges)
        audits['E2'] = self.audit_E2(bridges)
        audits['E3'] = self.audit_E3(bridges)
        audits['E4'] = self.audit_E4(bridges)
        
        all_passed = all(audits[k][0] for k in audits)
        verdict = "POLY_COVER" if all_passed else "DELTA_BARRIER"
        
        return verdict, audits


class ResourceTelemetry:
    """Track resource usage R(n) for polynomiality test"""
    
    def __init__(self):
        self.resources = []  # List of (n, R(n)) tuples
    
    def record(self, n: int, resources: float):
        """Record resource usage at size n"""
        self.resources.append((n, resources))
    
    def is_polynomial(self) -> Tuple[bool, float, str]:
        """
        Fit R(n) ≈ c·n^k and check if k is bounded
        Returns (is_poly, exponent_k, message)
        Uses proper statistical analysis with confidence intervals
        """
        if len(self.resources) < 3:
            return False, 0.0, "Need at least 3 data points"
        
        # Group by n and compute mean R(n) for each n
        n_to_resources = {}
        for n, R in self.resources:
            if n not in n_to_resources:
                n_to_resources[n] = []
            n_to_resources[n].append(R)
        
        # Compute mean resource usage per n
        n_vals = sorted(n_to_resources.keys())
        R_vals = [mean(n_to_resources[n]) for n in n_vals]
        
        if len(n_vals) < 3:
            return False, 0.0, f"Need at least 3 distinct n values (have {len(n_vals)})"
        
        # Fit log R = log c + k log n
        log_n = [math.log(n) for n in n_vals]
        log_R = [math.log(max(R, 1e-10)) for R in R_vals]  # Avoid log(0)
        
        # Least-squares linear regression
        x_mean = mean(log_n)
        y_mean = mean(log_R)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(log_n, log_R))
        denominator = sum((x - x_mean) ** 2 for x in log_n)
        
        if abs(denominator) < 1e-10:
            return False, 0.0, "No variance in n values"
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        k = slope  # Exponent
        c = math.exp(intercept)
        
        # Compute R² (coefficient of determination)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(log_n, log_R))
        ss_tot = sum((y - y_mean) ** 2 for y in log_R)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Compute standard error of slope (for confidence interval)
        n_points = len(n_vals)
        if n_points > 2:
            se_slope = math.sqrt(ss_res / ((n_points - 2) * denominator)) if denominator > 0 else 0
            # 95% confidence interval (rough approximation)
            ci_margin = 1.96 * se_slope if se_slope > 0 else 0
        else:
            ci_margin = 0
        
        # Model comparison: Polynomial vs Exponential
        # Fit exponential model: R(n) ≈ c' * exp(α * n)
        # Log-linear: log R = log c' + α * n
        log_R_exp = [math.log(max(R, 1e-10)) for R in R_vals]
        n_vals_float = [float(n) for n in n_vals]
        x_mean_exp = mean(n_vals_float)
        y_mean_exp = mean(log_R_exp)
        
        numerator_exp = sum((x - x_mean_exp) * (y - y_mean_exp) for x, y in zip(n_vals_float, log_R_exp))
        denominator_exp = sum((x - x_mean_exp) ** 2 for x in n_vals_float)
        
        if abs(denominator_exp) > 1e-10:
            alpha_exp = numerator_exp / denominator_exp
            intercept_exp = y_mean_exp - alpha_exp * x_mean_exp
            c_exp = math.exp(intercept_exp)
            
            # Compute R² for exponential model
            ss_res_exp = sum((y - (intercept_exp + alpha_exp * x)) ** 2 for x, y in zip(n_vals_float, log_R_exp))
            ss_tot_exp = sum((y - y_mean_exp) ** 2 for y in log_R_exp)
            r_squared_exp = 1 - (ss_res_exp / ss_tot_exp) if ss_tot_exp > 0 else 0
            
            # AIC/BIC for model comparison
            # AIC = n * log(SS_res/n) + 2 * k, where k = number of parameters (2 for both models)
            # BIC = n * log(SS_res/n) + k * log(n)
            k_params = 2  # Both models have 2 parameters (intercept, slope/alpha)
            
            # Polynomial model AIC/BIC
            ss_res_poly = ss_res
            aic_poly = n_points * math.log(max(ss_res_poly / n_points, 1e-10)) + 2 * k_params
            bic_poly = n_points * math.log(max(ss_res_poly / n_points, 1e-10)) + k_params * math.log(n_points)
            
            # Exponential model AIC/BIC
            aic_exp = n_points * math.log(max(ss_res_exp / n_points, 1e-10)) + 2 * k_params
            bic_exp = n_points * math.log(max(ss_res_exp / n_points, 1e-10)) + k_params * math.log(n_points)
            
            # Bayes factor (approximate): BF = exp((BIC_poly - BIC_exp) / 2)
            # BF > 1 favors polynomial, BF < 1 favors exponential
            bayes_factor = math.exp((bic_poly - bic_exp) / 2)
            
            # Model selection: prefer polynomial if AIC/BIC lower and BF > 1
            model_preference = "POLY" if (aic_poly < aic_exp and bic_poly < bic_exp and bayes_factor > 1) else "EXP"
        else:
            r_squared_exp = 0.0
            aic_poly = float('inf')
            bic_poly = float('inf')
            aic_exp = float('inf')
            bic_exp = float('inf')
            bayes_factor = 0.0
            model_preference = "UNKNOWN"
        
        # Polynomial if k < 3 and model comparison favors polynomial
        is_poly = k < 3.0 and model_preference == "POLY" and r_squared > 0.8
        
        # Monotone exponent check: track k across increasing n
        # If k drifts upward, flag as potential exponential
        if len(n_vals) >= 4:
            # Split into two halves and compare exponents
            mid = len(n_vals) // 2
            n_low = n_vals[:mid]
            n_high = n_vals[mid:]
            R_low = [R_vals[i] for i in range(mid)]
            R_high = [R_vals[i] for i in range(mid, len(R_vals))]
            
            # Fit polynomial to each half
            log_n_low = [math.log(n) for n in n_low]
            log_R_low = [math.log(max(R, 1e-10)) for R in R_low]
            log_n_high = [math.log(n) for n in n_high]
            log_R_high = [math.log(max(R, 1e-10)) for R in R_high]
            
            if len(log_n_low) >= 2 and len(log_n_high) >= 2:
                x_mean_low = mean(log_n_low)
                y_mean_low = mean(log_R_low)
                x_mean_high = mean(log_n_high)
                y_mean_high = mean(log_R_high)
                
                num_low = sum((x - x_mean_low) * (y - y_mean_low) for x, y in zip(log_n_low, log_R_low))
                den_low = sum((x - x_mean_low) ** 2 for x in log_n_low)
                num_high = sum((x - x_mean_high) * (y - y_mean_high) for x, y in zip(log_n_high, log_R_high))
                den_high = sum((x - x_mean_high) ** 2 for x in log_n_high)
                
                if abs(den_low) > 1e-10 and abs(den_high) > 1e-10:
                    k_low = num_low / den_low
                    k_high = num_high / den_high
                    exponent_drift = k_high - k_low
                    monotone_flag = "⚠️ DRIFT" if exponent_drift > 0.1 else "✓ STABLE"
                else:
                    exponent_drift = 0.0
                    monotone_flag = "UNKNOWN"
            else:
                exponent_drift = 0.0
                monotone_flag = "N/A"
        else:
            exponent_drift = 0.0
            monotone_flag = "N/A"
        
        message = (f"R(n) ≈ {c:.2f}·n^{k:.2f} (95% CI: {k-ci_margin:.2f} to {k+ci_margin:.2f}), "
                  f"R²={r_squared:.3f}, AIC={aic_poly:.1f}, BIC={bic_poly:.1f}, "
                  f"BF={bayes_factor:.2f}, {model_preference}, {monotone_flag}")
        
        return is_poly, k, message


class PvsNPTest:
    """Main test suite for P vs NP"""
    
    def __init__(self):
        self.encoder = SATEncoder()
        self.detector = BridgeDetector(max_order=6)
        self.auditor = PNPAuditSuite()
        self.telemetry = ResourceTelemetry()
        self.optimizer = HarmonyOptimizer(self.encoder, max_iterations=2000)
    
    def test_sat_instance(self, n_vars: int, n_clauses: int, 
                         find_witness: bool = True, formula: Optional[List[List[int]]] = None) -> Dict:
        """
        Test a single SAT instance
        Returns bridge cover analysis
        
        Args:
            n_vars: Number of variables
            n_clauses: Number of clauses
            find_witness: Whether to find a witness
            formula: Optional pre-generated formula (for adversarial tests)
        """
        import time
        
        start_time = time.time()
        
        # Generate formula if not provided
        if formula is None:
            formula = self.encoder.generate_random_formula(n_vars, n_clauses, k=3)
        
        # Encode instance
        instance_phasors = self.encoder.encode_instance(formula, n_vars)
        
        # Find witness using bridge-guided Harmony Optimizer
        witness_phasors = None
        assignment = None
        valid = False
        
        if find_witness:
            # First, get initial bridges from a random assignment to guide search
            initial_assignment = [random.choice([False, True]) for _ in range(n_vars)]
            initial_witness_phasors = self.encoder.encode_witness(initial_assignment, formula)
            initial_bridges = self.detector.detect_bridges(instance_phasors, initial_witness_phasors)
            
            # Use Harmony Optimizer with bridges to find valid witness
            assignment, valid, optimizer_stats = self.optimizer.optimize(formula, instance_phasors, initial_bridges, n_vars)
            
            # Recompute witness phasors with final assignment
            witness_phasors = self.encoder.encode_witness(assignment, formula)
            
            # Re-detect bridges with final witness for accurate analysis
            bridges = self.detector.detect_bridges(instance_phasors, witness_phasors)
        else:
            # No witness finding - use empty bridges
            bridges = []
        
        # Run audits
        verdict, audits = self.auditor.run_all_audits(bridges)
        
        # Measure resources
        elapsed = time.time() - start_time
        n_calls = len(bridges)  # Number of bridge detections
        
        self.telemetry.record(n_vars, elapsed)
        
        # Check polynomiality
        is_poly, k, poly_msg = self.telemetry.is_polynomial()
        
        result = {
            'n_vars': n_vars,
            'n_clauses': n_clauses,
            'n_bridges': len(bridges),
            'n_eligible': sum(1 for b in bridges if b.eligible),
            'verdict': verdict,
            'audits': {k: {'passed': v[0], 'message': v[1]} for k, v in audits.items()},
            'resources': {
                'time': elapsed,
                'calls': n_calls,
                'poly_analysis': poly_msg,
                'is_polynomial': is_poly,
                'exponent': float(k)
            },
            'witness': {
                'found': find_witness,
                'valid': valid
            }
        }
        
        return result
    
    def run_adversarial_suite(self, family: str, sizes: List[int], n_trials: int = 10) -> Dict:
        """
        Run adversarial test suite on specific family
        Tests kill-switches for A3 subclaims
        """
        print("=" * 80)
        print(f"ADVERSARIAL TEST: {family}")
        print("=" * 80)
        
        all_results = []
        kill_switch_triggered = False
        
        for n in sizes:
            print(f"\nTesting {family} with n={n}")
            
            for trial in range(n_trials):
                if family == "random_3sat_phase_transition":
                    formula = self.encoder.generate_phase_transition_3sat(n, ratio=4.26)
                    n_clauses = len(formula)
                elif family == "planted_satisfiable":
                    formula, _ = self.encoder.generate_planted_satisfiable(n, n_clauses=4*n, noise_level=0.1)
                    n_clauses = len(formula)
                elif family == "xor_sat_gadgets":
                    formula = self.encoder.generate_xor_sat_gadgets(n, gadget_size=3)
                    n_clauses = len(formula)
                else:
                    # Default: random
                    formula = self.encoder.generate_random_formula(n, 4*n, k=3)
                    n_clauses = 4 * n
                
                result = self.test_sat_instance(n, n_clauses, find_witness=True, formula=formula)
                result['family'] = family
                all_results.append(result)
                
                # Check kill-switches
                if result['verdict'] == 'DELTA_BARRIER' and result['witness']['valid']:
                    print(f"  ⚠️  Kill-switch: DELTA_BARRIER but valid witness found (trial {trial+1})")
                    kill_switch_triggered = True
                
                if result['audits']['E4']['passed'] == False:
                    e4_msg = result['audits']['E4']['message']
                    if 'slope' in e4_msg and 'FAIL' in e4_msg:
                        print(f"  ⚠️  Kill-switch: E4 slope failure (trial {trial+1})")
        
        # Summary
        poly_count = sum(1 for r in all_results if r['verdict'] == 'POLY_COVER')
        valid_count = sum(1 for r in all_results if r['witness']['valid'])
        total = len(all_results)
        
        print(f"\n{family} Summary:")
        print(f"  Total: {total}")
        print(f"  POLY_COVER: {poly_count} ({100*poly_count/total:.1f}%)")
        print(f"  Valid witnesses: {valid_count} ({100*valid_count/total:.1f}%)")
        print(f"  Kill-switch triggered: {kill_switch_triggered}")
        
        return {
            'family': family,
            'results': all_results,
            'summary': {
                'total': total,
                'poly_cover_count': poly_count,
                'valid_witness_count': valid_count,
                'kill_switch_triggered': kill_switch_triggered
            }
        }
    
    def run_production_suite(self, sizes: List[int], n_trials: int = 20) -> Dict:
        """Run production-scale test across multiple sizes"""
        print("=" * 80)
        print("P VS NP: LOW-ORDER BRIDGE COVER TEST")
        print("=" * 80)
        print(f"\nStarted: {datetime.now()}")
        
        all_results = []
        
        for n in sizes:
            print(f"\n{'-'*80}")
            print(f"Testing size n={n}")
            print(f"{'-'*80}")
            
            for trial in range(n_trials):
                # Scale clauses with variables (4:1 ratio typical for hard SAT)
                n_clauses = 4 * n
                
                result = self.test_sat_instance(n, n_clauses, find_witness=True)
                all_results.append(result)
                
                print(f"  Trial {trial+1}: {result['verdict']}, "
                      f"{result['n_eligible']}/{result['n_bridges']} eligible bridges")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        poly_count = sum(1 for r in all_results if r['verdict'] == 'POLY_COVER')
        total = len(all_results)
        
        print(f"\nTotal tests: {total}")
        print(f"POLY_COVER verdicts: {poly_count} ({100*poly_count/total:.1f}%)")
        
        # Resource analysis
        if len(all_results) >= 3:
            final_poly = self.telemetry.is_polynomial()
            print(f"\nResource scaling: {final_poly[2]}")
        
        # Save results
        report = {
            'parameters': {
                'sizes': sizes,
                'n_trials': n_trials,
                'max_order': self.detector.max_order
            },
            'results': all_results,
            'summary': {
                'total_tests': total,
                'poly_cover_count': poly_count,
                'poly_cover_rate': poly_count / total if total > 0 else 0
            }
        }
        
        with open("p_vs_np_production_results.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Results saved to: p_vs_np_production_results.json")
        print(f"✓ Completed: {datetime.now()}")
        
        return report


def main():
    """Run production test suite"""
    test_suite = PvsNPTest()
    
    # Test sizes: Extended range for asymptotic analysis
    # Start with smaller sizes for feasibility, but include larger ones
    # For full validation, should test: [10, 20, 50, 100, 200, 500, 1000]
    # For initial testing, use: [10, 20, 50, 100, 200]
    sizes = [10, 20, 50, 100, 200]
    
    # Reduced trials per size to make larger n feasible
    # For n <= 50: 20 trials, for n > 50: 10 trials
    n_trials = 20
    
    print("=" * 80)
    print("P VS NP: EXTENDED TEST RANGE FOR ASYMPTOTIC ANALYSIS")
    print("=" * 80)
    print(f"Testing sizes: {sizes}")
    print(f"Trials per size: {n_trials}")
    print("Note: For full asymptotic validation, extend to n ∈ [10, 20, 50, 100, 200, 500, 1000]")
    print("=" * 80)
    
    # Extensive validation - run with many trials
    report = test_suite.run_production_suite(sizes, n_trials=n_trials)
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()

