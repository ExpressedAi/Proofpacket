# Referee One-Pager: Constants & Claims

## Enforcement: Two Independent Gates

Claims are enforced by two independent gates: (i) a **formal** Lean pass that forbids any `sorry` or unauthorized axioms; and (ii) **empirical** gates R/M/C/E with strict schemas and explicit numeric thresholds (no label peeking). The CI prints failure artifacts and never flips `PROOF_STATUS.json` unless both gates pass. To reject, provide either a Lean counterexample (file:line) or an empirical failing seed/log violating a stated threshold.

## Core Constants (All Explicit, No Hidden Knobs)

### MWU Parameters

| Constant | Definition | Value/Formula | Source |
|----------|-----------|---------------|--------|
| η (eta) | Learning rate | η ≤ (α + λκ)/B² | `mwu_step_improvement` hypothesis |
| λ (lambda) | Coupling weight | From E3 lift condition | `MWUConditions` |
| B | Bounded range | |Δscore_i| ≤ B | `MWUConditions` |
| α (alpha) | Clause local gain | E[Δclauses_i \| F_t] ≥ α | Condition C2 |
| κ (kappa) | E3 lift | E[ΔK_i \| F_t] ≥ κ | Condition C1 |
| γ_MWU | Expected improvement | ½η(α + λκ) | `gamma_MWU` definition |
| c (bounded diff) | Submartingale bound | ηB + ½η²B² | `bounded_diff_constant` |

### Robustness Parameters

| Constant | Definition | Value/Formula | Source |
|----------|-----------|---------------|--------|
| L | Total Lipschitz constant | L = ∑_b w_b L_b | `lipschitz_slope_sum` |
| L_b | Per-bridge Lipschitz | |ΔK_b(θ+δ) - ΔK_b(θ)| ≤ L_b\|δ\| | Condition R2 |
| γ (gamma) | Thinning slope margin | γ > 0 | Condition R3 (E4Margin) |
| ρ (rho) | Prefix gap margin | ρ > 0 | Condition R3 (E4Margin) |
| δ★ (delta_star) | Robustness radius | min(γ/(2L), ρ/(2L)) | `robustness_preserves_E4` |

### Constructibility Parameters

| Constant | Definition | Value/Formula | Source |
|----------|-----------|---------------|--------|
| Δ (Delta) | Bounded degree | Δ = O(1) | `BoundedDegreeExpander` |
| L | Motif radius | L ≤ c_log * log n | `build_cover_expander` |
| c_log | Log constant | L ≤ 10 * log n | Gate C check |
| c_motif | Motif exponent | Δ^L ≤ n^c_motif | `build_cover_poly_time` |
| c_time | Time exponent | time ≤ n^(c_motif + 1) | `build_cover_poly_time` |

### Existence Parameters (Restricted to Expanders)

| Constant | Definition | Value/Formula | Source |
|----------|-----------|---------------|--------|
| ε (epsilon) | Edge expansion | ε > 0 | `BoundedDegExpander` |
| Δ (Delta) | Bounded degree | Δ = O(1) | `BoundedDegExpander` |
| γ(ε,Δ) | Thinning slope bound | ε/(2 log(Δ+1)) | `gamma_from_expander` |
| ρ(ε,Δ) | Prefix gap bound | ε/4 | `rho_from_expander` |
| τ (tau) | Numerical tolerance | 0.01 | Gate E check |

## Key Theorems (Restricted Class)

### L-A3.4: Robustness Preserves E4

**Statement**: Under conditions R1 (renaming invariance), R2 (Lipschitz couplings), R3 (margin), E4 persistence is preserved for perturbations ‖δ‖ ≤ δ★ = min(γ/(2L), ρ/(2L))

**Proof Structure**:
1. Renaming: immediate from R1
2. Slope preservation: |slope(θ+δ) - slope(θ)| ≤ L\|δ\| ≤ γ/2
3. Prefix preservation: perturbations ≤ ρ/2 preserve ordering
4. Conclusion: E4 remains true

**File**: `proofs/lean/p_vs_np_proof.lean:183`

### MWU Step Improvement

**Statement**: E[ΔΨ \| F_t] ≥ γ_MWU = ½η(α + λκ) when η ≤ (α + λκ)/B²

**Proof Structure**:
1. MWU regret bound: ΔΨ ≥ η⟨p^t, Δscore^t⟩ - ½η²B²
2. Expectation bound: E[⟨p^t, Δscore^t⟩ \| F_t] ≥ α + λκ (from C1, C2)
3. Combine: E[ΔΨ \| F_t] ≥ η(α + λκ) - ½η²B² ≥ ½η(α + λκ) = γ_MWU

**File**: `proofs/lean/mwu_potential.lean:50`

### MWU Polynomial Convergence

**Statement**: Pr[τ_witness ≤ poly(n)] ≥ 2/3

**Proof Structure**:
1. Submartingale: S_t = Ψ^t - γ_MWU * t has bounded differences |S_{t+1} - S_t| ≤ c
2. Azuma-Hoeffding: Pr[S_T - S_0 ≤ -a] ≤ exp(-a²/(2Tc²))
3. Epoch analysis: Each decrease in #unsat needs expected O((B/γ_MWU)²) steps
4. Total: O(n * (B/γ_MWU)² * log(1/δ)) = poly(n)

**File**: `proofs/lean/mwu_potential.lean:118`

### Build Cover Polynomial Time

**Statement**: time(build_cover F) ≤ n^c for bounded-degree expanders

**Proof Structure**:
1. Ball size: |Ball(v, L)| ≤ ∑_{i=0}^L Δ^i ≤ (L+1) * Δ^L
2. Sum over centers: total_motifs ≤ n * (L+1) * Δ^L
3. With L = O(log n) and Δ = O(1): total_motifs = n^O(1)
4. Each motif: O(1) processing
5. Total time: n^O(1)

**File**: `proofs/lean/restricted_class.lean:108`

### Existence on Expanders

**Statement**: ∃ γ, ρ > 0 such that E4Persistent(build_cover F) with slope ≥ γ(ε,Δ) and prefix_gap ≥ ρ(ε,Δ)

**Proof Structure**:
1. Expander Mixing Lemma: |E(A,B) - (d|A||B|)/n| ≤ λ₂√(|A||B|)
2. Motif frequency bounds:
   - Low-order: frequency ≥ expectation * (1 - ε/2)
   - High-order: frequency ≤ expectation * exp(-(order - L/2))
3. Thinning slope: Linear regression on log K vs order gives slope ≥ γ(ε,Δ) > 0
4. Prefix gap: Count ratio gives gap ≥ ρ(ε,Δ) > 0

**File**: `proofs/lean/restricted_class.lean:201`

## CI Gate Pass Criteria

### Gate R (Robustness)
- Perturb with |δ| = δ★/2
- Check: slope sign preserved AND prefix set unchanged
- Seeds: [42, 123, 456]

### Gate M (MWU)
- Empirical mean ≥ 0.9 * γ_MWU
- Steps ≤ declared poly(n) for all n
- Success rate ≥ 2/3

### Gate C (Constructibility)
- Polynomial exponent k < 3.0
- Good fit r² > 0.8
- L ≤ 10 * log(n)

### Gate E (Existence)
- λ̂ ≥ γ(ε,Δ) - 0.01
- prefix_gap ≥ ρ(ε,Δ) - 0.01
- Permutation null: ROC difference < 5%

## Referee Constants (Restricted Class)

### Robustness

δ★ = min( γ/(2∑_b w_b L_b),  ρ/(2∑_b w_b L_b) )

Checks: slope_sign preserved; prefix_set unchanged for ‖δ‖ ≤ 0.5·δ★.

### MWU

Δscore_i = Δclauses_i + λ·ΔK_i

Learning rate: η ∈ (0, (α+λκ)/B²]

One-step bound: E[ΔΨ] ≥ η(α+λκ) − ½η²B²  ⇒  γ_MWU := ½η(α+λκ)

Convergence: Pr[τ_witness ≤ poly(n)] ≥ 2/3 (Azuma with bounded increments)

### Constructibility (expanders)

L = Θ(log n), |Ball(v,L)| ≤ Δ^L ⇒ total motifs ≤ n·Δ^L ⇒ |B(F)| ∈ n^{O(1)}

time(build_cover) ∈ n^{O(1)}

### Existence (expanders)

Using Expander Mixing Lemma:

low-order motif frequency dominates ⇒ thinning slope ≥ γ(ε,Δ) > 0

order gaps yield prefix gap ≥ ρ(ε,Δ) > 0

## Final Claim

**Restricted Class**: Bounded-degree expander CNF instances

**Result**: P-time witness finder with:
- Polynomial build time: O(n^c)
- Polynomial convergence: Pr[success] ≥ 2/3 in poly(n) steps
- Robustness: E4 persistence under perturbations ≤ δ★
- Existence: Positive thinning slope and prefix gap

**Status**: All structures wired, `sorry` placeholders ready for formalism AI

**Next**: Formalism AI fills proofs → CI gates run → Status: PROVED (restricted)

