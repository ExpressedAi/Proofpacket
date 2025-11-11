# Formalism AI Requirements: Exact Specifications

## Overview

The formalism AI needs to fill **43 `sorry` placeholders** in the Lean proof files. Each placeholder has:
- Exact file:line location
- Clear proof strategy already documented
- References to standard theory (MWU, Azuma, expander mixing, etc.)
- Explicit constants already defined

## Priority Order (Recommended)

### Priority 1: Core Helper Lemmas (Most Critical)

These are the foundation for the main theorems:

#### 1. `lipschitz_slope_sum` 
**File**: `proofs/lean/p_vs_np_proof.lean:100`

**What's needed**: Complete the `calc` chain proving that sum of Lipschitz functions is Lipschitz.

**Current state**: Structure is 90% complete, just needs:
- Apply sum linearity: `|∑ w_b (ΔK_b θ - ΔK_b θ')| = |∑ w_b * (ΔK_b θ - ΔK_b θ')|`
- Apply triangle inequality: `≤ ∑ |w_b * (ΔK_b θ - ΔK_b θ')|`
- Use `hw_nonneg`: `≤ ∑ w_b * |ΔK_b θ - ΔK_b θ'|`
- Apply Lipschitz property: `≤ ∑ w_b * L_b * ‖θ - θ'‖`
- Factor out: `= (∑ w_b * L_b) * ‖θ - θ'‖`

**Standard theory**: Use `LipschitzWith.sum` from mathlib or prove directly with triangle inequality.

#### 2. `prefix_stability_gap`
**File**: `proofs/lean/p_vs_np_proof.lean:133`

**What's needed**: Complete the calculation showing that if gaps ≥ ρ and perturbations ≤ ρ/2, then ordering is preserved.

**Current state**: Structure is 80% complete, needs:
- Complete calculation: `s' j ≤ s j + ε ≤ (s i - ρ) + ε ≤ (s' i + ε) - ρ + ε = s' i - (ρ - 2ε)`
- Since `ρ - 2ε ≥ 0`, we have `s' j < s' i`
- Symmetric argument for reverse direction

**Standard theory**: Use `Finset` operations and `lt_of_le_of_lt`.

#### 3. `mwu_regret_bound`
**File**: `proofs/lean/mwu_potential.lean:38`

**What's needed**: Prove standard MWU regret bound using Hoeffding's lemma.

**Current state**: Structure is 70% complete, needs:
- Apply Hoeffding's lemma: `E[exp(η X)] ≤ exp(η E[X] + ½η²B²)` for `|X| ≤ B`
- Apply to exponential weights: `log(∑ p_i exp(η g_i)) ≥ η ∑ p_i g_i - ½η²B²`

**Standard theory**: This is standard MWU regret analysis. Can import from mathlib or prove using convexity of log-sum-exp.

#### 4. `azuma_hoeffding_bounded`
**File**: `proofs/lean/mwu_potential.lean:157`

**What's needed**: Prove Azuma-Hoeffding for bounded increments.

**Current state**: Structure is 60% complete, needs:
- Use Chernoff bound: `Pr[S_T - S_0 ≤ -a] ≤ E[exp(-λ(S_T - S_0))] * exp(λa)`
- Bound expectation using submartingale property and bounded increments
- Optimize λ to get `exp(-a²/(2Tc²))`

**Standard theory**: Standard Azuma-Hoeffding. Can import from mathlib or prove using Chernoff method.

### Priority 2: Constructibility (Restricted Class)

#### 5. `ball_size_le`
**File**: `proofs/lean/restricted_class.lean:58`

**What's needed**: Complete induction proof for BFS tree bound.

**Current state**: Structure is 80% complete, needs:
- Base case: `|Ball(v, 0)| = 1 ≤ Δ^0 = 1` ✓ (already done)
- Inductive step: `|Ball(v, L+1)| ≤ Δ · |Ball(v, L)| ≤ Δ · (∑_{i=0}^L Δ^i) = ∑_{i=0}^{L+1} Δ^i`

**Standard theory**: Standard BFS tree expansion bound.

#### 6. `sum_motifs_poly`
**File**: `proofs/lean/restricted_class.lean:77`

**What's needed**: Complete sum bound over centers.

**Current state**: Structure is 70% complete, needs:
- Connect motifs to ball: `motifs(v, L) ⊆ Ball(v, L)`
- Sum over vertices: `total_motifs = ∑_v |motifs(v, L)| ≤ ∑_v |Ball(v, L)|`
- Apply `ball_size_le`: `≤ n * (∑_{i=0}^L Δ^i)`
- Bound geometric sum: `≤ n * (L+1) * Δ^L`
- With L = O(log n): `≤ n * log(n) * n^O(1) = n^O(1)`

**Standard theory**: Standard sum bounds and geometric series.

### Priority 3: Existence (Restricted Class)

#### 7. `existence_on_expanders` and helpers
**Files**: `proofs/lean/restricted_class.lean:152-199`

**What's needed**: 
- Apply expander mixing lemma to motif structure
- Derive frequency bounds (low-order vs high-order)
- Prove thinning slope > 0 from frequency bounds
- Prove prefix gap > 0 from count ratio

**Current state**: Structure is 50% complete, needs:
- `motif_frequency_low_order`: Apply EML to show short motifs appear with near-product frequency
- `motif_frequency_high_order`: Show long motifs decay exponentially
- `thinning_slope_positive`: Linear regression on log K vs order gives slope ≥ γ(ε,Δ)
- `prefix_gap_positive`: Count ratio gives gap ≥ ρ(ε,Δ)

**Standard theory**: Expander Mixing Lemma + standard probability bounds.

### Priority 4: Supporting Lemmas

#### 8. `robustness_preserves_E4` connections
**File**: `proofs/lean/p_vs_np_proof.lean:183-228`

**What's needed**: Connect helper lemmas to main theorem.

**Current state**: Structure is 80% complete, needs:
- Instantiate `lipschitz_slope_sum` with cover structure
- Extract gap from `E4Margin`
- Connect perturbation to score change via Lipschitz

#### 9. `mwu_step_improvement` connections
**File**: `proofs/lean/mwu_potential.lean:50-102`

**What's needed**: Connect MWU regret to expectation bound.

**Current state**: Structure is 75% complete, needs:
- Extract normalized weights from MWU update
- Connect C1, C2 to expectation bound
- Complete calculation chain

#### 10. `mwu_poly_convergence` connections
**File**: `proofs/lean/mwu_potential.lean:118-185`

**What's needed**: Connect submartingale to epoch analysis.

**Current state**: Structure is 65% complete, needs:
- Formalize bounded differences from MWU update
- Connect to `mwu_step_improvement`
- Formalize epoch analysis (decreases in #unsat)

## Complete List of All `sorry` Placeholders

Run this to get the exact list:
```bash
python3 tools/lean_no_sorry_check.py proofs/lean
```

Current count: **43 `sorry` placeholders** across:
- `p_vs_np_proof.lean`: ~15
- `mwu_potential.lean`: ~10
- `restricted_class.lean`: ~14
- `complexity_accounting.lean`: ~4

## What Each Proof Needs

### For Each `sorry`:

1. **Exact location**: File:line number
2. **Context**: What's being proved (already documented in comments)
3. **Strategy**: How to prove it (already documented)
4. **Standard theory**: What to use (MWU, Azuma, EML, etc.)
5. **Constants**: All explicit (no hidden knobs)

### Example: `lipschitz_slope_sum`

**Location**: `p_vs_np_proof.lean:100-128`

**What to fill**: The `sorry` placeholders in the `calc` chain:
- Line 113: Apply sum linearity
- Line 116: Apply triangle inequality  
- Line 119: Use `hw_nonneg`
- Lines 120-125: Apply Lipschitz property (partially done)
- Line 128: Factor out (done)

**Expected result**: Complete `calc` chain with no `sorry`.

## Deliverables from Formalism AI

For each `sorry`, provide:

1. **Complete Lean proof** (no `sorry`)
2. **Proof strategy** (if different from documented)
3. **Dependencies** (what needs to be imported from mathlib)

### Format

```lean
-- Replace this:
sorry  -- TODO: Apply sum linearity

-- With this:
-- Apply sum linearity: |∑ w_b (ΔK_b θ - ΔK_b θ')| = |∑ w_b * (ΔK_b θ - ΔK_b θ')|
rw [Finset.sum_sub_distrib]
simp [mul_sub]
```

## Testing

After filling proofs:

1. **Check syntax**: `lean --check proofs/lean/*.lean`
2. **Run formal gate**: `python3 tools/lean_no_sorry_check.py proofs/lean`
3. **Should pass**: Exit code 0, no `sorry` found

## Priority Recommendation

**Start with Priority 1** (core helper lemmas):
1. `lipschitz_slope_sum` - Foundation for robustness
2. `prefix_stability_gap` - Foundation for robustness
3. `mwu_regret_bound` - Foundation for MWU
4. `azuma_hoeffding_bounded` - Foundation for convergence

These 4 lemmas unlock the main theorems. Once they're done, the rest is mostly connecting them together.

## Questions for Formalism AI

If stuck on any `sorry`, the formalism AI should:
1. Check the comments above the `sorry` (proof strategy is documented)
2. Check the helper lemmas already defined
3. Check standard theory references (MWU, Azuma, EML)
4. Ask for clarification on specific file:line if needed

## Success Criteria

**All proofs complete when**:
- `python3 tools/lean_no_sorry_check.py proofs/lean` returns exit code 0
- `lean --check proofs/lean/*.lean` compiles without errors
- All 43 `sorry` placeholders are filled

**Then**: Formal gate passes → Can proceed to empirical gates → Full CI can run → Status = PROVED (restricted)

