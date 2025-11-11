    # Quick Start: What Your Formalism AI Needs to Do

    ## TL;DR

    **Fill 43 `sorry` placeholders** in Lean proof files. Each has:
    - ✅ Exact file:line location
    - ✅ Proof strategy already documented
    - ✅ Standard theory references (MWU, Azuma, EML)
    - ✅ Explicit constants

    ## Get the Exact List

    ```bash
    cd P_VS_NP_SUBMISSION_CLEAN
    python3 tools/lean_no_sorry_check.py proofs/lean
    ```

    This outputs JSON with every `sorry` location.

    ## Priority Order (Do These First)

    ### 1. Core Helper Lemmas (4 proofs)

    These unlock everything else:

    1. **`lipschitz_slope_sum`** (`p_vs_np_proof.lean:100`)
    - Complete `calc` chain: sum of Lipschitz = Lipschitz
    - Use: triangle inequality + `LipschitzWith.sum` from mathlib

    2. **`prefix_stability_gap`** (`p_vs_np_proof.lean:133`)
    - Complete calculation: gaps ≥ ρ + perturbations ≤ ρ/2 → ordering preserved
    - Use: `Finset` operations + `lt_of_le_of_lt`

    3. **`mwu_regret_bound`** (`mwu_potential.lean:38`)
    - Prove: `log(∑ p_i exp(η g_i)) ≥ η ∑ p_i g_i - ½η²B²`
    - Use: Hoeffding's lemma or standard MWU regret

    4. **`azuma_hoeffding_bounded`** (`mwu_potential.lean:157`)
    - Prove: `Pr[S_T ≤ -a] ≤ exp(-a²/(2Tc²))` for bounded increments
    - Use: Chernoff method or import from mathlib

    ### 2. Constructibility (2 proofs)

    5. **`ball_size_le`** (`restricted_class.lean:58`)
    - Complete induction: `|Ball(v, L)| ≤ ∑_{i=0}^L Δ^i`
    - Use: BFS tree expansion

    6. **`sum_motifs_poly`** (`restricted_class.lean:77`)
    - Complete sum bound: `total_motifs ≤ n * Δ^L`
    - Use: `ball_size_le` + geometric series

    ### 3. Existence (4 proofs)

    7. **`motif_frequency_low_order`** (`restricted_class.lean:152`)
    - Apply Expander Mixing Lemma to short motifs

    8. **`motif_frequency_high_order`** (`restricted_class.lean:162`)
    - Show exponential decay for long motifs

    9. **`thinning_slope_positive`** (`restricted_class.lean:173`)
    - Linear regression on log K vs order

    10. **`prefix_gap_positive`** (`restricted_class.lean:188`)
        - Count ratio bound

    ### 4. Connections (33 remaining)

    Connect the helpers to main theorems:
    - `robustness_preserves_E4` connections
    - `mwu_step_improvement` connections
    - `mwu_poly_convergence` connections
    - `build_cover_poly_time` connections
    - `existence_on_expanders` connections

    ## What Each Proof Needs

    For each `sorry`:

    1. **Read the comment above it** - Proof strategy is documented
    2. **Check helper lemmas** - May already be defined
    3. **Use standard theory** - MWU, Azuma, EML, etc.
    4. **Fill the `sorry`** - Replace with actual proof

    ## Example

    **Before**:
    ```lean
    have h_sum : |∑ b, w_b * (ΔK_b θ - ΔK_b θ')| ≤ ∑ |w_b * (ΔK_b θ - ΔK_b θ')| := by
    sorry  -- TODO: Apply triangle inequality
    ```

    **After**:
    ```lean
    have h_sum : |∑ b, w_b * (ΔK_b θ - ΔK_b θ')| ≤ ∑ |w_b * (ΔK_b θ - ΔK_b θ')| := by
    -- Apply triangle inequality for absolute value of sum
    exact abs_sum_le_sum_abs _ _
    ```

    ## Testing

    After filling proofs:

    ```bash
    # Check for remaining sorry
    python3 tools/lean_no_sorry_check.py proofs/lean

    # Should output: {"ok": true, "issues": []}
    # Exit code: 0
    ```

    ## Success Criteria

    ✅ **All 43 `sorry` filled**
    ✅ **Formal gate passes**: `python3 tools/lean_no_sorry_check.py proofs/lean` → exit 0
    ✅ **Lean compiles**: `lean --check proofs/lean/*.lean` → no errors

    **Then**: Can run full CI → Status = PROVED (restricted)

    ## Questions?

    - Check `FORMALISM_AI_REQUIREMENTS.md` for detailed specs
    - Check comments above each `sorry` for proof strategy
    - Check `REFEREE_ONEPAGER.md` for constants and claims

