# Handoff Note (for the Formalism AI)

## Contract

* Fill **all 43 `sorry`** in the Lean sources listed.

* Do **not** add new axioms or change statement types.

* Use only standard mathlib facts noted below.

* When done, run the **formal gate** and then the **empirical gates**.

* If any obligation cannot be met, return the **minimal counterexample** (file:line + failing hypothesis).

## Priority order (unblocks everything)

1. Core helpers (4):

   * `lipschitz_slope_sum` — p_vs_np_proof.lean:100

   * `prefix_stability_gap` — p_vs_np_proof.lean:133

   * `mwu_regret_bound` — mwu_potential.lean:38

   * `azuma_hoeffding_bounded` — mwu_potential.lean:157

2. Constructibility helpers (2):

   * `ball_size_le` — restricted_class.lean:58

   * `sum_motifs_poly` — restricted_class.lean:77

3. Existence helpers (4):

   * `motif_frequency_low_order` — restricted_class.lean:152

   * `motif_frequency_high_order` — restricted_class.lean:162

   * `thinning_slope_positive` — restricted_class.lean:173

   * `prefix_gap_positive` — restricted_class.lean:188

4. Connectors (rest): finish the five main theorems by invoking the helpers.

## Allowed mathlib tools (use these; don't reinvent)

* **Analysis / Lipschitz / Norm:**

  * `LipschitzWith.comp`, `LipschitzWith.add`, `LipschitzWith.sum`.

  * `norm_sum_le`, `abs_sum_le_sum_abs`, `norm_add_le`, `mul_le_mul_of_nonneg_left`.

* **Order / Finset / Argmin stability:**

  * `Finset.max'`, `Finset.min'`, `lt_of_le_of_lt`, `by_cases`, `split_ifs`.

* **Prob / MWU / Log-sum-exp:**

  * `Real.log`, convexity of `log` + `log_sum_exp` lemmas if available, else Hoeffding-style bound for bounded RVs.

  * Linearity of expectation; bounded differences schema.

* **Probability inequalities:**

  * Azuma–Hoeffding (if available); otherwise prove bounded-increment version using mgf/Chernoff.

* **Graph theory (expanders):**

  * If a library EML is not available, use the declared axiom stub `expander_mixing_lemma` as permitted and build from counting inequalities.

## Exact proof obligations (drop-in results)

### A. Robustness

* `lipschitz_slope_sum`

  Goal: slope as a weighted sum of Lipschitz maps is Lipschitz with constant (\sum_b w_b L_b).

  Use linearity + triangle inequality + `LipschitzWith.sum`.

* `prefix_stability_gap`

  Goal: if order gaps ≥ ρ and each score perturbation ≤ ρ/2, the low-order prefix set is unchanged.

  Use pairwise comparisons and `lt_of_le_of_lt`.

* `robustness_preserves_E4`

  Plug the two lemmas to keep slope sign and prefix under (|\delta| \le \delta^\star = \min(\gamma/(2\sum w_b L_b), \rho/(2\sum w_b L_b))).

### B. MWU

* `mwu_regret_bound`

  Prove one-step bound:

  \[
  \log\sum_i p_i e^{\eta g_i}\ \ge\ \eta\textstyle\sum_i p_i g_i - \tfrac12\eta^2B^2,\quad |g_i|\le B.
  \]

  Either invoke a mathlib lemma for log-sum-exp or derive via Hoeffding's lemma for bounded variables.

* `mwu_step_improvement`

  Combine `mwu_regret_bound` with (C1)–(C3) to get

  \[
  \mathbb{E}[\Delta\Psi]\ \ge\ \eta(\alpha+\lambda\kappa) - \tfrac12\eta^2 B^2 \ \ge\ \tfrac12\eta(\alpha+\lambda\kappa)=:\gamma_{\text{MWU}}.
  \]

* `azuma_hoeffding_bounded` + `mwu_poly_convergence`

  Define (S_t:=\Psi_t-\gamma_{\text{MWU}} t), show bounded increments, apply Azuma. Tie increments to decreases in `#unsat` to get poly hitting time with success ≥ 2/3.

### C. Constructibility (restricted)

* `ball_size_le`

  BFS tree bound: (|\mathrm{Ball}(v,L)| \le \sum_{i=0}^L \Delta^i).

* `sum_motifs_poly`

  Sum over centers; with (L=Θ(\log n)), total motifs ≤ (n·\Delta^L = n^{O(1)}).

* `build_cover_poly_time`

  Combine the above with O(1) work per motif → (n^{O(1)}).

### D. Existence (restricted)

* `existence_on_expanders` via helpers:

  * `motif_frequency_low_order` using EML for short motifs.

  * `motif_frequency_high_order` exponential thinning for long motifs.

  * `thinning_slope_positive` from frequency gap → positive slope (\hat\lambda \ge \gamma(ε,Δ)).

  * `prefix_gap_positive` → fixed low-order prefix with gap (\rho(ε,Δ)).

    Conclude `E4Persistent (build_cover F)` with explicit (\gamma,\rho>0).

## Main theorems (connect helpers)

* `robustness_preserves_E4` — use `lipschitz_slope_sum` + `prefix_stability_gap`.

* `mwu_step_improvement` — use `mwu_regret_bound` + linearity of expectation.

* `mwu_poly_convergence` — use `azuma_hoeffding_bounded` + epoch accounting.

* `build_cover_poly_time` — use `ball_size_le` + `sum_motifs_poly`.

* `existence_on_expanders` — use four existence helpers + EML.

## Formal gate (must pass)

```bash
python3 tools/lean_no_sorry_check.py proofs/lean
lean --check proofs/lean/*.lean
```

* Output must be `{"ok": true, "issues": []}` and Lean must report no errors.

## Empirical gates (then run)

```bash
python3 code/run_ci_gates.py
```

* Uses your `AUDIT_SPECS.yaml` thresholds and the JSONL manifest.

* Produces artifacts on fail; updates `PROOF_STATUS.json` only when all pass.

## Rejection protocol

* If a lemma cannot be closed, return: `{file, line, lemma_name, dependency, minimal counterexample (input instance/seed)}`

* If an empirical gate fails, emit the failing run (JSONL line) + metric deltas against thresholds.

