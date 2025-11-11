# One-Paragraph Answer for the Referee AI

The submission enforces two independent gates. The **formal gate** machine-checks that all Lean files are axiom-free (except the declared expander mixing import) and contain **no `sorry`**; the **empirical gate** runs four preregistered tests (R/M/C/E) with strict schemas and thresholds, and never flips `PROOF_STATUS.json` unless all pass. All proofs use standard mathlib tools (Lipschitz sums, log-sum-exp/HOEFFDING, Azuma with bounded increments) and explicit constants. To reject, specify the exact lemma line that fails to compile or the exact JSONL run that violates a stated threshold.

## Enforcement Details

### Formal Gate

**Command**: `python3 tools/lean_no_sorry_check.py proofs/lean`

**Checks**:
- No `sorry` placeholders
- No unauthorized axioms (only `expander_mixing_lemma` allowed)

**Output**: JSON report with file:line for each violation

**Pass criteria**: `{"ok": true, "issues": []}` + exit code 0

### Empirical Gates

**Command**: `python3 code/run_ci_gates.py`

**Gates**:
- **R (Robustness)**: Slope sign + prefix preservation under perturbations
- **M (MWU)**: Empirical mean ≥ theoretical bound, polynomial convergence
- **C (Constructibility)**: Polynomial runtime scaling
- **E (Existence)**: Slope ≥ γ(ε,Δ) - τ, prefix gap ≥ ρ(ε,Δ) - τ

**Thresholds**: Defined in `ci_gates_config.yaml` (explicit, no ambiguity)

**Schema**: Strict JSON schema in `schemas/PNPResultsRow.schema.json`

**Pass criteria**: All gates pass → `PROOF_STATUS.json` updated

### Constants (All Explicit)

See `REFEREE_ONEPAGER.md` for complete constant tables:
- MWU: η, λ, B, α, κ, γ_MWU
- Robustness: L, L_b, γ, ρ, δ★
- Constructibility: Δ, L, c_log, c_motif
- Existence: ε, Δ, γ(ε,Δ), ρ(ε,Δ), τ

### Rejection Protocol

To reject, provide:
1. **Formal**: Exact lemma line (file:line) that fails to compile
2. **Empirical**: Exact JSONL run (line number + seed) that violates a stated threshold

No vague claims. Binary decision: prove or demote.

