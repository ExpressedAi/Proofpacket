# CI Gates: Formal + Empirical Enforcement

## Two Independent Gates

Claims are enforced by two independent gates:

1. **Formal Gate**: Lean no-sorry check - forbids any `sorry` or unauthorized axioms
2. **Empirical Gates**: R/M/C/E with strict schemas and explicit numeric thresholds (no label peeking)

The CI prints failure artifacts and never flips `PROOF_STATUS.json` unless both gates pass.

**To reject**: Provide either a Lean counterexample (file:line) or an empirical failing seed/log violating a stated threshold.

## Running CI

### Full CI (Recommended)

```bash
cd P_VS_NP_SUBMISSION_CLEAN
python3 tools/run_full_ci.py
```

This runs:
1. Formal gate (Lean no-sorry check)
2. Empirical gates (R/M/C/E)
3. Schema validation (if results exist)
4. Updates PROOF_STATUS.json

### Individual Gates

**Formal Gate**:
```bash
python3 tools/lean_no_sorry_check.py proofs/lean
# Exit code 0 = pass, nonzero = fail with JSON report
```

**Empirical Gates**:
```bash
cd code
python3 run_ci_gates.py
```

**Schema Validation**:
```bash
python3 tools/validate_results_jsonl.py RESULTS/adversarial_manifest.jsonl
```

## Configuration

Thresholds are defined in `ci_gates_config.yaml`:
- `tolerance.slope_epsilon`: Sign check robustness
- `tolerance.prefix_required_rate`: Must preserve prefix for all perturbed seeds (1.0 = 100%)
- `tolerance.success_required_rate`: ≥ 2/3 for MWU convergence
- `gate_E.gamma_tau`: Numeric slack for slope lower bound (1e-3)
- `gate_E.rho_tau`: Numeric slack for prefix gap (1e-3)

## Artifacts

On failure, artifacts are saved to:
- `RESULTS/ci_artifacts/gate_*_failure.json` - Failure details
- Console output - JSON reports with file:line for formal gate

## Status Updates

`PROOF_STATUS.json` is updated only if:
- ✅ Formal gate passes (no `sorry`, no unauthorized axioms)
- ✅ All empirical gates pass (R, M, C, E)
- ✅ Schema validation passes (if results exist)

Then: `ci.restricted_class_proved = true`

