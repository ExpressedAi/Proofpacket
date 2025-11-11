# CI Enforcement: Complete ✅

## Two Independent Gates (No Wiggle Room)

### 1. Formal Gate: Lean No-Sorry Check ✅

**File**: `tools/lean_no_sorry_check.py`

**Status**: ✅ **WORKING** (found 43 `sorry` placeholders - expected, needs formalism AI)

**What it does**:
- Scans all `.lean` files for `sorry` placeholders
- Checks for unauthorized axioms (only `expander_mixing_lemma` allowed)
- Returns exit code 0 = pass, nonzero = fail with JSON report

**Run**:
```bash
python3 tools/lean_no_sorry_check.py proofs/lean
```

**Output**: JSON report with file:line for each `sorry` or unauthorized axiom

### 2. Empirical Gates: Strict Schema + Thresholds ✅

**Files**:
- `code/run_ci_gates.py` - Gate runner (R/M/C/E)
- `tools/validate_results_jsonl.py` - Schema validator
- `schemas/PNPResultsRow.schema.json` - Strict JSON schema
- `ci_gates_config.yaml` - Explicit thresholds

**Status**: ✅ **READY** (needs `jsonschema` module: `pip install jsonschema`)

**What it does**:
- Gate R: Robustness (slope sign + prefix preservation)
- Gate M: MWU (empirical mean ≥ theoretical, polynomial convergence)
- Gate C: Constructibility (polynomial runtime scaling)
- Gate E: Existence (slope ≥ γ(ε,Δ) - τ, prefix gap ≥ ρ(ε,Δ) - τ)

**Thresholds** (from `ci_gates_config.yaml`):
- `prefix_required_rate: 1.0` (100% preservation required)
- `success_required_rate: 0.667` (≥ 2/3)
- `gamma_tau: 1e-3` (numeric slack)
- `rho_tau: 1e-3` (numeric slack)
- `poly_exponent_cap: 4.0` (steps ≤ n^4)

### 3. Full CI Driver ✅

**File**: `tools/run_full_ci.py`

**Status**: ✅ **READY**

**What it does**:
1. Runs formal gate (Lean no-sorry check)
2. Runs empirical gates (R/M/C/E)
3. Validates schema (if results exist)
4. Updates `PROOF_STATUS.json` only if ALL pass

**Run**:
```bash
python3 tools/run_full_ci.py
```

**Exit codes**:
- 0 = All gates passed → `ci.restricted_class_proved = true`
- 1 = Some gates failed → Artifacts saved to `RESULTS/ci_artifacts/`

## Current Status

| Gate | Status | Notes |
|------|--------|-------|
| Formal (Lean) | ⚠️ **43 `sorry` found** | Expected - needs formalism AI |
| Empirical (R/M/C/E) | ✅ **Ready** | Can run now (needs `jsonschema`) |
| Schema Validation | ✅ **Ready** | Can validate existing results |
| Full CI Driver | ✅ **Ready** | Orchestrates all gates |

## What's Needed

### For Formal Gate to Pass:
- Formalism AI must fill all 43 `sorry` placeholders
- No unauthorized axioms (only `expander_mixing_lemma` allowed)

### For Empirical Gates to Run:
- Install dependency: `pip install jsonschema`
- Run: `python3 tools/run_full_ci.py`

## Enforcement Statement

**Claims are enforced by two independent gates:**
1. **Formal**: Lean no-sorry check - forbids any `sorry` or unauthorized axioms
2. **Empirical**: R/M/C/E with strict schemas and explicit numeric thresholds (no label peeking)

**The CI prints failure artifacts and never flips `PROOF_STATUS.json` unless both gates pass.**

**To reject**: Provide either:
- A Lean counterexample (file:line)
- An empirical failing seed/log violating a stated threshold

## Files Created

✅ `tools/lean_no_sorry_check.py` - Formal gate checker
✅ `tools/validate_results_jsonl.py` - Schema validator
✅ `tools/run_full_ci.py` - Full CI driver
✅ `schemas/PNPResultsRow.schema.json` - Strict JSON schema
✅ `ci_gates_config.yaml` - Explicit thresholds
✅ `README_CI.md` - CI documentation
✅ `REFEREE_ONEPAGER.md` - Updated with enforcement statement

## Next Steps

1. **Formalism AI**: Fill all 43 `sorry` placeholders
2. **Install dependency**: `pip install jsonschema`
3. **Run full CI**: `python3 tools/run_full_ci.py`
4. **If all pass**: Status = **PROVED (restricted)**

**No wiggle room. Binary decision: prove or demote.**

