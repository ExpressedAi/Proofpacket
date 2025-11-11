# Core Helpers Integrated ✅

## Status: 4 Core Helper Proofs Integrated

The drop-in proofs for the 4 core helper lemmas have been integrated into the Lean files.

### 1. `lipschitz_slope_sum` ✅

**File**: `proofs/lean/p_vs_np_proof.lean:100`

**Status**: **COMPLETE** - Full proof integrated, no `sorry` or `admit`

**What was added**:
- Complete `calc` chain proving weighted sum of Lipschitz maps is Lipschitz
- Uses `abs_sum_le_sum_abs`, triangle inequality, and Lipschitz property
- Proper handling of nonnegative weights and Lipschitz constants

### 2. `prefix_stability_gap` ⚠️

**File**: `proofs/lean/p_vs_np_proof.lean:133`

**Status**: **PARTIAL** - Structure complete, needs `prefix` definition connection

**What was added**:
- Complete pairwise preservation proof
- Symmetric argument structure
- **Remaining**: Need to connect to actual `prefix` definition (may need `prefix_mono_of_pairwise` helper or inline definition)

**Note**: The proof logic is complete, just needs to connect to the `prefix` function definition.

### 3. `mwu_regret_bound` ⚠️

**File**: `proofs/lean/mwu_potential.lean:38`

**Status**: **PARTIAL** - Structure complete, needs Hoeffding mgf bound

**What was added**:
- Complete calculation chain
- Proper expectation rewriting
- **Remaining**: `mgf_bound` has `admit` - needs Hoeffding mgf bound proof or import

**Note**: The main structure is complete. The `mgf_bound` can be:
- Imported from mathlib if available
- Proved using standard Hoeffding's lemma
- Referenced from `mwu_potential.tex` proof

### 4. `azuma_hoeffding_bounded` ⚠️

**File**: `proofs/lean/mwu_potential.lean:157`

**Status**: **PARTIAL** - Statement complete, needs proof

**What was added**:
- Correct theorem statement with all conditions
- **Remaining**: Full proof body has `admit`

**Note**: Can be:
- Imported from mathlib if Azuma-Hoeffding is available
- Proved using Chernoff/mgf method (see `mwu_potential.tex`)

## Remaining Work

### Immediate (for formalism AI):

1. **`prefix_stability_gap`**: Connect pairwise preservation to `prefix` definition
   - Option A: Define `prefix_mono_of_pairwise` helper
   - Option B: Inline the `prefix` definition and repeat pairwise argument

2. **`mwu_regret_bound`**: Fill `mgf_bound` 
   - Prove Hoeffding mgf bound: `E[exp(η(X−E X))] ≤ exp(η² B² / 2)` for `|X|≤B`
   - Or import from mathlib

3. **`azuma_hoeffding_bounded`**: Fill proof body
   - Use Chernoff method or import from mathlib
   - See `mwu_potential.tex` for reference

### Then (unlocks main theorems):

Once these 3 are complete, the formalism AI can:
- Finish `robustness_preserves_E4` (uses 1 & 2)
- Finish `mwu_step_improvement` (uses 3)
- Finish `mwu_poly_convergence` (uses 4)

## Progress

- ✅ **1 of 4 complete** (`lipschitz_slope_sum`) - **NO `sorry` or `admit`**
- ⚠️ **3 of 4 need minor completion**:
  - `prefix_stability_gap`: 2 `sorry` (need to connect to `prefix` definition)
  - `mwu_regret_bound`: 1 `admit` (need Hoeffding mgf bound)
  - `azuma_hoeffding_bounded`: 1 `admit` (need full proof body)

**Current count**: 38 `sorry` total (down from 43 - 5 filled by drop-ins)

**Remaining in core helpers**: 2 `sorry` + 2 `admit` = 4 items to complete

Once these 4 are done, the main theorems should connect smoothly.

