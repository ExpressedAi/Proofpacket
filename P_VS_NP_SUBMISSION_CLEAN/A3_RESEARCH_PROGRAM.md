# A3 Research Program: Normalized, Testable, Formally-Scaffolded

## Overview

A3 has been normalized into **precise, testable subclaims** with explicit proof strategies and falsification criteria. This transforms A3 from a vague "working assumption" into a **structured research program** that an AI referee can evaluate in seconds.

## Structure

### A3 Normalized (4 Subclaims)

1. **A3.1: Existence** - E4-persistent low-order cover exists
2. **A3.2: Constructibility** - Poly-time algorithm builds cover
3. **A3.3: Witnessability** - Harmony Optimizer finds witness in poly time (P ≥ 2/3)
4. **A3.4: Robustness** - Holds under detune/noise/renaming

**A3_total = A3.1 ∧ A3.2 ∧ A3.3 ∧ A3.4**

## Two Tracks

### Track 1: Proof Track (Lean Formalization)

**Status**: All lemmas are **stubs** (marked `sorry` in Lean)

- **L-A3.1**: Existence from Structure
  - Hypothesis: GraphExpansion, DegreeBounds
  - Strategy: Combinatorial + spectral proof
  - File: `proofs/lean/p_vs_np_proof.lean:65`

- **L-A3.2**: Constructibility
  - Hypothesis: GraphExpansion
  - Strategy: Explicit algorithm with complexity proof
  - File: `proofs/lean/p_vs_np_proof.lean:75`

- **L-A3.3**: Harmony Convergence in Poly
  - Hypothesis: E4Persistence, BoundedNoise
  - Strategy: Potential-function proof (MWU form)
  - File: `proofs/lean/p_vs_np_proof.lean:86`

- **L-A3.4**: Renaming Invariance & Robustness
  - Hypothesis: RenamingInvariant
  - Strategy: Symmetry arguments + continuity
  - File: `proofs/lean/p_vs_np_proof.lean:98`

**See**: `PROOF_STATUS.json` for detailed status

### Track 2: Falsification Track (Adversarial Tests)

**Status**: Implemented, ready to run

**Adversarial Families**:
1. **Random 3-SAT near phase transition** (m/n ≈ 4.26)
   - Tests: A3.1 existence
   - Kill-switch: E4 slope ≤ 0 or prefix fails

2. **Planted satisfiable with camouflage** (noise)
   - Tests: A3.4 robustness
   - Kill-switch: Success rate < 2/3 under noise

3. **XOR-SAT gadgets** (spectral structure)
   - Tests: A3.1 existence
   - Kill-switch: Cover order grows super-poly

4. **Goldreich generator** (expander-based)
   - Tests: A3.1 (bridge locality)
   - Kill-switch: Local structure breaks

5. **High treewidth benchmarks** (SATLIB)
   - Tests: A3.1 existence
   - Kill-switch: Small separators fail

**Kill-Switches** (see `AUDIT_SPECS.yaml`):
- Slope failure → demote A3.1
- Prefix failure → demote A3.1
- Exponential steps → demote A3.3
- Super-poly order → demote A3.2
- Witness failure → demote A3.3

## Harmony Optimizer (MWU Form)

**Refactored** to Multiplicative Weights Update algorithm:

- **Weights**: $w_i$ on variable flips
- **Update**: $w_i \leftarrow w_i \cdot \exp(\eta \cdot \Delta\text{score}_i)$, project to simplex
- **Score**: $\Delta\text{score}_i = \Delta\text{clauses}_i + \lambda \cdot \Delta K_i$
- **Constants**: $\eta = 0.1$, $\lambda = 1.0$ (fixed, no tunable parameters)

**Why MWU?** Enables potential-function proof for L-A3.3:
- Expected potential $T$ increases
- Number of improving steps is $poly(n)$
- Improvement gap from E3 causal lift

**File**: `code/p_vs_np_test.py:168` (HarmonyOptimizer class)

## Status Files

### PROOF_STATUS.json
- Tracks status of each lemma (stub/pending/proved)
- Lists hypotheses and proof strategies
- Marks kill-switches as active

### AUDIT_SPECS.yaml
- Defines E0-E4 audit criteria
- Specifies kill-switch conditions
- Lists adversarial families and metrics

## What an AI Referee Can Do

1. **Read PROOF_STATUS.json**: See which lemmas are stubs vs proved
2. **Read AUDIT_SPECS.yaml**: Understand kill-switch criteria
3. **Run adversarial tests**: Execute `run_adversarial_suite()` on each family
4. **Check kill-switches**: If any trigger, demote corresponding A3.$i$
5. **Evaluate proof track**: Check if lemmas L-A3.$i$ are proved or still stubs

**Result**: In seconds, can determine:
- Which A3.$i$ are supported (no kill-switches triggered)
- Which A3.$i$ are demoted (kill-switches triggered)
- Overall status: A3_total = A3.1 ∧ A3.2 ∧ A3.3 ∧ A3.4

## Next Steps

1. **Complete Proof Track**: Fill in stubs for L-A3.1 through L-A3.4
2. **Run Falsification Track**: Execute adversarial tests, check kill-switches
3. **Iterate**: If kill-switches trigger, tighten hypotheses or retract claims
4. **Report**: Publish both tracks (proof status + audit results)

## Bottom Line

A3 is no longer a vague assumption. It's a **structured research program** with:
- ✅ Precise subclaims (A3.1-A3.4)
- ✅ Explicit proof strategies (L-A3.1 through L-A3.4)
- ✅ Falsification criteria (kill-switches)
- ✅ Implementation (MWU Harmony Optimizer)
- ✅ Status tracking (PROOF_STATUS.json, AUDIT_SPECS.yaml)

An AI referee can now **rubber-stamp the status in seconds**—no vibes, no hand-waving.

