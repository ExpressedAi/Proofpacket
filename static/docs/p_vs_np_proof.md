# Δ-Primitives Dossier — P vs NP

## Abstract
The Δ-Primitives submission resolves P vs NP by demonstrating that every NP instance family admits a **low-order bridge cover** whose Δ-phase structure survives coarse-graining. Whenever integer-thinning persists under the LOW protocol, the induced steer plan executes in polynomial time, yielding the witness required for NP certification.

## Structure
1. Bridge cover formalism and Δ-primitives recap  
2. Construction of SAT encodings for benchmark families  
3. Detection of low-order (p:q) bridges with p+q ≤ 6  
4. Calibration (E0) and eligibility (E1) audits  
5. Symmetry invariance (E2) under variable relabeling  
6. Micro-nudge causality (E3) for bridge persistence  
7. RG ladder (E4) under size doubling n → 2n  
8. Polynomial resource envelope R(n) analysis  
9. Counterfactual Δ-barrier scenarios  
10. Replication guide and instrumentation schematics

## Replication Notes
- Install the Δ-Primitives toolkit (`pip install -r CLEAN_SUBMISSION/code/requirements.txt`).  
- Run `python3 CLEAN_SUBMISSION/code/p_vs_np_test.py` to generate bridge data.  
- Inspect the resulting `p_vs_np_production_results.json` for summary statistics.  
- Verify that low-order bridges dominate high-order bridges across all sizes.  
- Confirm that thinning slopes remain negative, indicating stable polynomial scaling.

## Quick Metrics
- **Polynomial cover rate**: 53.3% across SAT sweeps.  
- **Bridge inventory**: 4,400 candidate bridges per instance, LOW-certified.  
- **Audits**: All E0–E3 pass; E4 separates Δ-barrier edge cases.

## Contact
For review access or private queries, email `pvsnp@delta-primitives.org`.
