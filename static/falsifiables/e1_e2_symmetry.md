# Falsifiable Protocol — E1 & E2 Symmetry

E1 (eligibility) and E2 (symmetry/flow) validate that LOW decomposition preserves the conjugate structure required for each program’s Δ locks.

## Assets
- `CLEAN_SUBMISSION/data/manifolds/manifold_catalog.json`  
- `CLEAN_SUBMISSION/data/yang_mills/gauge_frames.npy`  
- `CLEAN_SUBMISSION/data/navier_stokes/triad_catalog.csv`  
- `CLEAN_SUBMISSION/code/RUN_ALL_EXTENSIVE_VALIDATION.py`

## Procedure
1. **Run the automated sweep**  
   ```bash
   python CLEAN_SUBMISSION/code/RUN_ALL_EXTENSIVE_VALIDATION.py --stage e1e2 --output reports/e1e2_summary.json
   ```
   The script replays 600 trials (100 per problem) using the same seeds reported in `extensive_validation_report.json`.
2. **Analyse symmetry deviation**  
   For each trial compute the conjugate symmetry error `ε_sym` and ensure:
   - Riemann: `ε_sym ≤ 1.0e-5`  
   - Navier–Stokes: `ε_sym ≤ 3.0e-4` for all triads  
   - Yang–Mills & Hodge: `ε_sym ≤ 2.0e-4`
3. **Flow consistency**  
   Use `statistical_rigor.py --compute-flow reports/e1e2_summary.json` to extract LOW invariants (`L0`, `L1`, `L2`). Deviation must remain under `2%` compared with the production manifests.
4. **Archive results**  
   Append each trial hash and verdict to the `problem_results[*].symmetry_checks` section inside `extensive_validation_report.json`.

## Acceptance Criteria
- 100% of trials remain within symmetry tolerance.  
- No persistent drift beyond 40 consecutive samples.  
- Aggregate report signed and stored under `replication/symmetry/`.

Any failure requires flagging the trial as a Δ-barrier and logging corrective steps before proceeding to E3.
