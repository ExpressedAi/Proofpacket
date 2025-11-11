# Falsifiable Protocol — E3 & E4 RG Persistence

These stages validate micro-perturbation resilience (E3) and renormalisation-group persistence (E4) across all programs.

## Required Files
- `CLEAN_SUBMISSION/data/rg/rg_seeds.csv` — canonical seeds for each proof program  
- `CLEAN_SUBMISSION/data/rg/micro_nudges.json` — perturbation schedules  
- `CLEAN_SUBMISSION/code/navier_stokes_production.py`, `yang_mills_test.py`, `p_vs_np_test.py`, etc. — stage-specific harnesses  
- `CLEAN_SUBMISSION/code/extensive_validation_report.json` — append RG verdicts and hashes

## Execution
1. **Micro-nudge stage (E3)**  
   ```
   python RUN_ALL_EXTENSIVE_VALIDATION.py --stage e3 --output reports/e3_micro_nudges.json
   ```
   Record restoration time `t_restore` for each Δ lock; acceptable bound is `t_restore ≤ 12 iterations`.
2. **Coarse-grain stage (E4)**  
   ```
   python RUN_ALL_EXTENSIVE_VALIDATION.py --stage e4 --output reports/e4_rg_ladder.json
   ```
   Verify the LOW-preserving statistic `K_low / K_high ≥ 0.92` after the final stage.
3. **Telemetry archive**  
   Combine E3 and E4 outputs into `reports/rg_persistence.parquet` and store under `replication/rg/`.
4. **Cross-check**  
   Compare aggregate medians against `extensive_validation_report.json` (`problem_results[*].rg_persistence`). Discrepancies >1% require investigation.

## Acceptance Thresholds
- 100% of trials restore within the time bound.  
- No Δ lock drops below 0.92 baseline across eight RG stages.  
- Signed telemetry (hash + timestamp) provided in the audit log.

Failures should be tagged as Δ-barriers with remediation steps documented before the program is considered valid.
