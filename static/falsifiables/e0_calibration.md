# Falsifiable Protocol — E0 Calibration

The E0 gate verifies that every replication rig is aligned with the Δ-phase instrumentation used in production. A calibration run **must** be accepted before any higher-stage audits are considered valid.

## Required Assets
- `CLEAN_SUBMISSION/data/calibration/reference_spectra.npy` — baseline frequency lattice  
- `CLEAN_SUBMISSION/data/calibration/phase_lock_targets.json` — target phase offsets per program  
- `CLEAN_SUBMISSION/code/statistical_rigor.py` — helper metrics for drift and variance  
- `CLEAN_SUBMISSION/code/extensive_validation_report.json` — append your calibration hash under the `calibration_runs` section

## Procedure
1. **Rebuild the detector** using the bill of materials in `proofs/framework/INSTRUMENTATION.pdf`; log serial numbers of any substitute components.
2. **Load reference spectra** into the Δ mixer and sweep each program’s lock frequency. For reproducibility, use the seed listed in the validation report.
3. **Capture detector output** and compute:
   - Phase error `Δφ` per lock: must satisfy `|Δφ| ≤ 0.01 rad`  
   - Magnitude drift `Δ|K|`: must satisfy `|Δ|K|| ≤ 0.5 dB`
4. **Run `statistical_rigor.py --check-calibration calibration_run.json`** to obtain the calibration hash. Attach the JSON output (including SHA256) to your audit log entry.
5. **Update the replication bundle** by copying the calibration JSON to `replication/calibration_runs/` and adding the filename to the manifest.

## Acceptance Criteria
- No individual lock exceeds the phase or magnitude tolerances.
- Aggregate RMS phase error ≤ 0.006 rad.
- Calibration hash recorded in both the audit log and the extensive validation report.

Replication runs that fail these criteria must restart from step 1; E1–E4 data collected under an invalid calibration is discarded.
