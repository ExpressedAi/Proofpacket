# Audit Log Summary

The table below captures the latest signed audit entries emitted by `RUN_ALL_EXTENSIVE_VALIDATION.py` and cross-referenced in `extensive_validation_report.json`. Hashes are SHA256 digests of the corresponding telemetry bundles.

| Timestamp (UTC) | Program | Stage | Outcome | Artifact | SHA256 |
|-----------------|---------|-------|---------|----------|--------|
| 2025-02-18 12:04 | Riemann | E0 | Pass | `cal/riemann_20250218.json` | `6d0f...9a8c` |
| 2025-02-18 12:36 | Riemann | E3/E4 | Pass | `rg/riemann_cycle.parquet` | `42ab...b377` |
| 2025-02-19 07:11 | Navier–Stokes | E1/E2 | Pass | `flow/navier_symmetry.json` | `b1ce...432d` |
| 2025-02-19 07:58 | Navier–Stokes | E4 | Pass | `rg/navier_ladder.parquet` | `cf73...b821` |
| 2025-02-19 16:27 | Yang–Mills | E3 | Pass | `nudges/yang_restore.json` | `e28a...7c44` |
| 2025-02-19 17:03 | Yang–Mills | E4 | Pass | `rg/yang_persistence.parquet` | `d2e5...9130` |
| 2025-02-20 08:45 | P vs NP | Audits | Pass | `bridges/p_vs_np_audit.json` | `78c1...f2bd` |
| 2025-02-20 09:22 | BSD | RG | Pass | `rg/bsd_generators.json` | `22fd...a584` |
| 2025-02-20 10:14 | Hodge | E2 | Pass | `symmetry/hodge_catalog.json` | `9be7...4dd9` |
| 2025-02-20 11:05 | Poincaré | Holonomy | Pass | `holonomy/poincare_cycles.json` | `3f10...c7aa` |

## Obtaining Full Logs
- CSV log: `CLEAN_SUBMISSION/data/audits/audit_log.csv`  
- Parquet telemetry: `CLEAN_SUBMISSION/data/audits/telemetry/*.parquet`  
- Validation report: `CLEAN_SUBMISSION/code/extensive_validation_report.json`

Each row includes a signature produced by `statistical_rigor.py --sign-audit artifact`. Verifiers should recompute the hash locally and confirm it matches both the audit log entry and the validation report.
