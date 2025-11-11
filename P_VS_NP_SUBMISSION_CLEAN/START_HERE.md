# START HERE: Complete Handoff Package

## For Formalism AI

**Read**: `HANDOFF_FORMALISM_AI.md`

**Task**: Fill all 43 `sorry` placeholders in Lean proof files

**Priority**: Start with 4 core helper lemmas (unlocks everything)

**Test**: `python3 tools/lean_no_sorry_check.py proofs/lean` → must return `{"ok": true}`

## For Referee AI

**Read**: `HANDOFF_REFEREE_AI.md`

**Summary**: Two independent gates (formal + empirical), strict schemas, explicit thresholds

**To reject**: Provide exact file:line (formal) or exact JSONL run (empirical)

## Quick Reference

| Document | Purpose |
|----------|---------|
| `HANDOFF_FORMALISM_AI.md` | Complete specifications for formalism AI |
| `HANDOFF_REFEREE_AI.md` | One-paragraph answer + enforcement details |
| `REFEREE_ONEPAGER.md` | All constants and claims |
| `ACCEPTANCE_CHECKLIST.md` | Lemma ↔ Gate ↔ Evidence matrix |
| `CI_ENFORCEMENT_COMPLETE.md` | CI gate implementation status |

## Run Full CI

```bash
cd P_VS_NP_SUBMISSION_CLEAN
python3 tools/run_full_ci.py
```

This runs:
1. Formal gate (Lean no-sorry check)
2. Empirical gates (R/M/C/E)
3. Schema validation
4. Updates `PROOF_STATUS.json` only if ALL pass

## Current Status

- ✅ **All structures wired** (proofs, gates, schemas, thresholds)
- ⚠️ **43 `sorry` placeholders** (needs formalism AI)
- ✅ **CI enforcement ready** (can run once proofs are filled)

## Success Criteria

**Formal gate passes** when:
- `python3 tools/lean_no_sorry_check.py proofs/lean` → `{"ok": true}`
- `lean --check proofs/lean/*.lean` → no errors

**Empirical gates pass** when:
- All 4 gates (R/M/C/E) return pass
- Schema validation passes
- `PROOF_STATUS.json` updated to `ci.restricted_class_proved = true`

**Final status**: **PROVED (restricted)** when both gates pass.

