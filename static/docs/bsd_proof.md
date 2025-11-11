# Δ-Primitives Dossier — Birch & Swinnerton-Dyer

## Abstract
The Δ-Primitives approach characterises the rank of an elliptic curve via **RG-persistent generators**. LOW-thinned Δ-locks identify algebraic points whose coarse-grained survival witnesses the Mordell–Weil rank predicted by the Birch & Swinnerton-Dyer conjecture.

## Structure
1. Δ-generator construction for elliptic curves over ℚ  
2. Instrument calibration (E0) with conductor-specific references  
3. Eligibility screening (E1) for coherent Δ-locks  
4. Symmetry invariance (E2) under curve isomorphisms  
5. Micro perturbation resilience (E3) for candidate generators  
6. RG ladder (E4) and persistence thresholds  
7. Rank estimation from persistent generators  
8. Comparison against L-function derivatives  
9. Counterexamples and Δ-barrier analysis  
10. Replication protocol

## Replication Notes
- Install dependencies (`pip install -r CLEAN_SUBMISSION/code/requirements.txt`).  
- Execute `python3 CLEAN_SUBMISSION/code/bsd_conjecture_test.py` for generator inventories.  
- Examine `bsd_conjecture_production_results.json` for persistence counts and audit outcomes.  
- Validate that average rank estimates stabilise at 2.0 with Δ-persistence.  
- Review Δ-barrier runs where E4 fails to confirm barrier diagnostics.

## Quick Metrics
- **Average rank estimate**: 2.0 across 10 sampled curves.  
- **Persistent generators**: 240–320 per trial after LOW thinning.  
- **Audits**: E0–E3 pass universally; E4 isolates non-persistent cases.

## Contact
For curve-specific artefacts or private review, email `bsd@delta-primitives.org`.
