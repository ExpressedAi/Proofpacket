# Δ-Primitives Dossier — Navier–Stokes Smoothness

## Abstract
The Navier–Stokes submission characterises bounded energy and smoothness for incompressible flow in ℝ³, using Δ-Primitives to suppress supercritical triads and enforce LOW-consistent dissipation.

## Structure
1. Governing PDEs and Δ decomposition  
2. Instrument calibration and resolution window (E0)  
3. Vorticity confinement and symmetry auditing (E1-E2)  
4. Micro-nudge stability sweep (E3)  
5. Renormalisation cascade with LOW guardrails (E4)  
6. Production run catalogue (9/9 configurations)  
7. χ<sub>max</sub> upper bounds and sensitivity analysis  
8. Discussion of blow-up countermodels  
9. Replication environment (containers + notebooks)  
10. Reviewer questionnaire

## Replication Notes
- Execute the provided container images with `docker compose up`.  
- Validate χ<sub>max</sub> ≤ 8.95×10⁻⁶ for each configuration.  
- Confirm SMOOTH verdict within tolerance ε = 1.0e-5.  
- Archive solver logs and attach to audit log on submission.

## Contact
For reproducibility support, email `navier@delta-primitives.org`.
