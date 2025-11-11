# Δ-Primitives Replication Bundle

- `environment.yml` — Conda environment definition for the Δ toolkit (Python 3.11 + Plotly + JAX).  
- `notebooks/riemann_phase_lock.ipynb` — Reproduces the critical-line persistence statistics, including micro-nudge replay.  
- `notebooks/navier_stokes_smoothness.ipynb` — Executes the CFD ladder with LOW damping and audits.  
- `notebooks/yang_mills_mass_gap.ipynb` — Lattice resonance analysis with mass-gap extraction.  
- `notebooks/p_vs_np_bridge_covers.ipynb` — Recreates the low-order bridge inventory and Δ-barrier detection.  
- `scripts/run_validation.sh` — Thin wrapper around `RUN_ALL_EXTENSIVE_VALIDATION.py` for automated sweeps.  
- `data/` — Reference spectra, manifolds, RG seeds, calibration assets, and telemetry samples (see manifest below).

## Usage
1. **Create the environment**  
   ```bash
   mamba env create -f environment.yml
   mamba activate delta-primitives
   ```
2. **Pull the latest artifacts**  
   Download `/api/proofs` and place the JSON payload under `data/api_snapshot.json`. Extract additional assets referenced in the manifest.
3. **Run the extensive validation suite**  
   ```bash
   ./scripts/run_validation.sh --all
   ```
   The script emits per-program summaries to `reports/` and signs each artifact.
4. **Launch the notebooks**  
   `jupyter lab` and open the notebooks in order. Each notebook references the validation outputs and reproduces the plots visible on the dashboard.
5. **Submit reports**  
   After completion, run `statistical_rigor.py --sign-bundle reports/` and append the resulting hash and timestamp to `CLEAN_SUBMISSION/code/extensive_validation_report.json` under your validator ID.

## Manifest

| Path | Description |
|------|-------------|
| `data/riemann/zero_catalog.parquet` | Prime-zero dataset used in production and validation. |
| `data/navier_stokes/triads/` | Triad energy telemetry and RG traces. |
| `data/yang_mills/lattice_frames/` | Gauge configurations and channel spectra. |
| `data/p_vs_np/bridge_samples/` | Low-order bridge inventories and Δ-barrier examples. |
| `data/bsd/generator_sets/` | Elliptic curve generator inventories for RG persistence. |
| `data/hodge/lock_catalog/` | (p,p) lock listings with algebraicity status. |

Every replication should leave the bundle in a signed state with SHA256 hashes recorded in the audit log.
