# Reproduction Seeds

## Fixed Seed List

For reproducibility, all tests use these fixed seeds:

```python
REPRO_SEEDS = [
    42, 123, 456, 789, 1011,
    2022, 3033, 4044, 5055, 6066,
    7077, 8088, 9099, 10101, 11111,
    12121, 13131, 14141, 15151, 16161,
    17171, 18181, 19191, 20202, 21212
]
```

## Code Hashes

### Main Test File
- File: `code/p_vs_np_test.py`
- Hash: `sha256:...` (to be computed)
- Last modified: (timestamp)

### Baselines
- File: `code/baselines.py`
- Hash: `sha256:...` (to be computed)
- Last modified: (timestamp)

### Lean Proofs
- File: `proofs/lean/p_vs_np_proof.lean`
- Hash: `sha256:...` (to be computed)
- Last modified: (timestamp)

### LaTeX Proofs
- File: `proofs/tex/P_vs_NP_theorem.tex`
- Hash: `sha256:...` (to be computed)
- Last modified: (timestamp)

## Parameter Hashes

All fixed parameters:

```python
PARAMETERS = {
    "harmony_eta": 0.1,
    "harmony_lambda_bridge": 1.0,
    "harmony_max_iterations": 2000,
    "bridge_max_order": 6,
    "e4_threshold_slope": 0.0,
    "e4_threshold_prefix_ratio": 1.0,
    "test_sizes": [10, 20, 50, 100, 200],
    "n_trials_per_size": 20,
    "adversarial_n_trials": 10
}
```

Hash: `sha256:...` (to be computed)

## Environment

- Python: 3.9+
- Dependencies: See `code/requirements.txt`
- Lean: 4.0+
- OS: (to be specified)

## Reproduction Command

```bash
cd P_VS_NP_SUBMISSION_CLEAN
python code/p_vs_np_test.py
```

This will:
1. Use fixed seeds from `REPRO_SEEDS`
2. Run all test families
3. Generate `RESULTS/*.jsonl` files
4. Generate `RESULTS/CONFUSION_MATRIX.csv`
5. Generate `RESULTS/LEADERBOARD.md`
6. Generate `RESULTS/ABLATIONS.json`

