# RESULTS Schema

## JSONL Format (One line per run)

Each line in `RESULTS/*.jsonl` follows this schema:

```json
{
  "family": "random_3sat_phase_transition" | "planted_satisfiable" | "xor_sat_gadgets" | "goldreich_generator" | "high_treewidth_benchmarks" | "standard",
  "n": 10,
  "seed": 12345,
  "slope": 0.15,
  "prefix_pass": true,
  "steps": 234,
  "success": true,
  "time": 0.123,
  "verdict": "POLY_COVER" | "DELTA_BARRIER",
  "witness_valid": true,
  "n_bridges": 45,
  "n_eligible": 12,
  "e4_slope": 0.15,
  "e4_prefix_ratio": 1.2,
  "harmony_iterations": 234,
  "harmony_potential_increases": 156,
  "baseline_walksat": {
    "success": true,
    "flips": 567,
    "time": 0.567
  },
  "baseline_gsat": {
    "success": false,
    "flips": 10000,
    "time": 10.0
  },
  "baseline_random_restart": {
    "success": true,
    "restarts": 23,
    "flips": 2300,
    "time": 2.3
  },
  "model_selection": {
    "r_squared_poly": 0.95,
    "r_squared_exp": 0.87,
    "aic_poly": 12.3,
    "aic_exp": 15.6,
    "bic_poly": 14.1,
    "bic_exp": 17.4,
    "bayes_factor": 2.5,
    "model_preference": "POLY",
    "exponent_drift": 0.02,
    "monotone_flag": "✓ STABLE"
  }
}
```

## Confusion Matrix Schema

`RESULTS/CONFUSION_MATRIX.csv`:

```csv
family,true_positive,false_positive,true_negative,false_negative,precision,recall,accuracy,f1_score
random_3sat_phase_transition,45,5,30,20,0.90,0.69,0.75,0.78
planted_satisfiable,60,2,35,3,0.97,0.95,0.95,0.96
xor_sat_gadgets,20,10,40,30,0.67,0.40,0.60,0.50
...
```

## Leaderboard Schema

`RESULTS/LEADERBOARD.md`:

```markdown
# Algorithm Leaderboard

## Success Rate (%)

| Algorithm | n=10 | n=20 | n=50 | n=100 | n=200 | Overall |
|-----------|------|------|------|-------|-------|---------|
| Harmony Optimizer | 95% | 92% | 88% | 85% | 80% | 88% |
| WalkSAT | 90% | 85% | 75% | 65% | 55% | 74% |
| GSAT | 85% | 70% | 50% | 30% | 15% | 50% |
| Random Restart | 60% | 45% | 30% | 20% | 10% | 33% |

## Average Time (seconds)

| Algorithm | n=10 | n=20 | n=50 | n=100 | n=200 |
|-----------|------|------|------|-------|-------|
| Harmony Optimizer | 0.05 | 0.12 | 0.45 | 1.2 | 3.5 |
| WalkSAT | 0.10 | 0.25 | 1.0 | 3.5 | 12.0 |
| GSAT | 0.15 | 0.50 | 2.5 | 10.0 | 40.0 |
| Random Restart | 0.20 | 0.80 | 5.0 | 25.0 | 100.0 |

## Average Flips/Iterations

| Algorithm | n=10 | n=20 | n=50 | n=100 | n=200 |
|-----------|------|------|------|-------|-------|
| Harmony Optimizer | 150 | 300 | 800 | 2000 | 5000 |
| WalkSAT | 500 | 1500 | 5000 | 15000 | 50000 |
| GSAT | 1000 | 5000 | 20000 | 80000 | 300000 |
| Random Restart | 2000 | 8000 | 50000 | 250000 | 1000000 |
```

## Ablation Results Schema

`RESULTS/ABLATIONS.json`:

```json
{
  "clause_only_vs_clause_bridge": {
    "clause_only": {
      "success_rate": 0.65,
      "avg_steps": 5000,
      "description": "Harmony Optimizer with only clause satisfaction (no bridge terms)"
    },
    "clause_bridge": {
      "success_rate": 0.88,
      "avg_steps": 2000,
      "description": "Harmony Optimizer with clause + bridge terms (full MWU)"
    },
    "improvement": {
      "success_rate_delta": 0.23,
      "steps_reduction": 0.60
    }
  },
  "bridge_removal": {
    "remove_top_10_percent": {
      "slope": 0.12,
      "prefix_pass_rate": 0.85,
      "success_rate": 0.82
    },
    "remove_top_25_percent": {
      "slope": 0.08,
      "prefix_pass_rate": 0.70,
      "success_rate": 0.75
    },
    "remove_top_50_percent": {
      "slope": 0.03,
      "prefix_pass_rate": 0.50,
      "success_rate": 0.60
    },
    "dose_response": "As bridges removed, slope/prefix/success all decrease"
  },
  "permutation_null": {
    "original_labels": {
      "success_rate": 0.88,
      "roc_auc": 0.92
    },
    "permuted_labels": {
      "success_rate": 0.87,
      "roc_auc": 0.91
    },
    "difference": {
      "success_rate_delta": 0.01,
      "roc_auc_delta": 0.01
    },
    "interpretation": "No significant difference → no label peeking"
  }
}
```

