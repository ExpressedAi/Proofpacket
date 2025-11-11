#!/usr/bin/env python3
"""
Generate RESULTS files: JSONL, confusion matrices, leaderboard, ablations
"""

import json
import csv
import hashlib
from pathlib import Path
from typing import List, Dict
from p_vs_np_test import PvsNPTest, REPRO_SEEDS
from baselines import run_baselines

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def generate_jsonl(results: List[Dict], output_file: str):
    """Generate JSONL file (one line per run)"""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def generate_confusion_matrix(results: List[Dict], output_file: str):
    """Generate confusion matrix CSV"""
    # Group by family
    families = {}
    for r in results:
        family = r.get('family', 'standard')
        if family not in families:
            families[family] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        
        verdict = r.get('verdict', 'DELTA_BARRIER')
        valid = r.get('witness', {}).get('valid', False)
        
        if verdict == 'POLY_COVER' and valid:
            families[family]['tp'] += 1
        elif verdict == 'POLY_COVER' and not valid:
            families[family]['fp'] += 1
        elif verdict == 'DELTA_BARRIER' and not valid:
            families[family]['tn'] += 1
        elif verdict == 'DELTA_BARRIER' and valid:
            families[family]['fn'] += 1
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['family', 'true_positive', 'false_positive', 'true_negative', 
                        'false_negative', 'precision', 'recall', 'accuracy', 'f1_score'])
        
        for family, counts in families.items():
            tp, fp, tn, fn = counts['tp'], counts['fp'], counts['tn'], counts['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            writer.writerow([family, tp, fp, tn, fn, f"{precision:.3f}", 
                           f"{recall:.3f}", f"{accuracy:.3f}", f"{f1_score:.3f}"])

def generate_leaderboard(results: List[Dict], output_file: str):
    """Generate leaderboard markdown"""
    # Group by algorithm and n
    algorithms = ['Harmony Optimizer', 'WalkSAT', 'GSAT', 'RandomRestart']
    sizes = sorted(set(r.get('n', 0) for r in results))
    
    # Aggregate statistics
    stats = {alg: {size: {'success': 0, 'total': 0, 'time': [], 'flips': []} 
                   for size in sizes} for alg in algorithms}
    
    for r in results:
        n = r.get('n', 0)
        # Harmony Optimizer
        if r.get('witness', {}).get('valid', False):
            stats['Harmony Optimizer'][n]['success'] += 1
        stats['Harmony Optimizer'][n]['total'] += 1
        stats['Harmony Optimizer'][n]['time'].append(r.get('resources', {}).get('time', 0))
        stats['Harmony Optimizer'][n]['flips'].append(r.get('harmony_iterations', 0))
        
        # Baselines
        for baseline in ['WalkSAT', 'GSAT', 'RandomRestart']:
            baseline_data = r.get(f'baseline_{baseline.lower().replace(" ", "_")}', {})
            if baseline_data.get('success', False):
                stats[baseline][n]['success'] += 1
            stats[baseline][n]['total'] += 1
            stats[baseline][n]['time'].append(baseline_data.get('time', 0))
            stats[baseline][n]['flips'].append(baseline_data.get('flips', 0))
    
    # Write markdown
    with open(output_file, 'w') as f:
        f.write("# Algorithm Leaderboard\n\n")
        
        # Success rate table
        f.write("## Success Rate (%)\n\n")
        f.write("| Algorithm | " + " | ".join(f"n={s}" for s in sizes) + " | Overall |\n")
        f.write("|-----------|" + "|".join("------" for _ in sizes) + "|---------|\n")
        
        for alg in algorithms:
            row = [alg]
            overall_success = 0
            overall_total = 0
            for size in sizes:
                s = stats[alg][size]
                rate = 100 * s['success'] / s['total'] if s['total'] > 0 else 0
                row.append(f"{rate:.0f}%")
                overall_success += s['success']
                overall_total += s['total']
            overall_rate = 100 * overall_success / overall_total if overall_total > 0 else 0
            row.append(f"{overall_rate:.0f}%")
            f.write("| " + " | ".join(row) + " |\n")
        
        # Time table
        f.write("\n## Average Time (seconds)\n\n")
        f.write("| Algorithm | " + " | ".join(f"n={s}" for s in sizes) + " |\n")
        f.write("|-----------|" + "|".join("------" for _ in sizes) + "|\n")
        
        for alg in algorithms:
            row = [alg]
            for size in sizes:
                times = stats[alg][size]['time']
                avg_time = sum(times) / len(times) if times else 0
                row.append(f"{avg_time:.2f}")
            f.write("| " + " | ".join(row) + " |\n")

def main():
    """Generate all RESULTS files"""
    test_suite = PvsNPTest()
    results_dir = Path("RESULTS")
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    # Run production suite
    print("Running production suite...")
    sizes = [10, 20, 50, 100, 200]
    report = test_suite.run_production_suite(sizes, n_trials=20)
    all_results.extend(report['results'])
    
    # Run adversarial suites
    families = ["random_3sat_phase_transition", "planted_satisfiable", "xor_sat_gadgets"]
    for family in families:
        print(f"\nRunning adversarial suite: {family}")
        adv_report = test_suite.run_adversarial_suite(family, sizes[:3], n_trials=10)
        all_results.extend(adv_report['results'])
    
    # Generate files
    print("\nGenerating RESULTS files...")
    generate_jsonl(all_results, str(results_dir / "all_results.jsonl"))
    generate_confusion_matrix(all_results, str(results_dir / "CONFUSION_MATRIX.csv"))
    generate_leaderboard(all_results, str(results_dir / "LEADERBOARD.md"))
    
    # Compute hashes
    print("\nComputing file hashes...")
    files_to_hash = [
        "code/p_vs_np_test.py",
        "code/baselines.py",
        "proofs/lean/p_vs_np_proof.lean",
        "proofs/tex/P_vs_NP_theorem.tex"
    ]
    
    hashes = {}
    for filepath in files_to_hash:
        if Path(filepath).exists():
            hashes[filepath] = compute_file_hash(filepath)
    
    # Write hashes to REPRO_SEEDS.md
    with open("REPRO_SEEDS.md", 'a') as f:
        f.write("\n## Code Hashes (Computed)\n\n")
        for filepath, hash_val in hashes.items():
            f.write(f"- {filepath}: `sha256:{hash_val[:16]}...`\n")
    
    print("✓ All RESULTS files generated!")

if __name__ == "__main__":
    main()

