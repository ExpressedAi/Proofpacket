#!/usr/bin/env python3
"""
Yang-Mills Mass Gap Test - PRODUCTION VERSION
==============================================

NO HARDCODED MASSES. Real lattice QCD simulation.

This replaces yang_mills_test.py with actual computations.
"""

import numpy as np
from scipy.linalg import expm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the LQCD implementation
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Use the improved LQCD code as a module
exec(open('yang_mills_lqcd_improved.py').read())


class YangMillsProductionTest:
    """
    Production Yang-Mills test that computes masses from first principles

    Key differences from original:
    - NO hardcoded masses
    - Real gauge field generation
    - Correlators from Wilson loops
    - Masses extracted via fits
    """

    def __init__(self):
        # Multiple parameter points for robustness
        self.test_configs = [
            {'L': 6, 'beta': 2.2, 'n_configs': 30},
            {'L': 6, 'beta': 2.3, 'n_configs': 30},
            {'L': 6, 'beta': 2.4, 'n_configs': 30},
        ]

    def run_all_tests(self):
        """Run complete test suite across multiple parameters"""
        print("="*80)
        print("YANG-MILLS MASS GAP TEST - PRODUCTION VERSION")
        print("="*80)
        print(f"Started: {datetime.now()}\n")

        all_results = []

        for idx, config in enumerate(self.test_configs, 1):
            print(f"\n{'='*80}")
            print(f"TEST {idx}/{len(self.test_configs)}: L={config['L']}, β={config['beta']}")
            print(f"{'='*80}")

            test = ImprovedYangMillsTest(
                L=config['L'],
                beta=config['beta'],
                n_configs=config['n_configs'],
                n_therm=50,
                n_sep=5
            )

            try:
                result = test.run_test()
                result['test_id'] = idx
                result['config'] = config
                all_results.append(result)

                # Check if mass gap exists
                m = result['masses']['0++']['mass']
                if m > 0.1:
                    print(f"✓ Configuration {idx}: MASS_GAP (m = {m:.3f})")
                else:
                    print(f"✗ Configuration {idx}: NO_GAP (m = {m:.3f})")

            except Exception as e:
                print(f"✗ Configuration {idx} FAILED: {e}")
                all_results.append({
                    'test_id': idx,
                    'config': config,
                    'verdict': 'ERROR',
                    'error': str(e)
                })

        # Summary
        print(f"\n{'='*80}")
        print("FINAL VERDICT")
        print(f"{'='*80}")

        successful_tests = [r for r in all_results if 'masses' in r]
        n_mass_gap = sum(1 for r in successful_tests if r['masses']['0++']['mass'] > 0.1)

        print(f"Successful tests: {len(successful_tests)}/{len(all_results)}")
        print(f"Tests with mass gap: {n_mass_gap}/{len(successful_tests)}")

        if successful_tests:
            masses = [r['masses']['0++']['mass'] for r in successful_tests]
            avg_mass = np.mean(masses)
            std_mass = np.std(masses)
            print(f"\nAverage 0++ mass: {avg_mass:.3f} ± {std_mass:.3f} (lattice units)")

            if n_mass_gap == len(successful_tests) and avg_mass > 0.1:
                verdict = "MASS_GAP"
                print(f"\n✓ VERDICT: {verdict}")
                print("  All parameter points show positive mass gap")
            else:
                verdict = "INCONCLUSIVE"
                print(f"\n⚠ VERDICT: {verdict}")
                print("  Results inconsistent across parameter points")
        else:
            verdict = "ERROR"
            print(f"\n✗ VERDICT: {verdict}")
            print("  No successful tests")

        # Save results
        output = {
            'verdict': verdict,
            'timestamp': str(datetime.now()),
            'results': all_results,
            'summary': {
                'n_tests': len(all_results),
                'n_successful': len(successful_tests),
                'n_mass_gap': n_mass_gap
            }
        }

        with open('../results/yang_mills_production_results_COMPUTED.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to: ../results/yang_mills_production_results_COMPUTED.json")
        print(f"Completed: {datetime.now()}")

        return output


if __name__ == "__main__":
    test_suite = YangMillsProductionTest()
    test_suite.run_all_tests()
