#!/usr/bin/env python3
"""
Test runner for Yang-Mills submission
Runs tests and validates all results
"""

import sys
import os
import json
from pathlib import Path

# Add code directory to path
code_dir = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(code_dir))

# Change to code directory for imports
os.chdir(code_dir)
from yang_mills_test import YangMillsTest

def main():
    print("="*80)
    print("YANG-MILLS MASS GAP SUBMISSION TEST SUITE")
    print("="*80)
    
    # Run test
    test_suite = YangMillsTest()
    result = test_suite.run_test()
    
    # Load production results for validation
    results_file = Path(__file__).parent.parent / "results" / "yang_mills_production_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    all_mass_gap = all(r['verdict'] == 'MASS_GAP' for r in results['results'])
    all_omega_positive = all(r['omega_min'] > 0 for r in results['results'])
    all_audits_pass = all(
        all(audit[0] for audit in r['audits'].values())
        for r in results['results']
    )
    
    print(f"\n✓ All MASS_GAP verdicts: {all_mass_gap}")
    print(f"✓ All ω_min > 0: {all_omega_positive}")
    print(f"✓ All audits passing: {all_audits_pass}")
    
    if all_mass_gap and all_omega_positive and all_audits_pass:
        print("\n✅ ALL TESTS PASSED - SUBMISSION READY")
        if results.get('results'):
            print(f"   • Mass gap confirmed: ω_min = {results['results'][0]['omega_min']}")
            print(f"   • All {len(results['results'])} configurations passed")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

