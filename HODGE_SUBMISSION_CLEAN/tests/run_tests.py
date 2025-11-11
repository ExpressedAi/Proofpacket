#!/usr/bin/env python3
"""
Test runner for Hodge Conjecture submission
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
from hodge_conjecture_test import HodgeConjectureTest

def main():
    print("="*80)
    print("HODGE CONJECTURE SUBMISSION TEST SUITE")
    print("="*80)
    
    # Run test
    test_suite = HodgeConjectureTest()
    result = test_suite.run_production_tests()
    
    # Load production results for validation
    results_file = Path(__file__).parent.parent / "results" / "hodge_conjecture_production_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Check for (p,p) lock detection
    all_hodge_barrier = all(
        r.get('verdict') == 'HODGE_BARRIER' for r in results.get('results', [])
    )
    all_audits_pass = all(
        all(audit.get('passed', False) for audit in r.get('audits', {}).values())
        for r in results.get('results', [])
    )
    
    print(f"\n✓ Hodge barrier detection: {all_hodge_barrier}")
    print(f"✓ All audits passing: {all_audits_pass}")
    
    if all_hodge_barrier and all_audits_pass:
        print("\n✅ ALL TESTS PASSED - SUBMISSION READY")
        print(f"   • (p,p) locks detected")
        print(f"   • All {len(results.get('results', []))} configurations passed")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

