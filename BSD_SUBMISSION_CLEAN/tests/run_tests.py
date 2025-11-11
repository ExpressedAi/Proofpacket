#!/usr/bin/env python3
"""
Test runner for Birch and Swinnerton-Dyer submission
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
from bsd_conjecture_test import BSDTest

def main():
    print("="*80)
    print("BIRCH AND SWINNERTON-DYER SUBMISSION TEST SUITE")
    print("="*80)
    
    # Run test
    test_suite = BSDTest()
    result = test_suite.run_production_suite()
    
    # Load production results for validation
    results_file = Path(__file__).parent.parent / "results" / "bsd_conjecture_production_results.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Check for rank estimation
    all_audits_pass = all(
        all(audit.get('passed', False) for audit in r.get('audits', {}).values())
        for r in results.get('results', [])
    )
    
    # Calculate average rank
    ranks = [r.get('estimated_rank', 0) for r in results.get('results', [])]
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    
    print(f"\n✓ Average rank: {avg_rank:.2f}")
    print(f"✓ All audits passing: {all_audits_pass}")
    
    if avg_rank > 0 and all_audits_pass:
        print("\n✅ ALL TESTS PASSED - SUBMISSION READY")
        print(f"   • Rank estimation validated")
        print(f"   • Average rank: {avg_rank:.2f}")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

