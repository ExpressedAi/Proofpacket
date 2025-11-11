#!/usr/bin/env python3
"""
Test runner for Poincaré submission
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
from poincare_conjecture_test import PoincareConjectureTest

def main():
    print("="*80)
    print("POINCARÉ CONJECTURE SUBMISSION TEST SUITE")
    print("="*80)
    
    # Run tests
    test_suite = PoincareConjectureTest()
    results = test_suite.run_production_tests()
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Check for trivial holonomy detection
    all_trivial = all(
        r.get('all_m_zero', False) for r in results.get('results', [])
    )
    all_audits_pass = all(
        all(audit[0] for audit in r.get('audits', {}).values())
        for r in results.get('results', [])
    )
    
    print(f"\n✓ Trivial holonomy detected: {all_trivial}")
    print(f"✓ All audits passing: {all_audits_pass}")
    
    if all_trivial and all_audits_pass:
        print("\n✅ ALL TESTS PASSED - SUBMISSION READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

