#!/usr/bin/env python3
"""
Test runner for Navier-Stokes submission
Runs production tests and validates all results
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
from navier_stokes_production import ProductionNavierStokesTest

def main():
    print("="*80)
    print("NAVIER-STOKES SUBMISSION TEST SUITE")
    print("="*80)
    
    # Run production tests
    test_suite = ProductionNavierStokesTest()
    results = test_suite.run_test_suite()
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    all_smooth = all(r['verdict'] == 'SMOOTH' for r in results['results'])
    all_audits_pass = all(
        all(audit[0] for audit in r['audits'].values())
        for r in results['results']
    )
    zero_supercritical = all(r['n_supercritical'] == 0 for r in results['results'])
    
    print(f"\n✓ All SMOOTH verdicts: {all_smooth}")
    print(f"✓ All audits passing: {all_audits_pass}")
    print(f"✓ Zero supercritical triads: {zero_supercritical}")
    
    if all_smooth and all_audits_pass and zero_supercritical:
        print("\n✅ ALL TESTS PASSED - SUBMISSION READY")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

