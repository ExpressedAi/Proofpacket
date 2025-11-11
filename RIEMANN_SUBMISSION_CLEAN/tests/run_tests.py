#!/usr/bin/env python3
"""
Test runner for Riemann Hypothesis submission
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
from riemann_hypothesis_test_FIXED import RiemannHypothesisTest

def main():
    print("="*80)
    print("RIEMANN HYPOTHESIS SUBMISSION TEST SUITE")
    print("="*80)
    
    # Run test on first zero
    test_suite = RiemannHypothesisTest(precision=25)
    result = test_suite.test_with_proper_audits(14.134725142, window=2.0, delta_sigma=0.3)
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Check for critical line exclusivity
    on_line_K = result['locks_on'].get('1:1', {}).get('K', 0)
    off_line_K = result['locks_off'].get('1:1', {}).get('K', 0)
    e3_pass = result['audits']['E3']['passed']
    e4_pass = result['audits']['E4']['passed']
    verdict = result['verdict']
    
    print(f"\n✓ On-line K₁:₁: {on_line_K:.4f}")
    print(f"✓ Off-line K₁:₁: {off_line_K:.4f}")
    print(f"✓ E3 audit: {'PASS' if e3_pass else 'FAIL'}")
    print(f"✓ E4 audit: {'PASS' if e4_pass else 'FAIL'}")
    print(f"✓ Verdict: {verdict}")
    
    # Check thresholds
    on_line_strong = on_line_K >= 0.9
    off_line_weak = off_line_K < 0.7
    all_pass = e3_pass and e4_pass and (verdict == "SUPPORTED")
    
    if on_line_strong and off_line_weak and all_pass:
        print("\n✅ ALL TESTS PASSED - SUBMISSION READY")
        print("   • On-line lock is strong (K ≥ 0.9)")
        print("   • Off-line lock is weak (K < 0.7)")
        print("   • All audits passing")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        if not on_line_strong:
            print(f"   • On-line K too weak: {on_line_K:.4f} < 0.9")
        if not off_line_weak:
            print(f"   • Off-line K too strong: {off_line_K:.4f} >= 0.7")
        if not all_pass:
            print(f"   • Audits or verdict failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

