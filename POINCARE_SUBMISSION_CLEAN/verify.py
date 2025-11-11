#!/usr/bin/env python3
"""
Quick verification script - checks all required files exist
"""

import os
from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "MANIFEST.json",
    "CHECKLIST.md",
    "FRAMEWORK_LEGITIMACY.md",
    "Ricci_Flow_Bridge_Audit.md",
    "proofs/tex/Poincare_theorem.tex",
    "proofs/tex/Ricci_Bridge_Analysis.tex",
    "proofs/lean/poincare_proof.lean",
    "code/poincare_conjecture_test.py",
    "code/requirements.txt",
    "data/poincare_delta_receipts.jsonl",
    "results/poincare_conjecture_production_results.json",
    "tests/run_tests.py"
]

def verify_structure():
    """Verify all required files exist"""
    base = Path(__file__).parent
    missing = []
    present = []
    
    for file_path in REQUIRED_FILES:
        full_path = base / file_path
        if full_path.exists():
            present.append(file_path)
        else:
            missing.append(file_path)
    
    print("="*80)
    print("POINCARÉ SUBMISSION FILE VERIFICATION")
    print("="*80)
    
    if missing:
        print(f"\n❌ Missing files ({len(missing)}):")
        for f in missing:
            print(f"  - {f}")
    
    if present:
        print(f"\n✅ Present files ({len(present)}):")
        for f in present:
            print(f"  + {f}")
    
    print(f"\n{'='*80}")
    if missing:
        print("❌ VERIFICATION FAILED - Some files are missing")
        return False
    else:
        print("✅ VERIFICATION PASSED - All files present")
        print("\n📦 Package includes:")
        print("  • Formal proof (LaTeX + Lean)")
        print("  • Bridge analysis to Perelman's Ricci flow")
        print("  • Framework legitimacy documentation")
        print("  • Complete test suite")
        print("  • Production results")
        return True

if __name__ == "__main__":
    success = verify_structure()
    exit(0 if success else 1)

