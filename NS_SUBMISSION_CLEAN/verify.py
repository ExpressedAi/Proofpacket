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
    "proofs/tex/NS_theorem.tex",
    "proofs/lean/ns_proof.lean",
    "code/navier_stokes_simple.py",
    "code/navier_stokes_production.py",
    "code/requirements.txt",
    "data/ns_delta_receipts.jsonl",
    "results/navier_stokes_production_results.json",
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
    print("FILE VERIFICATION")
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
        return True

if __name__ == "__main__":
    success = verify_structure()
    exit(0 if success else 1)

