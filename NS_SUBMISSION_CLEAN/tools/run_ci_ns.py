#!/usr/bin/env python3
"""
Full CI Driver for Navier-Stokes Submission
Runs: (1) Lean no-sorry check, (2) Empirical reference check, (3) Constants validation
"""

import subprocess
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(cmd, description):
    """Run command and return (success, output)."""
    print(f"[CI] {description}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True, result.stdout
        else:
            print(f"❌ {description} FAILED")
            print(result.stderr, file=sys.stderr)
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} ERROR: {e}", file=sys.stderr)
        return False, str(e)

def main():
    all_passed = True
    
    # 1) Lean no-sorry check
    success, output = run(
        ["python3", "tools/lean_no_sorry_check.py", "proofs/lean"],
        "Lean no-sorry check"
    )
    if not success:
        all_passed = False
    
    # 2) Empirical reference check (forbid in theorems)
    success, output = run(
        ["python3", "tools/check_no_empirical_in_theorems.py", "proofs/tex"],
        "Empirical reference check"
    )
    if not success:
        all_passed = False
    
    # 3) Constants file exists and is symbolic
    constants_file = ROOT / "NS_CONSTANTS.toml"
    if constants_file.exists():
        print("✅ Constants file exists")
        # Check for numeric delta (should be symbolic)
        import re
        content = constants_file.read_text()
        if re.search(r'delta\s*=\s*-?[0-9]', content):
            print("❌ Constants file has numeric delta (should be symbolic)")
            all_passed = False
        else:
            print("✅ Constants file is symbolic")
    else:
        print("❌ Constants file missing: NS_CONSTANTS.toml")
        all_passed = False
    
    # 4) No circular axioms (shell_absorb_*)
    success, output = run(
        ["python3", "tools/check_no_circular_axioms.py", "proofs/lean"],
        "Circular axioms check"
    )
    if not success:
        all_passed = False
    
    # 5) No sorry/admit in NS Lean files (additional check)
    import subprocess
    result = subprocess.run(
        ["grep", "-nE", r'\b(sorry|admit)\b', "proofs/lean/ns_*.lean"],
        cwd=ROOT,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"❌ Found sorry/admit in NS Lean files:\n{result.stdout}")
        all_passed = False
    else:
        print("✅ No sorry/admit in NS Lean files")
    
    # 4) Summary
    if all_passed:
        print("\n✅ ALL CI CHECKS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME CI CHECKS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()

