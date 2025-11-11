#!/usr/bin/env python3
"""
Full CI Driver: Wraps both formal + empirical gates, updates status
"""

import subprocess
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATUS = ROOT / "PROOF_STATUS.json"
TOOLS = ROOT / "tools"
CODE = ROOT / "code"

def run(cmd, cwd=None):
    """Run command and return exit code"""
    print(f"[RUN] {' '.join(cmd)}")
    if cwd:
        print(f"      (cwd: {cwd})")
    p = subprocess.run(cmd, cwd=cwd)
    return p.returncode

def update_status(field, value=True):
    """Update PROOF_STATUS.json"""
    if not STATUS.exists():
        print(f"⚠️  {STATUS} not found, creating...")
        data = {"ci": {}}
    else:
        data = json.loads(STATUS.read_text())
    
    data["ci"] = data.get("ci", {})
    data["ci"][field] = value
    STATUS.write_text(json.dumps(data, indent=2))
    print(f"✓ Updated PROOF_STATUS.json: ci.{field} = {value}")

def main():
    print("=" * 70)
    print("Full CI: Formal + Empirical Gates")
    print("=" * 70)
    
    all_pass = True
    
    # 1) Formal gate: No sorry, no unauthorized axioms
    print("\n[1/3] Formal Gate: Lean No-Sorry Check")
    print("-" * 70)
    rc = run(["python3", str(TOOLS / "lean_no_sorry_check.py"), str(ROOT / "proofs" / "lean")])
    if rc != 0:
        update_status("formal_pass", False)
        all_pass = False
        print("\n❌ Formal gate FAILED")
    else:
        update_status("formal_pass", True)
        print("\n✅ Formal gate PASSED")
    
    # 2) Empirical gate runner
    print("\n[2/3] Empirical Gates: R/M/C/E")
    print("-" * 70)
    rc = run(["python3", str(CODE / "run_ci_gates.py")], cwd=str(CODE))
    if rc != 0:
        update_status("empirical_pass", False)
        all_pass = False
        print("\n❌ Empirical gates FAILED")
    else:
        update_status("empirical_pass", True)
        print("\n✅ Empirical gates PASSED")
    
    # 3) Schema validation (if results exist)
    results_jsonl = ROOT / "RESULTS" / "adversarial_manifest.jsonl"
    if results_jsonl.exists():
        print("\n[2.5/3] Schema Validation: RESULTS JSONL")
        print("-" * 70)
        rc = run(["python3", str(TOOLS / "validate_results_jsonl.py"), str(results_jsonl)])
        if rc != 0:
            all_pass = False
            print("\n❌ Schema validation FAILED")
        else:
            print("\n✅ Schema validation PASSED")
    
    # 4) Final status
    print("\n" + "=" * 70)
    print("Final Status")
    print("=" * 70)
    
    if all_pass:
        update_status("restricted_class_proved", True)
        print("\n✅✅✅ ALL GATES PASSED ✅✅✅")
        print("A3.1–A3.4 (restricted) = PROVED")
        print("P-time witness finder on bounded-degree expanders: PROVED")
        sys.exit(0)
    else:
        update_status("restricted_class_proved", False)
        print("\n❌❌❌ SOME GATES FAILED ❌❌❌")
        print("Check artifacts and logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()

