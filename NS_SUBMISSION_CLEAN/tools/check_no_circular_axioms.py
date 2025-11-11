#!/usr/bin/env python3
"""
CI Gate: Check for circular axioms (shell_absorb_*)
These should be lemmas, not axioms, as they assume the conclusion we're trying to prove.
"""
import re
import os
import sys
from pathlib import Path

def check_circular_axioms(filepath):
    """Check for problematic circular axioms."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find axiom declarations
    axiom_pattern = re.compile(r'^\s*axiom\s+(\w+)\s*', re.MULTILINE)
    
    # List of forbidden axiom names (circular assumptions)
    forbidden_axioms = [
        'shell_absorb_low_to_dissipation',
        'shell_absorb_high_to_dissipation',
        'shell_absorb_far_to_dissipation',
    ]
    
    issues = []
    for match in axiom_pattern.finditer(content):
        axiom_name = match.group(1)
        if axiom_name in forbidden_axioms:
            # Find line number
            line_num = content.count('\n', 0, match.start()) + 1
            issues.append({
                "file": filepath,
                "line": line_num,
                "kind": "circular_axiom",
                "axiom": axiom_name,
                "message": f"Circular axiom '{axiom_name}' found. This should be a lemma proved from base tools, not an axiom."
            })
    
    return issues

if __name__ == "__main__":
    lean_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("proofs/lean")
    
    if not lean_dir.is_dir():
        print(f"Error: {lean_dir} not found.")
        sys.exit(1)
    
    all_issues = []
    for filepath in lean_dir.rglob("*.lean"):
        all_issues.extend(check_circular_axioms(filepath))
    
    if all_issues:
        print("❌ CI CHECK FAILED: Circular axioms found")
        for issue in all_issues:
            print(f"  - {issue['file']}:{issue['line']} - {issue['message']}")
        sys.exit(1)
    else:
        print("✅ CI CHECK PASSED: No circular axioms found")
        sys.exit(0)

