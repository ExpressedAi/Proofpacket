#!/usr/bin/env python3
"""
Lean No-Sorry Gate: Fail fast if any `sorry` or admitted axioms beyond allowlist
"""

import sys
import re
import json
import pathlib

ALLOW_AXIOMS = {
    # keep tight; add only if you truly import them explicitly
    "classical.choice": False,  # example: disallow by default
    "expander_mixing_lemma": True,  # We explicitly state this as an axiom
}

def scan(filepath: pathlib.Path):
    txt = filepath.read_text(encoding="utf-8", errors="ignore")
    hits = []
    
    # 1) any 'sorry'
    for m in re.finditer(r'\bsorry\b', txt):
        line = txt.count("\n", 0, m.start()) + 1
        hits.append({"file": str(filepath), "line": line, "kind": "sorry"})
    
    # 2) admitted axioms outside allowlist
    for m in re.finditer(r'\baxiom\s+([A-Za-z0-9_\.]+)', txt):
        name = m.group(1)
        if not ALLOW_AXIOMS.get(name, False):
            line = txt.count("\n", 0, m.start()) + 1
            hits.append({"file": str(filepath), "line": line, "kind": "axiom", "name": name})
    
    return hits

def main(root):
    rootp = pathlib.Path(root)
    lean_files = list(rootp.rglob("*.lean"))
    all_hits = []
    
    for f in lean_files:
        all_hits += scan(f)
    
    result = {"ok": len(all_hits) == 0, "issues": all_hits}
    print(json.dumps(result, indent=2))
    
    if not result["ok"]:
        print(f"\n❌ Formal gate FAILED: {len(all_hits)} issue(s) found", file=sys.stderr)
        for issue in all_hits:
            print(f"  - {issue['file']}:{issue['line']} - {issue['kind']}", file=sys.stderr)
        sys.exit(2)
    else:
        print("\n✅ Formal gate PASSED: No `sorry` or unauthorized axioms")
        sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

