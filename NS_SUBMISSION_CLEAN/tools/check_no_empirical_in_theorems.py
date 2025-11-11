#!/usr/bin/env python3
"""
CI Check: Forbid empirical references in theorem statements
Kill-switch: Fail if "chi_max", "empirical", or "8.95" appears in theorem/lemma/proof sections
"""

import sys
import re
import pathlib

FORBIDDEN_PATTERNS = [
    r'\bchi_max\b',
    r'\b8\.95\s*[×x]\s*10\s*[-\^]?\s*6',
    r'empirical\s+(evidence|data|observation|result)',
    r'Based on empirical',
]

def scan_tex_file(filepath: pathlib.Path):
    """Scan TEX file for forbidden patterns in theorem/lemma/proof sections."""
    txt = filepath.read_text(encoding="utf-8", errors="ignore")
    hits = []
    
    # Find theorem/lemma/proof environments
    env_pattern = r'\\(?:begin|end)\{(theorem|lemma|proposition|proof|corollary)\}'
    env_matches = list(re.finditer(env_pattern, txt))
    
    # Track which environment we're in
    in_theorem = False
    current_env_start = 0
    
    for i, match in enumerate(env_matches):
        is_begin = match.group(0).startswith('\\begin')
        
        if is_begin:
            # Check previous environment for forbidden patterns
            if in_theorem:
                env_text = txt[current_env_start:match.start()]
                for pattern in FORBIDDEN_PATTERNS:
                    for m in re.finditer(pattern, env_text, re.IGNORECASE):
                        line = txt.count("\n", 0, current_env_start + m.start()) + 1
                        hits.append({
                            "file": str(filepath),
                            "line": line,
                            "pattern": pattern,
                            "context": env_text[max(0, m.start()-50):m.end()+50]
                        })
            
            in_theorem = True
            current_env_start = match.end()
        else:
            # End of environment
            if in_theorem:
                env_text = txt[current_env_start:match.start()]
                for pattern in FORBIDDEN_PATTERNS:
                    for m in re.finditer(pattern, env_text, re.IGNORECASE):
                        line = txt.count("\n", 0, current_env_start + m.start()) + 1
                        hits.append({
                            "file": str(filepath),
                            "line": line,
                            "pattern": pattern,
                            "context": env_text[max(0, m.start()-50):m.end()+50]
                        })
            in_theorem = False
    
    # Check final environment if file ends in theorem
    if in_theorem:
        env_text = txt[current_env_start:]
        for pattern in FORBIDDEN_PATTERNS:
            for m in re.finditer(pattern, env_text, re.IGNORECASE):
                line = txt.count("\n", 0, current_env_start + m.start()) + 1
                hits.append({
                    "file": str(filepath),
                    "line": line,
                    "pattern": pattern,
                    "context": env_text[max(0, m.start()-50):m.end()+50]
                })
    
    return hits

def main(root):
    rootp = pathlib.Path(root)
    tex_files = list(rootp.rglob("*.tex"))
    all_hits = []
    
    for f in tex_files:
        all_hits += scan_tex_file(f)
    
    if all_hits:
        print("❌ CI CHECK FAILED: Empirical references found in theorem sections", file=sys.stderr)
        print(f"Found {len(all_hits)} violation(s):\n", file=sys.stderr)
        for hit in all_hits:
            print(f"  {hit['file']}:{hit['line']} - Pattern: {hit['pattern']}", file=sys.stderr)
            print(f"    Context: ...{hit['context']}...\n", file=sys.stderr)
        sys.exit(1)
    else:
        print("✅ CI CHECK PASSED: No empirical references in theorem sections")
        sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "proofs/tex")

