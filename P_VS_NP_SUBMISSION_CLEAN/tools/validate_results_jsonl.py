#!/usr/bin/env python3
"""
Validate RESULTS JSONL against strict schema
"""

import sys
import json
import pathlib

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "PNPResultsRow.schema.json"

def main(path):
    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema not found: {SCHEMA_PATH}", file=sys.stderr)
        sys.exit(1)
    
    SCHEMA = json.loads(SCHEMA_PATH.read_text())
    ok = True
    line_no = 0
    errors = []
    
    with open(path, "r") as f:
        for line in f:
            line_no += 1
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                jsonschema.validate(obj, SCHEMA)
            except json.JSONDecodeError as e:
                ok = False
                errors.append({"line": line_no, "error": f"JSON decode: {e}", "line_content": line.strip()[:100]})
            except jsonschema.ValidationError as e:
                ok = False
                errors.append({"line": line_no, "error": str(e), "obj": obj})
    
    if not ok:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        print(f"\n❌ Schema validation FAILED: {len(errors)} error(s)", file=sys.stderr)
        sys.exit(3)
    else:
        print(json.dumps({"ok": True, "lines_validated": line_no}, indent=2))
        print(f"\n✅ Schema validation PASSED: {line_no} line(s)")
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_results_jsonl.py <path_to_jsonl>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

