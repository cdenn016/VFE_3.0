"""Read-only syntax and source inventory for the audited Python scope."""
from __future__ import annotations
import ast
import json
import tokenize
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
groups = {"package": sorted((ROOT / "vfe3").rglob("*.py")),
          "tests": sorted((ROOT / "tests").rglob("*.py")),
          "root_scripts": sorted(ROOT.glob("*.py"))}
result = {"groups": {}, "syntax_errors": [], "duplicate_literal_keys": []}
for group, paths in groups.items():
    lines = 0
    for path in paths:
        with tokenize.open(path) as stream:
            source = stream.read()
        lines += len(source.splitlines())
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            result["syntax_errors"].append({"path": str(path.relative_to(ROOT)), "line": exc.lineno, "message": exc.msg})
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = {}
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, (str, int, float, bool)):
                    if key.value in seen:
                        result["duplicate_literal_keys"].append({"path": str(path.relative_to(ROOT)), "key": repr(key.value),
                                                                  "first_line": seen[key.value], "line": key.lineno})
                    seen[key.value] = key.lineno
    result["groups"][group] = {"files": len(paths), "physical_lines": lines}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
