import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


source = os.environ["STYLE_SOURCE"]
test = os.environ["STYLE_TEST"]
base = os.environ["STYLE_BASE"]
paths = [source, test]
before = {path: Path(path).read_text() for path in paths}


def lint(paths):
    result = subprocess.run(["ruff", "check", "--output-format", "json", *paths],
                            capture_output=True, text=True)
    return result.returncode, json.loads(result.stdout)


candidate_rc, candidate_diagnostics = lint(paths)
candidate_format = subprocess.run(["ruff", "format", "--check", *paths],
                                  capture_output=True, text=True)
baseline_source = subprocess.check_output(["git", "show", f"{base}:{source}"], text=True)
Path(source).write_text(baseline_source)
baseline_rc, baseline_diagnostics = lint([source])
baseline_format = subprocess.run(["ruff", "format", "--check", source],
                                 capture_output=True, text=True)
Path(source).write_text(before[source])


def normalized(diagnostics):
    return Counter((Path(item["filename"]).relative_to(Path.cwd()).as_posix(),
                    item["code"], item["message"]) for item in diagnostics)


introduced = normalized(candidate_diagnostics) - normalized(baseline_diagnostics)
subprocess.check_call(["ruff", "format", *paths])
formatted = {path: Path(path).read_text() for path in paths}
for path in paths:
    if ast.dump(ast.parse(before[path]), include_attributes=False) != ast.dump(ast.parse(formatted[path]), include_attributes=False):
        raise RuntimeError(f"Formatter changed the AST: {path}")
formatted_rc, formatted_diagnostics = lint(paths)
formatted_check = subprocess.run(["ruff", "format", "--check", *paths], capture_output=True, text=True)
report = {
    "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "base": base,
    "ruff": subprocess.check_output(["ruff", "--version"], text=True).strip(),
    "paths": paths,
    "candidate_lint_rc": candidate_rc,
    "candidate_diagnostics": candidate_diagnostics,
    "candidate_format_rc": candidate_format.returncode,
    "candidate_format_output": candidate_format.stdout + candidate_format.stderr,
    "baseline_lint_rc": baseline_rc,
    "baseline_diagnostics": baseline_diagnostics,
    "baseline_format_rc": baseline_format.returncode,
    "baseline_format_output": baseline_format.stdout + baseline_format.stderr,
    "introduced": [list(key) + [value] for key, value in introduced.items()],
    "formatted_lint_rc": formatted_rc,
    "formatted_diagnostics": formatted_diagnostics,
    "formatted_check_rc": formatted_check.returncode,
    "ast_equivalent": True,
    "changed": [path for path in paths if before[path] != formatted[path]],
    "files": formatted,
}
Path("style-result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key not in ("files", "candidate_diagnostics", "baseline_diagnostics", "formatted_diagnostics")}, indent=2))
if introduced or candidate_format.returncode or formatted_check.returncode:
    sys.exit(1)
