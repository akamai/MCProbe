import os
import subprocess
import shutil
import json
from helpers import dprint


def run_semgrep_direct(args: list[str]):
    semgrep_bin = shutil.which("semgrep")
    if not semgrep_bin:
        return 2, ""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [semgrep_bin] + args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=300,
        )
        return proc.returncode, proc.stdout
    except subprocess.TimeoutExpired:
        dprint("[SEMGREP] Timed out after 300s")
        return 2, ""
    except Exception as e:
        dprint(f"[SEMGREP] Subprocess error: {e}")
        return 2, ""


def analyze_js_repo(repo_path: str, output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    args = [
        "--config=p/ci",
        "--json",
        "--quiet",
        "--exclude=node_modules",
        "--exclude=tests",
        "--exclude=test",
        "--exclude=__tests__",
        repo_path,
    ]
    exit_code, output = run_semgrep_direct(args)
    if exit_code not in (0, 1):
        dprint(f"[SEMGREP] Semgrep failed")
        return 0
    try:
        data = json.loads(output)
        issues = data.get("results", [])
    except Exception as e:
        dprint(f"[SEMGREP] Failed to parse JSON output: {e}")
        issues = []
    severities = {"HIGH", "MEDIUM"}
    filtered_issues = [
        i for i in issues if i.get("extra", {}).get("severity") in severities
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        if not filtered_issues:
            f.write("[SEMGREP] 0 issues found by Semgrep.\n")
            dprint("[SEMGREP] 0 issues found by Semgrep.")
        else:
            dprint(f"[SEMGREP] Semgrep found: {len(filtered_issues)} issues")
            f.write(f"{len(filtered_issues)} issues found by Semgrep:\n\n")
            for i in issues:
                path = i.get("path")
                line = i.get("start", {}).get("line")
                msg = i.get("extra", {}).get("message")
                sev = i.get("extra", {}).get("severity")
                f.write(f"{path}:{line} [{sev}] {msg}\n")

    dprint(f"[SEMGREP] Done")
    return len(filtered_issues)
