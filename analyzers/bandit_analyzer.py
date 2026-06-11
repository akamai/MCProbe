import os
import logging
from bandit.core.manager import BanditManager
from bandit.core.config import BanditConfig
from bandit.core.metrics import Metrics
from helpers import dprint, EXCLUDE_DIRS, should_skip_path

def analyze_with_bandit(repo_path: str, output_path: str):
    dprint(f"[BANDIT] Scanning Python code in: {repo_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for name in ("bandit", "bandit.core", "bandit.core.node_visitor",
                  "bandit.core.tester"):
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.ERROR)
        lgr.handlers.clear()
        lgr.propagate = False

    config = BanditConfig()

    manager = BanditManager(config, agg_type='file', debug=False)
    manager.discover_files([repo_path], True)
    # Filter out excluded directories and test/fixture files
    before = len(manager.files_list)
    manager.files_list = [
        f for f in manager.files_list
        if not should_skip_path(f)
    ]
    after = len(manager.files_list)
    dprint(f"[BANDIT] Filtered {before - after} excluded files; {after} remain to scan.")
    manager.run_tests()  # perform the actual scan

    # Threshold: HIGH severity (any confidence)
    #         OR MEDIUM severity with HIGH confidence
    all_raw = manager.get_issue_list(sev_level='LOW', conf_level='LOW')
    filtered_issues = [
        i for i in all_raw
        if i.severity == "HIGH"
        or (i.severity == "MEDIUM" and i.confidence == "HIGH")
    ]

    lines = []
    for issue in sorted(filtered_issues,
                        key=lambda x: (0 if x.severity == "HIGH" else 1, x.fname, x.lineno)):
        lines.append(
            f"{issue.fname}:{issue.lineno} "
            f"[{issue.test_id}] {issue.text} "
            f"(Severity: {issue.severity}, Confidence: {issue.confidence})"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        if not lines:
            f.write("No security issues found by Bandit.\n")
        else:
            f.write(f"{len(lines)} potential issues found by Bandit:\n\n")
            f.write("\n".join(lines))

    dprint(f"[BANDIT] Done — {len(filtered_issues)} issue(s) at threshold")
    return len(filtered_issues)