import os, time, re, json, threading
from typing import Optional
from html import escape as _esc

DEBUG_SLOW = os.getenv("DEBUG_SLOW", "0") == "1"
DEBUG_DELAY = float(os.getenv("DEBUG_DELAY", "0.6"))  # seconds

_thread_local = threading.local()


def dprint(msg: str, delay: Optional[float] = None):
    """Print a line immediately, or buffer if running in a worker thread."""
    buf = getattr(_thread_local, "buffer", None)
    if buf is not None:
        buf.write(str(msg) + "\n")
    else:
        print(msg, flush=True)
        if DEBUG_SLOW:
            time.sleep(delay if delay is not None else DEBUG_DELAY)


# ---------------------------------------------------------------------------
# Shared file-exclusion helpers (used by every analyzer)
# ---------------------------------------------------------------------------

# Directory names that are never worth scanning for security issues.
EXCLUDE_DIRS: frozenset = frozenset({
    # dependency installs
    "node_modules", "venv", ".venv", "env", "vendor",
    # client / SDK consumer code (typically test harnesses)
    "client", "clients",
    # build artefacts
    "__pycache__", "dist", "build", ".pytest_cache", ".nyc_output",
    "coverage", "htmlcov", ".tox",
    # VCS
    ".git", ".svn", ".hg",
    # test suites — the main source of false positives
    "tests", "test", "testing", "__tests__", "spec", "specs",
    "integration_tests", "e2e", "functional_tests",
    # documentation & examples
    "docs", "doc", "documentation", "examples", "example",
    "samples", "sample", "demo", "demos", "tutorial", "tutorials",
    # fixtures / mocks / stubs
    "fixtures", "fixture", "mocks", "mock", "stubs", "stub",
    "testdata", "test_data", "testfiles",
    # benchmarks
    "benchmark", "benchmarks", "perf",
    # CI helpers
    "scripts",        # often contain CI/dev-only shell + Python
})

# File-name patterns that indicate a test / fixture file regardless of directory.
_TEST_FILE_RE = re.compile(
    r"^(test_|tests_)|(_test|_spec)\.(py|js|ts|jsx|tsx|mjs|mts)$"
    r"|^(conftest|setup_tests?|jest\.config|vitest\.config|karma\.conf)\.",
    re.IGNORECASE,
)

# Common example/fixture patterns in file names
_EXAMPLE_FILE_RE = re.compile(
    r"^(example|sample|demo|fixture|mock|stub|fake)[_\-\.]",
    re.IGNORECASE,
)


def is_test_file(filename: str) -> bool:
    """Return True if the bare filename looks like a test or fixture file."""
    return bool(_TEST_FILE_RE.search(filename) or _EXAMPLE_FILE_RE.match(filename))


def should_skip_path(path: str) -> bool:
    """
    Return True if any component of *path* is an excluded directory,
    or if the file itself looks like a test / fixture.
    """
    parts = path.replace("\\", "/").split("/")
    filename = parts[-1]
    for part in parts[:-1]:           # check directory components
        if part.lower() in EXCLUDE_DIRS:
            return True
    return is_test_file(filename)


# ---------------------------------------------------------------------------
# AI output parsing
# ---------------------------------------------------------------------------

def parse_ai_findings(raw: str) -> dict:
    """Parse AI JSON output into {"summary": str, "findings": list}."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "findings" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return {"summary": raw[:500] if raw else "No AI output.", "findings": []}


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

_SEV_COLORS = {
    "CRITICAL": "#d63031", "HIGH": "#e17055",
    "MEDIUM": "#fdcb6e", "LOW": "#00b894", "INFO": "#74b9ff",
}


def _read_report_file(path: str, max_chars: int = 60_000) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
        if len(data) > max_chars:
            return data[:int(max_chars * 0.7)] + "\n\n...[TRUNCATED]...\n\n" + data[-int(max_chars * 0.3):]
        return data
    except Exception:
        return ""


def generate_html_report(state: dict) -> str:
    """Build an HTML security report for a single repo. Returns the file path."""
    repo_name = _esc(state.get("name", "unknown"))
    language = _esc(state.get("language", "unknown"))
    analysis_root = state.get("analysis_root", "")

    ai_data = parse_ai_findings(state.get("ai_analysis", ""))
    findings = ai_data.get("findings", [])
    summary = ai_data.get("summary", "")

    module_reports = [
        ("MCP Flow / Command Injection", state.get("code_analysis_path", ""), state.get("cfg_issues", -1)),
        ("Network Analysis",             state.get("net_analysis_path", ""),  state.get("net_issues", -1)),
        ("SSE / Streaming",              state.get("sse_analysis_path", ""),  state.get("sse_issues", -1)),
        ("Auth & Authorization",         state.get("auth_analysis_path", ""), state.get("auth_issues", -1)),
        ("Bandit Static Analysis",       state.get("bandit_analysis_path", ""), state.get("bandit_high", 0)),
        ("Semgrep Static Analysis",      state.get("semgrep_analysis_path", ""), state.get("semgrep_issues", 0)),
    ]

    module_html = ""
    for title, path, count in module_reports:
        body = _read_report_file(path)
        badge = f'<span class="badge">{count}</span>' if count > 0 else ""
        content = _esc(body) if body.strip() else "<em>No findings or report unavailable.</em>"
        module_html += f"""
        <div class="section">
            <h2>{_esc(title)} {badge}</h2>
            <pre>{content}</pre>
        </div>"""

    if findings:
        rows = ""
        for f in findings:
            sev = _esc(str(f.get("severity", "INFO")))
            color = _SEV_COLORS.get(sev, "#74b9ff")
            loc = _esc(str(f.get("file", "")))
            if f.get("line"):
                loc += f":{f['line']}"
            rows += f"""
            <tr>
                <td><span class="sev" style="background:{color}">{sev}</span></td>
                <td>{_esc(str(f.get('title', '')))}</td>
                <td class="mono">{loc}</td>
                <td>{_esc(str(f.get('description', '')))}</td>
                <td>{_esc(str(f.get('recommendation', '')))}</td>
            </tr>"""
        ai_section = f"""
        <div class="section">
            <h2>AI Security Review</h2>
            <p>{_esc(summary)}</p>
            <table>
                <thead><tr>
                    <th>Severity</th><th>Title</th><th>Location</th>
                    <th>Description</th><th>Recommendation</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""
    else:
        ai_section = f"""
        <div class="section">
            <h2>AI Security Review</h2>
            <p>{_esc(summary) if summary else 'No AI findings.'}</p>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>MCProbe Report — {repo_name}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:#0d1117; color:#e6edf3; padding:24px; }}
  h1 {{ color:#58a6ff; margin-bottom:4px; }}
  .meta {{ color:#8b949e; margin-bottom:24px; }}
  .section {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:16px; }}
  .section h2 {{ color:#c9d1d9; font-size:16px; margin-bottom:10px; }}
  pre {{ background:#0d1117; color:#8b949e; padding:12px; border-radius:6px; overflow-x:auto; font-size:12px; white-space:pre-wrap; word-break:break-word; max-height:400px; overflow-y:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#21262d; color:#8b949e; text-align:left; padding:8px; }}
  td {{ padding:8px; border-bottom:1px solid #21262d; vertical-align:top; }}
  .sev {{ padding:2px 8px; border-radius:4px; color:#fff; font-weight:bold; font-size:11px; }}
  .mono {{ font-family:Consolas,monospace; font-size:12px; color:#8b949e; }}
  .badge {{ background:#f85149; color:#fff; font-size:11px; padding:1px 7px; border-radius:10px; margin-left:6px; }}
  .summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:8px; margin-bottom:20px; }}
  .stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; text-align:center; }}
  .stat .num {{ font-size:28px; font-weight:bold; color:#58a6ff; }}
  .stat .lbl {{ font-size:11px; color:#8b949e; }}
</style>
</head><body>
<h1>MCProbe Security Report</h1>
<p class="meta">{repo_name} &middot; {language}</p>

<div class="summary-grid">
  <div class="stat"><div class="num">{max(0, state.get('cfg_issues', 0))}</div><div class="lbl">MCP Flow</div></div>
  <div class="stat"><div class="num">{max(0, state.get('net_issues', 0))}</div><div class="lbl">Network</div></div>
  <div class="stat"><div class="num">{max(0, state.get('sse_issues', 0))}</div><div class="lbl">SSE</div></div>
  <div class="stat"><div class="num">{max(0, state.get('auth_issues', 0))}</div><div class="lbl">Auth</div></div>
  <div class="stat"><div class="num">{max(0, state.get('bandit_high', 0)) + max(0, state.get('semgrep_issues', 0))}</div><div class="lbl">Static</div></div>
  <div class="stat"><div class="num">{len(findings)}</div><div class="lbl">AI Findings</div></div>
</div>

{ai_section}
{module_html}
</body></html>"""

    out_path = os.path.join(analysis_root, "report.html")
    os.makedirs(analysis_root, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# ---------------------------------------------------------------------------
# Cost estimation (per 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-6":            (15.00, 75.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "claude-sonnet-4-6":          (3.00,  15.00),
    "claude-haiku-4-5-20251001":  (0.80,  4.00),
    # OpenAI
    "gpt-4o":                     (2.50,  10.00),
    "gpt-4o-mini":                (0.15,  0.60),
    "gpt-4.1":                    (2.00,  8.00),
    "gpt-4.1-mini":               (0.40,  1.60),
    "gpt-4.1-nano":               (0.10,  0.40),
    "o3":                         (10.00, 40.00),
    "o3-mini":                    (1.10,  4.40),
    "o4-mini":                    (1.10,  4.40),
}

MAX_OUTPUT_TOKENS = 4096


def estimate_tokens(chars: int) -> int:
    return max(1, int(chars / 3.5))


def print_cost_summary(results: list, model_name: str, validate: bool = False):
    input_price, output_price = MODEL_PRICING.get(model_name, (0, 0))
    ok_results = [r for r in results if "_error" not in r]

    print(f"\n{'='*60}")
    print(f"[MCPROBE] Cost Estimate — model: {model_name}")
    print(f"{'='*60}")

    if not input_price:
        print(f"  Unknown model pricing for '{model_name}'.")
        print(f"  Known models: {', '.join(sorted(MODEL_PRICING))}")
        print()

    total_r_input = 0
    total_v_input = 0
    for r in ok_results:
        r_chars = r.get("prompt_chars", 0)
        r_tokens = estimate_tokens(r_chars)
        total_r_input += r_tokens
        name = r.get("name", "?")
        if validate:
            v_chars = r.get("validate_chars", 0)
            v_tokens = estimate_tokens(v_chars)
            total_v_input += v_tokens
            n_files = len(r.get("interesting_files", []))
            print(f"  {name}: ~{r_tokens:,} review tokens | ~{v_tokens:,} validation tokens ({n_files} files)")
        else:
            print(f"  {name}: ~{r_tokens:,} input tokens ({r_chars:,} chars)")

    passes = 2 if validate else 1
    total_output_tokens = MAX_OUTPUT_TOKENS * len(ok_results) * passes
    total_input_tokens = total_r_input + total_v_input
    grand_tokens = total_input_tokens + total_output_tokens

    print()
    print(f"  Total review input tokens:               ~{total_r_input:,}")
    if validate:
        print(f"  Total validation input tokens:           ~{total_v_input:,}")
    print(f"  Total output tokens (AI responses, max): ~{total_output_tokens:,}")

    if input_price:
        r_in_cost  = (total_r_input / 1_000_000) * input_price
        v_in_cost  = (total_v_input / 1_000_000) * input_price if validate else 0.0
        out_cost   = (total_output_tokens / 1_000_000) * output_price
        total_cost = r_in_cost + v_in_cost + out_cost
        print()
        print(f"  Pricing: ${input_price:.2f}/1M input  ·  ${output_price:.2f}/1M output")
        print()
        print(f"  Review input cost:                ${r_in_cost:.4f}  ({total_r_input:,} × ${input_price:.2f}/1M)")
        if validate:
            print(f"  Validation input cost:            ${v_in_cost:.4f}  ({total_v_input:,} × ${input_price:.2f}/1M)")
        print(f"  Output cost (AI responses, max):  ${out_cost:.4f}  ({total_output_tokens:,} × ${output_price:.2f}/1M)")
        print(f"  {'─'*52}")
        print(f"  Total cost:                       ${total_cost:.4f}  (~{grand_tokens:,} tokens)")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Estimate prompt size from an existing analysis folder
# ---------------------------------------------------------------------------

_REPORT_FILES = [
    ("CFG / Command Injection Analysis", "mcp_flow_analysis.txt"),
    ("Network Analysis",                 "network_analysis.txt"),
    ("SSE / Streaming Analysis",         "sse_analysis.txt"),
    ("Auth & Authorization Analysis",    "auth_analysis.txt"),
    ("Bandit Static Analysis",           "bandit_analysis.txt"),
    ("Semgrep Static Analysis",          "semgrep_analysis.txt"),
]

_PROMPT_OVERHEAD = 800


def estimate_prompt_chars_from_folder(analysis_root: str) -> int:
    total = _PROMPT_OVERHEAD
    for title, filename in _REPORT_FILES:
        path = os.path.join(analysis_root, filename)
        body = _read_report_file(path, max_chars=120_000)
        section = f"## {title}\n\n{body}\n" if body.strip() else f"## {title}\n\n(No findings or report unavailable.)\n"
        total += len(section)
    return total


# ---------------------------------------------------------------------------
# Interesting-file extraction from scanner report files (for --validate)
# ---------------------------------------------------------------------------

_SRC_EXTENSIONS = frozenset({".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"})

# Matches "File  : path/to/file.py" or "File : path/to/file.py:42" (auth/sse/network)
_FILE_LABEL_RE = re.compile(r'File\s*:\s*([^\s\|]+)', re.IGNORECASE)
# Matches "[path/to/file.py:42]" in mcp_flow reports
_BRACKET_PATH_RE = re.compile(r'\[([^\[\]\s]+\.(?:py|js|ts|jsx|tsx|mjs|mts)):\d+\]')
# Matches "path/to/file.py:42 [" — bandit / semgrep line-start pattern
_LINE_START_PATH_RE = re.compile(r'^([^\s#\[]+\.(?:py|js|ts|jsx|tsx|mjs|mts)):\d+\s', re.MULTILINE)


def extract_files_from_report(report_path: str, repo_local_path: str) -> list:
    """
    Parse a scanner report file and return deduplicated absolute paths of
    source files mentioned in it.  Only paths that actually exist on disk
    are included.
    """
    if not report_path or not os.path.isfile(report_path):
        return []
    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return []

    raw_paths = []

    for m in _FILE_LABEL_RE.finditer(text):
        # Strip trailing :lineno and any surrounding whitespace
        p = re.sub(r':\d+$', '', m.group(1).strip())
        raw_paths.append(p)

    for m in _BRACKET_PATH_RE.finditer(text):
        raw_paths.append(m.group(1))

    for m in _LINE_START_PATH_RE.finditer(text):
        raw_paths.append(m.group(1))

    result = []
    seen: set = set()
    for p in raw_paths:
        p = p.strip().strip('"\'')
        if not p or p in seen:
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in _SRC_EXTENSIONS:
            continue
        seen.add(p)
        abs_p = p if os.path.isabs(p) else os.path.join(repo_local_path, p)
        abs_p = os.path.normpath(abs_p)
        if os.path.isfile(abs_p):
            result.append(abs_p)

    return result