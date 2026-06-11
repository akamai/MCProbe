"""
MCProbe — Autonomous Claude Agent Scanner
==========================================
Gives Claude a set of callable tools so it can actively explore a repository,
run the built-in static analyzers, and produce its own vulnerability report.

Uses the Anthropic tool_use API (client.messages.create with tools=[...]).
Called from orchestrator.use_ai() when backend == "claude-agent".
"""

import os
import re
import fnmatch
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from helpers import EXCLUDE_DIRS, should_skip_path, dprint

# ---------------------------------------------------------------------------
# Safety / cost limits
# ---------------------------------------------------------------------------

MAX_ITERATIONS       = 15    # max API round-trips per repo
MAX_READ_CALLS_TOTAL = 30    # total read_file calls allowed per scan
MAX_READS_PER_PATH   = 2     # how many times the same file may be read
MAX_LINES_PER_READ   = 400   # hard cap on lines returned per read_file call
MAX_SEARCH_RESULTS   = 30    # max lines returned by search_code
MAX_ANALYZER_CHARS   = 8_000 # analyzer report is truncated to this length

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert security researcher specializing in MCP (Model Context Protocol) servers.
Your task: perform a thorough autonomous security audit of the repository "{repo_name}".

Investigation strategy:
1. Call list_directory("") to understand the top-level repo layout.
2. Identify MCP tool entry points — search for @mcp.tool and call_tool decorators.
3. Call run_analyzer("mcp_flow") to get automated call-graph results.
4. Read key source files; trace how user inputs reach dangerous operations.
5. Run additional analyzers (auth, network, sse, bandit) as the code warrants.
6. Use search_code for targeted pattern searches (subprocess, eval, open, requests, exec).
7. Once you have a complete picture, call report_findings with every vulnerability found.

Focus on:
- Command injection: user input reaching subprocess / os.system / eval / exec
- SSRF: user-controlled URLs passed to HTTP libraries
- Path traversal: user input used in file paths without sanitisation
- Hardcoded credentials / secrets in source or config files
- Missing or easily bypassable authentication
- Prompt injection risk in MCP tool descriptions or names

Be specific — cite relative file paths and line numbers.
You MUST call report_findings before finishing; do not stop before submitting your report.\
"""

# ---------------------------------------------------------------------------
# Tool JSON schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict] = [
    {
        "name": "list_directory",
        "description": (
            "List files and sub-directories inside the repository. "
            "Use path='' for the repo root. "
            "Set recursive=true to get a full flat listing (skip generated dirs automatically)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":      {"type": "string",
                              "description": "Relative path from repo root. Empty string = root."},
                "recursive": {"type": "boolean",
                              "description": "If true, walk all subdirectories recursively.",
                              "default": False},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a source file from the repository with line numbers. "
            f"Returns at most {MAX_LINES_PER_READ} lines per call. "
            "Use start_line / end_line to read different sections of a large file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string",
                               "description": "Relative path from repo root."},
                "start_line": {"type": "integer",
                               "description": "1-based first line to return (default 1).",
                               "default": 1},
                "end_line":   {"type": "integer",
                               "description": f"Last line to return (default start_line + {MAX_LINES_PER_READ} - 1)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search for a regex pattern across repository source files. "
            f"Returns up to {MAX_SEARCH_RESULTS} matching lines with file:line context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":     {"type": "string",
                                "description": "Python regex pattern to search for."},
                "file_glob":   {"type": "string",
                                "description": "Glob filter, e.g. '**/*.py' or '**/*.js'. Default: '**/*.py'",
                                "default": "**/*.py"},
                "max_results": {"type": "integer",
                                "description": f"Cap on results returned (max {MAX_SEARCH_RESULTS}).",
                                "default": 25},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_analyzer",
        "description": (
            "Run one of the built-in MCProbe security analyzers and return its report. "
            "Use this to get systematic coverage before diving into specific files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["mcp_flow", "network", "auth", "sse", "bandit"],
                    "description": (
                        "mcp_flow — traces MCP tool params to dangerous sinks; "
                        "network — exposed network bindings; "
                        "auth — hardcoded creds / JWT misuse / auth bypass; "
                        "sse — SSE/streaming SSRF and CORS issues; "
                        "bandit — Python static analysis (HIGH-severity only)"
                    ),
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "report_findings",
        "description": (
            "Submit your complete security findings and end the audit. "
            "Call this exactly once when you have finished your investigation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": "List of vulnerability findings (empty list if none found).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity":       {"type": "string",
                                               "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]},
                            "title":          {"type": "string",
                                               "description": "Short one-line title."},
                            "file":           {"type": "string",
                                               "description": "Relative file path (empty if repo-wide)."},
                            "line":           {"type": "integer",
                                               "description": "Line number (0 if not applicable)."},
                            "description":    {"type": "string",
                                               "description": "What is wrong and why it is exploitable."},
                            "recommendation": {"type": "string",
                                               "description": "Concrete remediation advice."},
                        },
                        "required": ["severity", "title", "description", "recommendation"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "1-3 paragraph executive summary of your audit.",
                },
            },
            "required": ["findings", "summary"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _list_directory(path: str, recursive: bool, repo_path: str) -> str:
    """Return a formatted directory listing."""
    target = os.path.normpath(os.path.join(repo_path, path)) if path else repo_path

    # Safety: don't escape the repo
    if not os.path.abspath(target).startswith(os.path.abspath(repo_path)):
        return "ERROR: path is outside the repository."

    if not os.path.exists(target):
        return f"ERROR: path '{path}' does not exist in this repository."

    if not os.path.isdir(target):
        return f"ERROR: '{path}' is a file, not a directory. Use read_file to read it."

    lines: List[str] = []

    if recursive:
        for root, dirs, files in os.walk(target):
            dirs[:] = sorted(d for d in dirs if d.lower() not in EXCLUDE_DIRS)
            rel_root = os.path.relpath(root, repo_path)
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                if not should_skip_path(fpath):
                    lines.append(os.path.join(rel_root, fname).replace("\\", "/"))
        if not lines:
            return "(no source files found)"
        if len(lines) > 300:
            lines = lines[:300]
            lines.append(f"… (+{len(lines) - 300} more files, use a sub-path to narrow down)")
        return "\n".join(lines)
    else:
        entries = sorted(os.listdir(target))
        for entry in entries:
            full = os.path.join(target, entry)
            rel  = os.path.relpath(full, repo_path).replace("\\", "/")
            if os.path.isdir(full):
                if entry.lower() not in EXCLUDE_DIRS:
                    lines.append(f"[dir]  {rel}/")
            else:
                lines.append(f"[file] {rel}")
        return "\n".join(lines) if lines else "(empty directory)"


def _read_file(path: str, start_line: int, end_line: Optional[int],
               repo_path: str,
               call_counts: Dict[str, int], total_reads_ref: List[int]) -> str:
    """Return numbered lines from a file; enforces per-path and global caps."""
    norm_path = path.lstrip("/\\").replace("\\", "/")
    abs_path  = os.path.normpath(os.path.join(repo_path, norm_path))

    if not os.path.abspath(abs_path).startswith(os.path.abspath(repo_path)):
        return "ERROR: path is outside the repository."
    if not os.path.exists(abs_path):
        return f"ERROR: file '{norm_path}' not found."
    if not os.path.isfile(abs_path):
        return f"ERROR: '{norm_path}' is a directory. Use list_directory."

    # Enforce read caps
    if total_reads_ref[0] >= MAX_READ_CALLS_TOTAL:
        return (f"READ LIMIT REACHED: maximum {MAX_READ_CALLS_TOTAL} read_file calls "
                "per scan has been hit. Use search_code or run_analyzer for remaining coverage.")
    if call_counts.get(norm_path, 0) >= MAX_READS_PER_PATH:
        return f"LIMIT: '{norm_path}' has already been read {MAX_READS_PER_PATH} times this scan."

    call_counts[norm_path] = call_counts.get(norm_path, 0) + 1
    total_reads_ref[0] += 1

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except Exception as e:
        return f"ERROR reading file: {e}"

    start_line = max(1, start_line)
    if end_line is None:
        end_line = start_line + MAX_LINES_PER_READ - 1
    end_line = min(end_line, start_line + MAX_LINES_PER_READ - 1, len(all_lines))

    selected = all_lines[start_line - 1: end_line]
    out_lines = [f"{start_line + i:4d} | {line.rstrip()}"
                 for i, line in enumerate(selected)]

    header = f"# {norm_path}  (lines {start_line}–{end_line} of {len(all_lines)})\n"
    footer = ""
    if end_line < len(all_lines):
        footer = f"\n… ({len(all_lines) - end_line} more lines — call read_file again with start_line={end_line + 1})"

    return header + "\n".join(out_lines) + footer


def _search_code(pattern: str, file_glob: str, max_results: int,
                 repo_path: str) -> str:
    """Regex-search across files matching file_glob; return matching lines."""
    max_results = min(max_results, MAX_SEARCH_RESULTS)
    glob_pattern = file_glob or "**/*.py"

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"ERROR: invalid regex pattern: {e}"

    # Normalise glob: strip leading **/ for fnmatch basename match
    matches: List[str] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            if should_skip_path(fpath):
                continue
            rel = os.path.relpath(fpath, repo_path).replace("\\", "/")
            # Check glob match against relative path or just the filename
            glob_norm = glob_pattern.lstrip("*").lstrip("/")
            if not fnmatch.fnmatch(rel, glob_pattern) and not fnmatch.fnmatch(fname, glob_norm):
                # Also try matching just the extension pattern like *.py
                ext_pat = glob_pattern.lstrip("*")
                if not fname.endswith(ext_pat.lstrip(".")):
                    continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if compiled.search(line):
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                break
            except Exception:
                continue
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches found for pattern '{pattern}' in files matching '{glob_pattern}'."

    result = "\n".join(matches)
    if len(matches) >= max_results:
        result += f"\n… (capped at {max_results} results — refine your pattern or glob)"
    return result


def _run_analyzer(name: str, repo_path: str, analysis_root: str) -> str:
    """Run a built-in analyzer and return its report text."""
    out_file = os.path.join(analysis_root, f"agent_sub_{name}.txt")
    os.makedirs(analysis_root, exist_ok=True)

    try:
        if name == "mcp_flow":
            from analyzers.mcp_flow_analyzer import analyze_mcp_flow
            repo_name = os.path.basename(repo_path.rstrip("/\\"))
            analyze_mcp_flow(repo_path, repo_name, analysis_root)
            report_path = os.path.join(analysis_root, "mcp_flow_analysis.txt")
        elif name == "network":
            from analyzers.network_analysis import analyze_network
            repo_name = os.path.basename(repo_path.rstrip("/\\"))
            analyze_network(name=repo_name, repo_path=repo_path, analysis_path=out_file)
            report_path = out_file
        elif name == "auth":
            from analyzers.auth_analyzer import analyze_auth
            analyze_auth(repo_path, out_file)
            report_path = out_file
        elif name == "sse":
            from analyzers.sse_analyzer import analyze_sse
            analyze_sse(repo_path, out_file)
            report_path = out_file
        elif name == "bandit":
            from analyzers.bandit_analyzer import analyze_with_bandit
            analyze_with_bandit(repo_path, out_file)
            report_path = out_file
        else:
            return f"ERROR: unknown analyzer '{name}'. Valid: mcp_flow, network, auth, sse, bandit"
    except Exception as e:
        return f"ERROR running analyzer '{name}': {type(e).__name__}: {e}"

    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except Exception:
        return f"Analyzer '{name}' ran but its report could not be read."

    if len(text) > MAX_ANALYZER_CHARS:
        head = text[:int(MAX_ANALYZER_CHARS * 0.75)]
        tail = text[-int(MAX_ANALYZER_CHARS * 0.25):]
        text = head + "\n\n...[TRUNCATED — report exceeds display limit]...\n\n" + tail

    return text or f"(Analyzer '{name}' produced an empty report.)"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, tool_input: Dict[str, Any],
                   repo_path: str, analysis_root: str,
                   call_counts: Dict[str, int],
                   total_reads_ref: List[int]) -> str:
    """Route a tool_use block to the appropriate implementation."""
    if name == "list_directory":
        return _list_directory(
            path      = tool_input.get("path", ""),
            recursive = bool(tool_input.get("recursive", False)),
            repo_path = repo_path,
        )
    elif name == "read_file":
        return _read_file(
            path            = tool_input.get("path", ""),
            start_line      = int(tool_input.get("start_line", 1)),
            end_line        = tool_input.get("end_line"),          # None → auto
            repo_path       = repo_path,
            call_counts     = call_counts,
            total_reads_ref = total_reads_ref,
        )
    elif name == "search_code":
        return _search_code(
            pattern     = tool_input.get("pattern", ""),
            file_glob   = tool_input.get("file_glob", "**/*.py"),
            max_results = int(tool_input.get("max_results", 25)),
            repo_path   = repo_path,
        )
    elif name == "run_analyzer":
        return _run_analyzer(
            name          = tool_input.get("name", ""),
            repo_path     = repo_path,
            analysis_root = analysis_root,
        )
    elif name == "report_findings":
        # Handled in the main loop; we still return an ack so the message is valid
        return "Findings received. Audit complete."
    else:
        return f"ERROR: unknown tool '{name}'."


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_agent_report(
    repo_name:    str,
    findings:     List[Dict],
    summary:      str,
    stats:        Dict[str, Any],   # iterations, tool_call_counts
    output_path:  str,
) -> int:
    """Write agent_analysis.txt and return the issue count (excludes INFO)."""
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    findings_sorted = sorted(findings,
                             key=lambda f: sev_order.get(f.get("severity", "INFO"), 9))

    sev_counts: Dict[str, int] = {}
    for f in findings_sorted:
        s = f.get("severity", "INFO")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    issue_count = sum(v for k, v in sev_counts.items() if k != "INFO")

    # Build tool-call summary string
    tc = stats.get("tool_calls", {})
    tc_parts = [f"{t} ×{n}" for t, n in sorted(tc.items()) if n > 0]
    tc_str = ", ".join(tc_parts) if tc_parts else "none"

    lines: List[str] = [
        f"MCProbe Autonomous Agent Analysis — {repo_name}",
        "=" * 60,
        f"Model      : {stats.get('model', 'claude')}",
        f"Iterations : {stats.get('iterations', 0)}  |  "
        f"Tool calls : {tc_str}",
        "",
    ]

    if not findings_sorted:
        lines += [
            "━━━ FINDINGS ━━━",
            "",
            "No vulnerabilities found.",
            "",
        ]
    else:
        lines += ["━━━ FINDINGS ━━━", ""]
        for f in findings_sorted:
            sev   = f.get("severity", "?")
            title = f.get("title", "(untitled)")
            fpath = f.get("file", "")
            lineno = f.get("line", 0)
            loc   = f"{fpath}:{lineno}" if fpath and lineno else (fpath or "")
            lines.append(f"[{sev}] {title}")
            if loc:
                lines.append(f"  File           : {loc}")
            lines.append(f"  Description    : {f.get('description', '')}")
            lines.append(f"  Recommendation : {f.get('recommendation', '')}")
            lines.append("")

    lines += ["━━━ SUMMARY ━━━", ""]
    lines.append(summary or "(No summary provided.)")
    lines += [
        "",
        "━━━ STATISTICS ━━━",
        f"Total findings : {len(findings_sorted)}  "
        f"(CRITICAL: {sev_counts.get('CRITICAL', 0)} · "
        f"HIGH: {sev_counts.get('HIGH', 0)} · "
        f"MEDIUM: {sev_counts.get('MEDIUM', 0)} · "
        f"LOW: {sev_counts.get('LOW', 0)} · "
        f"INFO: {sev_counts.get('INFO', 0)})",
        f"Actionable     : {issue_count}",
    ]
    if stats.get("truncated"):
        lines += [
            "",
            "NOTE: Scan was cut short — maximum iterations reached before report_findings was called.",
            "The findings above represent partial coverage.",
        ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return issue_count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent_scan(state: dict) -> Tuple[str, int]:
    """
    Run the autonomous Claude agent scanner.
    Called from orchestrator.use_ai() when backend == "claude-agent".

    Returns (report_text_for_ai_analysis, issue_count).
    The report is also written to <analysis_root>/agent_analysis.txt.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package is required for Claude Agent mode. "
            "Install it with:  pip install anthropic"
        )

    _use_os = state.get("use_os_env", False)
    api_key = state.get("anthropic_api_key") or (os.getenv("ANTHROPIC_API_KEY", "") if _use_os else "")
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key found. Set it via 'Set API Keys…' in the GUI, "
            "provide it in .env, or pass --use-os-env to read from environment variables."
        )

    base_url = state.get("anthropic_base_url") or (os.getenv("ANTHROPIC_BASE_URL") if _use_os else None) or None
    repo_path     = state.get("repo_local_path", "")
    analysis_root = state.get("analysis_root",   "")
    repo_name     = state.get("name", os.path.basename(repo_path.rstrip("/\\")) or "repo")
    model         = os.getenv("MCPROBE_CLAUDE_MODEL", "claude-sonnet-4-6")
    output_path   = os.path.join(analysis_root, "agent_analysis.txt")

    os.makedirs(analysis_root, exist_ok=True)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**client_kwargs)

    system  = SYSTEM_PROMPT.format(repo_name=repo_name)
    initial = (
        f"Please perform a thorough security audit of the MCP server repository: {repo_name}\n\n"
        f"The repository is available for inspection via the provided tools. "
        f"Start by listing the directory structure, then investigate systematically."
    )

    messages: List[Dict] = [{"role": "user", "content": initial}]

    # Mutable state threaded through tool dispatch
    call_counts: Dict[str, int] = {}   # file path → number of reads
    total_reads: List[int]       = [0] # wrapped in list so it's mutable in closures

    # Telemetry
    tool_call_counts: Dict[str, int] = {
        "list_directory": 0, "read_file": 0,
        "search_code": 0,    "run_analyzer": 0,
    }
    iterations  = 0
    truncated   = False
    findings:  List[Dict] = []
    summary    = ""
    final_text = ""

    dprint(f"[AGENT] Starting autonomous scan of {repo_name} (model={model})")

    while iterations < MAX_ITERATIONS:
        try:
            resp = client.messages.create(
                model      = model,
                max_tokens = 4096,
                system     = system,
                tools      = TOOL_SCHEMAS,
                messages   = messages,
            )
        except Exception as e:
            dprint(f"[AGENT] API error on iteration {iterations + 1}: {e}")
            raise

        iterations += 1
        dprint(f"[AGENT] Iteration {iterations}/{MAX_ITERATIONS} — "
               f"stop_reason={resp.stop_reason}")

        # Append assistant turn to conversation
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            # Claude finished without calling report_findings — extract any text
            for block in resp.content:
                if hasattr(block, "text"):
                    final_text = block.text
            break

        if resp.stop_reason != "tool_use":
            break

        # Process tool calls
        tool_results = []
        done = False
        for block in resp.content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue

            tool_name  = block.name
            tool_input = block.input or {}

            # Track telemetry
            if tool_name in tool_call_counts:
                tool_call_counts[tool_name] += 1

            dprint(f"[AGENT]   → {tool_name}({list(tool_input.keys())})")

            result_text = _dispatch_tool(
                name            = tool_name,
                tool_input      = tool_input,
                repo_path       = repo_path,
                analysis_root   = analysis_root,
                call_counts     = call_counts,
                total_reads_ref = total_reads,
            )

            # Capture findings if this is the report_findings call
            if tool_name == "report_findings":
                findings = tool_input.get("findings", [])
                summary  = tool_input.get("summary", "")
                done     = True

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_text,
            })

        messages.append({"role": "user", "content": tool_results})

        if done:
            dprint(f"[AGENT] report_findings called — {len(findings)} finding(s)")
            break
    else:
        truncated = True
        dprint(f"[AGENT] Max iterations ({MAX_ITERATIONS}) reached without report_findings")

    stats = {
        "model":      model,
        "iterations": iterations,
        "tool_calls": tool_call_counts,
        "truncated":  truncated,
    }

    issue_count = _write_agent_report(
        repo_name   = repo_name,
        findings    = findings,
        summary     = summary,
        stats       = stats,
        output_path = output_path,
    )

    dprint(f"[AGENT] Done — {issue_count} actionable finding(s), report → {output_path}")

    # Also build a plain-text summary for the orchestrator's ai_analysis field
    if findings:
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        top = sorted(findings, key=lambda f: sev_order.get(f.get("severity", "INFO"), 9))[:5]
        bullets = "\n".join(
            f"- [{f['severity']}] {f['title']} ({f.get('file', '')}:{f.get('line', 0)})"
            for f in top
        )
        ai_text = (
            f"## Autonomous Agent Scan — {repo_name}\n\n"
            f"{summary}\n\n"
            f"**Top findings ({len(findings)} total):**\n{bullets}\n\n"
            f"See agent_analysis.txt for full details."
        )
    else:
        ai_text = (
            f"## Autonomous Agent Scan — {repo_name}\n\n"
            f"{summary or 'No vulnerabilities found.'}\n\n"
            "agent_analysis.txt: no actionable findings."
        )

    return ai_text, issue_count
