"""
MCProbe — SSE/Streaming Communication Analyzer

Detects security issues in MCP servers using Server-Sent Events transport:
  - SSE endpoints without authentication (HIGH)
  - Dangerous CORS configurations: wildcard + credentials, Origin reflection (HIGH)
  - /messages POST handlers without session-ID binding (MEDIUM)
  - SSE endpoints lacking Origin validation — DNS rebinding risk (MEDIUM)
  - Generic SSRF via streaming requests (MEDIUM)
  - Unvalidated redirect targets in streaming handlers (HIGH)
"""
import os
import re
from typing import List, Dict, Tuple

from helpers import EXCLUDE_DIRS, should_skip_path

# ---------------------------------------------------------------------------
# Patterns: SSE detection (gate)
# ---------------------------------------------------------------------------

SSE_MARKERS_RE = re.compile(
    r"text/event-stream"
    r"|EventSourceResponse"
    r"|sse_starlette"
    r"|flask_sse"
    r"|SSEHandler"
    r"|new\s+EventSource\s*\("
    r"|yield\s+['\"]data:"
    r"|res\.write\s*\(\s*['\"]data:",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Patterns: auth markers (used to decide if an SSE endpoint has auth)
# ---------------------------------------------------------------------------

# Decorators + dependencies + inline calls. Order doesn't matter; we just
# substring-search the function source for any of these.
AUTH_MARKERS = [
    # Python decorators
    "@require_auth", "@login_required", "@authenticated", "@requires_auth",
    "@protected", "@authorize", "@auth_required", "@token_required",
    # FastAPI Depends(...) auth dependencies
    "Depends(get_jwt_token)", "Depends(verify_token)", "Depends(get_current_user)",
    "Depends(authenticate)", "Depends(require_auth)", "Depends(auth)",
    "Depends(get_user)", "Depends(verify_jwt)",
    # Inline calls / attribute access
    "verify_jwt(", "decode_jwt(", "verify_token(", "verify_session(",
    "current_user", "request.user", "req.user",
    "Authorization", "x-api-key", "X-API-Key",
    # Express / Node middleware
    "passport.authenticate", "requireAuth", "ensureAuthenticated",
    "checkAuth(", "authMiddleware", "isAuthenticated(",
    "jwt.verify(", "jsonwebtoken",
]

# ---------------------------------------------------------------------------
# Patterns: CORS misconfig
# ---------------------------------------------------------------------------

CORS_WILDCARD_RE = re.compile(
    r"allow_origins\s*=\s*\[\s*['\"]\*['\"]"          # FastAPI/Starlette
    r"|origins\s*[:=]\s*['\"]\*['\"]"                  # flask-cors
    r"|origin\s*:\s*['\"]\*['\"]"                      # express cors options
    r"|Access-Control-Allow-Origin[^\n]*[:=]\s*['\"]\*['\"]",  # raw header
    re.IGNORECASE,
)

CORS_CREDENTIALS_RE = re.compile(
    r"allow_credentials\s*=\s*True"
    r"|credentials\s*[:=]\s*[Tt]rue"
    r"|supports_credentials\s*=\s*True"
    r"|Access-Control-Allow-Credentials[^\n]*[:=]\s*['\"]?true",
    re.IGNORECASE,
)

# Origin reflection: setting ACAO equal to req.headers.origin
CORS_REFLECTION_RE = re.compile(
    r"Access-Control-Allow-Origin[^\n]{0,80}"
    r"(?:request|req)\.headers"
    r"(?:\.get\s*\(\s*['\"]origin['\"]"
    r"|\[\s*['\"]origin['\"]\s*\]"
    r"|\.origin\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Patterns: /messages session-id reads + binding markers
# ---------------------------------------------------------------------------

MESSAGES_ROUTE_PY_RE = re.compile(
    r"@(?:\w+\.)?(?:post|route)\s*\(\s*['\"](/messages?(?:/[^'\"]*)?)['\"]",
    re.IGNORECASE,
)
MESSAGES_ROUTE_JS_RE = re.compile(
    r"(?:app|router)\.(?:post|use)\s*\(\s*['\"](/messages?(?:/[^'\"]*)?)['\"]",
    re.IGNORECASE,
)

SESSION_ID_READ_RE = re.compile(
    r"session_id\s*=\s*(?:request|req)\."
    r"(?:args|query|query_params|params|body|json|form)"
    r"|(?:request|req)\.(?:args|query|query_params|params)"
    r"\.get\s*\(\s*['\"]session_id['\"]"
    r"|(?:request|req)\.(?:args|query|query_params|params)\.session_id"
    r"|query_params\.get\s*\(\s*['\"]session_id['\"]",
    re.IGNORECASE,
)

SESSION_BIND_MARKERS = [
    "session.user_id", "current_user", "request.remote_addr",
    "request.client.host", "session.ip", "session.owner",
    "verify_session_binding", "request.cookies.get", "req.cookies",
    "session.user", "owner_id", "session_owner",
]

# ---------------------------------------------------------------------------
# Patterns: Origin validation (light DNS-rebinding check)
# ---------------------------------------------------------------------------

ORIGIN_VALIDATION_RE = re.compile(
    r"(?:request|req)\.headers\.get\s*\(\s*['\"]origin['\"]"
    r"|(?:request|req)\.headers\[\s*['\"]origin['\"]\s*\]"
    r"|(?:request|req)\.headers\.origin\b"
    r"|trusted_origins"
    r"|allowed_origins"
    r"|ALLOWED_ORIGINS"
    r"|ORIGIN_WHITELIST"
    r"|origin_allowlist",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Patterns: SSRF + redirect (unchanged from previous version)
# ---------------------------------------------------------------------------

SSRF_PATTERNS = [
    r"requests\.(get|post)\s*\([^)]*stream\s*=\s*True[^)]*\)",
    r"httpx\.(get|post|stream)\s*\([^)]*stream\s*=\s*True[^)]*\)",
    r"new\s+EventSource\s*\(\s*(?![\'\"]http)",
]

REDIRECT_PATTERNS = [
    r"redirect\s*\(\s*request\.(args|params|query|form|data|json)",
    r"return\s+redirect\s*\(\s*(?:url|target|next|dest|location)\s*\)",
    r"res\.redirect\s*\(\s*req\.(query|body|params)\.",
]

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"}


def _iter_source_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if os.path.splitext(fname)[1] in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(root, fname)
                if not should_skip_path(fpath):
                    yield fpath


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------

def _extract_python_function(content: str, signal_pos: int) -> Tuple[int, str]:
    """
    Extract the source of the Python function relevant to signal_pos.

    - If signal_pos is on a decorator line, find the function below the
      decorator chain.
    - Otherwise (signal inside body), find the OUTERMOST enclosing function
      (so nested helpers like inner generators don't shadow the route handler).

    Returns (start_line_1based, source) or (0, "") if not found.
    """
    lines = content.split("\n")
    pos_line = content[:signal_pos].count("\n")
    if pos_line >= len(lines):
        return 0, ""

    signal_stripped = lines[pos_line].lstrip()
    def_line = -1

    if signal_stripped.startswith("@"):
        # Decorator case: walk forward through @ / blank lines until we hit def
        i = pos_line
        while i < len(lines):
            stripped = lines[i].lstrip()
            if stripped.startswith("@") or stripped == "":
                i += 1
                continue
            if stripped.startswith("def ") or stripped.startswith("async def "):
                def_line = i
            break
        if def_line < 0:
            return 0, ""
        deco_start = pos_line
        # Pull additional decorators above (in case signal hit a non-first one)
        while deco_start > 0 and (lines[deco_start - 1].lstrip().startswith("@")
                                    or lines[deco_start - 1].strip() == ""):
            deco_start -= 1
    else:
        # Body case: find the outermost enclosing function. We want a def at
        # an indent STRICTLY LESS than the signal line's indent (so sibling
        # nested helpers like `async def gen()` at the same indent are skipped).
        signal_indent = len(lines[pos_line]) - len(lines[pos_line].lstrip())
        cur_min_indent = signal_indent
        for i in range(pos_line, -1, -1):
            stripped = lines[i].lstrip()
            if not (stripped.startswith("def ") or stripped.startswith("async def ")):
                continue
            def_indent = len(lines[i]) - len(lines[i].lstrip())
            if def_indent < cur_min_indent:
                def_line = i
                cur_min_indent = def_indent
                if def_indent == 0:
                    break
        if def_line < 0:
            return 0, ""
        deco_start = def_line
        while deco_start > 0 and (lines[deco_start - 1].lstrip().startswith("@")
                                    or lines[deco_start - 1].strip() == ""):
            deco_start -= 1

    # Find end of function: next non-empty line at same-or-lower indent that
    # begins a new def/class/decorator.
    base_indent = len(lines[def_line]) - len(lines[def_line].lstrip())
    end_line = def_line + 1
    while end_line < len(lines):
        line = lines[end_line]
        stripped = line.strip()
        if stripped == "":
            end_line += 1
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= base_indent and (
            stripped.startswith("def ")
            or stripped.startswith("async def ")
            or stripped.startswith("class ")
            or stripped.startswith("@")
        ):
            break
        end_line += 1

    return deco_start + 1, "\n".join(lines[deco_start:end_line])


def _extract_js_window(content: str, signal_pos: int, window: int = 15) -> Tuple[int, str]:
    """JS doesn't lend itself to clean function extraction via regex — return
    a window of lines around the signal."""
    lines = content.split("\n")
    pos_line = content[:signal_pos].count("\n")
    start = max(0, pos_line - window)
    end = min(len(lines), pos_line + window + 1)
    return start + 1, "\n".join(lines[start:end])


def _extract_handler(content: str, signal_pos: int, filepath: str) -> Tuple[int, str]:
    if filepath.endswith(".py"):
        return _extract_python_function(content, signal_pos)
    return _extract_js_window(content, signal_pos)


def _has_marker(src: str, markers: List[str]) -> bool:
    low = src.lower()
    return any(m.lower() in low for m in markers)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_sse_auth(content: str, filepath: str) -> List[Dict]:
    """SSE handler functions that don't reference any auth marker."""
    issues = []
    seen_starts = set()
    for m in re.finditer(r"text/event-stream|EventSourceResponse|yield\s+['\"]data:",
                          content, re.IGNORECASE):
        start_line, src = _extract_handler(content, m.start(), filepath)
        if not src or start_line in seen_starts:
            continue
        seen_starts.add(start_line)

        if _has_marker(src, AUTH_MARKERS):
            continue

        issues.append({
            "type": "SSE_NO_AUTH",
            "file": filepath,
            "line": start_line,
            "match": "SSE handler",
            "detail": (
                "SSE endpoint handler has no recognizable authentication wiring "
                "(no auth decorator, FastAPI Depends, or inline JWT/session check). "
                "Anyone reaching this endpoint can subscribe to the event stream."
            ),
            "severity": "HIGH",
        })
    return issues


def _check_cors_misconfig(content: str, filepath: str) -> List[Dict]:
    """Wildcard origin + credentials, or Origin-reflection ACAO."""
    issues = []

    # Pattern 1: wildcard origin + credentials nearby (same call / middleware block)
    for m in CORS_WILDCARD_RE.finditer(content):
        window = content[max(0, m.start() - 400): m.end() + 400]
        if CORS_CREDENTIALS_RE.search(window):
            lineno = content[: m.start()].count("\n") + 1
            issues.append({
                "type": "CORS_WILDCARD_WITH_CREDENTIALS",
                "file": filepath,
                "line": lineno,
                "match": m.group(0).strip()[:120],
                "detail": (
                    "CORS allow_origins='*' combined with credentials=true. "
                    "Browsers refuse this combination, but the misconfig signals "
                    "intent to widely accept cross-origin credentialed requests."
                ),
                "severity": "HIGH",
            })

    # Pattern 2: ACAO reflected from request Origin header
    for m in CORS_REFLECTION_RE.finditer(content):
        lineno = content[: m.start()].count("\n") + 1
        issues.append({
            "type": "CORS_ORIGIN_REFLECTION",
            "file": filepath,
            "line": lineno,
            "match": m.group(0).strip()[:120],
            "detail": (
                "Access-Control-Allow-Origin reflected from the request's Origin "
                "header without an explicit allowlist. Functionally equivalent to "
                "'*' for any caller."
            ),
            "severity": "HIGH",
        })

    return issues


def _check_messages_session_binding(content: str, filepath: str) -> List[Dict]:
    """MCP-style /messages POST handlers reading session_id without binding it."""
    issues = []
    is_py = filepath.endswith(".py")
    route_re = MESSAGES_ROUTE_PY_RE if is_py else MESSAGES_ROUTE_JS_RE

    seen_starts = set()
    for m in route_re.finditer(content):
        start_line, src = _extract_handler(content, m.start(), filepath)
        if not src or start_line in seen_starts:
            continue
        seen_starts.add(start_line)

        # Only care if the handler actually reads session_id from the request
        if not SESSION_ID_READ_RE.search(src):
            continue

        if _has_marker(src, SESSION_BIND_MARKERS):
            continue

        issues.append({
            "type": "MCP_MESSAGES_NO_SESSION_BIND",
            "file": filepath,
            "line": start_line,
            "match": f"POST {m.group(1)}",
            "detail": (
                "Handler reads session_id from the request but never binds it "
                "to source IP, cookie, or authenticated subject. Anyone who "
                "learns a valid session_id (logs, referer leaks) can inject "
                "messages into another user's stream."
            ),
            "severity": "MEDIUM",
        })
    return issues


def _check_origin_validation(content: str, filepath: str) -> List[Dict]:
    """Light DNS-rebinding check: SSE endpoint files with no Origin reference."""
    issues = []
    sse_m = SSE_MARKERS_RE.search(content)
    if not sse_m:
        return issues
    if ORIGIN_VALIDATION_RE.search(content):
        return issues

    lineno = content[: sse_m.start()].count("\n") + 1
    issues.append({
        "type": "SSE_NO_ORIGIN_VALIDATION",
        "file": filepath,
        "line": lineno,
        "match": "SSE endpoint",
        "detail": (
            "SSE endpoint defined here but no Origin header validation found "
            "in this file. Localhost MCP servers are vulnerable to DNS "
            "rebinding when Origin is not validated against an allowlist."
        ),
        "severity": "MEDIUM",
    })
    return issues


def _check_ssrf(content: str, filepath: str) -> List[Dict]:
    issues = []
    for pat in SSRF_PATTERNS:
        for m in re.finditer(pat, content, re.DOTALL):
            lineno = content[: m.start()].count("\n") + 1
            issues.append({
                "type": "SSRF_STREAM",
                "file": filepath,
                "line": lineno,
                "match": m.group(0).strip()[:120],
                "detail": "Streaming request — verify URL source is not user-controlled (SSRF risk)",
                "severity": "MEDIUM",
            })
    return issues


def _check_redirect(content: str, filepath: str) -> List[Dict]:
    issues = []
    for pat in REDIRECT_PATTERNS:
        for m in re.finditer(pat, content):
            lineno = content[: m.start()].count("\n") + 1
            issues.append({
                "type": "OPEN_REDIRECT",
                "file": filepath,
                "line": lineno,
                "match": m.group(0).strip(),
                "detail": "Redirect target may be user-controlled — validate against allowlist",
                "severity": "HIGH",
            })
    return issues


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


def analyze_sse(repo_path: str, output_path: str) -> int:
    """
    Scan the repository for SSE/streaming security issues.
    Writes a report to output_path and returns the issue count.
    """
    all_issues: List[Dict] = []

    for filepath in _iter_source_files(repo_path):
        content = _read_file(filepath)
        if not content:
            continue

        # Gate: skip files that don't touch streaming at all
        has_sse_signal = bool(SSE_MARKERS_RE.search(content))
        has_cors_signal = bool(CORS_WILDCARD_RE.search(content) or
                                CORS_REFLECTION_RE.search(content))
        has_messages_route = bool(
            MESSAGES_ROUTE_PY_RE.search(content) or MESSAGES_ROUTE_JS_RE.search(content)
        )

        if not (has_sse_signal or has_cors_signal or has_messages_route):
            continue

        if has_sse_signal:
            all_issues.extend(_check_sse_auth(content, filepath))
            all_issues.extend(_check_origin_validation(content, filepath))
            all_issues.extend(_check_ssrf(content, filepath))
            all_issues.extend(_check_redirect(content, filepath))

        if has_cors_signal:
            all_issues.extend(_check_cors_misconfig(content, filepath))

        if has_messages_route:
            all_issues.extend(_check_messages_session_binding(content, filepath))

    # Deduplicate by (type, file, line)
    seen = set()
    deduped = []
    for issue in all_issues:
        key = (issue["type"], issue["file"], issue["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if not deduped:
            f.write("No SSE/streaming security issues found.\n")
        else:
            f.write(f"{len(deduped)} SSE/streaming issue(s) found:\n\n")
            sorted_issues = sorted(
                deduped,
                key=lambda x: (_SEV_ORDER.get(x["severity"], 9), x["file"], x["line"]),
            )
            for issue in sorted_issues:
                rel = os.path.relpath(issue["file"], repo_path)
                f.write(
                    f"[{issue['severity']}] {issue['type']}\n"
                    f"  File : {rel}:{issue['line']}\n"
                    f"  Match: {issue['match']}\n"
                    f"  Note : {issue['detail']}\n\n"
                )

    return len(deduped)
