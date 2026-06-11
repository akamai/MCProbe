"""
MCProbe — Authentication & Authorization Analyzer
Detects auth/authz security issues:
  - Hardcoded credentials and secrets
  - Missing authentication decorators on routes
  - JWT misuse (signature bypass, missing validation)
  - Authorization bypass patterns (wildcard roles, missing permission checks)
  - Insecure session / token handling
"""
import os
import re
import ast
from typing import List, Dict

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Hardcoded credential patterns
HARDCODED_CRED_PATTERNS = [
    (r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]{3,}['\"]", "Hardcoded password"),
    (r"(?i)(secret|secret_key|app_secret)\s*=\s*['\"][^'\"]{6,}['\"]", "Hardcoded secret"),
    (r"(?i)(api_key|apikey|access_key|auth_token|bearer)\s*=\s*['\"][^'\"]{6,}['\"]", "Hardcoded API key / token"),
    (r"(?i)(private_key)\s*=\s*['\"][^'\"]{6,}['\"]", "Hardcoded private key"),
    # JWT-shaped hardcoded tokens
    (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "Hardcoded JWT token"),
    # AWS-style keys
    (r"(?i)aws_secret_access_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded AWS secret"),
    (r"AKIA[0-9A-Z]{16}", "Hardcoded AWS access key"),
]

# ---------------------------------------------------------------------------
# Well-known fake / placeholder credentials published in official docs.
# Any match whose full text contains one of these substrings is skipped.
# ---------------------------------------------------------------------------
KNOWN_FAKE_CREDENTIALS: frozenset = frozenset({
    # AWS documentation canonical example keys (used in every AWS getting-started guide)
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    # Generic obvious placeholders
    "your-api-key", "your_api_key", "your-secret", "your_secret",
    "YOUR_API_KEY", "YOUR_SECRET_KEY", "YOUR_ACCESS_KEY",
    "INSERT_KEY_HERE", "REPLACE_WITH_YOUR",
    "<api_key>", "<secret>", "<token>",
    "xxxxxxxxxxxx", "000000000000",
    "changeme", "change_me", "placeholder",
    "example_key", "example_secret", "example_token",
    "test_key", "test_secret", "test_token",
    "dummy_key", "dummy_secret",
    "fake_key", "fake_secret",
    "my_api_key", "my_secret_key",
    # Common CI/docs placeholders
    "sk-XXXXXXXX", "sk-xxxx",
})

# File-name patterns that indicate validation / format-checking utilities
# where fake credentials are intentional.
_VALIDATION_FILE_RE = re.compile(
    r"(validation|validate|validator|format_check|format_test"
    r"|schema|model_check|credential_format)\w*\.py$",
    re.IGNORECASE,
)

# JWT misuse
JWT_MISUSE_PATTERNS = [
    (r"jwt\.decode\s*\([^)]*verify\s*=\s*False", "JWT signature verification disabled"),
    (r"jwt\.decode\s*\([^)]*algorithms\s*=\s*\[\s*['\"]none['\"]\s*\]", "JWT 'none' algorithm accepted"),
    (r"PyJWT.*algorithms.*none", "JWT 'none' algorithm"),
    (r"verify_signature\s*=\s*False", "JWT signature verification disabled"),
    # JS/TS
    (r"jwt\.verify\s*\([^,]+,\s*null\s*\)", "JWT verified with null secret"),
    (r"jsonwebtoken.*algorithms.*none", "JWT 'none' algorithm (JS)"),
]

# Missing auth decorators on Python route definitions
# Matches @app.route or @router.X without a following @login_required / @require_auth etc.
PYTHON_AUTH_DECORATOR_RE = re.compile(
    r"@(?:app|router|blueprint)\.(route|get|post|put|patch|delete)\s*\([^)]*\)"
)
AUTH_DECORATOR_RE = re.compile(
    r"@(?:login_required|require_auth|authenticated|requires_auth|jwt_required"
    r"|token_required|permission_required|requires_permission|auth\.login_required)"
)

# Authorization bypass patterns
AUTHZ_BYPASS_PATTERNS = [
    (r"(?i)role\s*==\s*['\"]?\*['\"]?", "Wildcard role check — anyone passes"),
    (r"(?i)if\s+True\s*:", "Hardcoded True auth guard — always passes"),
    (r"(?i)#\s*TODO.*auth", "TODO comment near auth logic — possibly unimplemented"),
    (r"(?i)skip_auth\s*=\s*True", "Auth skip flag enabled"),
    (r"(?i)bypass_auth\s*=\s*True", "Auth bypass flag enabled"),
    # JS patterns
    (r"isAdmin\s*=\s*true", "Hardcoded admin flag"),
    (r"authenticated\s*=\s*true", "Hardcoded authenticated flag"),
]

# Insecure session / cookie flags
SESSION_PATTERNS = [
    (r"(?i)secure\s*=\s*False", "Cookie secure flag disabled"),
    (r"(?i)httponly\s*=\s*False", "Cookie HttpOnly flag disabled"),
    (r"(?i)samesite\s*=\s*['\"]?None['\"]?.*secure\s*=\s*False", "SameSite=None without Secure"),
    (r"(?i)SESSION_COOKIE_SECURE\s*=\s*False", "Flask session cookie not secured"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"}
from helpers import EXCLUDE_DIRS, should_skip_path


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


def _lineno(content: str, pos: int) -> int:
    return content[:pos].count("\n") + 1


# ---------------------------------------------------------------------------
# Regex-based checks (Python + JS/TS)
# ---------------------------------------------------------------------------

def _is_fake_credential(match_text: str, filepath: str) -> bool:
    """Return True if this match is a known placeholder and should be skipped."""
    # Check against the known-fake denylist (case-insensitive substring search)
    upper = match_text.upper()
    for fake in KNOWN_FAKE_CREDENTIALS:
        if fake.upper() in upper:
            return True
    # Skip validation / format-checking utility files
    if _VALIDATION_FILE_RE.search(os.path.basename(filepath)):
        return True
    return False


def _check_patterns(content: str, filepath: str,
                    patterns: List[tuple], issue_type: str) -> List[Dict]:
    issues = []
    for pat, detail in patterns:
        for m in re.finditer(pat, content):
            # Skip lines that are comments (crude check)
            line_start = content.rfind("\n", 0, m.start()) + 1
            line_text = content[line_start: content.find("\n", m.start())]
            stripped = line_text.lstrip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            match_text = m.group(0).strip()
            # Skip well-known placeholder / example credentials
            if issue_type == "HARDCODED_CREDENTIAL" and _is_fake_credential(match_text, filepath):
                continue
            issues.append({
                "type": issue_type,
                "file": filepath,
                "line": _lineno(content, m.start()),
                "match": match_text[:120],
                "detail": detail,
                "severity": "HIGH",
            })
    return issues


# ---------------------------------------------------------------------------
# Python-specific: missing auth decorator on routes
# ---------------------------------------------------------------------------

def _check_missing_auth_decorator_python(content: str, filepath: str) -> List[Dict]:
    if not filepath.endswith(".py"):
        return []
    issues = []
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if PYTHON_AUTH_DECORATOR_RE.search(line):
            # Look at the next few decorator lines before the `def`
            block = "\n".join(lines[max(0, i - 3): i + 4])
            if not AUTH_DECORATOR_RE.search(block):
                issues.append({
                    "type": "MISSING_AUTH_DECORATOR",
                    "file": filepath,
                    "line": i + 1,
                    "match": line.strip(),
                    "detail": "Route defined without an auth decorator — endpoint may be publicly accessible",
                    "severity": "MEDIUM",
                })
    return issues


# ---------------------------------------------------------------------------
# Python AST: detect functions that accept 'token'/'user'/'auth' params
# but never validate them
# ---------------------------------------------------------------------------

def _check_unvalidated_auth_param_python(content: str, filepath: str) -> List[Dict]:
    if not filepath.endswith(".py"):
        return []
    issues = []
    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError:
        return []

    auth_param_names = {"token", "auth", "authorization", "user_token", "api_key", "secret"}
    validation_calls = {"verify", "validate", "decode", "check", "authenticate",
                        "is_valid", "authorized", "permission"}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {arg.arg.lower() for arg in node.args.args}
        auth_params = params & auth_param_names
        if not auth_params:
            continue
        # Check if any call within the body mentions validation keywords
        body_src = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if not any(v in body_src.lower() for v in validation_calls):
            issues.append({
                "type": "UNVALIDATED_AUTH_PARAM",
                "file": filepath,
                "line": node.lineno,
                "match": f"def {node.name}(..., {', '.join(auth_params)}, ...)",
                "detail": "Function receives auth/token param but contains no obvious validation call",
                "severity": "MEDIUM",
            })
    return issues


# ---------------------------------------------------------------------------
# Repo-wide: no authentication at all
# ---------------------------------------------------------------------------

# Imports that suggest auth is at least present somewhere in the project
_AUTH_IMPORT_SIGNALS = [
    "flask_login", "flask_jwt", "flask_security",
    "jwt", "authlib", "itsdangerous",
    "passlib", "bcrypt", "argon2",
    "django.contrib.auth", "rest_framework.authentication",
    "fastapi.security", "starlette.authentication",
    "pyjwt", "python-jose", "jose",
    "oauth", "oidc", "saml",
    "login_required", "require_auth", "authenticated",
    "jwt_required", "token_required", "permission_required",
    "verify_token", "validate_token", "decode_token",
    "authenticate", "authorization",
]

_ROUTE_SIGNALS = [
    r"@(?:app|router|blueprint)\.(route|get|post|put|patch|delete)\s*\(",
    r"app\.listen\s*\(",
    r"server\.listen\s*\(",
    r"router\.(get|post|put|patch|delete)\s*\(",
]


def _check_no_auth_in_repo(repo_path: str) -> List[Dict]:
    """
    If the repo defines HTTP routes but contains zero authentication-related
    code anywhere, emit a single HIGH-severity repo-level finding.
    """
    has_routes = False
    has_auth   = False

    for filepath in _iter_source_files(repo_path):
        content = _read_file(filepath)
        if not content:
            continue
        content_lower = content.lower()

        # Check for route definitions
        if not has_routes:
            for pat in _ROUTE_SIGNALS:
                if re.search(pat, content, re.IGNORECASE):
                    has_routes = True
                    break

        # Check for any auth signal anywhere in the file
        if not has_auth:
            for signal in _AUTH_IMPORT_SIGNALS:
                if signal.lower() in content_lower:
                    has_auth = True
                    break

        if has_routes and has_auth:
            break  # both found — no need to continue scanning

    if has_routes and not has_auth:
        return [{
            "type":     "NO_AUTHENTICATION",
            "file":     repo_path,
            "line":     0,
            "match":    "(repo-wide)",
            "detail":   (
                "HTTP routes are defined but no authentication library, decorator, "
                "or token-validation logic was detected anywhere in this repository. "
                "All endpoints may be publicly accessible."
            ),
            "severity": "HIGH",
        }]
    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_auth(repo_path: str, output_path: str) -> int:
    """
    Scan repo for authentication & authorization issues.
    Writes a report to output_path and returns the issue count.
    """
    all_issues: List[Dict] = []

    # Repo-wide check first
    all_issues.extend(_check_no_auth_in_repo(repo_path))

    for filepath in _iter_source_files(repo_path):
        content = _read_file(filepath)
        if not content:
            continue

        all_issues.extend(_check_patterns(content, filepath, HARDCODED_CRED_PATTERNS, "HARDCODED_CREDENTIAL"))
        all_issues.extend(_check_patterns(content, filepath, JWT_MISUSE_PATTERNS, "JWT_MISUSE"))
        all_issues.extend(_check_patterns(content, filepath, AUTHZ_BYPASS_PATTERNS, "AUTHZ_BYPASS"))
        all_issues.extend(_check_patterns(content, filepath, SESSION_PATTERNS, "INSECURE_SESSION"))
        all_issues.extend(_check_missing_auth_decorator_python(content, filepath))
        all_issues.extend(_check_unvalidated_auth_param_python(content, filepath))

    # Deduplicate
    seen = set()
    deduped = []
    for issue in all_issues:
        key = (issue["type"], issue["file"], issue["line"])
        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    deduped.sort(key=lambda x: (severity_order.get(x["severity"], 9), x["file"], x["line"]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if not deduped:
            f.write("No authentication or authorization issues found.\n")
        else:
            f.write(f"{len(deduped)} auth/authz issue(s) found:\n\n")
            for issue in deduped:
                rel = os.path.relpath(issue["file"], repo_path)
                f.write(
                    f"[{issue['severity']}] {issue['type']}\n"
                    f"  File  : {rel}:{issue['line']}\n"
                    f"  Match : {issue['match']}\n"
                    f"  Detail: {issue['detail']}\n\n"
                )

    return len(deduped)
