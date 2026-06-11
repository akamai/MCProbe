"""
MCProbe — CFG-Based Command Injection Analyzer v2
Improved over code_analyzer.py with:
  - Deeper interprocedural taint tracking (arg → dangerous sink)
  - SQL injection detection via string concatenation / f-strings passed to DB execute
  - Expanded dangerous sink list (pickle, yaml.load, marshal, ctypes)
  - JS/TS token-walk for multi-line expression handling
  - Reports include taint source → sink call chain, not just the sink line

The original code_analyzer.py is left untouched for side-by-side comparison.
"""
import ast
import os
import re
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Python dangerous sinks: module -> {method, ...}
DANGEROUS_SINKS_PY: Dict[str, Set[str]] = {
    "os":         {"system", "popen", "execl", "execle", "execlp", "execv", "execve",
                   "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnv",
                   "spawnve", "spawnvp"},
    "subprocess": {"run", "call", "check_call", "check_output", "Popen"},
    "builtins":   {"eval", "exec", "compile", "__import__"},
    "requests":   {"get", "post", "put", "delete", "patch", "request", "head"},
    "pickle":     {"loads", "load"},
    "marshal":    {"loads", "load"},
    "yaml":       {"load"},          # yaml.safe_load is OK; yaml.load without Loader is not
    "ctypes":     {"cdll", "CDLL", "WinDLL", "windll"},
    "open":       {"open"},          # bare open() — file access sink
}

# SQL execute patterns: attribute calls we consider DB sinks
SQL_EXECUTE_METHODS = {"execute", "executemany", "executescript", "raw", "query"}

from helpers import EXCLUDE_DIRS, should_skip_path

# ---------------------------------------------------------------------------
# Python AST helpers
# ---------------------------------------------------------------------------

def _get_attr_chain(node: ast.expr) -> str:
    """Return dotted name like 'os.path.join' or '' if not resolvable."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_attr_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_string_concat_or_fstring(node: ast.expr) -> bool:
    """Return True if the node is built via string concatenation or f-string."""
    if isinstance(node, ast.JoinedStr):  # f-string
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return True
    if isinstance(node, ast.Call):
        func = _get_attr_chain(node.func)
        if func in ("str.format", "format") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "format"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Interprocedural taint tracker
# ---------------------------------------------------------------------------

class TaintedCallFinder(ast.NodeVisitor):
    """
    Two-pass visitor:
      Pass 1 — build a map of function_name → list of parameter names that
                are passed directly to dangerous sinks or tainted return values.
      Pass 2 — for each call site, check if any argument comes from a tainted
                parameter (interprocedural taint) or is a string-concat expression
                (SQL injection risk).
    """

    def __init__(self, source: str):
        self.source = source
        self.lines = source.splitlines()

        # function name → set of parameter indices that are "tainted to sink"
        self._tainted_params: Dict[str, Set[int]] = {}
        # function name → ast node
        self._func_nodes: Dict[str, ast.FunctionDef] = {}

        # Final findings list
        self.findings: List[Dict] = []

        # Current scope stack: list of (func_name, {param_name: param_index})
        self._scope_stack: List[Tuple[str, Dict[str, int]]] = []

    # ------------------------------------------------------------------
    # Pass 1: collect function definitions and their parameters
    # ------------------------------------------------------------------

    def _collect_functions(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._func_nodes[node.name] = node
                params = {arg.arg: i for i, arg in enumerate(node.args.args)}
                # Check if any param is forwarded to a dangerous sink
                tainted = set()
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    sink_name = self._resolve_sink(child)
                    if not sink_name:
                        continue
                    for arg_idx, arg in enumerate(child.args):
                        if isinstance(arg, ast.Name) and arg.id in params:
                            tainted.add(params[arg.id])
                if tainted:
                    self._tainted_params[node.name] = tainted

    # ------------------------------------------------------------------
    # Pass 2: visit call sites
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        params = {arg.arg: i for i, arg in enumerate(node.args.args)}
        self._scope_stack.append((node.name, params))
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        self._check_direct_sink(node)
        self._check_sql_injection(node)
        self._check_interprocedural(node)
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Direct dangerous-sink detection
    # ------------------------------------------------------------------

    def _resolve_sink(self, node: ast.Call) -> Optional[str]:
        """Return 'module.method' if this call is a known dangerous sink, else None."""
        chain = _get_attr_chain(node.func)
        if not chain:
            return None
        parts = chain.split(".")

        # Single-name sinks (eval, exec, open, compile, __import__)
        if len(parts) == 1:
            name = parts[0]
            if name in DANGEROUS_SINKS_PY.get("builtins", set()) or \
               name in DANGEROUS_SINKS_PY.get("open", set()):
                return name

        # Two-part sinks: module.method
        if len(parts) >= 2:
            mod, method = parts[-2], parts[-1]
            if method in DANGEROUS_SINKS_PY.get(mod, set()):
                return chain

        return None

    def _check_direct_sink(self, node: ast.Call):
        sink = self._resolve_sink(node)
        if not sink:
            return
        scope = self._scope_stack[-1][0] if self._scope_stack else "<module>"
        # Check if yaml.load is called without a Loader keyword arg
        if "yaml" in sink and "load" in sink:
            has_loader = any(kw.arg == "Loader" for kw in node.keywords)
            if has_loader:
                return  # yaml.load(data, Loader=yaml.SafeLoader) is fine
        self.findings.append({
            "type": "DANGEROUS_SINK",
            "sink": sink,
            "scope": scope,
            "line": node.lineno,
            "source": "direct call",
            "severity": "HIGH",
        })

    # ------------------------------------------------------------------
    # SQL injection detection
    # ------------------------------------------------------------------

    def _check_sql_injection(self, node: ast.Call):
        method = ""
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
        if method not in SQL_EXECUTE_METHODS:
            return
        # First positional arg is the query string
        if not node.args:
            return
        query_arg = node.args[0]
        if _is_string_concat_or_fstring(query_arg):
            scope = self._scope_stack[-1][0] if self._scope_stack else "<module>"
            self.findings.append({
                "type": "SQL_INJECTION",
                "sink": f".{method}()",
                "scope": scope,
                "line": node.lineno,
                "source": "string concatenation / f-string in SQL query",
                "severity": "HIGH",
            })

    # ------------------------------------------------------------------
    # Interprocedural taint check
    # ------------------------------------------------------------------

    def _check_interprocedural(self, node: ast.Call):
        callee = _get_attr_chain(node.func)
        if callee not in self._tainted_params:
            return
        tainted_indices = self._tainted_params[callee]
        if not self._scope_stack:
            return
        caller_name, caller_params = self._scope_stack[-1]
        caller_param_names = set(caller_params.keys())

        for arg_idx, arg in enumerate(node.args):
            if arg_idx not in tainted_indices:
                continue
            if isinstance(arg, ast.Name) and arg.arg if hasattr(arg, "arg") else False:
                pass
            # Taint flows if caller is also passing a parameter through
            if isinstance(arg, ast.Name) and arg.id in caller_param_names:
                self.findings.append({
                    "type": "TAINT_PROPAGATION",
                    "sink": callee,
                    "scope": caller_name,
                    "line": node.lineno,
                    "source": f"param '{arg.id}' → {callee}() → dangerous sink",
                    "severity": "HIGH",
                })


# ---------------------------------------------------------------------------
# JavaScript / TypeScript analysis (token-walk)
# ---------------------------------------------------------------------------

# Extended JS dangerous patterns — covers multi-line via token walk
JS_DANGEROUS_TOKENS: List[Tuple[str, str, str]] = [
    # pattern, description, severity
    (r"\beval\s*\(", "eval() call", "HIGH"),
    (r"\bnew\s+Function\s*\(", "new Function() — dynamic code execution", "HIGH"),
    (r"child_process\s*[\.\[]\s*['\"]?exec(?:Sync)?\s*[\'\"]?\s*[\]\)]?\s*\(",
     "child_process.exec/execSync", "HIGH"),
    (r"child_process\s*[\.\[]\s*['\"]?spawn(?:Sync)?\s*[\'\"]?\s*[\]\)]?\s*\(",
     "child_process.spawn/spawnSync", "HIGH"),
    (r"require\s*\(\s*['\"]child_process['\"]\s*\)", "child_process imported", "MEDIUM"),
    (r"\.exec\s*\(\s*(?:[`'\"]|\w+\s*\+)", "exec() with string/variable arg — injection risk", "HIGH"),
    (r"fs\.(?:readFile|writeFile|appendFile|createReadStream|createWriteStream|open)\s*\(",
     "Filesystem access via fs.*", "MEDIUM"),
    (r"\bfetch\s*\(\s*(?!\s*['\"`]https?://)", "fetch() with non-literal URL — possible SSRF", "MEDIUM"),
    (r"\baxios\s*\.\s*(?:get|post|put|delete|request)\s*\(\s*(?!\s*['\"`]https?://)",
     "axios call with non-literal URL", "MEDIUM"),
    (r"(?:http|https)\.request\s*\(", "http.request — inspect URL source", "LOW"),
    (r"\bpickle\b", "pickle usage (wrong language but pattern present)", "MEDIUM"),
    (r"__proto__\s*\[", "Prototype pollution risk", "HIGH"),
    (r"\.\s*constructor\s*\[", "Prototype pollution via constructor", "HIGH"),
]

JS_EXTENSIONS = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"}


def _analyze_js_file_v2(filepath: str) -> List[Dict]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []

    findings = []
    for pat, description, severity in JS_DANGEROUS_TOKENS:
        for m in re.finditer(pat, content, re.DOTALL):
            lineno = content[: m.start()].count("\n") + 1
            # Skip pure comment lines
            line_start = content.rfind("\n", 0, m.start()) + 1
            line_text = content[line_start: content.find("\n", m.start())].lstrip()
            if line_text.startswith("//") or line_text.startswith("*"):
                continue
            findings.append({
                "type": "JS_DANGEROUS_CALL",
                "file": filepath,
                "line": lineno,
                "match": m.group(0).strip()[:100],
                "detail": description,
                "severity": severity,
            })
    return findings


# ---------------------------------------------------------------------------
# Directory scanners
# ---------------------------------------------------------------------------

def scan_python_repo_v2(repo_path: str) -> List[Dict]:
    all_findings: List[Dict] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            if should_skip_path(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
                tree = ast.parse(source, filename=fpath)
            except (SyntaxError, OSError):
                continue

            finder = TaintedCallFinder(source)
            finder._collect_functions(tree)  # pass 1
            finder.visit(tree)               # pass 2

            for finding in finder.findings:
                finding["file"] = fpath
            all_findings.extend(finder.findings)

    return all_findings


def scan_js_repo_v2(repo_path: str) -> List[Dict]:
    all_findings: List[Dict] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if os.path.splitext(fname)[1] in JS_EXTENSIONS:
                fpath = os.path.join(root, fname)
                if should_skip_path(fpath):
                    continue
                findings = _analyze_js_file_v2(fpath)
                all_findings.extend(findings)
    return all_findings


# ---------------------------------------------------------------------------
# Public entry point (mirrors analyze_repo_path from code_analyzer.py)
# ---------------------------------------------------------------------------

def analyze_repo_path_v2(repo_path: str, repo_name: str, output_dir: str = None) -> str:
    """
    Analyze a repository using the improved v2 CFG analyzer.
    Returns the path to the written report file.
    """
    output_dir = output_dir or os.path.join(os.getcwd(), "out", "analyses", repo_name)
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "code_analysis_v2.txt")

    # Detect language by file count
    py_count = js_count = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for f in files:
            fpath = os.path.join(root, f)
            if should_skip_path(fpath):
                continue
            ext = os.path.splitext(f)[1]
            if ext == ".py":
                py_count += 1
            elif ext in JS_EXTENSIONS:
                js_count += 1

    findings: List[Dict] = []
    language = "unknown"
    if py_count >= js_count and py_count > 0:
        language = "python"
        findings = scan_python_repo_v2(repo_path)
    elif js_count > py_count:
        language = "js"
        findings = scan_js_repo_v2(repo_path)

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    findings.sort(key=lambda x: (severity_order.get(x.get("severity", "LOW"), 9),
                                  x.get("file", ""), x.get("line", 0)))

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"MCProbe CFG Analyzer v2 — {repo_name} ({language})\n")
        f.write("=" * 60 + "\n\n")
        if not findings:
            f.write("No dangerous calls or injection sinks found.\n")
        else:
            f.write(f"{len(findings)} finding(s):\n\n")
            for finding in findings:
                rel = os.path.relpath(finding.get("file", ""), repo_path)
                f.write(f"[{finding.get('severity', '?')}] {finding.get('type', '?')}\n")
                f.write(f"  File  : {rel}:{finding.get('line', '?')}\n")
                f.write(f"  Sink  : {finding.get('sink', finding.get('match', '?'))}\n")
                f.write(f"  Scope : {finding.get('scope', finding.get('detail', ''))}\n")
                if finding.get("source"):
                    f.write(f"  Chain : {finding['source']}\n")
                f.write("\n")

    return result_file, len(findings)
