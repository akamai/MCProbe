"""
MCProbe — JS/TS Flow Analyzer
Regex-based MCP tool entry point detection and taint/sink analysis
for JavaScript and TypeScript MCP servers.

Produces FunctionRecord / CallSite / SinkHit objects compatible with
mcp_flow_analyzer.py's tracing and report-writing code.
"""
from __future__ import annotations

import os
import re
import concurrent.futures
from typing import Dict, List, Set, Tuple, Optional

from helpers import EXCLUDE_DIRS, should_skip_path, dprint
from analyzers.mcp_flow_analyzer import (
    ParamInfo, CallSite, SinkHit, FunctionRecord,
    MAX_WORKERS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JS_EXTENSIONS = frozenset({".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"})

_JS_KEYWORDS = frozenset({
    "if", "else", "return", "new", "true", "false", "null", "undefined",
    "typeof", "instanceof", "await", "async", "const", "let", "var",
    "function", "class", "this", "super", "import", "export", "default",
    "from", "of", "in", "for", "while", "do", "switch", "case", "break",
    "continue", "throw", "try", "catch", "finally", "delete", "void",
    "yield", "static", "get", "set", "extends", "implements", "interface",
    "type", "enum", "namespace", "module", "declare", "abstract", "public",
    "private", "protected", "readonly", "override", "require", "then",
    "resolve", "reject", "Promise", "Array", "Object", "String", "Number",
    "Boolean", "console", "process", "Buffer", "JSON", "Math", "Error",
    "Map", "Set", "Symbol", "BigInt", "RegExp", "Date", "setTimeout",
    "clearTimeout", "setInterval", "clearInterval",
    # TypeScript type keywords that appear in param annotations
    "any", "unknown", "never", "void", "object", "Record", "Partial",
    "Required", "Readonly", "Pick", "Omit", "Exclude", "Extract",
})

# ---------------------------------------------------------------------------
# JS/TS dangerous sink catalogue
# ---------------------------------------------------------------------------

_JS_SINKS: Dict[str, List[Tuple[str, str, str]]] = {
    "COMMAND": [
        ("child_process", "exec", "HIGH"),
        ("child_process", "execSync", "HIGH"),
        ("child_process", "spawn", "HIGH"),
        ("child_process", "spawnSync", "HIGH"),
        ("child_process", "execFile", "HIGH"),
        ("child_process", "execFileSync", "HIGH"),
        ("", "exec", "HIGH"),
        ("", "execSync", "HIGH"),
        ("", "spawn", "HIGH"),
        ("execa", "execa", "HIGH"),
        ("execa", "execaSync", "HIGH"),
        ("execa", "execaCommand", "HIGH"),
        ("shelljs", "exec", "HIGH"),
    ],
    "EXEC": [
        ("", "eval", "HIGH"),
        ("vm", "runInContext", "HIGH"),
        ("vm", "runInNewContext", "HIGH"),
        ("vm", "runInThisContext", "HIGH"),
        ("vm", "compileFunction", "HIGH"),
        ("vm", "Script", "HIGH"),
    ],
    "FILE": [
        ("fs", "readFile", "MEDIUM"),
        ("fs", "readFileSync", "MEDIUM"),
        ("fs", "writeFile", "MEDIUM"),
        ("fs", "writeFileSync", "MEDIUM"),
        ("fs", "appendFile", "MEDIUM"),
        ("fs", "appendFileSync", "MEDIUM"),
        ("fs", "unlink", "MEDIUM"),
        ("fs", "unlinkSync", "MEDIUM"),
        ("fs", "rename", "MEDIUM"),
        ("fs", "renameSync", "MEDIUM"),
        ("fs", "mkdir", "MEDIUM"),
        ("fs", "mkdirSync", "MEDIUM"),
        ("fs", "rmdir", "MEDIUM"),
        ("fs", "rmdirSync", "MEDIUM"),
        ("fs", "rm", "MEDIUM"),
        ("fs", "rmSync", "MEDIUM"),
        ("fs", "createReadStream", "MEDIUM"),
        ("fs", "createWriteStream", "MEDIUM"),
    ],
    "SSRF": [
        ("", "fetch", "MEDIUM"),
        ("axios", "get", "MEDIUM"),
        ("axios", "post", "MEDIUM"),
        ("axios", "put", "MEDIUM"),
        ("axios", "delete", "MEDIUM"),
        ("axios", "request", "MEDIUM"),
        ("axios", "patch", "MEDIUM"),
        ("http", "request", "MEDIUM"),
        ("https", "request", "MEDIUM"),
        ("http", "get", "MEDIUM"),
        ("https", "get", "MEDIUM"),
        ("got", "get", "MEDIUM"),
        ("got", "post", "MEDIUM"),
        ("request", "get", "MEDIUM"),
        ("request", "post", "MEDIUM"),
        ("superagent", "get", "MEDIUM"),
        ("superagent", "post", "MEDIUM"),
        ("needle", "get", "MEDIUM"),
        ("needle", "post", "MEDIUM"),
    ],
    "DESER": [
        ("", "deserialize", "HIGH"),
        ("", "unserialize", "HIGH"),
    ],
}

_JS_SQL_METHODS = frozenset({
    "query", "execute", "raw", "run", "all", "prepare",
})

# ---------------------------------------------------------------------------
# Source preparation — two passes
# ---------------------------------------------------------------------------

def _strip_js_comments(source: str) -> str:
    """Strip block and line comments only, preserving strings and newlines."""
    result = re.sub(
        r'/\*.*?\*/',
        lambda m: re.sub(r'[^\n]', ' ', m.group()),
        source, flags=re.DOTALL,
    )
    result = re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group()), result)
    return result


def _prepare_js_source(source: str) -> str:
    """
    Strip comments AND string/template literal content with spaces, preserving
    newlines. The result has the same length as source, so positions map 1-to-1.
    Used for structural analysis (brace counting, function detection).
    Template literal ${...} interpolations are kept visible.
    """
    chars = list(source)
    i = 0
    n = len(source)

    while i < n:
        c = source[i]

        # Block comment
        if c == '/' and i + 1 < n and source[i + 1] == '*':
            i += 2
            while i < n - 1:
                if source[i] == '*' and source[i + 1] == '/':
                    chars[i] = ' '; chars[i + 1] = ' '
                    i += 2
                    break
                if source[i] != '\n':
                    chars[i] = ' '
                i += 1
            continue

        # Line comment
        if c == '/' and i + 1 < n and source[i + 1] == '/':
            i += 2
            while i < n and source[i] != '\n':
                chars[i] = ' '
                i += 1
            continue

        # Double-quoted string
        if c == '"':
            i += 1
            while i < n and source[i] != '"':
                if source[i] == '\\':
                    if source[i] != '\n': chars[i] = ' '
                    i += 1
                    if i < n and source[i] != '\n': chars[i] = ' '
                elif source[i] != '\n':
                    chars[i] = ' '
                i += 1
            i += 1
            continue

        # Single-quoted string
        if c == "'":
            i += 1
            while i < n and source[i] != "'":
                if source[i] == '\\':
                    if source[i] != '\n': chars[i] = ' '
                    i += 1
                    if i < n and source[i] != '\n': chars[i] = ' '
                elif source[i] != '\n':
                    chars[i] = ' '
                i += 1
            i += 1
            continue

        # Template literal — blank content but keep ${...} interpolations
        if c == '`':
            i += 1
            while i < n and source[i] != '`':
                if source[i] == '\\':
                    if source[i] != '\n': chars[i] = ' '
                    i += 1
                    if i < n and source[i] != '\n': chars[i] = ' '
                    continue
                if source[i] == '$' and i + 1 < n and source[i + 1] == '{':
                    # Keep interpolation visible — skip past matching }
                    i += 2
                    depth = 1
                    while i < n and depth > 0:
                        if source[i] == '{': depth += 1
                        elif source[i] == '}': depth -= 1
                        i += 1
                    continue
                if source[i] != '\n':
                    chars[i] = ' '
                i += 1
            if i < n:
                i += 1
            continue

        i += 1

    return ''.join(chars)

# ---------------------------------------------------------------------------
# Brace / paren matching
# ---------------------------------------------------------------------------

def _find_matching_brace(source: str, open_pos: int) -> int:
    """Return index of } matching the { at open_pos. Operates on prepared source."""
    depth = 1
    i = open_pos + 1
    n = len(source)
    while i < n and depth > 0:
        c = source[i]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        i += 1
    return i - 1


def _find_matching_paren(source: str, open_pos: int) -> int:
    """Return index of ) matching the ( at open_pos."""
    depth = 1
    i = open_pos + 1
    n = len(source)
    while i < n and depth > 0:
        c = source[i]
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        i += 1
    return i - 1

# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def _extract_js_params(param_str: str) -> List[ParamInfo]:
    """
    Extract parameter names from a JS/TS parameter string.
    Handles: plain names, typed (name: Type), destructured ({ a, b }),
    rest (...name), and default values (name = val).
    """
    params: List[ParamInfo] = []
    param_str = param_str.strip()
    if not param_str:
        return params

    # Destructured object first arg: { a, b, c } or { a, b }: Type
    destr_m = re.match(r'^\{([^}]*)\}', param_str)
    if destr_m:
        for m in re.finditer(r'\b(\w+)\b', destr_m.group(1)):
            name = m.group(1)
            if name not in _JS_KEYWORDS:
                params.append(ParamInfo(name=name))
        return params

    for part in param_str.split(','):
        part = part.strip().lstrip('.')           # remove rest (...)
        part = re.sub(r'\s*:\s*\S.*$', '', part).strip()   # strip : Type
        part = re.sub(r'\s*=.*$', '', part).strip()         # strip = default
        m = re.match(r'(\w+)', part)
        if m and m.group(1) not in _JS_KEYWORDS:
            params.append(ParamInfo(name=m.group(1)))

    return params

# ---------------------------------------------------------------------------
# Taint / influenced-variable tracking
# ---------------------------------------------------------------------------

def _collect_js_influenced_vars(body: str, initial: Set[str]) -> Set[str]:
    """
    Fixed-point propagation: find every variable that derives from `initial`
    via assignment, destructuring, or property access.
    """
    influenced = set(initial)

    assignments: List[Tuple[Set[str], Set[str]]] = []

    # const/let/var { a, b } = expr
    for m in re.finditer(r'(?:const|let|var)\s*\{([^}]+)\}\s*=\s*(\w+)', body):
        lhs = {n for n in re.findall(r'\b(\w+)\b', m.group(1)) if n not in _JS_KEYWORDS}
        rhs = {m.group(2)}
        assignments.append((lhs, rhs))

    # const/let/var name = expr.prop  or  expr["prop"]
    for m in re.finditer(r'(?:const|let|var)\s+(\w+)\s*=\s*(\w+)(?:\.(\w+)|\[)', body):
        lhs = m.group(1)
        if lhs not in _JS_KEYWORDS:
            assignments.append(({lhs}, {m.group(2)}))

    # const/let/var name = expr  (general)
    for m in re.finditer(r'(?:const|let|var)\s+(\w+)\s*=\s*([^;{\n]+)', body):
        lhs = m.group(1)
        if lhs not in _JS_KEYWORDS:
            rhs_idents = {n for n in re.findall(r'\b(\w+)\b', m.group(2))
                          if n not in _JS_KEYWORDS}
            assignments.append(({lhs}, rhs_idents))

    changed = True
    while changed:
        changed = False
        for lhs_names, rhs_idents in assignments:
            if rhs_idents & influenced:
                new = lhs_names - influenced
                if new:
                    influenced |= new
                    changed = True

    return influenced

# ---------------------------------------------------------------------------
# Call collection
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r'\b((?:\w+\.)*\w+)\s*\(')

def _collect_js_calls(body: str, influenced: Set[str]) -> List[CallSite]:
    """Find outgoing function calls in a JS/TS function body."""
    calls: List[CallSite] = []
    seen: Set[Tuple] = set()

    for m in _CALL_RE.finditer(body):
        dotted = m.group(1)
        parts = dotted.split('.')
        if parts[0] in _JS_KEYWORDS:
            continue
        last = parts[-1]
        if last in _JS_KEYWORDS:
            continue

        open_paren = m.end() - 1
        close_paren = _find_matching_paren(body, open_paren)
        arg_text = body[m.end():close_paren] if close_paren > m.end() else ""

        lineno = body[:m.start()].count('\n') + 1
        key = (dotted, lineno)
        if key in seen:
            continue
        seen.add(key)

        arg_idents = [n for n in re.findall(r'\b(\w+)\b', arg_text)
                      if n not in _JS_KEYWORDS]
        has_star = bool(
            set(re.findall(r'\.\.\.(\w+)', arg_text)) & influenced
        )

        calls.append(CallSite(
            callee_name=last,
            callee_dotted=dotted,
            line=lineno,
            pos_arg_vars=[arg_idents] if arg_idents else [],
            kw_arg_vars={},
            has_star_kwargs=has_star,
        ))

    return calls

# ---------------------------------------------------------------------------
# Sink collection
# ---------------------------------------------------------------------------

def _arg_touches_influenced(arg_text: str, influenced: Set[str]) -> Tuple[List[str], bool]:
    """
    Return (list_of_influenced_idents, has_dynamic_construction).
    Dynamic = template literal ${tainted} or string concat + tainted var.
    """
    idents = set(re.findall(r'\b(\w+)\b', arg_text)) - _JS_KEYWORDS
    influenced_found = sorted(idents & influenced)

    # Template literal with tainted variable
    tpl_vars = set(re.findall(r'\$\{(\w+)', arg_text))
    has_dynamic = bool(tpl_vars & influenced)

    return influenced_found, has_dynamic


def _collect_js_sinks(body: str, influenced: Set[str]) -> List[SinkHit]:
    """Find calls to dangerous sinks in a JS/TS function body."""
    sinks: List[SinkHit] = []
    seen: Set[Tuple] = set()

    for m in _CALL_RE.finditer(body):
        dotted = m.group(1)
        parts = dotted.split('.')
        if parts[0] in _JS_KEYWORDS:
            continue
        last = parts[-1]
        prefix = '.'.join(parts[:-1]) if len(parts) > 1 else ''

        open_paren = m.end() - 1
        close_paren = _find_matching_paren(body, open_paren)
        arg_text = body[m.end():close_paren] if close_paren > m.end() else ""

        lineno = body[:m.start()].count('\n') + 1
        key = (dotted, lineno)
        if key in seen:
            continue
        seen.add(key)

        influenced_found, has_dynamic = _arg_touches_influenced(arg_text, influenced)

        # Check against sink catalogue
        for sink_type, patterns in _JS_SINKS.items():
            for cat_prefix, cat_name, severity in patterns:
                if last != cat_name:
                    continue
                if cat_prefix and not (
                    prefix == cat_prefix or dotted.startswith(cat_prefix + '.')
                ):
                    continue

                note = ""
                if has_dynamic and sink_type == "COMMAND":
                    note = "template literal injection — CRITICAL"
                    severity = "HIGH"

                if influenced_found or (has_dynamic and sink_type in ("COMMAND", "EXEC")):
                    sinks.append(SinkHit(
                        sink_type=sink_type,
                        sink_expr=dotted,
                        severity=severity,
                        line=lineno,
                        influenced_args=influenced_found,
                        note=note,
                    ))
                break

        # SQL sink detection
        if last in _JS_SQL_METHODS:
            if influenced_found or has_dynamic:
                note = "dynamic SQL via template literal" if has_dynamic else ""
                sinks.append(SinkHit(
                    sink_type="SQL",
                    sink_expr=dotted,
                    severity="HIGH",
                    line=lineno,
                    influenced_args=influenced_found,
                    note=note,
                ))

        # new Function(tainted)
        pre = body[max(0, m.start() - 4):m.start()]
        if last == 'Function' and 'new' in pre and (influenced_found or has_dynamic):
            sinks.append(SinkHit(
                sink_type="EXEC",
                sink_expr="new Function",
                severity="HIGH",
                line=lineno,
                influenced_args=influenced_found,
                note="dynamic code construction",
            ))

    return sinks

# ---------------------------------------------------------------------------
# Helper: build a FunctionRecord from body + metadata
# ---------------------------------------------------------------------------

def _make_record(
    name: str,
    params: List[ParamInfo],
    body: str,
    filepath: str,
    line_start: int,
    line_end: int,
    is_mcp_tool: bool = False,
    is_call_tool: bool = False,
    is_http_handler: bool = False,
) -> FunctionRecord:
    initial = {p.name for p in params}
    influenced = _collect_js_influenced_vars(body, initial)
    calls = _collect_js_calls(body, influenced)
    sinks = _collect_js_sinks(body, influenced)
    return FunctionRecord(
        name=name,
        file=filepath,
        line_start=line_start,
        line_end=line_end,
        params=params,
        decorators=[],
        is_mcp_tool=is_mcp_tool,
        is_call_tool=is_call_tool,
        is_http_handler=is_http_handler,
        calls=calls,
        sinks=sinks,
    )

# ---------------------------------------------------------------------------
# Callback extraction: find the last inline function inside a call's args
# ---------------------------------------------------------------------------

# Callback patterns — do NOT include the { so we can locate it separately
# Handles: async (params) => , (params) => , async param => , param =>
_ARROW_CB_RE = re.compile(
    r'(?:async\s+)?(?:\(([^)]*)\)|(\w+))\s*(?::\s*[\w<>\[\]|&,\s]+\s*)?=>'
)
_FUNC_CB_RE = re.compile(
    r'(?:async\s+)?function\s*\w*\s*\(([^)]*)\)\s*(?::\s*[\w<>\[\]|&,\s]+\s*)?(?=\{)'
)

# addTool({name: 'toolName', ..., execute: async (args) => { or function(args) {
_ADD_TOOL_RE = re.compile(
    r'\baddTool\s*\(\s*\{'
)
_TOOL_NAME_PROP_RE = re.compile(r"""name\s*:\s*['"](\w+)['"]""")
_EXECUTE_PROP_RE   = re.compile(r'\bexecute\s*:\s*')


def _extract_callback(
    stripped: str,
    prepared: str,
    after_pos: int,
) -> Optional[Tuple[str, List[ParamInfo], int, int]]:
    """
    Starting at `after_pos` (inside the arg list of a tool() call), find the
    last inline arrow/function callback.  Uses `prepared` for structure and
    `stripped` for body content.
    Returns (body_from_stripped, params, line_start, line_end) or None.
    """
    # Find end of the enclosing call using paren counting on prepared
    depth = 1
    i = after_pos
    n = len(prepared)
    while i < n and depth > 0:
        c = prepared[i]
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        i += 1
    end_of_args = i - 1   # index of the closing )

    if end_of_args <= after_pos:
        return None

    prepared_args = prepared[after_pos:end_of_args]

    # Find the LAST callback pattern in the (prepared) arg region
    last_match: Optional[re.Match] = None
    last_pos = -1

    for pat in (_ARROW_CB_RE, _FUNC_CB_RE):
        for m in pat.finditer(prepared_args):
            if m.start() > last_pos:
                last_pos = m.start()
                last_match = m

    if last_match is None:
        return None

    # Locate the opening { after the match end in prepared
    brace_search_from = after_pos + last_match.end()
    brace_pos = prepared.find('{', brace_search_from)
    if brace_pos == -1 or brace_pos > end_of_args + 5:
        return None

    close_brace = _find_matching_brace(prepared, brace_pos)
    body = stripped[brace_pos + 1:close_brace]   # use stripped for content

    # Extract params from the stripped version at the same absolute position
    abs_match_start = after_pos + last_match.start()
    abs_match_end   = after_pos + last_match.end()
    param_snippet   = stripped[abs_match_start:abs_match_end]

    # Try parenthesised params first, then bare single-param arrow (async x =>)
    pm = re.search(r'\(([^)]*)\)', param_snippet)
    if pm:
        params = _extract_js_params(pm.group(1))
    else:
        # bare identifier: async request => ...
        bare = re.match(r'(?:async\s+)?(\w+)\s*=>', param_snippet.strip())
        if bare and bare.group(1) not in _JS_KEYWORDS:
            params = [ParamInfo(name=bare.group(1))]
        else:
            params = []

    line_start = stripped[:brace_pos].count('\n') + 1
    line_end   = stripped[:close_brace].count('\n') + 1

    return body, params, line_start, line_end

# ---------------------------------------------------------------------------
# MCP tool registration detection
# ---------------------------------------------------------------------------

# Matches any .tool( call: server.tool(, this.tool(, mcpServer.tool(
# Does NOT require a string literal — handles variable names too
_TOOL_CALL_RE = re.compile(
    r'(?:\bthis|\b\w+)\s*\.\s*tool\s*\('
)

# Matches: setRequestHandler(CallToolRequestSchema,
_HANDLER_RE = re.compile(
    r'setRequestHandler\s*\(\s*(?:\w+\.)*CallToolRequestSchema\s*,'
)

# Extract name from object-first arg: { name: 'literal' } or { name: VARIABLE }
_OBJ_NAME_STR_RE = re.compile(r"""name\s*:\s*['"]([^'"]+)['"]""")
_OBJ_NAME_VAR_RE = re.compile(r'name\s*:\s*(\w+)')


def _find_mcp_tool_records(
    stripped: str, prepared: str, filepath: str
) -> List[FunctionRecord]:
    records: List[FunctionRecord] = []

    for m in _TOOL_CALL_RE.finditer(stripped):
        after_open = m.end()  # position right after the opening (

        # Peek at first non-whitespace character to detect arg style
        peek = stripped[after_open:after_open + 300].lstrip()

        if peek.startswith('{'):
            # Object-first pattern: server.tool({ name: ..., schema: ... }, callback)
            brace_pos = prepared.find('{', after_open)
            if brace_pos == -1:
                continue
            obj_close = _find_matching_brace(prepared, brace_pos)
            obj_content = stripped[brace_pos:obj_close + 1]

            # Extract name — try string literal first, then variable name
            nm = _OBJ_NAME_STR_RE.search(obj_content)
            if nm:
                tool_name = nm.group(1)
            else:
                nm = _OBJ_NAME_VAR_RE.search(obj_content)
                tool_name = nm.group(1) if nm else "unknown"

            # Callback is the next argument after the object closes
            callback_start = obj_close + 1
            result = _extract_callback(stripped, prepared, callback_start)

        elif peek.startswith("'") or peek.startswith('"'):
            # String literal first arg: server.tool('name', ...)
            name_m = re.match(r"""['"]([^'"]+)['"]""", peek)
            tool_name = name_m.group(1) if name_m else "unknown"
            result = _extract_callback(stripped, prepared, after_open)

        else:
            # Variable/property first arg: server.tool(varName, ...) or this.tool(...)
            var_m = re.match(r'([\w.]+)', peek)
            raw = var_m.group(1) if var_m else "unknown"
            last_seg = raw.split('.')[-1]
            # If the last segment is a generic word like 'name', add line number
            if last_seg in ('name', 'toolName', 'unknown', 'tool'):
                line_num = stripped[:after_open].count('\n') + 1
                tool_name = f"{last_seg}_L{line_num}"
            else:
                tool_name = last_seg
            result = _extract_callback(stripped, prepared, after_open)

        if result is None:
            continue
        body, params, line_start, line_end = result
        if not params:
            params = [ParamInfo(name="args")]

        # Deduplicate: skip if we already have a record at this exact line
        if any(r.line_start == line_start and r.file == filepath for r in records):
            continue

        records.append(_make_record(
            name=f"tool__{tool_name}",
            params=params,
            body=body,
            filepath=filepath,
            line_start=line_start,
            line_end=line_end,
            is_mcp_tool=True,
        ))

    return records


def _find_add_tool_records(
    stripped: str, prepared: str, filepath: str
) -> List[FunctionRecord]:
    """
    Handle: server.addTool({ name: 'toolName', ..., execute: async (args) => { ... } })
    """
    records: List[FunctionRecord] = []

    for m in _ADD_TOOL_RE.finditer(prepared):
        # Find the closing } of the addTool({...}) object arg
        obj_open = prepared.find('{', m.start())
        if obj_open == -1:
            continue
        obj_close = _find_matching_brace(prepared, obj_open)
        obj_body_prepared = prepared[obj_open:obj_close + 1]
        obj_body_stripped  = stripped[obj_open:obj_close + 1]

        # Extract tool name from stripped (string content visible)
        name_m = _TOOL_NAME_PROP_RE.search(obj_body_stripped)
        tool_name = name_m.group(1) if name_m else "unknown"

        # Find execute: callback
        exec_m = _EXECUTE_PROP_RE.search(obj_body_prepared)
        if exec_m is None:
            continue

        exec_start_in_obj = exec_m.end()
        exec_abs_start = obj_open + exec_start_in_obj

        # Find callback after execute:
        result = _extract_callback(stripped, prepared, exec_abs_start)
        if result is None:
            continue

        body, params, line_start, line_end = result
        if not params:
            params = [ParamInfo(name="args")]

        records.append(_make_record(
            name=f"tool__{tool_name}",
            params=params,
            body=body,
            filepath=filepath,
            line_start=line_start,
            line_end=line_end,
            is_mcp_tool=True,
        ))

    return records


def _find_request_handler_records(
    stripped: str, prepared: str, filepath: str
) -> List[FunctionRecord]:
    records: List[FunctionRecord] = []

    for m in _HANDLER_RE.finditer(stripped):
        result = _extract_callback(stripped, prepared, m.end())
        if result is None:
            continue
        body, params, line_start, line_end = result
        if not params:
            params = [ParamInfo(name="request")]
        records.append(_make_record(
            name="call_tool_handler",
            params=params,
            body=body,
            filepath=filepath,
            line_start=line_start,
            line_end=line_end,
            is_call_tool=True,
        ))

    return records

# ---------------------------------------------------------------------------
# Named function indexing (for call graph)
# ---------------------------------------------------------------------------

# function [*] name<Generics>(params)
_FUNC_DECL_RE = re.compile(
    r'(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s+(\w+)'
    r'\s*(?:<[^>]*>)?\s*\(([^)]*)\)'
)

# const/let/var name = [async] [function*] (params) [=>]
_FUNC_ASSIGN_RE = re.compile(
    r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?\s*='
    r'\s*(?:async\s+)?(?:function\s*\*?\s*\w*\s*(?:<[^>]*>)?\s*)?\(([^)]*)\)'
    r'\s*(?::\s*[\w<>\[\]|&,\s]+\s*)?(?:=>)?'
)


def _find_named_function_records(
    stripped: str, prepared: str, filepath: str
) -> List[FunctionRecord]:
    records: List[FunctionRecord] = []
    seen: Set[str] = set()

    for pat in (_FUNC_DECL_RE, _FUNC_ASSIGN_RE):
        for m in pat.finditer(prepared):
            func_name = m.group(1)
            if func_name in _JS_KEYWORDS or func_name in seen:
                continue

            # Find opening brace — must be close to the match end
            brace_pos = prepared.find('{', m.end())
            if brace_pos == -1 or brace_pos > m.end() + 60:
                continue

            close_brace = _find_matching_brace(prepared, brace_pos)
            body = stripped[brace_pos + 1:close_brace]

            # Get params from stripped at same position
            param_snippet = stripped[m.start():m.end()]
            pm = re.search(r'\(([^)]*)\)', param_snippet)
            params = _extract_js_params(pm.group(1)) if pm else []

            line_start = prepared[:m.start()].count('\n') + 1
            line_end   = prepared[:close_brace].count('\n') + 1

            records.append(_make_record(
                name=func_name,
                params=params,
                body=body,
                filepath=filepath,
                line_start=line_start,
                line_end=line_end,
            ))
            seen.add(func_name)

    return records

# ---------------------------------------------------------------------------
# Per-file indexer
# ---------------------------------------------------------------------------

def _index_js_file(filepath: str) -> List[FunctionRecord]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception:
        return []

    stripped = _strip_js_comments(source)
    prepared = _prepare_js_source(source)

    records: List[FunctionRecord] = []
    records.extend(_find_named_function_records(stripped, prepared, filepath))
    records.extend(_find_mcp_tool_records(stripped, prepared, filepath))
    records.extend(_find_add_tool_records(stripped, prepared, filepath))
    records.extend(_find_request_handler_records(stripped, prepared, filepath))
    return records

# ---------------------------------------------------------------------------
# File iterator
# ---------------------------------------------------------------------------

def _iter_js_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if os.path.splitext(fname)[1] not in JS_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            if not should_skip_path(fpath):
                yield fpath

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_js_function_index(
    repo_path: str,
) -> Tuple[Dict[str, List[FunctionRecord]], int, int]:
    """
    Scan all JS/TS files in parallel.
    Returns (index, total_functions, total_files).
    """
    files = list(_iter_js_files(repo_path))
    index: Dict[str, List[FunctionRecord]] = {}
    total_funcs = 0

    dprint(f"[FLOW] Indexing {len(files)} JS/TS file(s) with <={MAX_WORKERS} threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_index_js_file, f): f for f in files}
        for fut in concurrent.futures.as_completed(futures):
            try:
                records = fut.result()
            except Exception:
                continue
            for rec in records:
                index.setdefault(rec.name, []).append(rec)
                total_funcs += 1

    return index, total_funcs, len(files)


def find_js_entry_points(
    index: Dict[str, List[FunctionRecord]],
) -> List[FunctionRecord]:
    """Return all MCP tool handler functions found in the JS index."""
    return [
        rec
        for records in index.values()
        for rec in records
        if rec.is_mcp_tool or rec.is_call_tool
    ]
