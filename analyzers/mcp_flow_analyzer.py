"""
MCProbe — MCP Flow Analyzer
Traces user-controlled inputs from MCP tool entry points through the
project-wide call graph, flagging dangerous sinks at each reachable function.

Two parallel phases:
  Phase 1 — index every Python function in the repo (<=10 threads, one per file)
  Phase 2 — trace each MCP tool entry point through the call graph (<=10 threads)

For HTTP handlers a lightweight single-pass scan checks the handler body and
its direct callees for dangerous calls.
"""
from __future__ import annotations

import ast
import os
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

from helpers import EXCLUDE_DIRS, should_skip_path, dprint

MAX_WORKERS = 10
DEFAULT_DEPTH = 4

# ---------------------------------------------------------------------------
# Dangerous sink catalogue
# (prefix, function_name, severity)
# prefix="" means a builtin / any module
# ---------------------------------------------------------------------------

_SINKS: Dict[str, List[Tuple[str, str, str]]] = {
    "COMMAND": [
        ("os", "system", "HIGH"),
        ("os", "popen", "HIGH"),
        ("os", "execl", "HIGH"), ("os", "execle", "HIGH"), ("os", "execlp", "HIGH"),
        ("os", "execv", "HIGH"), ("os", "execve", "HIGH"),
        ("os", "execvp", "HIGH"), ("os", "execvpe", "HIGH"),
        ("os", "spawnl", "HIGH"), ("os", "spawnle", "HIGH"), ("os", "spawnlp", "HIGH"),
        ("os", "spawnv", "HIGH"), ("os", "spawnve", "HIGH"), ("os", "spawnvp", "HIGH"),
        ("subprocess", "run", "HIGH"),
        ("subprocess", "call", "HIGH"),
        ("subprocess", "check_call", "HIGH"),
        ("subprocess", "check_output", "HIGH"),
        ("subprocess", "Popen", "HIGH"),
    ],
    "EXEC": [
        ("", "eval", "HIGH"),
        ("", "exec", "HIGH"),
        ("", "compile", "HIGH"),
        ("", "__import__", "HIGH"),
    ],
    "FILE": [
        ("", "open", "MEDIUM"),
        ("shutil", "rmtree", "MEDIUM"),
        ("shutil", "move", "MEDIUM"),
        ("shutil", "copy", "MEDIUM"),
        ("os", "unlink", "MEDIUM"),
        ("os", "remove", "MEDIUM"),
        ("os", "rename", "MEDIUM"),
        ("os", "makedirs", "MEDIUM"),
    ],
    "DESER": [
        ("pickle", "loads", "HIGH"),
        ("pickle", "load", "HIGH"),
        ("marshal", "loads", "HIGH"),
        ("marshal", "load", "HIGH"),
        ("yaml", "load", "HIGH"),
    ],
    "SSRF": [
        ("requests", "get", "MEDIUM"),
        ("requests", "post", "MEDIUM"),
        ("requests", "put", "MEDIUM"),
        ("requests", "delete", "MEDIUM"),
        ("requests", "patch", "MEDIUM"),
        ("requests", "request", "MEDIUM"),
        ("requests", "head", "MEDIUM"),
        ("httpx", "get", "MEDIUM"),
        ("httpx", "post", "MEDIUM"),
        ("httpx", "put", "MEDIUM"),
        ("httpx", "delete", "MEDIUM"),
        ("urllib.request", "urlopen", "MEDIUM"),
        ("urllib", "urlopen", "MEDIUM"),
        ("aiohttp", "get", "MEDIUM"),
        ("aiohttp", "post", "MEDIUM"),
    ],
}

# SQL sinks — detected by method name on a cursor/ORM object
_SQL_METHODS: frozenset = frozenset({
    "execute", "executemany", "executescript", "raw", "filter",
})

# HTTP route decorator attribute names (Flask / FastAPI / Starlette)
_HTTP_ROUTE_ATTRS: frozenset = frozenset({
    "route", "get", "post", "put", "delete", "patch", "head",
})

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    name: str
    annotation: str = ""


@dataclass
class CallSite:
    """An outgoing function call within a function body."""
    callee_name: str            # last dotted segment, used for index lookup
    callee_dotted: str          # full expression, e.g. "adfin.upload_invoice"
    line: int
    # For each positional argument: the set of variable names appearing in it
    pos_arg_vars: List[List[str]] = field(default_factory=list)
    # For each keyword argument: kwarg_name → variable names appearing in value
    kw_arg_vars: Dict[str, List[str]] = field(default_factory=dict)
    # True if any **name was used where name derives from user input
    has_star_kwargs: bool = False


@dataclass
class SinkHit:
    """A call to a dangerous function inside a function body."""
    sink_type: str              # "COMMAND", "SQL", "FILE", etc.
    sink_expr: str              # e.g. "subprocess.run"
    severity: str
    line: int
    # Variable names (from the function's influenced set) that appear in the args
    influenced_args: List[str] = field(default_factory=list)
    note: str = ""


@dataclass
class FunctionRecord:
    name: str
    file: str
    line_start: int
    line_end: int
    params: List[ParamInfo]
    decorators: List[str]
    is_mcp_tool: bool
    is_call_tool: bool
    is_http_handler: bool
    calls: List[CallSite]
    sinks: List[SinkHit]


@dataclass
class TraceNode:
    """One node in the call tree produced during flow tracing."""
    func_name: str
    file: str
    line_call: int              # line in the CALLER where this function was called
    call_expr: str              # dotted expression used at the call site
    tracked_params: Set[str]    # params tracked at this level
    sink_hits: List[SinkHit]
    children: List["TraceNode"] = field(default_factory=list)
    is_external: bool = False   # callee not found in index


@dataclass
class HttpFinding:
    handler_name: str
    file: str
    line: int
    sink_type: str
    sink_expr: str
    severity: str
    sink_line: int
    note: str = ""


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _annotation_str(node) -> str:
    """Convert a type-annotation AST node to a string, with fallback."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)          # Python 3.9+
    except AttributeError:
        pass
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_annotation_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_annotation_str(node.value)}[...]"
    return ""


def _decorator_str(node) -> str:
    """Return a string representation of a decorator node."""
    if isinstance(node, ast.Call):
        return _decorator_str(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_decorator_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _iter_body_nodes(func_node):
    """
    Yield all AST nodes within func_node's body WITHOUT descending into
    nested function or class definitions.
    """
    def _walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                continue
            yield child
            yield from _walk(child)

    for stmt in getattr(func_node, "body", []):
        yield stmt
        yield from _walk(stmt)


def _names_in(node) -> List[str]:
    """Return all ast.Name.id values found anywhere in node's subtree."""
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


def _extract_target_names(node) -> Set[str]:
    """Return variable names on the left-hand side of an assignment."""
    names: Set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names |= _extract_target_names(elt)
    elif isinstance(node, ast.Starred):
        names |= _extract_target_names(node.value)
    return names


def _get_call_expr(call: ast.Call) -> Tuple[str, str]:
    """
    Return (dotted_expression, last_segment) for a Call node.
    e.g. "adfin.upload_invoice", "upload_invoice"
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, func.id
    if isinstance(func, ast.Attribute):
        parts = []
        node = func
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        parts.reverse()
        return ".".join(parts), func.attr
    return "", ""


# ---------------------------------------------------------------------------
# Influenced-variable tracker (local to one function)
# ---------------------------------------------------------------------------

def _collect_influenced_vars(func_node, initial: Set[str]) -> Set[str]:
    """
    Walk the function body and find every variable that derives (directly or
    indirectly) from the initial set.  Uses a simple fixed-point iteration
    over assignments — no SSA, but handles the common patterns.
    """
    influenced = set(initial)

    # Collect (target, value) pairs from the function body only
    assignments: List[Tuple] = []
    for node in _iter_body_nodes(func_node):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assignments.append((t, node.value))
        elif isinstance(node, ast.AugAssign):
            assignments.append((node.target, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value:
            assignments.append((node.target, node.value))

    changed = True
    while changed:
        changed = False
        for target, value in assignments:
            tnames = _extract_target_names(target)
            vnames = set(_names_in(value))
            if vnames & influenced:
                new = tnames - influenced
                if new:
                    influenced |= new
                    changed = True

    return influenced


# ---------------------------------------------------------------------------
# Decorator / entry-point detection
# ---------------------------------------------------------------------------

def _is_tool_decorator(node) -> bool:
    """True for @X.tool() or @X.tool."""
    func = node.func if isinstance(node, ast.Call) else node
    return isinstance(func, ast.Attribute) and func.attr == "tool"


def _is_call_tool_decorator(node) -> bool:
    """True for @X.call_tool() or @X.call_tool."""
    func = node.func if isinstance(node, ast.Call) else node
    return isinstance(func, ast.Attribute) and func.attr == "call_tool"


def _is_http_decorator(node) -> bool:
    """True for Flask/FastAPI/Starlette route decorators."""
    func = node.func if isinstance(node, ast.Call) else node
    return isinstance(func, ast.Attribute) and func.attr in _HTTP_ROUTE_ATTRS


def _has_request_param(params: List[ParamInfo]) -> bool:
    """True if any parameter looks like an HTTP request object."""
    return any("Request" in p.annotation for p in params)


# ---------------------------------------------------------------------------
# Call / sink collection inside a function body
# ---------------------------------------------------------------------------

def _collect_calls(func_node, influenced: Set[str]) -> List[CallSite]:
    """
    Extract every outgoing function call in the body.
    `influenced` is used to detect **kwargs star-expansion with user-controlled dicts.
    """
    calls: List[CallSite] = []
    seen: Set[Tuple] = set()

    for node in _iter_body_nodes(func_node):
        if not isinstance(node, ast.Call):
            continue
        dotted, last = _get_call_expr(node)
        if not last:
            continue

        key = (dotted, node.lineno)
        if key in seen:
            continue
        seen.add(key)

        pos_arg_vars: List[List[str]] = []
        for arg in node.args:
            pos_arg_vars.append(_names_in(arg))

        kw_arg_vars: Dict[str, List[str]] = {}
        has_star = False
        for kw in node.keywords:
            if kw.arg is None:
                # **name — check if it's influenced
                if set(_names_in(kw.value)) & influenced:
                    has_star = True
            else:
                kw_arg_vars[kw.arg] = _names_in(kw.value)

        calls.append(CallSite(
            callee_name=last,
            callee_dotted=dotted,
            line=node.lineno,
            pos_arg_vars=pos_arg_vars,
            kw_arg_vars=kw_arg_vars,
            has_star_kwargs=has_star,
        ))

    return calls


def _collect_sinks(func_node, influenced: Set[str]) -> List[SinkHit]:
    """
    Find calls to dangerous sinks inside the function body and record which
    influenced variables appear in their arguments.
    """
    sinks: List[SinkHit] = []
    seen: Set[Tuple] = set()

    for node in _iter_body_nodes(func_node):
        if not isinstance(node, ast.Call):
            continue
        dotted, last = _get_call_expr(node)
        if not dotted:
            continue

        key = (dotted, node.lineno)
        if key in seen:
            continue
        seen.add(key)

        # All argument variable names (positional + keyword values)
        all_arg_names: List[str] = []
        for arg in node.args:
            all_arg_names.extend(_names_in(arg))
        for kw in node.keywords:
            all_arg_names.extend(_names_in(kw.value))

        influenced_found = sorted(set(all_arg_names) & influenced)

        # --- Catalogue sinks ---
        for sink_type, patterns in _SINKS.items():
            for prefix, name, severity in patterns:
                if last != name:
                    continue
                if prefix and not (dotted == f"{prefix}.{name}"
                                   or dotted.startswith(f"{prefix}.")):
                    continue

                note = ""
                sev = severity
                # Escalate subprocess with shell=True → CRITICAL note
                if sink_type == "COMMAND":
                    for kw in node.keywords:
                        if (kw.arg == "shell"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True):
                            note = "shell=True — CRITICAL"
                            sev = "HIGH"

                sinks.append(SinkHit(
                    sink_type=sink_type,
                    sink_expr=dotted,
                    severity=sev,
                    line=node.lineno,
                    influenced_args=influenced_found,
                    note=note,
                ))

        # --- SQL sinks (method-name based) ---
        if last in _SQL_METHODS:
            has_dynamic = any(
                isinstance(a, (ast.JoinedStr, ast.BinOp))
                for a in node.args
            )
            if influenced_found or has_dynamic:
                note = "dynamic SQL construction" if has_dynamic else ""
                sinks.append(SinkHit(
                    sink_type="SQL",
                    sink_expr=dotted,
                    severity="HIGH",
                    line=node.lineno,
                    influenced_args=influenced_found,
                    note=note,
                ))

    return sinks


# ---------------------------------------------------------------------------
# Phase 1 — per-file indexing
# ---------------------------------------------------------------------------

def _iter_python_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            if not should_skip_path(fpath):
                yield fpath


def _index_file(filepath: str) -> List[FunctionRecord]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except Exception:
        return []

    records: List[FunctionRecord] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Decorators
        dec_strs = [_decorator_str(d) for d in node.decorator_list]
        is_mcp_tool  = any(_is_tool_decorator(d)      for d in node.decorator_list)
        is_call_tool = any(_is_call_tool_decorator(d) for d in node.decorator_list)

        # Parameters (skip self / cls)
        params: List[ParamInfo] = []
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            params.append(ParamInfo(
                name=arg.arg,
                annotation=_annotation_str(arg.annotation),
            ))

        # HTTP handler detection
        is_http = (
            any(_is_http_decorator(d) for d in node.decorator_list)
            or _has_request_param(params)
        )

        # Influenced variables — start from all params
        initial = {p.name for p in params}
        influenced = _collect_influenced_vars(node, initial)

        calls = _collect_calls(node, influenced)
        sinks = _collect_sinks(node, influenced)

        line_end = getattr(node, "end_lineno", node.lineno + 20)

        records.append(FunctionRecord(
            name=node.name,
            file=filepath,
            line_start=node.lineno,
            line_end=line_end,
            params=params,
            decorators=dec_strs,
            is_mcp_tool=is_mcp_tool,
            is_call_tool=is_call_tool,
            is_http_handler=is_http,
            calls=calls,
            sinks=sinks,
        ))

    return records


def build_function_index(
    repo_path: str,
) -> Tuple[Dict[str, List[FunctionRecord]], int, int]:
    """
    Scan all Python files in parallel.
    Returns (index, total_functions, total_files).
    """
    files = list(_iter_python_files(repo_path))
    index: Dict[str, List[FunctionRecord]] = {}
    total_funcs = 0

    dprint(f"[FLOW] Indexing {len(files)} Python file(s) with <={MAX_WORKERS} threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_index_file, f): f for f in files}
        for fut in concurrent.futures.as_completed(futures):
            try:
                records = fut.result()
            except Exception:
                continue
            for rec in records:
                index.setdefault(rec.name, []).append(rec)
                total_funcs += 1

    return index, total_funcs, len(files)


def find_entry_points(
    index: Dict[str, List[FunctionRecord]],
) -> List[FunctionRecord]:
    """Return all MCP tool handler functions found in the index."""
    return [
        rec
        for records in index.values()
        for rec in records
        if rec.is_mcp_tool or rec.is_call_tool
    ]


# ---------------------------------------------------------------------------
# Phase 2 — call-graph tracing
# ---------------------------------------------------------------------------

def _map_tracked_to_callee(
    call: CallSite,
    caller_tracked: Set[str],
    callee: FunctionRecord,
) -> Set[str]:
    """
    Given which params are tracked in the caller, return which of the callee's
    params should be tracked based on how arguments are passed.
    """
    if call.has_star_kwargs:
        # Conservative: any **dict expansion may carry all tracked params
        return {p.name for p in callee.params}

    tracked: Set[str] = set()

    # Positional args
    for i, arg_vars in enumerate(call.pos_arg_vars):
        if i >= len(callee.params):
            break
        if set(arg_vars) & caller_tracked:
            tracked.add(callee.params[i].name)

    # Keyword args
    for kw_name, arg_vars in call.kw_arg_vars.items():
        if set(arg_vars) & caller_tracked:
            for p in callee.params:
                if p.name == kw_name:
                    tracked.add(p.name)
                    break

    return tracked


def _dfs(
    parent_node: TraceNode,
    parent_record: FunctionRecord,
    index: Dict[str, List[FunctionRecord]],
    remaining: int,
    visited: Set[str],
) -> None:
    """Recursive DFS — mutates parent_node.children in place."""
    if remaining < 0:
        return

    for call in parent_record.calls:
        # Check whether any tracked param flows into this call's arguments
        all_call_vars: Set[str] = set()
        for vars_list in call.pos_arg_vars:
            all_call_vars.update(vars_list)
        for vars_list in call.kw_arg_vars.values():
            all_call_vars.update(vars_list)

        carries_tracked = bool(all_call_vars & parent_node.tracked_params) or call.has_star_kwargs
        callee_records = index.get(call.callee_name, [])

        if not callee_records:
            # External call — only show if it carries tracked input
            if carries_tracked:
                parent_node.children.append(TraceNode(
                    func_name=call.callee_name,
                    file="",
                    line_call=call.line,
                    call_expr=call.callee_dotted,
                    tracked_params=set(),
                    sink_hits=[],
                    is_external=True,
                ))
            continue

        for callee_rec in callee_records:
            if callee_rec.name in visited:
                continue

            new_tracked = _map_tracked_to_callee(call, parent_node.tracked_params, callee_rec)
            if not new_tracked:
                if carries_tracked:
                    # Conservative fallback: propagate all caller tracked params
                    new_tracked = set(parent_node.tracked_params)
                else:
                    continue  # nothing flows here

            reachable_sinks = [
                s for s in callee_rec.sinks
                if not s.influenced_args or (set(s.influenced_args) & new_tracked)
            ]

            child = TraceNode(
                func_name=callee_rec.name,
                file=callee_rec.file,
                line_call=call.line,
                call_expr=call.callee_dotted,
                tracked_params=new_tracked,
                sink_hits=reachable_sinks,
            )
            parent_node.children.append(child)

            if remaining > 0:
                _dfs(child, callee_rec, index, remaining - 1,
                     visited | {callee_rec.name})


def _trace_entry(
    entry: FunctionRecord,
    index: Dict[str, List[FunctionRecord]],
    depth: int = DEFAULT_DEPTH,
) -> TraceNode:
    """Trace user-controlled input from one MCP tool entry point."""
    initial_tracked = {p.name for p in entry.params}

    # For call_tool handlers, the `arguments` dict is the primary input source;
    # all other params are still tracked conservatively.
    if entry.is_call_tool:
        args_param = next((p.name for p in entry.params if p.name == "arguments"), None)
        if args_param:
            initial_tracked = {args_param}

    entry_sinks = [
        s for s in entry.sinks
        if not s.influenced_args or (set(s.influenced_args) & initial_tracked)
    ]

    root = TraceNode(
        func_name=entry.name,
        file=entry.file,
        line_call=entry.line_start,
        call_expr=entry.name,
        tracked_params=initial_tracked,
        sink_hits=entry_sinks,
    )

    _dfs(root, entry, index, depth - 1, {entry.name})
    return root


def trace_all_entries(
    entries: List[FunctionRecord],
    index: Dict[str, List[FunctionRecord]],
    depth: int = DEFAULT_DEPTH,
) -> List[TraceNode]:
    """Trace all entry points in parallel."""
    results: List[TraceNode] = []
    dprint(f"[FLOW] Tracing {len(entries)} entry point(s) with <={MAX_WORKERS} threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_trace_entry, entry, index, depth): entry
            for entry in entries
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                dprint(f"[FLOW] Trace failed: {e}")

    return results


# ---------------------------------------------------------------------------
# HTTP handler scan (lightweight)
# ---------------------------------------------------------------------------

def _scan_http_handlers(
    index: Dict[str, List[FunctionRecord]],
) -> List[HttpFinding]:
    """
    Find HTTP handler functions and flag dangerous calls in their body or
    direct callees (depth 1).
    """
    findings: List[HttpFinding] = []

    http_handlers = [
        rec
        for records in index.values()
        for rec in records
        if rec.is_http_handler
    ]

    for handler in http_handlers:
        for sink in handler.sinks:
            findings.append(HttpFinding(
                handler_name=handler.name,
                file=handler.file,
                line=handler.line_start,
                sink_type=sink.sink_type,
                sink_expr=sink.sink_expr,
                severity=sink.severity,
                sink_line=sink.line,
                note=sink.note,
            ))

        # One level of callees
        for call in handler.calls:
            for callee_rec in index.get(call.callee_name, []):
                for sink in callee_rec.sinks:
                    findings.append(HttpFinding(
                        handler_name=handler.name,
                        file=handler.file,
                        line=handler.line_start,
                        sink_type=sink.sink_type,
                        sink_expr=f"{call.callee_dotted} → {sink.sink_expr}",
                        severity=sink.severity,
                        sink_line=sink.line,
                        note=f"via {call.callee_name}() at line {call.line}",
                    ))

    return findings


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _count_sinks(node: TraceNode) -> int:
    n = len(node.sink_hits)
    for child in node.children:
        n += _count_sinks(child)
    return n


def _render_tree(
    node: TraceNode,
    repo_path: str,
    indent: str = "  ",
    is_last: bool = True,
) -> List[str]:
    lines: List[str] = []
    branch  = "└─" if is_last else "├─"
    padding = "   " if is_last else "│  "

    # Location label
    if node.is_external:
        loc = "(external library)"
    elif node.file:
        try:
            rel = os.path.relpath(node.file, repo_path)
        except ValueError:
            rel = node.file
        loc = f"[{rel}:{node.line_call}]"
    else:
        loc = f"[line {node.line_call}]"

    call_label = (node.call_expr
                  if node.call_expr != node.func_name
                  else node.func_name)
    lines.append(f"{indent}{branch} {call_label}  {loc}")

    if node.is_external:
        return lines

    child_indent = indent + padding

    # Sink hits at this node
    for sink in node.sink_hits:
        inf = f"(via: {', '.join(sink.influenced_args)})" if sink.influenced_args else ""
        note = f" — {sink.note}" if sink.note else ""
        lines.append(
            f"{child_indent}⚡ [{sink.severity}] {sink.sink_type}: "
            f"{sink.sink_expr}  {inf}{note}"
        )

    # Children
    for i, child in enumerate(node.children):
        lines.extend(
            _render_tree(child, repo_path, child_indent, i == len(node.children) - 1)
        )

    return lines


def _write_report(
    repo_name: str,
    repo_path: str,
    traces: List[TraceNode],
    http_findings: List[HttpFinding],
    total_funcs: int,
    total_files: int,
    output_path: str,
) -> int:
    lines: List[str] = []
    total_sinks = 0

    lines.append(f"MCProbe MCP Flow Analyzer — {repo_name}")
    lines.append("=" * 60)
    lines.append(f"Indexed {total_funcs} function(s) across {total_files} file(s).")
    entry_word = "entry point" if len(traces) == 1 else "entry points"
    lines.append(f"Found {len(traces)} MCP tool {entry_word}.")
    lines.append("")

    # Sort traces for stable output
    for trace in sorted(traces, key=lambda t: (t.file, t.func_name)):
        sink_count = _count_sinks(trace)
        total_sinks += sink_count

        try:
            rel_file = os.path.relpath(trace.file, repo_path)
        except ValueError:
            rel_file = trace.file

        lines.append(
            f"━━━ TOOL: {trace.func_name}  "
            f"[{rel_file}:{trace.line_call}] ━━━"
        )

        if trace.tracked_params:
            lines.append("Parameters (all user-controlled):")
            for p in sorted(trace.tracked_params):
                lines.append(f"  • {p}")
        else:
            lines.append("Parameters: (none)")
        lines.append("")

        lines.append(f"Call tree (depth {DEFAULT_DEPTH}):")
        lines.append(f"  {trace.func_name}")

        # Sink hits at entry level
        for sink in trace.sink_hits:
            inf = f"(via: {', '.join(sink.influenced_args)})" if sink.influenced_args else ""
            note = f" — {sink.note}" if sink.note else ""
            lines.append(
                f"  ⚡ [{sink.severity}] {sink.sink_type}: "
                f"{sink.sink_expr}  {inf}{note}"
            )

        if trace.children:
            for i, child in enumerate(trace.children):
                lines.extend(
                    _render_tree(child, repo_path, "  ", i == len(trace.children) - 1)
                )
        elif not trace.sink_hits:
            lines.append("  └─ (no outgoing calls found or no input flows out)")

        lines.append("")
        if sink_count == 0:
            lines.append(f"Result: CLEAN — no dangerous sinks reached within depth {DEFAULT_DEPTH}")
        else:
            lines.append(f"Result: {sink_count} sink(s) reached")
        lines.append("")

    # HTTP handler section
    if http_findings:
        lines.append("━━━ HTTP HANDLERS ━━━")
        for hf in http_findings:
            total_sinks += 1
            try:
                rel = os.path.relpath(hf.file, repo_path)
            except ValueError:
                rel = hf.file
            note = f" — {hf.note}" if hf.note else ""
            lines.append(
                f"  [{hf.severity}] {hf.handler_name} [{rel}:{hf.line}]"
            )
            lines.append(
                f"    {hf.sink_type}: {hf.sink_expr} [line {hf.sink_line}]{note}"
            )
        lines.append("")

    # Footer
    if total_sinks == 0:
        lines.append(
            "No dangerous sinks reachable from MCP tool handlers or HTTP request parsers."
        )
    else:
        lines.append(f"{total_sinks} total finding(s).")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return total_sinks


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _count_source_files(repo_path: str) -> Tuple[int, int]:
    """Return (python_count, js_ts_count) of non-test source files."""
    _JS_EXT = frozenset({".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"})
    py_count = js_count = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            if should_skip_path(fpath):
                continue
            ext = os.path.splitext(fname)[1]
            if ext == ".py":
                py_count += 1
            elif ext in _JS_EXT:
                js_count += 1
    return py_count, js_count


def analyze_mcp_flow(
    repo_path: str,
    repo_name: str,
    output_dir: str,
    depth: int = DEFAULT_DEPTH,
) -> Tuple[str, int]:
    """
    Orchestrator-compatible entry point.
    Returns (report_path, issue_count).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mcp_flow_analysis.txt")

    py_count, js_count = _count_source_files(repo_path)
    if js_count > py_count and js_count > 0:
        from analyzers.js_flow_analyzer import build_js_function_index, find_js_entry_points
        index, total_funcs, total_files = build_js_function_index(repo_path)
        entries = find_js_entry_points(index)
        lang_label = "JS/TS"
    else:
        index, total_funcs, total_files = build_function_index(repo_path)
        entries = find_entry_points(index)
        lang_label = "Python"

    dprint(f"[FLOW] Indexed {total_funcs} function(s) in {total_files} {lang_label} file(s)")
    dprint(f"[FLOW] Found {len(entries)} MCP tool entry point(s)")

    if not entries:
        if lang_label == "JS/TS":
            hint = "No MCP tool entry points detected (server.tool() or setRequestHandler patterns).\n"
        else:
            hint = "No MCP tool entry points detected (@*.tool() decorated functions).\n"
        msg = (
            f"MCProbe MCP Flow Analyzer — {repo_name}\n"
            + "=" * 60 + "\n\n"
            + f"Indexed {total_funcs} function(s) across {total_files} file(s).\n"
            + hint
            + "This may be a non-Python repo or tools registered dynamically.\n"
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(msg)
        return output_path, 0

    traces = trace_all_entries(entries, index, depth)
    http_findings = _scan_http_handlers(index)

    issue_count = _write_report(
        repo_name, repo_path, traces, http_findings,
        total_funcs, total_files, output_path,
    )
    dprint(f"[FLOW] Done — {issue_count} finding(s) across {len(entries)} tool(s)")
    return output_path, issue_count


def analyze(repo_path: str, output_path: str) -> int:
    """
    PROTOCOL_FUNC — matches the MCProbe custom module interface.
    Signature: analyze(repo_path: str, output_path: str) -> int
    """
    repo_name = os.path.basename(repo_path.rstrip("/\\")) or "repo"
    output_dir = os.path.dirname(output_path) or "."
    _, count = analyze_mcp_flow(repo_path, repo_name, output_dir)
    return count
