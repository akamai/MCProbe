"""
MCProbe — Network Analyzer
Detects exposed network bindings and HTTP server patterns.
"""
import re
import os
import shutil
from typing import List, Dict
from helpers import EXCLUDE_DIRS, should_skip_path

# ── Listening / bind patterns ─────────────────────────────────────────────

LISTENING_PATTERNS = [
    r"app\.run\(\s*host\s*=\s*['\"](?P<host>[\d\.]+)['\"](?:\s*,\s*port\s*=\s*(?P<port>\d+))?",
    r"flask\.run\(\s*host\s*=\s*['\"](?P<host>[\d\.]+)['\"](?:\s*,\s*port\s*=\s*(?P<port>\d+))?",
    r"uvicorn\.run\s*\([^)]*host\s*=\s*['\"](?P<host>[\d\.]+)['\"](?:[^)]*port\s*=\s*(?P<port>\d+))?",
    r"socket\.bind\(\s*\(?\s*['\"]?(?P<host>[\d\.]+)['\"]?\s*,\s*(?P<port>\d+)\s*\)?\)",
    r"['\"](?P<host>0\.0\.0\.0|127\.0\.0\.1|localhost)['\"]\s*[:,]\s*(?P<port>\d+)",
    r"app\.listen\(\s*(?:(?P<port1>\d+)|['\"](?P<port2>\d+)['\"])\s*(?:,\s*['\"](?P<host>[\d\.]+)['\"])?",
    r"server\.listen\(\s*(?:(?P<port3>\d+)|['\"](?P<port4>\d+)['\"])\s*(?:,\s*['\"](?P<host2>[\d\.]+)['\"])?",
    # FastAPI / starlette
    r"uvicorn\.run\s*\(['\"][\w\.]+['\"]\s*,\s*host\s*=\s*['\"](?P<host>[\d\.]+)['\"]",
    # Generic 0.0.0.0 string literal
    r"\"(?P<host>0\.0\.0\.0)\"",
    r"'(?P<host>0\.0\.0\.0)'",
]

# ── HTTP server heuristics ────────────────────────────────────────────────

HTTP_HEURISTICS = [
    ("http.server",       "Python http.server",       "MEDIUM"),
    ("BaseHTTPRequestHandler", "Python BaseHTTPRequestHandler", "MEDIUM"),
    ("http.createServer", "Node.js http.createServer", "MEDIUM"),
    ("https.createServer","Node.js https.createServer","MEDIUM"),
    ("express",           "Express framework",         "INFO"),
    ("fastapi",           "FastAPI framework",         "INFO"),
    ("flask",             "Flask framework",           "INFO"),
    ("starlette",         "Starlette framework",       "INFO"),
    ("aiohttp",           "aiohttp server",            "INFO"),
    ("tornado",           "Tornado web server",        "INFO"),
]


def _iter_source_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for fname in files:
            if fname.endswith((".py", ".js", ".ts", ".jsx", ".tsx", ".mjs")):
                fpath = os.path.join(root, fname)
                if not should_skip_path(fpath):
                    yield fpath


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _lineno(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


# ── Scanner functions ─────────────────────────────────────────────────────

def _scan_bindings(repo_path: str) -> List[Dict]:
    findings = []
    seen = set()
    for fpath in _iter_source_files(repo_path):
        text = _read(fpath)
        if not text:
            continue
        for pat in LISTENING_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                g = m.groupdict()
                host = (g.get("host") or g.get("host2") or "").strip("'\"") or None
                port_raw = (g.get("port") or g.get("port1") or g.get("port2") or
                            g.get("port3") or g.get("port4"))
                port = int(port_raw) if port_raw and port_raw.isdigit() else None
                line = _lineno(text, m.start())
                snippet = text[m.start(): m.end() + 60].splitlines()[0].strip()
                key = (fpath, line)
                if key in seen:
                    continue
                seen.add(key)

                if host == "0.0.0.0":
                    severity = "HIGH"
                    note = "Binding to 0.0.0.0 exposes the service on ALL network interfaces"
                elif host in ("127.0.0.1", "localhost"):
                    severity = "MEDIUM"
                    note = f"Binding to {host} — restricted to localhost but verify proxy/container configuration"
                elif host:
                    severity = "MEDIUM"
                    note = f"Explicit host binding to {host}"
                else:
                    severity = "MEDIUM"
                    note = "Network binding detected — verify host/interface scope"

                findings.append({
                    "type":     "NETWORK_BINDING",
                    "severity": severity,
                    "file":     fpath,
                    "line":     line,
                    "host":     host or "(unresolved)",
                    "port":     port,
                    "snippet":  snippet,
                    "note":     note,
                })
    return findings


def _scan_http_indicators(repo_path: str) -> List[Dict]:
    """
    Emit ONE finding per framework per repo, listing how many files contain it.
    This avoids hundreds of identical INFO lines for monorepos that import
    the same framework everywhere.
    """
    # framework_key → {"label", "severity", "files": []}
    framework_hits: Dict[str, dict] = {}

    for fpath in _iter_source_files(repo_path):
        text_lower = _read(fpath).lower()
        if not text_lower:
            continue
        for keyword, label, severity in HTTP_HEURISTICS:
            if keyword.lower() in text_lower:
                if keyword not in framework_hits:
                    framework_hits[keyword] = {
                        "label":    label,
                        "severity": severity,
                        "files":    [],
                    }
                framework_hits[keyword]["files"].append(fpath)

    findings = []
    for keyword, info in framework_hits.items():
        file_list = info["files"]
        count = len(file_list)
        # Show up to 3 example files; mention total if more
        examples = [os.path.relpath(f, repo_path) for f in file_list[:3]]
        example_str = ", ".join(examples)
        if count > 3:
            example_str += f" … (+{count - 3} more)"
        findings.append({
            "type":     "HTTP_SERVER_INDICATOR",
            "severity": info["severity"],
            "file":     file_list[0],          # representative file for dedup
            "line":     None,
            "note":     f"{info['label']} detected in {count} file(s): {example_str}",
        })
    return findings


# ── Public entry point ────────────────────────────────────────────────────

def analyze_network(name: str, repo_path: str, analysis_path: str) -> int:
    """
    Scan for network bindings and HTTP server usage.
    Writes a structured report to analysis_path.
    Returns the number of findings.
    """
    if not repo_path:
        raise ValueError("repo_path is required")

    os.makedirs(os.path.dirname(analysis_path) or ".", exist_ok=True)

    binding_findings = _scan_bindings(repo_path)
    http_findings    = _scan_http_indicators(repo_path)

    # Only include HTTP indicators that don't duplicate a binding file already reported
    binding_files = {f["file"] for f in binding_findings}
    # HTTP indicators are informational — always include but mark as INFO when
    # the same file already has a binding finding
    all_findings = binding_findings[:]
    for hf in http_findings:
        if hf["file"] not in binding_files or hf["severity"] != "INFO":
            all_findings.append(hf)

    # If there are binding findings AND the http heuristic section would say
    # "nothing found", that's fine — we already have real findings. Only add
    # a "no HTTP server found" note when there are truly zero findings.
    total = len(all_findings)

    sev_order = {"HIGH": 0, "MEDIUM": 1, "INFO": 2}
    all_findings.sort(key=lambda x: (sev_order.get(x["severity"], 9),
                                      x["file"], x.get("line") or 0))

    lines = []
    if total == 0:
        lines.append("0 network findings.")
        lines.append("")
        lines.append("No network bindings or HTTP server patterns detected.")
    else:
        lines.append(f"{total} network finding(s) for {name}:")
        lines.append("")
        for f in all_findings:
            rel = os.path.relpath(f["file"], repo_path)
            loc = f"{rel}:{f['line']}" if f.get("line") else rel
            lines.append(f"[{f['severity']}] {f['type']}")
            lines.append(f"  File   : {loc}")
            if f.get("host"):
                port_str = f":{f['port']}" if f.get("port") else ""
                lines.append(f"  Address: {f['host']}{port_str}")
            if f.get("snippet"):
                lines.append(f"  Code   : {f['snippet']}")
            lines.append(f"  Detail : {f['note']}")
            lines.append("")

    with open(analysis_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return total
