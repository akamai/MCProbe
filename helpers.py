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
# Batch results dashboard (interactive HTML) — shared by the CLI and the
# standalone tools/results_to_html.py generator.
# ---------------------------------------------------------------------------

_BATCH_SNIPPET_CONTEXT = 8


def read_source_snippet(repo_root: str, file: str, line, context: int = _BATCH_SNIPPET_CONTEXT):
    """Return [{"n": lineno, "text": str}, ...] around `line` in
    repo_root/file, or None if the file can't be read."""
    if not repo_root or not file or not line:
        return None
    try:
        path = os.path.join(repo_root, *str(file).split("/"))
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read().splitlines()
    except Exception:
        return None
    line = int(line)
    lo = max(1, line - context)
    hi = min(len(src), line + context)
    return [{"n": i, "text": src[i - 1]} for i in range(lo, hi + 1)]


def next_results_html_path(analyses_dir: str, stem: str = "results") -> str:
    """Return a new, non-colliding versioned path:
    <analyses_dir>/00_<stem>.html, then 01_, 02_, ...  The zero-padded numeric
    prefix keeps the dashboards sorted at the top of the folder, and never
    overwrites a previous run's dashboard."""
    os.makedirs(analyses_dir, exist_ok=True)
    n = 0
    while True:
        cand = os.path.join(analyses_dir, f"{n:02d}_{stem}.html")
        if not os.path.exists(cand):
            return cand
        n += 1


# The renderer expects a payload with precomputed hrefs so it is agnostic to
# where the HTML file lives:
#   payload = {
#     "meta": {total, errors, with_findings, sev_counts:{SEV:int}},
#     "repos": [{name, lang, ai(bool), error, high, total, stats, report_href,
#                findings:[{sev, source, title, location, file, line, description,
#                           verdict, orig_sev, snippet, src_href, github_href}]}]
#   }
_BATCH_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCProbe — Batch Results</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --panel2:#0d1117; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --row:#161b22;
    --chip:#21262d; --hl:rgba(210,153,34,.28);
  }
  [data-theme="light"] {
    --bg:#f6f8fa; --panel:#ffffff; --panel2:#f6f8fa; --border:#d0d7de;
    --text:#1f2328; --muted:#636c76; --accent:#0969da; --row:#ffffff;
    --chip:#eaeef2; --hl:#fff8c5;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg);
         color:var(--text); padding:20px; transition:background .15s,color .15s; }
  h1 { color:var(--accent); font-size:22px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
           gap:10px; margin-bottom:16px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:8px;
          padding:12px; text-align:center; }
  .stat .num { font-size:26px; font-weight:700; }
  .stat .lbl { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .controls { position:sticky; top:0; z-index:5; background:var(--bg);
              border-bottom:1px solid var(--border); padding:10px 0 12px;
              display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:14px; }
  .group { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
  .group > .label { font-size:11px; color:var(--muted); text-transform:uppercase; margin-right:2px; }
  button, .pill { cursor:pointer; border:1px solid var(--border); background:var(--chip);
          color:var(--text); border-radius:20px; padding:5px 12px; font-size:12px; user-select:none; }
  .pill.on { border-color:transparent; color:#fff; }
  #search { border-radius:8px; padding:6px 10px; min-width:200px; }
  .repo { background:var(--row); border:1px solid var(--border); border-radius:8px;
          margin-bottom:8px; overflow:hidden; }
  .repo-head { display:flex; align-items:center; gap:10px; padding:10px 12px; }
  .repo-head input { width:16px; height:16px; cursor:pointer; }
  .repo-name { font-weight:600; }
  .repo-name a { color:var(--accent); text-decoration:none; }
  .repo-meta { color:var(--muted); font-size:12px; display:flex; gap:8px; flex-wrap:wrap;
               margin-left:auto; align-items:center; }
  .tag { background:var(--chip); border-radius:10px; padding:1px 8px; font-size:11px; }
  .tag.err { background:#d6303122; color:#ff7b72; }
  .findings { padding:2px 12px 10px 40px; }
  .finding-wrap { border-top:1px solid var(--border); }
  .finding-wrap:first-child { border-top:none; }
  .finding { padding:7px 6px; cursor:pointer; border-radius:6px; }
  .finding:hover { background:var(--chip); }
  .finding .caret { color:var(--muted); font-size:10px; margin-right:6px; }
  .sev { display:inline-block; padding:1px 8px; border-radius:4px; font-size:11px; font-weight:700; margin-right:6px; }
  .src { color:var(--muted); font-size:11px; margin-left:4px; }
  .vd { font-size:10px; padding:0 6px; border-radius:4px; margin-left:6px; color:#fff; }
  .vd-disc { background:#1f6feb; } .vd-rc { background:#bf8700; } .vd-ok { background:#238636; }
  .loc { font-family:Consolas,monospace; font-size:12px; color:var(--muted); margin:3px 0 0 20px; }
  .drawer { padding:2px 6px 10px 20px; }
  .drawer.hidden { display:none; }
  .desc { color:var(--muted); font-size:12px; margin:4px 0; }
  pre.code { background:var(--panel2); border:1px solid var(--border); border-radius:6px;
             padding:8px 0; overflow-x:auto; margin:6px 0; }
  .cline { font-family:Consolas,monospace; font-size:12px; white-space:pre; padding:0 10px; }
  .cline .ln { display:inline-block; width:40px; color:var(--muted); text-align:right; margin-right:14px; user-select:none; }
  .cline.hl { background:var(--hl); }
  .drawer-actions { display:flex; gap:14px; align-items:center; margin-top:8px; }
  .fp-btn { font-size:11px; padding:3px 10px; border-radius:6px; }
  .filelink { color:var(--accent); font-size:12px; text-decoration:none; }
  .fp-badge { display:none; background:#8957e5; color:#fff; border-radius:4px; padding:0 6px; font-size:10px; font-weight:700; margin-right:6px; }
  .finding-wrap.fp { border-left:3px solid #8957e5; }
  .finding-wrap.fp .fp-badge { display:inline-block; }
  .none { color:var(--muted); font-size:12px; padding:2px 0 6px 0; }
  .hidden { display:none !important; }
  .count { color:var(--muted); font-size:12px; }
  .theme-fixed { position:fixed; top:12px; right:12px; z-index:50; }
</style>
</head><body>
<button id="theme" class="theme-fixed">&#127769; Dark</button>
<h1>MCProbe — Batch Results</h1>
<div class="sub" id="subtitle"></div>
<div class="stats" id="stats"></div>
<div class="controls">
  <div class="group"><span class="label">Severity</span><span id="sevPills"></span></div>
  <div class="group">
    <button id="selAll">All repos</button>
    <button id="selNone">None</button>
    <label class="pill"><input type="checkbox" id="hideEmpty" checked> Hide repos w/o shown findings</label>
    <button id="clearFP">Clear FP marks</button>
  </div>
  <div class="group" style="margin-left:auto"><input id="search" placeholder="Filter repos by name…"></div>
</div>
<div class="count" id="visCount"></div>
<div id="list"></div>
<script>
const DATA = __DATA__;
const SEV_ORDER = ["CRITICAL","HIGH","MEDIUM","LOW","INFO"];
const SEV_COLORS = {CRITICAL:"#d63031",HIGH:"#e17055",MEDIUM:"#fdcb6e",LOW:"#00b894",INFO:"#74b9ff"};
const SEV_TEXT   = {CRITICAL:"#fff",HIGH:"#fff",MEDIUM:"#1a1a1a",LOW:"#1a1a1a",INFO:"#0a0a0a"};
const active = new Set(["CRITICAL"]);
function loadFP(){ try { return JSON.parse(localStorage.getItem("mcprobe-fp")||"[]"); } catch(e){ return []; } }
function saveFP(){ try { localStorage.setItem("mcprobe-fp", JSON.stringify([...fpSet])); } catch(e){} }
const fpSet = new Set(loadFP());
const FKEY = {}; let _gid = 0;
function keyOf(name, f){ return name + "|" + (f.file||"") + "|" + (f.line||"") + "|" + f.title; }
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function repoLink(r){
  if(!r.report_href) return esc(r.name);
  return `<a href="${esc(r.report_href)}" target="_blank" rel="noopener">${esc(r.name)}</a>`;
}
function verdictTag(f){
  if(f.verdict==="discovered") return `<span class="vd vd-disc">discovered</span>`;
  if(f.verdict==="severity_changed") return `<span class="vd vd-rc">reclassified${f.orig_sev?" from "+esc(f.orig_sev):""}</span>`;
  if(f.verdict==="confirmed") return `<span class="vd vd-ok">confirmed</span>`;
  return "";
}
function renderStats(){
  const m = DATA.meta, c = m.sev_counts;
  document.getElementById("subtitle").textContent =
    `${m.total} repos analyzed · ${m.errors} errors · ${m.with_findings} with findings`;
  const tiles = [["Repos",m.total],["Errors",m.errors],["Critical",c.CRITICAL],
                 ["High",c.HIGH],["Medium",c.MEDIUM],["Low",c.LOW]];
  document.getElementById("stats").innerHTML = tiles.map(([l,n]) =>
    `<div class="stat"><div class="num">${n}</div><div class="lbl">${l}</div></div>`).join("");
}
function renderPills(){
  const box = document.getElementById("sevPills");
  let html = SEV_ORDER.map(s => {
    const on = active.has(s);
    const style = on ? `background:${SEV_COLORS[s]};color:${SEV_TEXT[s]}` : "";
    return `<span class="pill sev-pill ${on?"on":""}" data-sev="${s}" style="${style}">${s} (${DATA.meta.sev_counts[s]})</span>`;
  }).join("");
  const fpOn = active.has("FP");
  html += `<span class="pill sev-pill ${fpOn?"on":""}" data-sev="FP" style="${fpOn?"background:#8957e5;color:#fff":""}">FP (${fpSet.size})</span>`;
  box.innerHTML = html;
  box.querySelectorAll(".sev-pill").forEach(p => p.onclick = () => {
    const s = p.dataset.sev; active.has(s) ? active.delete(s) : active.add(s);
    renderPills(); applyFilters();
  });
}
function drawerHTML(r, f, gid){
  const snip = (f.snippet||[]).map(s =>
    `<div class="cline${s.n===f.line?" hl":""}"><span class="ln">${s.n}</span>${esc(s.text)}</div>`).join("");
  const code = snip ? `<pre class="code">${snip}</pre>`
             : (f.location ? `<div class="none">Source not available locally.</div>` : "");
  const desc = f.description ? `<div class="desc">${esc(f.description)}</div>` : "";
  const openFile = f.src_href ? `<a class="filelink" href="${esc(f.src_href)}" target="_blank" rel="noopener">Open full file &#8599;</a>` : "";
  const gh = f.github_href ? `<a class="filelink" href="${esc(f.github_href)}" target="_blank" rel="noopener">GitHub &#8599;</a>` : "";
  const actions = `<div class="drawer-actions"><button class="fp-btn" data-fid="${gid}">Mark as FP</button>${openFile}${gh}</div>`;
  return `<div class="drawer hidden">${desc}${code}${actions}</div>`;
}
function renderList(){
  _gid = 0;
  document.getElementById("list").innerHTML = DATA.repos.map((r,i) => {
    const findings = r.findings.map(f => {
      const gid = _gid++; FKEY[gid] = keyOf(r.name, f);
      return `
      <div class="finding-wrap" data-sev="${f.sev}" data-fid="${gid}">
        <div class="finding">
          <span class="caret">&#9654;</span><span class="fp-badge">FP</span>
          <span class="sev" style="background:${SEV_COLORS[f.sev]||"#999"};color:${SEV_TEXT[f.sev]||"#fff"}">${esc(f.sev)}</span>
          <span>${esc(f.title)}</span><span class="src">[${esc(f.source)}]</span>${verdictTag(f)}
          ${f.location?`<div class="loc">↳ ${esc(f.location)}</div>`:""}
        </div>
        ${drawerHTML(r, f, gid)}
      </div>`;
    }).join("");
    const meta = [];
    if(r.error){ meta.push(`<span class="tag err">ERROR</span>`); }
    else {
      meta.push(`<span class="tag">${esc(r.lang||"?")}</span>`);
      if(r.high!=null) meta.push(`<span class="tag">${r.high} high / ${r.total} total</span>`);
      if(r.stats) meta.push(`<span class="tag">${esc(r.stats)}</span>`);
      meta.push(`<span class="tag">AI ${r.ai?"✓":"—"}</span>`);
    }
    return `
    <div class="repo" data-idx="${i}" data-name="${esc(r.name.toLowerCase())}">
      <div class="repo-head">
        <input type="checkbox" class="repo-cb" checked data-idx="${i}">
        <span class="repo-name">${repoLink(r)}</span>
        <span class="repo-meta">${meta.join("")}</span>
      </div>
      <div class="findings">${findings || (r.error?`<div class="none">${esc(r.error)}</div>`:`<div class="none">No findings.</div>`)}</div>
    </div>`;
  }).join("");
  document.querySelectorAll(".repo-cb").forEach(cb => cb.onchange = applyFilters);
  document.querySelectorAll(".finding").forEach(el => el.onclick = () => {
    const d = el.parentNode.querySelector(".drawer");
    if(d) d.classList.toggle("hidden");
    const caret = el.querySelector(".caret");
    if(caret) caret.innerHTML = d && d.classList.contains("hidden") ? "&#9654;" : "&#9660;";
  });
  document.querySelectorAll(".fp-btn").forEach(b => b.onclick = (e) => {
    e.stopPropagation();
    const key = FKEY[b.dataset.fid]; const wrap = b.closest(".finding-wrap");
    if(fpSet.has(key)){ fpSet.delete(key); wrap.classList.remove("fp"); b.textContent = "Mark as FP"; }
    else { fpSet.add(key); wrap.classList.add("fp"); b.textContent = "Unmark FP"; }
    saveFP(); renderPills(); applyFilters();
  });
  document.querySelectorAll(".finding-wrap").forEach(w => {
    const b = w.querySelector(".fp-btn"); if(!b) return;
    if(fpSet.has(FKEY[b.dataset.fid])){ w.classList.add("fp"); b.textContent = "Unmark FP"; }
  });
  renderPills();
}
function applyFilters(){
  const q = document.getElementById("search").value.trim().toLowerCase();
  const hideEmpty = document.getElementById("hideEmpty").checked;
  let visible = 0;
  document.querySelectorAll(".repo").forEach(el => {
    const cb = el.querySelector(".repo-cb");
    let shown = 0;
    el.querySelectorAll(".finding-wrap").forEach(f => {
      const on = f.classList.contains("fp") ? active.has("FP") : active.has(f.dataset.sev);
      f.classList.toggle("hidden", !on);
      if(on) shown++;
    });
    const nameMatch = !q || el.dataset.name.includes(q);
    let show = cb.checked && nameMatch;
    if(hideEmpty && shown === 0) show = false;
    el.classList.toggle("hidden", !show);
    if(show) visible++;
  });
  document.getElementById("visCount").textContent = `Showing ${visible} repo(s)`;
}
document.getElementById("selAll").onclick = () => { document.querySelectorAll(".repo-cb").forEach(cb => cb.checked = true); applyFilters(); };
document.getElementById("selNone").onclick = () => { document.querySelectorAll(".repo-cb").forEach(cb => cb.checked = false); applyFilters(); };
document.getElementById("hideEmpty").onchange = applyFilters;
document.getElementById("clearFP").onclick = () => {
  fpSet.clear(); saveFP();
  document.querySelectorAll(".finding-wrap.fp").forEach(w => {
    w.classList.remove("fp"); const b = w.querySelector(".fp-btn"); if(b) b.textContent = "Mark as FP";
  });
  renderPills(); applyFilters();
};
document.getElementById("search").oninput = applyFilters;
document.getElementById("theme").onclick = () => {
  const h = document.documentElement; const light = h.getAttribute("data-theme") === "light";
  h.setAttribute("data-theme", light ? "dark" : "light");
  document.getElementById("theme").innerHTML = light ? "&#9728; Light" : "&#127769; Dark";
};
renderStats(); renderPills(); renderList(); applyFilters();
</script>
</body></html>"""


def render_batch_html(payload: dict) -> str:
    """Render the interactive batch dashboard from a prepared payload."""
    return _BATCH_HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Cost estimation (per 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-8":            (15.00, 75.00),
    "claude-opus-4-7":            (15.00, 75.00),
    "claude-opus-4-6":            (15.00, 75.00),
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


def usage_tokens(usage) -> tuple:
    """Extract (input_tokens, output_tokens) from an API usage object.

    Anthropic reports input_tokens/output_tokens; OpenAI reports
    prompt_tokens/completion_tokens. Returns (0, 0) when unavailable.
    """
    if usage is None:
        return 0, 0
    inp = getattr(usage, "input_tokens", None)
    if inp is None:
        inp = getattr(usage, "prompt_tokens", 0)
    out = getattr(usage, "output_tokens", None)
    if out is None:
        out = getattr(usage, "completion_tokens", 0)
    try:
        return int(inp or 0), int(out or 0)
    except (TypeError, ValueError):
        return 0, 0


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