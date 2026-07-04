import io
import re
import os
import json
import logging
import threading
from typing import Optional, Dict as TypingDict, List

from typing_extensions import TypedDict
from helpers import dprint, extract_files_from_report, estimate_prompt_chars_from_folder, usage_tokens


def _silence_sdk_loggers():
    for name in ("anthropic", "openai", "httpx", "httpcore"):
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.WARNING)
        lgr.handlers.clear()
        lgr.propagate = False

_silence_sdk_loggers()

PROGRESS_FILE = os.path.join("out", "analyzed_repos.json")


def load_analyzed_repos() -> set:
    if not os.path.exists(PROGRESS_FILE):
        return set()
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data)
    except Exception:
        return set()


def save_analyzed_repo(repo_url: str):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    repos = load_analyzed_repos()
    repos.add(repo_url)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(repos)), f, indent=2)


def parse_repo_url(url: str) -> tuple:
    """
    Parse a repo URL into (clone_url, ref_or_None).

    Supported input formats:
      https://github.com/owner/repo                 → (url, None)
      https://github.com/owner/repo/tree/v1.1.0     → (base, "v1.1.0")
      https://github.com/owner/repo/blob/main/f.py  → (base, "main")
      https://github.com/owner/repo/commit/<sha>    → (base, "<sha>")
      https://github.com/owner/repo@v1.1.0          → (base, "v1.1.0")
      https://github.com/owner/repo.git@main        → (base, "main")
    """
    url = url.strip().rstrip("/")
    ref = None

    # 1) @ref suffix (custom shorthand):  url@v1.1.0
    #    Only treat @ as a ref separator when it appears after the repo path,
    #    not inside the scheme (git@github.com style SSH URLs are left untouched
    #    because their @ comes right after "git").
    at_idx = url.rfind("@")
    scheme_end = url.find("//")
    if at_idx > 0 and (scheme_end == -1 or at_idx > scheme_end + 2):
        ref = url[at_idx + 1:] or None
        url = url[:at_idx]

    # 2) /tree/<ref>, /blob/<ref>/..., /commit/<sha>/...
    tree_match = re.search(
        r"/(tree|blob|commit)/([^/?#]+).*$", url
    )
    if tree_match:
        ref = ref or tree_match.group(2)   # @ref takes precedence if both given
        url = url[: tree_match.start()]

    # 3) Strip any remaining GitHub web-UI suffixes that carry no ref info
    url = re.sub(
        r"/(releases|actions|issues|pull|pulls|wiki|discussions)(/.*)?" + r"$",
        "",
        url,
    )

    return url, ref


def extract_repo_name(repo_url: str) -> str:
    base_url, ref = parse_repo_url(repo_url)
    parts = base_url.rstrip("/").split("/")
    last = parts[-1].removesuffix(".git") if parts else "unknown"
    name = f"{parts[-2]}_{last}" if len(parts) >= 2 else last
    if ref:
        # Sanitize ref so it's safe as a directory component
        safe_ref = re.sub(r"[^\w.\-]", "_", ref)
        name = f"{name}@{safe_ref}"
    return name


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    name: str
    github_repo: str
    repo_local_path: str
    code_analysis_path: str
    cfg_issues: int
    mcp_tool_count: int         # MCP tool entry points found by the flow analyzer
    needs_oversight: bool       # True when 0 tools found — halt analysis, flag repo
    analysis_root: str
    net_analysis_path: str
    net_issues: int
    bandit_analysis_path: str
    bandit_high: int
    semgrep_analysis_path: str
    semgrep_issues: int
    sse_analysis_path: str
    sse_issues: int
    auth_analysis_path: str
    auth_issues: int
    agent_analysis_path: str   # autonomous agent scanner output
    agent_issues: int
    ai_analysis: str
    language: str
    # Pipeline control
    offline_mode: bool          # skip AI node when True
    ai_backend: str             # "claude", "openai", or "claude-agent"
    ai_model: str               # model name override (empty = use default)
    calc_cost: bool             # dry-run: measure prompt size, skip API call
    ai_only: bool               # --ai-only: analyzers skipped, don't announce them
    refresh: bool               # --refresh: ignore all caches, re-run analyzers + AI
    use_cached_analysis: bool   # runtime flag: reusing cached analyzer reports
    tokens_in: int              # cumulative AI input tokens (review + validation)
    tokens_out: int             # cumulative AI output tokens (review + validation)
    validate_tokens_in: int     # input tokens used by the validation pass only
    validate_tokens_out: int    # output tokens used by the validation pass only
    prompt_chars: int           # char count of AI prompt (set by calc_cost mode)
    validate: bool              # --validate flag
    cost_threshold: float       # --cost-threshold (default 5.0, -1 = no limit)
    interesting_files: List[str]  # absolute paths of interesting source files
    validate_chars: int         # estimated char count for validation prompt
    enabled_modules: TypingDict[str, bool]  # {"cfg": True, "static": True, ...}
    use_os_env: bool            # when False (default), never fall back to os.environ for keys/URLs
    module_timeout: int         # per-module timeout in seconds (0 = no limit)
    # API keys & base URLs — passed through state so os.environ is never mutated at runtime
    anthropic_api_key: str
    openai_api_key: str
    anthropic_base_url: str
    openai_base_url: str


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------

def clone_repo_node(state: State) -> State:
    from git import Repo
    clone_url, ref = parse_repo_url(state["github_repo"])
    name = state["name"]
    base_out_dir = os.path.join(os.getcwd(), "out")
    repos_dir = os.path.join(base_out_dir, "all_repos")
    analyses_dir = os.path.join(base_out_dir, "analyses")
    repo_path = os.path.join(repos_dir, name)
    project_analysis_dir = os.path.join(analyses_dir, name)
    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(analyses_dir, exist_ok=True)
    os.makedirs(project_analysis_dir, exist_ok=True)
    if not os.path.exists(repo_path):
        ref_label = f" @ {ref}" if ref else ""
        dprint(f"[MCPROBE] Cloning {clone_url}{ref_label}")
        gh_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
        auth_url = clone_url
        if gh_token and "github.com" in clone_url:
            auth_url = clone_url.replace("https://github.com",
                                         f"https://x-access-token:{gh_token}@github.com")
        clone_kwargs = {}
        if ref:
            clone_kwargs["branch"] = ref
        try:
            Repo.clone_from(auth_url, repo_path, **clone_kwargs)
        except Exception as e:
            err = str(e)
            if "not found" in err.lower() or "exit code(128)" in err:
                raise RuntimeError(f"Repository not found: {clone_url}")
            raise
    else:
        dprint(f"[MCPROBE] Repo {name} already cloned")
    state["repo_local_path"] = repo_path
    state["analysis_root"] = project_analysis_dir
    return state


def detect_language_node(state: State) -> State:
    repo_path = state["repo_local_path"]
    py_files = js_files = 0
    exclude = {"node_modules", "venv", ".venv"}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if f.endswith(".py"):
                py_files += 1
            elif f.endswith((".js", ".ts", ".jsx", ".tsx")):
                js_files += 1
    if js_files > py_files:
        lang = "js"
    elif py_files > js_files:
        lang = "python"
    else:
        lang = "unknown"
    state["language"] = lang
    dprint(f"[MCPROBE] {state['name']} detected as {lang}")
    return state


_CACHED_REPORTS = ("mcp_flow_analysis.txt", "network_analysis.txt", "sse_analysis.txt",
                   "auth_analysis.txt", "bandit_analysis.txt", "semgrep_analysis.txt")
_ANALYSIS_META = "analysis_meta.json"
_META_KEYS = ("language", "cfg_issues", "mcp_tool_count", "net_issues", "sse_issues",
              "auth_issues", "bandit_high", "semgrep_issues", "needs_oversight")


def load_analysis_meta(analysis_root: str):
    """Return cached analyzer counts for a repo, or None if there's no valid
    cache (requires both the meta file and the flow report to exist)."""
    if not analysis_root:
        return None
    meta_p = os.path.join(analysis_root, _ANALYSIS_META)
    flow_p = os.path.join(analysis_root, "mcp_flow_analysis.txt")
    if not (os.path.isfile(meta_p) and os.path.isfile(flow_p)):
        return None
    try:
        with open(meta_p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_analysis_meta(state: dict):
    """Persist analyzer counts so a later run can reuse them (unless --refresh)."""
    root = state.get("analysis_root", "")
    if not root:
        return
    try:
        os.makedirs(root, exist_ok=True)
        meta = {k: state.get(k) for k in _META_KEYS}
        with open(os.path.join(root, _ANALYSIS_META), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def analyze_repo_code(state: State) -> State:
    if not state.get("enabled_modules", {}).get("cfg", True):
        if not state.get("ai_only"):
            dprint("[CFG] Module disabled — skipping")
        return state

    repo_path = state["repo_local_path"]
    analysis_root = state["analysis_root"]

    # Cache: reuse prior analyzer outputs unless --refresh. This is the first
    # analyzer node, so setting use_cached_analysis here makes every later
    # analyzer node short-circuit (see _timeout_wrapper).
    if not state.get("refresh"):
        meta = load_analysis_meta(analysis_root)
        if meta:
            for k, v in meta.items():
                if v is not None:
                    state[k] = v
            state["use_cached_analysis"] = True
            state["code_analysis_path"] = os.path.join(analysis_root, "mcp_flow_analysis.txt")
            # Repopulate interesting_files from the cached reports so validation
            # still has source context.
            if state.get("validate"):
                files = []
                for fname in _CACHED_REPORTS:
                    files += extract_files_from_report(
                        os.path.join(analysis_root, fname), repo_path)
                state["interesting_files"] = list(dict.fromkeys(files))
            dprint(f"[CACHE] Reusing analyzer outputs for {state['name']} "
                   "— use --refresh to re-run")
            return state

    dprint("[CFG] Running MCP flow analyzer")
    from analyzers.code_analyzer import analyze_repo_path
    from analyzers.mcp_flow_analyzer import analyze_mcp_flow, has_mcp_tool_signal
    os.makedirs(analysis_root, exist_ok=True)
    analyze_repo_path(repo_path, repo_name=state["name"], output_dir=analysis_root)
    mcp_report_path, cfg_count, tool_count = analyze_mcp_flow(repo_path, state["name"], analysis_root)
    state["code_analysis_path"] = mcp_report_path
    state["cfg_issues"] = cfg_count
    state["mcp_tool_count"] = tool_count
    dprint(f"[CFG] Done — {cfg_count} finding(s) across {tool_count} MCP tool(s)")

    # The flow tracer found no traceable entry points. Only flag for manual
    # oversight if the repo shows NO MCP tool/SDK signal at all — otherwise it's
    # a real MCP server the tracer just couldn't resolve, so keep analyzing it.
    if tool_count == 0:
        if has_mcp_tool_signal(repo_path):
            dprint(f"[CFG] 0 traceable tool entry points in {state['name']}, but "
                   "MCP tool/SDK signals present — continuing full analysis")
        else:
            state["needs_oversight"] = True
            dprint(f"[CFG] No MCP tools or SDK signals in {state['name']} — "
                   "flagging MANUAL OVERSIGHT NEEDED; skipping remaining analyzers")
    if state.get("validate"):
        pass  # extract_files_from_report already imported at top level
        _f = extract_files_from_report(mcp_report_path, state["repo_local_path"])
        state["interesting_files"] = list(dict.fromkeys(state.get("interesting_files", []) + _f))
    return state


def analyze_repo_network(state: State) -> State:
    if not state.get("enabled_modules", {}).get("network", True):
        if not state.get("ai_only"):
            dprint("[NET] Module disabled — skipping")
        return state
    dprint("[NET] Running network analyzer")
    from analyzers.network_analysis import analyze_network
    name = state["name"]
    repo_path = state["repo_local_path"]
    analysis_root = state["analysis_root"]
    network_report_path = os.path.join(analysis_root, "network_analysis.txt")
    os.makedirs(analysis_root, exist_ok=True)
    net_count = analyze_network(name=name, repo_path=repo_path, analysis_path=network_report_path)
    state["net_analysis_path"] = network_report_path
    state["net_issues"] = net_count if isinstance(net_count, int) else 0
    dprint(f"[NET] Done — {state['net_issues']} finding(s)")
    if state.get("validate"):
        pass  # extract_files_from_report already imported at top level
        _f = extract_files_from_report(network_report_path, state["repo_local_path"])
        state["interesting_files"] = list(dict.fromkeys(state.get("interesting_files", []) + _f))
    return state


def analyze_repo_sse(state: State) -> State:
    if not state.get("enabled_modules", {}).get("sse", True):
        if not state.get("ai_only"):
            dprint("[SSE] Module disabled — skipping")
        return state
    dprint("[SSE] Running SSE/streaming communication analyzer")
    from analyzers.sse_analyzer import analyze_sse
    repo_path = state["repo_local_path"]
    analysis_root = state["analysis_root"]
    sse_path = os.path.join(analysis_root, "sse_analysis.txt")
    try:
        issues = analyze_sse(repo_path, sse_path)
        state["sse_analysis_path"] = sse_path
        state["sse_issues"] = issues
        dprint(f"[SSE] {issues} issue(s) found")
        if state.get("validate"):
            pass  # extract_files_from_report already imported at top level
            _f = extract_files_from_report(sse_path, state["repo_local_path"])
            state["interesting_files"] = list(dict.fromkeys(state.get("interesting_files", []) + _f))
    except Exception as e:
        dprint(f"[SSE] Failed: {e}")
        state["sse_analysis_path"] = ""
        state["sse_issues"] = -1
    return state


def analyze_repo_auth(state: State) -> State:
    if not state.get("enabled_modules", {}).get("auth", True):
        if not state.get("ai_only"):
            dprint("[AUTH] Module disabled — skipping")
        return state
    dprint("[AUTH] Running authentication & authorization analyzer")
    from analyzers.auth_analyzer import analyze_auth
    repo_path = state["repo_local_path"]
    analysis_root = state["analysis_root"]
    auth_path = os.path.join(analysis_root, "auth_analysis.txt")
    try:
        issues = analyze_auth(repo_path, auth_path)
        state["auth_analysis_path"] = auth_path
        state["auth_issues"] = issues
        dprint(f"[AUTH] {issues} issue(s) found")
        if state.get("validate"):
            pass  # extract_files_from_report already imported at top level
            _f = extract_files_from_report(auth_path, state["repo_local_path"])
            state["interesting_files"] = list(dict.fromkeys(state.get("interesting_files", []) + _f))
    except Exception as e:
        dprint(f"[AUTH] Failed: {e}")
        state["auth_analysis_path"] = ""
        state["auth_issues"] = -1
    return state


def analyze_repo_bandit(state: State) -> State:
    if not state.get("enabled_modules", {}).get("static", True):
        if not state.get("ai_only"):
            dprint("[STATIC] Module disabled — skipping Bandit")
        return state
    dprint("[STATIC] Running Bandit static analyzer")
    from analyzers.bandit_analyzer import analyze_with_bandit
    repo_path = state["repo_local_path"]
    analysis_root = state["analysis_root"]
    bandit_path = os.path.join(analysis_root, "bandit_analysis.txt")
    # Use the language already detected by the pipeline rather than the
    # root-dir-only is_python_project() check, which fails for monorepos
    # whose pyproject.toml / setup.py live under src/ or sub-packages.
    lang = state.get("language", "")
    if lang != "python":
        dprint(f"[STATIC] Skipping Bandit — language is '{lang}', not Python")
        state["bandit_analysis_path"] = ""
        return state
    highs = analyze_with_bandit(repo_path, bandit_path)
    state["bandit_high"] = highs
    state["bandit_analysis_path"] = bandit_path
    if state.get("validate"):
        pass  # extract_files_from_report already imported at top level
        _f = extract_files_from_report(bandit_path, state["repo_local_path"])
        state["interesting_files"] = list(dict.fromkeys(state.get("interesting_files", []) + _f))
    return state


def analyze_repo_semgrep(state: State) -> State:
    if not state.get("enabled_modules", {}).get("static", True):
        if not state.get("ai_only"):
            dprint("[STATIC] Module disabled — skipping Semgrep")
        return state
    dprint("[STATIC] Running Semgrep static analyzer")
    from analyzers.semgrep_analyzer import analyze_js_repo
    repo_path = state["repo_local_path"]
    analysis_root = state["analysis_root"]
    semgrep_path = os.path.join(analysis_root, "semgrep_analysis.txt")
    try:
        issues = analyze_js_repo(repo_path, semgrep_path)
        state["semgrep_analysis_path"] = semgrep_path
        state["semgrep_issues"] = issues
        if state.get("validate"):
            pass  # extract_files_from_report already imported at top level
            _f = extract_files_from_report(semgrep_path, state["repo_local_path"])
            state["interesting_files"] = list(dict.fromkeys(state.get("interesting_files", []) + _f))
    except Exception as e:
        dprint(f"[STATIC] Semgrep failed: {e}")
        state["semgrep_analysis_path"] = ""
        state["semgrep_issues"] = -1
    return state


# ---------------------------------------------------------------------------
# AI node — supports both Claude and OpenAI
# ---------------------------------------------------------------------------

def _read_report(path: str, max_chars: int = 120_000) -> str:
    if not path:
        return ""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            if len(data) > max_chars:
                head = data[: int(max_chars * 0.7)]
                tail = data[-int(max_chars * 0.3):]
                return head + "\n\n...[TRUNCATED]...\n\n" + tail
            return data
    except Exception:
        pass
    return ""


def use_ai(state: State) -> State:
    if state.get("needs_oversight", False):
        dprint("[AI] Manual oversight needed (no MCP tools) — skipping AI review")
        state["ai_analysis"] = "Skipped: manual oversight needed — no MCP tools detected."
        return state

    if state.get("offline_mode", False):
        dprint("[AI] Offline mode — skipping AI review")
        state["ai_analysis"] = "Offline mode: AI review disabled."
        if state.get("validate"):
            from analyzers.ai.validate import estimate_validate_chars
            state["validate_chars"] = estimate_validate_chars(state)
        return state

    if not state.get("enabled_modules", {}).get("ai", True):
        dprint("[AI] AI module disabled — skipping")
        state["ai_analysis"] = "AI review disabled."
        if state.get("validate"):
            from analyzers.ai.validate import estimate_validate_chars
            state["validate_chars"] = estimate_validate_chars(state)
        return state

    analysis_root = state.get("analysis_root", "")
    repo_name = state.get("name", "<unknown>")

    existing = os.path.join(analysis_root, "ai_security_review.json") if analysis_root else ""
    if existing and os.path.isfile(existing):
        try:
            with open(existing, "r", encoding="utf-8", errors="replace") as f:
                cached = f.read().strip()
        except Exception:
            cached = ""
        # Never reuse a cached review that represents a failure or a placeholder
        # (e.g. a prior run where the AI call errored) — regenerate instead.
        low = cached.lower()
        is_bad = (not cached) or any(m in low for m in (
            "analysis failed", "scan failed", "skipped:", "offline mode:",
            "ai review disabled", "unknown ai backend"))
        if cached and not is_bad and not state.get("refresh"):
            state["ai_analysis"] = cached
            dprint(f"[AI] Using existing review for {repo_name}")
            if state.get("calc_cost", False):
                state["prompt_chars"] = estimate_prompt_chars_from_folder(analysis_root)
            if state.get("validate"):
                from analyzers.ai.validate import estimate_validate_chars
                state["validate_chars"] = estimate_validate_chars(state)
            return state
        if cached and state.get("refresh"):
            dprint(f"[AI] --refresh — ignoring cached review for {repo_name}, re-running")
        elif cached:
            dprint(f"[AI] Cached review for {repo_name} was a failure/placeholder — re-running")

    dprint("[AI] Running AI security review")
    repo_path = state.get("repo_local_path", "")
    language = (state.get("language") or "unknown").lower()

    # Map state keys to well-known filenames so --ai-only can discover
    # existing reports when the analyzer nodes were skipped.
    _FALLBACK = {
        "code_analysis_path":    "mcp_flow_analysis.txt",
        "net_analysis_path":     "network_analysis.txt",
        "sse_analysis_path":     "sse_analysis.txt",
        "auth_analysis_path":    "auth_analysis.txt",
        "bandit_analysis_path":  "bandit_analysis.txt",
        "semgrep_analysis_path": "semgrep_analysis.txt",
    }

    def _resolve(key):
        p = state.get(key, "")
        if p:
            return p
        if analysis_root:
            fb = os.path.join(analysis_root, _FALLBACK[key])
            if os.path.isfile(fb):
                return fb
        return ""

    candidate_reports = [
        ("CFG / Command Injection Analysis", _resolve("code_analysis_path")),
        ("Network Analysis",                 _resolve("net_analysis_path")),
        ("SSE / Streaming Analysis",         _resolve("sse_analysis_path")),
        ("Auth & Authorization Analysis",    _resolve("auth_analysis_path")),
        ("Bandit Static Analysis",           _resolve("bandit_analysis_path")),
        ("Semgrep Static Analysis",          _resolve("semgrep_analysis_path")),
    ]

    report_sections = []
    for title, p in candidate_reports:
        body = _read_report(p)
        if body.strip():
            report_sections.append(f"## {title}\n\n{body}\n")
        else:
            report_sections.append(f"## {title}\n\n(No findings or report unavailable.)\n")

    combined_report = "\n\n".join(report_sections).strip()

    bandit_high = state.get("bandit_high", 0)
    semgrep_issues = state.get("semgrep_issues", 0)
    sse_issues = state.get("sse_issues", 0)
    auth_issues = state.get("auth_issues", 0)

    prompt = f"""You are an expert software security auditor specializing in MCP (Model Context Protocol) servers.

You are reviewing automated static-analysis outputs and must produce a structured security review.
Be concrete — cite specific files/patterns from the reports. Prioritize by severity.

Repository: {repo_name}
Detected language: {language}

Scanner signals:
- Bandit HIGH findings: {bandit_high}
- Semgrep issues: {semgrep_issues}
- SSE/streaming issues: {sse_issues}
- Auth/authz issues: {auth_issues}

You MUST respond with ONLY valid JSON — no markdown, no commentary, no code fences.
Use this exact schema:

{{
  "summary": "2-4 sentence executive summary",
  "findings": [
    {{
      "id": 1,
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "title": "Short one-line title",
      "file": "relative/path.py",
      "line": 42,
      "description": "What is wrong and why it is exploitable",
      "recommendation": "Concrete fix"
    }}
  ]
}}

Rules:
- "findings" must be an array (empty [] if nothing found).
- "id" is a sequential integer starting at 1.
- "file" and "line" may be empty string and 0 if repo-wide.
- Sort findings by severity (CRITICAL first).
- Severity guidance: SQL injection and user-controlled code execution
  (eval/exec/compile, or a shell/subprocess call built from untrusted input)
  are CRITICAL.

--- BEGIN REPORTS ---
{combined_report}
--- END REPORTS ---
"""

    if state.get("calc_cost", False):
        state["prompt_chars"] = len(prompt)
        state["ai_analysis"] = ""
        dprint(f"[AI] Cost calc mode — prompt is {len(prompt):,} chars")
        if state.get("validate"):
            from analyzers.ai.validate import estimate_validate_chars
            state["validate_chars"] = estimate_validate_chars(state)
        return state

    backend = state.get("ai_backend", os.getenv("MCPROBE_AI_BACKEND", "claude")).lower()
    ai_output = ""
    ai_failed = False

    _use_os = state.get("use_os_env", False)
    _silence_sdk_loggers()

    if backend == "claude":
        try:
            import anthropic
            api_key = state.get("anthropic_api_key") or (os.getenv("ANTHROPIC_API_KEY", "") if _use_os else "")
            base_url = state.get("anthropic_base_url") or (os.getenv("ANTHROPIC_BASE_URL") if _use_os else None) or None
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = anthropic.Anthropic(**client_kwargs)
            model = state.get("ai_model") or os.getenv("MCPROBE_CLAUDE_MODEL", "claude-sonnet-4-6")
            _silence_sdk_loggers()
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                system="You are an expert application security reviewer specializing in MCP servers. Respond ONLY with valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            )
            ai_output = message.content[0].text.strip()
            _ti, _to = usage_tokens(getattr(message, "usage", None))
            state["tokens_in"] = state.get("tokens_in", 0) + _ti
            state["tokens_out"] = state.get("tokens_out", 0) + _to
        except Exception as e:
            ai_output = f'{{"summary":"Claude analysis failed: {type(e).__name__}: {e}","findings":[]}}'
            ai_failed = True

    elif backend == "openai":
        try:
            from openai import OpenAI
            api_key = state.get("openai_api_key") or (os.getenv("OPENAI_API_KEY", "") if _use_os else "")
            base_url = state.get("openai_base_url") or (os.getenv("OPENAI_BASE_URL") if _use_os else None) or None
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            model = state.get("ai_model") or os.getenv("MCPROBE_OPENAI_MODEL", "gpt-4o-mini")
            _silence_sdk_loggers()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert application security reviewer. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            ai_output = (response.choices[0].message.content or "").strip()
            _ti, _to = usage_tokens(getattr(response, "usage", None))
            state["tokens_in"] = state.get("tokens_in", 0) + _ti
            state["tokens_out"] = state.get("tokens_out", 0) + _to
        except Exception as e:
            ai_output = f'{{"summary":"OpenAI analysis failed: {type(e).__name__}: {e}","findings":[]}}'
            ai_failed = True

    elif backend == "claude-agent":
        try:
            from analyzers.ai.claude_agent_scanner import run_agent_scan
            ai_output, agent_count = run_agent_scan(state)
            state["agent_issues"] = agent_count
            state["agent_analysis_path"] = os.path.join(analysis_root, "agent_analysis.txt")
        except Exception as e:
            ai_output = f"Agent scan failed: {type(e).__name__}: {e}"
            state["agent_issues"] = -1
            ai_failed = True

    else:
        ai_output = f'{{"summary":"Unknown AI backend: {backend}","findings":[]}}'
        ai_failed = True

    state["ai_analysis"] = ai_output

    # Only persist SUCCESSFUL reviews — never cache a failure, or the next run
    # would silently reuse it and report zero findings.
    if analysis_root and not ai_failed:
        try:
            os.makedirs(analysis_root, exist_ok=True)
            out_path = os.path.join(analysis_root, "ai_security_review.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(ai_output)
        except Exception:
            pass

    if ai_failed:
        dprint(f"[AI] ⚠ FAILED for {repo_name} — {ai_output[:200]}")
    else:
        dprint(f"[AI] Review complete — saved to {analysis_root}")
    if state.get("validate"):
        from analyzers.ai.validate import estimate_validate_chars
        state["validate_chars"] = estimate_validate_chars(state)
    return state


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------

def route_by_language(state: State):
    lang = (state.get("language") or "").lower()
    if lang == "python":
        return "analyze_bandit"
    if lang in ("js", "javascript", "ts", "typescript"):
        return "analyze_semgrep"
    return "analyze_sse"


def route_after_static(state: State):
    """After bandit/semgrep, always continue to SSE."""
    return "analyze_sse"


def route_to_ai_or_end(state: State):
    """After auth analysis: go to AI unless offline or flagged for oversight."""
    from langgraph.graph import END as _END
    if (state.get("offline_mode", False)
            or state.get("needs_oversight", False)
            or not state.get("enabled_modules", {}).get("ai", True)):
        return _END
    return "use_ai"


# ---------------------------------------------------------------------------
# Per-module timeout wrapper
# ---------------------------------------------------------------------------

def _timeout_wrapper(func, label):
    """Run a pipeline node with a per-module timeout.

    The actual work runs in a daemon thread. If it exceeds the timeout the
    node is skipped and the original state is returned unchanged.
    """
    def wrapper(state):
        # Skip remaining analyzers when the repo is flagged for manual oversight
        # (no MCP tools) or when reusing cached analyzer outputs.
        if state.get("needs_oversight") or state.get("use_cached_analysis"):
            return state
        timeout = state.get("module_timeout", 0)
        if not timeout:
            return func(state)
        state_copy = dict(state)
        result_box = [None]

        def target():
            result_box[0] = func(state_copy)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            dprint(f"[{label}] TIMEOUT after {timeout}s — skipping")
            return state
        return result_box[0]

    return wrapper


# ---------------------------------------------------------------------------
# Build the LangGraph app
# ---------------------------------------------------------------------------

def build_app():
    from langgraph.graph import StateGraph, START, END
    builder = StateGraph(State)

    builder.add_node("clone", clone_repo_node)
    builder.add_node("detect_language", detect_language_node)
    builder.add_node("analyze_code", _timeout_wrapper(analyze_repo_code, "CFG"))
    builder.add_node("analyze_network", _timeout_wrapper(analyze_repo_network, "NET"))
    builder.add_node("analyze_bandit", _timeout_wrapper(analyze_repo_bandit, "BANDIT"))
    builder.add_node("analyze_semgrep", _timeout_wrapper(analyze_repo_semgrep, "SEMGREP"))
    builder.add_node("analyze_sse", _timeout_wrapper(analyze_repo_sse, "SSE"))
    builder.add_node("analyze_auth", _timeout_wrapper(analyze_repo_auth, "AUTH"))
    builder.add_node("use_ai", use_ai)

    builder.add_edge(START, "clone")
    builder.add_edge("clone", "detect_language")
    builder.add_edge("detect_language", "analyze_code")
    builder.add_edge("analyze_code", "analyze_network")
    builder.add_conditional_edges("analyze_network", route_by_language,
                                  {"analyze_bandit": "analyze_bandit",
                                   "analyze_semgrep": "analyze_semgrep",
                                   "analyze_sse": "analyze_sse"})
    builder.add_edge("analyze_bandit", "analyze_sse")
    builder.add_edge("analyze_semgrep", "analyze_sse")
    builder.add_edge("analyze_sse", "analyze_auth")
    builder.add_conditional_edges("analyze_auth", route_to_ai_or_end,
                                  {"use_ai": "use_ai", END: END})
    builder.add_edge("use_ai", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _default_state_extras() -> dict:
    return {
        "repo_local_path": "",
        "analysis_root": "",
        "code_analysis_path": "",
        "cfg_issues": -1,
        "mcp_tool_count": -1,
        "needs_oversight": False,
        "net_analysis_path": "",
        "net_issues": -1,
        "bandit_analysis_path": "",
        "bandit_high": 0,
        "semgrep_analysis_path": "",
        "semgrep_issues": 0,
        "sse_analysis_path": "",
        "sse_issues": 0,
        "auth_analysis_path": "",
        "auth_issues": 0,
        "agent_analysis_path": "",
        "agent_issues": -1,
        "ai_analysis": "",
        "language": "",
        "offline_mode": False,
        "use_os_env": False,
        "ai_backend": os.getenv("MCPROBE_AI_BACKEND", "claude"),
        "ai_model": "",
        "calc_cost": False,
        "ai_only": False,
        "refresh": False,
        "use_cached_analysis": False,
        "tokens_in": 0,
        "tokens_out": 0,
        "validate_tokens_in": 0,
        "validate_tokens_out": 0,
        "prompt_chars": 0,
        "validate": True,
        "cost_threshold": 5.0,
        "interesting_files": [],
        "validate_chars": 0,
        "module_timeout": 0,
        "enabled_modules": {
            "cfg": True, "network": True, "static": True,
            "sse": True, "auth": True, "ai": True,
        },
        # Keys & base URLs default to empty; GUI/CLI passes them through state
        "anthropic_api_key": "",
        "openai_api_key":    "",
        "anthropic_base_url": "",
        "openai_base_url":    "",
    }


def run_all_repos(repo_urls: list,
                   offline_mode: bool = False,
                   ai_backend: str = None,
                   enabled_modules: dict = None):
    dprint(f"[MCPROBE] Found {len(repo_urls)} repos")
    analyzed = load_analyzed_repos()
    risky_repos = []

    for repo_url in repo_urls:
        if repo_url in analyzed:
            print(f"[MCPROBE] Skipping {repo_url} (already analyzed)")
            continue
        repo_name = extract_repo_name(repo_url)
        dprint(f"\n[MCPROBE] Starting analysis: {repo_name}")
        extras = _default_state_extras()
        extras["offline_mode"] = offline_mode
        if ai_backend:
            extras["ai_backend"] = ai_backend
        if enabled_modules:
            extras["enabled_modules"] = enabled_modules
        initial_state = {"name": repo_name, "github_repo": repo_url, **extras}
        try:
            final_state = app.invoke(initial_state)
            save_analyzed_repo(repo_url)
            bandit_high = final_state.get("bandit_high", 0)
            if bandit_high and bandit_high > 0:
                risky_repos.append((repo_name, bandit_high))
            dprint(f"[MCPROBE] Done: {repo_name}")
        except Exception as e:
            dprint(f"[MCPROBE] Failed to analyze {repo_name}: {e}")

    if not risky_repos:
        dprint("[MCPROBE] All scanned repos are clean (no high-confidence Bandit findings).")
    else:
        print("\n[MCPROBE] Repos with high-severity Bandit findings:")
        for name, count in sorted(risky_repos, key=lambda x: -x[1]):
            print(f"  {name}: {count} issue(s)")
        print(f"\n[MCPROBE] {len(risky_repos)} / {len(repo_urls)} repos flagged")
