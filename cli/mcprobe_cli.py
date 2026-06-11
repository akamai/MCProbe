"""
MCProbe — CLI entry point
Runs the same LangGraph pipeline as the GUI via command-line switches.

Usage examples:
  python mcprobe_cli.py https://github.com/owner/repo
  python mcprobe_cli.py https://github.com/owner/repo --offline --no-ai
  python mcprobe_cli.py --batch repos.txt --offline
  python mcprobe_cli.py <url> --backend openai
"""
import argparse
import io
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import dotenv_values

from orchestrator import build_app, extract_repo_name, _default_state_extras
from helpers import (parse_ai_findings, generate_html_report, print_cost_summary,
                     estimate_prompt_chars_from_folder, estimate_tokens, MAX_OUTPUT_TOKENS)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(_PROJECT_ROOT, ".env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_modules(args) -> dict:
    ai_only = getattr(args, "ai_only", False)
    return {
        "cfg":     not args.no_cfg and not ai_only,
        "network": not args.no_network and not ai_only,
        "static":  not args.no_static and not ai_only,
        "sse":     not args.no_sse and not ai_only,
        "auth":    not args.no_auth and not ai_only,
        "ai":      not args.no_ai and not args.offline,
    }


def _resolve_value(env_key: str, dot_env: dict, use_os_env: bool) -> str:
    """.env file > (if use_os_env) OS environment > empty."""
    if dot_env.get(env_key):
        return dot_env[env_key]
    if use_os_env:
        return os.getenv(env_key, "")
    return ""


def _prompt_api_config(args, env_file: str) -> dict:
    """
    In online mode, resolve API keys and base URLs.
    Priority: .env file > (if --use-os-env) OS env > interactive prompt.
    Returns dict with anthropic_api_key, openai_api_key, anthropic_base_url, openai_base_url.
    """
    backend = args.backend
    use_os = getattr(args, "use_os_env", False)
    dot_env = dotenv_values(env_file) if os.path.exists(env_file) else {}

    model_key = "ANTHROPIC_MODEL" if backend in ("claude", "claude-agent") else "OPENAI_MODEL"
    result = {
        "anthropic_api_key":  _resolve_value("ANTHROPIC_API_KEY", dot_env, use_os),
        "openai_api_key":     _resolve_value("OPENAI_API_KEY", dot_env, use_os),
        "anthropic_base_url": _resolve_value("ANTHROPIC_BASE_URL", dot_env, use_os),
        "openai_base_url":    _resolve_value("OPENAI_BASE_URL", dot_env, use_os),
        "ai_model":           _resolve_value(model_key, dot_env, use_os),
    }

    # Determine which keys are needed based on backend
    if backend in ("claude", "claude-agent"):
        needed_key = "anthropic_api_key"
        needed_url = "anthropic_base_url"
        label = "Anthropic"
        key_placeholder = "sk-ant-..."
    elif backend == "openai":
        needed_key = "openai_api_key"
        needed_url = "openai_base_url"
        label = "OpenAI"
        key_placeholder = "sk-..."
    else:
        return result

    if not result[needed_key]:
        try:
            print(f"\n[MCPROBE] No {label} API key found in .env or environment.")
            result[needed_key] = input(f"  Enter {label} API Key ({key_placeholder}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[MCPROBE] Cancelled.")
            sys.exit(1)

    if not result[needed_url]:
        try:
            url_input = input(f"  Enter {label} Base URL (press Enter to use default): ").strip()
            if url_input:
                result[needed_url] = url_input
        except (EOFError, KeyboardInterrupt):
            pass

    return result


def _build_initial_state(repo_url: str, repo_name: str, args, api_config: dict) -> dict:
    extras = _default_state_extras()
    extras["offline_mode"]       = args.offline or args.no_ai
    extras["use_os_env"]         = getattr(args, "use_os_env", False)
    extras["ai_backend"]         = args.backend
    extras["ai_model"]           = args.model or api_config.get("ai_model", "")
    extras["anthropic_api_key"]  = api_config["anthropic_api_key"]
    extras["openai_api_key"]     = api_config["openai_api_key"]
    extras["anthropic_base_url"] = api_config["anthropic_base_url"]
    extras["openai_base_url"]    = api_config["openai_base_url"]
    extras["_env_file"]          = api_config.get("_env_file", "")
    extras["calc_cost"]          = getattr(args, "calc_cost", False)
    extras["validate"]           = not getattr(args, "no_validate", False)
    extras["cost_threshold"]     = getattr(args, "cost_threshold", 5.0)
    extras["module_timeout"]     = getattr(args, "timeout", 0) or 0
    extras["enabled_modules"]    = _build_modules(args)

    # Override the base output directory when --output is given
    if args.output:
        base = os.path.abspath(args.output)
        extras["analysis_root"] = os.path.join(base, "analyses", repo_name)

    return {"name": repo_name, "github_repo": repo_url, **extras}


def _resolve_model_name(args, api_config: dict) -> str:
    default = "claude-sonnet-4-6" if args.backend in ("claude", "claude-agent") else "gpt-4o-mini"
    return args.model or api_config.get("ai_model") or default


_print_lock = threading.Lock()
_shutdown = threading.Event()

# ---------------------------------------------------------------------------
# Validation progress bar with elapsed clock
# ---------------------------------------------------------------------------

_BAR_WIDTH = 30


class _ValidationProgress:
    """
    Progress bar with a ticking elapsed-time clock.
    A background thread redraws every second so the user sees the timer
    advance while waiting for API responses.
    """

    def __init__(self, total: int):
        self._total = total
        self._completed = 0
        self._repo_name = ""
        self._start = time.monotonic()
        self._stop = False
        self._lock = threading.Lock()
        self._ticker = threading.Thread(target=self._tick, daemon=True)
        self._ticker.start()

    def _elapsed(self) -> str:
        secs = int(time.monotonic() - self._start)
        m, s = divmod(secs, 60)
        return f"{m}:{s:02d}"

    def _render(self):
        with self._lock:
            pct = self._completed / self._total * 100 if self._total else 100
            filled = int(pct / 100 * _BAR_WIDTH)
            unfilled = _BAR_WIDTH - filled

            GREEN = "\033[32m"
            DIM = "\033[90m"
            RESET = "\033[0m"
            CLEAR = "\033[2K"

            label = f" {self._repo_name}" if self._repo_name else ""
            bar = (f"{CLEAR}\r  Validating: "
                   f"0{GREEN}{'=' * filled}{RESET}"
                   f"{DIM}{'=' * unfilled}{RESET}100"
                   f" ({pct:.0f}%) [{self._elapsed()}]{label}")
            print(bar, end="", flush=True)

    def _tick(self):
        while not self._stop:
            self._render()
            time.sleep(1)

    def update(self, completed: int, repo_name: str = ""):
        with self._lock:
            self._completed = completed
            self._repo_name = repo_name
        self._render()

    def finish(self):
        self._stop = True
        self._ticker.join(timeout=2)
        with self._lock:
            self._completed = self._total
            self._repo_name = ""

        GREEN = "\033[32m"
        RESET = "\033[0m"
        CLEAR = "\033[2K"
        bar = (f"{CLEAR}\r  Validating: "
               f"0{GREEN}{'=' * _BAR_WIDTH}{RESET}100"
               f" (100%) [{self._elapsed()}]")
        print(bar, flush=True)


def _run_threaded_validation(ok_results: list, per_repo_costs: list,
                             num_threads: int, run_validate) -> None:
    """
    Run validation across multiple worker threads, streaming a per-thread
    activity log to stdout (lock-protected).
    """
    import queue as _queue

    work_q: "_queue.Queue" = _queue.Queue()
    for r in ok_results:
        work_q.put(r)

    cost_by_name = {n: c for n, c in per_repo_costs}
    workers = min(num_threads, len(ok_results))
    out_lock = threading.Lock()

    with out_lock:
        print(f"\n[VALIDATOR] Opening {workers} thread(s):")

    def _worker(thread_id: int):
        while True:
            try:
                r = work_q.get_nowait()
            except _queue.Empty:
                with out_lock:
                    print(f"  Thread {thread_id}/{workers} - Done")
                return

            name = r.get("name", "?")
            cost = cost_by_name.get(name, 0.0)
            with out_lock:
                print(f"  Thread {thread_id}/{workers} - working - {name} - ${cost:.4f}")

            try:
                run_validate(r)
            except Exception as e:
                with out_lock:
                    print(f"  {name} - Validation Failed: {e}")
            else:
                with out_lock:
                    print(f"  {name} - Validation Done")

    threads = [threading.Thread(target=_worker, args=(i + 1,), daemon=True)
               for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _safe_exit(code: int):
    stale = [t for t in threading.enumerate()
             if t is not threading.main_thread() and not t.daemon and t.is_alive()]
    if stale:
        sys.stdout.flush()
        os._exit(code)
    sys.exit(code)


def _sigint_handler(signum, frame):
    if _shutdown.is_set():
        os._exit(130)
    print("\n[MCPROBE] Interrupted — press Ctrl+C again to force quit...",
          flush=True)
    _shutdown.set()


def _run_repo(app, repo_url: str, args, api_config: dict,
              threaded: bool = False) -> dict:
    import helpers
    repo_name = extract_repo_name(repo_url)

    header = (f"\n{'='*60}\n"
              f"[MCPROBE] Analyzing: {repo_name}\n"
              f"{'='*60}")

    if threaded:
        buf = io.StringIO()
        helpers._thread_local.buffer = buf
        buf.write(header + "\n")
    else:
        print(header)

    initial_state = _build_initial_state(repo_url, repo_name, args, api_config)
    try:
        result = app.invoke(initial_state)
    except Exception as e:
        if threaded:
            buf.write(f"[MCPROBE] ERROR: {e}\n")
        else:
            print(f"[MCPROBE] ERROR: {e}")
        result = {"_error": str(e), "name": repo_name}
    finally:
        if threaded:
            helpers._thread_local.buffer = None
            with _print_lock:
                print(buf.getvalue(), end="", flush=True)

    return result


def _fmt(val) -> str:
    if val is None or val == -1:
        return "—"
    if val == 0:
        return "0"
    return str(val)


def _print_summary(results: list, out=None):
    if not results:
        return
    p = lambda *a, **kw: print(*a, **kw, file=out) if out else print(*a, **kw)

    col_repo   = max(len(r.get("name", "")) for r in results)
    col_repo   = max(col_repo, 28)

    header = (
        f"{'Repo':<{col_repo}}  {'Lang':<8}  "
        f"{'MCP Flow':>8}  {'Network':>7}  {'Auth':>4}  {'Static':>6}  {'AI':>3}"
    )
    sep = "-" * len(header)
    p(f"\n{sep}")
    p(header)
    p(sep)

    for r in results:
        if "_error" in r:
            p(f"{r['name']:<{col_repo}}  ERROR: {r['_error'][:60]}")
            continue
        ai_done = bool(
            r.get("ai_analysis") and
            "disabled" not in (r.get("ai_analysis") or "").lower()
        )
        bandit  = max(0, r.get("bandit_high", 0))
        semgrep = max(0, r.get("semgrep_issues", 0))
        static  = bandit + semgrep if (bandit + semgrep) > 0 else (
            -1 if r.get("bandit_analysis_path", "") == "" and
                  r.get("semgrep_analysis_path", "") == "" else 0
        )
        p(
            f"{r.get('name', '?'):<{col_repo}}  "
            f"{r.get('language', '?'):<8}  "
            f"{_fmt(r.get('cfg_issues')):>8}  "
            f"{_fmt(r.get('net_issues')):>7}  "
            f"{_fmt(r.get('auth_issues')):>4}  "
            f"{_fmt(static):>6}  "
            f"{'✓' if ai_done else '—':>3}"
        )
    p(sep)


_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


def _load_validation(result: dict) -> dict | None:
    """Load validation_result.json for a result if it exists."""
    import json as _json
    root = result.get("analysis_root", "")
    if not root:
        return None
    json_path = os.path.join(root, "validation_result.json")
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _build_merged_findings(result: dict) -> tuple:
    """
    Merge original AI findings with validation verdicts into one list.

    Returns (merged_findings, stats_dict_or_None).
    Each merged finding has keys: severity, title, file, line,
    and when validated: _verdict, _evidence, _original_severity.
    stats_dict has: confirmed, false_positives, reclassified, discovered.
    """
    ai_data = parse_ai_findings(result.get("ai_analysis", ""))
    original = ai_data.get("findings", [])
    summary = ai_data.get("summary", "")

    vdata = _load_validation(result)
    if not vdata:
        return original, None, summary

    validated = vdata.get("validated_findings", [])
    new_findings = vdata.get("new_findings", [])

    verdict_map = {}
    for v in validated:
        vid = v.get("id")
        if vid is not None:
            verdict_map[vid] = v

    n_confirmed = sum(1 for v in validated if v.get("verdict") == "confirmed")
    n_fp = sum(1 for v in validated if v.get("verdict") == "false_positive")
    n_changed = sum(1 for v in validated if v.get("verdict") == "severity_changed")

    merged = []
    for i, f in enumerate(original, 1):
        v = verdict_map.get(i, {})
        verdict = v.get("verdict", "confirmed")

        if verdict == "false_positive":
            continue

        entry = dict(f)
        entry["_verdict"] = verdict
        entry["_evidence"] = v.get("evidence", "")
        entry["_original_severity"] = f.get("severity", "?")

        if verdict == "severity_changed" and v.get("new_severity"):
            entry["severity"] = v["new_severity"]

        merged.append(entry)

    for nf in new_findings:
        entry = dict(nf)
        entry["_verdict"] = "discovered"
        entry["_evidence"] = ""
        merged.append(entry)

    stats = {
        "confirmed": n_confirmed,
        "false_positives": n_fp,
        "reclassified": n_changed,
        "discovered": len(new_findings),
    }

    return merged, stats, vdata.get("summary", summary or "")


def _print_ai_single(result: dict, out=None):
    """Unified AI + validation output for a single repo."""
    p = lambda *a, **kw: print(*a, **kw, file=out) if out else print(*a, **kw)
    ai_data = parse_ai_findings(result.get("ai_analysis", ""))
    ai_summary = ai_data.get("summary", "")

    if not ai_data.get("findings") and not ai_summary:
        return

    p(f"\n--- AI Security Review ---")
    if ai_summary:
        p(f"  {ai_summary}")

    findings, stats, _ = _build_merged_findings(result)

    if not findings:
        p("\n  No findings.")
        return

    p(f"\n  Findings ({len(findings)}):")

    if stats:
        parts = []
        if stats["confirmed"]:
            parts.append(f"{stats['confirmed']} confirmed")
        if stats["false_positives"]:
            parts.append(f"{stats['false_positives']} false positive(s)")
        if stats["reclassified"]:
            parts.append(f"{stats['reclassified']} reclassified")
        if stats["discovered"]:
            parts.append(f"{stats['discovered']} discovered")
        if parts:
            p(f"  {', '.join(parts)}")

    grouped = {}
    for f in findings:
        sev = f.get("severity", "INFO")
        grouped.setdefault(sev, []).append(f)

    n = 1
    for sev in sorted(grouped, key=lambda s: _SEV_ORDER.get(s, 9)):
        items = grouped[sev]
        p(f"\n  {sev} ({len(items)}):")
        for f in items:
            title = f.get("title", "")
            loc = f.get("file", "")
            if f.get("line"):
                loc += f":{f['line']}"
            verdict = f.get("_verdict", "")
            orig_sev = f.get("_original_severity", "")
            source = "[VALIDATOR]" if verdict == "discovered" else "[ANALYZER]"

            if verdict == "severity_changed" and orig_sev and orig_sev != sev:
                p(f"    {n}. [{orig_sev} -> {sev}] {source} {title}")
            else:
                p(f"    {n}. [{sev}] {source} {title}")
            if loc:
                p(f"              Location: {loc}")

            evidence = f.get("_evidence", "")
            if evidence and verdict != "discovered":
                oneliner = evidence.split("\n")[0].split(". ")[0]
                p(f"              Description: {oneliner}")

            n += 1

    p(f"\n  See HTML report for full details.")


def _print_ai_batch(results: list, out=None):
    """Per-repo validated findings for batch mode. Only CRITICAL/HIGH shown."""
    p = lambda *a, **kw: print(*a, **kw, file=out) if out else print(*a, **kw)
    p(f"\n--- Findings (CRITICAL / HIGH only) ---")
    for r in results:
        if "_error" in r:
            continue
        name = r.get("name", "?")

        all_findings, stats, _ = _build_merged_findings(r)
        high_findings = [f for f in all_findings
                         if f.get("severity", "INFO") in ("CRITICAL", "HIGH")]

        header = f"\n  {name} ({len(high_findings)} high"
        if stats:
            total = len(all_findings)
            header += f" / {total} total"
            parts = []
            if stats["confirmed"]:
                parts.append(f"{stats['confirmed']} confirmed")
            if stats["false_positives"]:
                parts.append(f"{stats['false_positives']} FP")
            if stats["reclassified"]:
                parts.append(f"{stats['reclassified']} reclassified")
            if stats["discovered"]:
                parts.append(f"{stats['discovered']} discovered")
            if parts:
                header += f" — {', '.join(parts)}"
        header += "):"
        p(header)

        if not high_findings:
            p(f"    No CRITICAL/HIGH findings.")
            continue

        for i, f in enumerate(high_findings, 1):
            sev = f.get("severity", "?")
            title = f.get("title", "")
            loc = f.get("file", "")
            if f.get("line"):
                loc += f":{f['line']}"
            verdict = f.get("_verdict", "")
            orig_sev = f.get("_original_severity", "")
            source = "[VALIDATOR]" if verdict == "discovered" else "[ANALYZER]"

            if verdict == "severity_changed" and orig_sev and orig_sev != sev:
                p(f"    {i}. [{orig_sev} -> {sev}] {source} {title}")
            else:
                p(f"    {i}. [{sev}] {source} {title}")
            if loc:
                p(f"              Location: {loc}")
            evidence = f.get("_evidence", "")
            if evidence and verdict != "discovered":
                oneliner = evidence.split("\n")[0].split(". ")[0]
                p(f"              Description: {oneliner}")


def _print_reports(results: list, out=None):
    """Print HTML report paths."""
    p = lambda *a, **kw: print(*a, **kw, file=out) if out else print(*a, **kw)
    paths = [(r.get("name", "?"), r.get("_html_report", ""))
             for r in results if r.get("_html_report")]
    if not paths:
        return
    p(f"\n--- HTML Reports ---")
    for name, path in paths:
        p(f"  {name}: {path}")


def _print_ai_batch_critical(results: list):
    """Print only CRITICAL findings to screen for large batches."""
    critical_count = 0
    print(f"\n--- CRITICAL Findings ---")
    for r in results:
        if "_error" in r:
            continue
        name = r.get("name", "?")
        ai_data = parse_ai_findings(r.get("ai_analysis", ""))
        findings = ai_data.get("findings", [])
        crits = [f for f in findings if f.get("severity") == "CRITICAL"]
        if not crits:
            continue
        print(f"\n  {name} ({len(crits)} critical):")
        for i, f in enumerate(crits, 1):
            title = f.get("title", "")
            loc = f.get("file", "")
            if f.get("line"):
                loc += f":{f['line']}"
            print(f"    {i}. [CRITICAL] {title}")
            if loc:
                print(f"              Location: {loc}")
        critical_count += len(crits)
    if not critical_count:
        print("  None.")
    print()


def _print_validation(results: list, is_batch: bool, out=None):
    """Print validation verdicts for each repo."""
    p = lambda *a, **kw: print(*a, **kw, file=out) if out else print(*a, **kw)

    validated_results = []
    for r in results:
        if "_error" in r:
            continue
        root = r.get("analysis_root", "")
        if not root:
            continue
        json_path = os.path.join(root, "validation_result.json")
        if not os.path.isfile(json_path):
            continue
        try:
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            validated_results.append((r.get("name", "?"), vdata))
        except Exception:
            continue

    if not validated_results:
        return

    p(f"\n--- Validation Results ---")

    for repo_name, vdata in validated_results:
        canary_ok = vdata.get("canary_verified", False)
        validated = vdata.get("validated_findings", [])
        new_findings = vdata.get("new_findings", [])
        n_confirmed = sum(1 for v in validated if v.get("verdict") == "confirmed")
        n_fp = sum(1 for v in validated if v.get("verdict") == "false_positive")
        n_changed = sum(1 for v in validated if v.get("verdict") == "severity_changed")
        warnings = vdata.get("guardrail_warnings", [])

        if is_batch:
            p(f"\n  {repo_name}:")
            indent = "    "
        else:
            indent = "  "

        p(f"{indent}{n_confirmed} confirmed, {n_fp} false positive(s), "
          f"{n_changed} reclassified, {len(new_findings)} new")

        if not canary_ok:
            p(f"{indent}⚠ CANARY FAILED — results may be unreliable")

        if warnings:
            for w in warnings:
                p(f"{indent}⚠ {w}")

        for v in validated:
            vid = v.get("id", "?")
            verdict = v.get("verdict", "unknown")
            orig_sev = v.get("original_severity", "?")
            new_sev = v.get("new_severity", "")
            evidence = v.get("evidence", "")

            if verdict == "confirmed":
                tag = f"[{orig_sev}] CONFIRMED"
            elif verdict == "false_positive":
                tag = f"[{orig_sev}] FALSE POSITIVE"
            elif verdict == "severity_changed" and new_sev:
                tag = f"[{orig_sev} -> {new_sev}] RECLASSIFIED"
            else:
                tag = f"[{orig_sev}] {verdict.upper()}"

            p(f"{indent}  #{vid} {tag}")
            if evidence:
                short = evidence[:120] + ("..." if len(evidence) > 120 else "")
                p(f"{indent}       {short}")

        if new_findings:
            p(f"{indent}  New findings:")
            for nf in new_findings:
                sev = nf.get("severity", "?")
                title = nf.get("title", "")
                loc = nf.get("file", "")
                if nf.get("line"):
                    loc += f":{nf['line']}"
                p(f"{indent}    [{sev}] {title}")
                if loc:
                    p(f"{indent}           {loc}")


def _has_high_findings(results: list) -> bool:
    for r in results:
        if "_error" in r:
            continue
        if (r.get("cfg_issues") or 0) > 0:
            return True
        if (r.get("auth_issues") or 0) > 0:
            return True
        if (r.get("bandit_high") or 0) > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# Post-run: offer to open analysis folder
# ---------------------------------------------------------------------------

def _open_folder(path: str):
    import subprocess
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _prompt_open_folder(results: list, args):
    """Ask the user if they want to open the analysis folder. Enter = yes."""
    # Collect folders that actually exist
    folders = []
    for r in results:
        if "_error" in r:
            continue
        root = r.get("analysis_root", "")
        if not root and args.output:
            root = os.path.join(os.path.abspath(args.output), "analyses",
                                r.get("name", ""))
        if not root:
            root = os.path.join(os.getcwd(), "out", "analyses", r.get("name", ""))
        if os.path.isdir(root):
            folders.append((r.get("name", "?"), root))

    if not folders:
        return

    if len(folders) == 1:
        name, folder = folders[0]
        try:
            answer = input(f"\nOpen analysis folder for {name}? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer in ("", "y", "yes"):
            _open_folder(folder)
    else:
        # Batch: offer to open the parent analyses directory
        parent = os.path.dirname(folders[0][1])
        try:
            answer = input(f"\nOpen analyses folder ({len(folders)} repos)? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if answer in ("", "y", "yes"):
            _open_folder(parent)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="mcprobe",
        description="MCProbe — MCP server vulnerability scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic scan (uses Claude AI by default)\n"
            "  mcprobe_cli.py https://github.com/owner/repo\n"
            "\n"
            "  # Offline scan — no AI review\n"
            "  mcprobe_cli.py https://github.com/owner/repo --offline\n"
            "\n"
            "  # Use OpenAI backend with a specific model\n"
            "  mcprobe_cli.py <url> --backend openai --model gpt-4o\n"
            "\n"
            "  # Use Claude with a specific model\n"
            "  mcprobe_cli.py <url> --backend claude --model claude-opus-4-0-20250115\n"
            "\n"
            "  # Skip specific modules\n"
            "  mcprobe_cli.py <url> --no-static --no-sse --no-auth\n"
            "\n"
            "  # Batch scan from a file\n"
            "  mcprobe_cli.py --batch repos.txt\n"
            "\n"
            "  # Batch scan, skip already-analyzed repos\n"
            "  mcprobe_cli.py --batch repos.txt --skip-analyzed\n"
            "\n"
            "  # Custom .env file and output directory\n"
            "  mcprobe_cli.py <url> --env-file /path/to/.env --output /path/to/results\n"
            "\n"
            "  # Allow reading API keys from OS environment variables\n"
            "  mcprobe_cli.py <url> --use-os-env\n"
            "\n"
            "  # Parallel batch scan with 10 threads\n"
            "  mcprobe_cli.py --batch repos.txt --threads 10\n"
            "\n"
            "  # Estimate AI costs without calling the API\n"
            "  mcprobe_cli.py --batch repos.txt --calc-cost --model claude-opus-4-6\n"
            "\n"
            "  # Scan without AI validation of findings\n"
            "  mcprobe_cli.py <url> --no-validate\n"
            "\n"
            "  # Re-run only AI review on already-analyzed repos\n"
            "  mcprobe_cli.py --batch repos.txt --ai-only\n"
            "\n"
            "Environment variables (read from .env file, or OS env with --use-os-env):\n"
            "  ANTHROPIC_API_KEY     Anthropic API key\n"
            "  ANTHROPIC_BASE_URL    Anthropic API base URL (optional)\n"
            "  ANTHROPIC_MODEL       Anthropic model name (default: claude-sonnet-4-6)\n"
            "  OPENAI_API_KEY        OpenAI API key\n"
            "  OPENAI_BASE_URL       OpenAI API base URL (optional)\n"
            "  OPENAI_MODEL          OpenAI model name (default: gpt-4o-mini)\n"
            "  GH_TOKEN              GitHub token for authenticated cloning (avoids rate limits)\n"
            "  GITHUB_TOKEN          Alternative to GH_TOKEN\n"
        ),
    )

    # ── Repo target ──
    parser.add_argument("repo_url", nargs="?", default=None,
                        help="GitHub URL or local path to analyze")
    parser.add_argument("--batch", metavar="FILE",
                        help="Text file with one GitHub repo URL per line")

    # ── Module toggles ──
    mod = parser.add_argument_group("modules (all enabled by default)")
    mod.add_argument("--no-cfg",     action="store_true", help="Skip MCP flow analyzer")
    mod.add_argument("--no-network", action="store_true", help="Skip network analyzer")
    mod.add_argument("--no-static",  action="store_true", help="Skip Bandit / Semgrep")
    mod.add_argument("--no-sse",     action="store_true", help="Skip SSE / streaming analyzer")
    mod.add_argument("--no-auth",    action="store_true", help="Skip auth & authorization analyzer")
    mod.add_argument("--no-ai",      action="store_true", help="Skip AI security review")
    mod.add_argument("--ai-only",    action="store_true",
                     help="Skip all analyzers, run only AI review on existing reports")

    # ── AI ──
    ai = parser.add_argument_group("AI backend")
    ai.add_argument("--backend", choices=["claude", "openai", "claude-agent"],
                    default="claude", help="AI backend (default: claude)")
    ai.add_argument("--model", metavar="NAME", default="",
                    help="AI model name (default: claude-sonnet-4-6 / gpt-4o-mini)")
    ai.add_argument("--env-file", metavar="PATH", default=None,
                    help="Path to .env file (default: .env next to this script)")
    ai.add_argument("--offline", action="store_true",
                    help="Disable AI review (equivalent to --no-ai)")
    ai.add_argument("--use-os-env", action="store_true",
                    help="Also read API keys/URLs from OS environment variables "
                         "(by default only .env file is used)")
    ai.add_argument("--calc-cost", action="store_true",
                    help="Dry run: scan repos and estimate AI cost without calling the API")
    ai.add_argument("--no-validate", action="store_true",
                    help="Skip AI validation of findings (validation runs by default)")
    ai.add_argument("--cost-threshold", type=float, default=5.0, metavar="$",
                    help="Auto-confirm validation if cost is below threshold in USD "
                         "(default: 5.0; use -1 to run without asking)")

    # ── Output ──
    out = parser.add_argument_group("output")
    out.add_argument("--output", metavar="DIR",
                     help="Base output directory (default: ./out)")
    out.add_argument("--depth", type=int, default=4, metavar="N",
                     help="MCP flow trace depth (default: 4)")
    out.add_argument("--skip-analyzed", "--sa", action="store_true",
                     help="Skip repos that already have an analysis folder")
    out.add_argument("--threads", type=int, default=1, metavar="N",
                     help="Number of parallel workers for batch mode (default: 1)")
    out.add_argument("--timeout", type=int, default=300, metavar="SEC",
                     help="Per-module timeout in seconds; each analyzer gets this "
                          "limit independently (default: 300, 0=no limit)")

    args = parser.parse_args()

    # Validate: need repo_url or --batch
    if not args.repo_url and not args.batch:
        parser.error("provide a repo_url or use --batch FILE")

    # Collect URLs
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"[MCPROBE] ERROR: batch file not found: {args.batch}")
            sys.exit(2)
        with open(args.batch, "r", encoding="utf-8", errors="replace") as f:
            urls = [line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")]
        if not urls:
            print(f"[MCPROBE] No URLs found in {args.batch}")
            sys.exit(2)
        print(f"[MCPROBE] Batch mode — {len(urls)} repo(s) from {args.batch}")
    else:
        urls = [args.repo_url]

    # Resolve .env file location
    env_file = os.path.abspath(args.env_file) if args.env_file else ENV_FILE

    # Resolve API keys/URLs: .env file → (--use-os-env) OS env → prompt
    offline = args.offline or args.no_ai
    if offline or args.calc_cost:
        use_os = args.use_os_env
        dot_env = dotenv_values(env_file) if os.path.exists(env_file) else {}
        model_key = "ANTHROPIC_MODEL" if args.backend in ("claude", "claude-agent") else "OPENAI_MODEL"
        api_config = {
            "anthropic_api_key":  _resolve_value("ANTHROPIC_API_KEY", dot_env, use_os),
            "openai_api_key":     _resolve_value("OPENAI_API_KEY", dot_env, use_os),
            "anthropic_base_url": _resolve_value("ANTHROPIC_BASE_URL", dot_env, use_os),
            "openai_base_url":    _resolve_value("OPENAI_BASE_URL", dot_env, use_os),
            "ai_model":           _resolve_value(model_key, dot_env, use_os),
            "_env_file":          env_file,
        }
    else:
        api_config = _prompt_api_config(args, env_file)
        api_config["_env_file"] = env_file

    if not offline and not args.calc_cost:
        default_model = "claude-sonnet-4-6" if args.backend in ("claude", "claude-agent") else "gpt-4o-mini"
        model_name = args.model or api_config.get("ai_model") or default_model
        print(f"[MCPROBE] Using model: {model_name}")

    want_validate = not getattr(args, "no_validate", False) and not offline
    do_validate = want_validate and not args.calc_cost

    results = []
    errors  = 0
    is_batch = len(urls) > 1
    num_threads = max(1, args.threads) if is_batch else 1

    # Pre-filter: handle --skip-analyzed before dispatching work
    urls_to_scan = []
    for url in urls:
        repo_name = extract_repo_name(url)
        if args.skip_analyzed:
            if args.output:
                analysis_root = os.path.join(os.path.abspath(args.output),
                                             "analyses", repo_name)
            else:
                analysis_root = os.path.join(os.getcwd(), "out", "analyses",
                                             repo_name)
            has_reports = (os.path.isdir(analysis_root)
                          and any(f.endswith(".txt") for f in os.listdir(analysis_root)))
            if has_reports:
                if args.calc_cost:
                    chars = estimate_prompt_chars_from_folder(analysis_root)
                    results.append({"name": repo_name, "prompt_chars": chars})
                    print(f"[MCPROBE] Using existing analysis for {repo_name}")
                else:
                    print(f"[MCPROBE] MCP Server already analyzed - {analysis_root}")
                continue
        urls_to_scan.append(url)

    app = build_app() if urls_to_scan else None

    def _process_url(url, thread_app=None):
        a = thread_app or app
        final = _run_repo(a, url, args, api_config, threaded=(num_threads > 1))
        if "_error" not in final and not args.calc_cost:
            try:
                final["_html_report"] = generate_html_report(final)
            except Exception as e:
                print(f"[MCPROBE] HTML report failed: {e}")
        return final

    if num_threads > 1 and len(urls_to_scan) > 1:
        print(f"[MCPROBE] Running with {num_threads} threads")

        def _worker(url):
            if _shutdown.is_set():
                return {"_error": "cancelled", "name": extract_repo_name(url)}
            thread_app = build_app()
            return _process_url(url, thread_app=thread_app)

        _shutdown.clear()
        prev_handler = signal.signal(signal.SIGINT, _sigint_handler)
        pool = ThreadPoolExecutor(max_workers=num_threads)
        pending = {pool.submit(_worker, url): url for url in urls_to_scan}

        try:
            while pending:
                done = [f for f in pending if f.done()]
                for f in done:
                    url = pending.pop(f)
                    try:
                        final = f.result()
                    except Exception as e:
                        final = {"_error": str(e),
                                 "name": extract_repo_name(url)}
                    results.append(final)
                    if "_error" in final:
                        errors += 1

                if _shutdown.is_set() or not pending:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            _shutdown.set()

        if _shutdown.is_set():
            for f in pending:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            signal.signal(signal.SIGINT, prev_handler)
            n_done = len(urls_to_scan) - len(pending)
            print(f"[MCPROBE] {n_done}/{len(urls_to_scan)} scans completed "
                  f"before interrupt.")
            os._exit(130)

        pool.shutdown(wait=True)
        signal.signal(signal.SIGINT, prev_handler)
    else:
        try:
            for url in urls_to_scan:
                final = _process_url(url)
                results.append(final)
                if "_error" in final:
                    errors += 1
        except KeyboardInterrupt:
            print("\n[MCPROBE] Interrupted.")
            sys.exit(130)

    if not results:
        print("[MCPROBE] No repos analyzed.")
        _safe_exit(0)

    if args.calc_cost:
        print_cost_summary(results, _resolve_model_name(args, api_config),
                           validate=want_validate)
        _safe_exit(0)

    if do_validate:
        from analyzers.ai.validate import estimate_validate_cost, run_validate
        _model = _resolve_model_name(args, api_config)
        ok_results = [r for r in results if "_error" not in r]
        per_repo = [(r.get("name", "?"),
                     estimate_validate_cost(r.get("validate_chars", 0), _model))
                    for r in ok_results]
        total_v_cost = sum(c for _, c in per_repo)

        threshold = getattr(args, "cost_threshold", 5.0)
        if threshold != -1 and total_v_cost >= threshold:
            print(f"\n[MCProbe] Estimated validation cost:")
            for name, cost in per_repo:
                print(f"  {name}: ${cost:.4f}")
            print(f"  {'─' * 40}")
            print(f"  Total: ${total_v_cost:.4f}  (threshold: ${threshold:.2f})")
            try:
                ans = input("[MCProbe] Continue with validation? [y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _safe_exit(0)
            if ans != "y":
                do_validate = False
        else:
            print(f"[MCProbe] Validation cost: ${total_v_cost:.4f} "
                  f"(under ${threshold:.2f} threshold — auto-confirming)")

        if do_validate:
            import helpers as _h
            _orig_dprint = _h.dprint
            _h.dprint = lambda *a, **kw: None
            try:
                if num_threads > 1 and len(ok_results) > 1:
                    _run_threaded_validation(
                        ok_results, per_repo, num_threads, run_validate)
                else:
                    show_bar = sys.stdout.isatty()
                    total = len(ok_results)
                    progress = _ValidationProgress(total) if show_bar else None
                    if progress:
                        progress.update(0)
                    for i, r in enumerate(ok_results):
                        if progress:
                            progress.update(i, repo_name=r.get("name", ""))
                        run_validate(r)
                        if progress:
                            progress.update(i + 1)
                    if progress:
                        progress.finish()
            finally:
                _h.dprint = _orig_dprint

    large_batch = is_batch and len(results) > 10

    if large_batch:
        results_path = os.path.join(os.getcwd(), "results.txt")
        with open(results_path, "w", encoding="utf-8") as rf:
            _print_summary(results, out=rf)
            _print_ai_batch(results, out=rf)
            _print_reports(results, out=rf)
        print(f"[MCPROBE] Full results written to {results_path}")
        _print_summary(results)
        _print_ai_batch(results)
    else:
        _print_summary(results)
        if is_batch:
            _print_ai_batch(results)
        else:
            _print_ai_single(results[0])
        _print_reports(results)

    _prompt_open_folder(results, args)

    if errors:
        exit_code = 2
    elif _has_high_findings(results):
        exit_code = 1
    else:
        exit_code = 0

    _safe_exit(exit_code)


if __name__ == "__main__":
    main()
