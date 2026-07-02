"""
MCProbe — results.txt → interactive HTML (standalone dev tool)

Parses a batch `results.txt` and emits the interactive dashboard using the
SAME renderer the CLI uses (helpers.render_batch_html), so there is only one
HTML/JS template to maintain. Use this to (re)build a dashboard from an old
results.txt; the CLI produces the same thing automatically at the end of a run.

Usage:
    python tools/results_to_html.py results.txt results.html [report_base] [src_base]

    report_base = path prefix (relative to the HTML file) under which each repo's
                  per-repo report.html lives. Default "out/analyses".
    src_base    = path prefix under which cloned sources live, used to embed code
                  snippets and build the "open full file" link. Default "out/all_repos".
"""
import os
import re
import sys

# Allow running as `python tools/results_to_html.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import render_batch_html, read_source_snippet  # noqa: E402

SEV_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

_HDR_RE   = re.compile(r"^\s{2}(\S.*?) \((\d+) high / (\d+) total(?: — (.*?))?\):\s*$")
_FIND_RE  = re.compile(r"^\s+\d+\. \[(\w+)\] \[(\w+)\] (.*)$")
_LOC_RE   = re.compile(r"^\s+Location: (.*)$")
_DESC_RE  = re.compile(r"^\s+Description: (.*)$")
_RPT_RE   = re.compile(r"^\s{2}(\S.*?): (.*report\.html)\s*$")


def parse_results(text: str) -> dict:
    lines = text.splitlines()
    repos: dict = {}

    def get(name: str) -> dict:
        return repos.setdefault(name, {
            "name": name, "lang": "", "ai": False, "error": None,
            "high": None, "total": None, "stats": "", "findings": [],
        })

    section = None
    cur = None
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("--- Findings"):
            section, cur = "findings", None
            continue
        if stripped.startswith("--- HTML Reports"):
            section, cur = "reports", None
            continue

        if section is None:
            if stripped.startswith("Repo") and "Lang" in stripped:
                section = "summary"
            continue

        if section == "summary":
            if not stripped or set(stripped) <= set("-"):
                continue
            if "ERROR:" in line:
                get(line.split()[0])["error"] = line.split("ERROR:", 1)[1].strip()
                continue
            toks = line.split()
            if len(toks) >= 7:
                r = get(toks[0])
                r.update(lang=toks[1], ai=(toks[6] == "✓"))
            continue

        if section == "findings":
            m = _HDR_RE.match(line)
            if m:
                cur = get(m.group(1))
                cur["high"] = int(m.group(2))
                cur["total"] = int(m.group(3))
                cur["stats"] = m.group(4) or ""
                continue
            if stripped == "No CRITICAL/HIGH findings.":
                continue
            mf = _FIND_RE.match(line)
            if mf and cur is not None:
                cur["findings"].append({
                    "sev": mf.group(1), "source": mf.group(2),
                    "title": mf.group(3), "location": "", "description": "",
                })
                continue
            ml = _LOC_RE.match(line)
            if ml and cur and cur["findings"]:
                cur["findings"][-1]["location"] = ml.group(1)
                continue
            md = _DESC_RE.match(line)
            if md and cur and cur["findings"]:
                cur["findings"][-1]["description"] = md.group(1)
            continue

        if section == "reports":
            mr = _RPT_RE.match(line)
            if mr:
                pass  # report path parsed but per-repo href is derived below
    return repos


def build_payload(repos: dict, report_base: str, src_base: str) -> dict:
    report_base = report_base.replace("\\", "/").rstrip("/")
    src_base = src_base.replace("\\", "/").rstrip("/")
    repo_list = sorted(repos.values(), key=lambda r: r["name"].lower())
    sev_counts = {s: 0 for s in SEV_ORDER}

    for r in repo_list:
        name = r["name"]
        r["report_href"] = f"{report_base}/{name}/report.html"
        for f in r["findings"]:
            if f["sev"] in sev_counts:
                sev_counts[f["sev"]] += 1
            m = re.match(r"^(.*?):(\d+)$", f.get("location", ""))
            file = m.group(1) if m else f.get("location", "")
            line = int(m.group(2)) if m else None
            f["file"], f["line"] = file, line
            f["verdict"], f["orig_sev"] = "", ""
            f["github_href"] = ""
            f["src_href"] = f"{src_base}/{name}/{file}" if file else ""
            f["snippet"] = read_source_snippet(os.path.join(src_base, name), file, line)

    meta = {
        "total": len(repo_list),
        "errors": sum(1 for r in repo_list if r["error"]),
        "with_findings": sum(1 for r in repo_list if r["findings"]),
        "sev_counts": sev_counts,
    }
    return {"repos": repo_list, "meta": meta}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]
    report_base = sys.argv[3] if len(sys.argv) > 3 else "out/analyses"
    src_base = sys.argv[4] if len(sys.argv) > 4 else "out/all_repos"

    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    payload = build_payload(parse_results(text), report_base, src_base)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_batch_html(payload))

    m = payload["meta"]
    print(f"[results_to_html] {m['total']} repos, {m['errors']} errors, "
          f"CRITICAL={m['sev_counts']['CRITICAL']} HIGH={m['sev_counts']['HIGH']} "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
