import ast
import os
import subprocess
import tempfile
import shutil
import networkx as nx
import matplotlib.pyplot as plt

from helpers import dprint


DANGEROUS_CALLS_PY = {
    "os": ["system", "popen"],
    "subprocess": ["run", "call", "Popen"],
    "builtins": ["eval", "exec", "open"],
    "requests": ["get", "post", "put", "delete", "request"]
}

JS_KEYWORDS = ["eval", "child_process.exec", "child_process.execSync", "require('child_process')", "fs.readFile", "fs.createReadStream", "fs.open"]

class FunctionCallGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_function = None
        self.exec_calls = []

    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name
        self.graph.add_node(node.name)
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node):
        func_name = self.get_full_func_name(node.func)

        if isinstance(node.func, ast.Name):
            called_func = node.func.id
            if self.current_function:
                self.graph.add_edge(self.current_function, called_func)

        for mod, funcs in DANGEROUS_CALLS_PY.items():
            for fn in funcs:
                if func_name == f"{mod}.{fn}" or func_name == fn:
                    self.exec_calls.append({
                        "function": self.current_function or "<module>",
                        "call": func_name,
                        "lineno": node.lineno
                    })
        self.generic_visit(node)

    def get_full_func_name(self, func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            value = self.get_full_func_name(func_node.value)
            return f"{value}.{func_node.attr}"
        return "unknown"

def analyze_file_for_graph(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=path)
        except SyntaxError:
            return [], None

        builder = FunctionCallGraphBuilder()
        builder.visit(tree)
        return builder.exec_calls, builder.graph

def scan_directory_for_graph(dir_path):
    all_execs = []
    combined_graph = nx.DiGraph()

    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                execs, graph = analyze_file_for_graph(fpath)
                all_execs.extend([dict(e, file=fpath) for e in execs])
                if graph:
                    combined_graph = nx.compose(combined_graph, graph)
    return all_execs, combined_graph

def scan_directory_for_js_risks(dir_path):
    matches = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.endswith(".js") or fname.endswith(".ts"):
                fpath = os.path.join(root, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for lineno, line in enumerate(lines, 1):
                        for keyword in JS_KEYWORDS:
                            if keyword in line:
                                matches.append({
                                    "file": fpath,
                                    "lineno": lineno,
                                    "keyword": keyword,
                                    "line": line.strip()
                                })
    return matches

def draw_call_graph(graph, output_file="call_graph.png"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Call Graph")
    plt.savefig(output_file)
    plt.close()

def is_python_project(repo_path):
    return any(
        os.path.exists(os.path.join(repo_path, fname))
        for fname in ["setup.py", "pyproject.toml", "requirements.txt"]
    )

def is_node_project(repo_path):
    return os.path.exists(os.path.join(repo_path, "package.json"))

def clone_repo(url, dest):
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        dprint(f"[CODE] Failed to clone {url}")
        return False

def analyze_repo_path(repo_path, repo_name,output_dir=None):
    # Default destination folder
    output_dir = output_dir or os.path.join(os.getcwd(), "out", "analyses", repo_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build result file path
    result_file = os.path.join(output_dir, "code_analysis.txt")

    dprint(f"[CODE] Analyzing repository: {repo_name}")
    if is_python_project(repo_path):
        dprint("[CODE] Detected Python project. Analyzing...")
        exec_calls, _ = scan_directory_for_graph(repo_path)
        #result_file = os.path.join("outs\\official", f"{repo_name}_dangerous_calls.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            for call in exec_calls:
                line = f"{call['file']}:{call['lineno']} - {call['call']} inside {call['function']}"
                #print(line)
                f.write(line + "\n")
        #print(f"✅ Analysis written to {result_file}")
    elif is_node_project(repo_path):
        dprint("[CODE] Detected Node.js project. Scanning for JS/TS risks...")
        matches = scan_directory_for_js_risks(repo_path)
        #result_file = os.path.join("outs\\official", f"{repo_name}_js_risks.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            for m in matches:
                line = f"{m['file']}:{m['lineno']} - {m['keyword']} -> {m['line']}"
                #print(line)
                f.write(line + "\n")
        #print(f"✅ JS/TS risks written to {result_file}")
    else:
        with open(result_file, 'w', encoding='utf-8') as f:
            dprint("[CODE] Unknown project type.")
    return result_file

def analyze_repo(repo_url):
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo_path = os.path.join(tmpdir, repo_name)
        result_file = f"outs\\{repo_name}_dangerous_calls.txt"
        print(f"\n📥 Cloning {repo_url}...")
        if not clone_repo(repo_url, repo_path):
            return

        if is_python_project(repo_path):
            print("🐍 Detected Python project. Analyzing...")
            exec_calls, call_graph = scan_directory_for_graph(repo_path)


            graph_file = f"{repo_name}_call_graph.png"

            with open(result_file, 'w', encoding='utf-8') as f:
                for call in exec_calls:
                    line = f"{call['file']}:{call['lineno']} - {call['call']} inside {call['function']}"
                    print(line)
                    f.write(line + "\n")
            #draw_call_graph(call_graph, output_file=graph_file)
            print(f"✅ Analysis written to {result_file}, graph to {graph_file}")

        elif is_node_project(repo_path):
            print("🟦 Detected Node.js project. Scanning for JS/TS risks...")
            matches = scan_directory_for_js_risks(repo_path)
            #result_file = f"outs\\{repo_name}_js_risks.txt"
            with open(result_file, 'w', encoding='utf-8') as f:
                for m in matches:
                    line = f"{m['file']}:{m['lineno']} - {m['keyword']} -> {m['line']}"
                    print(line)
                    f.write(line + "\n")
            print(f"✅ JS/TS risks written to {result_file}")

        else:
            print("❓ Unknown project type.")

if __name__ == "__main__":
    official_dir = os.path.join(os.getcwd(), "official")
    if os.path.isdir(official_dir):
        for folder in os.listdir(official_dir):
            folder_path = os.path.join(official_dir, folder)
            if os.path.isdir(folder_path):
                print(f"\n📦 Analyzing local repo: {folder_path}")
                analyze_repo_path(folder_path, repo_name=folder)