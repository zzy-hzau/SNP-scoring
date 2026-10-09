"""Extract original prompt expressions and complete_json call bindings without execution."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROLE_MODULES = (
    "agents/supervisor.py", "agents/evidence_retrieval.py",
    "agents/hypothesis_reasoning.py", "agents/evidence_critic.py",
    "evidence_extractor.py",
)


def extract(source_root: Path) -> dict:
    modules = []
    for filename in ROLE_MODULES:
        path = source_root / filename
        code = path.read_text(encoding="utf-8")
        tree = ast.parse(code)
        owners = {}

        def locate(node, scope="module"):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope += "." + node.name
            owners[id(node)] = scope
            for child in ast.iter_child_nodes(node):
                locate(child, scope)

        locate(tree)
        bindings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = [ast.unparse(target) for target in node.targets]
                if any("prompt" in name.lower() or "system" in name.lower() for name in names) and isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                    bindings.append({"scope": owners[id(node)], "line": node.lineno, "binding": ", ".join(names), "kind": type(node.value).__name__, "expression": ast.get_source_segment(code, node.value)})
            if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "complete_json":
                bindings.append({
                    "scope": owners[id(node)], "line": node.lineno,
                    "binding": "complete_json arguments", "kind": "LLM call bindings",
                    "positional_arguments": [ast.get_source_segment(code, arg) for arg in node.args],
                    "keyword_arguments": {kw.arg: ast.get_source_segment(code, kw.value) for kw in node.keywords if kw.arg not in {"api_key", "password", "auth_token"}},
                })
        modules.append({"source_module": filename, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "prompt_bindings": bindings})
    return {
        "status": "SOURCE_EXTRACTED_NOT_STANDALONE_AGENT",
        "scope": "Exact source expressions and positional/keyword call bindings; no prompt rewriting or source execution. Dynamic payload construction and full role/tool execution remain separate source dependencies.",
        "modules": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
