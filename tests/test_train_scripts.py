"""Smoke tests for the training scripts themselves (no GPU, no downloads).

* every train.py must be valid Python and keep its `if __name__ == "__main__":` guard,
  so importing it never starts a training run;
* scripts whose top-level imports are installed are actually imported, which checks
  that their `genai_common.*` imports resolve and that module-level configuration parses.
"""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in ROOT.glob("*/*/train.py"))


def test_all_projects_found():
    assert len(SCRIPTS) >= 8


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.parent.name)
def test_script_has_docstring_and_main_guard(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), "module docstring explaining the project is required"
    guards = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)]
    assert guards, "train.py must not run on import"
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("__"):
            assert ast.get_docstring(node), f"{node.name} needs a docstring"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.parent.name)
def test_script_imports_without_running(path):
    spec = importlib.util.spec_from_file_location(f"train_{path.parent.name.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "genai_common":
            raise
        pytest.skip(f"optional dependency not installed: {e.name}")
    assert callable(module.main)
