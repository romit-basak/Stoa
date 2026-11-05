"""Paths and site rules come from configs, so a new site needs config, not code changes."""

import ast

import yaml

from src.ingest.common import REPO_ROOT

SITES = list(yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())["sites"])


def code_strings(path):
    """String constants in a module, leaving out docstrings, argparse defaults and SOURCE."""
    tree = ast.parse(path.read_text())
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                skip.add(id(body[0].value))
        if isinstance(node, ast.keyword) and node.arg == "default":
            skip.add(id(node.value))
        # SOURCE names a raw-data folder (data/raw/<source>), like melbourne_portal.
        is_source = isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == "SOURCE" for n in node.targets
        )
        if is_source:
            skip.add(id(node.value))
    for node in ast.walk(tree):
        is_str = isinstance(node, ast.Constant) and isinstance(node.value, str)
        if is_str and id(node) not in skip:
            yield node.lineno, node.value


def test_no_site_name_is_written_into_code():
    found = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        for line, s in code_strings(path):
            if s in SITES or any(f"/{site}" in s or f"{site}/" in s for site in SITES):
                found.append(f"{path.relative_to(REPO_ROOT)}:{line} {s!r}")
    assert found == []
