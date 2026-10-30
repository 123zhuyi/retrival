#!/usr/bin/env python3
"""Validate a npu-kernelbench solution.json file.

Checks:
1. solution.json parses against the Solution Pydantic model.
2. The entry file exists in sources and defines the entry function.
3. No forbidden build/package files appear in sources.
4. Source paths are relative and do not contain '..'.

Usage:
    uv run python validate_solution.py path/to/solution.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def _has_function(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return True
    return False


def _ensure_project_root_on_path() -> None:
    """Add the project src directory to sys.path if npu_kernelbench isn't importable."""
    try:
        import npu_kernelbench  # noqa: F401
    except ImportError:
        src_dir = Path(__file__).resolve().parents[4] / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))


def validate(path: Path) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    _ensure_project_root_on_path()
    try:
        from npu_kernelbench.data import Solution
    except ImportError as e:
        return False, [f"Cannot import npu_kernelbench.data: {e}. Run from project root with 'uv run'."], []

    try:
        solution = Solution(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        return False, [f"Failed to validate solution.json: {e}"], []

    # Entry-point conventions
    entry_file = solution.spec.entry_point.split("::")[0]
    entry_symbol = solution.spec.entry_point.split("::")[-1]

    if entry_file != "main.py":
        warnings.append(
            f"Recommended entry file is 'main.py', got '{entry_file}'"
        )
    if entry_symbol != "run":
        warnings.append(
            f"Recommended entry function is 'run', got '{entry_symbol}'"
        )

    source_paths = {s.path for s in solution.sources}
    if "main.py" not in source_paths:
        warnings.append("No 'main.py' found in sources")

    # Per-source checks
    forbidden_names = {
        "CMakeLists.txt",
        "Makefile",
        "setup.py",
        "pyproject.toml",
        "__init__.py",
    }
    forbidden_extensions = {".sh", ".mk", ".cmake"}

    for src in solution.sources:
        src_path = Path(src.path)
        if src_path.is_absolute():
            errors.append(f"Absolute source path not allowed: {src.path}")
        if ".." in src_path.parts:
            errors.append(f"Parent directory traversal not allowed: {src.path}")
        if src_path.name in forbidden_names:
            errors.append(f"Forbidden build/package file in sources: {src.path}")
        if src_path.suffix in forbidden_extensions:
            errors.append(f"Forbidden build script extension in sources: {src.path}")

    # Entry function check
    entry_source = solution.get_entry_source()
    if entry_source is not None:
        if not _has_function(entry_source.content, entry_symbol):
            errors.append(
                f"Entry function '{entry_symbol}' not found in '{entry_file}'"
            )

    return len(errors) == 0, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a solution.json file")
    parser.add_argument("solution", type=Path, help="Path to solution.json")
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output"
    )
    args = parser.parse_args()

    if not args.solution.exists():
        print(f"File not found: {args.solution}", file=sys.stderr)
        return 2

    ok, errors, warnings = validate(args.solution)

    if args.json:
        print(json.dumps({"valid": ok, "errors": errors, "warnings": warnings}, indent=2, ensure_ascii=False))
    else:
        if ok and not warnings:
            print(f"{args.solution}: OK")
        elif ok:
            print(f"{args.solution}: OK (with warnings)")
            for warn in warnings:
                print(f"  ! {warn}")
        else:
            print(f"{args.solution}: INVALID", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            for warn in warnings:
                print(f"  ! {warn}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
