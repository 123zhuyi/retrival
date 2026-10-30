#!/usr/bin/env python3
"""Pack kernel source files into a npu-kernelbench solution.json.

This is a convenience wrapper around the project-level pack script with
post-packing validation via the Solution Pydantic model.

Usage:
    uv run python pack_solution.py \
        --name my_matmul \
        --definition matmul_triton \
        --source main.py \
        --source kernel.py \
        --language triton \
        --hardware ascend910b \
        --output my_solutions/level2/MatmulTriton/solution.json

For AscendC:
    uv run python pack_solution.py \
        --name vector_add_ascendc \
        --definition vector_add \
        --source op_kernel/add_custom.cpp \
        --source op_host/my_add.cpp \
        --source main.py \
        --language ascendc \
        --hardware ascend910b \
        --build-method bisheng \
        --output solutions/level1/VectorAdd/solution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SUPPORTED_LANGUAGES = ("python", "triton", "tilelang", "ascendc", "catlass")
SUPPORTED_HARDWARE = ("ascend910b", "ascend910_93", "ascend950", "LOCAL")


def _ensure_project_root_on_path() -> None:
    """Add the project src directory to sys.path if npu_kernelbench isn't importable."""
    try:
        import npu_kernelbench  # noqa: F401
    except ImportError:
        src_dir = Path(__file__).resolve().parents[4] / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))


def _detect_language(source_paths: list[Path]) -> str | None:
    """Optional fallback language detection from source extensions."""
    exts = {p.suffix for p in source_paths}
    if ".py" in exts:
        return None  # ambiguous; could be python/triton/tilelang
    if any(e in {".cpp", ".cc", ".cxx", ".asc", ".cu", ".c"} for e in exts):
        return "ascendc"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack kernel source code into a solution.json file."
    )
    parser.add_argument("--name", required=True, help="Solution name")
    parser.add_argument(
        "--definition", required=True,
        help="Name of the target Definition (must match definition.json)",
    )
    parser.add_argument(
        "--source", required=True, action="append",
        help="Path to a source file (repeatable for multi-file solutions)",
    )
    parser.add_argument(
        "--language", choices=SUPPORTED_LANGUAGES,
        help="Implementation language (auto-detected if omitted and sources are C++)",
    )
    parser.add_argument(
        "--hardware", required=True, choices=SUPPORTED_HARDWARE,
        help="Target Ascend hardware",
    )
    parser.add_argument(
        "--entry-point", default="main.py::run",
        help='Entry point in "file_path::function_name" form',
    )
    parser.add_argument(
        "--destination-passing-style", action="store_true",
        help="Enable destination-passing style (default: false)",
    )
    parser.add_argument(
        "--build-method", choices=("bisheng", "cmake"),
        help="AscendC build method (default: bisheng)",
    )
    parser.add_argument(
        "--author", default="developer", help="Solution author"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output solution.json path"
    )
    args = parser.parse_args()

    if "::" not in args.entry_point or args.entry_point.count("::") != 1:
        print(
            f"Invalid entry point '{args.entry_point}'. Expected 'file_path::function_name'.",
            file=sys.stderr,
        )
        return 1

    entry_file = args.entry_point.split("::")[0]
    if entry_file != "main.py":
        print(
            "Warning: npu-kernelbench requires the entry file to be 'main.py'. "
            f"You provided '{entry_file}'.",
            file=sys.stderr,
        )

    source_paths = []
    for src_path_str in args.source:
        src_path = Path(src_path_str)
        if not src_path.exists():
            print(f"Source file not found: {src_path}", file=sys.stderr)
            return 1
        source_paths.append(src_path)

    language = args.language
    if language is None:
        language = _detect_language(source_paths)
        if language is None:
            print(
                "Could not auto-detect language; please pass --language.",
                file=sys.stderr,
            )
            return 1
        print(f"Auto-detected language: {language}")

    entry_found = False
    sources = []
    for src_path in source_paths:
        if src_path.name == entry_file:
            entry_found = True
        sources.append({
            "path": src_path.name,
            "content": src_path.read_text(encoding="utf-8"),
        })

    if not entry_found:
        print(
            f"Entry point file '{entry_file}' not found in sources: "
            f"{[p.name for p in source_paths]}",
            file=sys.stderr,
        )
        return 1

    spec: dict = {
        "languages": [language],
        "target_hardware": [args.hardware],
        "entry_point": args.entry_point,
        "destination_passing_style": args.destination_passing_style,
    }
    if args.build_method:
        spec["build_method"] = args.build_method

    solution_dict = {
        "name": args.name,
        "definition": args.definition,
        "author": args.author,
        "spec": spec,
        "sources": sources,
    }

    # Validate against the data model
    _ensure_project_root_on_path()
    try:
        from npu_kernelbench.data import Solution

        Solution(**solution_dict)
    except ImportError as e:
        print(
            f"Cannot import npu_kernelbench.data: {e}. Run from project root with 'uv run'.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Packed solution failed validation: {e}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(solution_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Solution written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
