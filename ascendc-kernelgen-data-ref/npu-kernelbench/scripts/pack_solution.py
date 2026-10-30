#!/usr/bin/env python3
"""Pack a kernel source file into NPUKernelBench solution.json format.

Usage:
    python scripts/pack_solution.py \
        --name my_matmul \
        --definition matmul_triton \
        --source main.py \
        --language triton \
        --hardware ascend910b \
        --entry-point main.py::run \
        --output my_solutions/level2/MatmulTriton/solution.json

For multi-file solutions (e.g. AscendC host + kernel), pass --source multiple times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_LANGUAGES = ("python", "triton", "tilelang", "ascendc", "catlass")
SUPPORTED_HARDWARE = ("ascend910b", "ascend910_93", "ascend950", "LOCAL")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack kernel source code into a solution.json file."
    )
    parser.add_argument("--name", required=True, help="Solution name")
    parser.add_argument(
        "--definition", required=True,
        help="Name of the target Definition (must match definition.json)"
    )
    parser.add_argument(
        "--source", required=True, action="append",
        help="Path to a source file (repeatable for multi-file solutions)"
    )
    parser.add_argument(
        "--language", required=True, choices=SUPPORTED_LANGUAGES,
        help="Implementation language"
    )
    parser.add_argument(
        "--hardware", required=True, choices=SUPPORTED_HARDWARE,
        help="Target Ascend hardware"
    )
    parser.add_argument(
        "--entry-point", required=True,
        help='Entry point in "file_path::function_name" form, e.g. "main.py::run"'
    )
    parser.add_argument(
        "--destination-passing-style", action="store_true",
        help="Enable destination-passing style (default: false)"
    )
    parser.add_argument(
        "--build-method", choices=("bisheng", "cmake"),
        help="AscendC build method (default: bisheng)"
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
        raise ValueError(
            f"Invalid entry point '{args.entry_point}'. "
            'Expected "file_path::function_name".'
        )

    entry_file = args.entry_point.split("::")[0]
    entry_found = False
    sources = []
    for src_path_str in args.source:
        src_path = Path(src_path_str)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        if src_path.name == entry_file:
            entry_found = True
        sources.append({
            "path": src_path.name,
            "content": src_path.read_text(encoding="utf-8"),
        })

    if not entry_found:
        raise ValueError(
            f"Entry point file '{entry_file}' not found in --source list: "
            f"{[Path(s).name for s in args.source]}"
        )

    spec: dict = {
        "languages": [args.language],
        "target_hardware": [args.hardware],
        "entry_point": args.entry_point,
        "destination_passing_style": args.destination_passing_style,
    }
    if args.build_method:
        spec["build_method"] = args.build_method

    solution = {
        "name": args.name,
        "definition": args.definition,
        "author": args.author,
        "spec": spec,
        "sources": sources,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(solution, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Solution written to {args.output}")


if __name__ == "__main__":
    main()
