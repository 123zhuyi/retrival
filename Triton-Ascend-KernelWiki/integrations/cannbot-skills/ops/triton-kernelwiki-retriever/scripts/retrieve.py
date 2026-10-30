#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys


def _load_main():
    try:
        from triton_ascend_kb.cannbot_retriever import main

        return main
    except ModuleNotFoundError as exc:
        if exc.name != "triton_ascend_kb":
            raise
        root_value = os.environ.get("TRITON_ASCEND_KB_ROOT")
        if root_value:
            project_src = Path(root_value).expanduser().resolve().parent / "src"
            if project_src.is_dir():
                sys.path.insert(0, str(project_src))
                from triton_ascend_kb.cannbot_retriever import main

                return main
        raise SystemExit(
            "triton_ascend_kb is not importable; install the knowledge-base package "
            "or set TRITON_ASCEND_KB_ROOT to <project>/kernelwiki"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(_load_main()())
