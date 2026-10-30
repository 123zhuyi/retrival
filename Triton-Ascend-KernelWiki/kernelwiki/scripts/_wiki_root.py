from __future__ import annotations

import os
from pathlib import Path


def resolve_wiki_root() -> Path:
    override = os.environ.get("TRITON_ASCEND_KERNELWIKI_ROOT")
    root = Path(override).expanduser().resolve() if override else Path(__file__).resolve().parent.parent
    if not (root / "data" / "tags.yaml").is_file() or not (root / "wiki").is_dir():
        raise RuntimeError(f"invalid Triton-Ascend KernelWiki root: {root}")
    return root


WIKI_ROOT = resolve_wiki_root()
