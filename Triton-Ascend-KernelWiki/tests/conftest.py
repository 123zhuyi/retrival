from __future__ import annotations

from pathlib import Path
import shutil

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def wiki_root() -> Path:
    return PROJECT_ROOT / "kernelwiki"


@pytest.fixture
def isolated_wiki(tmp_path: Path, wiki_root: Path) -> Path:
    target = tmp_path / "kernelwiki"
    shutil.copytree(wiki_root, target)
    return target

