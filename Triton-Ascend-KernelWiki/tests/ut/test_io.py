from __future__ import annotations

from pathlib import Path

import pytest

from triton_ascend_kb.io import atomic_write_text, dump_yaml, iter_markdown, load_yaml, read_markdown, split_frontmatter


def test_split_frontmatter_and_plain_markdown() -> None:
    fm, body = split_frontmatter("---\nid: page\ntags: [a, b]\n---\n\nBody")
    assert fm == {"id": "page", "tags": ["a", "b"]}
    assert body == "Body"
    assert split_frontmatter("plain text") == ({}, "plain text")
    with pytest.raises(ValueError, match="unterminated"):
        split_frontmatter("---\nid: broken")
    with pytest.raises(ValueError, match="mapping"):
        split_frontmatter("---\n- item\n---\nbody")


def test_atomic_markdown_yaml_and_iteration(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki" / "kernels"
    path = wiki / "page.md"
    atomic_write_text(path, "---\nid: page\n---\nbody\n")
    assert read_markdown(path).frontmatter["id"] == "page"
    assert [doc.path.name for doc in iter_markdown(tmp_path, ("wiki",))] == ["page.md"]
    assert list(iter_markdown(tmp_path, ("missing",))) == []

    yaml_path = tmp_path / "data.yaml"
    atomic_write_text(yaml_path, dump_yaml({"中文": [1, 2]}))
    assert load_yaml(yaml_path) == {"中文": [1, 2]}

