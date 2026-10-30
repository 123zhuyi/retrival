from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any, Iterator

import yaml


@dataclass(frozen=True)
class MarkdownDocument:
    path: Path
    frontmatter: dict[str, Any]
    body: str


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as exc:
        raise ValueError("unterminated YAML frontmatter") from exc
    raw = "\n".join(lines[1:end])
    frontmatter = yaml.safe_load(raw) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("frontmatter must be a mapping")
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return frontmatter, body


def read_markdown(path: Path) -> MarkdownDocument:
    frontmatter, body = split_frontmatter(path.read_text(encoding="utf-8"))
    return MarkdownDocument(path=path, frontmatter=frontmatter, body=body)


def iter_markdown(root: Path, subdirs: tuple[str, ...]) -> Iterator[MarkdownDocument]:
    for subdir in subdirs:
        base = root / subdir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            yield read_markdown(path)


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(data: Any, *, flow_style: bool = False) -> str:
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=flow_style,
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

