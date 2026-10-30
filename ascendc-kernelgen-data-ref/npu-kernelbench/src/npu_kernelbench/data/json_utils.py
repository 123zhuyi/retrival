"""JSON encoding/decoding utilities for Pydantic BaseModel objects."""

from pathlib import Path
from typing import Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_json_file(object: BaseModel, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(object.model_dump_json(indent=2, exclude_unset=True))


def load_json_file(model_cls: Type[T], path: Union[str, Path]) -> T:
    with open(Path(path), "r", encoding="utf-8") as f:
        return model_cls.model_validate_json(f.read())


def save_jsonl_file(objects: list[BaseModel], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        object_strs = [obj.model_dump_json(indent=None) for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)


def load_jsonl_file(model_cls: Type[T], path: Union[str, Path]) -> list[T]:
    out = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(model_cls.model_validate_json(line))
    return out


def append_jsonl_line(obj: BaseModel, path: Union[str, Path]) -> None:
    """Append a single JSON object as one line to a JSONL file.

    Creates the parent directory and file if they don't exist.
    Safe for incremental writes from long-running evaluation loops.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = obj.model_dump_json(indent=None) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def append_jsonl_file(objects: list[BaseModel], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_newline_prefix = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last_char = f.read(1)
            needs_newline_prefix = last_char != b"\n"

    with open(path, "a", encoding="utf-8") as f:
        if needs_newline_prefix:
            f.write("\n")
        object_strs = [obj.model_dump_json(indent=None) for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)
