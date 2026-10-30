from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from _wiki_root import WIKI_ROOT


RULES_PATH = WIKI_ROOT / "data" / "doc-tagging-rules.yaml"


def load_rules() -> dict[str, Any]:
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("data/doc-tagging-rules.yaml must be a mapping")
    return data


def _matches(rule: dict[str, Any], *, upstream_path: str, title: str, body: str) -> bool:
    fields = {
        "path_regex": upstream_path,
        "title_regex": title,
        "body_regex": body,
    }
    for key, text in fields.items():
        for pattern in rule.get(key) or []:
            if re.search(str(pattern), text, re.IGNORECASE):
                return True
    return False


def _values_for(field: str, rules: dict[str, Any], *, upstream_path: str, title: str, body: str) -> list[str]:
    values = []
    for rule in rules.get(field) or []:
        if isinstance(rule, dict) and rule.get("value") and _matches(
            rule, upstream_path=upstream_path, title=title, body=body
        ):
            values.append(str(rule["value"]))
    return list(dict.fromkeys(values))


def operators_from_path(upstream_path: str) -> list[str]:
    path = Path(upstream_path)
    stem = path.stem
    if stem == "index":
        return []
    normalized = upstream_path.replace("\\", "/")
    if "/triton_api/" in normalized or "/triton_api_extension/" in normalized:
        return [stem]
    example_match = re.search(r"/examples/\d+_(.+?)_example\.md$", normalized, re.IGNORECASE)
    if example_match:
        return [example_match.group(1).replace("_", "-").lower()]
    guide_match = re.search(r"/programming_guide/(.+?)_operator\.md$", normalized, re.IGNORECASE)
    if guide_match:
        return [f"{guide_match.group(1).replace('_', '-')}-operator".lower()]
    return []


def classify_document(*, upstream_path: str, title: str, body: str, rules: dict[str, Any] | None = None) -> dict[str, list[str]]:
    rules = rules or load_rules()
    return {
        "architectures": _values_for("architectures", rules, upstream_path=upstream_path, title=title, body=body),
        "hardware_features": _values_for("hardware_features", rules, upstream_path=upstream_path, title=title, body=body),
        "symptoms": _values_for("symptoms", rules, upstream_path=upstream_path, title=title, body=body),
        "kernel_types": _values_for("kernel_types", rules, upstream_path=upstream_path, title=title, body=body),
        "operators": operators_from_path(upstream_path),
    }
