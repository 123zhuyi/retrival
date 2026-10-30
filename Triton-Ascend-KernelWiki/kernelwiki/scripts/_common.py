from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

import yaml

from _wiki_root import WIKI_ROOT


TAG_FIELDS = ("tags", "techniques", "hardware_features", "kernel_types", "aliases", "symptoms")
CODE_EXTENSIONS = {".cu", ".cuh", ".ptx", ".py", ".pyx", ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".hxx", ".inl", ".txt", ".sh", ".yaml", ".yml", ".json", ".patch"}


def split_page(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    data = yaml.safe_load(match.group(1)) or {}
    return data if isinstance(data, dict) else {}, match.group(2)


def page_type(fm: dict[str, Any], relative: str) -> str:
    if fm.get("type"):
        return f"wiki-{fm['type']}"
    parts = relative.split("/")
    return f"source-{parts[1].rstrip('s')}" if len(parts) > 1 and parts[0] == "sources" else "unknown"


def pages() -> list[dict[str, Any]]:
    output = []
    for subdir in ("sources", "wiki"):
        for path in sorted((WIKI_ROOT / subdir).rglob("*.md")):
            fm, body = split_page(path)
            if fm:
                relative = path.relative_to(WIKI_ROOT).as_posix()
                output.append({"path": relative, "fm": fm, "body": body, "type": page_type(fm, relative)})
    return output


def aliases() -> dict[str, str]:
    raw = yaml.safe_load((WIKI_ROOT / "data" / "aliases.yaml").read_text(encoding="utf-8")) or {}
    output: dict[str, str] = {}
    for canonical, variants in raw.items():
        if isinstance(canonical, str):
            output[canonical.lower()] = canonical
            for variant in variants or []:
                if isinstance(variant, str):
                    output.setdefault(variant.lower(), canonical)
    return output


def architecture_aliases() -> dict[str, list[str]]:
    path = WIKI_ROOT / "data" / "architecture-aliases.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.is_file() else {}
    output: dict[str, list[str]] = {}
    for selector, models in raw.items():
        if not isinstance(selector, str):
            continue
        values = [models] if isinstance(models, str) else models or []
        key = re.sub(r"[\s_-]+", "", selector.strip().lower())
        output[key] = [str(model) for model in values if isinstance(model, str)]
    return output


def expand_architecture(term: str, architecture_map: dict[str, list[str]]) -> list[str]:
    key = re.sub(r"[\s_-]+", "", term.strip().lower())
    return architecture_map.get(key) or [term.strip()]


def expand(term: str, alias_map: dict[str, str], architecture_map: dict[str, list[str]] | None = None) -> list[str]:
    architecture_values = expand_architecture(term, architecture_map or {})
    if len(architecture_values) > 1 or architecture_values[0].lower() != term.lower():
        return list(dict.fromkeys([term, *architecture_values]))
    canonical = alias_map.get(term.lower())
    return [term, canonical] if canonical and canonical.lower() != term.lower() else [term]


def score(
    page: dict[str, Any],
    keywords: Iterable[str],
    alias_map: dict[str, str],
    architecture_map: dict[str, list[str]],
) -> int:
    fm, body = page["fm"], page["body"].lower()
    title = str(fm.get("title", "")).lower()
    tag_text = " ".join(str(item) for key in TAG_FIELDS for item in (([fm[key]] if isinstance(fm.get(key), str) else fm.get(key)) or [])).lower()
    total = 0
    for keyword in keywords:
        best = 0
        for variant in expand(keyword, alias_map, architecture_map):
            value = variant.lower()
            best = max(best, (10 if value in title else 0) + (5 if value in tag_text else 0) + min(body.count(value), 3))
        total += best
    return total


def _has_code(page: dict[str, Any]) -> bool:
    artifact_dir = page["fm"].get("artifact_dir")
    if not artifact_dir:
        return False
    base = WIKI_ROOT / str(artifact_dir)
    return base.is_dir() and any(path.is_file() and path.suffix.lower() in CODE_EXTENSIONS for path in base.rglob("*"))


def query_pages(keywords: list[str], args: Any) -> list[dict[str, Any]]:
    alias_map = aliases()
    architecture_map = architecture_aliases()
    result = []
    for page in pages():
        fm = page["fm"]
        if args.type and not page["type"].endswith(args.type) and page["type"] != args.type:
            continue
        if args.tag:
            all_tags = {str(item).lower() for key in TAG_FIELDS[:5] for item in (([fm[key]] if isinstance(fm.get(key), str) else fm.get(key)) or [])}
            if not all_tags.intersection(item.lower() for item in expand(args.tag, alias_map)):
                continue
        if args.repo and args.repo.lower() not in str(fm.get("repo", "")).lower():
            continue
        if args.architecture and not {item.lower() for item in expand_architecture(args.architecture, architecture_map)}.intersection(str(item).lower() for item in fm.get("architectures") or []):
            continue
        if args.symptom and args.symptom not in set(fm.get("symptoms") or []):
            continue
        if args.confidence and args.confidence != fm.get("confidence"):
            continue
        if args.has_code and not _has_code(page):
            continue
        page["score"] = score(page, keywords, alias_map, architecture_map) if keywords else 0
        if not keywords or page["score"]:
            result.append(page)
    result.sort(key=lambda page: (-page["score"], page["path"]) if keywords else (0, page["path"]))
    return result[: args.limit]


def find_page(lookup: str) -> tuple[Path, dict[str, Any], str]:
    direct = WIKI_ROOT / lookup
    if direct.is_file():
        fm, body = split_page(direct)
        return direct, fm, body
    for page in pages():
        if page["fm"].get("id") == lookup:
            path = WIKI_ROOT / page["path"]
            return path, page["fm"], page["body"]
    raise KeyError(f"page not found: {lookup}")


def source_ids(fm: dict[str, Any]) -> list[str]:
    value = fm.get("sources") or []
    return [str(value)] if isinstance(value, str) else [str(item) for item in value]


def artifact_files(fm: dict[str, Any]) -> list[Path]:
    value = fm.get("artifact_dir")
    if not value:
        return []
    base = (WIKI_ROOT / str(value)).resolve()
    if not base.is_relative_to(WIKI_ROOT.resolve()):
        raise ValueError("artifact_dir escapes wiki root")
    return [path for path in sorted(base.rglob("*")) if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS]


def validate() -> list[str]:
    errors: list[str] = []
    all_pages = pages()
    ids: dict[str, str] = {}
    source_page_ids = {str(page["fm"].get("id")) for page in all_pages if page["path"].startswith("sources/")}
    vocab = yaml.safe_load((WIKI_ROOT / "data" / "tags.yaml").read_text(encoding="utf-8")) or {}
    schemas = yaml.safe_load((WIKI_ROOT / "data" / "schemas.yaml").read_text(encoding="utf-8")) or {}
    architecture_map = architecture_aliases()
    allowed_models = set(vocab.get("architectures") or [])
    for selector, models in architecture_map.items():
        for model in models:
            if model not in allowed_models:
                errors.append(f"data/architecture-aliases.yaml: selector {selector} has unknown chip model {model}")
    for page in all_pages:
        fm, relative = page["fm"], page["path"]
        pid = str(fm.get("id", ""))
        if not pid:
            errors.append(f"{relative}: missing id")
        elif pid in ids:
            errors.append(f"duplicate page id {pid}: {ids[pid]} and {relative}")
        ids[pid] = relative
        if "source_pages" in fm:
            errors.append(f"{relative}: use canonical 'sources'")
        if "languages" in fm:
            errors.append(f"{relative}: legacy languages field is not allowed")
        schema = schemas.get(page["type"], {})
        for field in schema.get("required", []):
            if field not in fm or fm[field] in (None, ""):
                errors.append(f"{relative}: missing required field {field}")
        for source in source_ids(fm):
            if source not in source_page_ids:
                errors.append(f"{relative}: unresolved source page {source}")
        for field in ("architectures", "hardware_features", "techniques", "kernel_types", "symptoms"):
            allowed = set(vocab.get(field) or [])
            for value in fm.get(field) or []:
                if allowed and value not in allowed:
                    errors.append(f"{relative}: unknown {field} value {value}")
    for page in all_pages:
        artifact_dir = page["fm"].get("artifact_dir")
        if artifact_dir and not (WIKI_ROOT / str(artifact_dir) / "PROVENANCE.yaml").is_file():
            errors.append(f"{artifact_dir}: PROVENANCE.yaml missing")
    for provenance in (WIKI_ROOT / "artifacts").rglob("PROVENANCE.yaml"):
        data = yaml.safe_load(provenance.read_text(encoding="utf-8")) or {}
        for record in data.get("files") or []:
            file = provenance.parent / str(record.get("local_path", ""))
            if not file.is_file():
                errors.append(f"{provenance.relative_to(WIKI_ROOT)}: missing {record.get('local_path')}")
            elif hashlib.sha256(file.read_bytes()).hexdigest() != record.get("sha256"):
                errors.append(f"{provenance.relative_to(WIKI_ROOT)}: sha256 mismatch for {record.get('local_path')}")
    snapshot_path = WIKI_ROOT / "data" / "triton-ascend-doc-snapshot.yaml"
    if snapshot_path.is_file():
        snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        records = snapshot.get("pages") or []
        if snapshot.get("page_count") != len(records):
            errors.append("data/triton-ascend-doc-snapshot.yaml: page_count mismatch")
        page_map = {page["path"]: page for page in all_pages}
        manifest_paths = set()
        for record in records:
            if not isinstance(record, dict):
                errors.append("data/triton-ascend-doc-snapshot.yaml: pages entry must be a mapping")
                continue
            local_path = str(record.get("local_path", ""))
            manifest_paths.add(local_path)
            page = page_map.get(local_path)
            if page is None:
                errors.append(f"data/triton-ascend-doc-snapshot.yaml: missing page {local_path}")
                continue
            for field in ("id", "upstream_path", "content_sha256", "snapshot_body_sha256"):
                if str(page["fm"].get(field, "")) != str(record.get(field, "")):
                    errors.append(f"{local_path}: manifest {field} mismatch")
            if str(page["fm"].get("upstream_commit", "")) != str(snapshot.get("upstream_commit", "")):
                errors.append(f"{local_path}: manifest upstream_commit mismatch")
            raw_page = (WIKI_ROOT / local_path).read_bytes()
            snapshot_body = raw_page.partition(b"\n---\n\n")[2]
            if hashlib.sha256(snapshot_body).hexdigest() != record.get("snapshot_body_sha256"):
                errors.append(f"{local_path}: snapshot body sha256 mismatch")
        generated_paths = {
            page["path"]
            for page in all_pages
            if page["fm"].get("generated_by") == "scripts/ingest_triton_ascend_docs.py"
        }
        for local_path in sorted(generated_paths - manifest_paths):
            errors.append(f"{local_path}: generated doc missing from snapshot manifest")
    taxonomy_path = WIKI_ROOT / "data" / "operator-taxonomy-sources.yaml"
    if taxonomy_path.is_file():
        taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}
        allowed = set(vocab.get("kernel_types") or [])
        for source in taxonomy.get("sources") or []:
            for category in source.get("categories") or []:
                if category not in allowed:
                    errors.append(f"data/operator-taxonomy-sources.yaml: unknown kernel type {category}")
    rules_path = WIKI_ROOT / "data" / "doc-tagging-rules.yaml"
    if rules_path.is_file():
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        for field in ("symptoms", "kernel_types"):
            allowed = set(vocab.get(field) or [])
            for rule in rules.get(field) or []:
                value = str(rule.get("value", "")) if isinstance(rule, dict) else ""
                if value not in allowed:
                    errors.append(f"data/doc-tagging-rules.yaml: unknown {field} value {value}")
    governance_path = WIKI_ROOT / "data" / "tag-governance.yaml"
    if governance_path.is_file():
        governance = yaml.safe_load(governance_path.read_text(encoding="utf-8")) or {}
        if governance.get("update_mode") != "proposal-only":
            errors.append("data/tag-governance.yaml: update_mode must be proposal-only")
        controlled = {"architectures", "hardware_features", "techniques", "kernel_types", "symptoms"}
        for field in (governance.get("controlled_fields") or {}):
            if field not in controlled:
                errors.append(f"data/tag-governance.yaml: unknown controlled field {field}")
    return errors
