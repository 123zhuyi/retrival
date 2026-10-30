from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Sequence

from .indexer import build_indices, validate_corpus
from .io import atomic_write_text, dump_yaml, load_yaml, read_markdown
from .kernelwiki import KernelWikiSearch


CONTROLLED_FIELDS = (
    "architectures",
    "hardware_features",
    "techniques",
    "kernel_types",
    "symptoms",
)

TAG_AUTHORITIES = {
    "architectures": "cannBot npu-hardware-params.md chip-model table",
    "hardware_features": "cannBot ops/npu-arch references",
    "kernel_types": "CANN ops-nn/ops-math/ops-transformer classification",
    "symptoms": "reviewed verifier-error taxonomy",
    "techniques": "reviewed KernelWiki technique taxonomy",
}


class SourceBuildError(ValueError):
    """Raised when a source build violates a deterministic publication gate."""


def _json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, _json_text(data))


def _write_yaml(path: Path, data: Any) -> None:
    atomic_write_text(path, dump_yaml(data))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _slug(value: str, *, field: str) -> str:
    slug = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,78}[a-z0-9]", slug):
        raise SourceBuildError(f"{field} must use lowercase letters, digits, and hyphens: {value}")
    return slug


def _relative_target(value: str, *, prefix: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != prefix:
        raise SourceBuildError(f"target_path must stay below {prefix}/: {value}")
    if path.suffix.lower() != ".md":
        raise SourceBuildError(f"target_path must be a Markdown page: {value}")
    return path.as_posix()


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise SourceBuildError("tag fields must be strings or lists")
    return [str(item) for item in value]


def _render_page(frontmatter: dict[str, Any], body: str) -> str:
    return f"---\n{dump_yaml(frontmatter).rstrip()}\n---\n\n{body.strip()}\n"


class SourceBuilder:
    """Build reviewable KernelWiki candidates from an Agent-authored source packet."""

    def __init__(self, wiki_root: Path, workdir: Path):
        self.wiki_root = wiki_root.resolve()
        self.workdir = workdir.resolve()
        if not self.wiki_root.is_dir():
            raise SourceBuildError(f"KernelWiki root does not exist: {self.wiki_root}")

    def build_dir(self, build_id: str) -> Path:
        return self.workdir / _slug(build_id, field="build_id")

    @staticmethod
    def _validate_packet(packet: Any) -> dict[str, Any]:
        if not isinstance(packet, dict):
            raise SourceBuildError("source packet must be a mapping")
        if packet.get("schema_version") != 1:
            raise SourceBuildError("source packet schema_version must be 1")
        _slug(str(packet.get("build_id", "")), field="build_id")
        for section in ("source", "candidate"):
            value = packet.get(section)
            if not isinstance(value, dict):
                raise SourceBuildError(f"source packet requires a {section} mapping")
            if not isinstance(value.get("frontmatter"), dict):
                raise SourceBuildError(f"{section}.frontmatter must be a mapping")
            if not str(value.get("body", "")).strip():
                raise SourceBuildError(f"{section}.body must not be empty")
        evidence = packet.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise SourceBuildError("source packet requires at least one evidence record")
        return packet

    def collect(self, packet_path: Path) -> dict[str, Any]:
        packet_path = packet_path.resolve()
        packet = self._validate_packet(load_yaml(packet_path))
        build_id = str(packet["build_id"])
        build_dir = self.build_dir(build_id)
        evidence_dir = build_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        evidence_records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in packet["evidence"]:
            if not isinstance(item, dict):
                raise SourceBuildError("each evidence record must be a mapping")
            evidence_id = _slug(str(item.get("id", "")), field="evidence.id")
            if evidence_id in seen_ids:
                raise SourceBuildError(f"duplicate evidence id: {evidence_id}")
            seen_ids.add(evidence_id)
            source_path = item.get("path")
            excerpt = item.get("excerpt")
            if bool(source_path) == bool(excerpt):
                raise SourceBuildError(
                    f"evidence {evidence_id} must provide exactly one of path or excerpt"
                )
            if source_path:
                original = (packet_path.parent / str(source_path)).resolve()
                if not original.is_file():
                    raise SourceBuildError(f"evidence file does not exist: {original}")
                suffix = original.suffix or ".bin"
                snapshot = evidence_dir / f"{evidence_id}{suffix}"
                data = original.read_bytes()
                snapshot.write_bytes(data)
                locator = str(source_path).replace("\\", "/")
            else:
                snapshot = evidence_dir / f"{evidence_id}.txt"
                data = str(excerpt).encode("utf-8")
                atomic_write_text(snapshot, str(excerpt).rstrip() + "\n")
                data = snapshot.read_bytes()
                locator = str(item.get("locator", f"inline:{evidence_id}"))
            evidence_records.append(
                {
                    "id": evidence_id,
                    "kind": str(item.get("kind", "source-excerpt")),
                    "locator": locator,
                    "snapshot_path": snapshot.relative_to(build_dir).as_posix(),
                    "sha256": _sha256_bytes(data),
                    "bytes": len(data),
                    "summary": str(item.get("summary", "")),
                }
            )

        packet_bytes = packet_path.read_bytes()
        normalized = dict(packet)
        normalized["packet_sha256"] = _sha256_bytes(packet_bytes)
        normalized["packet_origin"] = packet_path.as_posix()
        _write_yaml(build_dir / "source-packet.yaml", normalized)
        _write_yaml(
            build_dir / "evidence-manifest.yaml",
            {"schema_version": 1, "build_id": build_id, "evidence": evidence_records},
        )
        report = {
            "stage": "collect",
            "build_id": build_id,
            "build_dir": build_dir.as_posix(),
            "evidence_count": len(evidence_records),
            "packet_sha256": normalized["packet_sha256"],
            "next": "build",
        }
        _write_json(build_dir / "collect-report.json", report)
        return report

    def _load_collected(self, build_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        build_dir = self.build_dir(build_id)
        packet_path = build_dir / "source-packet.yaml"
        manifest_path = build_dir / "evidence-manifest.yaml"
        if not packet_path.is_file() or not manifest_path.is_file():
            raise SourceBuildError(f"build {build_id} has not been collected")
        return build_dir, self._validate_packet(load_yaml(packet_path)), load_yaml(manifest_path)

    def _unknown_tags(self, packet: dict[str, Any]) -> list[dict[str, Any]]:
        vocabulary = load_yaml(self.wiki_root / "data" / "tags.yaml") or {}
        governance_path = self.wiki_root / "data" / "tag-governance.yaml"
        governance = load_yaml(governance_path) or {} if governance_path.is_file() else {}
        governed_fields = governance.get("controlled_fields") or {}
        evidence_ids = [str(item.get("id")) for item in packet.get("evidence") or []]
        proposals: list[dict[str, Any]] = []
        for section_name in ("source", "candidate"):
            frontmatter = packet[section_name]["frontmatter"]
            for field in CONTROLLED_FIELDS:
                allowed = set(_list(vocabulary.get(field)))
                for value in _list(frontmatter.get(field)):
                    if value not in allowed and not any(
                        item["field"] == field and item["value"] == value for item in proposals
                    ):
                        proposals.append(
                            {
                                "field": field,
                                "value": value,
                                "status": "needs-taxonomy-review",
                                "authority": str(
                                    (governed_fields.get(field) or {}).get(
                                        "authority", TAG_AUTHORITIES[field]
                                    )
                                ),
                                "evidence_ids": evidence_ids,
                                "requested_by": f"{section_name}.frontmatter",
                            }
                        )
        return proposals

    def build(self, build_id: str) -> dict[str, Any]:
        build_dir, packet, _ = self._load_collected(build_id)
        source_target = _relative_target(str(packet["source"].get("target_path", "")), prefix="sources")
        wiki_target = _relative_target(str(packet["candidate"].get("target_path", "")), prefix="wiki")
        candidate_dir = build_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            candidate_dir / "source.md",
            _render_page(packet["source"]["frontmatter"], str(packet["source"]["body"])),
        )
        atomic_write_text(
            candidate_dir / "wiki.md",
            _render_page(packet["candidate"]["frontmatter"], str(packet["candidate"]["body"])),
        )
        proposals = self._unknown_tags(packet)
        _write_yaml(
            build_dir / "tag-proposals.yaml",
            {
                "schema_version": 1,
                "build_id": build_id,
                "policy": "propose-only; never edit data/tags.yaml automatically",
                "proposals": proposals,
            },
        )
        report = {
            "stage": "build",
            "build_id": build_id,
            "candidate_files": ["candidates/source.md", "candidates/wiki.md"],
            "publication_targets": [source_target, wiki_target],
            "tag_proposal_count": len(proposals),
            "next": "verify",
        }
        _write_json(build_dir / "build-report.json", report)
        return report

    def verify(self, build_id: str) -> dict[str, Any]:
        build_dir, packet, manifest = self._load_collected(build_id)
        errors: list[str] = []
        source_target = _relative_target(str(packet["source"].get("target_path", "")), prefix="sources")
        wiki_target = _relative_target(str(packet["candidate"].get("target_path", "")), prefix="wiki")

        evidence_ok = True
        for record in manifest.get("evidence") or []:
            snapshot = build_dir / str(record.get("snapshot_path", ""))
            if not snapshot.is_file():
                errors.append(f"missing evidence snapshot: {record.get('snapshot_path')}")
                evidence_ok = False
            elif _sha256_bytes(snapshot.read_bytes()) != record.get("sha256"):
                errors.append(f"evidence sha256 mismatch: {record.get('id')}")
                evidence_ok = False

        source_path = build_dir / "candidates" / "source.md"
        wiki_path = build_dir / "candidates" / "wiki.md"
        if not source_path.is_file() or not wiki_path.is_file():
            raise SourceBuildError(f"build {build_id} has not produced candidates")
        source_doc = read_markdown(source_path)
        wiki_doc = read_markdown(wiki_path)
        source_id = str(source_doc.frontmatter.get("id", ""))
        linked_sources = _list(wiki_doc.frontmatter.get("sources"))
        source_linked = bool(source_id) and source_id in linked_sources
        if not source_linked:
            errors.append("wiki candidate must cite the generated source page id")

        if "languages" in source_doc.frontmatter or "languages" in wiki_doc.frontmatter:
            errors.append("legacy languages field is not allowed")

        schemas = load_yaml(self.wiki_root / "data" / "schemas.yaml") or {}
        search = KernelWikiSearch(self.wiki_root)
        typed_candidates = (
            (source_target, source_doc.frontmatter),
            (wiki_target, wiki_doc.frontmatter),
        )
        schema_ok = True
        for target, frontmatter in typed_candidates:
            page_type = search.detect_page_type(frontmatter, target)
            schema = schemas.get(page_type) or {}
            for field in schema.get("required") or []:
                if field not in frontmatter or frontmatter[field] in (None, ""):
                    errors.append(f"{target}: missing required field {field}")
                    schema_ok = False

        proposals = self._unknown_tags(packet)
        tags_ok = not proposals
        if proposals:
            errors.extend(
                f"unknown controlled tag {item['field']}={item['value']}" for item in proposals
            )
        gates = {
            "evidence_integrity": evidence_ok,
            "target_paths_safe": True,
            "source_linked": source_linked,
            "schema_fields_present": schema_ok,
            "controlled_tags_valid": tags_ok,
            "no_legacy_language_tags": not any("legacy languages" in error for error in errors),
            "human_approval": False,
        }
        decision = "ready-for-review" if not errors else "blocked"
        report = {
            "stage": "verify",
            "build_id": build_id,
            "decision": decision,
            "gates": gates,
            "errors": errors,
            "tag_proposals": proposals,
            "publication_targets": [source_target, wiki_target],
            "next": "publish" if decision == "ready-for-review" else "resolve-blockers",
        }
        _write_yaml(build_dir / "verification.yaml", report)
        return report

    def publish(
        self,
        build_id: str,
        *,
        approved_by: str,
        replace: bool = False,
    ) -> dict[str, Any]:
        if not approved_by.strip():
            raise SourceBuildError("publish requires approved_by")
        build_dir, packet, _ = self._load_collected(build_id)
        verification_path = build_dir / "verification.yaml"
        if not verification_path.is_file():
            raise SourceBuildError(f"build {build_id} has not been verified")
        verification = load_yaml(verification_path) or {}
        if verification.get("decision") != "ready-for-review":
            raise SourceBuildError("only a ready-for-review build can be published")

        publications = (
            (
                build_dir / "candidates" / "source.md",
                _relative_target(str(packet["source"]["target_path"]), prefix="sources"),
            ),
            (
                build_dir / "candidates" / "wiki.md",
                _relative_target(str(packet["candidate"]["target_path"]), prefix="wiki"),
            ),
        )
        previous: dict[Path, bytes | None] = {}
        for _, relative in publications:
            destination = self.wiki_root / relative
            if destination.exists() and not replace:
                raise SourceBuildError(f"publication target exists; pass --replace: {relative}")
            previous[destination] = destination.read_bytes() if destination.is_file() else None

        try:
            for candidate, relative in publications:
                destination = self.wiki_root / relative
                atomic_write_text(destination, candidate.read_text(encoding="utf-8"))
            errors = validate_corpus(self.wiki_root)
            if errors:
                raise SourceBuildError("published corpus failed validation: " + "; ".join(errors))
            generated = [path.relative_to(self.wiki_root).as_posix() for path in build_indices(self.wiki_root)]
        except Exception:
            for destination, content in previous.items():
                if content is None:
                    destination.unlink(missing_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
            raise

        report = {
            "stage": "publish",
            "build_id": build_id,
            "decision": "published",
            "approved_by": approved_by.strip(),
            "published_pages": [relative for _, relative in publications],
            "generated_indices": generated,
            "gates": {**verification.get("gates", {}), "human_approval": True},
        }
        _write_yaml(build_dir / "publication.yaml", report)
        return report

    def run(
        self,
        packet_path: Path,
        *,
        publish: bool = False,
        approved_by: str = "",
        replace: bool = False,
    ) -> dict[str, Any]:
        collected = self.collect(packet_path)
        build_id = str(collected["build_id"])
        built = self.build(build_id)
        verified = self.verify(build_id)
        result: dict[str, Any] = {
            "build_id": build_id,
            "collect": collected,
            "build": built,
            "verify": verified,
        }
        if publish:
            result["publish"] = self.publish(
                build_id, approved_by=approved_by, replace=replace
            )
        return result


def _root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"KernelWiki root does not exist: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triton-kb-source-builder")
    parser.add_argument("--root", type=_root, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--packet", type=Path, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--build-id", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--build-id", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--build-id", required=True)
    publish.add_argument("--approved-by", required=True)
    publish.add_argument("--replace", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--publish", action="store_true")
    run.add_argument("--approved-by", default="")
    run.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    builder = SourceBuilder(args.root, args.workdir)
    try:
        if args.command == "collect":
            result = builder.collect(args.packet)
        elif args.command == "build":
            result = builder.build(args.build_id)
        elif args.command == "verify":
            result = builder.verify(args.build_id)
        elif args.command == "publish":
            result = builder.publish(
                args.build_id, approved_by=args.approved_by, replace=args.replace
            )
        elif args.command == "run":
            result = builder.run(
                args.packet,
                publish=args.publish,
                approved_by=args.approved_by,
                replace=args.replace,
            )
        else:
            raise SourceBuildError(f"unsupported command: {args.command}")
        print(_json_text(result), end="")
        return 0
    except (OSError, SourceBuildError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
