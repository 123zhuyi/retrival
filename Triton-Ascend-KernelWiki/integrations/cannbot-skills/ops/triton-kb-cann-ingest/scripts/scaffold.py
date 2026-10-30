#!/usr/bin/env python3
"""Scaffold a CANN -> Triton-Ascend ingestion bundle for the triton-kb-cann-ingest skill.

Two commands:

  init                Create artifacts/<bundle>/ with a starter PROVENANCE.yaml and a
                      reports/<operator>-ingest-report.md skeleton. Inferred fields
                      (upstream_repo, upstream_sha, retrieved_at) are filled from the
                      source tree when possible and flagged with "<fill: ...>" otherwise.

  update-provenance   Recompute sha256 for every file under artifacts/<bundle>/ and
                      rewrite PROVENANCE.yaml's files[] list. Preserves author-chosen
                      role/mode for known files; assigns sensible defaults to new ones.
                      Run this after creating or editing any bundle file.

Standalone: stdlib + PyYAML only. It does not require the triton_ascend_kb package.
Mirrors the conventions of kernelwiki/scripts (pathlib, yaml, argparse).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("PyYAML is required (pip install pyyaml). The KernelWiki scripts already depend on it.")


DEFAULT_LICENSE = "CANN Open Software License Agreement Version 2.0"

SKILL_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SKILL_DIR.parent  # .../triton-kb-cann-ingest
REPORT_TEMPLATE = SKILL_ROOT / "references" / "report-template.md"

# Default (role, mode) per file extension when a file is first seen. Authors can edit
# PROVENANCE.yaml afterwards; update-provenance preserves existing role/mode.
ROLE_BY_EXT = {
    ".py": ("derived-source", "derived"),
    ".md": ("approach-notes", "derived"),
    ".yaml": ("approach-notes", "derived"),
    ".yml": ("approach-notes", "derived"),
    ".json": ("approach-notes", "derived"),
    ".c": ("derived-source", "derived"),
    ".cpp": ("derived-source", "derived"),
    ".h": ("derived-source", "derived"),
}
DEFAULT_ROLE = ("approach-notes", "derived")

FALLBACK_REPORT = """# {operator} CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | {source} | source-confirmed |
| KERNELWIKI_ROOT | {root} | source-confirmed |
| OPERATOR_NAME | {operator} | source-confirmed |
| OPERATOR_TYPE | {op_type} | inferred |
| VARIANTS | {variants} | inferred |
| mode | static_ingest | - |

## 2. Source file inventory
Full table in artifacts/{bundle}/source_file_inventory.md.

## 3. Operator semantics
inputs/outputs, attributes, shape/dtype/layout constraints, core formula, boundaries, variants.

## 4. Stage / pipeline table
| Stage | Entry | Unit | Op |

## 5. Tiling parameters
CANN original tiling/core-partition table.

## 6. Core partition strategy

## 7. Kernel templates & dispatch

## 8. Pipeline analysis
| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |

## 9. CANN -> Triton-Ascend migration classification
Full table in artifacts/{bundle}/migration-table.md.

## 10. Triton-Ascend skeleton
Reference artifacts/{bundle}/{operator}-triton-skeleton.py.
reproducibility: pseudocode | snippet. Not compiled / not benchmarked.

## 11. KB pages written
| Path | type | id | sources | confidence | reproducibility |

## 12. Validation & retrieval
- validate.py:
- generate-indices.py:
- query.py <OPERATOR_NAME> --compact:
- query.py <OPERATOR_TYPE> tiling --compact:
- query.py <OPERATOR_TYPE> mask --compact:
- query.py <OPERATOR_TYPE> triton ascend --compact:

## 13. Unverified / needs-hardware items

## 14. Modified / created files
"""


def slugify(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-").strip("-")


def resolve_root(args) -> Path:
    root = args.root or os.environ.get("TRITON_ASCEND_KB_ROOT")
    if not root:
        sys.exit("--root is required (or set TRITON_ASCEND_KB_ROOT to the kernelwiki/ directory)")
    root_path = Path(root).expanduser().resolve()
    if not (root_path / "data" / "schemas.yaml").is_file():
        sys.exit(f"{root_path} does not look like a KernelWiki root (no data/schemas.yaml)")
    return root_path


def _git(source: Path, *git_args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(source), *git_args],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_remote(url: str | None) -> str | None:
    """Reduce a git remote URL to an org/name slug (e.g. cann/ops-transformer)."""
    if not url:
        return None
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    if "://" in url:  # https://host/org/name  ->  org/name
        url = url.split("://", 1)[1]
        parts = url.split("/", 2)
        if len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
        return parts[-1]
    if ":" in url:  # scp-like git@host:org/name  ->  org/name
        url = url.split(":", 1)[1]
    return url or None


def derive_repo_short(upstream_repo: str | None, source: Path) -> str:
    if upstream_repo:
        tail = upstream_repo.rsplit("/", 1)[-1]
        if tail.lower().startswith("cann-"):
            return tail.lower()[len("cann-"):]
        return tail.lower()
    return source.name.lower()


def warn_unknown_kernel_type(root: Path, op_type: str) -> None:
    tags_path = root / "data" / "tags.yaml"
    if not tags_path.is_file():
        return
    try:
        vocab = yaml.safe_load(tags_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return
    allowed = vocab.get("kernel_types") or []
    if op_type not in allowed:
        print(f"WARNING: OPERATOR_TYPE {op_type!r} is not in tags.yaml#kernel_types {allowed}", file=sys.stderr)


def cmd_init(args) -> int:
    root = resolve_root(args)
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        sys.exit(f"--source not found: {source}")
    operator = args.operator
    op_type = args.type
    bundle = args.bundle or slugify(operator)
    warn_unknown_kernel_type(root, op_type)

    bundle_dir = root / "artifacts" / bundle
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        print(f"note: bundle dir already exists and is non-empty: {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    commit = _git(source, "rev-parse", "HEAD")
    remote = _git(source, "remote", "get-url", "origin")
    upstream_repo = args.repo or parse_remote(remote)
    repo_short = derive_repo_short(upstream_repo, source)
    source_page_id = f"{repo_short}-{bundle}"

    if source.is_relative_to(root):
        origin_url = f"local://{source.relative_to(root).as_posix()}"
    else:
        origin_url = remote or f"local://{source.name}"

    provenance = {
        "origin_url": origin_url,
        "upstream_repo": upstream_repo or "<fill: org/name>",
        "upstream_sha": commit or "<fill: 40-char commit>",
        "upstream_path": f"<op dir relative to repo root, e.g. attention/{operator}/>",
        "license": args.license,
        "retrieved_at": date.today().isoformat(),
        "asset_mode": "derived",
        "derived_from": source_page_id,
        "fixture": True,
        "files": [],  # run `update-provenance` after adding derived files
    }
    prov_path = bundle_dir / "PROVENANCE.yaml"
    prov_path.write_text(
        yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Report skeleton at the repo root (siblings of kernelwiki/), matching existing reports/.
    reports_dir = root.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{slugify(operator)}-ingest-report.md"
    variants = args.variants or "<none>"
    if REPORT_TEMPLATE.is_file():
        body = REPORT_TEMPLATE.read_text(encoding="utf-8")
        for placeholder, value in {
            "<OPERATOR_NAME>": operator,
            "<OPERATOR_TYPE>": op_type,
            "<bundle>": bundle,
            "<operator>": slugify(operator),
        }.items():
            body = body.replace(placeholder, value)
        if args.variants:
            body = body.replace("| VARIANTS | ... | inferred |", f"| VARIANTS | {args.variants} | inferred |")
    else:
        body = FALLBACK_REPORT.format(
            operator=operator, source=source, root=root, op_type=op_type,
            variants=variants, bundle=bundle,
        )
    report_path.write_text(body, encoding="utf-8")

    source_page_path = root / "sources" / repo_short / f"{slugify(operator)}.md"
    wiki_page_path = root / "wiki" / "kernels" / f"{slugify(operator)}.md"

    print(f"created PROVENANCE:    {prov_path}")
    print(f"created report stub:   {report_path}")
    print()
    print("Next steps (see SKILL.md S8 for frontmatter):")
    print(f"  1. source page:  {source_page_path}   (id: {source_page_id})")
    print(f"  2. wiki page(s): {wiki_page_path}   (sources: [{source_page_id}], artifact_dir: artifacts/{bundle})")
    print(f"  3. add derived files under {bundle_dir}/, then:")
    print(f"       python {Path(__file__).as_posix()} update-provenance --root {root} --bundle {bundle}")
    print(f"  4. validate:     python {root / 'scripts' / 'validate.py'}")
    return 0


def cmd_update_provenance(args) -> int:
    root = resolve_root(args)
    bundle_dir = root / "artifacts" / args.bundle
    if not bundle_dir.is_dir():
        sys.exit(f"bundle not found: {bundle_dir}")
    prov_path = bundle_dir / "PROVENANCE.yaml"
    if not prov_path.is_file():
        sys.exit(f"PROVENANCE.yaml not found in {bundle_dir} (run `init` first)")

    data = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        sys.exit(f"{prov_path} is not a mapping")
    existing = {
        rec["local_path"]: rec
        for rec in (data.get("files") or [])
        if isinstance(rec, dict) and rec.get("local_path")
    }

    records = []
    for f in sorted(p for p in bundle_dir.rglob("*") if p.is_file() and p.name != "PROVENANCE.yaml"):
        local_path = f.relative_to(bundle_dir).as_posix()
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        role, mode = ROLE_BY_EXT.get(f.suffix.lower(), DEFAULT_ROLE)
        prev = existing.get(local_path, {})
        records.append({
            "local_path": local_path,
            "role": prev.get("role", role),
            "mode": prev.get("mode", mode),
            "sha256": sha,
        })

    data["files"] = records
    prov_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"updated {prov_path} ({len(records)} file(s)):")
    for rec in records:
        print(f"  {rec['local_path']:<40} {rec['role']:<16} {rec['sha256'][:12]}…")
    if not records:
        print("  (no files yet — add derived files then re-run)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scaffold.py",
        description="Scaffold a CANN->Triton-Ascend ingestion bundle (triton-kb-cann-ingest skill).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", help="KernelWiki root (kernelwiki/), or set TRITON_ASCEND_KB_ROOT")

    p_init = sub.add_parser("init", help="create artifacts/<bundle>/ + PROVENANCE.yaml + report stub")
    add_root(p_init)
    p_init.add_argument("--source", required=True, help="SOURCE_REPO_PATH (CANN source tree)")
    p_init.add_argument("--operator", required=True, help="OPERATOR_NAME, e.g. flash_attention_score")
    p_init.add_argument("--type", required=True, help="OPERATOR_TYPE (tags.yaml#kernel_types), e.g. attention")
    p_init.add_argument("--variants", help="comma-separated, e.g. forward,inference")
    p_init.add_argument("--bundle", help="bundle name (default: slug of --operator)")
    p_init.add_argument("--repo", help="upstream org/name (default: parsed from git remote)")
    p_init.add_argument("--license", default=DEFAULT_LICENSE, help="license string")
    p_init.set_defaults(func=cmd_init)

    p_upd = sub.add_parser("update-provenance", help="recompute sha256 for all bundle files")
    add_root(p_upd)
    p_upd.add_argument("--bundle", required=True, help="bundle name under artifacts/")
    p_upd.set_defaults(func=cmd_update_provenance)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
