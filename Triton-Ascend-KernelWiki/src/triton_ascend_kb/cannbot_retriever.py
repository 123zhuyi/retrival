from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .agent_bundle import KernelWikiAgentAdapter
from .kernelwiki import SearchFilters


SCHEMA_VERSION = "1.0"
UPSTREAM_REPO = "triton-lang/triton-ascend"

# Only map categories whose CANN type is unambiguous. Broad categories must pass
# --kernel-type explicitly or run without a tag filter.
CATEGORY_KERNEL_TYPES: dict[str, str] = {
    "index-computation": "index",
    "math-compute": "math",
    "normalization": "norm",
    "quantization": "quant",
    "tensor-transform": "conversion",
}

ARCHITECTURE_ALIASES: dict[str, str] = {
    "1001": "DAV_1001",
    "dav1001": "DAV_1001",
    "ascend910": "Ascend910",
    "2002": "DAV_2002",
    "dav2002": "DAV_2002",
    "ascend310p": "Ascend310P",
    "ascend310p1": "Ascend310P1",
    "ascend310p3": "Ascend310P3",
    "ascend310p5": "Ascend310P5",
    "ascend310p7": "Ascend310P7",
    "2201": "DAV_2201",
    "dav2201": "DAV_2201",
    "arch22": "DAV_2201",
    "a2": "A2",
    "a3": "A3",
    "910b": "Ascend910B",
    "ascend910b": "Ascend910B",
    "ascend910b1": "Ascend910B1",
    "ascend910b2": "Ascend910B2",
    "ascend910b2c": "Ascend910B2C",
    "ascend910b3": "Ascend910B3",
    "ascend910b4": "Ascend910B4",
    "ascend91093": "Ascend910_93",
    "3002": "DAV_3002",
    "dav3002": "DAV_3002",
    "ascend310b": "Ascend310B",
    "ascend310b1": "Ascend310B1",
    "ascend310b2": "Ascend310B2",
    "ascend310b3": "Ascend310B3",
    "ascend310b4": "Ascend310B4",
    "3003": "DAV_3003",
    "dav3003": "DAV_3003",
    "kirinx90": "KirinX90",
    "3113": "DAV_3113",
    "dav3113": "DAV_3113",
    "kirin9030": "Kirin9030",
    "3510": "DAV_3510",
    "dav3510": "DAV_3510",
    "arch35": "DAV_3510",
    "a5": "A5",
    "ascend950": "Ascend950",
    "ascend950dt": "Ascend950DT",
    "ascend950pr": "Ascend950PR",
}

VALID_SYMPTOMS = {
    "alignment_error",
    "compile_error",
    "illegal_memory_access",
    "out_of_memory",
    "performance_bottleneck",
    "precision_mismatch",
    "runtime_error",
    "shape_mismatch",
    "unsupported_operation",
    "ub_overflow",
    "vecutil_low",
}

# Order matters: choose the most specific diagnosis before generic runtime or
# compilation failures.
SYMPTOM_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "illegal_memory_access",
        (
            r"illegal memory",
            r"out[- ]of[- ]bounds",
            r"\boob\b",
            r"invalid (?:device )?(?:address|pointer)",
        ),
    ),
    (
        "ub_overflow",
        (r"\bub(?:size)?\b.*(?:overflow|exceed|insufficient)", r"unified buffer.*(?:overflow|exceed)"),
    ),
    ("out_of_memory", (r"out of memory", r"\boom\b", r"memory allocation failed")),
    ("shape_mismatch", (r"shape mismatch", r"size mismatch", r"dimension mismatch")),
    (
        "precision_mismatch",
        (
            r"precision mismatch",
            r"output (?:mismatch|not equal)",
            r"allclose.*(?:false|failed)",
            r"relative error",
            r"absolute error",
        ),
    ),
    (
        "unsupported_operation",
        (r"unsupported (?:operation|op|api)", r"not supported", r"unimplemented"),
    ),
    (
        "alignment_error",
        (r"alignment error", r"misaligned", r"must be aligned"),
    ),
    (
        "compile_error",
        (
            r"compile error",
            r"compilation failed",
            r"mlir",
            r"syntaxerror",
            r"indentationerror",
            r"typeerror",
        ),
    ),
    ("runtime_error", (r"runtimeerror", r"runtime error", r"device error", r"launch failed")),
)

PRECEDENCE = [
    "task_desc semantics",
    "template Layer 1 constraints",
    "reviewed KernelWiki pages",
    "official source documents",
    "experimental or inferred pages",
]


def normalize_architecture(architecture: str | None) -> str | None:
    if not architecture:
        return None
    normalized = re.sub(r"[\s_-]+", "", architecture.strip().lower())
    return ARCHITECTURE_ALIASES.get(normalized, architecture.strip())


def resolve_kernel_type(category: str | None, kernel_type: str | None) -> str | None:
    if kernel_type:
        return kernel_type.strip().lower()
    if not category:
        return None
    return CATEGORY_KERNEL_TYPES.get(category.strip().lower())


# Strip a leading numeric index prefix such as "4_" in "4_MoeComputeExpertTokens".
# Benchmark task files are named "<idx>_<OpName>.py"; without normalization the
# numeric prefix poisons the query ("4_moecomputeexperttokens" no longer substring-
# matches the wiki page title "MoeComputeExpertTokens...").
_OP_NAME_PREFIX_RE = re.compile(r"^\d+[_-]+")


def _normalize_op_name(op_name: str) -> str:
    return _OP_NAME_PREFIX_RE.sub("", op_name.strip())


def infer_symptom(verifier_error: str) -> str | None:
    lowered = verifier_error.lower()
    for symptom, patterns in SYMPTOM_PATTERNS:
        if any(re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns):
            return symptom
    return None


def _unique_nonempty(values: Iterable[str | None]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        stripped = value.strip()
        if not stripped or stripped.lower() in seen:
            continue
        seen.add(stripped.lower())
        output.append(stripped)
    return output


def _dedupe_aliases(adapter: KernelWikiAgentAdapter, values: Iterable[str | None]) -> list[str]:
    aliases = adapter.search.load_alias_expansions()
    output: list[str] = []
    seen: set[str] = set()
    for value in _unique_nonempty(values):
        canonical = aliases.get(value.lower(), value).lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        output.append(value)
    return output


def _lane(
    adapter: KernelWikiAgentAdapter,
    *,
    name: str,
    consumer_field: str,
    purpose: str,
    query: Sequence[str],
    filters: SearchFilters,
    limit: int,
) -> dict[str, Any]:
    bundle = adapter.build(query, filters=filters, limit=limit).to_dict()
    fallback_reason: str | None = None
    if not bundle["pages"]:
        # Graceful fallback: when the strict search returns nothing, relax the most
        # restrictive filters one at a time (tag first, then architecture). This
        # recovers recall when a category->tag mapping is wrong (e.g. an op tagged
        # `moe` queried via category=index-computation -> tag=index) or when a page
        # lists a sibling architecture SKU. Precision is unaffected: the fallback
        # only fires when the strict search is empty.
        for reason, relaxed in (
            ("tag", dataclasses.replace(filters, tag=None)),
            ("architecture", dataclasses.replace(filters, architecture=None)),
        ):
            if getattr(filters, reason) is None:
                continue
            relaxed_bundle = adapter.build(query, filters=relaxed, limit=limit).to_dict()
            if relaxed_bundle["pages"]:
                bundle = relaxed_bundle
                fallback_reason = reason
                break
    lane: dict[str, Any] = {
        "name": name,
        "consumer_field": consumer_field,
        "purpose": purpose,
        "filters": {
            key: value
            for key, value in {
                "type": filters.type,
                "tag": filters.tag,
                "repo": filters.repo,
                "architecture": filters.architecture,
                "symptom": filters.symptom,
            }.items()
            if value is not None
        },
        "page_count": len(bundle["pages"]),
        "bundle": bundle,
    }
    if fallback_reason:
        lane["fallback"] = (
            f"relaxed {fallback_reason} filter (strict search returned 0 pages)"
        )
    return lane


def build_context(
    *,
    root: Path,
    mode: str,
    op_name: str,
    semantic_ops: Sequence[str] = (),
    category: str | None = None,
    kernel_type: str | None = None,
    architecture: str | None = "ascend910b1",
    verifier_error: str = "",
    symptom: str | None = None,
    keywords: Sequence[str] = (),
    limit: int = 2,
) -> dict[str, Any]:
    mode = mode.strip().lower()
    if mode not in {"design", "repair", "optimize"}:
        raise ValueError(f"unsupported mode: {mode}")
    if not op_name.strip():
        raise ValueError("op_name must not be empty")
    if limit < 1 or limit > 5:
        raise ValueError("limit must be between 1 and 5")

    root = root.resolve()
    if not (root / "data" / "tags.yaml").is_file():
        raise FileNotFoundError(f"not a KernelWiki root: {root}")

    effective_kernel_type = resolve_kernel_type(category, kernel_type)
    effective_architecture = normalize_architecture(architecture)
    effective_symptom = symptom.strip().lower() if symptom else None
    if effective_symptom and effective_symptom not in VALID_SYMPTOMS:
        raise ValueError(f"unknown symptom: {effective_symptom}")
    if mode == "repair" and not effective_symptom:
        effective_symptom = infer_symptom(verifier_error)
        if not effective_symptom:
            raise ValueError("repair mode requires --symptom or a recognizable verifier error")
    if mode == "optimize":
        effective_symptom = "performance_bottleneck"

    adapter = KernelWikiAgentAdapter(root)
    # Normalize query terms (strip numeric prefix from op_name / semantic_ops) so the
    # benchmark filename's "<idx>_" prefix does not break title/keyword matching.
    # The original op_name is preserved in request.op_name for traceability.
    base_query = _dedupe_aliases(
        adapter,
        [
            _normalize_op_name(op_name),
            *(_normalize_op_name(s) for s in semantic_ops),
            *keywords,
        ],
    )
    lanes: list[dict[str, Any]] = []

    if mode == "design":
        lanes.append(
            _lane(
                adapter,
                name="official",
                consumer_field="inspirations",
                purpose="Official Triton-Ascend APIs and implementation examples",
                query=base_query,
                filters=SearchFilters(repo=UPSTREAM_REPO),
                limit=limit,
            )
        )
        lanes.append(
            _lane(
                adapter,
                name="wiki-kernel",
                consumer_field="history_context",
                purpose="Reviewed or experimental kernel cases with explicit evidence metadata",
                query=base_query,
                filters=SearchFilters(type="kernel", tag=effective_kernel_type),
                limit=limit,
            )
        )
        lanes.append(
            _lane(
                adapter,
                name="wiki-technique",
                consumer_field="history_context",
                purpose="Reusable implementation techniques for the operator category",
                query=base_query,
                filters=SearchFilters(type="technique", tag=effective_kernel_type),
                limit=limit,
            )
        )
        if effective_architecture:
            lanes.append(
                _lane(
                    adapter,
                    name="hardware",
                    consumer_field="hardware_context",
                    purpose="Architecture-specific hardware constraints",
                    query=[effective_architecture],
                    filters=SearchFilters(type="hardware", architecture=effective_architecture),
                    limit=limit,
                )
            )
    else:
        diagnostic_query = _unique_nonempty([*base_query, effective_symptom])
        consumer = "repair_knowledge_context" if mode == "repair" else "optimization_knowledge_context"
        lanes.append(
            _lane(
                adapter,
                name="official-diagnostic",
                consumer_field=consumer,
                purpose="Official documentation tagged with the current failure or bottleneck",
                query=diagnostic_query,
                filters=SearchFilters(repo=UPSTREAM_REPO, symptom=effective_symptom),
                limit=limit,
            )
        )
        lanes.append(
            _lane(
                adapter,
                name="wiki-technique-diagnostic",
                consumer_field=consumer,
                purpose="Technique pages matching the current failure or bottleneck",
                query=diagnostic_query,
                filters=SearchFilters(
                    type="technique", tag=effective_kernel_type, symptom=effective_symptom
                ),
                limit=limit,
            )
        )
        lanes.append(
            _lane(
                adapter,
                name="wiki-kernel-diagnostic",
                consumer_field=consumer,
                purpose="Kernel cases matching the current failure or bottleneck",
                query=diagnostic_query,
                filters=SearchFilters(type="kernel", tag=effective_kernel_type, symptom=effective_symptom),
                limit=limit,
            )
        )

    warnings = [
        "KernelWiki score measures text relevance only; it is not confidence, correctness, or expected speedup.",
        "Do not copy retrieved code blindly; adapt it to task_desc and Triton-Ascend constraints.",
        "Verifier and benchmark outputs remain the only correctness and performance gates.",
        "This context is read-only and must not be written back to KernelWiki automatically.",
    ]
    empty_lanes = [lane["name"] for lane in lanes if lane["page_count"] == 0]
    if empty_lanes:
        warnings.append(f"No matching pages in lanes: {', '.join(empty_lanes)}")
    if category and not effective_kernel_type:
        warnings.append(
            f"Category '{category}' has no unambiguous CANN kernel_type; retrieval ran without a tag filter."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "request": {
            "op_name": op_name,
            "semantic_ops": list(semantic_ops),
            "category": category,
            "kernel_type": effective_kernel_type,
            "architecture": architecture,
            "normalized_architecture": effective_architecture,
            "symptom": effective_symptom,
            "keywords": list(keywords),
        },
        "retrieval": {
            "engine": "kernelwiki_title_tag_body",
            "second_stage_reranking": False,
            "limit_per_lane": limit,
            "lanes_are_independent": True,
        },
        "precedence": PRECEDENCE,
        "lanes": lanes,
        "warnings": warnings,
    }


def default_output(workdir: Path, mode: str, iteration: int) -> Path:
    names = {
        "design": "design_context.json",
        "repair": f"repair_iter_{iteration}.json",
        "optimize": f"optimize_iter_{iteration}.json",
    }
    return workdir / "knowledge" / names[mode]


def write_context(context: dict[str, Any], output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build read-only KernelWiki context for triton-op-generator")
    parser.add_argument("--mode", choices=("design", "repair", "optimize"), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=os.environ.get("TRITON_ASCEND_KB_ROOT"),
        help="KernelWiki root; defaults to TRITON_ASCEND_KB_ROOT",
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--semantic-op", action="append", default=[])
    parser.add_argument("--category")
    parser.add_argument("--kernel-type")
    parser.add_argument("--arch", default="ascend910b1")
    parser.add_argument("--symptom")
    parser.add_argument("--verifier-error", default="")
    parser.add_argument("--verifier-error-file", type=Path)
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root is None:
        raise SystemExit("--root or TRITON_ASCEND_KB_ROOT is required")
    verifier_error = args.verifier_error
    if args.verifier_error_file:
        verifier_error = args.verifier_error_file.read_text(encoding="utf-8")
    context = build_context(
        root=args.root,
        mode=args.mode,
        op_name=args.op_name,
        semantic_ops=args.semantic_op,
        category=args.category,
        kernel_type=args.kernel_type,
        architecture=args.arch,
        verifier_error=verifier_error,
        symptom=args.symptom,
        keywords=args.keyword,
        limit=args.limit,
    )
    output = args.output or default_output(args.workdir.resolve(), args.mode, args.iteration)
    written = write_context(context, output)
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
