# Schema

Every page has YAML frontmatter. Required fields are defined by page type in `data/schemas.yaml`. Wiki pages use canonical `sources` IDs to reference pages under `sources/`; controlled values come from `data/tags.yaml`. Every referenced `artifact_dir` contains a `PROVENANCE.yaml` with per-file SHA-256 values.

Imported Triton-Ascend documents use `symptoms` for error/problem types, CANN-aligned `kernel_types` for operator families, and `operators` for concrete API or example names. Classification rules are stored in `data/doc-tagging-rules.yaml`; their upstream CANN taxonomy commits are pinned in `data/operator-taxonomy-sources.yaml`.
