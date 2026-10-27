---
name: triton-kernelwiki-retriever
description: Retrieve source-grounded Triton-Ascend kernel knowledge from a KernelWiki corpus for triton-op-generator algorithm design, verifier-failure repair, and latency optimization. Use when an operator-generation orchestrator must build reproducible design, repair, or optimization context artifacts from official docs and reviewed Wiki pages; do not use it to verify kernels or automatically ingest new knowledge.
---

# Triton KernelWiki Retriever

Build a read-only, reproducible knowledge context for one stage of `triton-op-generator`. Keep task semantics, template constraints, retrieval evidence, and verification results separate.

## Prerequisites

Require:

- `TRITON_ASCEND_KB_ROOT` pointing to the `kernelwiki/` directory, or pass `--root`.
- A completed Phase 1 task and its operator work directory.
- The `triton_ascend_kb` Python package installed, or the skill located beside the knowledge-base project.

Never modify files below `TRITON_ASCEND_KB_ROOT` while using this skill.

## Select one mode

Use exactly one mode per invocation:

| Mode | Call point | Consumer |
|---|---|---|
| `design` | Phase 2, after template precheck and before designer | `triton-op-designer` |
| `repair` | Phase 3, after Conductor classifies a repairable verifier failure | `triton-op-coding` |
| `optimize` | Phase 4, after correctness passes and before an optimizer iteration | `triton-latency-optimizer` |

Do not call this skill from task extraction or verification. Do not use retrieval results to bypass AST, correctness, or benchmark gates.

## Build the context

Run the bundled script. Pass only facts available from the task, template precheck, verifier output, or benchmark output.

Design example:

```bash
python .claude/skills/triton-kernelwiki-retriever/scripts/retrieve.py \
  --mode design \
  --workdir "$WORKDIR" \
  --op-name LayerNorm \
  --semantic-op layer_norm \
  --category normalization \
  --kernel-type norm \
  --arch ascend910b1
```

Repair example:

```bash
python .claude/skills/triton-kernelwiki-retriever/scripts/retrieve.py \
  --mode repair \
  --workdir "$WORKDIR" \
  --op-name LayerNorm \
  --semantic-op layer_norm \
  --category normalization \
  --kernel-type norm \
  --verifier-error-file "$WORKDIR/output/iter_1/verifier_error.txt" \
  --iteration 1
```

Optimization example:

```bash
python .claude/skills/triton-kernelwiki-retriever/scripts/retrieve.py \
  --mode optimize \
  --workdir "$WORKDIR" \
  --op-name LayerNorm \
  --semantic-op layer_norm \
  --kernel-type norm \
  --keyword tiling \
  --keyword load_order \
  --iteration 0
```

Read [references/integration-contract.md](references/integration-contract.md) when adding this skill to an orchestrator, choosing `kernel_type`, diagnosing an unmapped error, or consuming the JSON schema.

## Consume the context

Read the generated JSON from `WORKDIR/knowledge/` and keep its lanes independent. Do not combine them into a new ranked list.

For `design`:

- Pass `official` pages to designer `inspirations`.
- Pass `wiki-kernel` and `wiki-technique` pages to designer `history_context`.
- Pass `hardware` pages as architecture context.
- Keep `.claude/template/{category}.md` Layer 1 constraints as the hard boundary.

For `repair`:

- Pass non-empty diagnostic lanes as `repair_knowledge_context` together with `previous_code`, `verifier_error`, and `conductor_suggestion`.
- Treat every retrieved fix as a hypothesis and rerun all verifier cases.

For `optimize`:

- Pass non-empty diagnostic lanes as `optimization_knowledge_context` together with baseline code and measured performance data.
- Accept an optimization only after optimized correctness passes and benchmark data shows no regression.

## Enforce evidence rules

Apply this order during reasoning:

1. Preserve `task_desc` semantics.
2. Obey template Layer 1 constraints.
3. Prefer reviewed Wiki pages over raw sources when both address the same decision.
4. Use official source documents as implementation evidence, not as task-specific proof.
5. Treat `experimental` and `inferred` pages as candidate ideas only.

Interpret KernelWiki `score` only as text relevance. Inspect `confidence`, `reproducibility`, `sources`, `upstream_commit`, and `performance_claims` before relying on a page. Never present retrieved performance values as results from the current task.

## Finish the stage

Record the generated context path in the iteration log or final report. Keep the JSON artifact unchanged for reproducibility. Continue through the existing designer, coding, verifier, and optimizer gates.

Do not write successful outputs back to KernelWiki. Knowledge ingestion requires the separate candidate, provenance, reproduction, and review workflow.
