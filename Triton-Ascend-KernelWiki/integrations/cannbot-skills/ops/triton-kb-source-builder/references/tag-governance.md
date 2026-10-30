# Controlled tag governance

Treat `kernelwiki/data/tags.yaml` as the only writable vocabulary authority used by page validation. Treat `kernelwiki/data/tag-governance.yaml` as the update policy.

## Existing tag selection

- `architectures`: use an exact chip model from the chip-model column of cannBot `npu-hardware-params.md`.
- `hardware_features`: use stable capabilities from cannBot `ops/npu-arch` references; do not encode numeric capacities or core counts.
- `kernel_types`: use CANN `ops-nn`, `ops-math`, or `ops-transformer` classification.
- `symptoms`: use a reviewed verifier-error class.
- `techniques`: use a reviewed KernelWiki optimization method.

Never add `languages`. Never write architecture selectors such as A2, A3, A5, `DAV_*`, `archXX`, or a product family into page tags.

## Unknown tag workflow

1. Keep the observed phrase and supporting evidence in the Source Packet.
2. Run `build`; inspect `tag-proposals.yaml`.
3. Confirm that the proposal has the correct controlled field and authoritative source.
4. Open a dedicated taxonomy change for human review.
5. Update the authoritative source manifest and controlled vocabulary in that change.
6. Update an alias or tagging rule only when the authoritative evidence supports it.
7. Run a dry retag, corpus validation, indices, UT, and ST.
8. Rerun the blocked Source Builder build after the taxonomy change lands.

The Source Builder never applies the proposal. A candidate containing an unknown controlled value remains `blocked`.

## Architecture-specific updates

For a new chip model:

1. Confirm the exact value in `ops/npu-arch/references/npu-hardware-params.md`.
2. Add that exact value to `tags.yaml/architectures`.
3. Pin the cannBot commit in `npu-architecture-taxonomy-source.yaml`.
4. Add query selectors to `architecture-aliases.yaml` only when their one-to-many mapping is explicit.
5. Add exact-text tagging rules only when documents name the chip model directly.

Do not infer a specific chip model from a family selector.
