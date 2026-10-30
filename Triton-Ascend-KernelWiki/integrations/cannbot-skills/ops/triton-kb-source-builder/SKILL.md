---
name: triton-kb-source-builder
description: Build source-grounded, reviewable KernelWiki candidates from Triton-Ascend documentation, source code, pull requests, issues, or operator-generation episodes. Use when cannBot must collect immutable evidence, write a structured source packet, generate source and Wiki candidate pages, propose unknown controlled tags, verify publication gates, or publish an explicitly approved knowledge update; do not use for read-only retrieval or direct unreviewed writes to the formal knowledge base.
---

# Triton KB Source Builder

Build knowledge through four explicit stages: `collect`, `build`, `verify`, and `publish`. Keep the formal KernelWiki layout and retrieval behavior unchanged.

## Establish the source packet

Read the source material and write one YAML Source Packet. Record facts, not hidden reasoning. Include:

- an immutable source page with repository, commit or reference, URL, capture date, and evidence summary;
- one reusable Wiki candidate that cites the generated source-page id;
- exact controlled tags already present in `kernelwiki/data/tags.yaml`;
- at least one evidence record containing a local file or a bounded excerpt.

Read [references/source-packet-schema.md](references/source-packet-schema.md) before creating or editing a packet. For a pull request, inspect the final diff, description, review resolution, linked issue, tests, and CI evidence. For source code, capture the symbol, pinned commit, path, and relevant tests. For an Agent episode, capture every attempt, verifier result, and benchmark environment.

## Apply tag policy

Read [references/tag-governance.md](references/tag-governance.md) when selecting architecture, hardware, operator, error, or technique tags.

- Write only exact chip models from `npu-hardware-params.md` into `architectures`.
- Use A2, A3, A5, DAV, archXX, SocVersion, and product-family names only as query selectors.
- Leave an unsupported tag out of the candidate. If the concept must be represented, include it in the packet so the builder emits a `tag-proposals.yaml` review item.
- Never edit `tags.yaml`, `architecture-aliases.yaml`, or `doc-tagging-rules.yaml` automatically.
- Never add a language tag or a `languages` field.

## Run the stages

Set `TRITON_ASCEND_KB_ROOT` to the KernelWiki directory or pass `--root`. Use a work directory outside the formal `kernelwiki/` tree.

Run all non-publishing stages:

```bash
python .claude/skills/triton-kb-source-builder/scripts/build.py \
  --root "$TRITON_ASCEND_KB_ROOT" \
  --workdir "$WORKDIR/knowledge-builds" \
  run --packet "$WORKDIR/source-packet.yaml"
```

The command creates:

```text
knowledge-builds/<build-id>/
  source-packet.yaml
  evidence-manifest.yaml
  evidence/*
  candidates/source.md
  candidates/wiki.md
  tag-proposals.yaml
  verification.yaml
```

Resolve every blocker in `verification.yaml`. Do not reinterpret a blocked result as approval.

## Publish an approved candidate

Require a named human reviewer. Publish only after `verification.yaml` reports `ready-for-review`:

```bash
python .claude/skills/triton-kb-source-builder/scripts/build.py \
  --root "$TRITON_ASCEND_KB_ROOT" \
  --workdir "$WORKDIR/knowledge-builds" \
  publish --build-id "$BUILD_ID" --approved-by "$REVIEWER"
```

The script writes the source and Wiki pages atomically, validates the complete corpus, regenerates KernelWiki indices, and rolls back page writes if validation fails. Pass `--replace` only when the reviewer explicitly approves replacement of existing pages.

## Finish

Report the build id, evidence manifest, candidate paths, verification decision, tag proposals, reviewer, and published paths. Keep the staging directory as an audit record. Use `triton-kernelwiki-retriever` separately when an operator-generation Agent needs to consume published knowledge.
