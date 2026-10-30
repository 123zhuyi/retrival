# Source Packet schema

Use YAML schema version `1`.

```yaml
schema_version: 1
build_id: lowercase-stable-id
source_type: pull_request | documentation | source_code | agent_episode
source:
  target_path: sources/prs/<repo>/PR-<number>.md
  frontmatter: {}
  body: |
    Source evidence summary.
candidate:
  target_path: wiki/techniques/<name>.md
  frontmatter: {}
  body: |
    Reusable method with applicability and verification.
evidence:
  - id: stable-evidence-id
    kind: diff | review | test | ci | documentation | source | episode
    locator: URL, commit/path/line, run id, or other stable locator
    excerpt: |
      Bounded evidence text.
    summary: What this evidence supports.
```

Use `path` instead of `excerpt` when a local evidence file exists. Resolve a relative path from the packet directory. Provide exactly one of `path` and `excerpt`.

## Required page shapes

Choose the source target according to the source:

| Source | Target | KernelWiki page type |
|---|---|---|
| Pull request | `sources/prs/.../*.md` | `source-pr` |
| Documentation | `sources/docs/.../*.md` | `source-doc` |
| Source-code snapshot | `sources/docs/.../*.md` | `source-doc` |
| Agent/reproduction episode | `sources/runs/.../*.md` | `source-run` |

Copy required frontmatter fields from `kernelwiki/data/schemas.yaml`. Give the Wiki candidate a canonical `type`, id prefix, confidence, reproducibility, related pages, and `sources` containing the generated source-page id.

## Content rules

- Pin a commit for mutable repositories.
- Separate the reported problem, implemented change, applicability, and verification.
- Do not claim a performance improvement without a baseline, environment, shapes, repeated measurement, and result evidence.
- Mark synthetic examples with `fixture: true` and a `local://` URL.
- Keep raw source facts in the source page and cross-task guidance in the Wiki page.
- Do not include chain-of-thought, private reasoning, secrets, or unbounded logs.
