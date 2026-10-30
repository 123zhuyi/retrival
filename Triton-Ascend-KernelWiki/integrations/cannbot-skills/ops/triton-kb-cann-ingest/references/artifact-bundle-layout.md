# Artifact bundle layout

Every ingestion produces one bundle under `kernelwiki/artifacts/<bundle-name>/`. The bundle holds derived material that does not belong in a wiki page body, plus a `PROVENANCE.yaml` that makes it traceable and integrity-checked. `validate.py` enforces PROVENANCE existence + per-file sha256, so a bundle with a wiki `artifact_dir` pointer but no PROVENANCE (or a stale sha256) will fail validation.

## Bundle name

`<operator>` or `<operator>-<variant>` - match the `artifact_dir` referenced by the wiki page. Examples in repo: `flash-attention-score`, `elementwise-abs`.

## Required file: `PROVENANCE.yaml`

Schema-canonical (see `kernelwiki/data/schemas.yaml#artifact-bundle-provenance`):

```yaml
origin_url: local://cann/<repo-short>          # or the upstream clone URL
upstream_repo: <org/name>                      # e.g. kunpengcompute/CANN-ops-transformer
upstream_sha: <40-char commit>                 # pinned commit of the analyzed source
upstream_path: <op dir relative to repo root>  # e.g. attention/flash_attention_score/
license: CANN Open Software License Agreement Version 2.0
retrieved_at: YYYY-MM-DD
asset_mode: derived                             # verbatim | extracted | derived
derived_from: <source-page-id or commit>        # e.g. source-cann-ops-math-abs
fixture: true                                   # true = excerpts for analysis, not full-repo copy
files:
  - local_path: <file>
    role: derived-source                        # upstream-file|extracted-block|derived-source|approach-notes|bench-record
    mode: derived                               # verbatim|extracted|derived
    sha256: <64-hex>                            # MUST match the file's actual sha256
```

`validate.py` checks: every `local_path` exists and its `sha256` matches. It does **not** enforce `role`/`mode` vocab, but use the schema-canonical values above for consistency with `flash-attention-score`. After editing any bundle file, recompute with `scaffold.py update-provenance --bundle <bundle>`.

## Typical derived files

Match the existing bundles; add only what the operator needs.

| File | role | Content |
|---|---|---|
| `source_file_inventory.md` | approach-notes | §3 source-file table: path, role, centrality, key symbols, line ranges, deps, op relation |
| `<operator>-triton-skeleton.py` | derived-source | §10 Triton skeleton (pseudocode/snippet); header states "not compiled/benchmarked" |
| `tiling_dispatch_tables.md` | approach-notes | §5/§6/§7: CANN tiling table + Triton abstraction map + template/dispatch table |
| `pipeline-diagram.md` | approach-notes | §8 pipeline stage table (CANN mechanism -> purpose -> Triton abstraction -> transferable?) |
| `migration-table.md` | approach-notes | §9 four-bucket classification table |
| `codegen_constraints.yaml` | approach-notes | derived constraints a later generator agent must obey (dtype rules, tail mask, FP32 path) |

## Do not

- Copy whole CANN source files into the bundle (use `fixture: true` excerpts; keep bounded snippets with `path Lxx-Lyy`).
- Put a file in `files[]` that you did not create, or leave a placeholder sha256.
- Reference an `artifact_dir` from a wiki page without creating the matching `PROVENANCE.yaml`.

## Relationship to sources

`PROVENANCE.derived_from` should point to the source page id (e.g. `source-cann-ops-math-abs`), and the source page's "derived references" section should list this bundle. The wiki page `sources:` lists the source page id; the wiki page `artifact_dir:` points here. Traceability chain: wiki `sources` -> source page -> `PROVENANCE.upstream_sha` -> CANN commit.
