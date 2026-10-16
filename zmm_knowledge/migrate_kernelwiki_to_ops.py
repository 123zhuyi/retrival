#!/usr/bin/env python3
"""migrate_kernelwiki_to_ops.py — 把 A(kernelwiki/wiki/kernels) 算子 wiki 迁成 B(triton-ascend-kb/ops) operator_spec 卡.

要点:
- 只转 frontmatter + 资源链接 + 内部链接脱敏, 不改内容结构.
- status: active, confidence: inferred.
- resource 优先从 artifact PROVENANCE / .git 读, 缺 sha 时回退到 A 仓库源文件链接.
- [[...]] 转成反引号, 避免 B lint 死链 warn.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import yaml


def _str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(str, _str_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(list, _list_representer, Dumper=yaml.SafeDumper)


SOURCE_REPO = Path("/home/w00836812/shixi_test02/Triton-Ascend-KernelWiki").resolve()
SOURCE_KERNEL_DIR = SOURCE_REPO / "kernelwiki" / "wiki" / "kernels"
ARTIFACTS_ROOT = SOURCE_REPO / "kernelwiki" / "artifacts"
TARGET_KB = Path("/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb").resolve()
TARGET_OPS = TARGET_KB / "ops"

RE_SHA_IN_URL = re.compile(r"https://gitcode\.com/([^/]+/[^/]+?)(?:\.git)?/?$")
RE_GITCODE_URL = re.compile(r"https://gitcode\.com/")


def git_info(path: Path) -> Optional[tuple[str, str]]:
    """返回 (owner/repo, HEAD_sha), 失败返回 None."""
    if not path.exists():
        return None
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        remote = subprocess.check_output(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    m = RE_SHA_IN_URL.match(remote)
    if not m:
        return None
    owner_repo = m.group(1).rstrip("/")
    return owner_repo, sha


def source_repo_info() -> tuple[str, str]:
    info = git_info(SOURCE_REPO)
    if info:
        return info
    # 兜底: 手动已知
    return "gcw_85ZVbQ4M/Triton-Ascend-KernelWiki", "HEAD"


def normalize_tag(value: str) -> str:
    s = value.strip().strip("\"'").lower()
    s = re.sub(r"[\s\-/\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    # 去掉纯结构性标签
    if s in {"cann", "triton", "triton_ascend", "example", "glossary", "index"}:
        return ""
    return s


def collect_tags(fm: dict) -> list[str]:
    raw: list[str] = []
    for key in ("tags", "techniques", "hardware_features", "architectures",
                "kernel_types", "operators", "symptoms"):
        val = fm.get(key)
        if isinstance(val, list):
            raw.extend(str(v) for v in val)
        elif isinstance(val, str):
            raw.extend(v.strip() for v in val.split(",") if v.strip())
    cleaned = []
    seen = set()
    for v in raw:
        t = normalize_tag(v)
        if t and t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


def extract_description(body: str, title: str) -> str:
    # 去掉标题行, 取第一个非空段落
    lines = body.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().startswith("#"):
        i += 1
    i += 1  # 跳过标题行
    paras = []
    cur = []
    for line in lines[i:]:
        stripped = line.strip()
        if not stripped:
            if cur:
                paras.append(" ".join(cur))
                cur = []
            if len(paras) >= 1:
                break
            continue
        if stripped.startswith("#"):
            break
        cur.append(stripped)
    if cur:
        paras.append(" ".join(cur))
    desc = paras[0] if paras else title
    # 去掉 markdown 标记
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
    desc = re.sub(r"[`*_~]+", "", desc)
    if len(desc) > 240:
        desc = desc[:237] + "..."
    return desc


def infer_upstream_path(basename: str, fm: dict) -> str:
    snake = basename.replace("-", "_")
    ktypes = [str(k).lower() for k in fm.get("kernel_types", [])]
    if "attention" in ktypes or "attention" in basename:
        return f"attention/{snake}/"
    if "moe" in ktypes or "moe" in basename:
        return f"moe/{snake}/"
    if "math" in ktypes or "math" in basename:
        return f"math/{snake}/"
    return ""


def resolve_resource(basename: str, fm: dict) -> tuple[str, str]:
    """返回 (resource_url, note)."""
    artifact_dir = fm.get("artifact_dir", f"artifacts/{basename}")
    artifact_path = SOURCE_REPO / "kernelwiki" / artifact_dir
    if not artifact_path.exists():
        artifact_path = ARTIFACTS_ROOT / basename

    owner_repo: Optional[str] = None
    sha: Optional[str] = None
    path: str = ""

    if artifact_path.exists():
        prov = artifact_path / "PROVENANCE.yaml"
        if prov.exists():
            try:
                with prov.open(encoding="utf-8") as f:
                    pdata = yaml.safe_load(f) or {}
            except Exception:
                pdata = {}
            upstream_repo = (pdata.get("upstream_repo", "") or "").lower()
            upstream_sha = pdata.get("upstream_sha") or pdata.get("derived_from", "")
            if isinstance(upstream_sha, str) and upstream_sha.startswith("commit-"):
                upstream_sha = upstream_sha[len("commit-"):]
            upstream_path = pdata.get("upstream_path", "")
            if upstream_path:
                path = upstream_path.strip().strip("/")

            # 本地 fixture / demo / 无法解析的 upstream 直接走 fallback
            if not upstream_repo or upstream_repo.startswith("local") or "fixture" in upstream_repo or "demo" in upstream_repo:
                owner_repo = None
                sha = None
            # 按 upstream_repo 映射到子模块/真实仓库, 从而拿到正确的远程和 sha
            elif "ops-transformer" in upstream_repo:
                ginfo = git_info(ARTIFACTS_ROOT / "ops-transformer")
                if ginfo:
                    owner_repo, sha = ginfo
            elif "ops-math" in upstream_repo:
                ginfo = git_info(ARTIFACTS_ROOT / "ops-math")
                if ginfo:
                    owner_repo, sha = ginfo
            else:
                # 自己的 .git, 但远程必须和声明的 upstream_repo 对上
                ginfo = git_info(artifact_path)
                if ginfo:
                    own, s = ginfo
                    # 简单匹配: upstream_repo 的仓库名应出现在实际 remote 中
                    if upstream_repo.split("/")[-1].replace("cann-", "").replace("ops-", "") in own:
                        owner_repo, sha = own, s
                    elif upstream_repo.split("/")[-1] in own:
                        owner_repo, sha = own, s
                    else:
                        owner_repo = None
                        sha = None
                else:
                    owner_repo = None
                    sha = None
        else:
            # 无 PROVENANCE 的目录当作无真实上游, 走 fallback
            owner_repo = None
            sha = None

    if path:
        path = path.strip("/")
    else:
        inferred = infer_upstream_path(basename, fm)
        if inferred:
            path = inferred.strip("/")

    if owner_repo and sha:
        url = f"https://gitcode.com/{owner_repo}/tree/{sha}/"
        if path:
            url += path + "/"
        return url, "artifact"

    # 回退: 指向 A 仓库的原始 wiki 文件(blob)
    sowner, ssha = source_repo_info()
    url = f"https://gitcode.com/{sowner}/blob/{ssha}/kernelwiki/wiki/kernels/{basename}.md"
    return url, "fallback"


def convert_body(body: str, basename: str, resource: str) -> str:
    # 去掉旧 frontmatter (调用前已去掉, 这里保险)
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            body = body[end + 4:].lstrip("\n")

    note = (
        f"> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/{basename}.md` 迁移而来，"
        f"原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。"
        f"资源 pinned 到 [{resource}]({resource})。\n\n"
    )
    return note + body.strip() + "\n"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate_one(src_path: Path) -> dict:
    basename = src_path.stem
    raw = src_path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{src_path} 没有 frontmatter")
    end = raw.find("\n---", 3)
    if end == -1:
        raise ValueError(f"{src_path} frontmatter 未闭合")
    fm = yaml.safe_load(raw[3:end]) or {}
    body = raw[end + 4:].lstrip("\n")

    resource, res_note = resolve_resource(basename, fm)

    # 去掉旧 frontmatter 并把 [[...]] 转成反引号
    if body.startswith("---"):
        body = body[body.find("\n---", 3) + 4:].lstrip("\n") if body.find("\n---", 3) != -1 else body
    body = re.sub(r"\[\[([^\]]+)\]\]", r"`\1`", body)

    description = extract_description(body, fm.get("title", basename))
    tags = collect_tags(fm)

    # confidence 诚实写 inferred
    confidence = "inferred"
    if isinstance(fm.get("confidence"), str) and fm["confidence"] in {
        "verified", "source_reported", "inferred", "experimental"
    }:
        # 即使 A 写 high/verified, 迁移后也降级为 inferred, 因为未在 B 环境实测
        confidence = "inferred"

    new_fm = {
        "schema_version": "okf.v1",
        "kind": "operator",
        "type": "operator_spec",
        "source_family": "ops",
        "title": fm.get("title", basename),
        "description": description,
        "tags": tags,
        "confidence": confidence,
        "status": "active",
        "resource": resource,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }

    sources_block = f"\nsources:\n  - url: {resource}\n    role: primary\n"

    out_body = convert_body(body, basename, resource)

    out_text = (
        "---\n"
        + yaml.safe_dump(new_fm, allow_unicode=True, sort_keys=False, width=float("inf"))
        + "---\n"
        + sources_block
        + "\n"
        + out_body
    )

    out_path = TARGET_OPS / f"{basename}.md"
    out_path.write_text(out_text, encoding="utf-8")

    return {
        "basename": basename,
        "resource": resource,
        "res_note": res_note,
        "tags": tags,
        "confidence": confidence,
    }


def update_ops_index() -> None:
    text = """---
kind: index
type: section_index
title: ops/ 算子参考卡
description: 从 Triton-Ascend-KernelWiki 迁移的 48 张算子语义 / tiling / dispatch 参考卡（operator_spec）。当前为静态分析级知识（inferred），后续挑高频算子改写成成功案例。
---

# ops/ 算子参考卡

本目录收录 48 张算子级参考卡，来源为 A 的 `kernelwiki/wiki/kernels/`。每张卡记录算子语义、输入输出、tiling key / dispatch、Agent actions 与 Triton 骨架，置信度 `inferred`（源码静态分析，未硬件实测）。

案例卡与 `runbooks/` 条目经「实例化 ⇄ 已知实例」双向回链，后续由 `triton-knowledge-*-ingest` skill 按需增补为真正的成功案例。
"""
    (TARGET_OPS / "index.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    args = parser.parse_args()

    if not SOURCE_KERNEL_DIR.exists():
        raise SystemExit(f"源目录不存在: {SOURCE_KERNEL_DIR}")
    if not args.dry_run:
        TARGET_OPS.mkdir(parents=True, exist_ok=True)

    files = sorted(SOURCE_KERNEL_DIR.glob("*.md"))
    print(f"发现源文件: {len(files)} 个")

    for src in files:
        if args.dry_run:
            print(f"[DRY] 将迁移 {src.name}")
            continue
        info = migrate_one(src)
        print(f"已写入 ops/{info['basename']}.md  resource={info['res_note']}")

    if not args.dry_run:
        update_ops_index()
        print(f"已更新 ops/index.md")


if __name__ == "__main__":
    main()
