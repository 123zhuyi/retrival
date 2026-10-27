#!/usr/bin/env python3
"""dispatch_parallel.py — 并行启动多个 route 子 CLI，每个 route 独立执行 .claude/references/algorithms/fusion-search/route.md 的完整 pipeline。

用法:
  python3 dispatch_parallel.py --manifests route-1/.task_manifest.json route-2/.task_manifest.json --timeout 3600
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="dispatch_parallel.py — 并行启动多个 route 子 CLI")
    parser.add_argument("--manifests", nargs="+", required=True, help="task manifest 文件路径列表")
    parser.add_argument("--timeout", type=int, default=3600, help="每个 route 的超时时间（秒），默认 3600")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 文件路径，默认写入第一个 manifest 的 work_dir 同级")
    return parser.parse_args()


def run_single_route(manifest_path: str, timeout: int) -> dict:
    """启动单个 route 子 CLI 并等待其完成。"""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        return {"route_id": None, "manifest_path": str(manifest_path), "status": "error", "error": "manifest not found"}

    manifest = json.loads(manifest_path.read_text())
    route_id = manifest.get("route_id", Path(manifest_path).parent.name)
    work_dir = Path(manifest.get("work_dir", Path(manifest_path).parent))

    work_dir.mkdir(parents=True, exist_ok=True)

    # 将 manifest_path 写入 work_dir 内的 .task_manifest.json
    target_manifest = work_dir / ".task_manifest.json"
    target_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # 写一个简单的 AGENTS.md 指向 route 规范
    agents_md = work_dir / "AGENTS.md"
    if not agents_md.exists():
        agents_md.write_text(f"""---
name: triton-route-worker-{route_id}
description: Route Worker for fusion-search — 执行 .claude/references/algorithms/fusion-search/route.md 完整 pipeline
mode: primary
temperature: 0.1
---

# Route Worker: {route_id}

你是 Fusion-Search 的 Route Worker。你的完整规范在 `.claude/references/algorithms/fusion-search/route.md`。

请读取 `{target_manifest}` 后按照规范执行。
""")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            ["claude", "--print", "--output-format", "json", "--model", "claude-sonnet-4-6"],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "CLAUDE_CODE_WORK_DIR": str(work_dir)},
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            elapsed = time.time() - start_time

            result = {
                "route_id": route_id,
                "manifest_path": str(manifest_path),
                "work_dir": str(work_dir),
                "status": "success" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "stderr": stderr.decode("utf-8", errors="replace")[:1000] if stderr else "",
            }

            # 尝试读取 round_result.json
            round_result_path = work_dir / "round_result.json"
            if round_result_path.exists():
                result["round_result"] = json.loads(round_result_path.read_text())

            return result

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.time() - start_time
            return {
                "route_id": route_id,
                "manifest_path": str(manifest_path),
                "work_dir": str(work_dir),
                "status": "timeout",
                "elapsed_seconds": round(elapsed, 1),
                "error": f"Route exceeded timeout of {timeout}s",
                "pid": proc.pid,
            }

    except Exception as e:
        return {
            "route_id": route_id,
            "manifest_path": str(manifest_path),
            "work_dir": str(work_dir),
            "status": "error",
            "error": str(e),
        }


def main():
    args = parse_args()

    manifests = [Path(m).resolve() for m in args.manifests]
    if not manifests:
        print("Error: No manifests provided", file=sys.stderr)
        sys.exit(1)

    print(f"[dispatch_parallel] Starting {len(manifests)} routes in parallel (timeout={args.timeout}s)")

    results = {}
    with ThreadPoolExecutor(max_workers=len(manifests)) as executor:
        futures = {executor.submit(run_single_route, str(m), args.timeout): str(m) for m in manifests}
        for future in as_completed(futures):
            result = future.result()
            manifest = futures[future]
            route_id = result.get("route_id", "unknown")
            status = result.get("status", "error")
            results[route_id] = result
            print(f"  [{route_id}] {status}")

    # 写入结果
    output_path = Path(args.output) if args.output else Path(manifests[0]).parent.parent / "parallel_result.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    print(f"[dispatch_parallel] Done: {success_count}/{len(manifests)} routes succeeded")
    print(f"  Result written to: {output_path}")


if __name__ == "__main__":
    main()