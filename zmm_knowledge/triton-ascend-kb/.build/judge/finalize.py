#!/usr/bin/env python3
"""Merge 3-window judge outputs, aggregate into edge_judgments.json (union with
existing judgments), then refresh graph injection/viz and verify.

Usage: python3 .build/judge/finalize.py
"""
import json
import os
import shutil
import subprocess
import sys

KB = "/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807"
SCRIPTS = "/home/w00836812/shixi_test02/zmm_knowledge/Triton-Ascend-KernelWiki/skills"
J = os.path.join(KB, ".build", "judge")

def main():
    # 1. merge three window outputs
    results, covered = [], set()
    for w in ("w1", "w2", "w3"):
        p = os.path.join(J, f"workflow_output_{w}.json")
        if not os.path.exists(p):
            sys.exit(f"MISSING: {p} — 窗口 {w} 尚未完成")
        raw = json.load(open(p))
        part = raw.get("result", raw).get("results", [])
        for r in part:
            covered.add(r.get("idx"))
        results += part
        print(f"{w}: {len(part)} focal results")
    expected = set(range(303))
    missing = sorted(expected - covered)
    if missing:
        sys.exit(f"idx 覆盖不全，缺 {len(missing)} 个，如 {missing[:10]}")
    merged = os.path.join(KB, ".build", "workflow_output.json")
    json.dump({"result": {"results": results}}, open(merged, "w"), ensure_ascii=False)
    print(f"merged {len(results)} focal results -> {merged}")

    # 2. backup + aggregate (aggregate 是全量覆盖语义)
    judgments_path = os.path.join(KB, "graph", "edge_judgments.json")
    old = json.load(open(judgments_path))
    shutil.copy(judgments_path, "/tmp/edge_judgments.pre-judge.json")
    agg = os.path.join(SCRIPTS, "triton-knowledge-ingest/scripts/okf_judge_aggregate.py")
    subprocess.run([sys.executable, agg, "--knowledge-root", KB, merged], check=True)

    # 3. union old judgments back (keys 为指纹对，旧判断端点已验证存在)
    new = json.load(open(judgments_path))
    added = 0
    for k, j in old["judgments"].items():
        if k not in new["judgments"]:
            new["judgments"][k] = j
            added += 1
    json.dump(new, open(judgments_path, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print(f"old judgments merged back: +{added}, total {len(new['judgments'])}")

    # 4. inject / viz / verify
    g = os.path.join(SCRIPTS, "triton-knowledge-ingest/scripts/okf_graph.py")
    for step in ("inject", "viz", "verify"):
        print(f"--- {step} ---")
        subprocess.run([sys.executable, g, "--knowledge-root", KB, step], check=False)

if __name__ == "__main__":
    main()
