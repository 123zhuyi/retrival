#!/usr/bin/env python3
"""kb_finalize.py - B 知识库图谱闭环收尾脚本

迁完卡后一键执行图谱闭环 + 体检（9 步）：
  1 knowledge_query build      建检索索引
  2 okf_graph build            建确定性图谱（same_topic/part_of/mentions 边）
  3 okf_graph candidates       召回候选关联边
  4 构造 judgment              读 candidates 生成 manifest + workflow_output.json
  5 okf_judge_aggregate        聚合 judgment 到 graph/edge_judgments.json
  6 okf_graph inject           把 judgment 写入卡片 #相关 块
  7 okf_graph verify           校验图谱闭环
  8 knowledge_query build      重建索引（inject 改了卡，必须重建否则 lint 报 stale）
  9 knowledge_lint             全库体检

用法：
  python3 kb_finalize.py --knowledge-root <KB> [--judge-mode auto|manual]

judge 模式：
  auto    基于内置 reason 表 + tags 交集自动判定（默认）
  manual  打印候选对 + 将生成的 reason，暂停等你检查/编辑 .build/workflow_output.json 后回车继续

扩展 reason 表：编辑下方 SPECIFIC_REASONS，或迁新卡后把新关联对加进去。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
Q = SCRIPT_DIR / "skills/triton-knowledge-query/scripts/knowledge_query.py"
G = SCRIPT_DIR / "skills/triton-knowledge-ingest/scripts/okf_graph.py"
J = SCRIPT_DIR / "skills/triton-knowledge-ingest/scripts/okf_judge_aggregate.py"
L = SCRIPT_DIR / "skills/triton-knowledge-lint/scripts/knowledge_lint.py"

# 具体理由表：frozenset([stem_a, stem_b]) -> reason
# 迁新卡后可在此补充新关联对的具体理由；未命中的对用 tags 交集兜底
SPECIFIC_REASONS = {
    frozenset(["fp32-reduction-accumulator", "two-pass-reduce-then-apply"]):
        "fp32 累加器服务于 two-pass reduce 阶段统计量",
    frozenset(["fp32-reduction-accumulator", "atomic-add-cross-tile-reduction"]):
        "atomic_add 跨 tile 合并部分和时仍需 fp32 累加保证精度",
    frozenset(["two-pass-reduce-then-apply", "atomic-add-cross-tile-reduction"]):
        "atomic_add 是 two-pass 双 kernel split 的跨 tile 归并原语",
    frozenset(["quant-fusion-int8-precision-diagnosis", "fp32-reduction-accumulator"]):
        "诊断链下钻到累加层时套用 fp32 累加原则定位根因",
    frozenset(["quant-fusion-int8-precision-diagnosis", "two-pass-reduce-then-apply"]):
        "量化算子 matmul 累加阶段的两阶段拆分复用 two-pass 模式",
    # ops/ 算子卡正向-反向/家族关联
    frozenset(["flash-attention-score", "flash-attention-score-grad"]):
        "同一 FlashAttentionScore 算子的前向与反向实现",
    frozenset(["fused-floyd-attention", "fused-floyd-attention-grad"]):
        "同一 FusedFloydAttention 算子的前向与反向实现",
    frozenset(["block-sparse-attention", "block-sparse-attention-grad"]):
        "同一 BlockSparseAttention 算子的前向与反向实现",
    frozenset(["moe-token-permute", "moe-token-permute-grad"]):
        "同一 MoeTokenPermute 算子的前向与反向实现",
    frozenset(["moe-token-unpermute", "moe-token-unpermute-grad"]):
        "同一 MoeTokenUnpermute 算子的前向与反向实现",
    frozenset(["moe-token-permute-with-ep", "moe-token-permute-with-ep-grad"]):
        "同一 MoeTokenPermuteWithEP 算子的前向与反向实现",
    frozenset(["moe-finalize-routing", "moe-finalize-routing-v2"]):
        "MoE finalize routing 的版本迭代",
    frozenset(["moe-finalize-routing-v2", "moe-finalize-routing-v2-grad"]):
        "同一 MoeFinalizeRoutingV2 算子的前向与反向实现",
    frozenset(["moe-init-routing", "moe-init-routing-v2"]):
        "MoE init routing 的版本迭代",
    frozenset(["moe-init-routing-v2", "moe-init-routing-v3"]):
        "MoE init routing V2/V3 版本迭代",
    frozenset(["moe-init-routing-quant", "moe-init-routing-quant-v2"]):
        "MoE init routing 量化版本迭代",
    frozenset(["moe-gating-top-k-softmax", "moe-gating-top-k-softmax-v2"]):
        "gating top-k softmax 版本迭代",
    frozenset(["nsa-compress-attention", "nsa-compress-attention-infer"]):
        "NSA 压缩注意力的训练/推理变体",
    frozenset(["nsa-compress-attention", "nsa-compress-grad"]):
        "同一 NSACompressAttention 算子的前向与反向实现",
    frozenset(["nsa-compress-attention", "nsa-compress-with-cache"]):
        "NSA 压缩注意力带 KV cache 的变体",
    frozenset(["nsa-compress-attention", "nsa-selected-attention"]):
        "NSA 压缩/选择注意力算子家族",
    frozenset(["mla-prolog", "mla-prolog-v3"]):
        "MLA prolog 的版本迭代",
    frozenset(["mla-prolog", "mla-preprocess"]):
        "MLA prolog/preprocess 算子家族",
    frozenset(["lightning-indexer", "quant-lightning-indexer"]):
        "lightning indexer 的量化变体",
    frozenset(["prompt-flash-attention", "incre-flash-attention"]):
        "FlashAttention 的 prompt/incremental 场景变体",
    frozenset(["flash-attention-score", "prompt-flash-attention"]):
        "FlashAttentionScore 与 prompt 场景变体同属 attention 家族",
    frozenset(["sparse-flash-attention", "kv-quant-sparse-flash-attention"]):
        "sparse FlashAttention 及其 KV 量化变体",
    frozenset(["elementwise-abs", "elementwise-add"]):
        "elementwise 基础算子",
    frozenset(["rmsnorm-rope", "rope-pointer-addressing"]):
        "RoPE 相关算子/指针处理模式",
}


def stem(path):
    return path.split("/")[-1].replace(".md", "")


def run(cmd, label):
    print(f"\n── {label} ──")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout.strip()
    if out:
        lines = out.split("\n")
        if len(lines) > 12:
            print("\n".join(lines[:7]))
            print(f"  ... ({len(lines) - 11} more lines)")
            print("\n".join(lines[-4:]))
        else:
            print(out)
    if r.stderr.strip():
        print("[stderr]", r.stderr.strip()[:400])
    return r.returncode


def build_judgment(kb, mode):
    cand_path = kb / ".build/candidates.json"
    if not cand_path.exists():
        print(f"[ERROR] {cand_path} 不存在，请确认 candidates 步骤成功")
        return None
    cand = json.load(open(cand_path))
    ids = list(cand.keys())
    manifest = [{"idx": i, "id": cid} for i, cid in enumerate(ids)]

    results = []
    total_pairs = 0
    for i, (focal, v) in enumerate(cand.items()):
        kept = []
        focal_tags = v.get("card", {}).get("tags", [])
        for c in v.get("candidates", []):
            tid = c["id"]
            pair = frozenset([stem(focal), stem(tid)])
            reason = SPECIFIC_REASONS.get(pair)
            if reason is None:
                common = set(c.get("tags", [])) & set(focal_tags)
                reason = (f"主题相关：共享 {', '.join(sorted(common))}"
                          if common else "同属优化知识，主题相关")
            kept.append({"target_id": tid, "type": "related", "reason": reason})
            total_pairs += 1
        results.append({"idx": i, "kept": kept})

    out_dir = kb / ".build/cand"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out_dir / "manifest.json", "w"),
              ensure_ascii=False, indent=1)
    wo_path = kb / ".build/workflow_output.json"
    json.dump({"result": {"results": results}}, open(wo_path, "w"),
              ensure_ascii=False, indent=1)
    print(f"  构造 judgment: {len(ids)} focal, {total_pairs} 候选对 -> {wo_path.name}")

    if mode == "manual" and total_pairs > 0:
        print("\n  候选对（可编辑 .build/workflow_output.json 调整 type/reason 后回车继续）：")
        for i, (focal, v) in enumerate(cand.items()):
            for c in v.get("candidates", []):
                r = next((k["reason"] for k in results[i]["kept"]
                          if k["target_id"] == c["id"]), "")
                print(f"    {stem(focal)} <-> {stem(c['id'])} : {r}")
        try:
            input("\n  回车继续 aggregate（或 Ctrl+C 中止，手动跑后续步骤）...")
        except KeyboardInterrupt:
            print("\n  已中止。workflow_output.json 已生成，编辑后可手动跑 aggregate。")
            return None
    return wo_path


def main():
    ap = argparse.ArgumentParser(description="B 知识库图谱闭环收尾脚本")
    ap.add_argument("--knowledge-root", required=True, help="知识库根目录")
    ap.add_argument("--judge-mode", choices=["auto", "manual"], default="auto",
                    help="judgment 模式：auto 全自动；manual 打印候选对并暂停")
    args = ap.parse_args()
    kb = Path(args.knowledge_root).resolve()
    if not kb.is_dir():
        print(f"[ERROR] knowledge-root 不是目录: {kb}")
        return 2
    for script, name in [(Q, "knowledge_query"), (G, "okf_graph"),
                         (J, "okf_judge_aggregate"), (L, "knowledge_lint")]:
        if not script.exists():
            print(f"[ERROR] 找不到 {name} 脚本: {script}")
            return 2

    print(f"知识库: {kb}")
    print(f"judge 模式: {args.judge_mode}")

    if run([sys.executable, str(Q), "--knowledge-root", str(kb), "build"],
           "1/9 knowledge_query build") != 0:
        return 1
    if run([sys.executable, str(G), "--knowledge-root", str(kb), "build"],
           "2/9 okf_graph build") != 0:
        return 1
    if run([sys.executable, str(G), "--knowledge-root", str(kb), "candidates"],
           "3/9 candidates") != 0:
        return 1

    print("\n── 4/9 构造 judgment ──")
    wo = build_judgment(kb, args.judge_mode)
    if wo is None:
        return 1

    if run([sys.executable, str(J), "--knowledge-root", str(kb), str(wo)],
           "5/9 aggregate") != 0:
        print("[FAIL] aggregate 失败")
        return 1
    if run([sys.executable, str(G), "--knowledge-root", str(kb), "inject"],
           "6/9 inject") != 0:
        print("[FAIL] inject 失败")
        return 1
    if run([sys.executable, str(G), "--knowledge-root", str(kb), "verify"],
           "7/9 okf_graph verify") != 0:
        print("[FAIL] okf_graph verify 失败")
        print("  可能原因：有卡 never judged（candidates 未召回或 judgment 缺失）")
        print("  建议：检查 .build/candidates.json 是否覆盖所有卡；手动跑 okf_graph verify 看详情")
        return 1
    if run([sys.executable, str(Q), "--knowledge-root", str(kb), "build"],
           "8/9 重建 knowledge_query 索引（inject 改了卡）") != 0:
        return 1

    rc = run([sys.executable, str(L), "--knowledge-root", str(kb)], "9/9 lint")
    if rc != 0:
        print("\n[WARN] lint 有 blocker/warn，请检查上方输出")
        print("  常见：watermark 缺失（根 index.md 未写 source pin）；frontmatter 字段缺失；死链")
        return 1

    print("\n✓ 闭环完成：build → candidates → judge → aggregate → inject → verify → 索引 → lint 全绿")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
