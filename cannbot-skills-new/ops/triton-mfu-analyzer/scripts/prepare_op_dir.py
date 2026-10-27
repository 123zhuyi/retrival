#!/usr/bin/env python3
"""把任意布局的算子目录适配成 ops-profiling `--quick` 要求的布局。

ops-profiling 的 quick/compare 模式写死了三样东西：

    model.py                 里面必须有 class Model      （参考实现）
    model_new_ascendc.py     里面必须有 class ModelNew   （被测实现，Triton 实现同样放这里）
    *_perf_cases.jsonl       JSON Lines 用例，每行一个 case

而实际算子目录往往是 `<OpName>.py` / `<OpName>.json` / `best_xxx/best_generated_code.py`
这类布局，直接跑会报 "no module named model" 或加载不到用例。

本脚本做三件事：

1. 按上面的命名复制文件（**复制而非改名**，原目录保持不动）。
2. **补齐 `__file__` 派生的用例路径**。很多参考实现是这样找用例的：

       json_path = os.path.splitext(__file__)[0] + ".json"     # → model.json

   改名成 model.py 之后它会去找 model.json。脚本会额外把用例复制成 `model.json`，
   并扫描两个实现文件里出现的 `*.json` 字面量，逐个复制一份同名文件。
   （这是本 skill 最容易踩的坑：跑起来不报错，但 case 加载失败会静默走兜底路径。）
3. 校验 `class Model` / `class ModelNew` 是否存在，不存在直接报错而不是等到采集时才崩。

用法
----
    python3 prepare_op_dir.py \
        --ref   <算子目录>/20_FlashAttentionV2.py \
        --impl  <算子目录>/best_n2/best_generated_code.py \
        --cases <算子目录>/20_FlashAttentionV2.json \
        --dst   <算子目录>/mfu_run
"""
import argparse
import re
import shutil
import sys
from pathlib import Path


def check_class(path: Path, cls: str):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not re.search(rf"^\s*class\s+{cls}\b", text, re.M):
        sys.exit(f"[ERR] {path} 里找不到 `class {cls}`。\n"
                 f"      ops-profiling 按固定类名反射，请先在源文件里改名或加一个别名："
                 f"\n        {cls} = <你的类名>")
    return text


def json_literals(text: str):
    """扫源码里的 *.json 字面量，用于补齐硬编码的用例文件名。"""
    return sorted(set(re.findall(r"[\"']([\w./-]+\.json)[\"']", text)))


def main():
    ap = argparse.ArgumentParser(description="适配算子目录为 ops-profiling 布局")
    ap.add_argument("--ref", required=True, help="参考实现 .py（须含 class Model）")
    ap.add_argument("--impl", required=True, help="被测实现 .py（须含 class ModelNew）")
    ap.add_argument("--cases", required=True, help="用例 .json / .jsonl（JSON Lines）")
    ap.add_argument("--dst", required=True, help="目标工作目录（会被创建）")
    args = ap.parse_args()

    ref, impl, cases = Path(args.ref), Path(args.impl), Path(args.cases)
    for p in (ref, impl, cases):
        if not p.is_file():
            sys.exit(f"[ERR] 不存在: {p}")

    ref_text = check_class(ref, "Model")
    impl_text = check_class(impl, "ModelNew")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    written = []
    shutil.copy(ref, dst / "model.py"); written.append("model.py")
    shutil.copy(impl, dst / "model_new_ascendc.py"); written.append("model_new_ascendc.py")

    # ops-profiling 优先读 *_perf_cases.jsonl
    stem = cases.stem
    shutil.copy(cases, dst / f"{stem}_perf_cases.jsonl")
    written.append(f"{stem}_perf_cases.jsonl")

    # __file__ 派生路径 + 源码里硬编码的 json 名
    aliases = {"model.json", cases.name}
    aliases |= {Path(j).name for j in json_literals(ref_text) + json_literals(impl_text)}
    for name in sorted(aliases):
        shutil.copy(cases, dst / name)
        written.append(name)

    n = sum(1 for l in cases.read_text(encoding="utf-8-sig").splitlines() if l.strip())
    print(f"[OK] {dst}")
    for w in written:
        print(f"     {w}")
    print(f"\n用例数: {n}")
    print("下一步：写 flops.py（见 references/flops-guide.md），然后按 SKILL.md 第 3 步采集。")
    print("注意：上面为兼容 __file__ 派生路径复制了多份同名用例，改用例时记得**全部**同步。")


if __name__ == "__main__":
    main()
