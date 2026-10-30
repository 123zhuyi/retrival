#!/usr/bin/env python3
"""批量跑算子精度测试：遍历算子目录，对每个调 evaluate_ascendc.sh。

evaluate_ascendc.sh 不在本 skill 里——自动从 .claude/skills/*/scripts/ 下查找。
查找顺序：
  1. --eval-script 参数（显式指定路径）
  2. 环境变量 EVAL_ASCENDC_SH
  3. 从本脚本位置向上找 .claude/skills/ 目录，遍历所有 skill 的 scripts/ 找 evaluate_ascendc.sh

用法：
    # 自动查找（最常见）
    python3 batch_evaluate.py <ops_dir> -o eval_report.json

    # 显式指定 evaluate_ascendc.sh
    python3 batch_evaluate.py <ops_dir> \\
        --eval-script /path/to/skills/ascendc-translator/scripts/evaluate_ascendc.sh \\
        -o eval_report.json

输出 JSON：
    {
      "summary": {"total": 64, "pass": 30, "fail": 20, "error": 10, "timeout": 2, "no_test": 2},
      "ops": {
        "20_BatchNormV3": {
          "status": "FAIL",          # PASS / FAIL / ERROR / TIMEOUT / NO_TEST
          "passed": 0, "failed": 6,
          "exit_code": 1,
          "duration_s": 5.0,
          "error_snippet": "...",
          "impl_regression": {       # 实现退化检测（AST 静态分析，不占 NPU）
              "checked": true,       # 是否检测了（无 model_new_ascendc.py 则 false）
              "regressed": false,    # true = 退化为 PyTorch 原生实现
              "regression_type": 3,  # 1-4 见 validate_ascendc_impl.py
              "suggestion": "..."
          }
        }
      }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 同目录的 AscendC 实现退化检测器（AST 静态分析）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_ascendc_impl import validate as validate_impl

RESULT_RE = re.compile(r"^Result[:：]\s*(pass|fail)\s*$", re.I | re.M)
TOTAL_RE = re.compile(
    r"Total[:：]\s*(\d+)\s*,\s*Passed[:：]\s*(\d+)\s*,\s*Failed[:：]\s*(\d+)",
    re.I,
)


def _search_skills_dir(skills_dir: Path) -> Optional[Path]:
    """在指定的 .claude/skills/ 目录下搜索 evaluate_ascendc.sh。"""
    if not skills_dir.is_dir():
        return None
    for skill_dir in sorted(skills_dir.iterdir()):
        candidate = skill_dir / "scripts" / "evaluate_ascendc.sh"
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_eval_script(explicit: Optional[str] = None) -> Optional[Path]:
    """查找 evaluate_ascendc.sh。

    优先级：
      1. 显式参数 --eval-script
      2. 环境变量 EVAL_ASCENDC_SH
      3. 从本脚本位置向上逐级找 .claude/skills/（不再找到第一个就停）
      4. 全局 ~/.claude/skills/（兜底）
    """
    # 1. 显式参数
    if explicit:
        p = Path(explicit).resolve()
        if p.is_file():
            return p
        print(f"[warn] --eval-script 路径不存在: {p}", file=sys.stderr)

    # 2. 环境变量
    env_path = os.environ.get("EVAL_ASCENDC_SH")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p
        print(f"[warn] EVAL_ASCENDC_SH 环境变量路径不存在: {p}", file=sys.stderr)

    # 3. 从本脚本位置向上逐级搜索 .claude/skills/
    this = Path(__file__).resolve()
    for parent in [this.parent] + list(this.parents):
        result = _search_skills_dir(parent / ".claude" / "skills")
        if result:
            return result

    # 4. 全局 ~/.claude/skills/（兜底）
    result = _search_skills_dir(Path.home() / ".claude" / "skills")
    if result:
        return result

    return None


def _find_impl_file(op_dir: Path) -> Optional[Path]:
    """在算子目录下找生成的 AscendC 实现文件 model_new_ascendc.py（含 kernel/ 子目录）。"""
    candidates = [op_dir / "model_new_ascendc.py",
                  op_dir / "kernel" / "model_new_ascendc.py"]
    for c in candidates:
        if c.is_file():
            return c
    # 兜底：目录下任意 model_new*.py
    for c in sorted(op_dir.rglob("model_new*.py")):
        if c.is_file():
            return c
    return None


def check_impl_regression(op_dir: Path) -> Dict[str, Any]:
    """对算子的 AscendC 实现跑退化检测（AST 静态分析，不占 NPU）。

    返回 impl_regression 字段；无实现文件时 checked=false 不判退化
    （是否算失败由调用方/无 kernel 目录的既有 NO_TEST 逻辑决定）。
    """
    impl = _find_impl_file(op_dir)
    if impl is None:
        return {"checked": False, "regressed": False,
                "regression_type": None, "suggestion": "未找到 model_new_ascendc.py"}

    try:
        code = impl.read_text(encoding="utf-8", errors="replace")
        result = validate_impl(code, filepath=str(impl))
    except Exception as e:  # 检测器崩溃不能拖垮评测
        return {"checked": False, "regressed": False,
                "regression_type": None,
                "suggestion": f"退化检测异常（不判定）: {e}"}

    if result["valid"]:
        return {"checked": True, "regressed": False,
                "regression_type": None, "suggestion": ""}
    return {"checked": True, "regressed": True,
            "regression_type": result["regression_type"],
            "suggestion": result.get("suggestion", "")}


def run_one(op_dir: Path, eval_script: Path, scripts_dir: Path, timeout: int, npu_id: str = None) -> Dict[str, Any]:
    """跑单个算子的 evaluate_ascendc.sh。用临时 WORKDIR 不污染算子根。"""
    # 实现退化检测先行（静态、毫秒级）；无 kernel/ 目录的算子照旧直接 NO_TEST
    impl_regression = check_impl_regression(op_dir) if (op_dir / "kernel").is_dir() else {
        "checked": False, "regressed": False, "regression_type": None,
        "suggestion": "算子目录无 kernel/ 子目录",
    }
    if not (op_dir / "kernel").is_dir():
        ret = {
            "status": "NO_TEST", "passed": 0, "failed": 0,
            "exit_code": None, "duration_s": 0.0,
            "error_snippet": "算子目录无 kernel/ 子目录",
        }
        ret["impl_regression"] = impl_regression
        return ret

    start = time.time()
    env = os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="batch_eval_wd_") as wd:
            workdir = Path(wd)
            # evaluate_ascendc.sh 内部硬编码找 WORKDIR/.claude/skills/<skill-name>/scripts/verification_ascendc.py
            # skill-name 可能是 "tilelang2ascend-translator" 或 "ascendc-translator" 等
            # 创建两个常见名字的软链接，指向找到 evaluate_ascendc.sh 的 scripts/ 目录
            for skill_name in ("tilelang2ascend-translator", "ascendc-translator"):
                link_dir = workdir / ".claude" / "skills" / skill_name / "scripts"
                link_dir.parent.mkdir(parents=True, exist_ok=True)
                if not link_dir.exists():
                    try:
                        link_dir.symlink_to(scripts_dir.resolve())
                    except OSError:
                        pass
            env["WORKDIR"] = str(workdir)
            env.setdefault("ASCENDC_SOC_VERSION", "Ascend910B3")
            if npu_id:
                env["ASCEND_RT_VISIBLE_DEVICES"] = npu_id
            env.setdefault("ASCENDC_CLEAN_BUILD", "1")

            proc = subprocess.run(
                ["bash", str(eval_script), str(op_dir.resolve())],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            duration = time.time() - start
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")

            rm = RESULT_RE.search(out)
            tm = TOTAL_RE.search(out)
            if rm:
                status = "PASS" if rm.group(1).lower() == "pass" else "FAIL"
                if tm:
                    passed, failed = int(tm.group(2)), int(tm.group(3))
                else:
                    passed, failed = (1, 0) if status == "PASS" else (0, -1)
                snippet = ""
                if status == "FAIL":
                    fail_lines = [l for l in out.splitlines() if "[FAIL]" in l][:5]
                    snippet = "\n".join(fail_lines)[:500] if fail_lines else out[-300:]
                return {
                    "status": status, "passed": passed, "failed": failed,
                    "exit_code": proc.returncode,
                    "duration_s": round(duration, 1),
                    "error_snippet": snippet,
                    "impl_regression": impl_regression,
                }
            snippet = out[-500:]
            return {
                "status": "ERROR", "passed": 0, "failed": -1,
                "exit_code": proc.returncode,
                "duration_s": round(duration, 1),
                "error_snippet": snippet,
                "impl_regression": impl_regression,
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT", "passed": 0, "failed": -1, "exit_code": None,
            "duration_s": round(time.time() - start, 1),
            "error_snippet": f"超时（>{timeout}s）",
            "impl_regression": impl_regression,
        }
    except Exception as e:
        return {
            "status": "ERROR", "passed": 0, "failed": -1, "exit_code": None,
            "duration_s": round(time.time() - start, 1),
            "error_snippet": f"批量脚本异常: {e}",
            "impl_regression": impl_regression,
        }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ops_dir", help="算子输出根目录（含 <num>_<OpName>/ 子目录）")
    ap.add_argument("-o", "--output", required=True, help="输出 JSON 报告路径")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="并发数（默认 4）")
    ap.add_argument("--timeout", type=int, default=300,
                    help="单算子超时秒数（默认 300）")
    ap.add_argument("--filter", action="append", default=[],
                    help="只跑目录名含该子串的算子，可重复")
    ap.add_argument("--eval-script", metavar="PATH",
                    help="显式指定 evaluate_ascendc.sh 路径"
                         "（默认自动从 .claude/skills/*/scripts/ 查找）")
    ap.add_argument("--npu-list", default=None,
                    help="NPU 卡列表，逗号分隔（如 4,5,6,7），按轮询分配给并发任务")
    args = ap.parse_args()

    # 解析 NPU 列表
    npu_list = []
    if args.npu_list:
        npu_list = [n.strip() for n in args.npu_list.split(",") if n.strip()]
    # 如果没传 --npu-list，从环境变量 EVAL_NPU_LIST 读
    if not npu_list:
        env_npu = os.environ.get("EVAL_NPU_LIST", "")
        if env_npu:
            npu_list = [n.strip() for n in env_npu.split(",") if n.strip()]

    # 查找 evaluate_ascendc.sh
    eval_script = find_eval_script(args.eval_script)
    if not eval_script:
        print("[err] 找不到 evaluate_ascendc.sh。", file=sys.stderr)
        print("      方法 1: --eval-script /path/to/evaluate_ascendc.sh", file=sys.stderr)
        print("      方法 2: export EVAL_ASCENDC_SH=/path/to/evaluate_ascendc.sh", file=sys.stderr)
        print("      方法 3: 把含 evaluate_ascendc.sh 的 skill 放到 .claude/skills/ 下", file=sys.stderr)
        return 2

    scripts_dir = eval_script.parent  # 含 verification_ascendc.py 等依赖的目录
    verif_py = scripts_dir / "verification_ascendc.py"
    if not verif_py.is_file():
        print(f"[err] {verif_py} 不存在（evaluate_ascendc.sh 依赖）", file=sys.stderr)
        return 2

    print(f"[eval] evaluate_ascendc.sh: {eval_script}")
    print(f"[eval] verification_ascendc.py: {verif_py}")

    ops_root = Path(args.ops_dir)
    if not ops_root.is_dir():
        print(f"[err] {ops_root} 不是目录", file=sys.stderr)
        return 2

    op_dirs = sorted([
        d for d in ops_root.iterdir()
        if d.is_dir() and re.match(r"^\d+_", d.name)
    ])
    if args.filter:
        op_dirs = [d for d in op_dirs if any(f in d.name for f in args.filter)]
    if not op_dirs:
        print(f"[err] {ops_root} 下没找到算子目录", file=sys.stderr)
        return 2

    npu_info = f"，NPU={','.join(npu_list)}" if npu_list else ""
    print(f"[eval] {len(op_dirs)} 个算子，并发 {args.jobs}，超时 {args.timeout}s{npu_info}")
    print()

    results: Dict[str, Any] = {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_to_op = {}
        for i, d in enumerate(op_dirs):
            npu_id = npu_list[i % len(npu_list)] if npu_list else None
            future_to_op[pool.submit(run_one, d, eval_script, scripts_dir, args.timeout, npu_id)] = d.name
        for i, fut in enumerate(as_completed(future_to_op), 1):
            op_name = future_to_op[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"status": "ERROR", "error_snippet": str(e),
                     "passed": 0, "failed": -1, "exit_code": None, "duration_s": 0.0}
            results[op_name] = r
            status = r["status"]
            extra = ""
            if status == "FAIL":
                extra = f" ({r['passed']}/{r['passed'] + r['failed']} passed)"
            elif status == "PASS":
                extra = f" ({r['passed']} passed)"
            elif status == "ERROR":
                extra = f" exit={r['exit_code']}"
            elif status == "TIMEOUT":
                extra = f" ({r['duration_s']}s)"
            print(f"  [{i}/{len(op_dirs)}] {status:7} {op_name}{extra}")

    duration = time.time() - start
    summary = {
        "total": len(results),
        "pass": sum(1 for r in results.values() if r["status"] == "PASS"),
        "fail": sum(1 for r in results.values() if r["status"] == "FAIL"),
        "error": sum(1 for r in results.values() if r["status"] == "ERROR"),
        "timeout": sum(1 for r in results.values() if r["status"] == "TIMEOUT"),
        "no_test": sum(1 for r in results.values() if r["status"] == "NO_TEST"),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "ops": results, "duration_s": round(duration, 1)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"[eval] 总结：PASS={summary['pass']}  FAIL={summary['fail']}  "
          f"ERROR={summary['error']}  TIMEOUT={summary['timeout']}  "
          f"NO_TEST={summary['no_test']}  （{duration:.1f}s）")
    print(f"[eval] 详细报告 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
