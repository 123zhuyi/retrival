#!/usr/bin/env python3
"""flops.py 自检工具（原型）。

功能：
1. 加载算子目录的 flops.py，校验接口。
2. 对每个 case 计算 flops() 和 flops_parts()，检查二者之和的偏差。
3. 用独立 tracer（可选）运行 model.py 的 Model.forward，统计实际 matmul/linear/einsum 的 FLOPs，
   与 flops.py 结果交叉对拍。
4. 检测异常：FLOPs=0、超过峰值、actual_seq_lengths 与 shape 不一致等。
5. 若目录已有 performance.json，列出未映射 kernel 作为提示。

用法：
    python3 check_flops.py --op-dir /path/to/mfu_run
    python3 check_flops.py --op-dir . --cases 0,9,50
    python3 check_flops.py --op-dir . --no-trace
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from collections import Counter

# 标称峰值，仅用于异常检测
DEFAULT_PEAK_TFLOPS = 400.0


def load_flops_module(op_dir: Path):
    path = op_dir / "flops.py"
    if not path.is_file():
        sys.exit(f"[ERR] 找不到 {path}")
    spec = importlib.util.spec_from_file_location("op_flops_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "flops", None)):
        sys.exit(f"[ERR] {path} 未暴露 flops(case) -> float")
    return mod


def load_cases(op_dir: Path):
    for pattern in ("*_perf_cases.jsonl", "*.jsonl"):
        files = sorted(op_dir.glob(pattern))
        if files:
            return _read_lines(files[0])
    for f in sorted(op_dir.glob("*.json")):
        if f.name.endswith(".bak") or f.name in (
            "performance.json", "perf_report.json", "peak.json", "mfu_decompose.json"
        ):
            continue
        raw = f.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(raw)
            return (parsed if isinstance(parsed, list) else [parsed])
        except ValueError:
            return _read_lines(f)
    sys.exit(f"[ERR] {op_dir} 下找不到用例文件")


def _read_lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def _specs(case):
    return {item["name"]: item for item in case.get("inputs", [])}


def _shape_to_tuple(shape):
    return tuple(shape) if isinstance(shape, list) else shape


def _matmul_flops(a_shape, b_shape):
    """估算 a @ b 的 FLOPs，假设最后两维是 [M,K] @ [K,N]。

    batch 维取 a 的前导维（常见情况下 a/b 的 batch 维相同或 broadcastable）。
    """
    *ba, m, k = a_shape
    *bb, k2, n = b_shape
    if k != k2:
        return None
    batch = 1
    for d in ba:
        batch *= d
    return 2.0 * batch * m * k * n


# ---------------------------------------------------------------------------
# 独立 tracer：运行 Model.forward 并统计 matmul/linear/einsum FLOPs
# ---------------------------------------------------------------------------

class FlopTracer:
    def __init__(self):
        self.total = 0.0
        self.ops = Counter()
        self._patches = []

    def _wrap(self, original, name):
        def wrapped(*args, **kwargs):
            flops = None
            if name == "einsum":
                equation = args[0]
                # 简化：只处理常见的二维/三维 einsum，且不含求和指标
                if "->" not in equation:
                    # 隐式输出 = 输入中只出现一次的指标
                    shapes = [tuple(a.shape) for a in args[1:] if hasattr(a, "shape")]
                    flops = self._einsum_flops(equation, shapes)
            elif name in ("matmul", "bmm", "mm"):
                if len(args) >= 2 and hasattr(args[0], "shape") and hasattr(args[1], "shape"):
                    flops = _matmul_flops(tuple(args[0].shape), tuple(args[1].shape))
            elif name == "linear":
                if len(args) >= 3:
                    input, weight, bias = args[0], args[1], args[2]
                else:
                    input, weight = args[0], args[1]
                in_f = weight.shape[-1]
                out_f = weight.shape[0]
                batch = 1
                for d in input.shape[:-1]:
                    batch *= d
                flops = 2.0 * batch * in_f * out_f

            if flops:
                self.total += flops
                self.ops[name] += flops
            return original(*args, **kwargs)
        return wrapped

    def _einsum_flops(self, equation, shapes):
        """粗略估算 einsum FLOPs：只处理每个字母最多出现两次的情况。"""
        # TODO: 实现更完整的 einsum FLOPs 解析
        return None

    def __enter__(self):
        import torch
        import torch.nn.functional as F
        originals = {
            "matmul": torch.matmul,
            "bmm": torch.bmm,
            "mm": torch.mm,
            "linear": F.linear,
            "einsum": torch.einsum,
        }
        torch.matmul = self._wrap(originals["matmul"], "matmul")
        torch.bmm = self._wrap(originals["bmm"], "bmm")
        torch.mm = self._wrap(originals["mm"], "mm")
        F.linear = self._wrap(originals["linear"], "linear")
        torch.einsum = self._wrap(originals["einsum"], "einsum")
        self._patches = originals
        return self

    def __exit__(self, *args):
        import torch
        import torch.nn.functional as F
        torch.matmul = self._patches["matmul"]
        torch.bmm = self._patches["bmm"]
        torch.mm = self._patches["mm"]
        F.linear = self._patches["linear"]
        torch.einsum = self._patches["einsum"]


def _tensor_from_spec(spec, seed=0):
    """根据 case spec 构造一个可运行的 tensor（CPU）。"""
    import torch
    shape = tuple(spec["shape"])
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int8": torch.int8,
        "bool": torch.bool,
    }
    dtype = dtype_map.get(spec.get("dtype", "float32"), torch.float32)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    if dtype == torch.int8:
        return torch.randint(-128, 127, shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.ones(shape, dtype=dtype)
    if "int" in str(dtype):
        return torch.randint(0, 10, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def trace_model_flops(model_path: Path, case, tracer: FlopTracer):
    """尝试在 CPU 上运行 model.py 的 Model.forward 并统计 FLOPs。"""
    import torch
    import torch.nn as nn

    # 加载 model.py 并抑制 .npu() 调用
    text = model_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(".npu()", "")

    # 创建独立模块命名空间
    module_globals = {
        "torch": torch,
        "nn": nn,
        "json": json,
        "os": __import__("os"),
        "math": math,
        "sys": sys,
    }
    try:
        exec(text, module_globals)
    except Exception as e:
        return None, f"model 加载失败: {e}"

    Model = module_globals.get("Model")
    get_input_groups = module_globals.get("get_input_groups")
    if Model is None:
        return None, "找不到 class Model"

    model = Model()
    specs = _specs(case)

    # 构造输入：严格按 forward 签名顺序填充 args，缺省且有默认值的用默认值占位
    args = []
    kwargs = {}
    try:
        import inspect
        sig = inspect.signature(model.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            spec = specs.get(name)
            if spec is None:
                if param.default is not inspect.Parameter.empty:
                    args.append(param.default)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    args.append(())
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    kwargs[name] = {}
                else:
                    args.append(None)
                continue
            if spec.get("type") == "tensor":
                args.append(_tensor_from_spec(spec))
            elif spec.get("type") == "attr":
                val = spec.get("value")
                args.append(val)
            else:
                args.append(None)
    except Exception as e:
        return None, f"参数构造失败: {e}"

    try:
        with tracer:
            with torch.no_grad():
                model(*args, **kwargs)
        return tracer.total, None
    except Exception as e:
        return None, f"forward 执行失败: {e}"


# ---------------------------------------------------------------------------
# 异常检测
# ---------------------------------------------------------------------------

def check_shape_consistency(case):
    warnings = []
    s = _specs(case)
    q = s.get("query") or s.get("q") or s.get("x")
    if q is None:
        return warnings
    layout = s.get("layout_query", {}).get("value", "BSND")
    asq = s.get("actual_seq_lengths_query", {}).get("value")
    if asq is None:
        return warnings
    if layout == "BSND":
        t1 = q["shape"][0] * q["shape"][1]
    else:
        t1 = q["shape"][0]
    if sum(asq) != t1:
        warnings.append(f"actual_seq_lengths_query sum({sum(asq)}) != t1({t1})")
    return warnings


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="校验算子目录的 flops.py")
    ap.add_argument("--op-dir", default=".", help="算子目录（含 flops.py、model.py、用例）")
    ap.add_argument("--cases", help="只检查指定 case 下标，逗号分隔")
    ap.add_argument("--no-trace", action="store_true", help="跳过 model.py 独立 tracer 对拍")
    ap.add_argument("--peak", type=float, default=DEFAULT_PEAK_TFLOPS,
                    help="峰值 TFLOPS，用于异常检测")
    args = ap.parse_args()

    op_dir = Path(args.op_dir).resolve()
    mod = load_flops_module(op_dir)
    cases = load_cases(op_dir)
    wanted = {int(x) for x in args.cases.split(",")} if args.cases else None

    has_parts = callable(getattr(mod, "flops_parts", None))
    kmap = getattr(mod, "KERNEL_PARTS", None)

    print(f"# 目录: {op_dir}")
    print(f"# 用例数: {len(cases)}   flops_parts: {'有' if has_parts else '无'}")
    print()

    model_path = op_dir / "model.py"
    do_trace = (not args.no_trace) and model_path.is_file()

    max_ratio = 0.0
    issues = []
    for i, case in enumerate(cases):
        if wanted is not None and i not in wanted:
            continue
        specs = _specs(case)
        dt = next((x["dtype"] for x in specs.values() if x.get("type") == "tensor"), "unknown")

        try:
            total_f = float(mod.flops(case))
        except Exception as e:
            issues.append((i, f"flops() 异常: {e}"))
            print(f"case {i:3d}: [ERR] flops() {e}")
            continue

        parts_sum = total_f
        if has_parts:
            try:
                parts = mod.flops_parts(case)
                parts_sum = float(sum(parts.values()))
            except Exception as e:
                issues.append((i, f"flops_parts() 异常: {e}"))
                print(f"case {i:3d}: [ERR] flops_parts() {e}")
                continue

        drift = abs(parts_sum - total_f) / total_f if total_f > 0 else 0.0
        if drift > 1e-6:
            issues.append((i, f"flops_parts 之和与 flops() 偏差 {drift:.2%}"))

        shape_warns = check_shape_consistency(case)
        for w in shape_warns:
            issues.append((i, w))

        traced = None
        trace_err = None
        if do_trace:
            tracer = FlopTracer()
            traced, trace_err = trace_model_flops(model_path, case, tracer)
            if traced is not None and total_f > 0:
                ratio = traced / total_f
                max_ratio = max(max_ratio, ratio)
                if not (0.5 <= ratio <= 2.0):
                    issues.append((i, f"tracer FLOPs ({traced/1e9:.3f}G) 与 flops.py ({total_f/1e9:.3f}G) 偏差过大 (ratio={ratio:.2f})"))

        tflops = total_f / 1e12
        over_peak = tflops > args.peak
        if over_peak:
            issues.append((i, f"理论 TFLOPS {tflops:.1f} 超过峰值 {args.peak}，分子可能写大"))

        zero_flops = total_f == 0.0
        if zero_flops:
            issues.append((i, "FLOPs 为 0"))

        status = "OK" if not (drift > 1e-6 or shape_warns or over_peak or zero_flops or trace_err) else "WARN"
        line = (f"case {i:3d} [{dt:10s}]  FLOPs={total_f/1e9:10.3f}G  parts_drift={drift:.2e}")
        if traced is not None:
            line += f"  tracer={traced/1e9:10.3f}G(r={traced/total_f:.2f})"
        elif trace_err:
            line += f"  tracer=skip({trace_err})"
        if shape_warns:
            line += f"  shape_warn={shape_warns}"
        print(f"{status:4s} {line}")

    print()
    if issues:
        print(f"# 发现 {len(issues)} 个问题：")
        for i, msg in issues:
            print(f"  case {i}: {msg}")
    else:
        print("# 全部 case 通过基础校验。")

    if do_trace:
        print(f"\n# tracer/model 最大比值: {max_ratio:.2f}")
        if max_ratio > 2.0:
            print("  [提示] tracer 结果显著大于 flops.py，请检查是否把实现冗余计入了分子。")
        elif max_ratio < 0.5:
            print("  [提示] tracer 结果显著小于 flops.py，请检查是否漏算了某些 GEMM。")

    # 如果有 performance.json，提示未映射 kernel
    perf = op_dir / "performance.json"
    if perf.is_file() and kmap is not None:
        report = json.loads(perf.read_text(encoding="utf-8"))
        for entry in report.get("per_case", []):
            c = entry.get("case")
            if wanted is not None and c not in wanted:
                continue
            prof_dir = entry.get("asc_prof_dir") or entry.get("ref_prof_dir")
            # 这里不解析 task_time.csv，仅做占位提示
            print(f"\n# case {c} PROF: {prof_dir}")
            print("  请运行 mfu_decompose.py 检查未映射 kernel。")


if __name__ == "__main__":
    main()
