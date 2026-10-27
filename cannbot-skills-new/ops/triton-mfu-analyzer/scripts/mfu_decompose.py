#!/usr/bin/env python3
"""把 ops-profiling `--quick --mfu` 的 MFU 按 kernel 拆开，并按 dtype 修正峰值。

解决两个问题
------------
1. **整体 MFU 说不清是谁拖的后腿。** quick 模式的分母是「一次迭代内所有 kernel
   的 task_time 之和」。当算子由多段语义组成（如 MHA = 4 次投影 GEMM + attention），
   整体 MFU 是各段的时间加权结果，看不出哪一段有问题。本脚本把理论 FLOPs 也按
   同样的缝拆开，算出每段各自的 MFU。

2. **峰值不区分 dtype。** 见 peak_calib.py 的说明。本脚本可读入 peak.json，
   按每个 case 的输入 dtype 选分母。

输入来自采集产出的 `performance.json`：它记录了本次实际测了哪几个 case、
每个 case 的 PROF 目录绝对路径、以及 warmup/repeats。所以默认不需要任何额外参数，
**采集测了哪三个 case，这里就拆哪三个**。

前置条件
--------
- 采集时加了 `--keep-prof`（否则 PROF 目录已被删除）
- 算子目录有 `flops.py`，除 ops-profiling 要求的 `flops(case) -> float` 外，
  可选再暴露：

      def flops_parts(case) -> dict[str, float]:
          '''把理论 FLOPs 拆成若干段，key 为段名。各段之和应等于 flops(case)。'''

      KERNEL_PARTS = {"linear_kernel": "proj", "flash_attention_kernel": "attn"}
      # kernel 名 → 段名。同一段可映射多个 kernel 名。

  两者都提供时才做拆分；只有 flops() 时退化为「整体 MFU + 各 kernel 耗时」。

用法
----
    python3 mfu_decompose.py --op-dir . --peak-json peak.json
    python3 mfu_decompose.py --op-dir . --report run1_top3_performance.json
    python3 mfu_decompose.py --op-dir . --cases 46,49        # 只看其中几个
"""
import argparse
import collections
import csv
import glob
import importlib.util
import json
import sys
from pathlib import Path

META_KERNEL_TYPES = ("PROFILING_ENABLE", "PROFILING_DISABLE", "TASK_TIMEOUT_SET", "")


def load_hook(op_dir: Path):
    path = op_dir / "flops.py"
    if not path.is_file():
        sys.exit(f"[ERR] 找不到 {path}；先按 references/flops-guide.md 写一个")
    spec = importlib.util.spec_from_file_location("op_flops_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "flops", None)):
        sys.exit(f"[ERR] {path} 未暴露 flops(case) -> float")
    return mod


def load_cases(op_dir: Path):
    """与 ops-profiling 的加载优先级保持一致，保证 case 下标对得上。"""
    for pattern in ("*_perf_cases.jsonl", "*.jsonl"):
        files = sorted(op_dir.glob(pattern))
        if files:
            return _read_lines(files[0]), files[0].name
    for f in sorted(op_dir.glob("*.json")):
        if f.name.endswith(".bak") or f.name in ("performance.json", "perf_report.json",
                                                 "peak.json", "mfu_decompose.json"):
            continue
        raw = f.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(raw)
            return (parsed if isinstance(parsed, list) else [parsed]), f.name
        except ValueError:
            return _read_lines(f), f.name
    sys.exit(f"[ERR] {op_dir} 下找不到用例文件")


def _read_lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def case_dtype(case):
    """取第一个 tensor 输入的 dtype，作为该 case 的峰值查表键。"""
    for inp in case.get("inputs", []):
        if inp.get("type") == "tensor" and inp.get("dtype"):
            return inp["dtype"]
    return "float16"


def find_task_time_csv(prof_dir, fallback_glob):
    """PROF 目录里的 task_time_*.csv。prof_dir 为空或已删除时退回 glob。"""
    if prof_dir and Path(prof_dir).is_dir():
        hits = sorted(glob.glob(f"{prof_dir}/mindstudio_profiler_output/task_time_*.csv"))
        if hits:
            return hits[0], None
        hits = sorted(glob.glob(f"{prof_dir}/PROF_*/mindstudio_profiler_output/task_time_*.csv"))
        if hits:
            return hits[0], None
    hits = sorted(glob.glob(fallback_glob))
    if hits:
        return hits[0], None
    return None, (f"找不到 task_time csv（PROF 目录 {prof_dir!r} 不存在或为空；"
                  f"回退 glob 也没命中）。采集时是否加了 --keep-prof？")


def kernel_times(csv_path, warmup, repeats):
    """返回 {kernel_name: 单次迭代平均 us}。

    与 ops-profiling 的解析口径一致：按 task_start 排序后切成 warmup+repeats 份，
    丢弃前 warmup 份，对剩下的 active 份取平均。注意是**对同名 kernel 的多次发射
    求和**（一次迭代内发射 4 次就计 4 次），而不是按名字去重。
    """
    kernels = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r.get("kernel_type", "") in META_KERNEL_TYPES:
                continue
            try:
                dur = float(r.get("task_time(us)") or 0)
                start = float((r.get("task_start(us)") or "0").strip())
            except ValueError:
                continue
            if dur > 0:
                kernels.append((start, r.get("kernel_name", "unknown"), dur))

    if not kernels:
        return None, "no kernel rows"
    n_runs = warmup + repeats
    if len(kernels) % n_runs:
        return None, (f"{len(kernels)} kernel 行不能被 warmup+repeats={n_runs} 整除，"
                      "无法可靠切分迭代")
    kernels.sort()
    per_run = len(kernels) // n_runs
    agg = collections.Counter()
    for _, name, dur in kernels[warmup * per_run:]:
        agg[name] += dur / repeats
    return dict(agg), None


def main():
    ap = argparse.ArgumentParser(description="按 kernel 拆解 MFU")
    ap.add_argument("--op-dir", default=".", help="算子目录（含 flops.py、用例、performance.json）")
    ap.add_argument("--report", default="performance.json",
                    help="采集产出的报告（相对 --op-dir 或绝对路径）")
    ap.add_argument("--cases", help="只看其中几个 case 下标，逗号分隔（默认报告里的全部）")
    ap.add_argument("--impl", default="asc", choices=["asc", "ref"])
    ap.add_argument("--warmup", type=int, help="覆盖报告里的 warmup（默认从报告读）")
    ap.add_argument("--repeats", type=int, help="覆盖报告里的 repeats（默认从报告读）")
    ap.add_argument("--nominal-peak", type=float, default=353.89,
                    help="标称 cube 峰值 TFLOPS（与 ops-profiling 报的口径一致）")
    ap.add_argument("--peak-json", help="peak_calib.py 产出的 peak.json，按 dtype 修正分母")
    ap.add_argument("--prof-root", default="/tmp", help="PROF 目录不在报告记录处时的回退根目录")
    ap.add_argument("--tag", default="", help="回退查找用：采集时的 --prof-tag")
    args = ap.parse_args()

    op_dir = Path(args.op_dir).resolve()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = op_dir / report_path
    if not report_path.is_file():
        sys.exit(f"[ERR] 找不到 {report_path}\n"
                 f"      先跑采集：msprof_perf_summary.py --quick --mfu --output-dir "
                 f"{op_dir} --keep-prof")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_case = report.get("per_case") or []
    if not per_case:
        sys.exit(f"[ERR] {report_path} 里没有 per_case")

    warmup = args.warmup if args.warmup is not None else int(report.get("warmup", 0))
    repeats = args.repeats if args.repeats is not None else int(report.get("repeats", 1))

    mod = load_hook(op_dir)
    cases, case_file = load_cases(op_dir)
    parts_fn = getattr(mod, "flops_parts", None)
    kmap = getattr(mod, "KERNEL_PARTS", None)
    do_split = callable(parts_fn) and isinstance(kmap, dict)

    achievable = {}
    if args.peak_json:
        achievable = json.loads(Path(args.peak_json).read_text())["peak_tflops"]

    wanted = {int(x) for x in args.cases.split(",")} if args.cases else None
    key_us, key_prof = f"{args.impl}_us", f"{args.impl}_prof_dir"

    print(f"# 报告: {report_path.name}   用例文件: {case_file}   "
          f"warmup={warmup} repeats={repeats}")
    print(f"# 本次采集测了 {len(per_case)} 个 case；拆分: "
          f"{'开' if do_split else '关（flops.py 未提供 flops_parts + KERNEL_PARTS）'}")
    print(f"# 标称峰值: {args.nominal_peak} TFLOPS"
          + (f"   实测上限: {achievable}" if achievable else ""))

    part_names = sorted(set(kmap.values())) if do_split else []
    head = (f"{'case':<5}{'dtype':<10}{'total us':>9}{'TFLOPS':>8}"
            f"{'MFU@nom':>9}{'MFU@ach':>9}")
    for p in part_names:
        head += f"{p + ' us':>10}{p + ' MFU':>10}"
    print(head)
    print("-" * len(head))

    results = []
    for entry in per_case:
        i = entry.get("case")
        if i is None or (wanted is not None and i not in wanted):
            continue
        if entry.get("skipped") or not entry.get(key_us):
            print(f"{i:<5}(skip: {entry.get('skipped') or entry.get(f'{args.impl}_error')})")
            continue

        fallback = (f"{args.prof_root}/msprof_quick_{args.impl}_{op_dir.name}_c{i}"
                    f"{'_' + args.tag if args.tag else ''}"
                    f"/PROF_*/mindstudio_profiler_output/task_time_*.csv")
        csv_path, err = find_task_time_csv(entry.get(key_prof), fallback)
        if csv_path is None:
            print(f"{i:<5}(skip: {err})")
            continue
        times, err = kernel_times(csv_path, warmup, repeats)
        if times is None:
            print(f"{i:<5}(skip: {err})")
            continue

        dt = case_dtype(cases[i])
        total_us = sum(times.values())
        try:
            total_f = float(mod.flops(cases[i]))
        except Exception as e:  # noqa: BLE001
            print(f"{i:<5}(skip: flops() 失败 {e})")
            continue
        if total_f <= 0 or total_us <= 0:
            print(f"{i:<5}(skip: flops={total_f} total_us={total_us})")
            continue

        tflops = total_f / (total_us * 1e-6) / 1e12
        ach = achievable.get(dt)
        row = {"case": i, "dtype": dt, "total_us": total_us, "actual_tflops": tflops,
               "mfu_nominal_pct": tflops / args.nominal_peak * 100,
               "mfu_achievable_pct": (tflops / ach * 100) if ach else None,
               "kernels": times}
        line = (f"{i:<5}{dt:<10}{total_us:>9.1f}{tflops:>8.1f}"
                f"{row['mfu_nominal_pct']:>8.2f}%"
                + (f"{row['mfu_achievable_pct']:>8.1f}%" if ach else f"{'-':>9}"))

        if do_split:
            pf = {k: float(v) for k, v in parts_fn(cases[i]).items()}
            pt, unmapped = collections.Counter(), []
            for kname, us in times.items():
                part = kmap.get(kname)
                (unmapped.append(kname) if part is None else pt.update({part: us}))
            if unmapped:
                print(f"      [warn] case {i} 有未映射 kernel（不计入任何段）: {unmapped}")
            row["parts"] = {}
            for p in part_names:
                us, f = pt.get(p, 0.0), pf.get(p, 0.0)
                denom = ach or args.nominal_peak
                m = f / (us * 1e-6) / 1e12 / denom * 100 if us > 0 and f > 0 else 0.0
                row["parts"][p] = {"us": us, "flops": f, "mfu_pct": m}
                line += f"{us:>10.1f}{m:>9.1f}%"
            drift = abs(sum(pf.values()) - total_f) / total_f
            if drift > 1e-6:
                print(f"      [warn] case {i} flops_parts 之和与 flops() 差 {drift:.2%}")
        print(line)
        results.append(row)

    if not results:
        sys.exit("[ERR] 没有可拆解的 case")

    best = max(results, key=lambda r: r["mfu_nominal_pct"])
    print(f"\n最大 MFU@标称: {best['mfu_nominal_pct']:.2f}%  (case {best['case']}, {best['dtype']})")
    ok = [r for r in results if r["mfu_achievable_pct"] is not None]
    if ok:
        b = max(ok, key=lambda r: r["mfu_achievable_pct"])
        print(f"最大 MFU@实测上限: {b['mfu_achievable_pct']:.1f}%  (case {b['case']}, {b['dtype']})")
    out = op_dir / "mfu_decompose.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] 明细写入 {out}")


if __name__ == "__main__":
    main()
