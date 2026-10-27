#!/usr/bin/env python3
"""标定本机 NPU 各 dtype 的实测算力上限，产出 peak.json。

为什么需要这一步
----------------
ops-profiling 的 `--mfu` 从 cube 规格推峰值：

    peak = 2 * cube_m * cube_k * cube_n * ai_core_cnt * cube_freq / 1e6

这个式子里没有 dtype，推出来的实际是 **fp16/bf16** 的 cube 峰值（910B2/B2C
= 353.89 TFLOPS）。fp32 case 用它当分母会系统性低估 —— 实测 fp32 的 cube
只有约 1/4 速率，用 fp16 峰值算出来的 MFU 要乘 ~4 才是真实利用率。

本脚本用一个足够大的方阵 GEMM（默认 4096³，已进入算力饱和区）测出各 dtype
的实测上限，写入 peak.json 供 mfu_decompose.py 做分母。

用法
----
    python3 peak_calib.py                      # 打屏 + 写 ./peak.json
    python3 peak_calib.py -o /path/peak.json -n 4096
"""
import argparse
import json
import time

import torch
import torch_npu  # noqa: F401  (注册 npu 后端)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def bench(dtype, n, warmup, iters):
    """返回 (us/iter, TFLOPS)。"""
    a = torch.randn(n, n, dtype=torch.float32).to(dtype).npu()
    b = torch.randn(n, n, dtype=torch.float32).to(dtype).npu()
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1e6, 2.0 * n * n * n / dt / 1e12


def main():
    ap = argparse.ArgumentParser(description="各 dtype 实测算力上限标定")
    ap.add_argument("-n", "--size", type=int, default=4096, help="方阵边长（默认 4096）")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("-o", "--output", default="peak.json")
    ap.add_argument("--dtypes", default="float16,bfloat16,float32")
    args = ap.parse_args()

    out = {"gemm_n": args.size, "iters": args.iters, "peak_tflops": {}, "us_per_iter": {}}
    print(f"{'dtype':<10} {'N':>6} {'us/iter':>10} {'TFLOPS':>10}")
    for name in args.dtypes.split(","):
        name = name.strip()
        us, tf = bench(DTYPES[name], args.size, args.warmup, args.iters)
        out["peak_tflops"][name] = round(tf, 2)
        out["us_per_iter"][name] = round(us, 1)
        print(f"{name:<10} {args.size:>6} {us:>10.1f} {tf:>10.2f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] 写入 {args.output}")
    print("注意：这是**实测可达上限**，略低于标称峰值（本机 fp16 约为标称的 90%）。")
    print("     用它当分母得到的是「离实际可达上限多远」；要对齐官方口径请用标称峰值。")


if __name__ == "__main__":
    main()
