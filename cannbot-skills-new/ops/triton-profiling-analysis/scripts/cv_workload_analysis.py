#!/usr/bin/env python3
"""按计算草图（kernel 结构）分析每个 C(cube)/V(vector) 侧的数据搬运量与计算量。

两路证据交叉：
  1. 实测（trace instr_exe.csv，core0 的 1C+2V 子核）：MMAD/向量算术指令计数、
     各搬运 pipe（MTE1/MTE2/MTE3/FIXP）指令计数与占用时间；
  2. 草图解析（源码循环结构）：每迭代的 load/store/dot 数 × 迭代数。

口径：
  - C 计算量 = MMAD 数 × 8192 FLOP（单条 MMAD = 16×16×16 MAC × 2）
  - V 计算量 = Σ向量算术指令 × 128 lane × 2（FMA 计 2；标量/掩码不计）——⚠ 近似
  - 搬运量 = 各搬运 pipe 指令计数（条数）+ 占用时间（µs，CSV 直读）；
    字节数若给 shapes 则按 tile 尺寸×条数估算（--flops-bytes 时给出）
  - 仿真为策略B收缩负载（T=256，4 链步），计数为单 core 单收缩负载值；
    --scale 给放大倍数（c1 全程为 ×8）估算 c1 全程量级

用法: cv_workload_analysis.py <roundsN_dir> [--scale 8]
"""
import argparse, csv, glob, os, sys
from collections import defaultdict

VEC_ARITH = {"VADD","VMUL","VSUB","VCONV","VSQRT","VEXP","VMAX","VMIN","VSEL","VABS",
             "VDIV","VADDS","VMULS","VCADD","VCMPV","VCMPVS","VMOVMASK_XN","VNCHWCONV",
             "VREC","VBRCB","VCOPY","VNOT","VSHR","VSHL"}
MTE_CUBE = {"MOV_SRC_TO_DST_ALIGN","MOV_OUT_TO_L1_MULTI_ND2NZ","LOAD_L1_TO_DST_3DV2",
            "LOAD_2D","FIX_L0C_TO_DST"}
MTE_VEC = {"MOV_SRC_TO_DST_ALIGN","MOV_OUT_TO_L1_MULTI_ND2NZ","LOAD_L1_TO_DST_3DV2"}


def agg_kernel(rdir, kern):
    """core0 的 1C+2V 子核指令聚合。返回 {side: {"compute":…, "mte":…, "mte_time":…}}"""
    out = {}
    for f in glob.glob(os.path.join(
            rdir, f"round*___gdnv_{kern}*kernel*/OPPROF_*/simulator/core0.*/*instr_exe.csv")):
        sub = os.path.basename(os.path.dirname(f)).split("_instr_exe")[0]
        side = "C" if "cubecore" in sub else "V"
        cnt, tm = defaultdict(int), defaultdict(float)
        for r in csv.DictReader(open(f)):
            cnt[r["instr"]] += int(r["call_count"])
            tm[r["instr"]] += float(r["running_time(us)"])
        d = out.setdefault(side, {"mmad": 0, "vec": 0, "mte": 0, "mte_time": 0.0, "fixp_time": 0.0})
        d["mmad"] += cnt.get("MMAD", 0)
        d["vec"] += sum(cnt[i] for i in VEC_ARITH if i in cnt)
        mte_set = MTE_CUBE if side == "C" else MTE_VEC
        d["mte"] += sum(cnt[i] for i in mte_set if i in cnt)
        d["mte_time"] += sum(tm[i] for i in mte_set if i in tm)
        d["fixp_time"] += tm.get("FIX_L0C_TO_DST", 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round_dir")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="收缩→全程放大倍数（c1 全程 = ×8，T2048/T256）")
    a = ap.parse_args()
    s = a.scale
    print(f"## 每 kernel C/V 侧 搬运量与计算量（{os.path.basename(a.round_dir)}，"
          f"core0 子核，缩放 ×{s}）\n")
    print(f"| kernel | C 计算(MFLOP) | C 搬运(条/µs) | V 计算(MFLOP) | V 搬运(条/µs) |")
    print("|---|---|---|---|---|")
    for kern in ["prep_is", "state", "prep_g", "out", "gate"]:
        r = agg_kernel(a.round_dir, kern)
        if not r: continue
        C = r.get("C", {}); V = r.get("V", {})
        c_flop = C.get("mmad", 0) * 8192 * s / 1e6
        v_flop = V.get("vec", 0) * 128 * 2 * s / 1e6
        print(f"| {kern} | {c_flop:.1f} | {C.get('mte',0)*s:.0f} 条 / {C.get('mte_time',0)*s:.1f}µs"
              f" | {v_flop:.1f} | {V.get('mte',0)*s:.0f} 条 / {V.get('mte_time',0)*s:.1f}µs |")
    print("\n> 口径：C 计算=MMAD×8192FLOP；V 计算=向量算术指令×128lane×2(FMA，近似)；"
          "搬运=各 MTE pipe 指令条数与占用时间（instr_exe.csv 直读，core0 单核）。")


if __name__ == "__main__":
    main()
