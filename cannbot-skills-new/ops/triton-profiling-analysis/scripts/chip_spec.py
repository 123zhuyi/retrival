#!/usr/bin/env python3
"""读取当前机器 SOC 的平台配置（$ASCEND_HOME_PATH/{arm64-linux,aarch64-linux,x86_64-linux}/data/platform_config/<soc>.ini），
输出关键硬件规格与推导的算力峰值。

用途：让 roofline/MFU 分析用本机真实规格，不硬编码。
  python3 chip_spec.py            # 打印规格 + 推导峰值
  python3 chip_spec.py --json     # 机器可读 JSON

推导公式（cube 稠密峰值，与 triton-mfu-analyzer/peak_calib.py 口径一致）：
  peak_TFLOPS(fp16/bf16) = 2 * cube_m * cube_n * cube_k * cube_core_cnt * cube_freq(MHz) / 1e6
"""
import argparse, json, os, sys

def detect_soc():
    try:
        import torch, torch_npu  # noqa
        return torch_npu.npu.get_device_name(0)
    except Exception:
        return os.environ.get("SOC_VERSION")

def load_ini(path):
    """返回 (flat, sections)。

    flat: 跨节合并的键值（同名字段先出现者优先）——SoCInfo/AICoreSpec/DtypeMKN
          等字段在各节唯一，可安全平铺。
    sections: {节名: {k: v}}——*MemoryRates 各节有同名字段但取值不同
          （如 ddr_rate 在 AICoreMemoryRates=31、VectorCoreMemoryRates=16，
          单位均为 B/cycle/核），必须按节读取，禁止平铺覆盖。
    """
    flat, sections = {}, {}
    cur = ""
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("["):
            cur = ln.strip("[]").strip()
            sections.setdefault(cur, {})
            continue
        if "=" in ln:
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip()
            flat.setdefault(k, v)
            sections.setdefault(cur, {})[k] = v
    return flat, sections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--soc", default=None, help="指定 SOC 版本（缺省自动探测当前计算卡）")
    a = ap.parse_args()
    soc = a.soc or detect_soc()
    if not soc:
        sys.exit("[chip_spec] 无法确定 SOC（torch_npu 不可用且 SOC_VERSION 未设置）")
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        sys.exit("[chip_spec] ASCEND_HOME_PATH 未设置（先 source set_env.sh）")
    ini = None
    for arch in ("arm64-linux", "aarch64-linux", "x86_64-linux"):
        p = os.path.join(home, arch, "data", "platform_config", f"{soc}.ini")
        if os.path.exists(p):
            ini = p; break
    if not ini:
        sys.exit(f"[chip_spec] 未找到 {soc} 的 platform_config（ASCEND_HOME_PATH={home}）")
    kv, sections = load_ini(ini)
    g = lambda k: int(kv.get(k, 0) or 0)
    AIC_SEC, VEC_SEC = "AICoreMemoryRates", "VectorCoreMemoryRates"
    rg = lambda sec, k: int(sections.get(sec, {}).get(k, 0) or 0)  # 按节读 rate
    # 各搬运路径理论带宽：rate(B/cycle/核) × 频率 × 核数聚合。
    # rate 按所属节读取并乘以对应侧核数（AICoreMemoryRates→cube 核，VectorCoreMemoryRates→vector 核），
    # 禁止跨节平铺——两节同名字段取值不同（如 ddr_rate: AIC=31 / Vector=16）。
    RATE_PATHS = {  # ini字段: (说明, 所属节, 聚合核数键)
        "l1_to_l0_a_rate": ("L1→L0A", AIC_SEC, "cube"), "l1_to_l0_b_rate": ("L1→L0B", AIC_SEC, "cube"),
        "l0_c_to_ub_rate": ("L0C→UB", AIC_SEC, "cube"), "l1_to_ub_rate": ("L1→UB", AIC_SEC, "cube"),
        "ub_to_l1_rate": ("UB→L1", AIC_SEC, "cube"), "ub_to_l2_rate": ("UB→L2", VEC_SEC, "vec"),
        "ub_to_ddr_rate": ("UB→DDR", VEC_SEC, "vec"), "l2_read_rate": ("L2读", AIC_SEC, "cube"),
        "l2_write_rate": ("L2写", AIC_SEC, "cube"),
    }
    # HBM 双侧各自展示（同名 ddr_rate 两节取值不同）
    DDR_SIDES = [("DDR(HBM·C侧)", AIC_SEC, "cube"), ("DDR(HBM·V侧)", VEC_SEC, "vec")]
    cube_cores, vec_cores, freq = g("cube_core_cnt"), g("vector_core_cnt"), g("cube_freq")
    m, n, k = g("cube_m_size"), g("cube_n_size"), g("cube_k_size")
    peak = 2 * m * n * k * cube_cores * freq / 1e6 if all([m, n, k, cube_cores, freq]) else None
    out = {
        "soc": soc, "ini": ini,
        "ai_core_cnt": g("ai_core_cnt"), "cube_core_cnt": cube_cores, "vector_core_cnt": vec_cores,
        "cube_freq_mhz": freq, "cube_mnk": [m, n, k],
        "ub_size": g("ub_size"), "l1_size": g("l1_size"),
        "l0_a_size": g("l0_a_size"), "l0_b_size": g("l0_b_size"), "l0_c_size": g("l0_c_size"),
        "l2_size": g("l2_size"), "memory_size": g("memory_size"),
        "cube_peak_tflops_fp16_bf16_dense": round(peak, 2) if peak else None,
        "pipe_bandwidth_gbs": {},
        "int8_cube_tops": None, "vector_tflops_fp32": None,
    }
    # 搬运带宽
    ncore = {"cube": cube_cores, "vec": vec_cores}
    for field, (label, sec, ck) in RATE_PATHS.items():
        r = rg(sec, field)
        if r and freq:
            out["pipe_bandwidth_gbs"][label] = {
                "rate_b_per_cycle": r, "per_core_gbs": round(r * freq / 1e3, 1),
                "aggregate_tbs": round(r * freq / 1e3 * ncore[ck] / 1e3, 2)}
    for label, sec, ck in DDR_SIDES:
        r = rg(sec, "ddr_rate")
        if r and freq:
            out["pipe_bandwidth_gbs"][label] = {
                "rate_b_per_cycle": r, "per_core_gbs": round(r * freq / 1e3, 1),
                "aggregate_tbs": round(r * freq / 1e3 * ncore[ck] / 1e3, 2)}
    # pipe 视角带宽/算力映射（profiling pipe 名 ← ini 速率字段）
    # MTE2:GM→片上  MTE1:L1→L0A/B  MTE3:UB→L2/GM  FIXP:L0C→UB  CUBE/VECTOR:算力
    # 跨 GM 边界的 pipe 取瓶颈端（本侧 ddr_rate=HBM 方向锚点）
    # pipe 视角带宽，按 C(cube)/V(vector) 侧拆分——split 架构下两侧各有 MTE2/MTE3；
    # MTE1(L1→L0) 与 FIXP(L0C→UB) 仅 cube 侧（vector 无 L0/L0C）；
    # 凡跨 GM 的路径共享同一条 HBM（两侧 ddr_rate 不同：AIC 侧/V 侧各自聚合），
    # 芯片级 HBM 物理上限对两侧流量合并封顶（不分侧叠加）。
    PIPE_MAP = [  # (pipe, 侧, 路径, [ini字段], 所属节, 聚合核数键, 是否HBM共享)
        ("MTE1", "C", "L1→L0A/B 装载",  ["l1_to_l0_a_rate"], AIC_SEC, "cube", False),
        ("FIXP", "C", "L0C→UB 搬出",    ["l0_c_to_ub_rate"], AIC_SEC, "cube", False),
        ("MTE2", "C", "GM→L1 装载",     ["ddr_rate"], AIC_SEC, "cube", True),
        ("MTE3", "C", "L1→GM 写回",     ["ddr_rate"], AIC_SEC, "cube", True),
        ("MTE2", "V", "GM→UB 装载",     ["ddr_rate"], VEC_SEC, "vec",  True),
        ("MTE3", "V", "UB→GM 写回",     ["ddr_rate"], VEC_SEC, "vec",  True),
    ]
    out["pipe_view"] = {}
    for pipe, side, desc, fields, sec, ck, hbm_shared in PIPE_MAP:
        vals = [rg(sec, f) for f in fields if rg(sec, f)]
        if not vals or not freq: continue
        r = min(vals)  # 瓶颈链路（跨边界取窄端）
        key = f"{pipe}({side})"
        out["pipe_view"][key] = {"desc": desc, "rate_b_per_cycle": r,
                                 "per_core_gbs": round(r * freq / 1e3, 1),
                                 "aggregate_tbs": round(r * freq / 1e3 * ncore[ck] / 1e3, 2),
                                 "hbm_shared": hbm_shared}
    # int8 cube 峰值（DtypeMKN: DT_INT8=m,k,n）
    mkn = (kv.get("DT_INT8") or "").split(",")
    if len(mkn) == 3 and cube_cores and freq:
        im, ik, in_ = (int(x) for x in mkn)
        out["int8_cube_tops"] = round(2 * im * ik * in_ * cube_cores * freq / 1e6, 1)
    # vector fp32（vec_calc_size lanes × FMA(2) × vec核 × 频率）——lane 口径为假设
    vcs = g("vec_calc_size")
    if vcs and vec_cores and freq:
        out["vector_tflops_fp32"] = round(vcs * 2 * vec_cores * freq / 1e6, 1)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print(f"SOC={soc}  ini={ini}")
        print(f"  核数: ai={out['ai_core_cnt']} cube={cube_cores} vector={vec_cores}  freq={freq}MHz")
        print(f"  存储: UB={out['ub_size']//1024}KB L1={out['l1_size']//1024}KB "
              f"L0A/B={out['l0_a_size']//1024}KB L0C={out['l0_c_size']//1024}KB "
              f"L2={out['l2_size']//1024//1024}MB HBM={out['memory_size']//1024**3}GB")
        if peak:
            print(f"  cube 稠密峰值(fp16/bf16): {peak:.1f} TFLOPS  (2×{m}×{n}×{k}×{cube_cores}×{freq}M)")
        if out["int8_cube_tops"]:
            print(f"  cube INT8 峰值: {out['int8_cube_tops']} TOPS  |  vector(fp32,FMA): {out['vector_tflops_fp32']} TFLOPS(假设)")
        print("  各 pipe 理论带宽（按 C/V 侧拆分；标 ★ 的共享同一条 HBM）:")
        for pipe, d in out["pipe_view"].items():
            tag = " ★HBM共享" if d.get("hbm_shared") else ""
            print(f"    {pipe:<10} {d['desc']:<16} 每核 {d['per_core_gbs']:>7.1f} GB/s → 聚合 {d['aggregate_tbs']:>6.2f} TB/s{tag}")
        hb = [d for d in out["pipe_view"].values() if d.get("hbm_shared")]
        if hb:
            cap = min(d["aggregate_tbs"] for d in hb)
            print(f"    ※ 标★的 C/V 各侧 GM 流量共享同一条 HBM 物理链路（两侧合并封顶 ≈{cap} TB/s，"
                  f"取各侧路径聚合的较窄者，不分侧叠加；芯片级 HBM 标称上限以白皮书为准）")
        print("  搬运路径理论带宽（rate B/cycle × 频率 × 核数，明细）:")
        for label, d in out["pipe_bandwidth_gbs"].items():
            print(f"    {label:<10} {d['rate_b_per_cycle']:>4} B/cyc → 每核 {d['per_core_gbs']:>7.1f} GB/s → 聚合 {d['aggregate_tbs']:>6.2f} TB/s")

if __name__ == "__main__":
    main()
