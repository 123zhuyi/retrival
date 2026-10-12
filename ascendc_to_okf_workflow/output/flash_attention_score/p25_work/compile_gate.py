"""P2.5 ①编译门：对 _sa_attn_fwd_kernel（self-attention 前向注意力核）做配置网格真编译。

[EXP-skeleton] Ascend910_9372 / CANN 9.1.0 / triton 3.2.0
每个配置：构造小输入直发 kernel（BLOCK_S/BLOCK_HD constexpr 为 cache key，必然触发 JIT），
记录 compile_ok / 错误信息。UB 上界估算按 mapping §4 静态表（910_93 192KB 校准）。

运行：`python p25_work/compile_gate.py npu`
"""
import os, sys, json, math

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # C-019
import torch
import torch_npu  # noqa
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triton_skeleton as sk  # noqa

DEV = sys.argv[1] if len(sys.argv) > 1 else "npu"


def try_kernel(dtype, S, hd, B=1):
    """直发 kernel 触发 JIT；返回 (ok, err)。shape 取 [B,S,hd] 单头（grid=1 编译即验）。"""
    bs = triton.next_power_of_2(S)
    bhd = triton.next_power_of_2(hd)
    g = torch.Generator(device="cpu").manual_seed(0)
    q = (torch.randn(B, S, hd, generator=g, dtype=torch.float32)).to(dtype).to(DEV)
    k = (torch.randn(B, S, hd, generator=g, dtype=torch.float32)).to(dtype).to(DEV)
    v = (torch.randn(B, S, hd, generator=g, dtype=torch.float32)).to(dtype).to(DEV)
    o = torch.empty(B, S, hd, device=DEV, dtype=torch.float32).contiguous()
    scale = 1.0 / math.sqrt(hd)
    try:
        sk._sa_attn_fwd_kernel[(1,)](
            q.contiguous(), k.contiguous(), v.contiguous(), o,
            B, S, hd, scale, BLOCK_S=bs, BLOCK_HD=bhd)
        torch.npu.synchronize()
        return True, ""
    except Exception as e:  # noqa: BLE001
        torch.npu.synchronize()
        return False, f"{type(e).__name__}: {str(e)[:600]}"


def _ub_est_kb(bs, bhd, dt):
    """粗估单 program 片上占用（fp32 计算域 4B）：scores + exp/pn + q/k/v/o。"""
    s = bs * bs * 4                 # scores [S,S]
    p = bs * bs * 4 * 2             # exp + pn
    mm = bs * bhd * 4 * 4           # q/k/v/o
    return (s + p + mm) / 1024.0


# 配置网格：level4 shape 代表面（S∈{5..59}、hd∈{20..65}，next_pow2 对齐）× 三 dtype
CONFIGS = []
for dt, tag in [(torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    for S, hd in [
        (7, 24), (11, 24), (13, 24),   # 主形状 hd=24 ≤64（%预算安全带 ≤45%）
        (5, 20), (53, 64), (59, 48),   # 边界 hd 档
        (31, 32), (13, 65),            # hd 超 64 档（BLOCK_HD=128 贴顶面，fp32 验证）
    ]:
        CONFIGS.append((tag, S, hd))

results = []
for tag, S, hd in CONFIGS:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[tag]
    bs, bhd = triton.next_power_of_2(S), triton.next_power_of_2(hd)
    ok, err = try_kernel(dtype, S, hd)
    chip = 950 if bhd <= 64 else 950          # UB 可用主体按 950KB（mapping §5 [SRC]）
    ub = _ub_est_kb(bs, bhd, dtype)
    if tag == "fp32":
        pass                                   # 计算域恒 fp32，估算已含 fp32 位宽
    row = dict(kernel="attention_fwd", dtype=tag, S=S, hd=hd, BLOCK_S=bs, BLOCK_HD=bhd,
               ub_est_kb=round(ub, 1), compile_ok=ok, err=err)
    results.append(row)
    print(f"[{'OK' if ok else 'FAIL'}] {tag:5s} S{S}/hd{hd} (BS{bs}/BHD{bhd}) "
          f"(est {ub:.0f}KB/{ub/chip:.0%} of {chip}KB)", flush=True)
    if not ok:
        print("      err:", err.replace("\n", " | ")[:300], flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compile_gate.json"), "w") as f:
    json.dump(results, f, indent=1)
n_ok = sum(r["compile_ok"] for r in results)
print(f"==== compile gate: {n_ok}/{len(results)} configs compiled OK ====")
sys.exit(0 if n_ok == len(results) else 1)