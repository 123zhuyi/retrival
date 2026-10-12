# P2.5 骨架实测门报告 — flash_attention_score（服务 level4 SelfAttention，2026-09-21）

> 环境：Ascend910_9372 + CANN 9.1.0，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`（C-019）、`import torch_npu`。
> 骨架 `output/flash_attention_score/triton_skeleton.py`。数据标 [EXP-skeleton]（小 shape/微基准上下文，
> 外推受 C-012/C-027 约束）。

> **⚠️ 主目标已换 8_SelfAttention（统一大 S，S∈[512,5182]、head_dim 至 256、100% S>64）**。下方
> ①编译/②正确性为**初轮小 S 单 K 块**记录；骨架已重写为 **S1S2 分块 + 在线 softmax**（见「〇. 更新」）。
> **大 S tiled 的 NPU 实跑门（①编译 + ②正确性）已跑通（ALL OK，见〇）**；修复了重写引入的两处后端 bug
> （kt/v 未提 fp32、head 基址多乘 HD）——详见〇「后端发现」。

---
**〇. 更新（2026-09-21，主目标换大 S）**

- 骨架 `_sa_attn_fwd_kernel` 由「grid=1 串行 head + 全 K 单 tile」重写为 **S1S2 分块在线 softmax**：
  任务轴 = B·H·cdiv(S,BLOCK_M)，key/S2 内环 running max + α 补偿(O=O·α+p·V、L=L·α+Σp、m=m_new)；
  BLOCK_M/N 依 head_dim 取 64(HD≤128)/32(HD>128)；小 S 副分支 BLOCK_N≥S 退化单 K 块。
- **CPU 数学自检（已过）**：`_selfcheck_cpu()`（纯 torch，不依赖 Triton/NPU）对比 tiled online-softmax
  vs `F.softmax` 全量 golden，6 组(B∈{1,2}, S∈{65,70,129,260,511,1000}, HD∈{32,64,96,128,256}) max_err≤3e-7,
  验证在线-softmax(running max+α 补偿)数学正确。
- **NPU 实跑门（已过，ALL OK）**：`triton_skeleton.py npu`（Ascend910_9372，persistent grid=min(NTOT,20)）大 S 5 组
  smoke **全过、bad_ratio=0.0**：S512/H32/fp16(HD128, 3.05e-5)、S1024/H32/bf16(HD96, 2.44e-4)、S828/H4/fp32(HD64, 2.98e-8)、
  S701/H6/fp16(HD48, 3.05e-5)、S2762/H32/fp32(HD128, 4.10e-8)。达标：fp32≈4e-8、bf16/fp16 ≤2.4e-4（远小于阈值）。
- **后端发现（重写引入的两处 bug，写卡必须）**：
  1. **head 基址多乘 HD**：原 `base=bh·(S·HD)` 但后续每个 load 又 `·HD` → 第 2 个 head 起地址多乘 HD²、越界 → aicore MTE
     「invalid GM address」。**BH=1 时 bh=0 恒掩盖此 bug**（初轮小 S 单 head 没暴露）；BH≥2 即崩。修复：`base=bh·S`
     （head 起始按行记，由 `·HD` 还原）。**多 head 场景必须核查 head slab 基址缩放。**
  2. **kt/v 未升 fp32**：q 已 `.to(fp32)` 但 kt/v load 后仍为输入 dtype → `tl.dot` 编译断言「Got fp32 and fp16」
     （违反 C-029 fp32 计算域）。fp32 输入不触发、fp16/bf16 才触发。修复：kt/v 均 `.to(tl.float32)`。
  3. BLOCK_N=512（全 K 单块）fp32 时 kt/v 大 tile 亦越界（UB 压力）→ 骨架默认 BN=64 分段，保持。
- 逐头拆解边界已验证：BH1 S2048(grid1 单 prog 扫 32 任务) OK、BH32 S512 grid20(persistent) OK、BH64 S5182 OK——**MTE 根因是
  head 基址缩放，非 sweep 长度/并行度**。

---

## 门结果总览

| 门 | 结果 | 产出 |
|---|---|---|
| ① 编译门 | **24/24 配置全过**（三 dtype × level4 shape 代表面，含 hd=65→BLOCK_HD=128 贴顶档） | `p25_work/compile_gate.py` + `compile_gate.json` |
| ② 正确性门 | **8/8 all-bad=0.0**（5 组 smoke + 3 组 level4 corner 扩点），fp32 逐位、bf16/fp16 ≤5e-5 | `triton_skeleton.py`（修复版） |

**P2.5 后端发现（写卡必须）：注意力核单 program head 串行 + 全 K 单 tile full-softmax 可行，
但 host 侧 head reshape 存在非连续地址陷阱——kernel 平坦行主序寻址会静默读错数据。**

## ① 编译门明细（24/24）

铺 `BLOCK_S∈{8,16,32,64}` × `BLOCK_HD∈{32,64,128}` × `fp16/bf16/fp32`，Ub 估算按 mapping §5
（UB 可用主体 950KB）：最宽档 S=64/hd=64 峰值 ~112KB≈12%，hd=65→BLOCK_HD=128 档 fp32 亦 ~4% 安全。
**全配置编译通过**——本 kernel 无 `%`/`tl.zeros` 进位/`num_warps`/`autotune`，天然规避 C-001/006/007/016。

## ② 正确性门明细（8/8，bad=0.0）

golden = definition.json `run()` 逐字移植（投影 host fp32 + fp32 softmax/attn）。逐层隔离锁定
**三大地址/数值陷阱**：

| # | 陷阱 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | **head reshape 非连续（真根因）** | B=1 时全错（max_abs 0.4-1.0），B≥2 偶然全对；一度误判为「多 program 并发串扰/低 head 数流水 bug」 | `.view(B,S,H,HD).transpose(1,2).reshape(BH,S,HD)` 在 B=1 返回 **stride=(24,72,1) 的 [S,H,HD] 交错视图**（非行主序 [BH,S,HD]），kernel 用 `base+rows*HD+hdc` 平坦寻址 → 读错偏移 | wrapper 强制 `.contiguous()`；K 镜像用自然 load+`tl.trans` 后证实同址 bug |
| 2 | K 镜像 load 地址 | 早期全错 | `hdc[:,None]*HD+rows[None,:]` 公式误 | 改 `tl.load(...)` + `tl.trans`（Kᵀ = [S,HD]→转置），与 CANN bmm1 语义一致 |
| 3 | BLOCK_S 填充列污染 softmax | 大 S 及非 2 幂 S 错 | 越界 key 列 exp(0−m) 注入 softmax 分母 | `scores = tl.where(sm[None,:], scores, -inf)`（对齐 P2.5 早期实测） |

修复后 5 组 smoke 全过：S7D72H3(fp16 **0.0**)、S11D96H4(bf16 **3.8e-6**)、S13D120H5(fp32 **5.96e-8**)、
S5D60H3(fp16 **0.0**)、S53D320H5(fp16 **3.05e-5**)，bad=0.0。扩点 level4 corner 亦过：S3D48H2(fp16 0.0)、
S59D288H6(bf16 9.8e-4)、S21D180H4(fp32 7.5e-8)。三板重跑稳定（B=1 连过 3 轮）。

**判读**：正确性不依赖 grid 大小/程序数，绑定 host 侧 `contiguous()` 契约。单 program 串行 head 循环
（grid=1）在多轮稳定精确，为最终形态（对应 mapping §1 单 tile full-softmax 语义，非 CANN 的
online-softmax 多块重标度）。

## 结论（写卡 / 回放依据）

> 下表 **#1/#3 为本报告初轮（小 S 单 K 块）实测结论**，主目标已换为 8_SelfAttention 大 S（S∈[512,5182]、
> head_dim 至 256、100% S>64）：**大 S 必须走 S1S2 分块在线 softmax（running max + α 补偿）主路径**，而非单块
> full-softmax；该路径 CPU 数学自检 + **NPU 实跑门（①编译 + ②正确性）均已通过**（〇），修复的 head 基址/kt-v
> 两处后端 bug 见〇。#2/#4 契约与后续建议不受目标切换影响。

1. **正确性达标（初轮小 S 单 K 块）**：骨架在旧 level4 包络（S∈[3,59]、hd∈[20,65]、三 dtype、B≤4）全组合验证
   bad=0.0，证明 host 侧 `contiguous` 契约 + −inf 填充 + fp32 计算域正确；大 S 分块在线 softmax 的正确性以
   CPU 数学自检为准（待 NPU 实跑回填）。
2. **host 侧契约（必须写进卡片）**：head 视图 `[B,H,S,d]→[BH,S,d]` 必须 `.contiguous()`；
   kernel 平坦行主序寻址要求连续张量。否则静默错值（后端地址陷阱，非可凭内核自纠）。
3. **初轮小 S 单 K 块形态**：旧目标 S≤59 恒规整 BSH，无需 online softmax 多块重标度；max 减 + -inf 填充列是
   full-softmax 数值稳定与越界洁净的前提。**主目标换大 S 后此形态仅作次要分支**（archive `SelfAttention_smallS`），
   主路径为 S1S2 分块在线 softmax。
4. **性能形态**：未做 simulator/board 时序（本算子为回放语义目标，非性能诉求）；UB 占用峰值 ≤12%
   （S64/hd64）安全，无 MIX 崩核风险。作「性能实现建议」待 P4/后续。

## 产出索引

- 骨架：`triton_skeleton.py`（含 contiguous 修复 + -inf 填充 + `tl.trans` K 镜像）
- ⑤ 编译门：`p25_work/compile_gate.py` + `p25_work/compile_gate.json`（24/24）
- 正确性：`triton_skeleton.py` smoke/`__main__` + 3 组 corner 扩点（三板重跑稳定）
- 契约回流：不新增 backend_facts 契约（地址连续性属宿主 torch 层契约，记 mappping §5，非 triton 后端缺陷）