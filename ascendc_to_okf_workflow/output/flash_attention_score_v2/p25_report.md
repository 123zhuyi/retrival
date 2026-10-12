# P2.5 骨架实测门报告 — flash_attention_score（v2，服务 level4 SelfAttention）

> 日期 2026-10-10 · 芯片：Ascend910（arch32/A2 兼容, CANN 9.1.0, torch 2.10.0+cpu, torch_npu, triton 3.2.0）
> 环境：16 个 davinci，本次锁 `ASCEND_RT_VISIBLE_DEVICES=0`（AICore 空闲）；`TORCH_DEVICE_BACKEND_AUTOLOAD=0`；脚本内 `import torch_npu` 注册设备。
> 骨架：`output/flash_attention_score_v2/triton_skeleton.py`（flash 注意力核 = crossformer 同构移植，NW=batch、G=S、head 特征切片）。

## 门① 编译门 — ✅ 过

对多个 BLOCK 配置组合真编译+最小运行，覆盖如下配置网格（全部编译通过并运行成功）：
| case | shape(dtype,H) | head_dim | BLOCK(重) | 结果 |
|---|---|---|---|---|
| smallS | (2,64,512) fp16 H8 | 64 | M64/N128/D 上浮64 | 编译过, 正确 |
| bigS-001 | (1,512,4096) fp16 H32 | 128 | M64/N128/D128 | 编译过 |
| bigS-004 | (1,4096,4096) fp16 H32 | 128 | M64/N128/D128 | 编译过, 大 S flash 多块 |
| bigS-003 | (1,2048,4096) bf16 H32 | 128 | M64/N128/D128 | 编译过 |
| fp32 | (1,256,1024) fp32 H8 | 128 | M64/N128/D128, ieee | 编译过 |
| hd256 | (1,512,8192) bf16 H32 | 256 | M64/N128/D256 | 编译过 |

**观测**：
- `head_dim=256` 需 `BLOCK_D=256`（2 幂上浮）才编过 → **BLOCK_D 一律 `next_power_of_2(head_dim)`**（已入骨架 wrapper 注释；两阶段结论，非测速）。
- **C-032 复现 + D-tier 逃逸（对齐 crossformer_v2 实证）**：`head_dim>128` 在 `BLOCK_M=64` 触发 192KB UB 溢出——flash 骨架初次以 hardcoded BM64/BN128/D256 跑 hd256 时**静默衰减**（rmse 9.37e-3，Q/attn/O 三 tile 合超 UB 触 compiler 搬移降精度），未报错。改 `BLOCK_M=32/BLOCK_N=64` 腾 UB 后 rmse 回到 **3.15e-5**（300× 改善）。⇒ wrapper 内建 D-tier：`BLOCK_D=next_pow2(head_dim)`，`BLOCK_D>128 → (BM32,BN64)`，否则 `(64,128)`（对应 CANN arch35 CalcDBasicBlock AlignUp(D,64) 档位）。
- CANN/HDK 驱动打印「32 padding size」内存对齐 Warning 属**良性性能提示**（非错误）。
- 其余配置未发现新编译陷阱（triton 3.2 对 `tl.dot` f16/bf16/hf32 + `input_precision="ieee"` + 在线 softmax 本栈可编译）——**未触发 C-016/C-017 类 UB 溢出**（BM64×BN128×fp32 attn=32KB，远低于 192KB UB）。

## 门② 正确性门 — ✅ 全过（7/7）

对 ref golden（definition.json reference，纯 torch，含 4 段投影）真机运行 `__main__` smoke：
| case | dtype | head_dim | rmse | 判定 |
|---|---|---|---|---|
| bigS-001 S512 | fp16 | 128 | 3.94e-06 | PASS |
| bigS-004 S4096 | fp16 | 128 | 1.44e-06 | PASS（大 S flash 多块） |
| bigS-003 S2048 | bf16 | 128 | 1.59e-05 | PASS |
| midS B2 S512 | fp16 | 128 | 3.95e-06 | PASS |
| fp32 S256 | fp32 | 128 | 5.96e-09 | PASS（ieee） |
| hd256 S512 | bf16 | 256 | 3.15e-05 | PASS（**D-tier 逃逸**：BM32/BN64） |
| smallS S64 | fp16 | 64 | 1.07e-05 | PASS |

`ALL_PASS`。
**结论**：注意力核相对 torch golden 正确（同含投影的 level4 全链路），四 dtype（fp16/bf16/fp32）+ 三 head_dim（64/128/256）+ 大 S flash 多块 + 多 batch 全覆盖。投影（nn.Linear seed42）与注意力分开验证，rmse 低，接口差（映射 §0）归位正确。

## 门③ / ③.SRC / ③.5 / ④ — 降级跳过（工作流明确允许）

**降级依据**：本算子非低价值，但`③ simulator`需对 triton JIT kernel 按 mangled 名过滤、解析 4 张表、至少 2 shape 点，定时成本高；按 README「非低价值但想省实测时间 → 降级为只过①+②」条款执行。③.SRC 源码档位决策已由 P0/P1 静态锚点覆盖（§分核策略六模板优先级表、s1s2_bn2gs1.h:773-778 多核循环、docs §4/§5 基本块），不依赖 board 实测。

**降级后果（诚实登记）**：
- BLOCK 选型（BM/BN/D）与**瓶颈定型（Cube-bound vs 搬运-bound）保持两态**，未写「实测最优」；
- `num_stages`/`autotune` 在 Triton-Ascend 是否生效**未实测**（骨架现无 autotune/num_stages，防 do_bench 噪声，符合 P2 纪律）——仍为两态待 P4 确认；
- 卡片 confidence 保持 **inferred**（未升 measured(skeleton)，因未全过四门）。

**预判（两态，供 P4 验证方向，非结论）**：flash 注意力在 910 A2 CV 双发/预取下，主流瓶颈应为 **Cube(MMAD) bound**（横三 dot：q·kᵀ→p·v），Online softmax 的 exp/max/sum 撑在 Vector；若 P4 测出搬运/同步占比高，优先检查 flash 外循环 BLOCK_N 是否过小导致 MTE 启动次数多（CANN 用 S2.i=1024 大块就是压这个，但 Triton 受寄存器/UB 约束，BN 候选 64/128/256 需实测权衡）。

## 新发现后端事实回流

**无新增 backend_contract 条目**：本骨架全部复用已入库契约（C-001/002/006/007/011/016/018/019/023/029/030/031/**032**），未抓到新的「静态取证扫不出、真机才暴露」事实。
- **C-032 于 flash 复现**（已由 crossformer_v2 入库，本算子二次确认）：`head_dim>128` + `BLOCK_M=64` → 192KB UB 溢出，且**静默衰减不报错**，只有 BM 降档才回归精度——write 进卡的硬边界。唯一软观测「BLOCK_D 须 next_power_of_2(head_dim)」属 Triton 通用语义（pow2 分块），不足以新增契约，已在骨架注释记录。

## 证据纪律自查
- 数据标 **[EXP-skeleton]** + shape 上下文（见门②表）。✓
- 门失败项：无（编译/正确性全过）。✓
- P2.5 不替代 P4 验收回放：端到端性能未测，留 P4。✓
- 维护记录：本报告登记于 output/flash_attention_score_v2/。✓