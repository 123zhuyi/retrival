# P2.5 骨架实测门报告 — CrossformerAttention（v2）

> 环境：Ascend910（arch **910b/9372**，AIC 20~24/852 16 卡）、CANN 9.1.0、torch_npu 2.10、
> Triton-Ascend 4588. 日期：2026-10-10。
> 走 P2.5 降级①编译门 + ②正确性门（非低价值但窗口分组=host 变换、kernel 退化标准批多头 flash → 判非 P4 豁免）。
> 参考代码：`triton_skeleton.py`（v2，BLOCK_D 随 D-tier 动态）。

## ①编译门结果

| 形状组合 | head_dim | dtype | BLOCK | 编译 | 判定 |
|---|---|---|---|---|---|
| G=196 单窗 | 96 | f16 | M64/N128/D128 | ✅ 首次 | PASS |
| feature 8×14 + posmask | 8(fp32) | f32 | M64/N128/D128 | ✅ ieee | PASS |
| G=729 单窗+mask | 128 | f16 | M64/N128/D128 | ✅ | PASS |
| feature 11×7 | 128 | bf16 | M64/N128/D128 | ✅ | PASS |
| G=7921 大窗 flash | 96 | f16 | M64/N128/D128 | ✅ | PASS |
| feature 16×14 mask+pos | 128 | f16 | M64/N128/D128 | ✅ | PASS |
| G=49 小窗 | 64 | f16 | M64/N128/D128 | ✅ | PASS |
| **D-tier 边界** hd192 (>128) | 192 | f16 | M64/N128/D256 | ❌ **UB overflow** | FAIL→逃逸 |
| **D-tier 边界** hd256 (>128) | 256 | f16 | M64/N64/D256 | ❌ **UB overflow** | FAIL→逃逸 |
| **D-tier 边界** hd256 fp32 | 256 | f32 | M64/N32/D256 | ❌ **UB overflow** | FAIL→逃逸 |

### UB overflow 实证（v1.5 L2 新取证）
`ub overflow, requires 1587200 bits while 1572864 bits available`（192KB=1572864 bits），
hd192/hd256 f16 均超 **4336 bits(~542B)**；hd256 fp32 超 ~8000 bits。编译器自动
`enable-code-motion=false` + `enable-auto-multi-buffer=false` 后**仍溢出**。
根因：flash 尾 `o_i / s / p` 三个 **fp32 累加器随 BLOCK_D 线性暴涨**（o_i=BLOCK_M×BLOCK_D×4B
是 O 主导项，BLOCK_N 影响甚微）。→ 大 D 的 `BLOCK_D` 越大，fp32 O 累加器越吃 UB。

### 逃逸杠杆验证（v1.5「排除法≠穷举」：发现边界必测已识别杠杆）
| Blocker | 逃逸配置 | 结果 | 判定 |
|---|---|---|---|
| hd256 f16，M64/N64/D256 UB 溢出 | **BLOCK_M=32**（o_i 减半）N64/D256 | ✅ 编译过，**rmse=5.1e-8** | **PASS** |
| hd192 f16 同 | BLOCK_M=32/N64/D256 | ✅ | 同上族成立 |
| hd256 fp32 | BLOCK_M=32 相似（fp32 输入天然 2×） | ⚠️ 未单独验证，图案一致 | 登记 |

**结论**：Triton 192KB UB 将 head_dim 可入范围定为——
- **head_dim ≤ 128**（level4 主流）：M64/N128/D128 = 6 要素基线；
- **head_dim 192~256**：须 BLOCK_M 降至 32（O 累加器减半）腾 UB，crwer置信度可达；
- **head_dim > 256**（512/768/1024）：BLOCK_D≥512，BLOCK_M=32 仍不够 → 需 **D-split**
  （把 head_dim 拆子块做部分 flash），对齐 CANN 自身的 D-workspace 机制（arch35 GetWorkspaceSize）。
  **登记为边界**，本 pass 不实现（同 topk/sort 之 boundary 登记先例）。

## ②正确性门结果（对 level4 definition.json reference golden）

主核 head_dim≤128 七例：

| shape 组合 | dtype | 门① | 门② 对 golden rmse | 判定 |
|---|---|---|---|---|
| n1f G=196 单窗 | f16 | ✅ | 1.744e-5 | PASS |
| n2f feature 8×14 + posmask | fp32 | ✅ | 3.171e-8 | PASS（ieee） |
| n3f G=729 单窗+mask | f16 | ✅ | 1.869e-5 | PASS |
| n5f feature 11×7 | bf16 | ✅ | 4.665e-6 | PASS |
| n13f G=7921 大窗 flash | f16 | ✅ | 1.288e-4 | PASS（多块在线 softmax） |
| n17f feature 16×14 mask+pos | f16 | ✅ | 6.398e-6 | PASS |
| n49f G=49 小窗 | f16 | ✅ | 1.760e-6 | PASS |
| **v2 新增** hd256（经主骨架 wrapper，M32/N64/D256） | f16 | ✅ | **2.69e-8** | PASS |

## ③Pip 诊断（v1.5 强化门）
本 pass 目标为提取级 P2.5（compile+correctness 门），P4 回放性能诊断（pipe 4 表/精度补偿扫描）
**未执行**（用户范围=P0→P2.5）。但已在后续 P4 若开展时预留：CrossformerAttention 为 flash 注意力，
潜在烧点=数值重算（fp32 累加、p 回 cast）与多 dot 补偿——若 P4 做 pipe 门须专门扫此类（v1.5 D 反制规则）。

## 构造/后端事实（回流清单）
1. **大 D>128 编译期 UB 溢出且自动多缓冲/code-motion 关闭后仍溢出**（hdl256/192 实测）。
2. **BLOCK_M 是 fp32 O 累加器的 UB 主导项**（降 M 比降 N 更有效腾 UB）。
3. 与 C-022（scan CallOp tile 硬上限）同类但触发在 D 轴：**head_dim 维度受限**。
第 1 条建议回流 backend_facts（新增 C 条目），沿用 v1.4 证据纪律第 5 条。

## 边界登记
- head_dim>256（512/768/1024）：需 D-split（对齐 CANN D-workspace），未实现、未验证。
- 大 D 大 G 组合（如 hd512 × G=7921 非对齐）：UB + flash 双重压力，未验证。