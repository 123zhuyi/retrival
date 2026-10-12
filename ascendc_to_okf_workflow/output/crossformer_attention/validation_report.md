# P2.5 骨架实测报告 — CrossformerAttention

> 证据标记 **[EXP-skeleton]**；芯片 Ascend910（NPU0）+ CANN 9.1.0 + torch_npu 2.10.0.dev；C1=Cube。
> 本算子 P0 判「非低价值」，按 P2.5 降级档只跑 **①编译门 + ②正确性门**（smoke 随骨架自带），
> ③simulator / ③.5 board / ④IR 取证门**未跑**——BLOCK 选型与瓶颈定型论断保持两态不写实测结论。

## ① 编译门（BLOCK 配置网格编译 + 最小运行）

| shape | dtype | BLOCK_M/BN/BD | 编译 | UB | 备注 |
|---|---|---|---|---|---|
| single window G=196 | fp16 | 64/128/128 | ✅ | 未 overflow | 基线 |
| feature 8×14 G=112 | fp32 | 64/128/128 | ✅ | 未 overflow | ieee dot 路径 |
| single G=729 | fp16 | 64/128/128 | ✅ | 未 overflow | +mask |
| feature 11×7 G=77 | bf16 | 64/128/128 | ✅ | 未 overflow | |
| **big single G=7921** | fp16 | 64/128/128 | ✅ | 未 overflow | **flash 多块** |
| feature 16×14 G=224 | fp16 | 64/128/128 | ✅ | 未 overflow | +mask+pos |
| small single G=49 | fp16 | 64/128/128 | ✅ | 未 overflow | G<BLOCK_N 尾块 |

编译门 7 组全过；UB 估算 §6 ≈51%，包络内。*单点实测值仅 Ascend910 上下文，换平台须重探。*

## ② 正确性门（对 torch golden 全开关组合）

| shape 组 | 对 golden rmse | 判定 | 门覆盖的路径 |
|---|---|---|---|
| n1f G=196 fp16 单窗 | 1.7e-5 | PASS | 基础 flash |
| n2f fp32 +pos_bias+mask | 3.1e-8 | PASS | **fp32 ieee + 加性 bias** |
| n3f G=729 fp16 +mask | 1.9e-5 | PASS | 全序列掩码 |
| n5f bf16 feature | 4.6e-6 | PASS | **bf16 + 多窗口分组** |
| n13f **G=7921** fp16 | 1.3e-4 | PASS | **大窗 flash（key 多块）** |
| n17f fp16 +mask+pos feature | 6.4e-6 | PASS | 组合 |
| n49f fp16 小窗 | 1.8e-6 | PASS | G<BLOCK_N |

**全 7/7 PASS**。覆盖 f16/bf16/fp32、单窗/feature 多窗、mask/pos_bias、G<BLOCK_N 与大 G flash。

## ②门调试中抓出的 bug（均已修复，回流成契约）

1. f16/bf16 `tl.dot` 写 `input_precision="tf32"` 被 Ascend 拒（须 hf32/ieee）→ **新增 C-029**。
2. `p@V` 的 V 误按 Kᵀ 的 `(D,N)` 方向加载 → 静默错值（rmse~1.0）→ 改正为 `(BLOCK_N,BLOCK_D)` → **新增 C-030**。
3. flash 在线 softmax 的 key 尾列（G<BLOCK_N）未置 `-inf` → softmax 分母虚增、输出被稀释 → `-inf` 掩码 → **新增 C-031**。
4. feature 输出漏逆排列（kernel 输出 grouped 布局 vs golden 原始布局）→ wrapper `o[_inv]`。
5. golden 非 feature 分支多 unsqueeze 一次（4D → 3D）→ 修正 golden。

## P4 状态

**P4 双盲回放验证【未执行】**。本报告仅覆盖 P2.5 ①②门（提取期实测）。P4 需独立会话跑对照组（仅算子名）
vs 实验组（仅本卡）+ 性能测量 + 归因分析，成本高且随机，作为后续动作。frontmatter `confidence` 保持
`inferred`；①②门证据已让 §6 预算、C-029~031 上升 `measured(skeleton)` 用法级结论。

## 证据纪律说明

- 所有数值为 **Ascend910 单机单点 [EXP-skeleton]**，换平台/版本须重探（C-012 上下文只有两态不升格禁令）。
- 新增契约 C-029/C-030/C-031 已当次回流 `backend_facts/backend_contracts.md`（README v1.4 第 5 条），
  含 preflight #20 行与维护记录；mapping §P2.5 与卡 §8/§10 双向同步。