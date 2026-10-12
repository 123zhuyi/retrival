# P4 验证报告 — layer_norm（layer_norm_v4）

> 2026-09-18｜环境：torch 2.10.0+torch_npu，Ascend910_9372，CANN 9.1.0，NPU Device 0（红线：TORCH_DEVICE_BACKEND_AUTOLOAD=0 + ASCEND_RT_VISIBLE_DEVICES=0）
> 判定回顾：非低价值（Norm 家族首卡、❌≥3：Welford 向量化内置/二分累加指令/融合指令与多变体分派、属性语义面丰富、rsqrt 后端未取证）→ P2.5 降级执行①编译门+②正确性门，③③.5④ 跳过，BLOCK 选型/瓶颈定型两态。

## ① 编译门记录（2 轮 + 探针，抓获 1 条新后端事实）

| 轮 | 现象 | 结论与处置 |
|---|---|---|
| 1 | f32 R=1000 / f32 R=700 **错值（FAIL）**，其余 PASS | 定位：非 2 幂 R 尾 tile 掩码元素 `xv=0`（other=0），pass2 方差累积 `d=xv−mean=−mean → d²=mean²` 被误加，var 高估 pad·mean²/R；mean/pass1（other=0 正确）与归一化不受影响。**骨架编码 bug（非后端契约）**，②门抓获修复 `d=tl.where(msk, xv−mean, 0.0)`（通用防坑：被掩码的「归约后置减」运算须显式归零） |
| 2 | 全绿 16/16 | — |
| 探针 | `tl.rsqrt` 单独编译+执行 | **OK**（910_9372+CANN 9.1.0）：对照 `1/torch.sqrt` maxerr 1.19e-7 正确 → 补证 lowering 通路 [EXP-skeleton]，**已回流 tl_op_lowering.md**（rsqrt 行 + 维护记录 2026-09-18 + updated_at）。主路径 layer_norm 用 `1/tl.sqrt`（T1 [SRC] 更保守），rsqrt 为经证备选 |

附带确认：`tl.sqrt` 逐 op 在本骨架内（var+eps 标量→向量化算子路径）编译执行正常——与 sum 卡 tl.sum（T2）/算术（T1）一致，不新增契约条目。

## ② 正确性门记录

- 覆盖：3 dtype（f32/f16/bf16）× gamma/beta 四组合 × R{32,64,128,700,1000,2048} × 尾部多维 normalized_shape(16,32)/(2,4,8) × 恒定行 × 大数值 1e6 × 混合精度（f16 x+f32 gamma/beta）× return_mean_rstd 两态 + 单独 y 对照 = **16 check**（y 对 `torch.nn.functional.layer_norm`、mean/rstd 对 fp64 手算参照）。
- 基准：fp64 参照（防抵消，呼应 Welford 设计文档 §3 精度动机）+ torch F.layer_norm；容差 f32 1e-4 / f16 1e-3 / bf16 1e-2；mean/rstd（f32）rtol 1e-4。
- 结果：**16/16 PASS**（日志 /tmp/ln_smoke1.log）。

## preflight 执行（契约清单 #1~#17 全扫）

- **#16（0-d 标量算术）**：kernel 内 `mean=tl.sum(s,0)/R`、`rstd=1.0/tl.sqrt(var+eps)` 均为**0-d 标量算术**——本栈编译+执行通过，未触发 C-025 的 materialization 失败（与 histc 卡 C-025 场景差异：histc 是「自构造 0-d 运算」，本骨架为归约结果标量→算术→store，路径不同），不命中（登记：归约结果标量算术可用）。
- #17（标量行循环+整型累加）：本算子全 fp32 域，非 i32；pass 循环用 1D tile 块状 load（非逐迭代换 base 行 load），结构上规避 C-026 同族风险，不命中。
- #1（%）不命中（无 % 解码）；#4（int64 加法链）不命中（fp32 域）；#15（scatter atomic）不命中（行主语独占写，无 atomic）。
- 其余 #2~#3、#5~#14 不命中（无 autotune/scan/gather/降序；stride 运行传参；HAS_*/NEED_* constexpr 分支）。

## 质量门结论

- [x] ①门新事实按 v1.4 第 5 条当次回流：`tl.rsqrt` 入 tl_op_lowering.md（T1 行 + 维护记录）——[EXP-skeleton]+芯片/CANN 上下文已注
- [x] ②门 16/16；卡片 §8 实测摘要与本文一致；双跑按位相等（y/mean/rstd 三输出 bitwise_equal=True，确定性确认）
- [x] ❌ 项（UB 复用/FusedMulDstAdd/行配对/多变体分派体系/半壳库 API）全部有「机制不可译+降级论证」（卡片 §6）；Welford→two-pass 精度同构论证（tl.sum 树形=二分累加 O(log n)）
- [x] 掩码尾 tile trap 作为通用防坑登记卡片 §4（②门实测抓出，非后端契约不回流）
- [x] BLOCK 选型两态、confidence=inferred（降级纪律未破）
- **P4 通过，卡片可入 P5 待审队列。**
