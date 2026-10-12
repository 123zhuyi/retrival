# P4 验证报告 — gelu（低价值降档：单组冒烟）

> 2026-09-17｜策略依据：manifest「低价值」判定 → P4 降为单组冒烟（README P4/P2.5 联动条款），验收标准为「无负引导」而非增益。
> 环境：torch 2.10.0 + torch_npu（Device 0），triton-ascend JIT 真编译真执行。

## 冒烟结果（PASS 3/3）

| dtype | shape | 值域 | max_err vs golden | 容差 | 结论 |
|---|---|---|---|---|---|
| fp32 | 257×1000 | N(0,3²) clamp[-7,7] | 4.768e-07 | 2e-3 | ✅ |
| fp16 | 同上 | 同上 | 1.221e-04 | 2e-2 | ✅ |
| bf16 | 同上 | 同上 | 0.000e+00 | 2e-2 | ✅ |

golden = `torch.nn.GELU(approximate="tanh")` fp32 域（对齐 tests/assets/golden.py）。fp32 误差 < 1e-6 证实：sigmoid 形态 + 照抄常数表达式与官方 golden 在 fp32 域内一致；bf16 零误差佐证 RTNE 回写路径同构。

## 冒烟抓到并已修复的缺陷（1 个）

- **JIT 全局变量访问**：骨架初版以普通 Python float 全局量引用两常数，Triton JIT 报 `NameError: Cannot access global variable ... from within @jit'ed function`——需 `tl.constexpr` 注解。已修复（triton_skeleton.py:29-30），修复后三型全过。
- 流程启示：此类错误属「骨架文本合法但 JIT 语义违禁」，静态 preflight 扫不出——低价值降档保留单组冒烟是对的（编译门缺失时的最小兜底）。已补充：若同类卡片骨架引用模块级常数，一律 `tl.constexpr` 注解或内联字面量。

## 结论

卡片「无负引导」验收通过：骨架一次编译通过（修复 constexpr 后）、三 dtype 精度达标、无运行时错误。低价值预判成立（正确性增益≈0，对照难度低）。卡片 confidence 维持 inferred 不升级——按纪律，单组冒烟不足以支撑 measured 评级，但 §11 复现性声明中「sigmoid 形 inf 支路未实测」之外的精度路径已获冒烟佐证。

未覆盖（按降档策略显式放弃）：inf 值域 case、多 shape 性能点、BLOCK autotune 优选、P2.5 各门（skipped: low-value）。
