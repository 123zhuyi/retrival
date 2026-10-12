# P4 验证报告 — abs（低价值降档：单组冒烟）

> 2026-09-17｜策略：manifest 低价值（实数域）→ P4 单组冒烟，验收标准「无负引导」。环境同 gelu（NPU Device 0，triton-ascend 真编译真执行）。

## 冒烟结果（PASS 4/4，全部零误差）

| dtype | shape | 值域 | max_err vs golden | 结论 |
|---|---|---|---|---|
| fp32 | 257×1000 | N(0,30²) | 0.000e+00 | ✅ |
| fp16 | 同上 | 同上 | 0.000e+00 | ✅ |
| bf16 | 同上 | 同上（升 fp32 路径） | 0.000e+00 | ✅ |
| int32 | 同上 | randint[-100,100] | 0.000e+00 | ✅ |

abs 是精确运算（无舍入链），全零误差符合预期；bf16 升 fp32 路径与 golden 的 float() 标杆逐位一致，佐证分档精度策略转化正确。

## 结论

「无负引导」验收通过：一次编译通过（沿用 gelu 的 tl.constexpr 经验，无新 bug）、四 dtype 全对。低价值预判成立。confidence 维持 inferred。

未覆盖（按降档策略显式放弃）：complex 域（❌ 扩展）、int64 标量化性能、int8/16、多 shape 性能点、P2.5 各门。
