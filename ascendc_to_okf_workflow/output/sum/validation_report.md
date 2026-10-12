# P4 验证报告 — sum（reduce_sum）

> 2026-09-18｜环境：torch 2.10.0+torch_npu，Ascend910_9372，CANN 9.1.0，NPU Device 0（红线：TORCH_DEVICE_BACKEND_AUTOLOAD=0 + ASCEND_RT_VISIBLE_DEVICES=0）
> 判定回顾：非低价值（规约家族首卡、❌≥2：跨核两阶段归约/NDDMA、属性语义面、i64 规约后端未取证）→ P2.5 降级①编译门+②正确性门，③③.5④ 跳过，BLOCK 选型/瓶颈论断两态，confidence=inferred。

## ① 编译门记录（2 轮 + 探针，抓获 1 条新后端事实）

| 轮 | 现象 | 结论与处置 |
|---|---|---|
| 1 | 2 FAIL：① f32 尾轴 (2048,4096)——纯精度（max abs 4.6e-5，无错值），累加序噪声，atol 1e-5 过紧；② **i32 中间轴 (8,64,128) 错值 1021/1024** | ① 容差修正（f32 atol 1e-4，对齐 ST 0.05/1e-4 档）；② 立案探针 |
| 探针 | i32 中间轴错值定位：N=128/256 失败、N=64/96/100 正常（**N≥BLOCK_N 即 mask 全真时触发**）；**同输入两次运行结果不同（非确定）**；**grid=1 单 program 复现**；f16/i64 同构形态正常；no-op `.to(int32)` 不解；错值无部分和/列移位规律 | **标量 r 循环「逐迭代换 base 的行 load + i32 向量累加」非确定错值**——排除核间竞争，指向循环流水 load→累加链同步插入不足（C-021 同族，机制未 IR 取证）→ **路线 M 改 2D tile (BLOCK_R=8, BLOCK_N=128) 块状加载**（尾轴 kernel 同构），回流 **C-026** + preflight #17 |
| 2 | 全绿 | — |

附带确认：i64 规约（vreduce i64，tail+mid 两路线）编译/正确性通过——与 dtype_support.md「A2/A3 i64 vreduce 标量化」[SRC] 一致（性能标量化、正确性不受限），不新增契约条目，维护记录登记。

## ② 正确性门记录

- 覆盖：6 dtype（f32/f16/bf16/i32/i64/bool）× {尾轴/首轴/中间轴/相邻多轴/非相邻多轴（permute 归一化路径）/负轴/全轴/空 axes 两态/keep_dims/大 R=100000/1D} = **23 check**。
- 基准：官方 golden.py 同构（f16/bf16 升 fp32 回原型；bool→i64；i32 按 np 默认升 i64 再回）；容差：f32 rtol/atol 1e-4、f16 1e-3、bf16 1e-2，整型精确相等。
- 结果：**23/23 PASS**（日志 /tmp/sum_smoke2.log）。

## preflight 执行（契约清单 #1~#17 全扫）

- #17（标量行循环+整型累加）命中初版路线 M——已改 2D tile 结构，终版不命中。
- #1（%）不命中（task 解码用 `//` + `a-(a//b)*b`）。
- #4（int64 加法链）命中 ACC=2 的 i64 累加——**登记豁免**：bool/i64 语义必需（def 表 y=i64），标量化预警已两态登记（§4/卡片）。
- #15（scatter atomic）不命中——本卡**禁用原子归约写法**（确定性声明 + C-024）。
- 其余 #2~#3、#5~#14、#16 不命中（无 autotune/scan/gather/降序/0-d 算术；stride 运行传参；ACC constexpr 分支）。

## 质量门结论

- [x] ①门新事实按 v1.4 第 5 条当次回流：C-026 入 backend_contracts.md（A 节）+ preflight #17 + 维护记录（含 i64 vreduce 一致性确认登记）
- [x] ②门 23/23；卡片 §8 实测摘要与本文一致
- [x] ❌ 项（跨核两阶段/NDDMA/ScheduleMode）全部有「结构性不可译+降级机制」论证（卡片 §6）；半壳缺省项（ReduceSch/GetPromoteType/常量表）标注机制推断
- [x] 源内差异登记：kernel BoolDag int64 域 vs aclnn 2²⁴ FLOAT 限流；GEIR store-cast vs aclnn cast-first；i32 promote 缺省
- [x] BLOCK 选型两态、confidence=inferred（降级纪律未破）
- **P4 通过，卡片可入 P5 待审队列。**
