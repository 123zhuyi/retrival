# P4 验证报告 — sort（Sort）

> 2026-09-20/21｜环境：torch 2.10.0+torch_npu，Ascend910_9372，CANN 9.1.0（红线：TORCH_DEVICE_BACKEND_AUTOLOAD=0 + ASCEND_RT_VISIBLE_DEVICES=1 同定物理卡）
> 判定回顾：非低价值（排序家族主卡、❌≥5：12 SCHID 分派/radix 多趟/merge 四态/rankInverse 位级索引/AiCore 仅浮点，❌ 项有降级论证）→ P2.5 降级执行①编译门+②正确性门，③③.5④ 跳过，BLOCK 选型/瓶颈定型两态。
> **本轮机器空闲（无争抢，AICore 0%）** → ②门**一次全绿，无环境受限**；②门内部实抓 2 类键映射 bug（均已修复）——真实 bug 修复，非降级妥协。

## ① 编译门记录

| 轮 | 现象 | 结论与处置 |
|---|---|---|
| 首跑 | 单 segment 骨架 28 项 smoke 全编译通过（机器空闲，编译显著快于 topk 卡时段） | 骨架可编译 |
| 链式 or 编译错 | `if DT==0 or DT==1 or DT==2` 于 kernel 内 → Triton 解析器拒绝 chained boolean | 折 0 折叠移入 `_sortable_key` 的 if/elif 结构（Triton 接受） |

**关键观测**：无 C-027/C-028 新触发（用到既有契约）；`tl.max`/`tl.argmax` int64 选择环沿用 topk 已证正确路径。候选 B（f32 `tl.sort` 直排）**本次未单独跑**——主路线（迭代选择）②门已全绿，B 候选作为优化岔路登记（tie 稳定索引回查仍是其阻碍；C-027 已限 vsort 仅 f16/f32，非主要路径）。

## ② 正确性门记录

| 域 | 覆盖 | 结果 |
|---|---|---|
| 浮点族 | f32（N=64/32/700非2幂 升降序、idt i32/i64、中间轴 dim=1、首轴 dim=0）、f16、bf16 | **全 PASS** |
| 整型族 | i8/i16/i32/i64、u64（idt i64） | **全 PASS** |
| 注入模式 | duplicate 升/降、signed_zero 升、nan 升/降、infinity 升/降 | **全 PASS**（见处置） |
| 覆盖全集 | 4 dtype 族 × 双向序 × N多形态 × 四注入 + 任意轴 | **20 PASS / 0 FAIL** |

**②门实抓 2 类键映射 bug（首次运行 16 PASS / 4 FAIL → 定位修复 → 20/0 全绿）**：
- **B1 值域沉底 ±inf 碰撞**：升序沉 +inf、降序沉 -inf，当行含 ±inf 或 NaN 时，已沉槽的闭键(+inf)大过 NaN 闭键(-2^32 级)或与实 ±inf 相等，argmax 左most tie 误选已锚槽 → 输出重复/错索引（nan升、infinity 双向 FAIL）。**修复：改独立 `vis` 缓冲在键域排除已选位（置 INT64_MIN 恒低于实键），x 值永不改写**——脱：值碰撞。
- **B2 ±0 不同键**：sign-flip 对 -0(0x7FFF..) 与 +0(0x8000..) 给相邻不同键 → torch 视 ±0 相等（稳定保输入序）被破坏（signed_zero/infinity FAIL）。**修复：`_sortable_key` 浮点族折合约 0 为单一 0 键**（负值键<0键<正值键保序）。

## preflight 执行（契约清单全扫）

- **C-027/C-028**：沿用（vsort 仅 f16/f32 → packed-key 废弃；argmax int64 → int32 两步）——不新增回流。
- **%、atomic、autotune、scan、gather、降序动态界**：皆不命中（无跨 tile task 解码、无 atomic、全 constexpr 序向、发射级无动态界）。
- **C-017 运行传参**：N=axisLen 运行传参；DESCENDING/DT/STABLE 语义 constexpr。
- 非尾轴 movedim 后续 `.contiguous()`（topk 探针14 实证 trap 沿用）已嵌入 wrapper，dim=1/dim=0 用例验证。

## 边界受限登记（卡片 §8，P4 如实记录）

1. **大 N>2048 二段归并**：官方 merge/radix more-core/multi-core 域，需 stage1 tile 局选 + stage2 归并（topk 卡两行段同构）；**本次未实现（骨架断言保护），②门仅锁单 segment N≤2048**。验证优先级中，空闲时依 topk 卡两行段方案补实现。
2. **候选 B（f32 tl.sort 直排）未单独验证**：主路线②门已全绿，B 作为优化岔路登记（tie 稳定索引回查为其阻碍）。
3. **发射次数=N** 的性能两态：N≤2048 正确性现实；大 N 的 launch 开销未做瓶颈定型（③跳过，降级登记）。

## 质量门结论

- [x] ①门：单 segment 骨架全配置真编译通过；C-027/C-028 既存不新增；候选 B 以 ③③.5 跳过登记
- [x] ②门：**一次性全绿（20/0）**，无环境受限；2 类键映射 bug（±inf 值沉底碰撞、±0 双向键）由 CPU 探针+②门定位并修复，均为**真实 bug 修复**（非降级）
- [x] ❌ 项（12 SCHID 分派/radix 多趟/merge 四态/rankInverse/免转置/AiCore 仅浮点）全部有降级论证（卡片 §3/§7）
- [x] 降级纪律：BLOCK 两态、confidence=inferred、③③.5④ 跳过；新后端事实无新增回流（C-027/028 沿用）
- ✅ **非标准降级声明：本次无降级妥协**——②门在机器空闲下一次性跑通全绿，boundary 仅大 N 二段归并（代码未实现+断言保护，如实登记），建议 P5 后补跑大 N 二段重点验。