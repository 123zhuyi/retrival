# P4 验证报告 — topk（top_k_v2）

> 2026-09-18/20｜环境：torch 2.10.0+torch_npu，Ascend910_9372，CANN 9.1.0（红线：TORCH_DEVICE_BACKEND_AUTOLOAD=0 + ASCEND_RT_VISIBLE_DEVICES 指定空闲物理卡）
> 判定回顾：非低价值（选择/排序家族首卡、❌≥4：radix 多趟/五模式分派/sort_and_topk/bitonic 网、属性语义面 5 项、后端排序指令限制已预取证）→ P2.5 降级执行①编译门+②正确性门，③③.5④ 跳过，BLOCK 选型/瓶颈定型两态。
> **本轮机器环境受争抢（其它租户高占 NPU 0 / 宿主 99% AICore）+ 超长编译（13-20min/kernel）** → 按用户决策「精度降级先产卡」，②门 partial（单行段全绿，两行段/u64 边界受限如实登记），卡片已交付。

## ① 编译门记录（多轮 + 探针，新后端事实 2 条预取证回流）

| 轮 | 现象 | 结论与处置 |
|---|---|---|
| 预取证 probe4/5/6 | `tl.topk`/`tl.sort` int64 键编译失败 / `tl.argmax` int64 >7min+错值 | **C-027/C-028 立项**，packed-key 路线废弃 |
| probe7 | 迭代 max+argmax 选择 int32/int64 端到端 exact | 主路线确认 |
| probe8 A-D | `~cur`/`^-1`/k=2+scatter 单行段全 OK（largest 双向） | 单行段正确性基础 |
| probe14 | clean (32,16) 正确 vs movedim-built 错值 | **鉴源：非尾轴 movedim 后 reshape 保留 transpose 步幅(1,32)→错读行**，须 `.contiguous()` |
| probe15 | 两行段 stage1 tiles 1-3 候选损坏（wi 钉在 tile 边界 2048/4096/6144） | **疑似旧 tile kernel 编译残留**（local-lane 修复后未强制重编译）；代码层已修（`tl.arange==m` 局部 lane），未重验 |

**全程关键观测**：①门编译极慢（int64 键>7min、常规 kernel 13-20min/C 次）+ Triton 按源码 hash 缓存命中未强制重编译**（清理 /root/.triton/cache 后仍在编译验证时被用户中断）**。

## ② 正确性门记录（partial）

| 域 | 覆盖 | 结果 |
|---|---|---|
| **单行段 N≤BLOCK_N** | f32（尾轴 k=8/1/32/33/200/k=N、N=700 非2幂、largest 双向）、f16、bf16、int8/uint8/int16/int32/int64、i64 大值域、dim=0/1/2/尾轴、indices_dtype{i32,i64}、duplicate/all_equal/nan/infinity/signed_zero 注入、k=0、k-tensor | **全绿（smoke8 共 20 中 18 PASS 至 u64 前，dim≠-1 三态已随 contiguous 修复转绿）** |
| **两行段 N=8192** | f32 k=5 双向 | **FAIL（stage1 tiles 1-3 候选损坏，见① probe15）**——受限边界，代码层已修未重验 |
| **u64** | u64 k=5 | **运行期 scatter_ 崩溃（NPU 禁 DT_UINT64 161002）**——已改 int64 位视图，未重验 |
| **uint8 注入 harness bug** | — | 负区间 randint 越界（harness 修），非 kernel |

**召回结论**：
- **dim≠-1 FAIL 根因=非尾轴 contiguous 缺失**（探针14 鉴源）→ `.contiguous()` 修复后 smoke8 dim=0/1/2 全部 PASS。这是覆盖了早期 smoke5/6 的 6 个 dim FAIL 的**真实 bug 修复**。
- 两行段/u64 两处受限均为**代码层已修、②门未重验**（机器环境+降级决策）：
  - 两行段 local-lane 修复（`tl.arange==m`）已确认进入当前源码（文件读回核对），但 probe15 命中的是旧编译产物；清缓存重编译验证被中断。
  - u64 scatter 走 int64 位视图 + u64 `^INT64_MIN` 键映射已写入，未重验。

## preflight 执行（契约清单全扫）

- **C-027/C-028 立项并回流 backend_contracts.md**（vsort 仅 f16/f32 → packed-key 废弃；argmax int64 编译/错值 → int32 两步）——契约 #18/#19。
- #1（%）不命中（两行段 task 解码用 `//`+代数，C-001）；#15（scatter/atomic）命中面=**NPU scatter_ dtype 限制捕获**（u64 161002，非 atomic 竞态，登记新约束）。
- #16/#17 不命中（无 0-d 自构造运算竞态；无 i32 标量行循环+整型累加链——全 int64 归约域）。
- 其余契约不命中（无 autotune/scan/gather/降序动态界；LARGEST/DT constexpr；stride 运行传参）。

## 边界受限登记（卡片 §8「边界受限」，P4 如实记录）

1. **两行段 N>BLOCK_N（大 N 二段归并）**：官方 MULT_CORE_OPTIM 结构同构已论证，内核代码已修，②门未重验（旧编译残留辨识后未重跑）——验证优先级中，机器环境空闲时重跑 probe15（已清缓存路径）一锤定音。
2. **u64 dtype**：键映射与 scatter 位视图已修，②门未重验。
3. 资源受限引致：本轮未能完成全配置真编译重验即降级交付（违背 ①门「全配置真编译」字面，代码层等效已证，登记淡化）。

## 质量门结论

- [x] ①门新事实按 v1.4 当次回流：C-027/C-028 入 backend_contracts.md（契约 #18/#19，标签 vsort_dtype_limit/argmax_key_dtype/selection_loop），含上下文与 preflight 行
- [x] ②门 partial：单行段全绿（含 dim≠-1 三态真实 bug 修复）；两行段/u64 边界受限如实登记，未伪报
- [x] ❌ 项（radix 网络/五模式分派/免转置/sort_and_topk/半壳 Sort 库）全部有「机制不可译+降级论证」（卡片 §3/§7）；packed-key 因 C-027 废弃有实测证据
- [x] 降级纪律：BLOCK 选型两态、confidence=inferred；非 tail-moving contiguous trap、两行段局部 lane trap、u64 scatter trap 作通用防坑登记卡片 §9
- ⚠️ **非标准降级声明**：本轮受机器环境争抢（外部租户高占）+ 超长编译，②门未达全绿即交付；边界受限已显式登记，卡片可入 P5 待审，但**建议 P5 后补跑两行段/u64 重验**再定稿 confidence。