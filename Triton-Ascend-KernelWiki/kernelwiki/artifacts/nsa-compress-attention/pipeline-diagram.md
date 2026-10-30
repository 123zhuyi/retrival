# NsaCompressAttention — Pipeline 抽象 (S6)

来源：`docs/aclnnNsaCompressAttention.md`、`op_kernel/nsa_compress_attention_s1s2_bn2gs1_sab.h`。**不复制 CANN 代码**。

## 1. 计算流程（2 阶段 Cube + 2 阶段 Vector）

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Cube1 | `IterateBmm1` | AIC | Q·K^T matmul → 原始 attention score |
| Vector1 | `ProcessVec1` | AIV | scale → atten_mask select → Softmax → attention score 写回 workspace |
| Cube2 | `IterateBmm2` | AIC | P·V matmul → attention_out |
| Vector2 | `ProcessVec2` | AIV | importance score 计算（滑动窗口求和 + 跨 head 累加 + topk_mask select）+ `TopkCompute` |

2 次 matmul：bmm1=Q·K^T、bmm2=P·V（P 为 Vector1 的 softmax 结果）。

## 2. 阶段映射表

| 阶段 | CANN 机制 | 目的 | Triton 抽象 | 可迁移? |
|------|----------|------|------------|--------|
| 数据读入 | DataCopy GM->UB | 载入 Q/K/V | tl.load + stride | 是 |
| QK^T | Cube IterateBmm1，L0C FP32 累加 | 步 1 | tl.dot(q,k.T) FP32 累加 | 是 |
| mask + scale | Vector ProcessVec1：SelectWithBytesMask + Muls | 步 1 | tl.where 掩码 + *sm_scale | 是 |
| Softmax | Vector ProcessVec1：SoftMax API | 步 1 | 在线 softmax（running max/sum + rescale） | 是 |
| P·V | Cube IterateBmm2 | 步 2 | tl.dot(p,v) | 是 |
| importance score 滑动窗口 | Vector ProcessVec2：DataCopyPad + ConfusionTranspose + Muls + Add + DataCopy | 滑动窗口求和 | 滑动窗口 reduce（简单循环或卷积） | 是 |
| 跨 head 累加 | Vector ProcessVec2：Add 循环 reduce g | 跨 head 求和 | tl.sum 跨 G 轴 | 是 |
| topk_mask | Vector ProcessVec2：SelectWithBytesMask | 可选掩码 | tl.where | 是 |
| topk 索引 | Vector TopkCompute：TopK API | 输出 topk 索引 | tl.topk（或手工排序） | 是 |
| 写回 | DataCopy + DataCopyPad GM | 输出 | tl.store masked | 是 |
| 双缓冲 | mm1Res[0]/[1] double buffer | 搬运/计算重叠 | 编译器排流 | 意图迁移 |
| CV 同步 | SetFlag/WaitFlag + PipeBarrier | CV 数据依赖 | 编译器排流 | 否（CANN-specific） |
| ConfusionTranspose | 手动 ND→ND transpose | 数据重排 | 不移植（Triton 无显式 transpose 需求） | 否（CANN-specific） |