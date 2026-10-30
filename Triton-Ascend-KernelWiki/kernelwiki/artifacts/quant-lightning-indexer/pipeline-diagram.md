# QuantLightningIndexer — Pipeline 抽象 (S6)

来源：`op_kernel/quant_lightning_indexer_kernel.h`、`op_kernel/quant_lightning_indexer_vector.h`。**不复制 CANN 代码**。

## 1. 计算流程

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Preload | QLIPreload::Init | AIC+AIV | 输入加载、量化参数读取 |
| Cube1 | QKCompute | AIC | Q·K^T matmul（int8/fp8 + 反量化 → FP32） |
| Vector1 | IndexCompute | AIV | Scale_Q ⊙ Scale_K^T 逐元素乘 → ReLU → W @ 相关性加权 |
| Vector2 | TopK | AIV | topk 索引选择 |
| WriteBack | WriteBack | AIV | 稀疏索引写回 GM |

## 2. 阶段映射表

| 阶段 | CANN 机制 | 目的 | Triton 抽象 | 可迁移? |
|------|----------|------|------------|--------|
| 数据读入 | DataCopy GM->UB | 载入 Q/K/weights/dequant_scale | tl.load | 是 |
| 反量化 | QuantDequant 向量算子 | int8/fp8 → FP32 | tl.load(..).to(tl.float32) * scale | 是 |
| Q·K^T | Cube matmul | 相关性计算 | tl.dot(q,k.T) FP32 | 是 |
| Scale_Q ⊙ Scale_K^T | Muls 逐元素乘 | 反量化系数融合 | * (q_scale * k_scale) | 是 |
| ReLU | ReLU 激活 | 过滤负相关 | tl.maximum(0, x) | 是 |
| W @ 加权 | Cube matmul / Vector 归约 | 跨 g 轴加权聚合 | tl.dot(weights, scores) | 是 |
| topk | TopK API | 稀疏索引选择 | tl.topk(x, sparse_count) | 是 |
| 写回 | DataCopy | 输出 sparse_indices | tl.store | 是 |
| 手动同步 | SetFlag/WaitFlag | CV 依赖 | 编译器排流 | 否（CANN-specific） |