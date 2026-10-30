# FusedFloydAttention — Pipeline 抽象 (S6)

## 1. 计算流程（4 次 matmul）

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Cube1 | IterateBmm1 | AIC | Q·K1^T matmul |
| Cube1k1 | IterateBmm1K1 | AIC | Q·K2^T matmul |
| Vector1 | ProcessVec1 | AIV | QK1 + QK2 相加 → scale → mask → Softmax |
| Cube2 | IterateBmm2 | AIC | P·V1 matmul |
| Cube2v2 | IterateBmm2V2 | AIC | P·V2 matmul |
| Vector2 | ProcessVec2 | AIV | P·V1 + P·V2 相加 → 写回 |

## 2. 阶段映射表

| 阶段 | CANN 机制 | 目的 | Triton 抽象 | 可迁移? |
|------|----------|------|------------|--------|
| Q·K1^T | Cube IterateBmm1 | 步 1 | `tl.dot(q,k1.T)` | 是 |
| Q·K2^T | Cube IterateBmm1K1 | 步 2 | `tl.dot(q,k2.T)` | 是 |
| scale+mask+softmax | Vector ProcessVec1 | 步 3 | 在线 softmax | 是 |
| P·V1 | Cube IterateBmm2 | 步 4 | `tl.dot(p,v1)` | 是 |
| P·V2 | Cube IterateBmm2V2 | 步 5 | `tl.dot(p,v2)` | 是 |
| 相加写回 | Vector ProcessVec2 | 步 6 | `out1+out2` | 是 |
| 手动同步 | SetFlag/WaitFlag | CV 依赖 | 编译器排流 | 否 |