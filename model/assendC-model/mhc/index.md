---
kind: index
type: section_index
title: ops/transformer/mhc/
description: mHC（流形约束超连接，arXiv:2512.24880）算子族设计 wiki 索引；mhc_sinkhorn_common 为共享 host 库不单独立卡。
---

# ops/transformer/mhc/

前向：
- [mhc_pre.md](mhc_pre.md) - pre 投影（RMSNorm+matmul）：9 tilingKey 4 条流水线（arch35 SPLIT_BS/ND/M_K + 910b 两段）
- [mhc_post.md](mhc_post.md) - post 广播缩放（MemBase：逐行现取 vs 整块预搬）｜[mhc_post_a5.md](mhc_post_a5.md) - arch35 三模板姊妹卡
- [mhc_sinkhorn_a5.md](mhc_sinkhorn_a5.md) - Sinkhorn 投影（纯 A5 / AIV_ONLY MicroAPI）

反向：
- [mhc_pre_backward.md](mhc_pre_backward.md) - pre 反向：四段 CV 流水（V→SyncAll→C→V→V）
- [mhc_post_backward.md](mhc_post_backward.md) - post 反向：arch35 MIX 1:2 / arch22 AIV_ONLY 双平台单卡，AIC/AIV 读写集不相交
- [mhc_pre_sinkhorn.md](mhc_pre_sinkhorn.md) - pre+Sinkhorn 融合：6 tilingKey 3 流水线（含预留不可达路径）
- [mhc_pre_sinkhorn_backward.md](mhc_pre_sinkhorn_backward.md) - 融合反向：确定性/非确定性/arch22 三流水线（arch35 恒走确定性路径）
- [mhc_sinkhorn_backward_a5.md](mhc_sinkhorn_backward_a5.md) - Sinkhorn 反向（纯 A5，与正向卡布局契约互链）
