/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist.cpp
 * \brief
 */

#include "arch35/cdist_simt.h"
#include "arch35/cdist.h"
#include "arch35/cdist_broadcast.h" // 方案2: NsCdist::CdistBroadcast<T> (M==1 Broadcast fast path)
#include "arch35/cdist_reduce_hl.h" // 方案三: NsCdist::CdistReduceHL<T> (M∈[2,256] broadcast+高层reduce, round2 crossover)
#include "cdist_tiling_data.h"
#include "cdist_brc_tilingdata.h" // 方案2: POD CdistBrcTilingData (host/kernel 共用)
#include "cdist_tiling_key.h"

// KERNEL_MODE 模板轴：0=Normal(M>256) / 1=SIMT(M<=256 && M>1) / 2=Broadcast(M==1 快路径)。
// 与 op_host RunTiling 的 SetTilingKey(KERNEL_MODE) 及 cdist_tiling_key.h 的 TPL_SEL 一一对应。
template <uint32_t KERNEL_MODE>
__global__ __aicore__ void cdist(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 全 kernel 只能有一个 REGISTER_TILING_DEFAULT（opc EB0500）。注册主结构 CdistTilingData；
    // Broadcast 分支用 GET_TILING_DATA_WITH_STRUCT 把同一 tiling GM 按 CdistBrcTilingData 解读即可。
    REGISTER_NONE_TILING;
    AscendC::TPipe pipe;
    if constexpr (KERNEL_MODE == 2) { // 方案2: M==1 Broadcast 快路径（用独立 CdistBrcTilingData）
        GET_TILING_DATA_WITH_STRUCT(CdistBrcTilingData, brcTilingData, tiling);
        NsCdist::CdistBroadcast<DTYPE_X1> op(&pipe, &brcTilingData);
        op.Init(x1, x2, y);
        op.Process();
    } else if constexpr (KERNEL_MODE == 3) { // 方案三: M∈[2,256] 矢量胜 broadcast+高层reduce（复用主 CdistTilingData）
        GET_TILING_DATA_WITH_STRUCT(CdistTilingData, tilingData, tiling);
        NsCdist::CdistReduceHL<DTYPE_X1> op;
        op.Init(x1, x2, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (KERNEL_MODE == 1) { // SIMT 场景
        GET_TILING_DATA_WITH_STRUCT(CdistTilingData, tilingData, tiling);
        NsCdist::CdistSimt<DTYPE_X1> op(&pipe, &tilingData);
        op.Init(x1, x2, y);
        op.Process();
    } else { // Normal 场景 (KERNEL_MODE == 0)
        GET_TILING_DATA_WITH_STRUCT(CdistTilingData, tilingData, tiling);
        NsCdist::Cdist<DTYPE_X1> op;
        op.Init(x1, x2, y, &tilingData, &pipe);
        op.Process();
    }
}
