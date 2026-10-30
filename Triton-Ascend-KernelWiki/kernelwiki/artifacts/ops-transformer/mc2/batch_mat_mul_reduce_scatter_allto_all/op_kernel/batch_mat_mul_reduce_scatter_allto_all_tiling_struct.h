/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_mat_mul_reduce_scatter_allto_all_tiling_struct.h
 * \brief
 */

#ifndef BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_TILING_STRUCT_H
#define BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL_TILING_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h")
#include "../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h"
#else
#include "../../3rd/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_data.h"
#endif

namespace optiling {
struct BMM_TileInfo {
    uint64_t tileCnt;
    uint64_t tileLen;
    uint64_t tailCnt;
    uint64_t tailLen;
};

struct BMM_Mc2MatmulTilingData {
    uint32_t rankDim;
    uint32_t rankM;
    uint32_t rankID;
    uint32_t enableL2Tile;
    Mc2BatchMatmulTilingData bmmTilingData;
};

struct Mc2RSATATiling {
    uint32_t epGroupSize;                   // 每个ep域内的并行运行的专家的个数
    uint32_t tpGroupSize;                   // 每个tp域内块的个数
    uint64_t expert;                        // 专家个数
    uint64_t EOverEp;                       // E/ep
    uint64_t C;
    uint64_t COverTp;                       // C/tp
    uint64_t H;
    uint64_t HOverTp;                       // H/tp
    uint64_t MOverTp;                       // M/tp
    uint32_t aivCoreNum;
    uint32_t inputDatasize;
    uint32_t biasDatasize;
    uint64_t ubCapacityForAdd;
    uint64_t totalUbSize;
    bool isBias;
    bool isWeightTrans;
    BMM_TileInfo localTileE;             // E 轴本地块切分信息
    BMM_TileInfo domesticTileE;          // E 轴非本地块切分信息
    BMM_TileInfo localTileC;             // C 轴本地块切分信息
    BMM_TileInfo domesticTileC;          // C 轴非本地块切分信息
    uint32_t yShardFlag;
};

struct BatchMatMulReduceScatterAlltoAllTilingData {
    AscendC::tiling::Mc2InitTiling hcclInitTiling;
    AscendC::tiling::Mc2CcTiling alltoAllCcTiling;
    AscendC::tiling::Mc2CcTiling reduceScatterCcTiling;
    Mc2RSATATiling commonTiling;
    BMM_Mc2MatmulTilingData localTiling;
    BMM_Mc2MatmulTilingData domesticTiling;
    BMM_Mc2MatmulTilingData localTailTiling;
    BMM_Mc2MatmulTilingData domesticTailTiling;
};
}
#endif
