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
 * \file allto_all_all_gather_batch_mat_mul_tiling_struct.h
 * \brief
 */

#ifndef MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_TILING_STRUCT_H_H
#define MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_TILING_STRUCT_H_H

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

struct Mc2CommonTiling {
    uint32_t epGroupSize; // ep
    uint32_t tpGroupSize; // tp
    uint64_t expert;      // E
    uint64_t EOverEp;     // E/ep
    uint64_t C;
    uint64_t COverTp; // C/tp
    uint64_t H;       // H
    uint64_t HOverTp; // H/tp
    uint64_t MOverTp; // M/tp
    uint32_t aivCoreNum;
    uint32_t inputDatasize;
    uint32_t biasDatasize;
    uint64_t ubCapacityForTrans;
    uint64_t ubCapacityForAddActivate;
    bool isBias;
    bool y2Flag;
    bool y3Flag;
    bool isWeightTrans;
    BMM_TileInfo localTileE;    // E 轴本地块切分信息
    BMM_TileInfo domesticTileE; // E 轴非本地块切分信息
    BMM_TileInfo localTileC;    // C 轴本地块切分信息
    BMM_TileInfo domesticTileC; // C 轴非本地块切分信息
    BMM_TileInfo localUbTranspose;
    BMM_TileInfo localTailUbTranspose;
    BMM_TileInfo domesticUbTranspose;
    BMM_TileInfo domesticTailUbTranspose;
    BMM_TileInfo localUbAdd;
    BMM_TileInfo localTailUbAdd;
    BMM_TileInfo domesticUbAdd;
    BMM_TileInfo domesticTailUbAdd;
    uint32_t activateType; // 激活措施：0为不激活，1为GELU，3为Relu，4为FastGELU
    uint32_t xShardFlag;
    uint32_t fastGeluBuffer;
    uint64_t totalUbSize;
};

struct AlltoAllAllGatherBatchMatMulTilingData {
    AscendC::tiling::Mc2InitTiling hcclInitTiling;
    AscendC::tiling::Mc2CcTiling alltoAllCcTiling;
    AscendC::tiling::Mc2CcTiling allGatherCcTiling;
    Mc2CommonTiling commonTiling;
    BMM_Mc2MatmulTilingData localTiling;
    BMM_Mc2MatmulTilingData domesticTiling;
    BMM_Mc2MatmulTilingData localTailTiling;
    BMM_Mc2MatmulTilingData domesticTailTiling;
    BMM_Mc2MatmulTilingData domesticTailETiling;
};
}

#endif