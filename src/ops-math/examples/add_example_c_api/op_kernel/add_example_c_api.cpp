/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "add_example_c_api_tiling_data.h"

constexpr uint32_t TILE_LENGTH = 2048;

__global__ __aicore__ void add_example_c_api(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddExampleCApiTilingData);
    AscendC::SetSysWorkspaceForce(workspace);
    asc_init();

    __ubuf__ float xLocal[TILE_LENGTH];
    __ubuf__ float yLocal[TILE_LENGTH];
    __ubuf__ float zLocal[TILE_LENGTH];

    const __gm__ AddExampleCApiTilingData* tilingData = (__gm__ AddExampleCApiTilingData*)tiling;
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    int64_t ubFactor = tilingData->ubFactor;

    int64_t blockIdx = asc_get_block_idx();
    int64_t blockLength = totalNum - blockFactor * blockIdx;
    blockLength = (blockLength > blockFactor) ? blockFactor : blockLength;

    __gm__ float* xGm = (__gm__ float*)x + blockFactor * blockIdx;
    __gm__ float* yGm = (__gm__ float*)y + blockFactor * blockIdx;
    __gm__ float* zGm = (__gm__ float*)z + blockFactor * blockIdx;

    int64_t loopCount = (blockLength + ubFactor - 1) / ubFactor;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength - ubFactor * i) : ubFactor;

        asc_copy_gm2ub_sync((__ubuf__ void*)xLocal, (__gm__ void*)(xGm + i * ubFactor), currentNum * sizeof(float));
        asc_copy_gm2ub_sync((__ubuf__ void*)yLocal, (__gm__ void*)(yGm + i * ubFactor), currentNum * sizeof(float));

        asc_add_sync(zLocal, xLocal, yLocal, currentNum);

        asc_copy_ub2gm_sync((__gm__ void*)(zGm + i * ubFactor), (__ubuf__ void*)zLocal, currentNum * sizeof(float));
    }
}
