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
 * \file lightning_indexer_grad_service_vector_pre.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_PRE_H
#define LIGHTNING_INDEXER_GRAD_SERVICE_VECTOR_PRE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_grad_common.h"
#include "lightning_indexer_grad_tiling.h"

namespace LigKernel {
using namespace LIGCommon;
using namespace AscendC;
using namespace optiling;

template <typename LIGT>
class LIGVectorPre {
public:
    using dataType = typename LIGT::dataType;
    
    __aicore__ inline LIGVectorPre(){};
    __aicore__ inline void Init(TPipe *pipe_in, __gm__ uint8_t *dk, __gm__ uint8_t *workspace, const LIGTilingData *__restrict orgTilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

protected:
    TPipe *pipe;
    GlobalTensor<float> dkWorkSpaceGm;
    GlobalTensor<float> dkCoreWorkspaceGM;

    const LIGTilingData *__restrict tilingData;

    uint32_t cBlockIdx;
    uint32_t kPreBlockFactor;
    uint32_t kPreBlockTotal;
    uint32_t kPreBlockTail;
    int64_t initdkSize;
    int64_t dkOffset;
};

template <typename LIGT>
__aicore__ inline void LIGVectorPre<LIGT>::Init(TPipe *pipe_in, __gm__ uint8_t *dk, __gm__ uint8_t *workspace, const LIGTilingData *__restrict orgTilingData)
{
    cBlockIdx = GetBlockIdx();
    pipe = pipe_in;
    tilingData = orgTilingData;

    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkWorkSpaceOffset / sizeof(float));

    // Init workspace to store per-core dKey partial results.
    dkCoreWorkspaceGM.SetGlobalBuffer((__gm__ float *)workspace + tilingData->dkCoreWorkspaceOffset / sizeof(float));

    uint32_t coreNum = tilingData->usedCoreNum;
    kPreBlockFactor = (tilingData->dkSize + coreNum - 1) / coreNum;
    kPreBlockTotal = (tilingData->dkSize + kPreBlockFactor - 1) / kPreBlockFactor;
    int64_t kPreTailNumTmp = tilingData->dkSize % kPreBlockFactor;
    kPreBlockTail = kPreTailNumTmp == 0 ? kPreBlockFactor : kPreTailNumTmp;

    initdkSize = cBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
    dkOffset = ((int64_t)cBlockIdx) * kPreBlockFactor;
}

template <typename LIGT> 
__aicore__ inline void LIGVectorPre<LIGT>::Process()
{
    // process clear dk workspace
    if (g_coreType == AIV && cBlockIdx < kPreBlockTotal) {
        InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
    }
}

template <typename LIGT> 
__aicore__ inline void LIGVectorPre<LIGT>::SyncALLCores()
{
    SyncAll();
}
} // namespace LigKernel
#endif