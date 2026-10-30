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
 * \file prompt_flash_attention_antiquant_preproc_q.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_PREPROC_Q_H
#define PROMPT_FLASH_ATTENTION_PREPROC_Q_H

#include "../comm/prompt_flash_attention_comm.h"

template <typename PFAT>
class PromptFlashAttentionAntiQuantPreProcessQ {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T, PFAT::calcMode>::softmaxType;
    __aicore__ inline PromptFlashAttentionAntiQuantPreProcessQ() {};
    __aicore__ inline void Process(TSCM<QuePosition::VECIN, 1, 0x4> &queryScmQueue,
                                   GlobalTensor <T> &queryGm,
                                   TQue<QuePosition::VECIN, 1> &queryInputQueue,
                                   TQue<QuePosition::VECOUT, 1> &queryOutputQueue,
                                   const TaskParam &params, const ConstParam &constParams,
                                   const PromptFlashAttentionTilingData* __restrict tilingData);

protected:
    __aicore__ inline void ProcessInner(TSCM<QuePosition::VECIN, 1, 0x4> &queryScmQueue,
                                   GlobalTensor <T> &queryGm,
                                   TQue<QuePosition::VECIN, 1> &queryInputQueue,
                                   TQue<QuePosition::VECOUT, 1> &queryOutputQueue,
                                   const TaskParam &params, const ConstParam &constParams,
                                   const PromptFlashAttentionTilingData* __restrict tilingData,
                                   LocalTensor<T> &queryScm);
    __aicore__ inline void DataCopyBNSD(LocalTensor<T> &querySrcUb, GlobalTensor <T> &queryGm, int64_t copySize, int64_t qOffset);
    __aicore__ inline void DataCopyND2NZ(int32_t headDim, uint32_t gSplitSize, LocalTensor<T> &queryUb, LocalTensor<T> &querySrcUb);
    __aicore__ inline void DataCopyL1(int32_t headDim, uint32_t gSplitSize, int64_t startRow, LocalTensor<T> &queryScm, LocalTensor<T> &queryUb, const TaskParam &params);

protected:
    static constexpr int32_t Q_BLOCK_BYTE_NUM = 32; // 32: size of block

    static constexpr uint32_t QUERY_BUFFER_SIZE_BYTE_4k = 4096; // 4096:ub size of query
};

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantPreProcessQ<PFAT>::Process(TSCM<QuePosition::VECIN, 1, 0x4> &queryScmQueue,
                                   GlobalTensor <T> &queryGm,
                                   TQue<QuePosition::VECIN, 1> &queryInputQueue,
                                   TQue<QuePosition::VECOUT, 1> &queryOutputQueue,
                                   const TaskParam &params, const ConstParam &constParams,
                                   const PromptFlashAttentionTilingData* __restrict tilingData)
{
    LocalTensor<T> queryScm = queryScmQueue.AllocTensor<T>();

    if (params.isFirstInnerIter) {
        ProcessInner(queryScmQueue,
        queryGm,
        queryInputQueue,
        queryOutputQueue,
        params, constParams,
        tilingData,
        queryScm);
    }
    queryScmQueue.EnQue(queryScm);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantPreProcessQ<PFAT>::DataCopyBNSD(LocalTensor<T> &querySrcUb, GlobalTensor<T> &queryGm, int64_t copySize, int64_t qOffset)
{
    DataCopy(querySrcUb, queryGm[qOffset], copySize);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantPreProcessQ<PFAT>::DataCopyND2NZ(int32_t headDim, uint32_t gSplitSize, LocalTensor<T> &queryUb, LocalTensor<T> &querySrcUb)
{
    DataCopyParams initParams;
    uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(T);
    // nd2nz
    initParams.blockCount = (headDim + elementTypeSize - 1) / elementTypeSize;
    initParams.blockLen = 1;
    initParams.srcStride = 0;
    initParams.dstStride = gSplitSize - 1;

    for (uint32_t j = 0;j < gSplitSize; j++) {
        DataCopy(queryUb[j * elementTypeSize], querySrcUb[j * headDim], initParams);
    }
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantPreProcessQ<PFAT>::DataCopyL1(int32_t headDim, uint32_t gSplitSize, int64_t startRow, LocalTensor<T> &queryScm, LocalTensor<T> &queryUb, const TaskParam &params)
{
    DataCopyParams initParams;
    uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(T);
    initParams.blockCount = (headDim + elementTypeSize - 1) / elementTypeSize;
    initParams.blockLen = gSplitSize;
    initParams.srcStride = 0;
    initParams.dstStride = 32 - gSplitSize; // 32:souter of basic block
    DataCopy(queryScm[startRow * headDim], queryUb, initParams);
}

template <typename PFAT>
__aicore__ inline void PromptFlashAttentionAntiQuantPreProcessQ<PFAT>::ProcessInner(TSCM<QuePosition::VECIN, 1, 0x4> &queryScmQueue,
                                   GlobalTensor <T> &queryGm,
                                   TQue<QuePosition::VECIN, 1> &queryInputQueue,
                                   TQue<QuePosition::VECOUT, 1> &queryOutputQueue,
                                   const TaskParam &params, const ConstParam &constParams,
                                   const PromptFlashAttentionTilingData* __restrict tilingData,
                                   LocalTensor<T> &queryScm)
{
    int64_t qOffset = static_cast<int64_t>(params.tensorAOffset);
    int64_t headDim = static_cast<int64_t>(tilingData->promptAttentionBaseParams.headSize);

    uint32_t gSplitSize = (QUERY_BUFFER_SIZE_BYTE_4k * 2) / headDim / sizeof(T); // 2:block of ub
    uint32_t loopCount = (params.singleProcessSOuterSize + gSplitSize - 1) / gSplitSize;
    loopCount = (loopCount == 0) ? 1 : loopCount;
    uint32_t tailSplitSize = params.singleProcessSOuterSize - (loopCount - 1) * gSplitSize;

    for (uint32_t i = 0; i < loopCount; i++) {
        int64_t startRow = i * gSplitSize;
        if (i + 1 == loopCount) {
            gSplitSize = tailSplitSize;
        }
        LocalTensor<T> querySrcUb = queryInputQueue.template AllocTensor<T>();
        DataCopyBNSD(querySrcUb, queryGm, headDim * gSplitSize, qOffset);

        queryInputQueue.template EnQue(querySrcUb);
        queryInputQueue.DeQue<T>();

        LocalTensor<T> queryUb = queryOutputQueue.template AllocTensor<T>();

        DataCopyND2NZ(headDim, gSplitSize, queryUb, querySrcUb);

        queryInputQueue.FreeTensor(querySrcUb);

        queryOutputQueue.template EnQue(queryUb);
        queryOutputQueue.DeQue<T>();

        DataCopyL1(headDim, gSplitSize, startRow, queryScm, queryUb, params);

        queryOutputQueue.FreeTensor(queryUb);
    }
}

#endif  // PROMPT_FLASH_ATTENTION_PREPROC_Q_H