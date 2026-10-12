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
 * \file pad_v3_grad_replicate_small_h_large_w_common.h
 * \brief Common methods shared by SmallHLargeW and SmallHLargeWF16 kernels
 */
#ifndef _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_COMMON_
#define _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_COMMON_

#include "pad_v3_grad_replicate_base.h"

template <typename T, typename FinalT>
class PadV3GradReplicateSmallHLargeWCommon : public PadV3GradReplicateKernelBase<T, FinalT> {
    using base = PadV3GradReplicateKernelBase<T, FinalT>;
    friend base;

public:
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx);
    __aicore__ inline void CopyWs2UB(const int32_t batchIdx, const int64_t copyCount, const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount);
    __aicore__ inline void CopyIn(const int32_t copyCount, const int64_t workspaceOffset);
    __aicore__ inline void Compute(const int32_t copyCount);
    __aicore__ inline void CopyOut(const int32_t copyCount, const int64_t offset);

protected:
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    event_t MTE3ToMTE2Event;
};

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateSmallHLargeWCommon<T, FinalT>::CopyGm2UB(const int32_t cycleIdx,
                                                                                  const int64_t copyCount,
                                                                                  const int32_t batchIdx)
{
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    int32_t alignCopyCount = CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    int64_t offset = 0;
    for (size_t i = 0; i < this->height; i++) {
        offset = i * this->width + cycleIdx * this->ubFactorElement + batchIdx * this->batchStride +
                 this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[i * this->ubFactorElement], this->mGmX[offset], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateSmallHLargeWCommon<T, FinalT>::CopyWs2UB(const int32_t batchIdx,
                                                                                  const int64_t copyCount,
                                                                                  const int32_t flag)
{
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < this->outHeight; i++) {
            workspaceOffset = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < this->outHeight; i++) {
            workspaceOffset = (i + 1) * this->width - COPY_ROWS_AND_COLS + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateSmallHLargeWCommon<T, FinalT>::CopyOut2Workspace(const int32_t tIdx,
                                                                                          const int64_t calCount)
{
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    for (size_t i = 0; i < this->outHeight; i++) {
        workspaceOffset = i * this->width + tIdx * this->ubFactorElement + this->blockIdx * this->workspacePerCore;
        DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateSmallHLargeWCommon<T, FinalT>::CopyIn(const int32_t copyCount,
                                                                               const int64_t workspaceOffset)
{
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyPad(xLocal, this->mGmWorkspace[workspaceOffset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateSmallHLargeWCommon<T, FinalT>::Compute(const int32_t copyCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    uint32_t alignCopyCount = CeilAlign(copyCount, this->perBlockCount);
    DataCopy(yLocal, xLocal, alignCopyCount);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateSmallHLargeWCommon<T, FinalT>::CopyOut(const int32_t copyCount,
                                                                                const int64_t offset)
{
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

#endif // _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_COMMON_
