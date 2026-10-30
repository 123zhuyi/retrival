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
 * \file pad_v3_grad_replicate_large_h_small_w_common.h
 * \brief Common methods shared by LargeHSmallW and LargeHSmallWF16 kernels
 */
#ifndef _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_COMMON_
#define _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_COMMON_

#include "pad_v3_grad_replicate_base.h"

template <typename T, typename FinalT>
class PadV3GradReplicateLargeHSmallWCommon : public PadV3GradReplicateKernelBase<T, FinalT> {
    using base = PadV3GradReplicateKernelBase<T, FinalT>;
    friend base;

public:
    __aicore__ inline void CopyGm2UB(const int32_t batchIdx, const int32_t flag);
    __aicore__ inline void CopyGmAndWs2UB1(const int32_t batchIdx);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx);
    __aicore__ inline void CopyOut2Ws(const int64_t calCount, const int32_t flag);

protected:
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    event_t MTE3ToMTE2Event;
};

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateLargeHSmallWCommon<T, FinalT>::CopyGm2UB(const int32_t batchIdx,
                                                                                  const int32_t flag)
{
    int64_t gmXOffset;
    int32_t alignCopyCount = CeilAlign(this->width, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - this->width), (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = i * this->width + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[gmXOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = (this->height - COPY_ROWS_AND_COLS + i) * this->width + batchIdx * this->batchStride +
                        this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[gmXOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateLargeHSmallWCommon<T, FinalT>::CopyGmAndWs2UB1(const int32_t batchIdx)
{
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
        workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
        DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
    }
    for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
        xGmOffset = i * this->width + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[(i - this->padTop) * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset], copyParams, padParams);
    }
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
        workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                           this->blockIdx * this->workspacePerCore;
        DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) * SMALL_WIDTH_LIMIT],
                    this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateLargeHSmallWCommon<T, FinalT>::CopyGmAndWorkspace2UB2(
    const int32_t transBlkIdx, const int32_t transTimes, const int32_t cycles, const int32_t batchIdx)
{
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t xGmOffset1;
    int64_t xGmOffset2;
    int64_t xGmOffset3;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();

    if (transBlkIdx == 0) {
        for (size_t i = 0; i < this->ubFactorElement; i++) {
            xGmOffset1 = (i + this->padTop) * this->width + batchIdx * this->batchStride +
                         this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset1], copyParams, padParams);
        }
        PipeBarrier<PIPE_MTE2>();
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
            workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }

    } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
        for (size_t i = 0; i < this->ubFactorElement; i++) {
            xGmOffset2 = (this->ubFactorElement * transBlkIdx + this->padTop + i) * this->width +
                         batchIdx * this->batchStride + this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset2], copyParams, padParams);
        }
    } else if (transBlkIdx == transTimes - 1) {
        if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                                   this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                xGmOffset3 = (i + (transTimes - 1) * this->ubFactorElement + this->padTop) * this->width +
                             batchIdx * this->batchStride + this->ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset3], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                workspaceOffset3 = (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                                   this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->padBottom) + i) * SMALL_WIDTH_LIMIT],
                            this->mGmWorkspace[workspaceOffset3], copyParams, padParams);
            }
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T, typename FinalT>
__aicore__ inline void PadV3GradReplicateLargeHSmallWCommon<T, FinalT>::CopyOut2Ws(const int64_t calCount,
                                                                                   const int32_t flag)
{
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
            workspaceOffset = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            workspaceOffset = (COPY_ROWS_AND_COLS - this->padTop + i) * this->width +
                              this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

#endif // _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_COMMON_
