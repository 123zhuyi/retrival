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
 * \file pad_v4_grad_h_w_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_H_W_PAD_H_
#define _PAD_V4_GRAD_H_W_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradPadHW : public PadV4GradBase<T, PadV4GradPadHW<T>> {
    using base = PadV4GradBase<T, PadV4GradPadHW<T>>;
    friend base;

public:
    __aicore__ inline PadV4GradPadHW(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset, const int32_t cycles,
                                      const int32_t transBlkIdx, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void implTransposeAndCompute(const int64_t transCount, const int32_t flag);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TQue<QuePosition::VECOUT, 1> transposeQue;

    event_t MTE3ToMTE2Event;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradPadHW<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * 16);
    pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * 16);
    pipe->InitBuffer(transposeQue, 1, this->ubFactorElement * sizeof(T) * 16);
}

template <typename T>
__aicore__ inline void PadV4GradPadHW<T>::CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset,
                                                     const int32_t cycles, const int32_t transBlkIdx,
                                                     const int32_t flag)
{
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    int64_t gmYOffset3 = 0;
    int64_t gmYOffset4 = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padLeft) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padRight) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> transposeData = transposeQue.DeQue<T>();
    if (flag == 0) {
        if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                gmYOffset1 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) +
                             batchIdx * this->outBatchStride + ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset1], transposeData[i * COPY_ROWS_AND_COLS], leftCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset3 = this->outWidth * (i + transBlkIdx * this->ubFactorElement) +
                             batchIdx * this->outBatchStride + ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset3], transposeData[i * COPY_ROWS_AND_COLS], leftCopyParams);
            }
        }

    } else {
        if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                gmYOffset2 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + 1 + i) -
                             (COPY_ROWS_AND_COLS - this->padRight) + batchIdx * this->outBatchStride +
                             ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset2], transposeData[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset4 = this->outWidth * (i + transBlkIdx * this->ubFactorElement + 1) -
                             (COPY_ROWS_AND_COLS - this->padRight) + batchIdx * this->outBatchStride +
                             ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset4], transposeData[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        }
    }
    transposeQue.FreeTensor(transposeData);
}

template <typename T>
__aicore__ inline void PadV4GradPadHW<T>::implTransposeAndCompute(const int64_t transCount, const int32_t flag)
{
    this->ImplTransposeAndComputeCommon(transCount, flag);
}

template <typename T>
__aicore__ inline void PadV4GradPadHW<T>::ComputeHGrad(const int32_t calCount, const int32_t flag)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    int64_t workspaceOffset1 = 0;
    int64_t workspaceOffset2 = 0;
    int64_t offset3 = 0;
    int64_t yOffset1 = 0;
    int64_t yOffset2 = 0;
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < this->padTop; i++) {
            Add(xLocal[(2 * this->padTop - i) * this->ubFactorElement], xLocal[i * this->ubFactorElement],
                xLocal[(2 * this->padTop - i) * this->ubFactorElement], calCount);
        }
        DataCopy(yLocal, xLocal[this->padTop * this->ubFactorElement],
                 (COPY_ROWS_AND_COLS - this->padTop) * this->ubFactorElement);
    } else {
        for (size_t i = 0; i < this->padBottom; i++) {
            Add(xLocal[(COPY_ROWS_AND_COLS - 2 * this->padBottom - 1 + i) * this->ubFactorElement],
                xLocal[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                xLocal[(COPY_ROWS_AND_COLS - 2 * this->padBottom - 1 + i) * this->ubFactorElement], calCount);
        }
        DataCopy(yLocal, xLocal, (COPY_ROWS_AND_COLS - this->padBottom) * this->ubFactorElement);
    }
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadHW<T>::Process()
{
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    uint32_t copyTimesOneRow = this->CeilDiv(this->width, this->ubFactorElement);
    uint32_t transTimesOneCol = this->CeilDiv(this->outHeight, this->ubFactorElement);
    uint32_t copyMidDataTimes = this->CeilDiv(this->width - 2 * COPY_ROWS_AND_COLS,
                                              COPY_ROWS_AND_COLS * this->ubFactorElement);
    this->ProcessHWPadLoopBody(copyTimesOneRow, transTimesOneCol, copyMidDataTimes, MTE3ToMTE2Event);
}
#endif // _PAD_V4_GRAD_H_W_PAD_H_
