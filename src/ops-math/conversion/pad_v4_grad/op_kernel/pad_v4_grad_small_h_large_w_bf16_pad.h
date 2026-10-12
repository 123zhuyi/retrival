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
 * \file pad_v4_grad_small_h_large_w_bf16_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_SMALL_H_LARGE_W_BF16_PAD_H_
#define _PAD_V4_GRAD_SMALL_H_LARGE_W_BF16_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradPadSamllHLargeWBf16 : public PadV4GradBase<T, PadV4GradPadSamllHLargeWBf16<T>> {
    using base = PadV4GradBase<T, PadV4GradPadSamllHLargeWBf16<T>>;
    friend base;

public:
    __aicore__ inline PadV4GradPadSamllHLargeWBf16(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount);
    __aicore__ inline void implTransposeAndCompute(const int64_t transCount, const int32_t flag);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTenosr;
    LocalTensor<float> transposeData;
    event_t MTE3ToMTE2Event;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, 1, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(T));
    pipe->InitBuffer(yOutQueue, 1, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(T));
    pipe->InitBuffer(transposeBuf, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(float));
    pipe->InitBuffer(floatCastResBuf, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(float));
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                                   const int32_t flag)
{
    int64_t gmYOffset = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padLeft) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padRight) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset = this->outWidth * i + batchIdx * this->outBatchStride + this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset = this->outWidth * (i + 1) - (COPY_ROWS_AND_COLS - this->padRight) +
                        batchIdx * this->outBatchStride + this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::implTransposeAndCompute(const int64_t transCount,
                                                                                const int32_t flag)
{
    uint32_t loopTimes = this->CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[16];
    uint64_t xDstLocalList0[16];
    uint64_t xSrcLocalList1[16];
    uint64_t xDstLocalList1[16];
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    this->Bf16TransDataForward(transDataParams, xSrcLocalList0, xDstLocalList0, floatTenosr, transposeData, loopTimes);
    this->Bf16PadReflectW(transposeData, flag);
    this->Bf16TransDataBackward(transDataParams, xSrcLocalList1, xDstLocalList1, floatTenosr, transposeData);
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::ComputeHGrad(const int32_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    // compute grad
    for (size_t i = 0; i < this->padTop; i++) {
        Add(floatTenosr[(2 * this->padTop - i) * this->ubFactorElement], floatTenosr[i * this->ubFactorElement],
            floatTenosr[(2 * this->padTop - i) * this->ubFactorElement], calCount);
    }
    for (size_t i = 0; i < this->padBottom; i++) {
        Add(floatTenosr[(this->height - 2 * this->padBottom - 1 + i) * this->ubFactorElement],
            floatTenosr[(this->height - 1 - i) * this->ubFactorElement],
            floatTenosr[(this->height - 2 * this->padBottom - 1 + i) * this->ubFactorElement], calCount);
    }
    Cast(yLocal, floatTenosr[this->padTop * this->ubFactorElement], RoundMode::CAST_RINT,
         this->outHeight * this->ubFactorElement);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadSamllHLargeWBf16<T>::Process()
{
    uint32_t copyTimesOneRow = this->CeilDiv(this->width, this->ubFactorElement);
    uint32_t copyMidDataTimes = this->CeilDiv(this->width - 2 * COPY_ROWS_AND_COLS,
                                              SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    floatTenosr = floatCastResBuf.Get<float>();
    transposeData = transposeBuf.Get<float>();
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    this->ProcessSmallHLargeWPadLoopBody(copyTimesOneRow, copyMidDataTimes, MTE3ToMTE2Event);
}
#endif // _PAD_V4_GRAD_SMALL_H_LARGE_W_BF16_PAD_H_
