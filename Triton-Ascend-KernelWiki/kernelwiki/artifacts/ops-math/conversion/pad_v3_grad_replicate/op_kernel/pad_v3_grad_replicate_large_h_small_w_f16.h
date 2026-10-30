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
 * \file pad_v3_grad_replicate_large_h_small_w_f16.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_F16_H_
#define _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_F16_H_

#include "pad_v3_grad_replicate_large_h_small_w_common.h"

template <typename T>
class PadV3GradReplicateLargeHSmallWF16
    : public PadV3GradReplicateLargeHSmallWCommon<T, PadV3GradReplicateLargeHSmallWF16<T>> {
    using base = PadV3GradReplicateLargeHSmallWCommon<T, PadV3GradReplicateLargeHSmallWF16<T>>;
    friend base;

public:
    __aicore__ inline PadV3GradReplicateLargeHSmallWF16(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t transBlkIdx);
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTenosr;
    LocalTensor<float> transposeData;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallWF16<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    pipe->InitBuffer(this->xInQueue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(this->yOutQueue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(transposeBuf, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(float));
    pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(float));
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallWF16<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                                        const int32_t transBlkIdx)
{
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    DataCopyExtParams copyParams{1, (uint32_t)(this->outWidth * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = this->yOutQueue.template DeQue<T>();
    if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            gmYOffset1 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) +
                         batchIdx * this->outBatchStride + this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset1], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset2 = this->outWidth * (i + transBlkIdx * this->ubFactorElement) + batchIdx * this->outBatchStride +
                         this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset2], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }
    this->yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallWF16<T>::ImplTransposeAndCompute(const int64_t transCount)
{
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[TRANSDATA_BASE_H];
    uint64_t xDstLocalList0[TRANSDATA_BASE_H];
    uint64_t xSrcLocalList1[TRANSDATA_BASE_H];
    uint64_t xDstLocalList1[TRANSDATA_BASE_H];
    LocalTensor<T> xLocal = this->xInQueue.template DeQue<T>();
    LocalTensor<T> yLocal = this->yOutQueue.template AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * SMALL_WIDTH_LIMIT);
    for (size_t time = 0; time < SMALL_WIDTH_LIMIT / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTenosr[SMALL_WIDTH_LIMIT * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList0[CONST_VALUE_2 * i] = (uint64_t)(transposeData[i * this->ubFactorElement +
                                                                         FLOAT_BLOCK_NUM * this->ubFactorElement * time]
                                                               .GetPhyAddr());
            xDstLocalList0[CONST_VALUE_2 * i +
                           1] = (uint64_t)(transposeData[i * this->ubFactorElement +
                                                         FLOAT_BLOCK_NUM * this->ubFactorElement * time +
                                                         FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        transDataParams.repeatTimes = loopTimes;
        transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(float) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList0, xSrcLocalList0, transDataParams);
    }
    for (size_t i = 0; i < this->padLeft; i++) {
        Add(transposeData[this->padLeft * this->ubFactorElement], transposeData[i * this->ubFactorElement],
            transposeData[this->padLeft * this->ubFactorElement], this->ubFactorElement);
    }
    for (size_t i = 0; i < this->padRight; i++) {
        Add(transposeData[(this->width - 1 - this->padRight) * this->ubFactorElement],
            transposeData[(this->width - 1 - i) * this->ubFactorElement],
            transposeData[(this->width - 1 - this->padRight) * this->ubFactorElement], this->ubFactorElement);
    }
    DataCopy(transposeData, transposeData[this->padLeft * this->ubFactorElement],
             this->outWidth * this->ubFactorElement);

    for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList1[i] = (uint64_t)(transposeData[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList1[CONST_VALUE_2 * i] = (uint64_t)(floatTenosr[SMALL_WIDTH_LIMIT * i +
                                                                       time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM]
                                                               .GetPhyAddr());
            xDstLocalList1[CONST_VALUE_2 * i +
                           1] = (uint64_t)(floatTenosr[SMALL_WIDTH_LIMIT * i +
                                                       time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
        transDataParams.srcRepStride = CONST_VALUE_2 * this->ubFactorElement;
        transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, this->ubFactorElement * SMALL_WIDTH_LIMIT);
    this->xInQueue.FreeTensor(xLocal);
    this->yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallWF16<T>::ComputeHGrad(const int32_t calCount, const int32_t flag)
{
    LocalTensor<T> xLocal = this->xInQueue.template DeQue<T>();
    LocalTensor<T> yLocal = this->yOutQueue.template AllocTensor<T>();
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * SMALL_WIDTH_LIMIT);
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < this->padTop; i++) {
            Add(floatTenosr[this->padTop * SMALL_WIDTH_LIMIT], floatTenosr[i * SMALL_WIDTH_LIMIT],
                floatTenosr[this->padTop * SMALL_WIDTH_LIMIT], calCount);
        }
        Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, this->ubFactorElement * SMALL_WIDTH_LIMIT);
        DataCopy(yLocal, yLocal[this->padTop * SMALL_WIDTH_LIMIT],
                 (COPY_ROWS_AND_COLS - this->padTop) * SMALL_WIDTH_LIMIT);
    } else {
        for (size_t i = 0; i < this->padBottom; i++) {
            Add(floatTenosr[(COPY_ROWS_AND_COLS - 1 - this->padBottom) * SMALL_WIDTH_LIMIT],
                floatTenosr[(COPY_ROWS_AND_COLS - 1 - i) * SMALL_WIDTH_LIMIT],
                floatTenosr[(COPY_ROWS_AND_COLS - 1 - this->padBottom) * SMALL_WIDTH_LIMIT], calCount);
        }
        Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, this->ubFactorElement * SMALL_WIDTH_LIMIT);
    }
    this->xInQueue.FreeTensor(xLocal);
    this->yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateLargeHSmallWF16<T>::Process()
{
    int64_t calCount = this->width;
    uint32_t transTimesOneCol = CeilDiv(this->outHeight, this->ubFactorElement);
    floatTenosr = floatCastResBuf.Get<float>();
    transposeData = transposeBuf.Get<float>();
    this->MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    this->ProcessLargeHSmallWLoopBody(transTimesOneCol, calCount, this->ubFactorElement, this->MTE3ToMTE2Event);
}
#endif // _PAD_V3_GRAD_REPLICATE_LARGE_H_SMALL_W_F16_H_
