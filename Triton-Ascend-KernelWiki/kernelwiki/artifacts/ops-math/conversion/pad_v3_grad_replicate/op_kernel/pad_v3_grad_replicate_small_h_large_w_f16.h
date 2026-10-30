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
 * \file pad_v3_grad_replicate_small_h_large_w_f16.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_F16_H_
#define _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_F16_H_

#include "pad_v3_grad_replicate_small_h_large_w_common.h"

template <typename T>
class PadV3GradReplicateSmallHLargeWF16
    : public PadV3GradReplicateSmallHLargeWCommon<T, PadV3GradReplicateSmallHLargeWF16<T>> {
    using base = PadV3GradReplicateSmallHLargeWCommon<T, PadV3GradReplicateSmallHLargeWF16<T>>;
    friend base;

public:
    __aicore__ inline PadV3GradReplicateSmallHLargeWF16(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount);
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount, const int32_t flag);
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
__aicore__ inline void PadV3GradReplicateSmallHLargeWF16<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    pipe->InitBuffer(this->xInQueue, 1, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(T));
    pipe->InitBuffer(this->yOutQueue, 1, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(T));
    pipe->InitBuffer(transposeBuf, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(float));
    pipe->InitBuffer(floatCastResBuf, SMALL_HEIGHT_LIMIT * this->ubFactorElement * sizeof(float));
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWF16<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                                        const int32_t flag)
{
    int64_t gmYOffset = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padLeft) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padRight) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = this->yOutQueue.template DeQue<T>();
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
    this->yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWF16<T>::ImplTransposeAndCompute(const int64_t transCount,
                                                                                     const int32_t flag)
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
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
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
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(float) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList0, xSrcLocalList0, transDataParams);
    }
    if (flag == 0) {
        for (size_t i = 0; i < this->padLeft; i++) {
            Add(transposeData[this->padLeft * this->ubFactorElement], transposeData[i * this->ubFactorElement],
                transposeData[this->padLeft * this->ubFactorElement], this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->padLeft * this->ubFactorElement],
                 (COPY_ROWS_AND_COLS - this->padLeft) * this->ubFactorElement);

    } else {
        for (size_t i = 0; i < this->padRight; i++) {
            Add(transposeData[(COPY_ROWS_AND_COLS - 1 - this->padRight) * this->ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - this->padRight) * this->ubFactorElement],
                this->ubFactorElement);
        }
    }
    for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList1[i] = (uint64_t)(transposeData[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList1[CONST_VALUE_2 * i] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i +
                                                                       time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                                               .GetPhyAddr());
            xDstLocalList1[CONST_VALUE_2 * i +
                           1] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i +
                                                       time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        transDataParams.repeatTimes = 1;
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    this->xInQueue.FreeTensor(xLocal);
    this->yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWF16<T>::ComputeHGrad(const int32_t calCount)
{
    LocalTensor<T> xLocal = this->xInQueue.template DeQue<T>();
    LocalTensor<T> yLocal = this->yOutQueue.template AllocTensor<T>();
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    // Compute grad
    for (size_t i = 0; i < this->padTop; i++) {
        Add(floatTenosr[this->padTop * this->ubFactorElement], floatTenosr[i * this->ubFactorElement],
            floatTenosr[this->padTop * this->ubFactorElement], calCount);
    }
    for (size_t i = 0; i < this->padBottom; i++) {
        Add(floatTenosr[(this->height - 1 - this->padBottom) * this->ubFactorElement],
            floatTenosr[(this->height - 1 - i) * this->ubFactorElement],
            floatTenosr[(this->height - 1 - this->padBottom) * this->ubFactorElement], calCount);
    }
    Cast(yLocal, floatTenosr[this->padTop * this->ubFactorElement], RoundMode::CAST_RINT,
         this->outHeight * this->ubFactorElement);
    this->xInQueue.FreeTensor(xLocal);
    this->yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateSmallHLargeWF16<T>::Process()
{
    uint32_t copyTimesOneRow = CeilDiv(this->width, this->ubFactorElement);
    uint32_t copyMidDataTimes = CeilDiv(this->width - CONST_VALUE_2 * COPY_ROWS_AND_COLS,
                                        SMALL_HEIGHT_LIMIT * this->ubFactorElement);
    floatTenosr = floatCastResBuf.Get<float>();
    transposeData = transposeBuf.Get<float>();
    this->MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    this->ProcessSmallHLargeWLoopBody(copyTimesOneRow, copyMidDataTimes, this->MTE3ToMTE2Event);
}
#endif // _PAD_V3_GRAD_REPLICATE_SMALL_H_LARGE_W_F16_H_
