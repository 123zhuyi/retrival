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
 * \file pad_v4_grad_large_h_small_w_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_LARGE_H_SMALL_W_PAD_H_
#define _PAD_V4_GRAD_LARGE_H_SMALL_W_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradLargeHSmallW : public PadV4GradBase<T, PadV4GradLargeHSmallW<T>> {
    using base = PadV4GradBase<T, PadV4GradLargeHSmallW<T>>;
    friend base;

public:
    __aicore__ inline PadV4GradLargeHSmallW(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int32_t cycles, const int32_t transBlkIdx);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void implTransposeAndCompute(const int64_t transCount);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> yOutQueue;
    TQue<QuePosition::VECOUT, 1> transposeQue;

    event_t MTE3ToMTE2Event;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(yOutQueue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
    pipe->InitBuffer(transposeQue, 1, this->ubFactorElement * SMALL_WIDTH_LIMIT * sizeof(T));
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::CopyOut2Gm(const int32_t batchIdx, const int32_t cycles,
                                                            const int32_t transBlkIdx)
{
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    DataCopyExtParams copyParams{1, (uint32_t)(this->outWidth * sizeof(T)), 0, 0, 0};
    LocalTensor<T> transposeData = transposeQue.DeQue<T>();
    if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            gmYOffset1 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) +
                         batchIdx * this->outBatchStride + this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset1], transposeData[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < cycles; i++) {
            gmYOffset2 = this->outWidth * (i + transBlkIdx * this->ubFactorElement) + batchIdx * this->outBatchStride +
                         this->ncOffset * this->outBatchStride;
            DataCopyPad(this->mGmY[gmYOffset2], transposeData[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }

    transposeQue.FreeTensor(transposeData);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::implTransposeAndCompute(const int64_t transCount)
{
    uint32_t loopTimes = this->CeilDiv(transCount, TRANSDATA_BASE_H);
    uint64_t xSrcLocalList0[16];
    uint64_t xDstLocalList0[16];
    uint64_t xSrcLocalList1[16];
    uint64_t xDstLocalList1[16];
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> transposeData = transposeQue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    if constexpr (AscendC::IsSameType<T, half>::value) {
        for (size_t time = 0; time < SMALL_WIDTH_LIMIT / HALF_BLOCK_NUM; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + HALF_BLOCK_NUM * time].GetPhyAddr());
                xDstLocalList0[i] = (uint64_t)(transposeData[this->ubFactorElement * i +
                                                             time * this->ubFactorElement * 16]
                                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }

        for (size_t i = 0; i < this->padLeft; i++) {
            Add(transposeData[(2 * this->padLeft - i) * this->ubFactorElement],
                transposeData[i * this->ubFactorElement],
                transposeData[(2 * this->padLeft - i) * this->ubFactorElement], this->ubFactorElement);
        }
        for (size_t i = 0; i < this->padRight; i++) {
            Add(transposeData[(this->width - 2 * this->padRight - 1 + i) * this->ubFactorElement],
                transposeData[(this->width - 1 - i) * this->ubFactorElement],
                transposeData[(this->width - 2 * this->padRight - 1 + i) * this->ubFactorElement],
                this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->padLeft * this->ubFactorElement],
                 this->outWidth * this->ubFactorElement);
        for (size_t time = 0; time < loopTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] = (uint64_t)(transposeData[this->ubFactorElement * i + time * HALF_BLOCK_NUM]
                                                   .GetPhyAddr());
                xDstLocalList1[i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + time * HALF_BLOCK_NUM * SMALL_WIDTH_LIMIT]
                                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = this->ubFactorElement;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(transposeData, xLocal, SMALL_WIDTH_LIMIT * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        transposeQue.EnQue(transposeData);
    } else {
        for (size_t time = 0; time < SMALL_WIDTH_LIMIT / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList0[2 * i] = (uint64_t)(transposeData[i * this->ubFactorElement +
                                                                 FLOAT_BLOCK_NUM * this->ubFactorElement * time]
                                                       .GetPhyAddr());
                xDstLocalList0[2 * i + 1] = (uint64_t)(transposeData[i * this->ubFactorElement +
                                                                     FLOAT_BLOCK_NUM * this->ubFactorElement * time +
                                                                     FLOAT_BLOCK_NUM]
                                                           .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * SMALL_WIDTH_LIMIT * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }
        for (size_t i = 0; i < this->padLeft; i++) {
            Add(transposeData[(2 * this->padLeft - i) * this->ubFactorElement],
                transposeData[i * this->ubFactorElement],
                transposeData[(2 * this->padLeft - i) * this->ubFactorElement], this->ubFactorElement);
        }
        for (size_t i = 0; i < this->padRight; i++) {
            Add(transposeData[(this->width - 2 * this->padRight - 1 + i) * this->ubFactorElement],
                transposeData[(this->width - 1 - i) * this->ubFactorElement],
                transposeData[(this->width - 2 * this->padRight - 1 + i) * this->ubFactorElement],
                this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->padLeft * this->ubFactorElement],
                 this->outWidth * this->ubFactorElement);

        for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] = (uint64_t)(transposeData[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM]
                                                   .GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList1[2 * i] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i +
                                                          time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM]
                                                       .GetPhyAddr());
                xDstLocalList1[2 * i +
                               1] = (uint64_t)(xLocal[SMALL_WIDTH_LIMIT * i +
                                                      time * SMALL_WIDTH_LIMIT * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = SMALL_WIDTH_LIMIT / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = 2 * this->ubFactorElement;
            transDataParams.dstRepStride = TRANSDATA_BASE_H / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(transposeData, xLocal, SMALL_WIDTH_LIMIT * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        transposeQue.EnQue(transposeData);
    }
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::ComputeHGrad(const int32_t calCount, const int32_t flag)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < this->padTop; i++) {
            Add(xLocal[(2 * this->padTop - i) * SMALL_WIDTH_LIMIT], xLocal[i * SMALL_WIDTH_LIMIT],
                xLocal[(2 * this->padTop - i) * SMALL_WIDTH_LIMIT], calCount);
        }
        DataCopy(yLocal, xLocal[this->padTop * SMALL_WIDTH_LIMIT],
                 (COPY_ROWS_AND_COLS - this->padTop) * SMALL_WIDTH_LIMIT);
    } else {
        for (size_t i = 0; i < this->padBottom; i++) {
            Add(xLocal[(COPY_ROWS_AND_COLS - 2 * this->padBottom - 1 + i) * SMALL_WIDTH_LIMIT],
                xLocal[(COPY_ROWS_AND_COLS - 1 - i) * SMALL_WIDTH_LIMIT],
                xLocal[(COPY_ROWS_AND_COLS - 2 * this->padBottom - 1 + i) * SMALL_WIDTH_LIMIT], calCount);
        }
        DataCopy(yLocal, xLocal, (COPY_ROWS_AND_COLS - this->padBottom) * SMALL_WIDTH_LIMIT);
    }
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradLargeHSmallW<T>::Process()
{
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    int64_t calCount = this->width;
    uint32_t transTimesOneCol = this->CeilDiv(this->outHeight, this->ubFactorElement);
    this->ProcessLargeHSmallWPadLoopBody(calCount, transTimesOneCol, MTE3ToMTE2Event);
}
#endif // _PAD_V4_GRAD_LARGE_H_SMALL_W_PAD_H_
