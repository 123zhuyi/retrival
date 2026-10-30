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
 * \file pad_v4_grad_h_w_bf16_pad.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_H_W_BF16_PAD_H_
#define _PAD_V4_GRAD_H_W_BF16_PAD_H_

#include "pad_v4_grad_base.h"

template <typename T>
class PadV4GradPadHWBf16 : public PadV4GradBase<T, PadV4GradPadHWBf16<T>> {
    using base = PadV4GradBase<T, PadV4GradPadHWBf16<T>>;
    friend base;

public:
    __aicore__ inline PadV4GradPadHWBf16(){};
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
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTenosr;
    LocalTensor<float> transposeData;
    event_t MTE3ToMTE2Event;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV4GradPadHWBf16<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
    pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
    pipe->InitBuffer(transposeBuf, this->ubFactorElement * sizeof(float) * COPY_ROWS_AND_COLS);
    pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * sizeof(float) * COPY_ROWS_AND_COLS);
}

template <typename T>
__aicore__ inline void PadV4GradPadHWBf16<T>::CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset,
                                                         const int32_t cycles, const int32_t transBlkIdx,
                                                         const int32_t flag)
{
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    int64_t gmYOffset3 = 0;
    int64_t gmYOffset4 = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->wPad1) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->wPad2) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                gmYOffset1 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) +
                             batchIdx * this->outBatchStride + ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset1], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset3 = this->outWidth * (i + transBlkIdx * this->ubFactorElement) +
                             batchIdx * this->outBatchStride + ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset3], yLocal[i * COPY_ROWS_AND_COLS], leftCopyParams);
            }
        }

    } else {
        if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                gmYOffset2 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + 1 + i) -
                             (COPY_ROWS_AND_COLS - this->wPad2) + batchIdx * this->outBatchStride +
                             ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset2], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset4 = this->outWidth * (i + transBlkIdx * this->ubFactorElement + 1) -
                             (COPY_ROWS_AND_COLS - this->wPad2) + batchIdx * this->outBatchStride +
                             ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset4], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadHWBf16<T>::implTransposeAndCompute(const int64_t transCount, const int32_t flag)
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
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * COPY_ROWS_AND_COLS);
    for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
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
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(float) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
        TransDataTo5HD<float>(xDstLocalList0, xSrcLocalList0, transDataParams);
    }
    if (flag == 0) {
        for (size_t i = 0; i < this->wPad1; i++) {
            Add(transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], transposeData[i * this->ubFactorElement],
                transposeData[(2 * this->wPad1 - i) * this->ubFactorElement], this->ubFactorElement);
        }
        DataCopy(transposeData, transposeData[this->wPad1 * this->ubFactorElement],
                 (COPY_ROWS_AND_COLS - this->wPad1) * this->ubFactorElement);

    } else {
        for (size_t i = 0; i < this->wPad2; i++) {
            Add(transposeData[(COPY_ROWS_AND_COLS - 2 * this->wPad2 - 1 + i) * this->ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                transposeData[(COPY_ROWS_AND_COLS - 2 * this->wPad2 - 1 + i) * this->ubFactorElement],
                this->ubFactorElement);
        }
    }
    for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList1[i] = (uint64_t)(transposeData[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
            xDstLocalList1[2 * i] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i +
                                                           time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                                   .GetPhyAddr());
            xDstLocalList1[2 * i +
                           1] = (uint64_t)(floatTenosr[COPY_ROWS_AND_COLS * i +
                                                       time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        transDataParams.repeatTimes = 1;
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, this->ubFactorElement * COPY_ROWS_AND_COLS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadHWBf16<T>::ComputeHGrad(const int32_t calCount, const int32_t flag)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    int64_t workspaceOffset1 = 0;
    int64_t workspaceOffset2 = 0;
    int64_t offset3 = 0;
    int64_t yOffset1 = 0;
    int64_t yOffset2 = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * COPY_ROWS_AND_COLS);
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < this->hPad1; i++) {
            Add(floatTenosr[(2 * this->hPad1 - i) * this->ubFactorElement], floatTenosr[i * this->ubFactorElement],
                floatTenosr[(2 * this->hPad1 - i) * this->ubFactorElement], calCount);
        }
        DataCopy(floatTenosr, floatTenosr[this->hPad1 * this->ubFactorElement],
                 (COPY_ROWS_AND_COLS - this->hPad1) * this->ubFactorElement);
    } else {
        for (size_t i = 0; i < this->hPad2; i++) {
            Add(floatTenosr[(COPY_ROWS_AND_COLS - 2 * this->hPad2 - 1 + i) * this->ubFactorElement],
                floatTenosr[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                floatTenosr[(COPY_ROWS_AND_COLS - 2 * this->hPad2 - 1 + i) * this->ubFactorElement], calCount);
        }
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, this->ubFactorElement * COPY_ROWS_AND_COLS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV4GradPadHWBf16<T>::Process()
{
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    uint32_t copyTimesOneRow = this->CeilDiv(this->width, this->ubFactorElement);
    uint32_t transTimesOneCol = this->CeilDiv(this->outHeight, this->ubFactorElement);
    uint32_t copyMidDataTimes = this->CeilDiv(this->width - 2 * COPY_ROWS_AND_COLS,
                                              COPY_ROWS_AND_COLS * this->ubFactorElement);
    floatTenosr = floatCastResBuf.Get<float>();
    transposeData = transposeBuf.Get<float>();
    this->ProcessHWPadLoopBody(copyTimesOneRow, transTimesOneCol, copyMidDataTimes, MTE3ToMTE2Event);
}
#endif // _PAD_V4_GRAD_H_W_BF16_PAD_H_
