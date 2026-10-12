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
 * \file pad_v3_grad_replicate_h_w_mini.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_W_MINI_
#define _PAD_V3_GRAD_REPLICATE_H_W_MINI_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateHWMini : public PadV3GradReplicateKernelBase<T, PadV3GradReplicateHWMini<T>> {
public:
    __aicore__ inline PadV3GradReplicateHWMini(){};
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyIn(const int32_t loop);
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeF16();
    __aicore__ inline void CopyOut(const int32_t loop);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<T> transposeData;
    LocalTensor<float> floatTransposeData;
    LocalTensor<float> floatTenosr;

    static constexpr bool isCastFp32 = AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value;
};

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    if constexpr (isCastFp32) {
        pipe->InitBuffer(xInQueue, 1, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(yOutQueue, 1, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(transposeBuf, this->ubFactorElement * sizeof(float) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * sizeof(float) * MINI_SHAPE_MAX_ROWS);
    } else {
        pipe->InitBuffer(xInQueue, 1, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(yOutQueue, 1, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
        pipe->InitBuffer(transposeBuf, this->ubFactorElement * sizeof(T) * MINI_SHAPE_MAX_ROWS);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::CopyIn(const int32_t loop)
{
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(this->width, this->perBlockCount) - this->width),
                                      (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    int64_t offset = 0;
    for (size_t i = 0; i < this->height; i++) {
        offset = i * this->width + loop * this->batchStride + this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[i * this->ubFactorElement], this->mGmX[offset], copyParams, padParams);
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::Compute()
{
    uint64_t srcLocalList0[16];
    uint64_t dstLocalList0[16];
    uint64_t srcLocalList1[16];
    uint64_t dstLocalList1[16];
    size_t fp16TransTimes = CeilDiv(this->alignWidth, HALF_BLOCK_NUM);
    size_t fp32TransTimes = CeilDiv(this->alignWidth, FLOAT_BLOCK_NUM);
    size_t fp16TransBackTimes = CeilDiv(MINI_SHAPE_MAX_ROWS, HALF_BLOCK_NUM);
    size_t fp32TransBackTimes = CeilDiv(MINI_SHAPE_MAX_ROWS, FLOAT_BLOCK_NUM);
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;

    // compute h grad
    for (size_t i = 0; i < this->padTop; i++) {
        Add(xLocal[this->padTop * this->ubFactorElement], xLocal[i * this->ubFactorElement],
            xLocal[this->padTop * this->ubFactorElement], this->ubFactorElement);
    }
    for (size_t i = 0; i < this->padBottom; i++) {
        Add(xLocal[(this->height - 1 - this->padBottom) * this->ubFactorElement],
            xLocal[(this->height - 1 - i) * this->ubFactorElement],
            xLocal[(this->height - 1 - this->padBottom) * this->ubFactorElement], this->ubFactorElement);
    }
    DataCopy(xLocal, xLocal[this->padTop * this->ubFactorElement], this->outHeight * this->ubFactorElement);
    if constexpr (AscendC::IsSameType<T, half>::value) {
        for (size_t time = 0; time < fp16TransTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList0[i] = (uint64_t)(xLocal[i * this->ubFactorElement + time * HALF_BLOCK_NUM].GetPhyAddr());
                dstLocalList0[i] = (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS +
                                                            time * MINI_SHAPE_MAX_ROWS * HALF_BLOCK_NUM]
                                                  .GetPhyAddr());
            }
            transDataParams.repeatTimes = MINI_SHAPE_MAX_ROWS / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = this->ubFactorElement;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(dstLocalList0, srcLocalList0, transDataParams);
        }
        // compute w grad
        for (size_t i = 0; i < this->padLeft; i++) {
            Add(transposeData[this->padLeft * MINI_SHAPE_MAX_ROWS], transposeData[i * MINI_SHAPE_MAX_ROWS],
                transposeData[this->padLeft * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        for (size_t i = 0; i < this->padRight; i++) {
            Add(transposeData[(this->width - 1 - this->padRight) * MINI_SHAPE_MAX_ROWS],
                transposeData[(this->width - 1 - i) * MINI_SHAPE_MAX_ROWS],
                transposeData[(this->width - 1 - this->padRight) * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        DataCopy(transposeData, transposeData[this->padLeft * MINI_SHAPE_MAX_ROWS],
                 (this->width - this->padLeft - this->padRight) * MINI_SHAPE_MAX_ROWS);
        for (size_t time = 0; time < fp16TransBackTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList1[i] = (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS + time * HALF_BLOCK_NUM]
                                                  .GetPhyAddr());
                dstLocalList1[i] = (uint64_t)(xLocal[i * this->ubFactorElement +
                                                     time * this->ubFactorElement * HALF_BLOCK_NUM]
                                                  .GetPhyAddr());
            }
            transDataParams.repeatTimes = this->ubFactorElement / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = MINI_SHAPE_MAX_ROWS;
            transDataParams.dstRepStride = 1;
            TransDataTo5HD<T>(dstLocalList1, srcLocalList1, transDataParams);
        }
        DataCopy(yLocal, xLocal, MINI_SHAPE_MAX_ROWS * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    } else {
        for (size_t time = 0; time < fp32TransTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList0[i] = (uint64_t)(xLocal[i * this->ubFactorElement + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
                dstLocalList0[CONST_VALUE_2 *
                              i] = (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS +
                                                            time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM]
                                                  .GetPhyAddr());
                dstLocalList0[CONST_VALUE_2 * i +
                              1] = (uint64_t)(transposeData[i * MINI_SHAPE_MAX_ROWS +
                                                            time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM +
                                                            FLOAT_BLOCK_NUM]
                                                  .GetPhyAddr());
            }
            transDataParams.repeatTimes = MINI_SHAPE_MAX_ROWS / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = CONST_VALUE_2 * this->ubFactorElement;
            transDataParams.dstRepStride = CONST_VALUE_2;
            TransDataTo5HD<T>(dstLocalList0, srcLocalList0, transDataParams);
        }
        for (size_t i = 0; i < this->padLeft; i++) {
            Add(transposeData[this->padLeft * MINI_SHAPE_MAX_ROWS], transposeData[i * MINI_SHAPE_MAX_ROWS],
                transposeData[this->padLeft * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        for (size_t i = 0; i < this->padRight; i++) {
            Add(transposeData[(this->width - 1 - this->padRight) * MINI_SHAPE_MAX_ROWS],
                transposeData[(this->width - 1 - i) * MINI_SHAPE_MAX_ROWS],
                transposeData[(this->width - 1 - this->padRight) * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
        }
        DataCopy(transposeData, transposeData[this->padLeft * MINI_SHAPE_MAX_ROWS],
                 (this->width - this->padLeft - this->padRight) * MINI_SHAPE_MAX_ROWS);
        for (size_t time = 0; time < fp32TransBackTimes; time++) {
            for (int i = 0; i < HALF_BLOCK_NUM; i++) {
                srcLocalList1[i] = (uint64_t)(transposeData[MINI_SHAPE_MAX_ROWS * i + time * FLOAT_BLOCK_NUM]
                                                  .GetPhyAddr());
            }
            for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
                dstLocalList1[CONST_VALUE_2 * i] = (uint64_t)(xLocal[i * this->ubFactorElement +
                                                                     time * this->ubFactorElement * FLOAT_BLOCK_NUM]
                                                                  .GetPhyAddr()); // 每行首地址
                dstLocalList1[CONST_VALUE_2 * i +
                              1] = (uint64_t)(xLocal[i * this->ubFactorElement +
                                                     time * this->ubFactorElement * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                                  .GetPhyAddr()); // 每行首地址
            }
            transDataParams.repeatTimes = this->ubFactorElement / TRANSDATA_BASE_H;
            transDataParams.srcRepStride = CONST_VALUE_2 * MINI_SHAPE_MAX_ROWS;
            transDataParams.dstRepStride = CONST_VALUE_2;
            TransDataTo5HD<T>(dstLocalList1, srcLocalList1, transDataParams);
        }
        DataCopy(yLocal, xLocal, MINI_SHAPE_MAX_ROWS * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::ComputeF16()
{
    uint64_t srcLocalList0[16];
    uint64_t dstLocalList0[16];
    uint64_t srcLocalList1[16];
    uint64_t dstLocalList1[16];
    size_t fp32TransTimes = CeilDiv(this->alignWidth, FLOAT_BLOCK_NUM);
    size_t fp32TransBackTimes = CeilDiv(MINI_SHAPE_MAX_ROWS, FLOAT_BLOCK_NUM);
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = 1;
    transDataParams.dstRepStride = 0;
    transDataParams.srcRepStride = 0;
    Cast(floatTenosr, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * MINI_SHAPE_MAX_ROWS);
    // compute h grad
    for (size_t i = 0; i < this->padTop; i++) {
        Add(floatTenosr[this->padTop * this->ubFactorElement], floatTenosr[i * this->ubFactorElement],
            floatTenosr[this->padTop * this->ubFactorElement], this->ubFactorElement);
    }
    for (size_t i = 0; i < this->padBottom; i++) {
        Add(floatTenosr[(this->height - 1 - this->padBottom) * this->ubFactorElement],
            floatTenosr[(this->height - 1 - i) * this->ubFactorElement],
            floatTenosr[(this->height - 1 - this->padBottom) * this->ubFactorElement], this->ubFactorElement);
    }
    DataCopy(floatTenosr, floatTenosr[this->padTop * this->ubFactorElement], this->outHeight * this->ubFactorElement);

    for (size_t time = 0; time < fp32TransTimes; time++) {
        for (int i = 0; i < HALF_BLOCK_NUM; i++) {
            srcLocalList0[i] = (uint64_t)(floatTenosr[i * this->ubFactorElement + time * FLOAT_BLOCK_NUM].GetPhyAddr());
        }
        for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
            dstLocalList0[CONST_VALUE_2 *
                          i] = (uint64_t)(floatTransposeData[i * MINI_SHAPE_MAX_ROWS +
                                                             time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM]
                                              .GetPhyAddr()); // 每行首地址
            dstLocalList0[CONST_VALUE_2 * i +
                          1] = (uint64_t)(floatTransposeData[i * MINI_SHAPE_MAX_ROWS +
                                                             time * MINI_SHAPE_MAX_ROWS * FLOAT_BLOCK_NUM +
                                                             FLOAT_BLOCK_NUM]
                                              .GetPhyAddr()); // 每行首地址
        }
        transDataParams.repeatTimes = MINI_SHAPE_MAX_ROWS / TRANSDATA_BASE_H;
        transDataParams.srcRepStride = CONST_VALUE_2 * this->ubFactorElement;
        transDataParams.dstRepStride = CONST_VALUE_2;
        TransDataTo5HD<float>(dstLocalList0, srcLocalList0, transDataParams);
    }
    for (size_t i = 0; i < this->padLeft; i++) {
        Add(floatTransposeData[this->padLeft * MINI_SHAPE_MAX_ROWS], floatTransposeData[i * MINI_SHAPE_MAX_ROWS],
            floatTransposeData[this->padLeft * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
    }
    for (size_t i = 0; i < this->padRight; i++) {
        Add(floatTransposeData[(this->width - 1 - this->padRight) * MINI_SHAPE_MAX_ROWS],
            floatTransposeData[(this->width - 1 - i) * MINI_SHAPE_MAX_ROWS],
            floatTransposeData[(this->width - 1 - this->padRight) * MINI_SHAPE_MAX_ROWS], MINI_SHAPE_MAX_ROWS);
    }
    DataCopy(floatTransposeData, floatTransposeData[this->padLeft * MINI_SHAPE_MAX_ROWS],
             (this->width - this->padLeft - this->padRight) * MINI_SHAPE_MAX_ROWS);

    for (size_t time = 0; time < fp32TransBackTimes; time++) {
        for (int i = 0; i < HALF_BLOCK_NUM; i++) {
            srcLocalList1[i] = (uint64_t)(floatTransposeData[MINI_SHAPE_MAX_ROWS * i + time * FLOAT_BLOCK_NUM]
                                              .GetPhyAddr());
        }
        for (int i = 0; i < FLOAT_BLOCK_NUM; i++) {
            dstLocalList1[CONST_VALUE_2 * i] = (uint64_t)(floatTenosr[i * this->ubFactorElement +
                                                                      time * this->ubFactorElement * FLOAT_BLOCK_NUM]
                                                              .GetPhyAddr());
            dstLocalList1[CONST_VALUE_2 * i +
                          1] = (uint64_t)(floatTenosr[i * this->ubFactorElement +
                                                      time * this->ubFactorElement * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                              .GetPhyAddr());
        }
        transDataParams.repeatTimes = this->ubFactorElement / TRANSDATA_BASE_H;
        transDataParams.srcRepStride = CONST_VALUE_2 * MINI_SHAPE_MAX_ROWS;
        transDataParams.dstRepStride = CONST_VALUE_2;
        TransDataTo5HD<float>(dstLocalList1, srcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTenosr, RoundMode::CAST_RINT, this->ubFactorElement * MINI_SHAPE_MAX_ROWS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::CopyOut(const int32_t loop)
{
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(this->outWidth * sizeof(T)), 0, 0, 0};
    int64_t offset = 0;
    for (size_t i = 0; i < this->outHeight; i++) {
        offset = i * this->outWidth + loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
        DataCopyPad(this->mGmY[offset], yLocal[i * this->ubFactorElement], copyParams);
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWMini<T>::Process()
{
    if constexpr (isCastFp32) {
        floatTransposeData = transposeBuf.Get<float>();
        floatTenosr = floatCastResBuf.Get<float>();
    } else {
        transposeData = transposeBuf.Get<T>();
    }
    for (size_t loop = 0; loop < this->loopNC; loop++) {
        CopyIn(loop);
        if constexpr (isCastFp32) {
            ComputeF16();
        } else {
            Compute();
        }
        CopyOut(loop);
    }
}
#endif // _PAD_V3_GRAD_REPLICATE_H_W_MINI_
