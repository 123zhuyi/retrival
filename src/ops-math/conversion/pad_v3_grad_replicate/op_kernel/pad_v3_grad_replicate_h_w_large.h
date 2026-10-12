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
 * \file pad_v3_grad_replicate_h_w_large.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_W_LARGE_
#define _PAD_V3_GRAD_REPLICATE_H_W_LARGE_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateHWLarge : public PadV3GradReplicateKernelBase<T, PadV3GradReplicateHWLarge<T>> {
    using base = PadV3GradReplicateKernelBase<T, PadV3GradReplicateHWLarge<T>>;
    friend base;

public:
    __aicore__ inline PadV3GradReplicateHWLarge(){};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR padding,
                                GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx,
                                     const int64_t ncOffset, const int32_t flag);
    __aicore__ inline void CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset, const int32_t cycles,
                                      const int32_t transBlkIdx, const int32_t flag);
    __aicore__ inline void CopyGmAndWorkspace2UB1(const int32_t batchIdx, const int64_t copyCount,
                                                  const int64_t ncOffset, const int32_t flag);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx, const int64_t ncOffset,
                                                  const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount, const int32_t flag);
    __aicore__ inline void ComputeHGrad(const int32_t calCount, const int32_t flag);
    __aicore__ inline void ComputeHGradF16(const int32_t calCount, const int32_t flag);
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount, const int32_t flag);
    __aicore__ inline void ImplTransposeAndComputeF16(const int64_t transCount, const int32_t flag);
    __aicore__ inline void CopyIn(const int32_t copyCount, const int64_t workspaceOffset);
    __aicore__ inline void Compute(const int32_t copyCount);
    __aicore__ inline void CopyOut(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void CopyInFromGm(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void CopyInput2OutGm(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> transposeBuf;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTensor;
    LocalTensor<float> transposeData;

    event_t MTE3ToMTE2Event;
    static constexpr bool isCastFp32 = AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::Init(const PadV3GradReplicateTilingData& __restrict tilingData,
                                                          GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace)
{
    base::Init(tilingData, x, padding, y, workspace);
    MTE3ToMTE2Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    if constexpr (isCastFp32) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(transposeBuf, this->ubFactorElement * sizeof(float) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * sizeof(float) * COPY_ROWS_AND_COLS);
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * COPY_ROWS_AND_COLS);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount,
                                                               const int32_t batchIdx, const int64_t ncOffset,
                                                               const int32_t flag)
{
    int64_t gmXOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, this->perBlockCount) - copyCount), (T)0};
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = i * this->width + cycleIdx * this->ubFactorElement + batchIdx * this->batchStride +
                        ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * this->ubFactorElement], this->mGmX[gmXOffset], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS; i++) {
            gmXOffset = (this->height - COPY_ROWS_AND_COLS + i) * this->width + cycleIdx * this->ubFactorElement +
                        batchIdx * this->batchStride + ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * this->ubFactorElement], this->mGmX[gmXOffset], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyInFromGm(const int32_t copyCount, const int64_t offset)
{
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, this->perBlockCount) - copyCount), (T)0};
    DataCopyPad(xLocal, this->mGmX[offset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyInput2OutGm(const int32_t copyCount, const int64_t offset)
{
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyOut2Gm(const int32_t batchIdx, const int64_t ncOffset,
                                                                const int32_t cycles, const int32_t transBlkIdx,
                                                                const int32_t flag)
{
    int64_t gmYOffset1 = 0;
    int64_t gmYOffset2 = 0;
    int64_t gmYOffset3 = 0;
    int64_t gmYOffset4 = 0;
    DataCopyExtParams leftCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padLeft) * sizeof(T)), 0, 0, 0};
    DataCopyExtParams rightCopyParams{1, (uint32_t)((COPY_ROWS_AND_COLS - this->padRight) * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                gmYOffset1 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) +
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
        if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                gmYOffset2 = this->outWidth * (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + 1 + i) -
                             (COPY_ROWS_AND_COLS - this->padRight) + batchIdx * this->outBatchStride +
                             ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset2], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                gmYOffset4 = this->outWidth * (i + transBlkIdx * this->ubFactorElement + 1) -
                             (COPY_ROWS_AND_COLS - this->padRight) + batchIdx * this->outBatchStride +
                             ncOffset * this->outBatchStride;
                DataCopyPad(this->mGmY[gmYOffset4], yLocal[i * COPY_ROWS_AND_COLS], rightCopyParams);
            }
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyGmAndWorkspace2UB1(const int32_t batchIdx,
                                                                            const int64_t copyCount,
                                                                            const int64_t ncOffset, const int32_t flag)
{
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    if (flag == 0) { // 左侧COPY_ROWS_AND_COLS列搬入到ub
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
            workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }
        for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
            xGmOffset = i * this->width + batchIdx * this->batchStride + ncOffset * this->batchStride;
            DataCopyPad(xLocal[(i - this->padTop) * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset], copyParams, padParams);
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                               this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) * COPY_ROWS_AND_COLS],
                        this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
        }
    } else { // 右侧COPY_ROWS_AND_COLS列搬入到ub
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
            workspaceOffset1 = (i + 1) * this->width - COPY_ROWS_AND_COLS + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }
        for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
            xGmOffset = (i + 1) * this->width - COPY_ROWS_AND_COLS + batchIdx * this->batchStride +
                        ncOffset * this->batchStride;
            DataCopyPad(xLocal[(i - this->padTop) * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset], copyParams, padParams);
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->padTop + 1) * this->width - COPY_ROWS_AND_COLS +
                               this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) * COPY_ROWS_AND_COLS],
                        this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyGmAndWorkspace2UB2(const int32_t transBlkIdx,
                                                                            const int32_t transTimes,
                                                                            const int32_t cycles,
                                                                            const int32_t batchIdx,
                                                                            const int64_t ncOffset, const int32_t flag)
{
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t workspaceOffset4;
    int64_t workspaceOffset5;
    int64_t workspaceOffset6;
    int64_t xGmOffset1;
    int64_t xGmOffset2;
    int64_t xGmOffset3;
    int64_t xGmOffset4;
    int64_t xGmOffset5;
    int64_t xGmOffset6;
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();

    if (flag == 0) {
        if (transBlkIdx == 0) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset1 = (i + this->padTop) * this->width + batchIdx * this->batchStride +
                             ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset1], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
                workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset1], copyParams,
                            padParams);
            }

        } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset2 = (this->ubFactorElement * transBlkIdx + this->padTop + i) * this->width +
                             batchIdx * this->batchStride + ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset2], copyParams, padParams);
            }
        } else if (transBlkIdx == transTimes - 1) {
            if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                    workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset2], copyParams,
                                padParams);
                }
            } else {
                for (size_t i = 0; i < cycles; i++) {
                    xGmOffset3 = (i + (transTimes - 1) * this->ubFactorElement + this->padTop) * this->width +
                                 batchIdx * this->batchStride + ncOffset * this->batchStride;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset3], copyParams, padParams);
                }
                PipeBarrier<PIPE_MTE2>();
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                    workspaceOffset3 = (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->padBottom) + i) * COPY_ROWS_AND_COLS],
                                this->mGmWorkspace[workspaceOffset3], copyParams, padParams);
                }
            }
        }
    } else {
        if (transBlkIdx == 0) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset4 = (i + this->padTop + 1) * this->width - 16 + batchIdx * this->batchStride +
                             ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset4], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
                workspaceOffset4 = (i + 1) * this->width - 16 + this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset4], copyParams,
                            padParams);
            }

        } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset5 = (this->ubFactorElement * transBlkIdx + this->padTop + i + 1) * this->width - 16 +
                             batchIdx * this->batchStride + ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset5], copyParams, padParams);
            }
        } else if (transBlkIdx == transTimes - 1) {
            if (cycles <= COPY_ROWS_AND_COLS - this->padBottom) {
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                    workspaceOffset5 = (i + COPY_ROWS_AND_COLS + 1 - this->padTop) * this->width - 16 +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset5], copyParams,
                                padParams);
                }
            } else {
                for (size_t i = 0; i < cycles; i++) {
                    xGmOffset6 = (i + (transTimes - 1) * this->ubFactorElement + this->padTop + 1) * this->width - 16 +
                                 batchIdx * this->batchStride + ncOffset * this->batchStride;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset6], copyParams, padParams);
                }
                PipeBarrier<PIPE_MTE2>();
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
                    workspaceOffset6 = (i + COPY_ROWS_AND_COLS + 1 - this->padTop) * this->width - 16 +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->padBottom) + i) * COPY_ROWS_AND_COLS],
                                this->mGmWorkspace[workspaceOffset6], copyParams, padParams);
                }
            }
        }
    }
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyOut2Workspace(const int32_t tIdx, const int64_t calCount,
                                                                       const int32_t flag)
{
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
            workspaceOffset = i * this->width + tIdx * this->ubFactorElement + this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            workspaceOffset = (COPY_ROWS_AND_COLS - this->padTop + i) * this->width + tIdx * this->ubFactorElement +
                              this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
        }
    }
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ImplTransposeAndCompute(const int64_t transCount,
                                                                             const int32_t flag)
{
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
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
    if constexpr (AscendC::IsSameType<T, half>::value) {
        for (int i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i].GetPhyAddr());
            xDstLocalList0[i] = (uint64_t)(yLocal[COPY_ROWS_AND_COLS * i * loopTimes].GetPhyAddr());
            xSrcLocalList1[i] = (uint64_t)(yLocal[COPY_ROWS_AND_COLS * i * loopTimes].GetPhyAddr());
            xDstLocalList1[i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i].GetPhyAddr());
        }
        transDataParams.repeatTimes = loopTimes;
        transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(T) / DATA_BLOCK_BYTES;
        transDataParams.dstRepStride = 1;
        TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        if (flag == 0) {
            for (size_t i = 0; i < this->padLeft; i++) {
                Add(yLocal[this->padLeft * this->ubFactorElement], yLocal[i * this->ubFactorElement],
                    yLocal[this->padLeft * this->ubFactorElement], this->ubFactorElement);
            }
            DataCopy(yLocal, yLocal[this->padLeft * this->ubFactorElement],
                     (COPY_ROWS_AND_COLS - this->padLeft) * this->ubFactorElement);

        } else {
            for (size_t i = 0; i < this->padRight; i++) {
                Add(yLocal[(COPY_ROWS_AND_COLS - 1 - this->padRight) * this->ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - this->padRight) * this->ubFactorElement], this->ubFactorElement);
            }
        }
        transDataParams.srcRepStride = 1;
        transDataParams.dstRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(T) / DATA_BLOCK_BYTES;
        TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        DataCopy(yLocal, xLocal, COPY_ROWS_AND_COLS * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    } else {
        for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList0[i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList0[CONST_VALUE_2 * i] = (uint64_t)(yLocal[i * this->ubFactorElement +
                                                                      FLOAT_BLOCK_NUM * this->ubFactorElement * time]
                                                                   .GetPhyAddr());
                xDstLocalList0[CONST_VALUE_2 * i +
                               1] = (uint64_t)(yLocal[i * this->ubFactorElement +
                                                      FLOAT_BLOCK_NUM * this->ubFactorElement * time + FLOAT_BLOCK_NUM]
                                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = loopTimes;
            transDataParams.srcRepStride = TRANSDATA_BASE_H * COPY_ROWS_AND_COLS * sizeof(T) / DATA_BLOCK_BYTES;
            transDataParams.dstRepStride = COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM;
            TransDataTo5HD<T>(xDstLocalList0, xSrcLocalList0, transDataParams);
        }
        if (flag == 0) {
            for (size_t i = 0; i < this->padLeft; i++) {
                Add(yLocal[this->padLeft * this->ubFactorElement], yLocal[i * this->ubFactorElement],
                    yLocal[this->padLeft * this->ubFactorElement], this->ubFactorElement);
            }
            DataCopy(yLocal, yLocal[this->padLeft * this->ubFactorElement],
                     (COPY_ROWS_AND_COLS - this->padLeft) * this->ubFactorElement);

        } else {
            for (size_t i = 0; i < this->padRight; i++) {
                Add(yLocal[(COPY_ROWS_AND_COLS - 1 - this->padRight) * this->ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                    yLocal[(COPY_ROWS_AND_COLS - 1 - this->padRight) * this->ubFactorElement], this->ubFactorElement);
            }
        }
        for (size_t time = 0; time < this->ubFactorElement / FLOAT_BLOCK_NUM; time++) {
            for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
                xSrcLocalList1[i] = (uint64_t)(yLocal[this->ubFactorElement * i + time * FLOAT_BLOCK_NUM].GetPhyAddr());
            }
            for (size_t i = 0; i < FLOAT_BLOCK_NUM; i++) {
                xDstLocalList1[CONST_VALUE_2 * i] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i +
                                                                      time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                                                   .GetPhyAddr());
                xDstLocalList1[CONST_VALUE_2 * i +
                               1] = (uint64_t)(xLocal[COPY_ROWS_AND_COLS * i +
                                                      time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                                   .GetPhyAddr());
            }
            transDataParams.repeatTimes = 1;
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
            TransDataTo5HD<T>(xDstLocalList1, xSrcLocalList1, transDataParams);
        }
        DataCopy(yLocal, xLocal, COPY_ROWS_AND_COLS * this->ubFactorElement);
        xInQueue.FreeTensor(xLocal);
        yOutQueue.EnQue(yLocal);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ImplTransposeAndComputeF16(const int64_t transCount,
                                                                                const int32_t flag)
{
    uint32_t loopTimes = CeilDiv(transCount, TRANSDATA_BASE_H);
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
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * COPY_ROWS_AND_COLS);
    for (size_t time = 0; time < COPY_ROWS_AND_COLS / FLOAT_BLOCK_NUM; time++) {
        for (size_t i = 0; i < HALF_BLOCK_NUM; i++) {
            xSrcLocalList0[i] = (uint64_t)(floatTensor[COPY_ROWS_AND_COLS * i + FLOAT_BLOCK_NUM * time].GetPhyAddr());
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
            xDstLocalList1[CONST_VALUE_2 * i] = (uint64_t)(floatTensor[COPY_ROWS_AND_COLS * i +
                                                                       time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM]
                                                               .GetPhyAddr());
            xDstLocalList1[CONST_VALUE_2 * i +
                           1] = (uint64_t)(floatTensor[COPY_ROWS_AND_COLS * i +
                                                       time * COPY_ROWS_AND_COLS * FLOAT_BLOCK_NUM + FLOAT_BLOCK_NUM]
                                               .GetPhyAddr());
        }
        transDataParams.repeatTimes = 1;
        transDataParams.srcRepStride = 0;
        transDataParams.dstRepStride = 0;
        TransDataTo5HD<float>(xDstLocalList1, xSrcLocalList1, transDataParams);
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_RINT, this->ubFactorElement * COPY_ROWS_AND_COLS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ComputeHGrad(const int32_t calCount, const int32_t flag)
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
            Add(xLocal[this->padTop * this->ubFactorElement], xLocal[i * this->ubFactorElement],
                xLocal[this->padTop * this->ubFactorElement], calCount);
        }
        DataCopy(yLocal, xLocal[this->padTop * this->ubFactorElement],
                 (COPY_ROWS_AND_COLS - this->padTop) * this->ubFactorElement);
    } else {
        for (size_t i = 0; i < this->padBottom; i++) {
            Add(xLocal[(COPY_ROWS_AND_COLS - 1 - this->padBottom) * this->ubFactorElement],
                xLocal[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                xLocal[(COPY_ROWS_AND_COLS - 1 - this->padBottom) * this->ubFactorElement], calCount);
        }
        DataCopy(yLocal, xLocal, (COPY_ROWS_AND_COLS - this->padBottom) * this->ubFactorElement);
    }
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::ComputeHGradF16(const int32_t calCount, const int32_t flag)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    int64_t workspaceOffset1 = 0;
    int64_t workspaceOffset2 = 0;
    int64_t offset3 = 0;
    int64_t yOffset1 = 0;
    int64_t yOffset2 = 0;
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, this->ubFactorElement * COPY_ROWS_AND_COLS);
    // compute grad
    if (flag == 0) {
        for (size_t i = 0; i < this->padTop; i++) { // i表示行数下标index
            Add(floatTensor[this->padTop * this->ubFactorElement], floatTensor[i * this->ubFactorElement],
                floatTensor[this->padTop * this->ubFactorElement], calCount);
        }
        DataCopy(floatTensor, floatTensor[this->padTop * this->ubFactorElement],
                 (COPY_ROWS_AND_COLS - this->padTop) * this->ubFactorElement);
    } else {
        for (size_t i = 0; i < this->padBottom; i++) { // i表示行数下标index
            Add(floatTensor[(COPY_ROWS_AND_COLS - 1 - this->padBottom) * this->ubFactorElement],
                floatTensor[(COPY_ROWS_AND_COLS - 1 - i) * this->ubFactorElement],
                floatTensor[(COPY_ROWS_AND_COLS - 1 - this->padBottom) * this->ubFactorElement], calCount);
        }
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_RINT, this->ubFactorElement * COPY_ROWS_AND_COLS);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyIn(const int32_t copyCount, const int64_t workspaceOffset)
{
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    LocalTensor<T> xLocal = xInQueue.AllocTensor<T>();
    DataCopyPad(xLocal, this->mGmWorkspace[workspaceOffset], copyParams, padParams);
    xInQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::Compute(const int32_t copyCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    uint32_t alignCopyCount = CeilAlign(copyCount, this->perBlockCount);
    DataCopy(yLocal, xLocal, alignCopyCount);
    xInQueue.FreeTensor(xLocal);
    yOutQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::CopyOut(const int32_t copyCount, const int64_t offset)
{
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateHWLarge<T>::Process()
{
    int64_t gmXOffset = 0;
    int64_t gmXOffset1;
    int64_t gmXOffset2;
    int64_t gmXOffset3;
    int64_t gmYOffset1;
    int64_t gmYOffset2;
    int64_t gmYOffset3;
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t calCount = this->ubFactorElement;
    int64_t cycleTimes = this->ubFactorElement;
    uint32_t copyCount1 = COPY_ROWS_AND_COLS * this->ubFactorElement;
    uint32_t copyCount2 = COPY_ROWS_AND_COLS * this->ubFactorElement;
    uint32_t copyCount = COPY_ROWS_AND_COLS * this->ubFactorElement;

    uint32_t copyTimesOneRow = CeilDiv(this->width, this->ubFactorElement);
    uint32_t transTimesOneCol = CeilDiv(this->outHeight, this->ubFactorElement);
    uint32_t copyMidDataTimes = CeilDiv(this->width - CONST_VALUE_2 * COPY_ROWS_AND_COLS,
                                        COPY_ROWS_AND_COLS * this->ubFactorElement);

    if constexpr (isCastFp32) {
        floatTensor = floatCastResBuf.Get<float>();
        transposeData = transposeBuf.Get<float>();
    }

    for (size_t loop = 0; loop < this->loopNC; loop++) {
        calCount = this->ubFactorElement;
        for (size_t time = 0; time < copyTimesOneRow; time++) {
            if (time == copyTimesOneRow - 1) {
                calCount = this->width - (copyTimesOneRow - 1) * this->ubFactorElement;
            }
            // 搬padTop16行
            CopyGm2UB(time, calCount, loop, this->ncOffset, 0);
            // padTop累加
            if constexpr (isCastFp32) {
                ComputeHGradF16(calCount, 0);
            } else {
                ComputeHGrad(calCount, 0);
            }
            // 累加轴搬到workspace上
            CopyOut2Workspace(time, calCount, 0);
            // 搬padBottom16行
            CopyGm2UB(time, calCount, loop, this->ncOffset, 1);
            // padBottom累加
            if constexpr (isCastFp32) {
                ComputeHGradF16(calCount, 1);
            } else {
                ComputeHGrad(calCount, 1);
            }
            // 累加轴搬到workspace上
            CopyOut2Workspace(time, calCount, 1);
        }
        SetFlag<HardEvent::MTE3_MTE2>(MTE3ToMTE2Event);
        WaitFlag<HardEvent::MTE3_MTE2>(MTE3ToMTE2Event);
        // workspace上padTop方向搬出到gm
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padTop; i++) {
            copyCount1 = COPY_ROWS_AND_COLS * this->ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount1 = this->width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                                 (copyMidDataTimes - 1) * this->ubFactorElement * COPY_ROWS_AND_COLS;
                }
                workspaceOffset1 = COPY_ROWS_AND_COLS + j * this->ubFactorElement * COPY_ROWS_AND_COLS +
                                   i * this->width + this->blockIdx * this->workspacePerCore;
                gmYOffset1 = COPY_ROWS_AND_COLS - this->padLeft + j * this->ubFactorElement * COPY_ROWS_AND_COLS +
                             i * this->outWidth + loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                CopyIn(copyCount1, workspaceOffset1);
                Compute(this->ubFactorElement * COPY_ROWS_AND_COLS);
                CopyOut(copyCount1, gmYOffset1);
            }
        }
        // workspace上padBottom方向搬出到gm
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->padBottom; i++) {
            copyCount2 = COPY_ROWS_AND_COLS * this->ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount2 = this->width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                                 (copyMidDataTimes - 1) * this->ubFactorElement * COPY_ROWS_AND_COLS;
                }
                workspaceOffset2 = COPY_ROWS_AND_COLS + j * this->ubFactorElement * COPY_ROWS_AND_COLS +
                                   (i + COPY_ROWS_AND_COLS - this->padTop) * this->width +
                                   this->blockIdx * this->workspacePerCore;
                gmYOffset2 = COPY_ROWS_AND_COLS - this->padLeft +
                             (this->outHeight - (COPY_ROWS_AND_COLS - this->padBottom) + i) * this->outWidth +
                             j * this->ubFactorElement * COPY_ROWS_AND_COLS + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                CopyIn(copyCount2, workspaceOffset2);
                Compute(copyCount);
                CopyOut(copyCount2, gmYOffset2);
            }
        }
        if (transTimesOneCol == 1) {
            CopyGmAndWorkspace2UB1(loop, COPY_ROWS_AND_COLS, this->ncOffset, 0);
            if constexpr (isCastFp32) {
                ImplTransposeAndComputeF16(this->ubFactorElement, 0);
            } else {
                ImplTransposeAndCompute(this->ubFactorElement, 0);
            }
            CopyOut2Gm(loop, this->ncOffset, this->outHeight, 0, 0);
            CopyGmAndWorkspace2UB1(loop, COPY_ROWS_AND_COLS, this->ncOffset, 1);
            if constexpr (isCastFp32) {
                ImplTransposeAndComputeF16(this->ubFactorElement, 1);
            } else {
                ImplTransposeAndCompute(this->ubFactorElement, 1);
            }
            CopyOut2Gm(loop, this->ncOffset, this->outHeight, 0, 1);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = this->ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = this->outHeight - (transTimesOneCol - 1) * this->ubFactorElement;
                }
                CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop, this->ncOffset, 0);
                if constexpr (isCastFp32) {
                    ImplTransposeAndComputeF16(this->ubFactorElement, 0);
                } else {
                    ImplTransposeAndCompute(this->ubFactorElement, 0);
                }
                CopyOut2Gm(loop, this->ncOffset, cycleTimes, transBlk, 0);
                CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop, this->ncOffset, 1);
                if constexpr (isCastFp32) {
                    ImplTransposeAndComputeF16(this->ubFactorElement, 1);
                } else {
                    ImplTransposeAndCompute(this->ubFactorElement, 1);
                }
                CopyOut2Gm(loop, this->ncOffset, cycleTimes, transBlk, 1);
            }
        }
        for (size_t rowIdx = COPY_ROWS_AND_COLS; rowIdx < this->height - COPY_ROWS_AND_COLS; rowIdx++) {
            copyCount = this->ubFactorElement * COPY_ROWS_AND_COLS;
            for (size_t i = 0; i < copyMidDataTimes; i++) {
                if (i == copyMidDataTimes - 1) {
                    copyCount = this->width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                                (copyMidDataTimes - 1) * this->ubFactorElement * COPY_ROWS_AND_COLS;
                }
                gmXOffset1 = COPY_ROWS_AND_COLS + rowIdx * this->width +
                             i * this->ubFactorElement * COPY_ROWS_AND_COLS + loop * this->batchStride +
                             this->ncOffset * this->batchStride;
                gmYOffset3 = (COPY_ROWS_AND_COLS - this->padLeft) + (rowIdx - this->padTop) * this->outWidth +
                             i * this->ubFactorElement * COPY_ROWS_AND_COLS + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                CopyInFromGm(copyCount, gmXOffset1);
                Compute(copyCount);
                CopyInput2OutGm(copyCount, gmYOffset3);
            }
        }
    }
}
#endif // _PAD_V3_GRAD_REPLICATE_H_W_LARGE_
