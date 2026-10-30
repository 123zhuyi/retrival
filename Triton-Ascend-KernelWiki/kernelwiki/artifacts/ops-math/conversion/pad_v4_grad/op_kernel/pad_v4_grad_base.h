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
 * \file pad_v4_grad_base.h
 * \brief
 */
#ifndef _PAD_V4_GRAD_BASE_H_
#define _PAD_V4_GRAD_BASE_H_

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PADDING_INPUT_INDEX = 2;
constexpr int32_t Y_OUTPUT_INDEX = 0;
constexpr int32_t BUFFER_APPLY_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t COPY_LOOP = 16;
constexpr uint32_t HALF_BLOCK_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_NUM = 8;
constexpr uint32_t CAL_COUNT = 32;
constexpr uint32_t W_PAD_LOWER_LIMIT = 16;
constexpr uint32_t COPY_ROWS_AND_COLS = 16;
constexpr uint32_t MINI_SHAPE_MAX_ROWS = 128;
constexpr uint32_t TRANSDATA_BASE_H = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t SMALL_WIDTH_LIMIT = 128;
constexpr uint32_t SMALL_HEIGHT_LIMIT = 64;

using namespace AscendC;

template <typename T, typename DerivedT = void>
class PadV4GradBase {
public:
    __aicore__ inline PadV4GradBase(){};
    __aicore__ inline void Init(const PadV4GradTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR padding, GM_ADDR y,
                                GM_ADDR workspace);
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == (T2)0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        if (b == (T2)0) {
            return a;
        }
        return (a + b - 1) / b * b;
    };

    // Loop body methods for Process() extraction
    __aicore__ inline void ProcessHWPadLoopBody(uint32_t copyTimesOneRow, uint32_t transTimesOneCol,
                                                uint32_t copyMidDataTimes, event_t mte3ToMte2Event);
    __aicore__ inline void ProcessLargeHSmallWPadLoopBody(int64_t calCount, uint32_t transTimesOneCol,
                                                          event_t mte3ToMte2Event);
    __aicore__ inline void ProcessSmallHLargeWPadLoopBody(uint32_t copyTimesOneRow, uint32_t copyMidDataTimes,
                                                          event_t mte3ToMte2Event);

    // Copy methods shared between h_w_pad and h_w_bf16_pad (CRTP dispatch)
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx,
                                     const int64_t ncOffset, const int32_t flag);
    __aicore__ inline void CopyInFromGm(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void CopyInput2OutGm(const int32_t copyCount, const int64_t offset);
    __aicore__ inline void CopyGmAndWorkspace2UB1(const int32_t batchIdx, const int64_t copyCount,
                                                  const int64_t ncOffset, const int32_t flag);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx, const int64_t ncOffset,
                                                  const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount, const int32_t flag);
    __aicore__ inline void CopyIn(const int32_t copyCount, const int64_t workspaceOffset);
    __aicore__ inline void compute(const int32_t copyCount);
    __aicore__ inline void CopyOut(const int32_t copyCount, const int64_t offset);

    // Copy methods for large_h_small_w pair (CRTP dispatch)
    __aicore__ inline void CopyGm2UB(const int32_t batchIdx, const int32_t flag);
    __aicore__ inline void CopyGmAndWs2UB1(const int32_t batchIdx);
    __aicore__ inline void CopyGmAndWorkspace2UB2(const int32_t transBlkIdx, const int32_t transTimes,
                                                  const int32_t cycles, const int32_t batchIdx);
    __aicore__ inline void CopyOut2Ws(const int64_t calCount, const int32_t flag);

    // Copy methods for small_h_large_w pair (CRTP dispatch)
    __aicore__ inline void CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount, const int32_t batchIdx);
    __aicore__ inline void CopyWs2UB(const int32_t batchIdx, const int64_t copyCount, const int32_t flag);
    __aicore__ inline void CopyOut2Workspace(const int32_t tIdx, const int64_t calCount);

public:
    uint32_t batch = 0;
    uint32_t ncPerCore = 0;
    uint32_t tailNC = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    uint32_t hPad1 = 0;
    uint32_t hPad2 = 0;
    uint32_t wPad1 = 0;
    uint32_t wPad2 = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
    uint32_t wPadCopyCount = 0;
    uint64_t workspacePerCore = 0;
    int64_t batchStride = 0;
    int64_t outBatchStride = 0;
    uint32_t loopNC = 0;
    int64_t ncOffset = 0;

    GlobalTensor<T> mGmX;
    GlobalTensor<T> mGmY;
    GlobalTensor<T> mGmWorkspace;
};

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::Init(const PadV4GradTilingData& __restrict tilingData, GM_ADDR x,
                                                        GM_ADDR padding, GM_ADDR y, GM_ADDR workspace)
{
    batch = tilingData.batch;
    ncPerCore = tilingData.ncPerCore;
    tailNC = tilingData.tailNC;
    height = tilingData.height;
    width = tilingData.width;
    outHeight = tilingData.outHeight;
    outWidth = tilingData.outWidth;
    alignHeight = tilingData.alignHeight;
    alignWidth = tilingData.alignWidth;
    alignOutHeight = tilingData.alignOutHeight;
    alignOutWidth = tilingData.alignOutWidth;
    hPad1 = tilingData.hPad1;
    hPad2 = tilingData.hPad2;
    wPad1 = tilingData.wPad1;
    wPad2 = tilingData.wPad2;
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
    wPadCopyCount = tilingData.wPadCopyCount;
    workspacePerCore = tilingData.workspacePerCore / sizeof(T);
    batchStride = height * width;
    outBatchStride = outHeight * outWidth;
    blockIdx = GetBlockIdx();
    perBlockCount = BLOCK_BYTES / sizeof(T);
    if (blockIdx < tailNC) {
        loopNC = ncPerCore + 1;
        ncOffset = blockIdx * loopNC;
    } else {
        loopNC = ncPerCore;
        ncOffset = blockIdx * ncPerCore + tailNC;
    }
    mGmX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    mGmY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    mGmWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));
}

// ========== LoopBody methods (CRTP dispatch to derived class) ==========
template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::ProcessHWPadLoopBody(uint32_t copyTimesOneRow,
                                                                        uint32_t transTimesOneCol,
                                                                        uint32_t copyMidDataTimes,
                                                                        event_t mte3ToMte2Event)
{
    int64_t calCount = this->ubFactorElement;
    int64_t cycleTimes = this->ubFactorElement;
    uint32_t copyCount1 = COPY_ROWS_AND_COLS * this->ubFactorElement;
    uint32_t copyCount2 = COPY_ROWS_AND_COLS * this->ubFactorElement;
    uint32_t copyCount = COPY_ROWS_AND_COLS * this->ubFactorElement;
    int64_t workspaceOffset1, workspaceOffset2, gmYOffset1, gmYOffset2, gmXOffset1, gmYOffset3;

    for (size_t loop = 0; loop < this->loopNC; loop++) {
        calCount = this->ubFactorElement;
        for (size_t time = 0; time < copyTimesOneRow; time++) {
            if (time == copyTimesOneRow - 1) {
                calCount = this->width - (copyTimesOneRow - 1) * this->ubFactorElement;
            }
            static_cast<DerivedT*>(this)->CopyGm2UB(time, calCount, loop, this->ncOffset, 0);
            static_cast<DerivedT*>(this)->ComputeHGrad(calCount, 0);
            static_cast<DerivedT*>(this)->CopyOut2Workspace(time, calCount, 0);
            static_cast<DerivedT*>(this)->CopyGm2UB(time, calCount, loop, this->ncOffset, 1);
            static_cast<DerivedT*>(this)->ComputeHGrad(calCount, 1);
            static_cast<DerivedT*>(this)->CopyOut2Workspace(time, calCount, 1);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            copyCount1 = COPY_ROWS_AND_COLS * this->ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount1 = this->width - 2 * COPY_ROWS_AND_COLS -
                                 (copyMidDataTimes - 1) * this->ubFactorElement * COPY_ROWS_AND_COLS;
                }
                workspaceOffset1 = COPY_ROWS_AND_COLS + j * this->ubFactorElement * COPY_ROWS_AND_COLS +
                                   i * this->width + this->blockIdx * this->workspacePerCore;
                gmYOffset1 = COPY_ROWS_AND_COLS - this->wPad1 + j * this->ubFactorElement * COPY_ROWS_AND_COLS +
                             i * this->outWidth + loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                static_cast<DerivedT*>(this)->CopyIn(copyCount1, workspaceOffset1);
                static_cast<DerivedT*>(this)->compute(this->ubFactorElement * COPY_ROWS_AND_COLS);
                static_cast<DerivedT*>(this)->CopyOut(copyCount1, gmYOffset1);
            }
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            copyCount2 = COPY_ROWS_AND_COLS * this->ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount2 = this->width - 2 * COPY_ROWS_AND_COLS -
                                 (copyMidDataTimes - 1) * this->ubFactorElement * COPY_ROWS_AND_COLS;
                }
                workspaceOffset2 = COPY_ROWS_AND_COLS + j * this->ubFactorElement * COPY_ROWS_AND_COLS +
                                   (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                                   this->blockIdx * this->workspacePerCore;
                gmYOffset2 = COPY_ROWS_AND_COLS - this->wPad1 +
                             (this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) * this->outWidth +
                             j * this->ubFactorElement * COPY_ROWS_AND_COLS + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                static_cast<DerivedT*>(this)->CopyIn(copyCount2, workspaceOffset2);
                static_cast<DerivedT*>(this)->compute(copyCount2);
                static_cast<DerivedT*>(this)->CopyOut(copyCount2, gmYOffset2);
            }
        }
        if (transTimesOneCol == 1) {
            static_cast<DerivedT*>(this)->CopyGmAndWorkspace2UB1(loop, COPY_ROWS_AND_COLS, this->ncOffset, 0);
            static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement, 0);
            static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->ncOffset, this->outHeight, 0, 0);
            static_cast<DerivedT*>(this)->CopyGmAndWorkspace2UB1(loop, COPY_ROWS_AND_COLS, this->ncOffset, 1);
            static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement, 1);
            static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->ncOffset, this->outHeight, 0, 1);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = this->ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = this->outHeight - (transTimesOneCol - 1) * this->ubFactorElement;
                }
                static_cast<DerivedT*>(this)->CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop,
                                                                     this->ncOffset, 0);
                static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement, 0);
                static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->ncOffset, cycleTimes, transBlk, 0);
                static_cast<DerivedT*>(this)->CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop,
                                                                     this->ncOffset, 1);
                static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement, 1);
                static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->ncOffset, cycleTimes, transBlk, 1);
            }
        }
        for (size_t rowIdx = COPY_ROWS_AND_COLS; rowIdx < this->height - COPY_ROWS_AND_COLS; rowIdx++) {
            copyCount = this->ubFactorElement * COPY_ROWS_AND_COLS;
            for (size_t i = 0; i < copyMidDataTimes; i++) {
                if (i == copyMidDataTimes - 1) {
                    copyCount = this->width - 2 * COPY_ROWS_AND_COLS -
                                (copyMidDataTimes - 1) * this->ubFactorElement * COPY_ROWS_AND_COLS;
                }
                gmXOffset1 = COPY_ROWS_AND_COLS + rowIdx * this->width +
                             i * this->ubFactorElement * COPY_ROWS_AND_COLS + loop * this->batchStride +
                             this->ncOffset * this->batchStride;
                gmYOffset3 = (COPY_ROWS_AND_COLS - this->wPad1) + (rowIdx - this->hPad1) * this->outWidth +
                             i * this->ubFactorElement * COPY_ROWS_AND_COLS + loop * this->outBatchStride +
                             this->ncOffset * this->outBatchStride;
                static_cast<DerivedT*>(this)->CopyInFromGm(copyCount, gmXOffset1);
                static_cast<DerivedT*>(this)->compute(copyCount);
                static_cast<DerivedT*>(this)->CopyInput2OutGm(copyCount, gmYOffset3);
            }
        }
    }
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::ProcessLargeHSmallWPadLoopBody(int64_t calCount,
                                                                                  uint32_t transTimesOneCol,
                                                                                  event_t mte3ToMte2Event)
{
    int64_t cycleTimes = this->ubFactorElement;
    for (size_t loop = 0; loop < this->loopNC; loop++) {
        static_cast<DerivedT*>(this)->CopyGm2UB(loop, 0);
        static_cast<DerivedT*>(this)->ComputeHGrad(calCount, 0);
        static_cast<DerivedT*>(this)->CopyOut2Ws(calCount, 0);
        static_cast<DerivedT*>(this)->CopyGm2UB(loop, 1);
        static_cast<DerivedT*>(this)->ComputeHGrad(calCount, 1);
        static_cast<DerivedT*>(this)->CopyOut2Ws(calCount, 1);

        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        if (transTimesOneCol == 1) {
            static_cast<DerivedT*>(this)->CopyGmAndWs2UB1(loop);
            static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement);
            static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->outHeight, 0);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = this->ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = this->outHeight - (transTimesOneCol - 1) * this->ubFactorElement;
                }
                static_cast<DerivedT*>(this)->CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop);
                static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement);
                static_cast<DerivedT*>(this)->CopyOut2Gm(loop, cycleTimes, transBlk);
            }
        }
    }
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::ProcessSmallHLargeWPadLoopBody(uint32_t copyTimesOneRow,
                                                                                  uint32_t copyMidDataTimes,
                                                                                  event_t mte3ToMte2Event)
{
    int64_t gmYOffset;
    int64_t workspaceOffset;
    int64_t calCount = this->ubFactorElement;
    uint32_t copyCount = SMALL_HEIGHT_LIMIT * this->ubFactorElement;
    for (size_t loop = 0; loop < this->loopNC; loop++) {
        calCount = this->ubFactorElement;
        for (size_t time = 0; time < copyTimesOneRow; time++) {
            if (time == copyTimesOneRow - 1) {
                calCount = this->width - (copyTimesOneRow - 1) * this->ubFactorElement;
            }
            static_cast<DerivedT*>(this)->CopyGm2UB(time, calCount, loop);
            static_cast<DerivedT*>(this)->ComputeHGrad(calCount);
            static_cast<DerivedT*>(this)->CopyOut2Workspace(time, calCount);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
        for (size_t i = 0; i < this->outHeight; i++) {
            copyCount = SMALL_HEIGHT_LIMIT * this->ubFactorElement;
            for (size_t j = 0; j < copyMidDataTimes; j++) {
                if (j == copyMidDataTimes - 1) {
                    copyCount = this->width - 2 * COPY_ROWS_AND_COLS -
                                (copyMidDataTimes - 1) * this->ubFactorElement * SMALL_HEIGHT_LIMIT;
                }
                workspaceOffset = COPY_ROWS_AND_COLS + j * this->ubFactorElement * SMALL_HEIGHT_LIMIT +
                                  i * this->width + this->blockIdx * this->workspacePerCore;
                gmYOffset = COPY_ROWS_AND_COLS - this->wPad1 + j * this->ubFactorElement * SMALL_HEIGHT_LIMIT +
                            i * this->outWidth + loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                static_cast<DerivedT*>(this)->CopyIn(copyCount, workspaceOffset);
                static_cast<DerivedT*>(this)->compute(this->ubFactorElement * SMALL_HEIGHT_LIMIT);
                static_cast<DerivedT*>(this)->CopyOut(copyCount, gmYOffset);
            }
        }
        static_cast<DerivedT*>(this)->CopyWs2UB(loop, COPY_ROWS_AND_COLS, 0);
        static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement, 0);
        static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->outHeight, 0);
        static_cast<DerivedT*>(this)->CopyWs2UB(loop, COPY_ROWS_AND_COLS, 1);
        static_cast<DerivedT*>(this)->implTransposeAndCompute(this->ubFactorElement, 1);
        static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->outHeight, 1);
    }
}

// ========== Copy methods (CRTP dispatch to derived class queues) ==========
template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount,
                                                             const int32_t batchIdx, const int64_t ncOffset,
                                                             const int32_t flag)
{
    int64_t gmXOffset;
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
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
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyInFromGm(const int32_t copyCount, const int64_t offset)
{
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    DataCopyPad(xLocal, this->mGmX[offset], copyParams, padParams);
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyInput2OutGm(const int32_t copyCount, const int64_t offset)
{
    LocalTensor<T> yLocal = static_cast<DerivedT*>(this)->yOutQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    static_cast<DerivedT*>(this)->yOutQueue.FreeTensor(yLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGmAndWorkspace2UB1(const int32_t batchIdx,
                                                                          const int64_t copyCount,
                                                                          const int64_t ncOffset, const int32_t flag)
{
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }
        for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
            xGmOffset = i * this->width + batchIdx * this->batchStride + ncOffset * this->batchStride;
            DataCopyPad(xLocal[(i - this->hPad1) * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset], copyParams, padParams);
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                               this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) * COPY_ROWS_AND_COLS],
                        this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset1 = (i + 1) * this->width - COPY_ROWS_AND_COLS + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }
        for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
            xGmOffset = (i + 1) * this->width - COPY_ROWS_AND_COLS + batchIdx * this->batchStride +
                        ncOffset * this->batchStride;
            DataCopyPad(xLocal[(i - this->hPad1) * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset], copyParams, padParams);
        }
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->hPad1 + 1) * this->width - COPY_ROWS_AND_COLS +
                               this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) * COPY_ROWS_AND_COLS],
                        this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
        }
    }
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGmAndWorkspace2UB2(const int32_t transBlkIdx,
                                                                          const int32_t transTimes,
                                                                          const int32_t cycles, const int32_t batchIdx,
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
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();

    if (flag == 0) {
        if (transBlkIdx == 0) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset1 = (i + this->hPad1) * this->width + batchIdx * this->batchStride +
                             ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset1], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
            ;
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
                workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset1], copyParams,
                            padParams);
            }

        } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset2 = (this->ubFactorElement * transBlkIdx + this->hPad1 + i) * this->width +
                             batchIdx * this->batchStride + ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset2], copyParams, padParams);
            }
        } else if (transBlkIdx == transTimes - 1) {
            if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                    workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset2], copyParams,
                                padParams);
                }
            } else {
                for (size_t i = 0; i < cycles; i++) {
                    xGmOffset3 = (i + (transTimes - 1) * this->ubFactorElement + this->hPad1) * this->width +
                                 batchIdx * this->batchStride + ncOffset * this->batchStride;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset3], copyParams, padParams);
                }
                PipeBarrier<PIPE_MTE2>();
                ;
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                    workspaceOffset3 = (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->hPad2) + i) * COPY_ROWS_AND_COLS],
                                this->mGmWorkspace[workspaceOffset3], copyParams, padParams);
                }
            }
        }
    } else {
        if (transBlkIdx == 0) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset4 = (i + this->hPad1 + 1) * this->width - 16 + batchIdx * this->batchStride +
                             ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset4], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
            ;
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
                workspaceOffset4 = (i + 1) * this->width - 16 + this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset4], copyParams,
                            padParams);
            }

        } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
            for (size_t i = 0; i < this->ubFactorElement; i++) {
                xGmOffset5 = (this->ubFactorElement * transBlkIdx + this->hPad1 + i + 1) * this->width - 16 +
                             batchIdx * this->batchStride + ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset5], copyParams, padParams);
            }
        } else if (transBlkIdx == transTimes - 1) {
            if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                    workspaceOffset5 = (i + COPY_ROWS_AND_COLS + 1 - this->hPad1) * this->width - 16 +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmWorkspace[workspaceOffset5], copyParams,
                                padParams);
                }
            } else {
                for (size_t i = 0; i < cycles; i++) {
                    xGmOffset6 = (i + (transTimes - 1) * this->ubFactorElement + this->hPad1 + 1) * this->width - 16 +
                                 batchIdx * this->batchStride + ncOffset * this->batchStride;
                    DataCopyPad(xLocal[i * COPY_ROWS_AND_COLS], this->mGmX[xGmOffset6], copyParams, padParams);
                }
                PipeBarrier<PIPE_MTE2>();
                ;
                for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                    workspaceOffset6 = (i + COPY_ROWS_AND_COLS + 1 - this->hPad1) * this->width - 16 +
                                       this->blockIdx * this->workspacePerCore;
                    DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->hPad2) + i) * COPY_ROWS_AND_COLS],
                                this->mGmWorkspace[workspaceOffset6], copyParams, padParams);
                }
            }
        }
    }
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyOut2Workspace(const int32_t tIdx, const int64_t calCount,
                                                                     const int32_t flag)
{
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = static_cast<DerivedT*>(this)->yOutQueue.template DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset = i * this->width + tIdx * this->ubFactorElement + this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            workspaceOffset = (COPY_ROWS_AND_COLS - this->hPad1 + i) * this->width + tIdx * this->ubFactorElement +
                              this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
        }
    }
    static_cast<DerivedT*>(this)->yOutQueue.FreeTensor(yLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyIn(const int32_t copyCount, const int64_t workspaceOffset)
{
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
    DataCopyPad(xLocal, this->mGmWorkspace[workspaceOffset], copyParams, padParams);
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::compute(const int32_t copyCount)
{
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template DeQue<T>();
    LocalTensor<T> yLocal = static_cast<DerivedT*>(this)->yOutQueue.template AllocTensor<T>();
    uint32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopy(yLocal, xLocal, alignCopyCount);
    static_cast<DerivedT*>(this)->xInQueue.FreeTensor(xLocal);
    static_cast<DerivedT*>(this)->yOutQueue.EnQue(yLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyOut(const int32_t copyCount, const int64_t offset)
{
    LocalTensor<T> yLocal = static_cast<DerivedT*>(this)->yOutQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], yLocal, copyParams);
    static_cast<DerivedT*>(this)->yOutQueue.FreeTensor(yLocal);
}

// ========== Copy methods for large_h_small_w pair ==========
template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGm2UB(const int32_t batchIdx, const int32_t flag)
{
    int64_t gmXOffset;
    int32_t alignCopyCount = this->CeilAlign(this->width, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, (uint8_t)(alignCopyCount - this->width), (T)0};
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
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
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGmAndWs2UB1(const int32_t batchIdx)
{
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t xGmOffset;
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
        workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
        DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
    }
    for (size_t i = COPY_ROWS_AND_COLS; i < this->height - COPY_ROWS_AND_COLS; i++) {
        xGmOffset = i * this->width + batchIdx * this->batchStride + this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[(i - this->hPad1) * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset], copyParams, padParams);
    }
    for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
        workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                           this->blockIdx * this->workspacePerCore;
        DataCopyPad(xLocal[(this->outHeight - (COPY_ROWS_AND_COLS - this->hPad2) + i) * SMALL_WIDTH_LIMIT],
                    this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
    }
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGmAndWorkspace2UB2(const int32_t transBlkIdx,
                                                                          const int32_t transTimes,
                                                                          const int32_t cycles, const int32_t batchIdx)
{
    DataCopyExtParams copyParams{1, (uint32_t)(this->width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t xGmOffset1;
    int64_t xGmOffset2;
    int64_t xGmOffset3;
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();

    if (transBlkIdx == 0) {
        for (size_t i = 0; i < this->ubFactorElement; i++) {
            xGmOffset1 = (i + this->hPad1) * this->width + batchIdx * this->batchStride +
                         this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset1], copyParams, padParams);
        }
        PipeBarrier<PIPE_MTE2>();
        ;
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset1 = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset1], copyParams, padParams);
        }

    } else if (transBlkIdx > 0 && transBlkIdx < transTimes - 1) {
        for (size_t i = 0; i < this->ubFactorElement; i++) {
            xGmOffset2 = (this->ubFactorElement * transBlkIdx + this->hPad1 + i) * this->width +
                         batchIdx * this->batchStride + this->ncOffset * this->batchStride;
            DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset2], copyParams, padParams);
        }
    } else if (transBlkIdx == transTimes - 1) {
        if (cycles <= COPY_ROWS_AND_COLS - this->hPad2) {
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                workspaceOffset2 = (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                                   this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmWorkspace[workspaceOffset2], copyParams, padParams);
            }
        } else {
            for (size_t i = 0; i < cycles; i++) {
                xGmOffset3 = (i + (transTimes - 1) * this->ubFactorElement + this->hPad1) * this->width +
                             batchIdx * this->batchStride + this->ncOffset * this->batchStride;
                DataCopyPad(xLocal[i * SMALL_WIDTH_LIMIT], this->mGmX[xGmOffset3], copyParams, padParams);
            }
            PipeBarrier<PIPE_MTE2>();
            ;
            for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
                workspaceOffset3 = (i + COPY_ROWS_AND_COLS - this->hPad1) * this->width +
                                   this->blockIdx * this->workspacePerCore;
                DataCopyPad(xLocal[(cycles - (COPY_ROWS_AND_COLS - this->hPad2) + i) * SMALL_WIDTH_LIMIT],
                            this->mGmWorkspace[workspaceOffset3], copyParams, padParams);
            }
        }
    }

    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyOut2Ws(const int64_t calCount, const int32_t flag)
{
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = static_cast<DerivedT*>(this)->yOutQueue.template DeQue<T>();
    if (flag == 0) {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad1; i++) {
            workspaceOffset = i * this->width + this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    } else {
        for (size_t i = 0; i < COPY_ROWS_AND_COLS - this->hPad2; i++) {
            workspaceOffset = (COPY_ROWS_AND_COLS - this->hPad1 + i) * this->width +
                              this->blockIdx * this->workspacePerCore;
            DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * SMALL_WIDTH_LIMIT], copyParams);
        }
    }
    static_cast<DerivedT*>(this)->yOutQueue.FreeTensor(yLocal);
}

// ========== Copy methods for small_h_large_w pair ==========
template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyGm2UB(const int32_t cycleIdx, const int64_t copyCount,
                                                             const int32_t batchIdx)
{
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
    int32_t alignCopyCount = this->CeilAlign(copyCount, this->perBlockCount);
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(alignCopyCount - copyCount), (T)0};
    int64_t offset = 0;
    for (size_t i = 0; i < this->height; i++) {
        offset = i * this->width + cycleIdx * this->ubFactorElement + batchIdx * this->batchStride +
                 this->ncOffset * this->batchStride;
        DataCopyPad(xLocal[i * this->ubFactorElement], this->mGmX[offset], copyParams, padParams);
    }
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyWs2UB(const int32_t batchIdx, const int64_t copyCount,
                                                             const int32_t flag)
{
    DataCopyExtParams copyParams{1, (uint32_t)(COPY_ROWS_AND_COLS * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, (T)0};
    int64_t workspaceOffset;
    LocalTensor<T> xLocal = static_cast<DerivedT*>(this)->xInQueue.template AllocTensor<T>();
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
    static_cast<DerivedT*>(this)->xInQueue.EnQue(xLocal);
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV4GradBase<T, DerivedT>::CopyOut2Workspace(const int32_t tIdx, const int64_t calCount)
{
    int64_t workspaceOffset;
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    LocalTensor<T> yLocal = static_cast<DerivedT*>(this)->yOutQueue.template DeQue<T>();
    for (size_t i = 0; i < this->outHeight; i++) {
        workspaceOffset = i * this->width + tIdx * this->ubFactorElement + this->blockIdx * this->workspacePerCore;
        DataCopyPad(this->mGmWorkspace[workspaceOffset], yLocal[i * this->ubFactorElement], copyParams);
    }
    static_cast<DerivedT*>(this)->yOutQueue.FreeTensor(yLocal);
}

#endif // _PAD_V4_GRAD_BASE_H_
