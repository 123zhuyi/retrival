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
 * \file pad_v3_grad_replicate_base.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_BASE_
#define _PAD_V3_GRAD_REPLICATE_BASE_

#include "kernel_operator.h"

constexpr int32_t INPUT_NUM = 2;
constexpr int32_t OUTPUT_NUM = 1;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PADDING_INPUT_INDEX = 2;
constexpr int32_t Y_OUTPUT_INDEX = 0;
constexpr int32_t BUFFER_APPLY_NUM = 2;
constexpr int32_t COPY_ROWS_AND_COLS = 16;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t COPY_LOOP = 16;
constexpr uint32_t CAL_COUNT = 32;
constexpr uint32_t FLOAT_BLOCK_NUM = 8;
constexpr uint32_t HALF_BLOCK_NUM = 16;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t TRANSDATA_BASE_H = 16;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t MINI_SHAPE_MAX_ROWS = 64;
constexpr uint32_t SMALL_WIDTH_LIMIT = 64;
constexpr uint32_t SMALL_HEIGHT_LIMIT = 64;

using namespace AscendC;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
};

template <typename T, typename DerivedT>
class PadV3GradReplicateKernelBase {
public:
    __aicore__ inline void Init(const PadV3GradReplicateTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR padding,
                                GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void ProcessLargeHSmallWLoopBody(uint32_t transTimesOneCol, int64_t calCount, int64_t cycleTimes,
                                                       event_t mte3ToMte2Event);
    __aicore__ inline void ProcessSmallHLargeWLoopBody(uint32_t copyTimesOneRow, uint32_t copyMidDataTimes,
                                                       event_t mte3ToMte2Event);

    // Wrapper methods for PadSmallHLargeWLoopBodyImpl CRTP dispatch
    __aicore__ inline void ComputeCopy(const int32_t copyCount) { static_cast<DerivedT*>(this)->Compute(copyCount); }
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount, const int32_t flag)
    {
        static_cast<DerivedT*>(this)->ImplTransposeAndCompute(transCount, flag);
    }
    __aicore__ inline void ImplTransposeAndCompute(const int64_t transCount)
    {
        static_cast<DerivedT*>(this)->ImplTransposeAndCompute(transCount);
    }
    __aicore__ inline uint32_t GetPadLeft() const { return this->padLeft; }
    __aicore__ inline uint32_t GetPadRight() const { return this->padRight; }
    __aicore__ inline uint32_t GetPadLeftMultiplier() const { return CONST_VALUE_2; }

protected:
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
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t blockNum = 0;
    uint32_t ubFactorElement = 0;
    uint32_t blockIdx = 0;
    uint32_t perBlockCount = 0;
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
__aicore__ inline void PadV3GradReplicateKernelBase<T, DerivedT>::Init(
    const PadV3GradReplicateTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace)
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
    padTop = tilingData.padTop;
    padBottom = tilingData.padBottom;
    padLeft = tilingData.padLeft;
    padRight = tilingData.padRight;
    blockNum = tilingData.blockNum;
    ubFactorElement = tilingData.ubFactorElement;
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

template <typename T, typename DerivedT>
__aicore__ inline void PadV3GradReplicateKernelBase<T, DerivedT>::ProcessLargeHSmallWLoopBody(uint32_t transTimesOneCol,
                                                                                              int64_t calCount,
                                                                                              int64_t cycleTimes,
                                                                                              event_t mte3ToMte2Event)
{
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
            static_cast<DerivedT*>(this)->ImplTransposeAndCompute(this->ubFactorElement);
            static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->outHeight, 0);
        } else if (transTimesOneCol > 1) {
            for (size_t transBlk = 0; transBlk < transTimesOneCol; transBlk++) {
                cycleTimes = this->ubFactorElement;
                if (transBlk == transTimesOneCol - 1) {
                    cycleTimes = this->outHeight - (transTimesOneCol - 1) * this->ubFactorElement;
                }
                static_cast<DerivedT*>(this)->CopyGmAndWorkspace2UB2(transBlk, transTimesOneCol, cycleTimes, loop);
                static_cast<DerivedT*>(this)->ImplTransposeAndCompute(this->ubFactorElement);
                static_cast<DerivedT*>(this)->CopyOut2Gm(loop, cycleTimes, transBlk);
            }
        }
    }
}

template <typename T, typename DerivedT>
__aicore__ inline void PadV3GradReplicateKernelBase<T, DerivedT>::ProcessSmallHLargeWLoopBody(uint32_t copyTimesOneRow,
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
                    copyCount = this->width - CONST_VALUE_2 * COPY_ROWS_AND_COLS -
                                (copyMidDataTimes - 1) * this->ubFactorElement * SMALL_HEIGHT_LIMIT;
                }
                workspaceOffset = COPY_ROWS_AND_COLS + j * this->ubFactorElement * SMALL_HEIGHT_LIMIT +
                                  i * this->width + this->blockIdx * this->workspacePerCore;
                gmYOffset = COPY_ROWS_AND_COLS - this->padLeft + j * this->ubFactorElement * SMALL_HEIGHT_LIMIT +
                            i * this->outWidth + loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                static_cast<DerivedT*>(this)->CopyIn(copyCount, workspaceOffset);
                static_cast<DerivedT*>(this)->Compute(this->ubFactorElement * SMALL_HEIGHT_LIMIT);
                static_cast<DerivedT*>(this)->CopyOut(copyCount, gmYOffset);
            }
        }
        static_cast<DerivedT*>(this)->CopyWs2UB(loop, COPY_ROWS_AND_COLS, 0);
        static_cast<DerivedT*>(this)->ImplTransposeAndCompute(this->ubFactorElement, 0);
        static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->outHeight, 0);
        static_cast<DerivedT*>(this)->CopyWs2UB(loop, COPY_ROWS_AND_COLS, 1);
        static_cast<DerivedT*>(this)->ImplTransposeAndCompute(this->ubFactorElement, 1);
        static_cast<DerivedT*>(this)->CopyOut2Gm(loop, this->outHeight, 1);
    }
}

#endif // _PAD_V3_GRAD_REPLICATE_BASE_
