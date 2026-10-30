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
 * \file pad_v3_grad_replicate_w.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_W_
#define _PAD_V3_GRAD_REPLICATE_W_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateW : public PadV3GradReplicateKernelBase<T, PadV3GradReplicateW<T>> {
    using base = PadV3GradReplicateKernelBase<T, PadV3GradReplicateW<T>>;
    friend base;

public:
    __aicore__ inline PadV3GradReplicateW(){};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR padding,
                                GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyGm2UBWhole(const int64_t offset, const int64_t copyCount);
    __aicore__ inline void CopyGm2UB(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void CopyWorkspace2Out(const int64_t offset1, const int64_t offset2, const int64_t copyCount);
    __aicore__ inline void CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2, const int64_t calCount);
    __aicore__ inline void CopyOut2Workspace(const int64_t offset, const int64_t calCount);
    __aicore__ inline void ComputeWGrad(const int32_t calCount);
    __aicore__ inline void ComputeWGradF16(const int32_t calCount);
    __aicore__ inline void ComputeWGradWhole(const int32_t calCount);
    __aicore__ inline void ComputeWGradWholeF16(const int32_t calCount);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TBuf<TPosition::VECCALC> floatCastResBuf;

    uint32_t wCalCount = 0;
    event_t eventId0;
    event_t eventId1;
    event_t eventId2;
    event_t eventId3;
    static constexpr bool isCastFp32 = AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::Init(const PadV3GradReplicateTilingData& __restrict tilingData,
                                                    GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace)
{
    base::Init(tilingData, x, padding, y, workspace);
    this->batchStride = this->width;
    this->outBatchStride = this->outWidth;
    wCalCount = tilingData.wCalCount;
    eventId0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    eventId3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    if constexpr (isCastFp32) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, wCalCount * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T));
        pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * sizeof(float));
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, wCalCount * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyGm2UBWhole(const int64_t offset, const int64_t copyCount)
{
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, this->perBlockCount) - copyCount), (T)0};

    DataCopyPad(dataLocal, this->mGmX[offset], copyParams, padParams);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyGm2UB(const int64_t offset1, const int64_t offset2,
                                                         const int64_t copyCount)
{
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, this->perBlockCount) - copyCount), (T)0};

    DataCopyPad(dataLocal[0], this->mGmX[offset1], copyParams, padParams);
    PipeBarrier<PIPE_MTE2>();
    DataCopyPad(dataLocal[copyCount], this->mGmX[offset2], copyParams, padParams);
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyWorkspace2Out(const int64_t offset1, const int64_t offset2,
                                                                 const int64_t copyCount)
{
    LocalTensor<T> dataLocal = yOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, this->perBlockCount) - copyCount), (T)0};
    WaitFlag<HardEvent::S_MTE2>(eventId0);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId1);
    DataCopyPad(dataLocal, this->mGmWorkspace[offset1], copyParams, padParams);
    SetFlag<HardEvent::S_MTE2>(eventId0);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::S_MTE3>(eventId2);
    DataCopyPad(this->mGmY[offset2], dataLocal, copyParams);
    SetFlag<HardEvent::S_MTE3>(eventId2);
    SetFlag<HardEvent::MTE3_MTE2>(eventId1);
    yOutQueue.FreeTensor(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2,
                                                               const int64_t calCount)
{
    LocalTensor<T> dstLocal = yOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(calCount, this->perBlockCount) - calCount), (T)0};
    WaitFlag<HardEvent::S_MTE2>(eventId0);
    DataCopyPad(dstLocal, this->mGmX[offset1], copyParams, padParams);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::S_MTE3>(eventId2);
    DataCopyPad(this->mGmY[offset2], dstLocal, copyParams);
    yOutQueue.FreeTensor(dstLocal);
    SetFlag<HardEvent::S_MTE2>(eventId0);
    SetFlag<HardEvent::S_MTE3>(eventId2);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::CopyOut2Workspace(const int64_t offset, const int64_t calCount)
{
    LocalTensor<T> yLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(CONST_VALUE_2 * calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmWorkspace[offset], yLocal, copyParams); // 拷贝到workspace，注意计算偏移
    yOutQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGrad(const int32_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    for (size_t i = 0; i < this->padLeft; i++) {
        val1 = xLocal.GetValue(i);             // index
        val2 = xLocal.GetValue(this->padLeft); // index 边缘轴
        if constexpr (AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            xLocal.SetValue(this->padLeft, tmp1);
        } else {
            xLocal.SetValue(this->padLeft, val1 + val2);
        }
    }
    for (size_t i = 0; i < this->padRight; i++) {
        val3 = xLocal.GetValue(CONST_VALUE_2 * calCount - 1 - i);              // index
        val4 = xLocal.GetValue(CONST_VALUE_2 * calCount - 1 - this->padRight); // index 边缘轴
        if constexpr (AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            xLocal.SetValue(CONST_VALUE_2 * calCount - 1 - this->padRight, tmp2);
        } else {
            xLocal.SetValue(CONST_VALUE_2 * calCount - 1 - this->padRight, val3 + val4);
        }
    }
    DataCopy(yLocal, xLocal, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGradF16(const int32_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatTensor = floatCastResBuf.Get<float>();
    float val1;
    float val2;
    float val3;
    float val4;
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, CONST_VALUE_2 * calCount);
    for (size_t i = 0; i < this->padLeft; i++) {
        val1 = floatTensor.GetValue(i);             // index
        val2 = floatTensor.GetValue(this->padLeft); // index 边缘轴
        floatTensor.SetValue(this->padLeft, val1 + val2);
    }
    for (size_t i = 0; i < this->padRight; i++) {
        val3 = floatTensor.GetValue(CONST_VALUE_2 * calCount - 1 - i);              // index
        val4 = floatTensor.GetValue(CONST_VALUE_2 * calCount - 1 - this->padRight); // index 边缘轴
        floatTensor.SetValue(CONST_VALUE_2 * calCount - 1 - this->padRight, val3 + val4);
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_ROUND, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
    floatCastResBuf.FreeTensor(floatTensor);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGradWhole(const int32_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    T val1;
    T val2;
    T val3;
    T val4;
    T tmp1;
    T tmp2;
    for (size_t i = 0; i < this->padLeft; i++) {
        val1 = xLocal.GetValue(i);             // index
        val2 = xLocal.GetValue(this->padLeft); // index 边缘轴
        if constexpr (AscendC::IsSameType<T, half>::value) {
            tmp1 = (T)((float)val1 + (float)val2);
            xLocal.SetValue(this->padLeft, tmp1);
        } else {
            xLocal.SetValue(this->padLeft, val1 + val2);
        }
    }
    for (size_t i = 0; i < this->padRight; i++) {
        val3 = xLocal.GetValue(this->width - 1 - i);              // index
        val4 = xLocal.GetValue(this->width - 1 - this->padRight); // index 边缘轴
        if constexpr (AscendC::IsSameType<T, half>::value) {
            tmp2 = (T)((float)val3 + (float)val4);
            xLocal.SetValue(this->width - 1 - this->padRight, tmp2);
        } else {
            xLocal.SetValue(this->width - 1 - this->padRight, val3 + val4);
        }
    }
    DataCopy(yLocal, xLocal, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::ComputeWGradWholeF16(const int32_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatTensor = floatCastResBuf.Get<float>();
    float val1;
    float val2;
    float val3;
    float val4;
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, CONST_VALUE_2 * calCount);
    for (size_t i = 0; i < this->padLeft; i++) {
        val1 = floatTensor.GetValue(i);             // index
        val2 = floatTensor.GetValue(this->padLeft); // index 边缘轴
        floatTensor.SetValue(this->padLeft, val1 + val2);
    }
    for (size_t i = 0; i < this->padRight; i++) {
        val3 = floatTensor.GetValue(this->width - 1 - i);              // index
        val4 = floatTensor.GetValue(this->width - 1 - this->padRight); // index 边缘轴
        floatTensor.SetValue(this->width - 1 - this->padRight, val3 + val4);
    }
    Cast(yLocal, floatTensor, RoundMode::CAST_ROUND, CONST_VALUE_2 * calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
    floatCastResBuf.FreeTensor(floatTensor);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateW<T>::Process()
{
    int64_t gmXOffset;
    int64_t gmXOffset1;
    int64_t gmXOffset2;
    int64_t gmXOffset3;
    int64_t gmYOffset;
    int64_t gmYOffset1;
    int64_t gmYOffset2;
    int64_t gmYOffset3;
    int64_t workspaceOffset;
    int64_t workspaceOffsetOut;
    int64_t workspaceOffset1;
    int64_t workspaceOffset2;
    int64_t workspaceOffset3;
    int64_t calCount = wCalCount;
    int64_t dataCountOneLine = this->width - CONST_VALUE_2 * calCount; // 中间body部分，不参与计算，复制即可
    // 对齐场景下，ubFactorElement应为16的倍数
    uint32_t copyTimesOneLine = CeilDiv(dataCountOneLine, this->ubFactorElement); // ubFactorElement：一行元素个数
    // 场景1：输入shape的W维度不超过2 * wCalCount，可以完全将整行搬到ub上，进行累加计算
    if (this->width <= CONST_VALUE_2 * wCalCount) {
        for (size_t loop = 0; loop < this->loopNC; loop++) {
            gmXOffset = loop * this->batchStride + this->ncOffset * this->batchStride;
            workspaceOffset = this->blockIdx * CONST_VALUE_2 * calCount; // workspace上的64空间
            SetFlag<HardEvent::S_MTE2>(eventId0);
            WaitFlag<HardEvent::S_MTE2>(eventId0);
            CopyGm2UBWhole(gmXOffset, this->width);
            // 左右两侧分别进行累加计算到edge
            if constexpr (isCastFp32) {
                ComputeWGradWholeF16(calCount);
            } else {
                ComputeWGradWhole(calCount);
            }
            // ub一共64列搬运到workspce上
            CopyOut2Workspace(workspaceOffset, calCount);
            // 计算需要搬出的worksapce偏移，需要搬出的起始位置，即左侧边缘行
            workspaceOffsetOut = this->padLeft + this->blockIdx * CONST_VALUE_2 * calCount;
            SetFlag<HardEvent::S_MTE2>(eventId0);
            // 计算搬出到gm上的偏移，outWidth左侧的起始位置，index0
            gmYOffset = loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
            SetFlag<HardEvent::S_MTE3>(eventId2);
            // workspace -> ub -> gm
            SetFlag<HardEvent::MTE3_MTE2>(eventId1);
            CopyWorkspace2Out(workspaceOffsetOut, gmYOffset, this->outWidth);
            WaitFlag<HardEvent::S_MTE2>(eventId0);
            WaitFlag<HardEvent::S_MTE3>(eventId2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId1);
        }
        return;
    }
    // 场景2：输入shape的W维度大于2 * wCalCount，需要分为padLeft、padRight和body三部分进行处理
    for (size_t loop = 0; loop < this->loopNC; loop++) {
        int64_t copyCount = this->ubFactorElement;
        gmXOffset1 = loop * this->batchStride + this->ncOffset * this->batchStride; // 包含padLeft的最左侧32列起始位置
        gmXOffset2 = (this->width - calCount) + loop * this->batchStride +
                     this->ncOffset * this->batchStride;              // 包含padRight的最右侧32列起始位置
        workspaceOffset1 = this->blockIdx * CONST_VALUE_2 * calCount; // workspace上的64空间
        // 左右两侧分别搬运到UB上
        SetFlag<HardEvent::S_MTE2>(eventId0);
        WaitFlag<HardEvent::S_MTE2>(eventId0);
        CopyGm2UB(gmXOffset1, gmXOffset2, calCount);
        // 左右两侧分别进行累加计算到edge
        if constexpr (isCastFp32) {
            ComputeWGradF16(calCount);
        } else {
            ComputeWGrad(calCount);
        }
        // 一共64列搬运到workspce上
        CopyOut2Workspace(workspaceOffset1, calCount);
        // 计算需要搬出的worksapce偏移
        // 左侧需要搬出的起始位置，即左侧边缘行~index31
        workspaceOffset2 = this->padLeft + this->blockIdx * CONST_VALUE_2 * calCount;
        // 右侧需要搬出的起始位置，即index32~右侧边缘行
        workspaceOffset3 = calCount + this->blockIdx * CONST_VALUE_2 * calCount;
        SetFlag<HardEvent::S_MTE2>(eventId0);
        // 计算搬出到gm上的偏移
        // outWidth左侧的起始位置，index0
        gmYOffset1 = loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
        // outWidth右侧的起始位置，(outWidth - calCount + padRight)
        gmYOffset2 = (this->outWidth - calCount + this->padRight) + loop * this->outBatchStride +
                     this->ncOffset * this->outBatchStride;
        SetFlag<HardEvent::S_MTE3>(eventId2);
        // 左侧workspace -> ub -> gm
        SetFlag<HardEvent::MTE3_MTE2>(eventId1);
        CopyWorkspace2Out(workspaceOffset2, gmYOffset1, calCount - this->padLeft);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId1);
        // 右侧workspace -> ub -> gm
        SetFlag<HardEvent::MTE3_MTE2>(eventId1);
        CopyWorkspace2Out(workspaceOffset3, gmYOffset2, calCount - this->padRight);
        WaitFlag<HardEvent::S_MTE2>(eventId0);
        WaitFlag<HardEvent::S_MTE3>(eventId2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId1);
        // 处理中间body，搬入ub再搬出到gm即可，不做计算
        for (size_t time = 0; time < copyTimesOneLine; time++) {
            if (time == copyTimesOneLine - 1) {
                copyCount = dataCountOneLine - (copyTimesOneLine - 1) * this->ubFactorElement; // 尾块搬运数量
            }
            // 输入body的起始位置
            gmXOffset3 = calCount + time * this->ubFactorElement + loop * this->batchStride +
                         this->ncOffset * this->batchStride;
            SetFlag<HardEvent::S_MTE2>(eventId0);
            // 输出body的起始位置
            gmYOffset3 = (calCount - this->padLeft) + time * this->ubFactorElement + loop * this->outBatchStride +
                         this->ncOffset * this->outBatchStride;
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::S_MTE3>(eventId2);
            CopyInAndOut2Gm(gmXOffset3, gmYOffset3, copyCount);
            WaitFlag<HardEvent::S_MTE2>(eventId0);
            WaitFlag<HardEvent::S_MTE3>(eventId2);
        }
    }
}
#endif // _PAD_V3_GRAD_REPLICATE_W_
