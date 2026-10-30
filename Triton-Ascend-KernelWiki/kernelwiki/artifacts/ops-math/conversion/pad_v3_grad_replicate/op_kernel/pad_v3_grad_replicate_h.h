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
 * \file pad_v3_grad_replicate_h.h
 * \brief
 */
#ifndef _PAD_V3_GRAD_REPLICATE_H_
#define _PAD_V3_GRAD_REPLICATE_H_

#include "kernel_operator.h"
#include "pad_v3_grad_replicate_base.h"

using namespace AscendC;

template <typename T>
class PadV3GradReplicateH : public PadV3GradReplicateKernelBase<T, PadV3GradReplicateH<T>> {
    using base = PadV3GradReplicateKernelBase<T, PadV3GradReplicateH<T>>;
    friend base;

public:
    __aicore__ inline PadV3GradReplicateH(){};
    __aicore__ inline void Init(const PadV3GradReplicateTilingData& __restrict tilingData, GM_ADDR x, GM_ADDR padding,
                                GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void CopyFromGm2UB(const int64_t offset, const int64_t copyCount);
    __aicore__ inline void CopyOut2Gm(const int64_t offset, const int64_t calCount);
    __aicore__ inline void CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2, const int64_t calCount,
                                           const int32_t blkIdx);
    __aicore__ inline void ComputeHGrad(const int64_t calCount);
    __aicore__ inline void ComputeHGradF16(const int64_t calCount);
    __aicore__ inline void FloatCast2F16(const int64_t calCount);
    __aicore__ inline void Process();

private:
    TPipe* pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> floatQueue;
    TBuf<TPosition::VECCALC> floatCastResBuf;
    LocalTensor<float> floatTensor;

    event_t eventId0;
    event_t eventId1;
    static constexpr bool isCastFp32 = AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value;
};

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::Init(const PadV3GradReplicateTilingData& __restrict tilingData,
                                                    GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace)
{
    base::Init(tilingData, x, padding, y, workspace);
    this->outBatchStride = this->outHeight * this->width;
    eventId0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
}

// init used buffer
template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::InitBuffer(TPipe* inputPipe)
{
    pipe = inputPipe;
    if constexpr (isCastFp32) {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(floatQueue, BUFFER_NUM, this->ubFactorElement * sizeof(float));
        pipe->InitBuffer(floatCastResBuf, this->ubFactorElement * sizeof(float));
    } else {
        pipe->InitBuffer(xInQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * CONST_VALUE_2);
        pipe->InitBuffer(yOutQueue, BUFFER_NUM, this->ubFactorElement * sizeof(T) * CONST_VALUE_2);
    }
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::CopyFromGm2UB(const int64_t offset, const int64_t copyCount)
{
    LocalTensor<T> dataLocal = xInQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(copyCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(copyCount, this->perBlockCount) - copyCount), (T)0};

    DataCopyPad(dataLocal[0], this->mGmX[offset], copyParams, padParams);
    PipeBarrier<PIPE_MTE2>();
    xInQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::CopyOut2Gm(const int64_t offset, const int64_t calCount)
{
    LocalTensor<T> dstLocal = yOutQueue.DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->mGmY[offset], dstLocal, copyParams);
    yOutQueue.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::CopyInAndOut2Gm(const int64_t offset1, const int64_t offset2,
                                                               const int64_t calCount, const int32_t blkIdx)
{
    LocalTensor<T> dstLocal = yOutQueue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(calCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, (uint8_t)(CeilAlign(calCount, this->perBlockCount) - calCount), (T)0};
    WaitFlag<HardEvent::S_MTE2>(eventId0);
    DataCopyPad(dstLocal[blkIdx * this->ubFactorElement], this->mGmX[offset1], copyParams, padParams);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::MTE2_MTE3>(eventID);
    DataCopyPad(this->mGmY[offset2], dstLocal[blkIdx * this->ubFactorElement], copyParams);
    yOutQueue.FreeTensor(dstLocal);
    SetFlag<HardEvent::S_MTE2>(eventId0);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::ComputeHGrad(const int64_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    LocalTensor<T> yLocal;
    if (yOutQueue.HasTensorInQue()) {
        yLocal = yOutQueue.DeQue<T>();
    } else {
        yLocal = yOutQueue.AllocTensor<T>();
        T inputValue(0.0);
        Duplicate<T>(yLocal, inputValue, calCount);
    }
    PipeBarrier<PIPE_V>();
    Add(yLocal, yLocal, xLocal[0], calCount);
    yOutQueue.EnQue(yLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::ComputeHGradF16(const int64_t calCount)
{
    LocalTensor<T> xLocal = xInQueue.DeQue<T>();
    Cast(floatTensor, xLocal, RoundMode::CAST_NONE, this->ubFactorElement);
    LocalTensor<float> floatLocal;
    if (floatQueue.HasTensorInQue()) {
        floatLocal = floatQueue.DeQue<float>();
    } else {
        floatLocal = floatQueue.AllocTensor<float>();
        float inputValue(0.0);
        Duplicate<float>(floatLocal, inputValue, calCount);
    }
    PipeBarrier<PIPE_V>();
    Add(floatLocal, floatLocal, floatTensor, calCount);
    floatQueue.EnQue(floatLocal);
    xInQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::FloatCast2F16(const int64_t calCount)
{
    LocalTensor<T> yLocal = yOutQueue.AllocTensor<T>();
    LocalTensor<float> floatLocal = floatQueue.DeQue<float>();
    Cast(yLocal, floatLocal, RoundMode::CAST_RINT, calCount);
    yOutQueue.EnQue(yLocal);
    floatQueue.FreeTensor(floatLocal);
}

template <typename T>
__aicore__ inline void PadV3GradReplicateH<T>::Process()
{
    int64_t gmXOffset;
    int64_t gmXOffset1;
    int64_t gmXOffset2;
    int64_t gmXOffset3;
    int64_t gmYOffset;
    int64_t gmYOffset1;
    int64_t gmYOffset2;
    int64_t gmYOffset3;
    // 对齐场景下，ubFactorElement应为16的倍数
    uint32_t copyTimesOneLine = CeilDiv(this->width, this->ubFactorElement); // ubFactorElement：一行元素个数

    if constexpr (isCastFp32) {
        floatTensor = floatCastResBuf.Get<float>();
    }

    for (size_t loop = 0; loop < this->loopNC; loop++) {
        int64_t calCount = this->ubFactorElement;
        for (size_t time = 0; time < copyTimesOneLine; time++) {
            if (time == copyTimesOneLine - 1) {
                calCount = this->width - (copyTimesOneLine - 1) * this->ubFactorElement; // 尾块搬运数量
            }
            // 场景1：输出shape的H维度为1，padTop和padBottom累加的边缘行重叠，梯度要全部累加到outHeight上
            if (this->outHeight == 1) {
                for (size_t i = 0; i < this->height; i++) {
                    gmXOffset = i * this->width + time * this->ubFactorElement + loop * this->batchStride +
                                this->ncOffset * this->batchStride;
                    SetFlag<HardEvent::S_MTE2>(eventId0);
                    WaitFlag<HardEvent::S_MTE2>(eventId0);
                    CopyFromGm2UB(gmXOffset, calCount);
                    if constexpr (isCastFp32) {
                        ComputeHGradF16(calCount);
                    } else {
                        ComputeHGrad(calCount);
                    }
                }
                if constexpr (isCastFp32) {
                    FloatCast2F16(calCount);
                }
                gmYOffset = time * this->ubFactorElement + loop * this->outBatchStride +
                            this->ncOffset * this->outBatchStride;
                SetFlag<HardEvent::S_MTE3>(eventId1);
                WaitFlag<HardEvent::S_MTE3>(eventId1);
                CopyOut2Gm(gmYOffset, calCount);
                continue;
            }

            // 场景2：输出shape的H维度不为1，即padTop和padBottom累加的边缘行不重叠，分三部分处理：padTop、padBottom和body
            // 处理padTop,梯度累加到边缘行
            for (size_t i = 0; i <= this->padTop; i++) {
                // 搬一行，padTop行一直到边缘行，梯度累加
                gmXOffset1 = i * this->width + time * this->ubFactorElement + loop * this->batchStride +
                             this->ncOffset * this->batchStride;
                SetFlag<HardEvent::S_MTE2>(eventId0);
                WaitFlag<HardEvent::S_MTE2>(eventId0);
                CopyFromGm2UB(gmXOffset1, calCount);
                if constexpr (isCastFp32) {
                    ComputeHGradF16(calCount);
                } else {
                    ComputeHGrad(calCount);
                }
            }
            if constexpr (isCastFp32) {
                FloatCast2F16(calCount);
            }
            // padTop累加完成，输出到边缘首行
            gmYOffset1 = time * this->ubFactorElement + loop * this->outBatchStride +
                         this->ncOffset * this->outBatchStride;
            SetFlag<HardEvent::S_MTE3>(eventId1);
            WaitFlag<HardEvent::S_MTE3>(eventId1);
            CopyOut2Gm(gmYOffset1, calCount);

            // 处理padBottom，梯度累加到边缘行
            for (size_t i = 0; i <= this->padBottom; i++) {
                // 搬一行，padBottom行一直到边缘行，梯度累加
                gmXOffset2 = (this->height - 1 - i) * this->width + time * this->ubFactorElement +
                             loop * this->batchStride + this->ncOffset * this->batchStride;
                SetFlag<HardEvent::S_MTE2>(eventId0);
                WaitFlag<HardEvent::S_MTE2>(eventId0);
                CopyFromGm2UB(gmXOffset2, calCount);
                if constexpr (isCastFp32) {
                    ComputeHGradF16(calCount);
                } else {
                    ComputeHGrad(calCount);
                }
            }
            if constexpr (isCastFp32) {
                FloatCast2F16(calCount);
            }
            // padBottom累加完成，输出到边缘尾行
            gmYOffset2 = (this->outHeight - 1) * this->width + time * this->ubFactorElement +
                         loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
            SetFlag<HardEvent::S_MTE3>(eventId1);
            WaitFlag<HardEvent::S_MTE3>(eventId1);
            CopyOut2Gm(gmYOffset2, calCount);

            // 处理中间body，搬入ub再搬出到gm即可，不做计算
            for (size_t i = this->padTop + 1; i < this->height - 1 - this->padBottom; i++) {
                // 输入body的起始位置
                gmXOffset3 = i * this->width + time * this->ubFactorElement + loop * this->batchStride +
                             this->ncOffset * this->batchStride;
                // 输出body的起始位置
                gmYOffset3 = (i - this->padTop) * this->width + time * this->ubFactorElement +
                             loop * this->outBatchStride + this->ncOffset * this->outBatchStride;
                PipeBarrier<PIPE_ALL>();
                SetFlag<HardEvent::S_MTE2>(eventId0);
                CopyInAndOut2Gm(gmXOffset3, gmYOffset3, calCount, 0);
                WaitFlag<HardEvent::S_MTE2>(eventId0);
            }
        }
    }
}
#endif // _PAD_V3_GRAD_REPLICATE_H_
