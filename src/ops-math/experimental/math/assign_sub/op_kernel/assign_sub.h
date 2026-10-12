/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASSIGNSUB_H
#define ASSIGNSUB_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "assign_sub_tiling_data.h"
#include "assign_sub_tiling_key.h"

namespace NsAssignSub {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;

template <typename T>
struct DtypeTrait {
    using ComputeT = T;
    static constexpr bool needCast = false;
    static constexpr bool needMod = false;
};
template <>
struct DtypeTrait<int8_t> {
    using ComputeT = half;
    static constexpr bool needCast = true;
    static constexpr bool needMod = true;
};
template <>
struct DtypeTrait<uint8_t> {
    using ComputeT = half;
    static constexpr bool needCast = true;
    static constexpr bool needMod = true;
};
template <>
struct DtypeTrait<bfloat16_t> {
    using ComputeT = float;
    static constexpr bool needCast = true;
    static constexpr bool needMod = false;
};
// Note: int64 uses int32 as intermediate type for Sub operation.
// This limits the valid input range to [-2^31+1, 2^31-1] per element.
// Values exceeding int32 range will produce incorrect results due to truncation.
template <>
struct DtypeTrait<int64_t> {
    using ComputeT = int32_t;
    static constexpr bool needCast = true;
    static constexpr bool needMod = false;
};

template <typename T>
class AssignSub {
public:
    __aicore__ inline AssignSub(){};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR value, GM_ADDR var_out, const AssignSubTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t offset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    using ComputeT = typename DtypeTrait<T>::ComputeT;
    static constexpr bool needCast = DtypeTrait<T>::needCast;
    static constexpr bool needMod = DtypeTrait<T>::needMod;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueVar;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueValue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;

    GlobalTensor<T> varGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> outGm;

    int64_t coreOffset_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubFactor_ = 0;
};

template <typename T>
__aicore__ inline void AssignSub<T>::Init(GM_ADDR var, GM_ADDR value, GM_ADDR var_out,
                                          const AssignSubTilingData* tilingData)
{
    int64_t totalNum = tilingData->totalNum;
    int64_t blockFactor = tilingData->blockFactor;
    ubFactor_ = tilingData->ubFactor;

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    coreOffset_ = blockIdx * blockFactor;
    if (coreOffset_ >= totalNum) {
        coreNum_ = 0;
        return;
    }
    coreNum_ = totalNum - coreOffset_;
    if (coreNum_ > blockFactor) {
        coreNum_ = blockFactor;
    }

    varGm.SetGlobalBuffer((__gm__ T*)var + coreOffset_, coreNum_);
    valueGm.SetGlobalBuffer((__gm__ T*)value + coreOffset_, coreNum_);
    outGm.SetGlobalBuffer((__gm__ T*)var_out + coreOffset_, coreNum_);

    pipe.InitBuffer(inputQueueVar, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(inputQueueValue, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubFactor_ * sizeof(T));
    if constexpr (needCast) {
        pipe.InitBuffer(tmpBuf0, ubFactor_ * sizeof(ComputeT));
        pipe.InitBuffer(tmpBuf1, ubFactor_ * sizeof(ComputeT));
    }
}

template <typename T>
__aicore__ inline void AssignSub<T>::CopyIn(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> varLocal = inputQueueVar.AllocTensor<T>();
    LocalTensor<T> valueLocal = inputQueueValue.AllocTensor<T>();
    constexpr int64_t alignElem = BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (currentNum % alignElem == 0) {
        DataCopy(varLocal, varGm[offset], static_cast<int32_t>(currentNum));
        DataCopy(valueLocal, valueGm[offset], static_cast<int32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(varLocal, varGm[offset], copyParams, padParams);
        DataCopyPad(valueLocal, valueGm[offset], copyParams, padParams);
    }
    inputQueueVar.EnQue(varLocal);
    inputQueueValue.EnQue(valueLocal);
}

template <typename T>
__aicore__ inline void AssignSub<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> varLocal = inputQueueVar.DeQue<T>();
    LocalTensor<T> valueLocal = inputQueueValue.DeQue<T>();
    LocalTensor<T> outLocal = outputQueue.AllocTensor<T>();

    int32_t count = static_cast<int32_t>(currentNum);

    if constexpr (!needCast) {
        Sub(outLocal, varLocal, valueLocal, count);
    } else if constexpr (needMod) {
        LocalTensor<half> h0 = tmpBuf0.Get<half>();
        LocalTensor<half> h1 = tmpBuf1.Get<half>();
        Cast(h0, varLocal, RoundMode::CAST_NONE, count);
        Cast(h1, valueLocal, RoundMode::CAST_NONE, count);
        Sub(h0, h0, h1, count);
        LocalTensor<int16_t> i16 = tmpBuf1.Get<int16_t>();
        Cast(i16, h0, RoundMode::CAST_ROUND, count);
        if constexpr (std::is_same_v<T, int8_t>) {
            ShiftLeft(i16, i16, static_cast<int16_t>(8), count);
            ShiftRight(i16, i16, static_cast<int16_t>(8), count);
        } else {
            LocalTensor<uint16_t> u16 = i16.template ReinterpretCast<uint16_t>();
            ShiftLeft(u16, u16, static_cast<uint16_t>(8), count);
            ShiftRight(u16, u16, static_cast<uint16_t>(8), count);
        }
        Cast(h0, i16, RoundMode::CAST_NONE, count);
        Cast(outLocal, h0, RoundMode::CAST_ROUND, count);
    } else {
        LocalTensor<ComputeT> c0 = tmpBuf0.Get<ComputeT>();
        LocalTensor<ComputeT> c1 = tmpBuf1.Get<ComputeT>();
        Cast(c0, varLocal, RoundMode::CAST_NONE, count);
        Cast(c1, valueLocal, RoundMode::CAST_NONE, count);
        Sub(c0, c0, c1, count);
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            Cast(outLocal, c0, RoundMode::CAST_RINT, count);
        } else {
            Cast(outLocal, c0, RoundMode::CAST_NONE, count);
        }
    }

    outputQueue.EnQue<T>(outLocal);
    inputQueueVar.FreeTensor(varLocal);
    inputQueueValue.FreeTensor(valueLocal);
}

template <typename T>
__aicore__ inline void AssignSub<T>::CopyOut(int64_t offset, int64_t currentNum)
{
    LocalTensor<T> outLocal = outputQueue.DeQue<T>();
    constexpr int64_t alignElem = BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    if (currentNum % alignElem == 0) {
        DataCopy(outGm[offset], outLocal, static_cast<int32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outGm[offset], outLocal, copyParams);
    }
    outputQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void AssignSub<T>::Process()
{
    if (coreNum_ <= 0) {
        return;
    }
    int64_t loopCount = (coreNum_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t offset = i * ubFactor_;
        int64_t currentNum = ubFactor_;
        if (offset + currentNum > coreNum_) {
            currentNum = coreNum_ - offset;
        }
        CopyIn(offset, currentNum);
        Compute(currentNum);
        CopyOut(offset, currentNum);
    }
}

} // namespace NsAssignSub
#endif // ASSIGNSUB_H
