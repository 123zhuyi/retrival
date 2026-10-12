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
 * \file split_all.h
 * \brief
 */
#ifndef SPLIT_ALL_TRIANGULATOR
#define SPLIT_ALL_TRIANGULATOR

#include "kernel_operator.h"
namespace Triangulator {
using namespace AscendC;

constexpr int32_t OUTPUT_ZERO_MODE = 0;
constexpr int32_t OUTPUT_INPUT_MODE = 1;

template <typename T, const int32_t MODE>
class SplitAll {
public:
    __aicore__ inline SplitAll(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void ParseTilingData(__tiling_data_ptr__ TriangulatorTilingData* tilingDataPtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerLoop(
        LocalTensor<T>& srcLocal, TEventID eventId, int64_t loopNum, int64_t loopBlockLength);

private:
    TPipe* pipe_;
    TBuf<QuePosition::VECCALC> ubBuf1_;
    TBuf<QuePosition::VECCALC> ubBuf2_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    int64_t bufferSize_;
    int64_t usedCoreNum_;

    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;
    int64_t curCoreProcessNum_;

    int64_t curUbLoopCount_;
    int64_t normalLoopBlockNum_;
    int64_t tailLoopBlockNum_;

    uint32_t blockIdx_;
    bool pingpong_;
};

template <typename T, const int32_t MODE>
__aicore__ inline void SplitAll<T, MODE>::ParseTilingData(__tiling_data_ptr__ TriangulatorTilingData* tilingDataPtr)
{
    usedCoreNum_ = tilingDataPtr->usedCoreNum;
    normalCoreProcessNum_ = tilingDataPtr->normalCoreProcessNum;
    tailCoreProcessNum_ = tilingDataPtr->tailCoreProcessNum;
    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        bufferSize_ = tilingDataPtr->bufferSize;
    } else {
        bufferSize_ = tilingDataPtr->bufferSize * 2;
    }
}

template <typename T, const int32_t MODE>
__aicore__ inline void SplitAll<T, MODE>::Init(
    GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorTilingData* tilingDataPtr, TPipe* pipeIn)
{
    // tiling_data
    ParseTilingData(tilingDataPtr);
    blockIdx_ = GetBlockIdx();
    pingpong_ = true;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    int64_t burstLen = bufferSize_ / sizeof(T);
    // shield normal core and tail core
    curCoreProcessNum_ = blockIdx_ < usedCoreNum_ - 1 ? normalCoreProcessNum_ : tailCoreProcessNum_;
    curUbLoopCount_ = (curCoreProcessNum_ + burstLen - 1) / burstLen;

    normalLoopBlockNum_ = (curCoreProcessNum_ >= burstLen) ? burstLen : curCoreProcessNum_;
    tailLoopBlockNum_ = curCoreProcessNum_ - (curUbLoopCount_ - 1) * normalLoopBlockNum_;

    // shield global memory address between different core
    int64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_;
    srcGm_.SetGlobalBuffer((__gm__ T*)x + intraCoreOffset);
    dstGm_.SetGlobalBuffer((__gm__ T*)y + intraCoreOffset);

    pipe_ = pipeIn;
    pipe_->InitBuffer(ubBuf1_, bufferSize_);
    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        pipe_->InitBuffer(ubBuf2_, bufferSize_);
    }
}

template <typename T, const int32_t MODE>
__aicore__ inline void SplitAll<T, MODE>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    LocalTensor<T> srcLocal1 = ubBuf1_.Get<T>();
    LocalTensor<T> srcLocal2;
    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        srcLocal2 = ubBuf2_.Get<T>();
    } else {
        srcLocal2 = srcLocal1;
    }

    TEventID eventFirst = 0;
    TEventID eventSecond = 0;
    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        eventFirst = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        eventSecond = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventFirst);
        SetFlag<HardEvent::MTE3_MTE2>(eventSecond);
    } else {
        eventFirst = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        Duplicate(srcLocal1, T(0), bufferSize_ / sizeof(T));
        SetFlag<HardEvent::V_MTE3>(eventFirst);
        WaitFlag<HardEvent::V_MTE3>(eventFirst);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventFirst);
    }

    for (int64_t loopNum = 0; loopNum < curUbLoopCount_ - 1; loopNum++) {
        if (pingpong_) {
            ProcessPerLoop(srcLocal1, eventFirst, loopNum, normalLoopBlockNum_);
        } else {
            ProcessPerLoop(srcLocal2, eventSecond, loopNum, normalLoopBlockNum_);
        }
        pingpong_ = !pingpong_;
    }

    if (pingpong_) {
        ProcessPerLoop(srcLocal1, eventFirst, curUbLoopCount_ - 1, tailLoopBlockNum_);
    } else {
        ProcessPerLoop(srcLocal2, eventSecond, curUbLoopCount_ - 1, tailLoopBlockNum_);
    }

    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventFirst);
        WaitFlag<HardEvent::MTE3_MTE2>(eventSecond);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventFirst);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventSecond);
    } else {
        PipeBarrier<PIPE_MTE3>();
    }
}

template <typename T, const int32_t MODE>
__aicore__ inline void SplitAll<T, MODE>::ProcessPerLoop(
    LocalTensor<T>& srcLocal, TEventID eventId, int64_t loopNum, int64_t loopBlockLength)
{
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(loopBlockLength * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};

    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        DataCopyPad(srcLocal, srcGm_[normalLoopBlockNum_ * loopNum], copyParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
    }

    DataCopyPad(dstGm_[normalLoopBlockNum_ * loopNum], srcLocal, copyParams);
    if constexpr (MODE == OUTPUT_INPUT_MODE) {
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
}
} // namespace Triangulator

#endif // SPLIT_ALL_TRIANGULATOR