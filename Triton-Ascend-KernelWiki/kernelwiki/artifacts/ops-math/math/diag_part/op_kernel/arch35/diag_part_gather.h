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
 * \file diag_part_gather.h
 * \brief diag_part_gather
 */

#ifndef DIAG_PART_GATHER_H
#define DIAG_PART_GATHER_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace DiagPart {
using namespace AscendC;

template <typename T>
class DiagPartGather {
public:
    __aicore__ inline DiagPartGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DiagPartTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore();

    constexpr static int32_t bufferNum = 2;
    constexpr static int32_t blockSize = 32;
    constexpr static int32_t BLOCK_NUM = blockSize / sizeof(T);
    constexpr static int32_t cacheLine = 128;

private:
    int64_t blockIdx_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;

    GlobalTensor<T> xGM_;
    GlobalTensor<T> yGM_;

    // tiling params
    const DiagPartTilingData* tiling_;

    int64_t curBlockLoopNumStart_ = 0;
    int64_t curBlockLoopNumEnd_ = 0;
    int64_t curProcessBlock_ = 0;
    int64_t xGmOffset_ = 0;
    int64_t yGmOffset_ = 0;
    int64_t maxBlockNum_ = 0;

    int64_t blockNum_ = 0;

    // Datacopy params
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void DiagPartGather<T>::Init(GM_ADDR x, GM_ADDR y, const DiagPartTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;

    xGM_.SetGlobalBuffer((__gm__ T*)x);
    yGM_.SetGlobalBuffer((__gm__ T*)y);
    int64_t ubSize = tiling_->ubSize;

    maxBlockNum_ = (ubSize - tiling_->numPerCore * sizeof(int32_t)) / (2 * cacheLine * (tiling_->numPerCore + 1));

    pipe->InitBuffer(xQueue_, bufferNum, tiling_->numPerCore * tiling_->numPerCore * maxBlockNum_ * sizeof(T));
    pipe->InitBuffer(yQueue_, bufferNum, tiling_->numPerCore * maxBlockNum_ * sizeof(T));
    pipe->InitBuffer(indexBuf_, tiling_->numPerCore * maxBlockNum_ * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void DiagPartGather<T>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum || tiling_->sideLength == 0) {
        return;
    }
    blockNum_ = Ops::Base::CeilDiv(tiling_->sideLength, tiling_->numPerCore);
    int64_t loopNum = blockNum_ / tiling_->realCoreNum;
    int64_t mainCoreNum = blockNum_ % tiling_->realCoreNum;

    curBlockLoopNumStart_ = blockIdx_ < mainCoreNum ? blockIdx_ * (loopNum + 1) :
                                                      mainCoreNum * (loopNum + 1) + (blockIdx_ - mainCoreNum) * loopNum;

    xGmOffset_ = curBlockLoopNumStart_ * tiling_->numPerCore * (tiling_->sideLength + 1);
    yGmOffset_ = curBlockLoopNumStart_ * tiling_->numPerCore;

    curBlockLoopNumEnd_ =
        blockIdx_ < mainCoreNum ? curBlockLoopNumStart_ + loopNum : curBlockLoopNumStart_ + loopNum - 1;
    ProcessPerCore();
}

template <typename T>
__aicore__ inline void DiagPartGather<T>::ProcessPerCore()
{
    int64_t mnAlignSize = 0;
    int64_t loopIdx = curBlockLoopNumStart_;
    curProcessBlock_ = loopIdx;
    copyInParams_.blockCount = tiling_->numPerCore;
    copyInParams_.blockLen = tiling_->numPerCore * sizeof(T);                          // unit Byte
    copyInParams_.srcStride = (tiling_->sideLength - tiling_->numPerCore) * sizeof(T); // unit Byte
    copyInParams_.dstStride = 0;                                                       // unit block(32byte)

    copyOutParams_.blockCount = 1;
    copyOutParams_.blockLen = tiling_->numPerCore * sizeof(T); // unit Byte
    copyOutParams_.srcStride = 0;                              // unit Byte
    copyOutParams_.dstStride = 0;                              // unit block(32byte)

    LocalTensor<int32_t> indexLocal = indexBuf_.AllocTensor<int32_t>();
    ArithProgression<int32_t>(
        indexLocal, static_cast<int32_t>(0), static_cast<int32_t>((tiling_->numPerCore + 1) * sizeof(T)),
        static_cast<int32_t>(tiling_->numPerCore));

    while (loopIdx <= curBlockLoopNumEnd_) {
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        int64_t xUbOffset = 0;
        int64_t yUbOffset = 0;
        int64_t ubLoopNum = 0;
        for (loopIdx = curProcessBlock_; loopIdx <= curBlockLoopNumEnd_; loopIdx++) {
            if (loopIdx == blockNum_ - 1 && tiling_->tailNum != 0) {
                copyInParams_.blockCount = tiling_->tailNum;
                copyInParams_.blockLen = tiling_->tailNum * sizeof(T);                          // unit Byte
                copyInParams_.srcStride = (tiling_->sideLength - tiling_->tailNum) * sizeof(T); // unit Byte

                copyOutParams_.blockLen = tiling_->tailNum * sizeof(T); // unit Byte

                int32_t alignSideLength =
                    Ops::Base::CeilAlign(static_cast<int32_t>(tiling_->tailNum * sizeof(T)), blockSize) / sizeof(T);
                ArithProgression<int32_t>(
                    indexLocal, static_cast<int32_t>(0), static_cast<int32_t>((alignSideLength + 1) * sizeof(T)),
                    static_cast<int32_t>(tiling_->tailNum));
            }
            if (ubLoopNum >= maxBlockNum_) {
                xQueue_.FreeTensor(xLocal);
                yQueue_.FreeTensor(yLocal);
                break;
            }
            int64_t mnAlignSize = Ops::Base::CeilAlign(copyInParams_.blockCount * copyInParams_.blockCount, BLOCK_NUM);
            DataCopyPad(xLocal[xUbOffset], xGM_[xGmOffset_], copyInParams_, padParams_);
            SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            LocalTensor<uint32_t> interpreIndexTensor = indexLocal.ReinterpretCast<uint32_t>();
            Gather(yLocal[yUbOffset], xLocal[xUbOffset], interpreIndexTensor, 0, tiling_->numPerCore);
            SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            DataCopyPad(yGM_[yGmOffset_], yLocal[yUbOffset], copyOutParams_);

            xGmOffset_ += tiling_->numPerCore * (tiling_->sideLength + 1);
            yGmOffset_ += tiling_->numPerCore;
            xUbOffset += mnAlignSize;
            yUbOffset += tiling_->numPerCore;
            curProcessBlock_++;
            ubLoopNum++;
        }
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
        SetFlag<HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        WaitFlag<HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }
}
} // namespace DiagPart
#endif // DIAG_PART_GATHER_H