/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ROUND_H
#define ROUND_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "round_tiling_data.h"
#include "round_tiling_key.h"

namespace MyRound {

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>

class KernelRound {
    using T = TYPE_X;

public:
    __aicore__ inline KernelRound() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RoundTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf,
        tmpBuf1; // tmpBuf用于格式转换临时存储变量，tmpBuf1用于x缩放存储变量
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    float decimals;
};
template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void KernelRound<TYPE_X, TYPE_Y, IsExistBigCore>::Init(
    GM_ADDR x, GM_ADDR y, const RoundTilingData& tilingData)

{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->tileDataNum = tilingData.tileDataNum;
    this->decimals = tilingData.decimals;
    uint64_t blockIdx = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData.bigCoreDataNum * AscendC::GetBlockIdx();
    // 是否存在大核
    if constexpr (IsExistBigCore) {
        if (blockIdx < tilingData.tailBlockNum) // 与大核个数对比，小于，那就在大核里，大于就在小核里
        {
            this->coreDataNum = tilingData.bigCoreDataNum;
            this->tileNum = tilingData.finalBigTileNum;
            this->tailDataNum = tilingData.bigTailDataNum;
        } else {
            this->coreDataNum = tilingData.smallCoreDataNum;
            this->tileNum = tilingData.finalSmallTileNum;
            this->tailDataNum = tilingData.smallTailDataNum;
            globalBufferIndex -= (tilingData.bigCoreDataNum - tilingData.smallCoreDataNum) *
                                 (AscendC::GetBlockIdx() - tilingData.tailBlockNum);
        }
    } else {
        this->coreDataNum = tilingData.smallCoreDataNum;
        this->tileNum = tilingData.finalSmallTileNum;
        this->tailDataNum = tilingData.smallTailDataNum;
        globalBufferIndex = tilingData.smallCoreDataNum * AscendC::GetBlockIdx();
    }

    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, int32_t>) {
        pipe.InitBuffer(tmpBuf, this->tileDataNum * sizeof(float)); // 存储格式转换临时变量
    }
    if (decimals != 1) {
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float)); // 存储缩放临时变量
    }
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void KernelRound<TYPE_X, TYPE_Y, IsExistBigCore>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void KernelRound<TYPE_X, TYPE_Y, IsExistBigCore>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void KernelRound<TYPE_X, TYPE_Y, IsExistBigCore>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

    if constexpr (std::is_same_v<T, int32_t>) {
        AscendC::DataCopy(yLocal, xLocal, this->processDataNum);
    }

    else if constexpr (!std::is_same_v<T, float>) {
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::Cast(tmp, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        if (decimals > 1.0f) {
            AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
            AscendC::Duplicate<float>(tmp1, decimals, this->processDataNum); // 开辟一个向量空间
            AscendC::Mul(tmp, tmp, tmp1, this->processDataNum);
            AscendC::Round(tmp, tmp, this->processDataNum);
            AscendC::Div(tmp, tmp, tmp1, this->processDataNum);
        } else {
            AscendC::Round(tmp, tmp, this->processDataNum);
        }
        AscendC::Cast(yLocal, tmp, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    } else {
        if (decimals > 1.0f) {
            AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
            AscendC::Duplicate<float>(tmp1, decimals, this->processDataNum);
            AscendC::Mul(xLocal, xLocal, tmp1, this->processDataNum);
            AscendC::Round(yLocal, xLocal, this->processDataNum);
            AscendC::Div(yLocal, yLocal, tmp1, this->processDataNum);
        } else {
            AscendC::Round(yLocal, xLocal, this->processDataNum);
        }
    }

    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
__aicore__ inline void KernelRound<TYPE_X, TYPE_Y, IsExistBigCore>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}
} // namespace MyRound
#endif // SQRT_H
