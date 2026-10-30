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
 * \file select_v2.h
 * \brief
 */
#ifndef SELECTV2_H
#define SELECTV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "select_v2_tiling_data.h"
#include "select_v2_tiling_key.h"

namespace MySelectV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ONE_REPEAT_SIZE = 256;

template <typename TYPE_X, typename TYPE_Y>
class KernelSelectV2 {
public:
    __aicore__ inline KernelSelectV2(){};

    __aicore__ inline void Init(
        GM_ADDR condition, GM_ADDR self, GM_ADDR other, GM_ADDR out, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
        uint64_t finalBigTileNum, uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum,
        uint64_t bigTailDataNum, uint64_t tailBlockNuma);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueC, inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fp16Buf, tmpBuf;
    AscendC::GlobalTensor<uint8_t> condGm;
    AscendC::GlobalTensor<TYPE_X> selfGm, otherGm;
    AscendC::GlobalTensor<TYPE_Y> outGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint32_t condCopyLength;
};

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSelectV2<TYPE_X, TYPE_Y>::Init(
    GM_ADDR condition, GM_ADDR self, GM_ADDR other, GM_ADDR out, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
    uint64_t finalBigTileNum, uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum,
    uint64_t bigTailDataNum, uint64_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
    }

    condGm.SetGlobalBuffer((__gm__ uint8_t*)condition + globalBufferIndex, this->coreDataNum);
    selfGm.SetGlobalBuffer((__gm__ TYPE_Y*)self + globalBufferIndex, this->coreDataNum);
    otherGm.SetGlobalBuffer((__gm__ TYPE_Y*)other + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ TYPE_Y*)out + globalBufferIndex, this->coreDataNum);

    uint32_t tileDataNumAlign256 = (this->tileDataNum + ONE_REPEAT_SIZE - 1) / ONE_REPEAT_SIZE * ONE_REPEAT_SIZE;

    pipe.InitBuffer(inQueueC, BUFFER_NUM, tileDataNumAlign256 * sizeof(uint8_t));
    pipe.InitBuffer(inQueueX, BUFFER_NUM, 2 * this->tileDataNum * sizeof(TYPE_Y));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

    pipe.InitBuffer(fp16Buf, tileDataNumAlign256 * sizeof(half));

    if constexpr (
        (std::is_same_v<TYPE_Y, uint8_t>) || (std::is_same_v<TYPE_Y, int8_t>) || (std::is_same_v<TYPE_Y, bool>)) {
        pipe.InitBuffer(tmpBuf, this->tileDataNum * sizeof(half));
    }
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSelectV2<TYPE_X, TYPE_Y>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<uint8_t> condLocal = inQueueC.AllocTensor<uint8_t>();
    AscendC::LocalTensor<TYPE_Y> xLocal = inQueueX.AllocTensor<TYPE_Y>();

    AscendC::DataCopy(condLocal, condGm[progress * this->tileDataNum], this->condCopyLength);
    AscendC::DataCopy(xLocal, selfGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(xLocal[this->tileDataNum], otherGm[progress * this->tileDataNum], this->processDataNum);

    inQueueC.EnQue(condLocal);
    inQueueX.EnQue(xLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSelectV2<TYPE_X, TYPE_Y>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    AscendC::DataCopy(outGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSelectV2<TYPE_X, TYPE_Y>::Compute(int32_t progress)
{
    AscendC::LocalTensor<uint8_t> condLocal = inQueueC.DeQue<uint8_t>();

    AscendC::LocalTensor<half> cond_fp16Local = fp16Buf.Get<half>();

    AscendC::Cast(cond_fp16Local, condLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    uint32_t compareCnt = (this->processDataNum + 127) / 128 * 128;
    AscendC::CompareScalar(condLocal, cond_fp16Local, (half)(0.0), AscendC::CMPMODE::NE, compareCnt);

    if constexpr (
        (std::is_same_v<TYPE_Y, float>) || (std::is_same_v<TYPE_Y, uint32_t>) || (std::is_same_v<TYPE_Y, int32_t>)) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> selfLocal = xLocal;
        AscendC::LocalTensor<float> otherLocal = xLocal[this->tileDataNum];
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Select(
            yLocal.ReinterpretCast<float>(), condLocal, selfLocal.ReinterpretCast<float>(),
            otherLocal.ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        outQueueY.EnQue<float>(yLocal);
        inQueueC.FreeTensor(condLocal);
        inQueueX.FreeTensor(xLocal);
    }
    if constexpr (
        (std::is_same_v<TYPE_Y, half>) || (std::is_same_v<TYPE_Y, bfloat16_t>) || (std::is_same_v<TYPE_Y, uint16_t>) ||
        (std::is_same_v<TYPE_Y, int16_t>)) {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> selfLocal = xLocal;
        AscendC::LocalTensor<half> otherLocal = xLocal[this->tileDataNum];
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        AscendC::Select(
            yLocal.ReinterpretCast<half>(), condLocal, selfLocal.ReinterpretCast<half>(),
            otherLocal.ReinterpretCast<half>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        outQueueY.EnQue<half>(yLocal);
        inQueueC.FreeTensor(condLocal);
        inQueueX.FreeTensor(xLocal);
    }
    if constexpr (
        (std::is_same_v<TYPE_Y, uint8_t>) || (std::is_same_v<TYPE_Y, int8_t>) || (std::is_same_v<TYPE_Y, bool>)) {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> selfLocal = xLocal;
        AscendC::LocalTensor<int8_t> otherLocal = xLocal[this->tileDataNum];
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> tmp = tmpBuf.Get<half>();
        AscendC::LocalTensor<half> p1 = tmp;
        AscendC::LocalTensor<half> p2 = tmp[this->tileDataNum];
        AscendC::Cast(p1, selfLocal.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, otherLocal.ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Select(p1, condLocal, p1, p2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        AscendC::Cast(yLocal.ReinterpretCast<int8_t>(), p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        outQueueY.EnQue<int8_t>(yLocal);
        inQueueC.FreeTensor(condLocal);
        inQueueX.FreeTensor(xLocal);
    }
}

template <typename TYPE_X, typename TYPE_Y>
__aicore__ inline void KernelSelectV2<TYPE_X, TYPE_Y>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    this->condCopyLength = (this->processDataNum + 31) / 32 * 32;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    this->condCopyLength = (this->processDataNum + 31) / 32 * 32;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace MySelectV2
#endif // SELECTV2_H
