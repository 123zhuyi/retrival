/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pow2.h
 * \brief
 */
#ifndef __POW2_H__
#define __POW2_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pow2_tiling_data.h"
#include "pow2_tiling_key.h"

namespace NsPow2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
class Pow2 {
public:
    __aicore__ inline Pow2(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Pow2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t GetBroadcastIndexEff(uint32_t linearIdx, const uint32_t* effStride);
    __aicore__ inline void CopyIn(uint32_t progress);
    __aicore__ inline void CopyOut(uint32_t progress);
    __aicore__ inline void PowCompute(
        LocalTensor<float>& x1Local, LocalTensor<float>& x2Local, LocalTensor<float>& yLocal,
        LocalTensor<uint8_t>& mask);
    __aicore__ inline void Compute(uint32_t progress);

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> x1CastTmp;
    TBuf<QuePosition::VECCALC> x2CastTmp;
    TBuf<QuePosition::VECCALC> yCastTmp;
    TBuf<QuePosition::VECCALC> tranCastTmp;
    TBuf<QuePosition::VECCALC> absTmp;
    TBuf<QuePosition::VECCALC> maskTmp;
    TBuf<QuePosition::VECCALC> zeroTmp;
    GlobalTensor<TYPE_X1> x1Gm;
    GlobalTensor<TYPE_X2> x2Gm;
    GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;

    uint64_t globalBufferIndex;
    // 标量判断
    bool is_input0_scalar;
    bool is_input1_scalar;
    // 形状广播
    uint32_t yDim;
    bool isSameX1;
    bool isSameX2;
    uint32_t strideX1[10];
    uint32_t strideX2[10];
    uint32_t strideY[10];
};

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline void Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Pow2TilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreNum = GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * coreNum;
    this->tileDataNum = tilingData->tileDataNum;
    this->is_input0_scalar = tilingData->is_input0_scalar;
    this->is_input1_scalar = tilingData->is_input1_scalar;
    this->yDim = tilingData->yDim;
    const uint32_t* strideX1Src = tilingData->strideX1;
    const uint32_t* strideX2Src = tilingData->strideX2;
    const uint32_t* strideYSrc = tilingData->strideY;
    for (int i = 0; i < 10; ++i) {
        this->strideX1[i] = strideX1Src[i];
        this->strideX2[i] = strideX2Src[i];
        this->strideY[i] = strideYSrc[i];
    }
    this->isSameX1 = tilingData->isSameX1;
    this->isSameX2 = tilingData->isSameX2;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreNum - tilingData->tailBlockNum);
    }
    if (tilingData->isSameX1) {
        // 无广播，按输出偏移访问
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + globalBufferIndex, coreDataNum);
    } else {
        // 有广播，直接访问原始数据（不叠加输出偏移）
        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1, tilingData->X1TotalNum); // x1TotalNum是x的原始总元素数
    }
    if (tilingData->isSameX2) {
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + globalBufferIndex, coreDataNum);
    } else {
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2, tilingData->X2TotalNum); // x2TotalNum是y的原始总元素数
    }
    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, coreDataNum);
    pipe.InitBuffer(inQueueX1, BUFFER_NUM, tileDataNum * sizeof(TYPE_X1));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, tileDataNum * sizeof(TYPE_X2));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileDataNum * sizeof(TYPE_Y));
    // === Init Tbuf ===
    pipe.InitBuffer(absTmp, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(zeroTmp, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(maskTmp, this->tileDataNum * sizeof(uint8_t));

    pipe.InitBuffer(x1CastTmp, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(x2CastTmp, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(yCastTmp, this->tileDataNum * sizeof(float));
    if constexpr (
        IsSameType<TYPE_Y, uint8_t>::value || IsSameType<TYPE_Y, int8_t>::value ||
        IsSameType<TYPE_X1, uint8_t>::value || IsSameType<TYPE_X1, int8_t>::value ||
        IsSameType<TYPE_X2, uint8_t>::value || IsSameType<TYPE_X2, int8_t>::value) {
        pipe.InitBuffer(tranCastTmp, this->tileDataNum * sizeof(half));
    }
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline uint32_t Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::GetBroadcastIndexEff(
    uint32_t linearIdx,
    const uint32_t* effStride // 预计算后的有效stride
)
{
    // 利用预计算stride快速定位
    // linearIdx 按输出 shape 展开
    uint32_t offset = 0;
    uint32_t tmp = linearIdx;
    for (int i = 0; i < yDim; ++i) {
        uint32_t coord = tmp / strideY[i];
        tmp %= strideY[i];
        offset += coord * effStride[i];
    }
    return offset;
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline void Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::CopyIn(uint32_t progress)
{
    LocalTensor<TYPE_X1> x1Local = inQueueX1.AllocTensor<TYPE_X1>();
    LocalTensor<TYPE_X2> x2Local = inQueueX2.AllocTensor<TYPE_X2>();
    if (this->is_input0_scalar && !this->is_input1_scalar) {
        TYPE_X1 scalarValX;
        scalarValX = x1Gm.GetValue(0); // duplicate不支持int8和uint8
        if constexpr (std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>) {
            for (int i = 0; i < this->processDataNum; ++i) {
                x1Local.SetValue(i, scalarValX);
            }
        } else {
            Duplicate(x1Local, scalarValX, this->processDataNum);
        }
        DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
    } else if (!this->is_input0_scalar && this->is_input1_scalar) {
        DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
        TYPE_X2 scalarValX2;
        scalarValX2 = x2Gm.GetValue(0);
        if constexpr (std::is_same_v<TYPE_X2, int8_t> || std::is_same_v<TYPE_X2, uint8_t>) {
            for (int i = 0; i < this->processDataNum; ++i) {
                x2Local.SetValue(i, scalarValX2);
            }
        } else {
            Duplicate(x2Local, scalarValX2, this->processDataNum);
        }
    } else {                        // ====== 判断是否完全匹配 ======
        if (isSameX1 && isSameX2) { // 无广播，直接DMA搬运
            DataCopy(x1Local, x1Gm[progress * tileDataNum], processDataNum);
            DataCopy(x2Local, x2Gm[progress * tileDataNum], processDataNum);
        } else if (!isSameX1 && isSameX2) { // x广播yDMA搬运
            for (uint32_t i = 0; i < processDataNum; ++i) {
                uint32_t globalIdx = globalBufferIndex + i;
                uint32_t idxX = GetBroadcastIndexEff(globalIdx, this->strideX1);
                x1Local.SetValue(i, x1Gm.GetValue(idxX));
            }
            DataCopy(x2Local, x2Gm[progress * tileDataNum], processDataNum);
        } else if (isSameX1 && !isSameX2) { // xDMA搬运y广播
            DataCopy(x1Local, x1Gm[progress * tileDataNum], processDataNum);
            for (uint32_t i = 0; i < processDataNum; ++i) {
                uint32_t globalIdx = globalBufferIndex + i;
                uint32_t idxY = GetBroadcastIndexEff(globalIdx, this->strideX2);
                x2Local.SetValue(i, x2Gm.GetValue(idxY));
            }
        } else { // 全广播
            for (uint32_t i = 0; i < processDataNum; ++i) {
                uint32_t globalIdx = globalBufferIndex + i;
                uint32_t idxX = GetBroadcastIndexEff(globalIdx, this->strideX1);
                uint32_t idxX2 = GetBroadcastIndexEff(globalIdx, this->strideX2);
                x1Local.SetValue(i, x1Gm.GetValue(idxX));
                x2Local.SetValue(i, x2Gm.GetValue(idxX2));
            }
        }
    }
    inQueueX1.EnQue(x1Local);
    inQueueX2.EnQue(x2Local);
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline void Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::CopyOut(uint32_t progress)
{
    LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
    DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
    outQueueY.FreeTensor(yLocal);
}
template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline void Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::PowCompute(
    LocalTensor<float>& x1Local, LocalTensor<float>& x2Local, LocalTensor<float>& yLocal, LocalTensor<uint8_t>& mask)
{
    LocalTensor<float> p1 = absTmp.Get<float>();
    LocalTensor<float> zeros = zeroTmp.Get<float>();
    LocalTensor<float>& scalars = zeros;
    LocalTensor<float>& ones = x2Local;
    Duplicate(scalars, 2.0f, this->processDataNum);
    Abs(p1, x1Local, this->processDataNum);
    Ln(p1, p1, this->processDataNum);
    Mul(p1, p1, x2Local, this->processDataNum);
    Exp(yLocal, p1, this->processDataNum); // yLocal = exp(y*len(|x|))

    Abs(p1, x2Local, this->processDataNum);
    Div(zeros, p1, scalars, this->processDataNum);
    Cast(zeros, zeros, RoundMode::CAST_TRUNC, this->processDataNum);
    Mul(zeros, zeros, scalars, this->processDataNum);

    Sub(p1, p1, zeros, this->processDataNum);
    Muls(p1, p1, -2.0f, this->processDataNum);
    Adds(p1, p1, 1.0f, this->processDataNum);
    Mul(p1, p1, yLocal, this->processDataNum);

    Duplicate(zeros, 0.0f, this->processDataNum);

    Compare(mask, x1Local, zeros, CMPMODE::LT, this->processDataNum); // x小于0的部分
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    Compare(mask, x2Local, zeros, CMPMODE::EQ, this->processDataNum); // exp等于0的部分
    Duplicate(ones, 1.0f, this->processDataNum);
    Select(p1, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);

    Compare(mask, x1Local, zeros, CMPMODE::EQ, this->processDataNum); // x等于0的部分
    Select(yLocal, mask, p1, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline void Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::Compute(uint32_t progress)
{
    LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
    LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
    LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
    LocalTensor<float> x1LocalFp = x1CastTmp.Get<float>();
    LocalTensor<float> x2LocalFp = x2CastTmp.Get<float>();
    LocalTensor<float> yLocalFp = yCastTmp.Get<float>();
    LocalTensor<uint8_t> mask = maskTmp.Get<uint8_t>();
    if constexpr (
        IsSameType<TYPE_X1, half>::value || IsSameType<TYPE_X1, bfloat16_t>::value ||
        IsSameType<TYPE_X1, int16_t>::value || IsSameType<TYPE_X1, int32_t>::value) {
        Cast(x1LocalFp, x1Local, RoundMode::CAST_NONE, this->processDataNum);
    }
    if constexpr (IsSameType<TYPE_X1, uint8_t>::value || IsSameType<TYPE_X1, int8_t>::value) {
        LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
        Cast(tmpLocal, x1Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(x1LocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
    }
    if constexpr (IsSameType<TYPE_X1, float32_t>::value) {
        x1LocalFp = x1Local.template ReinterpretCast<float>();
    }
    if constexpr (
        IsSameType<TYPE_X2, half>::value || IsSameType<TYPE_X2, bfloat16_t>::value ||
        IsSameType<TYPE_X2, int16_t>::value || IsSameType<TYPE_X2, int32_t>::value) {
        Cast(x2LocalFp, x2Local, RoundMode::CAST_NONE, this->processDataNum);
    }
    if constexpr (IsSameType<TYPE_X2, uint8_t>::value || IsSameType<TYPE_X2, int8_t>::value) {
        LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
        Cast(tmpLocal, x2Local, RoundMode::CAST_NONE, this->processDataNum);
        Cast(x2LocalFp, tmpLocal, RoundMode::CAST_NONE, this->processDataNum);
    }
    if constexpr (IsSameType<TYPE_X2, float32_t>::value) {
        x2LocalFp = x2Local.template ReinterpretCast<float>();
    }
    PowCompute(x1LocalFp, x2LocalFp, yLocalFp, mask);
    if constexpr (
        IsSameType<TYPE_Y, half>::value || IsSameType<TYPE_Y, bfloat16_t>::value ||
        IsSameType<TYPE_Y, int16_t>::value || IsSameType<TYPE_Y, int32_t>::value) {
        Cast(yLocal, yLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    }
    if constexpr (IsSameType<TYPE_Y, uint8_t>::value || IsSameType<TYPE_Y, int8_t>::value) {
        LocalTensor<half> tmpLocal = tranCastTmp.Get<half>();
        Cast(tmpLocal, yLocalFp, RoundMode::CAST_NONE, this->processDataNum);
        Cast(yLocal, tmpLocal, RoundMode::CAST_RINT, this->processDataNum);
    }
    if constexpr (IsSameType<TYPE_Y, float32_t>::value) {
        yLocal = yLocalFp.template ReinterpretCast<float>();
    }
    outQueueY.EnQue<TYPE_Y>(yLocal);
    inQueueX1.FreeTensor(x1Local);
    inQueueX2.FreeTensor(x2Local);
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y>
__aicore__ inline void Pow2<TYPE_X1, TYPE_X2, TYPE_Y>::Process()
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

} // namespace NsPow2
#endif // POW2_H