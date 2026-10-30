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
 * \file tanh_grad.h
 * \brief
 */
#ifndef TANH_GRAD_H
#define TANH_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tanh_grad_tiling_data.h"
#include "tanh_grad_tiling_key.h"

namespace MyTanhGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_Y>
class KernelTanhGrad {
public:
    __aicore__ inline KernelTanhGrad(){};

    __aicore__ inline void Init(
        GM_ADDR y, GM_ADDR dy, GM_ADDR dx, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
        uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
        uint64_t tailBlockNum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueDX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp3;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Y> dyGm;
    AscendC::GlobalTensor<TYPE_Y> dxGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template <typename TYPE_Y>
__aicore__ inline void KernelTanhGrad<TYPE_Y>::Init(
    GM_ADDR y, GM_ADDR dy, GM_ADDR dx, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t finalBigTileNum,
    uint64_t finalSmallTileNum, uint64_t tileDataNum, uint64_t smallTailDataNum, uint64_t bigTailDataNum,
    uint64_t tailBlockNum)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }

    yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
    dyGm.SetGlobalBuffer((__gm__ TYPE_Y*)dy + globalBufferIndex, this->coreDataNum);
    dxGm.SetGlobalBuffer((__gm__ TYPE_Y*)dx + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    pipe.InitBuffer(outQueueDX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

    // 参数类型不为float32时需要为中间计算分配临时缓冲区
    if constexpr (!std::is_same_v<TYPE_Y, float32_t>) {
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_Y>
__aicore__ inline void KernelTanhGrad<TYPE_Y>::CopyIn(int32_t progress)
{
    uint32_t offset = progress * this->tileDataNum;
    uint32_t remain = this->coreDataNum - offset;
    uint32_t cpSize = (this->processDataNum < remain) ? this->processDataNum : remain;
    AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
    AscendC::LocalTensor<TYPE_Y> dyLocal = inQueueDY.AllocTensor<TYPE_Y>();
    AscendC::DataCopy(yLocal, yGm[offset], cpSize);
    AscendC::DataCopy(dyLocal, dyGm[offset], cpSize);
    inQueueY.EnQue(yLocal);
    inQueueDY.EnQue(dyLocal);
}

template <typename TYPE_Y>
__aicore__ inline void KernelTanhGrad<TYPE_Y>::CopyOut(int32_t progress)
{
    uint32_t offset = progress * this->tileDataNum;
    uint32_t remain = this->coreDataNum - offset;
    uint32_t cpSize = (this->processDataNum < remain) ? this->processDataNum : remain;
    AscendC::LocalTensor<TYPE_Y> dxLocal = outQueueDX.DeQue<TYPE_Y>();
    AscendC::DataCopy(dxGm[offset], dxLocal, cpSize);
    outQueueDX.FreeTensor(dxLocal);
}

template <typename TYPE_Y>
__aicore__ inline void KernelTanhGrad<TYPE_Y>::Compute(int32_t progress)
{
    AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
    AscendC::LocalTensor<TYPE_Y> dyLocal = inQueueDY.DeQue<TYPE_Y>();
    AscendC::LocalTensor<TYPE_Y> dxLocal = outQueueDX.AllocTensor<TYPE_Y>();

    if constexpr (std::is_same_v<TYPE_Y, float32_t>) {
        // float32类型直接计算
        AscendC::Mul(dxLocal, yLocal, yLocal, this->processDataNum);   // y²
        AscendC::Muls(dxLocal, dxLocal, -1.0f, this->processDataNum);  // -y²
        AscendC::Adds(dxLocal, dxLocal, 1.0f, this->processDataNum);   // 1 - y²
        AscendC::Mul(dxLocal, dxLocal, dyLocal, this->processDataNum); // dy * (1 - y²)

    } else {
        // 非float32类型
        AscendC::LocalTensor<float> yFloat = tmp1.Get<float>();
        AscendC::LocalTensor<float> dyFloat = tmp2.Get<float>();
        AscendC::LocalTensor<float> dxFloat = tmp3.Get<float>();

        // 转换为float32进行计算
        AscendC::Cast(yFloat, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(dyFloat, dyLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        AscendC::Mul(dxFloat, yFloat, yFloat, this->processDataNum);   // y²
        AscendC::Muls(dxFloat, dxFloat, -1.0f, this->processDataNum);  // -y²
        AscendC::Adds(dxFloat, dxFloat, 1.0f, this->processDataNum);   // 1 - y²
        AscendC::Mul(dxFloat, dxFloat, dyFloat, this->processDataNum); // dy * (1 - y²)

        // 转换回目标类型
        AscendC::Cast(dxLocal, dxFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
    }

    outQueueDX.EnQue<TYPE_Y>(dxLocal);
    inQueueY.FreeTensor(yLocal);
    inQueueDY.FreeTensor(dyLocal);
}

template <typename TYPE_Y>
__aicore__ inline void KernelTanhGrad<TYPE_Y>::Process()
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

} // namespace MyTanhGrad
#endif // TANH_GRAD_H