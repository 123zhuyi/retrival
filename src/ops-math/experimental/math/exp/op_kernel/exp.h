/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file exp.h
 * \brief
 */
#ifndef __EXP_H__
#define __EXP_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exp_tiling_data.h"
#include "exp_tiling_key.h"

namespace NsExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X, typename TYPE_Y, bool IsExistBigCore>
class Exp {
public:
    __aicore__ inline Exp() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum,
        uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum,
        uint64_t tailBlockNum, float base, float scale, float shift)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreNum = GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore) {
            if (coreNum < tailBlockNum) {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            } else {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * GetBlockIdx();
        }
        this->base = base;
        this->shift = shift;
        this->scale = scale;
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));
        if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
            pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
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

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        if constexpr (std::is_same_v<DTYPE_X, float32_t> || std::is_same_v<TYPE_X, float16_t>) {
            Muls(yLocal, xLocal, (DTYPE_X)this->scale, this->processDataNum);
            Adds(yLocal, yLocal, (DTYPE_X)this->shift, this->processDataNum);
            Muls(yLocal, yLocal, (DTYPE_X)this->base, this->processDataNum);
            AscendC::Exp(yLocal, yLocal, this->processDataNum);
        } else {
            LocalTensor<float> xLocalFp32 = tmp1.Get<float>();
            Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Muls(xLocalFp32, xLocalFp32, this->scale, this->processDataNum);
            Adds(xLocalFp32, xLocalFp32, this->shift, this->processDataNum);
            Muls(xLocalFp32, xLocalFp32, this->base, this->processDataNum);
            AscendC::Exp(xLocalFp32, xLocalFp32, this->processDataNum);
            Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1;
    TBuf<QuePosition::VECCALC> tmp2;
    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    float base;
    float shift;
    float scale;
};
} // namespace NsExp
#endif // EXP_H
