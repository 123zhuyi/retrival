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
 * \file logical_or.h
 * \brief
 */
#ifndef __LOGICAL_OR_H__
#define __LOGICAL_OR_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "logical_or_tiling_data.h"
#include "logical_or_tiling_key.h"

namespace NsLogicalOr {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <bool IsExistBigCore>
class LogicalOr {
public:
    __aicore__ inline LogicalOr() {}
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum,
        uint32_t smallCoreLoopNum, uint32_t ubPartDataNum, uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum,
        uint32_t tailBlockNum)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (IsExistBigCore) {
            if (blockIdx < tailBlockNum) {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            } else {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        x1Gm.SetGlobalBuffer((__gm__ int8_t*)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ int8_t*)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(int8_t));
        pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(half));
        pipe.InitBuffer(tmp2, this->ubPartDataNum * sizeof(half));
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
        AscendC::LocalTensor<int8_t> x1Local = inQueueX1.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> x2Local = inQueueX2.AllocTensor<int8_t>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->ubPartDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->ubPartDataNum], this->processDataNum);

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> x1Local = inQueueX1.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> x2Local = inQueueX2.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();

        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        AscendC::Cast(p1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Max(p1, p1, p2, this->processDataNum);

        AscendC::Cast(yLocal, p1, AscendC::RoundMode::CAST_NONE, this->processDataNum);

        outQueueY.EnQue<int8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.DeQue<int8_t>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::GlobalTensor<int8_t> x1Gm;
    AscendC::GlobalTensor<int8_t> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t ubPartDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};
} // namespace NsLogicalOr
#endif // LOGICAL_OR_H
