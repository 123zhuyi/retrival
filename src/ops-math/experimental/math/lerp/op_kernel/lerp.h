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
 * \file lerp.h
 * \brief
 */
#ifndef __LERP_H__
#define __LERP_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lerp_tiling_data.h"
#include "lerp_tiling_key.h"

namespace NsLerp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_START, typename TYPE_END, typename TYPE_WEIGHT, typename TYPE_Y, bool IsExistBigCore>
class Lerp {
public:
    __aicore__ inline Lerp() {}
    __aicore__ inline void Init(
        GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
        uint64_t bigCoreLoopNum, uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum,
        uint64_t bigCoreTailDataNum, uint64_t tailBlockNum)
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

        startGm.SetGlobalBuffer((__gm__ TYPE_START*)start + globalBufferIndex, this->coreDataNum);
        endGm.SetGlobalBuffer((__gm__ TYPE_END*)end + globalBufferIndex, this->coreDataNum);
        weightGm.SetGlobalBuffer((__gm__ TYPE_WEIGHT*)weight + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueStart, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_START));
        pipe.InitBuffer(inQueueEnd, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_END));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_Y));

        if constexpr (!std::is_same_v<TYPE_START, float>) {
            pipe.InitBuffer(tmp1, this->ubPartDataNum * sizeof(float));
            pipe.InitBuffer(tmp2, this->ubPartDataNum * sizeof(float));
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
        LocalTensor<TYPE_START> startLocal = inQueueStart.AllocTensor<TYPE_START>();
        LocalTensor<TYPE_END> endLocal = inQueueEnd.AllocTensor<TYPE_END>();
        LocalTensor<TYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<TYPE_WEIGHT>();

        DataCopy(startLocal, startGm[progress * this->ubPartDataNum], this->processDataNum);
        DataCopy(endLocal, endGm[progress * this->ubPartDataNum], this->processDataNum);
        DataCopy(weightLocal, weightGm[progress * this->ubPartDataNum], this->processDataNum);

        inQueueStart.EnQue(startLocal);
        inQueueEnd.EnQue(endLocal);
        inQueueWeight.EnQue(weightLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<TYPE_START> startLocal = inQueueStart.DeQue<TYPE_START>();
        LocalTensor<TYPE_END> endLocal = inQueueEnd.DeQue<TYPE_END>();
        LocalTensor<TYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<TYPE_WEIGHT>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<TYPE_START, float>) {
            // For float: y = start + weight * (end - start)
            Sub(yLocal, endLocal, startLocal, this->processDataNum);
            Mul(yLocal, weightLocal, yLocal, this->processDataNum);
            Add(yLocal, yLocal, startLocal, this->processDataNum);
        } else if constexpr (std::is_same_v<TYPE_START, half> || std::is_same_v<TYPE_START, bfloat16_t>) {
            // FP16 high precision path
            auto fstart = tmp1.Get<float>();
            auto fend = tmp2.Get<float>();

            // Convert to float with no rounding
            Cast(fstart, startLocal, RoundMode::CAST_NONE, this->processDataNum);
            Cast(fend, endLocal, RoundMode::CAST_NONE, this->processDataNum);

            // High precision computation: fy = start + weight * (end - start)
            Sub(fend, fend, fstart, this->processDataNum);
            Cast(fstart, weightLocal, RoundMode::CAST_NONE, this->processDataNum);
            Mul(fend, fstart, fend, this->processDataNum);
            Cast(fstart, startLocal, RoundMode::CAST_NONE, this->processDataNum);
            Add(fend, fend, fstart, this->processDataNum);

            // Convert back to FP16 with rounding to nearest even
            Cast(yLocal, fend, RoundMode::CAST_RINT, this->processDataNum);
        }
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueStart.FreeTensor(startLocal);
        inQueueEnd.FreeTensor(endLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueStart;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueEnd;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWeight;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1;
    TBuf<QuePosition::VECCALC> tmp2;
    GlobalTensor<TYPE_START> startGm;
    GlobalTensor<TYPE_END> endGm;
    GlobalTensor<TYPE_WEIGHT> weightGm;
    GlobalTensor<TYPE_Y> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};
} // namespace NsLerp
#endif
