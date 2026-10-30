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
 * \file rsqrt.h
 * \brief
 */
#ifndef __RSQRT_H__
#define __RSQRT_H__

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rsqrt_tiling_data.h"
#include "rsqrt_tiling_key.h"

namespace NsRsqrt {

using namespace AscendC;

template <typename TYPE_X, uint64_t BUFFER_NUM>
class KernelRsqrt {
    using T = std::conditional_t<std::is_same_v<TYPE_X, bool>, int8_t, TYPE_X>;

public:
    __aicore__ inline KernelRsqrt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RsqrtTilingData* tilingData, TPipe* pipeIn)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t blockIdx = GetBlockIdx();
        uint64_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
        this->tileDataNum = tilingData->tileDataNum;
        this->pipe = pipeIn;
        if (blockIdx < tilingData->tailBlockNum) {
            this->coreDataNum = tilingData->bigCoreDataNum;
            this->tileNum = tilingData->finalBigTileNum;
            this->tailDataNum = tilingData->bigTailDataNum;
        } else {
            this->coreDataNum = tilingData->smallCoreDataNum;
            this->tileNum = tilingData->finalSmallTileNum;
            this->tailDataNum = tilingData->smallTailDataNum;
            globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                                 (GetBlockIdx() - tilingData->tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        // For bool type, outQueueY buffer must be sized for int16_t access in ComputeBool
        // to avoid out-of-bounds write when processDataNum is odd
        if constexpr (std::is_same_v<TYPE_X, bool>) {
            pipe->InitBuffer(outQueueY, BUFFER_NUM, ((this->tileDataNum + 1) >> 1) * sizeof(int16_t));
        } else {
            pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        }

        if constexpr (std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, float>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(TYPE_X));
            LocalTensor<TYPE_X> oneLocal = tmp1.Get<TYPE_X>();
            Duplicate<TYPE_X>(oneLocal, (TYPE_X)1.0f, this->tileDataNum);
        }
        if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(float));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(float));
            LocalTensor<float> oneLocal = tmp1.Get<float>();
            Duplicate<float>(oneLocal, (float)1.0f, this->tileDataNum);
        }
        if constexpr (std::is_same_v<TYPE_X, int8_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
            pipe->InitBuffer(tmp2, this->tileDataNum * sizeof(half));
        }
        if constexpr (std::is_same_v<TYPE_X, uint8_t>) {
            pipe->InitBuffer(tmp1, this->tileDataNum * sizeof(half));
        }
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum - 1;
        this->processDataNum = this->tileDataNum;
        uint64_t offset = 0;
        for (int32_t i = 0; i < loopCount; i++, offset += this->tileDataNum) {
            CopyIn(offset);
            Compute(offset);
            CopyOut(offset);
        }
        this->processDataNum = this->tailDataNum;

        CopyIn(offset);
        Compute(offset);
        CopyOut(offset);
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset)
    {
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

        DataCopy(xLocal, xGm[offset], this->processDataNum);

        inQueueX.template EnQue<T>(xLocal);
    }

private:
    __aicore__ inline void Compute(uint64_t offset)
    {
        LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
        LocalTensor<T> xLocal = inQueueX.template DeQue<T>();

        ComputeImpl(yLocal, xLocal);

        outQueueY.template EnQue<T>(yLocal);
        inQueueX.template FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeImpl(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        if constexpr (std::is_same_v<TYPE_X, half> || std::is_same_v<TYPE_X, float>) {
            ComputeFloat(yLocal, xLocal);
        } else if constexpr (std::is_same_v<TYPE_X, bfloat16_t>) {
            ComputeBfloat16(yLocal, xLocal);
        } else if constexpr (std::is_same_v<TYPE_X, uint8_t>) {
            ComputeUint8(yLocal, xLocal);
        } else if constexpr (std::is_same_v<TYPE_X, bool>) {
            ComputeBool(yLocal);
        } else if constexpr (std::is_same_v<TYPE_X, int32_t>) {
            ComputeInt32(yLocal, xLocal);
        } else if constexpr (std::is_same_v<TYPE_X, int16_t>) {
            ComputeInt16(yLocal, xLocal);
        } else if constexpr (std::is_same_v<TYPE_X, int8_t>) {
            ComputeInt8(yLocal, xLocal);
        }
    }

    __aicore__ inline void ComputeFloat(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        LocalTensor<T> oneLocal = tmp1.Get<T>();
        Sqrt(xLocal, xLocal, this->processDataNum);
        Div(yLocal, oneLocal, xLocal, this->processDataNum);
    }

    __aicore__ inline void ComputeBfloat16(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        LocalTensor<float> oneLocal = tmp1.Get<float>();
        LocalTensor<float> xLocalFp = tmp2.Get<float>();
        Cast(xLocalFp, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Sqrt(xLocalFp, xLocalFp, this->processDataNum);
        Div(xLocalFp, oneLocal, xLocalFp, this->processDataNum);
        Cast(yLocal, xLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    }

    __aicore__ inline void ComputeUint8(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        LocalTensor<half> xLocalHalf = tmp1.Get<half>();
        Cast(xLocalHalf, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Rsqrt(xLocalHalf, xLocalHalf, this->processDataNum);
        Cast(yLocal, xLocalHalf, RoundMode::CAST_RINT, this->processDataNum);
    }

    __aicore__ inline void ComputeBool(LocalTensor<T>& yLocal)
    {
        LocalTensor<int16_t> p1 = yLocal.template ReinterpretCast<int16_t>();
        Duplicate<int16_t>(p1, 0x0101, (this->processDataNum + 1) >> 1);
    }

    __aicore__ inline void ComputeInt32(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        LocalTensor<float> p1 = xLocal.template ReinterpretCast<float>();
        LocalTensor<float> p2 = yLocal.template ReinterpretCast<float>();
        Cast(p2, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Rsqrt(p1, p2, this->processDataNum);
        Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
    }

    __aicore__ inline void ComputeInt16(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        LocalTensor<half> p1 = xLocal.template ReinterpretCast<half>();
        LocalTensor<half> p2 = yLocal.template ReinterpretCast<half>();
        Cast(p2, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Rsqrt(p1, p2, this->processDataNum);
        ComputeIntCorrection(p1, p2);
        Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
    }

    __aicore__ inline void ComputeInt8(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal)
    {
        LocalTensor<half> p1 = tmp1.Get<half>();
        LocalTensor<half> p2 = tmp2.Get<half>();
        Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
        Rsqrt(p1, p1, this->processDataNum);
        ComputeIntCorrection(p1, p2);
        Cast(yLocal, p1, RoundMode::CAST_RINT, this->processDataNum);
    }

    // Compute integer correction: p1 = min(p1, 2) - 3 * max(0, min(p1, 2) - 1)
    // Output range: [-1, 2]. Used for activation/value conditioning.
    __aicore__ inline void ComputeIntCorrection(LocalTensor<half>& p1, LocalTensor<half>& p2)
    {
        Mins(p1, p1, (half)2.0f, this->processDataNum);  // p1 = min(p1, 2)
        Adds(p2, p1, (half)-1.0f, this->processDataNum); // p2 = p1 - 1
        Maxs(p2, p2, (half)0.0f, this->processDataNum);  // p2 = max(p2, 0)
        Muls(p2, p2, (half)3.0f, this->processDataNum);  // p2 = p2 * 3
        Sub(p1, p1, p2, this->processDataNum);           // p1 = p1 - p2
    }

    __aicore__ inline void CopyOut(uint64_t offset)
    {
        LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

        DataCopy(yGm[offset], yLocal, this->processDataNum);

        outQueueY.template FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
};

} // namespace NsRsqrt

#endif // RSQRT_H
