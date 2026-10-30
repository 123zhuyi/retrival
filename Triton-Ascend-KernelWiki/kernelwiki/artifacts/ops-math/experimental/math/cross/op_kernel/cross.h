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
 * \file cross.h
 * \brief
 * */
#ifndef CROSS_H
#define CROSS_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cross_tiling_data.h"
#include "cross_tiling_key.h"

namespace NsCross {

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 1;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Cross {
public:
    __aicore__ inline Cross() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const CrossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool IsGroupMode() const;
    __aicore__ inline void GetTileWork(int64_t tileIndex, int64_t& baseIdx, int64_t& count) const;
    __aicore__ inline void CopyIn(int64_t baseIdx, int64_t count);
    __aicore__ inline void Compute(int64_t count);
    __aicore__ inline void CopyOut(int64_t baseIdx, int64_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> x0Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> x1Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> x2Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> y0Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> y1Queue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> y2Queue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> z0Queue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> z1Queue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> z2Queue;
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpBuf3;
    TBuf<QuePosition::VECCALC> tmpBuf4;
    TBuf<QuePosition::VECCALC> tmpBuf5;
    TBuf<QuePosition::VECCALC> tmpBuf6;
    TBuf<QuePosition::VECCALC> tmpBuf7;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;
    int64_t intervalNum = 0;
    int64_t loopTimes = 0;
    int64_t tileDataNum = 1;
    int64_t tilesPerLoop = 1;
    int64_t coreTileStart = 0;
    int64_t coreTileCount = 0;
    uint64_t totalElements = 0;
};

template <typename T>
__aicore__ inline void Cross<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const CrossTilingData* tilingData)
{
    this->intervalNum = tilingData->intervalNum;
    this->loopTimes = tilingData->loopTimes;
    this->tileDataNum = (tilingData->tileDataNum > 0) ? static_cast<int64_t>(tilingData->tileDataNum) : 1;
    this->totalElements = static_cast<uint64_t>(this->loopTimes) * static_cast<uint64_t>(this->intervalNum) * 3ULL;
    inputGMX.SetGlobalBuffer((__gm__ T*)x, this->totalElements);
    inputGMY.SetGlobalBuffer((__gm__ T*)y, this->totalElements);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z, this->totalElements);

    this->tilesPerLoop = (this->intervalNum + this->tileDataNum - 1) / this->tileDataNum;
    if (this->IsGroupMode()) {
        this->tilesPerLoop = (this->loopTimes + this->tileDataNum - 1) / this->tileDataNum;
    }
    if (this->tilesPerLoop <= 0) {
        this->tilesPerLoop = 1;
    }

    int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    if (blockNum <= 0) {
        blockNum = 1;
    }

    int64_t totalTileCount = this->loopTimes * this->tilesPerLoop;
    if (this->IsGroupMode()) {
        totalTileCount = this->tilesPerLoop;
    }
    int64_t tilesPerCore = totalTileCount / blockNum;
    int64_t tailCores = totalTileCount % blockNum;
    if (blockIdx < tailCores) {
        this->coreTileCount = tilesPerCore + 1;
        this->coreTileStart = blockIdx * this->coreTileCount;
    } else {
        this->coreTileCount = tilesPerCore;
        this->coreTileStart = tailCores * (tilesPerCore + 1) + (blockIdx - tailCores) * tilesPerCore;
    }

    if (this->IsGroupMode()) {
        uint32_t groupBufBytes = static_cast<uint32_t>(this->tileDataNum * 3 * static_cast<int64_t>(sizeof(T)));
        pipe.InitBuffer(x0Queue, BUFFER_NUM, groupBufBytes);
        pipe.InitBuffer(y0Queue, BUFFER_NUM, groupBufBytes);
        pipe.InitBuffer(z0Queue, BUFFER_NUM, groupBufBytes);
    } else {
        uint32_t bufBytes = static_cast<uint32_t>(this->tileDataNum * static_cast<int64_t>(sizeof(T)));
        pipe.InitBuffer(x0Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(x1Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(y0Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(y1Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(y2Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(z0Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(z1Queue, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(z2Queue, BUFFER_NUM, bufBytes);
        if constexpr (std::is_same_v<T, half>) {
            uint32_t fp32BufBytes = static_cast<uint32_t>(this->tileDataNum * static_cast<int64_t>(sizeof(float)));
            pipe.InitBuffer(tmpBuf0, fp32BufBytes);
            pipe.InitBuffer(tmpBuf1, fp32BufBytes);
            pipe.InitBuffer(tmpBuf2, fp32BufBytes);
            pipe.InitBuffer(tmpBuf3, fp32BufBytes);
            pipe.InitBuffer(tmpBuf4, fp32BufBytes);
            pipe.InitBuffer(tmpBuf5, fp32BufBytes);
            pipe.InitBuffer(tmpBuf6, fp32BufBytes);
            pipe.InitBuffer(tmpBuf7, fp32BufBytes);
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
            pipe.InitBuffer(tmpBuf0, bufBytes);
        }
    }
}

template <typename T>
__aicore__ inline bool Cross<T>::IsGroupMode() const
{
    return this->intervalNum == 1;
}

template <typename T>
__aicore__ inline void Cross<T>::GetTileWork(int64_t tileIndex, int64_t& baseIdx, int64_t& count) const
{
    if (this->IsGroupMode()) {
        int64_t groupOffset = tileIndex * this->tileDataNum;
        count = this->loopTimes - groupOffset;
        if (count > this->tileDataNum) {
            count = this->tileDataNum;
        }
        baseIdx = groupOffset * 3;
        return;
    }

    int64_t loopIdx = tileIndex / this->tilesPerLoop;
    int64_t intervalOffset = (tileIndex % this->tilesPerLoop) * this->tileDataNum;
    count = this->intervalNum - intervalOffset;
    if (count > this->tileDataNum) {
        count = this->tileDataNum;
    }
    int64_t loopBase = loopIdx * 3 * this->intervalNum;
    baseIdx = loopBase + intervalOffset;
}

template <typename T>
__aicore__ inline void Cross<T>::CopyIn(int64_t baseIdx, int64_t count)
{
    if (this->IsGroupMode()) {
        LocalTensor<T> xLocal = x0Queue.AllocTensor<T>();
        LocalTensor<T> yLocal = y0Queue.AllocTensor<T>();
        int64_t dataCount = count * 3;

        constexpr int64_t kBlockBytes = 32;
        int64_t alignedBytes = (dataCount * static_cast<int64_t>(sizeof(T)) / kBlockBytes) * kBlockBytes;
        int64_t alignedCount = alignedBytes / static_cast<int64_t>(sizeof(T));
        int64_t tailCount = dataCount - alignedCount;

        if (alignedCount > 0) {
            DataCopy(xLocal, inputGMX[baseIdx], alignedCount);
            DataCopy(yLocal, inputGMY[baseIdx], alignedCount);
        }

        if (tailCount > 0) {
            DataCopyExtParams copyInParams{
                1, static_cast<uint32_t>(tailCount * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
            DataCopyPadExtParams<T> padInParams{false, 0, 0, 0};
            DataCopyPad(xLocal[alignedCount], inputGMX[baseIdx + alignedCount], copyInParams, padInParams);
            DataCopyPad(yLocal[alignedCount], inputGMY[baseIdx + alignedCount], copyInParams, padInParams);
        }

        x0Queue.EnQue(xLocal);
        y0Queue.EnQue(yLocal);
        return;
    }

    LocalTensor<T> x0Local = x0Queue.AllocTensor<T>();
    LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
    LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
    LocalTensor<T> y0Local = y0Queue.AllocTensor<T>();
    LocalTensor<T> y1Local = y1Queue.AllocTensor<T>();
    LocalTensor<T> y2Local = y2Queue.AllocTensor<T>();

    constexpr int64_t kBlockBytes = 32;
    int64_t alignedBytes = (count * static_cast<int64_t>(sizeof(T)) / kBlockBytes) * kBlockBytes;
    int64_t alignedCount = alignedBytes / static_cast<int64_t>(sizeof(T));
    int64_t tailCount = count - alignedCount;

    if (alignedCount > 0) {
        DataCopy(x0Local, inputGMX[baseIdx], alignedCount);
        DataCopy(x1Local, inputGMX[baseIdx + this->intervalNum], alignedCount);
        DataCopy(x2Local, inputGMX[baseIdx + 2 * this->intervalNum], alignedCount);
        DataCopy(y0Local, inputGMY[baseIdx], alignedCount);
        DataCopy(y1Local, inputGMY[baseIdx + this->intervalNum], alignedCount);
        DataCopy(y2Local, inputGMY[baseIdx + 2 * this->intervalNum], alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(tailCount * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> padInParams{false, 0, 0, 0};
        DataCopyPad(x0Local[alignedCount], inputGMX[baseIdx + alignedCount], copyInParams, padInParams);
        DataCopyPad(
            x1Local[alignedCount], inputGMX[baseIdx + this->intervalNum + alignedCount], copyInParams, padInParams);
        DataCopyPad(
            x2Local[alignedCount], inputGMX[baseIdx + 2 * this->intervalNum + alignedCount], copyInParams, padInParams);
        DataCopyPad(y0Local[alignedCount], inputGMY[baseIdx + alignedCount], copyInParams, padInParams);
        DataCopyPad(
            y1Local[alignedCount], inputGMY[baseIdx + this->intervalNum + alignedCount], copyInParams, padInParams);
        DataCopyPad(
            y2Local[alignedCount], inputGMY[baseIdx + 2 * this->intervalNum + alignedCount], copyInParams, padInParams);
    }
    x0Queue.EnQue(x0Local);
    x1Queue.EnQue(x1Local);
    x2Queue.EnQue(x2Local);
    y0Queue.EnQue(y0Local);
    y1Queue.EnQue(y1Local);
    y2Queue.EnQue(y2Local);
}

template <typename T>
__aicore__ inline void Cross<T>::Compute(int64_t count)
{
    if (this->IsGroupMode()) {
        LocalTensor<T> xLocal = x0Queue.DeQue<T>();
        LocalTensor<T> yLocal = y0Queue.DeQue<T>();
        LocalTensor<T> zLocal = z0Queue.AllocTensor<T>();

        for (int64_t groupIdx = 0; groupIdx < count; ++groupIdx) {
            int64_t idx = groupIdx * 3;
            if constexpr (std::is_same_v<T, half>) {
                float x0 = static_cast<float>(xLocal.GetValue(idx));
                float x1 = static_cast<float>(xLocal.GetValue(idx + 1));
                float x2 = static_cast<float>(xLocal.GetValue(idx + 2));
                float y0 = static_cast<float>(yLocal.GetValue(idx));
                float y1 = static_cast<float>(yLocal.GetValue(idx + 1));
                float y2 = static_cast<float>(yLocal.GetValue(idx + 2));
                zLocal.SetValue(idx, static_cast<T>(x1 * y2 - x2 * y1));
                zLocal.SetValue(idx + 1, static_cast<T>(x2 * y0 - x0 * y2));
                zLocal.SetValue(idx + 2, static_cast<T>(x0 * y1 - x1 * y0));
            } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                int32_t x0 = static_cast<int32_t>(xLocal.GetValue(idx));
                int32_t x1 = static_cast<int32_t>(xLocal.GetValue(idx + 1));
                int32_t x2 = static_cast<int32_t>(xLocal.GetValue(idx + 2));
                int32_t y0 = static_cast<int32_t>(yLocal.GetValue(idx));
                int32_t y1 = static_cast<int32_t>(yLocal.GetValue(idx + 1));
                int32_t y2 = static_cast<int32_t>(yLocal.GetValue(idx + 2));
                zLocal.SetValue(idx, static_cast<T>(x1 * y2 - x2 * y1));
                zLocal.SetValue(idx + 1, static_cast<T>(x2 * y0 - x0 * y2));
                zLocal.SetValue(idx + 2, static_cast<T>(x0 * y1 - x1 * y0));
            } else {
                T x0 = xLocal.GetValue(idx);
                T x1 = xLocal.GetValue(idx + 1);
                T x2 = xLocal.GetValue(idx + 2);
                T y0 = yLocal.GetValue(idx);
                T y1 = yLocal.GetValue(idx + 1);
                T y2 = yLocal.GetValue(idx + 2);
                zLocal.SetValue(idx, x1 * y2 - x2 * y1);
                zLocal.SetValue(idx + 1, x2 * y0 - x0 * y2);
                zLocal.SetValue(idx + 2, x0 * y1 - x1 * y0);
            }
        }

        z0Queue.EnQue(zLocal);
        x0Queue.FreeTensor(xLocal);
        y0Queue.FreeTensor(yLocal);
        return;
    }

    LocalTensor<T> x0Local = x0Queue.DeQue<T>();
    LocalTensor<T> x1Local = x1Queue.DeQue<T>();
    LocalTensor<T> x2Local = x2Queue.DeQue<T>();
    LocalTensor<T> y0Local = y0Queue.DeQue<T>();
    LocalTensor<T> y1Local = y1Queue.DeQue<T>();
    LocalTensor<T> y2Local = y2Queue.DeQue<T>();
    LocalTensor<T> z0Local = z0Queue.AllocTensor<T>();
    LocalTensor<T> z1Local = z1Queue.AllocTensor<T>();
    LocalTensor<T> z2Local = z2Queue.AllocTensor<T>();

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
        LocalTensor<T> tmp0Local = tmpBuf0.Get<T>();

        AscendC::Mul(z0Local, x1Local, y2Local, count);
        AscendC::Mul(tmp0Local, x2Local, y1Local, count);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(z0Local, z0Local, tmp0Local, count);
        PipeBarrier<PIPE_V>();

        AscendC::Mul(z1Local, x2Local, y0Local, count);
        AscendC::Mul(tmp0Local, x0Local, y2Local, count);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(z1Local, z1Local, tmp0Local, count);
        PipeBarrier<PIPE_V>();

        AscendC::Mul(z2Local, x0Local, y1Local, count);
        AscendC::Mul(tmp0Local, x1Local, y0Local, count);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(z2Local, z2Local, tmp0Local, count);
        PipeBarrier<PIPE_V>();
    } else if constexpr (std::is_same_v<T, half>) {
        LocalTensor<float> x0Fp32 = tmpBuf0.Get<float>();
        LocalTensor<float> x1Fp32 = tmpBuf1.Get<float>();
        LocalTensor<float> x2Fp32 = tmpBuf2.Get<float>();
        LocalTensor<float> y0Fp32 = tmpBuf3.Get<float>();
        LocalTensor<float> y1Fp32 = tmpBuf4.Get<float>();
        LocalTensor<float> y2Fp32 = tmpBuf5.Get<float>();
        LocalTensor<float> tmpA = tmpBuf6.Get<float>();
        LocalTensor<float> tmpB = tmpBuf7.Get<float>();

        AscendC::Cast(x0Fp32, x0Local, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(y0Fp32, y0Local, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(y1Fp32, y1Local, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(y2Fp32, y2Local, AscendC::RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        AscendC::Mul(tmpA, x1Fp32, y2Fp32, count);
        AscendC::Mul(tmpB, x2Fp32, y1Fp32, count);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(tmpA, tmpA, tmpB, count);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(z0Local, tmpA, AscendC::RoundMode::CAST_ROUND, count);

        AscendC::Mul(tmpA, x2Fp32, y0Fp32, count);
        AscendC::Mul(tmpB, x0Fp32, y2Fp32, count);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(tmpA, tmpA, tmpB, count);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(z1Local, tmpA, AscendC::RoundMode::CAST_ROUND, count);

        AscendC::Mul(tmpA, x0Fp32, y1Fp32, count);
        AscendC::Mul(tmpB, x1Fp32, y0Fp32, count);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(tmpA, tmpA, tmpB, count);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(z2Local, tmpA, AscendC::RoundMode::CAST_ROUND, count);
    } else {
        for (int64_t idx = 0; idx < count; ++idx) {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                int32_t x0 = x0Local.GetValue(idx);
                int32_t x1 = x1Local.GetValue(idx);
                int32_t x2 = x2Local.GetValue(idx);
                int32_t y0 = y0Local.GetValue(idx);
                int32_t y1 = y1Local.GetValue(idx);
                int32_t y2 = y2Local.GetValue(idx);
                z0Local.SetValue(idx, static_cast<T>(x1 * y2 - x2 * y1));
                z1Local.SetValue(idx, static_cast<T>(x2 * y0 - x0 * y2));
                z2Local.SetValue(idx, static_cast<T>(x0 * y1 - x1 * y0));
            } else {
                T x0 = x0Local.GetValue(idx);
                T x1 = x1Local.GetValue(idx);
                T x2 = x2Local.GetValue(idx);
                T y0 = y0Local.GetValue(idx);
                T y1 = y1Local.GetValue(idx);
                T y2 = y2Local.GetValue(idx);
                z0Local.SetValue(idx, x1 * y2 - x2 * y1);
                z1Local.SetValue(idx, x2 * y0 - x0 * y2);
                z2Local.SetValue(idx, x0 * y1 - x1 * y0);
            }
        }
    }

    z0Queue.EnQue(z0Local);
    z1Queue.EnQue(z1Local);
    z2Queue.EnQue(z2Local);

    x0Queue.FreeTensor(x0Local);
    x1Queue.FreeTensor(x1Local);
    x2Queue.FreeTensor(x2Local);
    y0Queue.FreeTensor(y0Local);
    y1Queue.FreeTensor(y1Local);
    y2Queue.FreeTensor(y2Local);
}

template <typename T>
__aicore__ inline void Cross<T>::CopyOut(int64_t baseIdx, int64_t count)
{
    if (this->IsGroupMode()) {
        LocalTensor<T> zLocal = z0Queue.DeQue<T>();
        int64_t dataCount = count * 3;

        constexpr int64_t kBlockBytes = 32;
        int64_t alignedBytes = (dataCount * static_cast<int64_t>(sizeof(T)) / kBlockBytes) * kBlockBytes;
        int64_t alignedCount = alignedBytes / static_cast<int64_t>(sizeof(T));
        int64_t tailCount = dataCount - alignedCount;

        if (alignedCount > 0) {
            DataCopy(outputGMZ[baseIdx], zLocal, alignedCount);
        }

        if (tailCount > 0) {
            DataCopyExtParams copyOutParams{
                1, static_cast<uint32_t>(tailCount * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
            DataCopyPad(outputGMZ[baseIdx + alignedCount], zLocal[alignedCount], copyOutParams);
        }

        z0Queue.FreeTensor(zLocal);
        return;
    }

    LocalTensor<T> z0Local = z0Queue.DeQue<T>();
    LocalTensor<T> z1Local = z1Queue.DeQue<T>();
    LocalTensor<T> z2Local = z2Queue.DeQue<T>();

    constexpr int64_t kBlockBytes = 32;
    int64_t alignedBytes = (count * static_cast<int64_t>(sizeof(T)) / kBlockBytes) * kBlockBytes;
    int64_t alignedCount = alignedBytes / static_cast<int64_t>(sizeof(T));
    int64_t tailCount = count - alignedCount;

    if (alignedCount > 0) {
        DataCopy(outputGMZ[baseIdx], z0Local, alignedCount);
        DataCopy(outputGMZ[baseIdx + this->intervalNum], z1Local, alignedCount);
        DataCopy(outputGMZ[baseIdx + 2 * this->intervalNum], z2Local, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(tailCount * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGMZ[baseIdx + alignedCount], z0Local[alignedCount], copyOutParams);
        DataCopyPad(outputGMZ[baseIdx + this->intervalNum + alignedCount], z1Local[alignedCount], copyOutParams);
        DataCopyPad(outputGMZ[baseIdx + 2 * this->intervalNum + alignedCount], z2Local[alignedCount], copyOutParams);
    }

    z0Queue.FreeTensor(z0Local);
    z1Queue.FreeTensor(z1Local);
    z2Queue.FreeTensor(z2Local);
}

template <typename T>
__aicore__ inline void Cross<T>::Process()
{
    if (this->coreTileCount <= 0) {
        return;
    }

    int64_t currentBaseIdx = 0;
    int64_t currentCount = 0;
    GetTileWork(this->coreTileStart, currentBaseIdx, currentCount);
    CopyIn(currentBaseIdx, currentCount);

    for (int64_t tileOffset = 0; tileOffset + 1 < this->coreTileCount; ++tileOffset) {
        Compute(currentCount);

        int64_t nextBaseIdx = 0;
        int64_t nextCount = 0;
        GetTileWork(this->coreTileStart + tileOffset + 1, nextBaseIdx, nextCount);
        CopyIn(nextBaseIdx, nextCount);

        CopyOut(currentBaseIdx, currentCount);
        currentBaseIdx = nextBaseIdx;
        currentCount = nextCount;
    }

    Compute(currentCount);
    CopyOut(currentBaseIdx, currentCount);
}
} // namespace NsCross
#endif // CROSS_H
