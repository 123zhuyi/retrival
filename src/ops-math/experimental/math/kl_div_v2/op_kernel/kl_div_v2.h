/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KLDIVV2_H
#define KLDIVV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kl_div_v2_tiling_data.h"
#include "kl_div_v2_tiling_key.h"

namespace NsKLDivV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float KLDIV_TINY = 1.18e-38f;
constexpr uint32_t WORK_GM_SIZE = 8;

template <typename T>
class KLDivV2 {
public:
    __aicore__ inline KLDivV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR workspace,
                                const KLDivV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t num);
    __aicore__ inline void ComputePointwise(uint32_t num);
    __aicore__ inline void CopyOut(int32_t progress, uint32_t num);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueT;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcXBuf;
    TBuf<QuePosition::VECCALC> calcTBuf;
    TBuf<QuePosition::VECCALC> calcResBuf;
    TBuf<QuePosition::VECCALC> accumBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMT;
    GlobalTensor<T> outputGMY;
    GlobalTensor<float> workGM;

    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t reduction_ = 0;
    uint32_t logTarget_ = 0;
    float cof_ = 1.0f;
};

template <typename T>
__aicore__ inline void KLDivV2<T>::Init(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR workspace,
                                        const KLDivV2TilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t blockIdx = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * blockIdx;
    this->tileDataNum_ = tilingData->tileDataNum;
    this->reduction_ = tilingData->reduction;
    this->logTarget_ = tilingData->logTarget;
    this->cof_ = tilingData->cof;
    if (blockIdx < tilingData->tailBlockNum) {
        this->coreDataNum_ = tilingData->bigCoreDataNum;
        this->tileNum_ = tilingData->finalBigTileNum;
        this->tailDataNum_ = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum_ = tilingData->smallCoreDataNum;
        this->tileNum_ = tilingData->finalSmallTileNum;
        this->tailDataNum_ = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (blockIdx - tilingData->tailBlockNum);
    }
    inputGMX.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum_);
    inputGMT.SetGlobalBuffer((__gm__ T*)target + globalBufferIndex, this->coreDataNum_);
    if (this->reduction_ == 0) {
        outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum_);
    } else {
        outputGMY.SetGlobalBuffer((__gm__ T*)y, 1);
        workGM.SetGlobalBuffer((__gm__ float*)workspace, GetBlockNum());
    }

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum_ * sizeof(T));
    pipe.InitBuffer(inQueueT, BUFFER_NUM, this->tileDataNum_ * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum_ * sizeof(T));
    pipe.InitBuffer(calcResBuf, this->tileDataNum_ * sizeof(float));
    pipe.InitBuffer(calcXBuf, this->tileDataNum_ * sizeof(float));
    if constexpr (!std::is_same_v<T, float>) {
        pipe.InitBuffer(calcTBuf, this->tileDataNum_ * sizeof(float));
    }
    if (this->reduction_ != 0) {
        pipe.InitBuffer(accumBuf, this->tileDataNum_ * sizeof(float));
    }
    pipe.InitBuffer(reduceBuf, WORK_GM_SIZE * sizeof(float));
}

template <typename T>
__aicore__ inline void KLDivV2<T>::CopyIn(int32_t progress, uint32_t num)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    LocalTensor<T> tLocal = inQueueT.AllocTensor<T>();
    DataCopyParams copyParams{1, (uint16_t)(num * sizeof(T)), 0, 0};
    DataCopyPad(xLocal, inputGMX[progress * this->tileDataNum_], copyParams, {false, 0, 0, 0});
    DataCopyPad(tLocal, inputGMT[progress * this->tileDataNum_], copyParams, {false, 0, 0, 0});
    inQueueX.EnQue(xLocal);
    inQueueT.EnQue(tLocal);
}

template <typename T>
__aicore__ inline void KLDivV2<T>::ComputePointwise(uint32_t num)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> tLocal = inQueueT.DeQue<T>();
    LocalTensor<float> res = calcResBuf.Get<float>();

    LocalTensor<float> xf;
    LocalTensor<float> tf;
    if constexpr (std::is_same_v<T, float>) {
        xf = xLocal;
        tf = tLocal;
    } else {
        xf = calcXBuf.Get<float>();
        tf = calcTBuf.Get<float>();
        Cast(xf, xLocal, RoundMode::CAST_NONE, num);
        Cast(tf, tLocal, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
    }

    if (this->logTarget_ == 1) {
        Exp(res, tf, num);
        Sub(xf, tf, xf, num);
        Mul(res, res, xf, num);
    } else {
        Maxs(res, tf, KLDIV_TINY, num);
        Log(res, res, num);
        Sub(res, res, xf, num);
        Mul(res, tf, res, num);
        Maxs(xf, tf, 0.0f, num);
        Adds(tf, xf, KLDIV_TINY, num);
        Div(xf, xf, tf, num);
        Mul(res, res, xf, num);
    }
    PipeBarrier<PIPE_V>();

    if (this->reduction_ == 0) {
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(yLocal, res, 0.0f, num);
        } else {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Cast(yLocal, res, RoundMode::CAST_RINT, num);
            } else {
                Cast(yLocal, res, RoundMode::CAST_NONE, num);
            };
        }
        outQueueY.EnQue<T>(yLocal);
    }
    inQueueX.FreeTensor(xLocal);
    inQueueT.FreeTensor(tLocal);
}

template <typename T>
__aicore__ inline void KLDivV2<T>::CopyOut(int32_t progress, uint32_t num)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyParams copyParams{1, (uint16_t)(num * sizeof(T)), 0, 0};
    DataCopyPad(outputGMY[progress * this->tileDataNum_], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void KLDivV2<T>::Process()
{
    int32_t loopCount = this->tileNum_;
    if (this->reduction_ == 0) {
        uint32_t num = this->tileDataNum_;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1) {
                num = this->tailDataNum_;
            }
            CopyIn(i, num);
            ComputePointwise(num);
            CopyOut(i, num);
        }
        return;
    }

    uint32_t blockNum = GetBlockNum();

    LocalTensor<float> accum = accumBuf.Get<float>();
    Duplicate(accum, 0.0f, this->tileDataNum_);
    PipeBarrier<PIPE_V>();

    uint32_t num = this->tileDataNum_;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == loopCount - 1) {
            num = this->tailDataNum_;
        }
        CopyIn(i, num);
        ComputePointwise(num);
        LocalTensor<float> res = calcResBuf.Get<float>();
        Add(accum, accum, res, num);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> work = calcXBuf.Get<float>();
    LocalTensor<float> reduceOut = reduceBuf.Get<float>();
    ReduceSum(reduceOut, accum, work, this->tileDataNum_);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    float coreSum = reduceOut.GetValue(0);

    if (blockNum == 1) {
        float total = coreSum * this->cof_;
        LocalTensor<float> resf = reduceBuf.Get<float>();
        resf.SetValue(0, total);
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        if constexpr (std::is_same_v<T, float>) {
            Adds(yLocal, resf, 0.0f, 1);
        } else {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Cast(yLocal, resf, RoundMode::CAST_RINT, 1);
            } else {
                Cast(yLocal, resf, RoundMode::CAST_NONE, 1);
            };
        }
        outQueueY.EnQue<T>(yLocal);
        LocalTensor<T> outLocal = outQueueY.DeQue<T>();
        DataCopyParams copyParams{1, (uint16_t)(sizeof(T)), 0, 0};
        DataCopyPad(outputGMY, outLocal, copyParams);
        outQueueY.FreeTensor(outLocal);
        return;
    }

    LocalTensor<float> partial = reduceBuf.Get<float>();
    partial.SetValue(0, coreSum);
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyPad(workGM[GetBlockIdx()], partial, {1, (uint16_t)sizeof(float), 0, 0});
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
    SyncAll();

    if (GetBlockIdx() == 0) {
        float total = 0.0f;
        for (uint32_t c = 0; c < blockNum; c++) {
            total += workGM.GetValue(c);
        }
        total *= this->cof_;
        LocalTensor<float> resf = reduceBuf.Get<float>();
        resf.SetValue(0, total);
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        if constexpr (std::is_same_v<T, float>) {
            Adds(yLocal, resf, 0.0f, 1);
        } else {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Cast(yLocal, resf, RoundMode::CAST_RINT, 1);
            } else {
                Cast(yLocal, resf, RoundMode::CAST_NONE, 1);
            };
        }
        outQueueY.EnQue<T>(yLocal);
        LocalTensor<T> outLocal = outQueueY.DeQue<T>();
        DataCopyParams copyParams{1, (uint16_t)(sizeof(T)), 0, 0};
        DataCopyPad(outputGMY, outLocal, copyParams);
        outQueueY.FreeTensor(outLocal);
    }
}

} // namespace NsKLDivV2
#endif // KLDIVV2_H
