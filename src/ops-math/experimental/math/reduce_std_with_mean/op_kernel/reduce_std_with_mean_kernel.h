/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_STD_WITH_MEAN_H
#define REDUCE_STD_WITH_MEAN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_std_with_mean_tiling_data.h"
#include "reduce_std_with_mean_tiling_key.h"

namespace NsReduceStdWithMean {
using namespace AscendC;

template <typename T, int schMode>
class ReduceStdWithMean {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline ReduceStdWithMean() {}
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR mean, GM_ADDR output, GM_ADDR workspace,
                                const ReduceStdWithMeanTilingData* tiling)
    {
        totalNonReduce_ = tiling->totalNonReduce;
        reduceLength_ = tiling->reduceLength;
        blockFactor_ = tiling->blockFactor;
        ubLength_ = tiling->ubLength;
        correction_ = tiling->correction;
        eps_ = tiling->eps;
        invert_ = tiling->invert;

        int64_t blockIdx = AscendC::GetBlockIdx();
        coreStartM_ = blockIdx * blockFactor_;
        int64_t rem = totalNonReduce_ - coreStartM_;
        coreM_ = (rem > blockFactor_) ? blockFactor_ : rem;
        if (coreM_ < 0) {
            coreM_ = 0;
        }

        selfGM.SetGlobalBuffer((__gm__ T*)self + coreStartM_ * reduceLength_, coreM_ * reduceLength_);
        meanGM.SetGlobalBuffer((__gm__ T*)mean + coreStartM_ * reduceLength_, coreM_ * reduceLength_);
        outputGM.SetGlobalBuffer((__gm__ T*)output + coreStartM_, coreM_);

        pipe.InitBuffer(inQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(inQueueMean, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(T));
        pipe.InitBuffer(reduceBuf, 8 * sizeof(float));
        pipe.InitBuffer(scratchBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(workBuf, ubLength_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>)
            pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
    }
    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteOutput(int64_t m, float stdVal);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMean;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> reduceBuf, scratchBuf, workBuf, castBuf;
    GlobalTensor<T> selfGM, meanGM, outputGM;

    int64_t totalNonReduce_, reduceLength_, blockFactor_, ubLength_;
    int64_t correction_;
    float eps_;
    bool invert_;
    int64_t coreStartM_, coreM_;
};

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMean<T, schMode>::Process()
{
    if (coreM_ <= 0 || reduceLength_ <= 0) {
        return;
    }
    int64_t numTiles = (reduceLength_ + ubLength_ - 1) / ubLength_;

    // Unified Two-Pass for both StdMeanCorrection and BatchNormStats.
    // Uses hardware Sub/Mul/ReduceSum (tree-reduction) for precision.
    // meanGM provides pre-computed broadcast mean.
    for (int64_t m = 0; m < coreM_; ++m) {
        float sum_sqr = 0.0f;

        for (int64_t i = 0; i < numTiles; ++i) {
            int64_t curLen = ((i + 1) * ubLength_ <= reduceLength_) ? ubLength_ : (reduceLength_ - i * ubLength_);
            uint32_t uCurLen = static_cast<uint32_t>(curLen);

            LocalTensor<T> sLocal = inQueueSelf.AllocTensor<T>();
            AscendC::DataCopyPad(sLocal, selfGM[m * reduceLength_ + i * ubLength_],
                                 {1, static_cast<uint16_t>(uCurLen * sizeof(T)), 0, 0}, {false, 0, 0, 0});
            inQueueSelf.EnQue(sLocal);
            sLocal = inQueueSelf.DeQue<T>();

            LocalTensor<T> mLocal = inQueueMean.AllocTensor<T>();
            AscendC::DataCopyPad(mLocal, meanGM[m * reduceLength_ + i * ubLength_],
                                 {1, static_cast<uint16_t>(uCurLen * sizeof(T)), 0, 0}, {false, 0, 0, 0});
            inQueueMean.EnQue(mLocal);
            mLocal = inQueueMean.DeQue<T>();

            LocalTensor<float> dBuf = reduceBuf.Get<float>();
            LocalTensor<float> tBuf = scratchBuf.Get<float>();
            LocalTensor<float> wBuf = workBuf.Get<float>();

            if constexpr (std::is_same_v<T, float>) {
                Sub(tBuf, sLocal, mLocal, uCurLen);
                PipeBarrier<PIPE_V>();
                Mul(wBuf, tBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                ReduceSum(dBuf, wBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> fBuf = castBuf.Get<float>();
                Cast(fBuf, sLocal, RoundMode::CAST_NONE, uCurLen);
                Cast(wBuf, mLocal, RoundMode::CAST_NONE, uCurLen);
                PipeBarrier<PIPE_V>();
                Sub(tBuf, fBuf, wBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                Mul(wBuf, tBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                ReduceSum(dBuf, wBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
            }

            sum_sqr += dBuf.GetValue(0);
            inQueueSelf.FreeTensor(sLocal);
            inQueueMean.FreeTensor(mLocal);
        }

        float denom = static_cast<float>(reduceLength_) - static_cast<float>(correction_);
        if (denom < 0.0f)
            denom = 0.0f;
        float var = (denom > 0.0f) ? (sum_sqr / denom) : 0.0f;
        float stdVal;
        if (invert_) {
            float stdTmp = sqrt(var + eps_);
            stdVal = (stdTmp > 0.0f) ? (1.0f / stdTmp) : 0.0f;
        } else {
            stdVal = sqrt(var);
        }
        WriteOutput(m, stdVal);
    }
}

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMean<T, schMode>::WriteOutput(int64_t m, float stdVal)
{
    LocalTensor<T> outLocal = outQueueY.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        outLocal.SetValue(0, stdVal);
    } else {
        LocalTensor<float> fBuf = reduceBuf.Get<float>();
        fBuf.SetValue(0, stdVal);
        Cast(outLocal, fBuf, RoundMode::CAST_ROUND, 1);
    }
    outQueueY.EnQue(outLocal);
    LocalTensor<T> outData = outQueueY.DeQue<T>();
    AscendC::DataCopyExtParams outCp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGM[m], outData, outCp);
    outQueueY.FreeTensor(outData);
}

} // namespace NsReduceStdWithMean
#endif
