/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file log_space.h
 * \brief LogSpace Kernel 类定义（arch35）
 *
 * 模板参数：
 *   - T:    输出数据类型（half / float / bfloat16_t）
 *   - MODE: 0 = NORMAL（steps>=2），1 = SINGLE（steps==0/1）
 *
 * 算法（NORMAL）：
 *   idx = ArithProgression(firstValue=base_idx, diffValue=1, count=N)
 *   val_fp32 = idx * stepF + startF
 *   val_fp32 = val_fp32 * logBase
 *   val_fp32 = Exp(val_fp32)
 *   out = Cast<T>(val_fp32)  [T==float 时直接搬出]
 *   DataCopyPad UB -> GM
 */
#ifndef LOG_SPACE_H
#define LOG_SPACE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log_space_tiling_data.h"
#include "log_space_tiling_key.h"

namespace NsLogSpace {

using namespace AscendC;

template <typename T, int MODE>
class LogSpace {
public:
    __aicore__ inline LogSpace() {}

    __aicore__ inline void Init(GM_ADDR result, const LogSpaceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void ProcessSingle();
    __aicore__ inline void ComputeChunk(int64_t chunkBase, int64_t currentNum);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> valBuf_;
    TQue<QuePosition::VECOUT, 2> outQueue_;

    GlobalTensor<T> outGM_;

    uint64_t totalLen_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t tileLen_ = 0;
    uint32_t tailCoreIdx_ = 0;
    uint32_t tailTileLen_ = 0;
    uint32_t ubChunk_ = 0;
    float startF_ = 0.0f;
    float stepF_ = 0.0f;
    float logBase_ = 0.0f;

    int64_t idxStart_ = 0;
    int64_t blockLen_ = 0;
};

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::Init(GM_ADDR result, const LogSpaceTilingData* tilingData)
{
    totalLen_ = tilingData->totalLen;
    coreNum_ = tilingData->coreNum;
    tileLen_ = tilingData->tileLen;
    tailCoreIdx_ = tilingData->tailCoreIdx;
    tailTileLen_ = tilingData->tailTileLen;
    ubChunk_ = tilingData->ubChunk;
    startF_ = tilingData->startF;
    stepF_ = tilingData->stepF;
    logBase_ = tilingData->logBase;

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    idxStart_ = blockIdx * static_cast<int64_t>(tileLen_);
    if (blockIdx == static_cast<int64_t>(tailCoreIdx_)) {
        blockLen_ = static_cast<int64_t>(tailTileLen_);
    } else if (blockIdx < static_cast<int64_t>(coreNum_)) {
        blockLen_ = static_cast<int64_t>(tileLen_);
    } else {
        blockLen_ = 0;
    }

    outGM_.SetGlobalBuffer((__gm__ T*)result + idxStart_, blockLen_);

    // UB 分配：index (fp32) + val (fp32)，out 队列按 T 分配
    pipe.InitBuffer(idxBuf_, ubChunk_ * sizeof(float));
    pipe.InitBuffer(valBuf_, ubChunk_ * sizeof(float));
    pipe.InitBuffer(outQueue_, 2, ubChunk_ * sizeof(T));
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ComputeChunk(int64_t chunkBase, int64_t currentNum)
{
    LocalTensor<float> idxLocal = idxBuf_.Get<float>();
    LocalTensor<float> valLocal = valBuf_.Get<float>();
    const float firstIdx = static_cast<float>(idxStart_ + chunkBase);
    const int32_t n = static_cast<int32_t>(currentNum);

    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    if constexpr (std::is_same_v<T, float>) {
        // fp32 路径：直接在 outLocal 上以 fp32 计算，省去中间 valBuf + DataCopy
        AscendC::ArithProgression<float>(outLocal, firstIdx, 1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(outLocal, outLocal, stepF_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float>(outLocal, outLocal, startF_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(outLocal, outLocal, logBase_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(outLocal, outLocal, n);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        // fp16 / bf16 路径：在 valBuf 中以 fp32 计算，最后 Cast RINT 到目标 dtype
        AscendC::ArithProgression<float>(idxLocal, firstIdx, 1.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(valLocal, idxLocal, stepF_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds<float>(valLocal, valLocal, startF_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float>(valLocal, valLocal, logBase_, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(valLocal, valLocal, n);
        AscendC::PipeBarrier<PIPE_V>();
        // CAST_RINT: IEEE 754 "Round to Nearest, ties to Even"（与 PyTorch 默认一致），
        // 适用于 float -> half/bf16 的精度最优舍入。
        AscendC::Cast<T, float>(outLocal, valLocal, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
    }
    outQueue_.EnQue(outLocal);

    LocalTensor<T> outDq = outQueue_.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM_[chunkBase], outDq, copyParams);
    outQueue_.FreeTensor(outDq);
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ProcessNormal()
{
    if (blockLen_ <= 0) {
        return;
    }
    const int64_t chunk = static_cast<int64_t>(ubChunk_);
    int64_t processed = 0;
    while (processed < blockLen_) {
        int64_t cur = blockLen_ - processed;
        if (cur > chunk)
            cur = chunk;
        ComputeChunk(processed, cur);
        processed += cur;
    }
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::ProcessSingle()
{
    // MODE=1: 仅核 0 写 1 个元素（steps==1）；steps==0 时 blockLen_ 为 0，直接返回
    if (AscendC::GetBlockIdx() != 0 || blockLen_ <= 0) {
        return;
    }
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        AscendC::Duplicate<float>(outLocal, startF_ * logBase_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(outLocal, outLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> valLocal = valBuf_.Get<float>();
        AscendC::Duplicate<float>(valLocal, startF_ * logBase_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp<float, false>(valLocal, valLocal, 1);
        AscendC::PipeBarrier<PIPE_V>();
        // CAST_RINT: IEEE 754 "Round to Nearest, ties to Even"，详见 ComputeChunk 注释。
        AscendC::Cast<T, float>(outLocal, valLocal, AscendC::RoundMode::CAST_RINT, 1);
        AscendC::PipeBarrier<PIPE_V>();
    }
    outQueue_.EnQue(outLocal);
    LocalTensor<T> outDq = outQueue_.template DeQue<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outGM_[0], outDq, copyParams);
    outQueue_.FreeTensor(outDq);
}

template <typename T, int MODE>
__aicore__ inline void LogSpace<T, MODE>::Process()
{
    if constexpr (MODE == 0) {
        ProcessNormal();
    } else {
        ProcessSingle();
    }
}

} // namespace NsLogSpace

#endif // LOG_SPACE_H
