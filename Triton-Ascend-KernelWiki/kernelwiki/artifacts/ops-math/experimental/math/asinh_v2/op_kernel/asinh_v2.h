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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_v2.h
 * \brief AsinhV2 算子 Kernel 类定义（arch32 架构）
 *
 * 模板参数说明：
 *   - T: 数据类型（half=float16, float=float32）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * 流水线设计：
 *   CopyIn (GM→UB) → Compute (Asinh) → CopyOut (UB→GM)
 *   双缓冲时（BUFFER_MODE=1）三阶段流水线并行，隐藏 DMA 延迟。
 *
 * 内存布局：
 *   inputQueue  : BUFFER_NUM × ubFactor × sizeof(T)
 *   outputQueue : BUFFER_NUM × ubFactor × sizeof(T)
 *   tmpQueue    : 1 × tmpBufSize  (Asinh sharedTmpBuffer)
 */

#ifndef ASINH_V2_ARCH32_H
#define ASINH_V2_ARCH32_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/math/asinh.h"
#include "asinh_v2_tiling_data.h"
#include "asinh_v2_tiling_key.h"

namespace NsAsinhV2 {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class AsinhV2 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline AsinhV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AsinhV2TilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TQue<QuePosition::VECCALC, 1> tmpQueue; // sharedTmpBuffer for Asinh

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tmpBufSize_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void AsinhV2<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const AsinhV2TilingData* td)
{
    int64_t offset = td->blockFactor * AscendC::GetBlockIdx();
    int64_t remaining = td->totalNum - offset;
    blockLength_ = (remaining > td->blockFactor) ? td->blockFactor : remaining;
    ubFactor_ = td->ubFactor;
    tmpBufSize_ = td->tmpBufSize;

    inputGM.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, (uint32_t)(ubFactor_ * sizeof(T)));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, (uint32_t)(ubFactor_ * sizeof(T)));
    pipe.InitBuffer(tmpQueue, 1, (uint32_t)tmpBufSize_);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AsinhV2<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGM[progress * ubFactor_], copyParams, padParams);
    inputQueue.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AsinhV2<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();
    LocalTensor<uint8_t> tmpBuf = tmpQueue.AllocTensor<uint8_t>();

    AscendC::Asinh(yLocal, xLocal, tmpBuf, (uint32_t)currentNum);

    tmpQueue.FreeTensor(tmpBuf);
    outputQueue.template EnQue<T>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AsinhV2<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGM[progress * ubFactor_], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AsinhV2<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i < loopCount - 1) ? ubFactor_ : (blockLength_ - i * ubFactor_);
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsAsinhV2

#endif // ASINH_V2_ARCH32_H
