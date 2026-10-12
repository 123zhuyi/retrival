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
 * \file acosh.h
 * \brief KernelAcosh 类实现（arch35 架构）
 *
 * 模板参数：
 *   T           — 数据类型 (half / float / bfloat16_t)
 *   BUFFER_MODE — 缓冲模式 (0=单缓冲, 1=双缓冲)
 *
 * 计算公式：y = acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1
 *
 * fp16/bf16 路径：AscendC::Acosh 对 fp16 大值（65504）会因中间步骤 x^2 溢出返回 0；
 *   bf16 原语不支持。因此 fp16 和 bf16 均走 Cast 回退路径：
 *   Cast(fp16/bf16->fp32, CAST_NONE) -> Acosh(fp32) -> Cast(fp32->fp16/bf16)
 *   其中 fp32->fp16 使用 CAST_ROUND，fp32->bf16 使用 CAST_RINT。
 * fp32 路径：直接调用 AscendC::Acosh(fp32)，无溢出问题。
 */

#ifndef ACOSH_ARCH35_H
#define ACOSH_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acosh_tiling_data.h"
#include "acosh_tiling_key.h"

namespace NsAcosh {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class KernelAcosh {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

    // fp16/bf16 路径需要 fp32 中转缓冲（Cast 回退策略）
    // fp16 大值（65504）会因 x^2 中间步骤溢出，必须走 fp32 中转路径
    static constexpr bool IS_CAST_PATH = !std::is_same_v<T, float>;

public:
    __aicore__ inline KernelAcosh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AcoshTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    // fp16/bf16 回退路径：fp32 中转缓冲（使用 VECCALC 位置）
    TBuf<TPosition::VECCALC> tmpBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t blockLength_ = 0; // 当前核处理的元素数量
    int64_t ubFactor_ = 0;    // 每次 UB 循环处理的元素数量
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelAcosh<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const AcoshTilingData* tilingData)
{
    // 计算当前核的实际处理元素数（尾部核可能少于 blockFactor）
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ <= 0) {
        blockLength_ = 0;
        return;
    }
    ubFactor_ = tilingData->ubFactor;

    xGm_.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    yGm_.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * blockIdx, blockLength_);

    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubFactor_ * sizeof(T));

    // fp16/bf16 路径：初始化 fp32 中转缓冲（Cast 回退策略）
    if constexpr (IS_CAST_PATH) {
        pipe_.InitBuffer(tmpBuf_, ubFactor_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelAcosh<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue_.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, xGm_[progress * ubFactor_], copyParams, {false, 0, 0, 0});
    inputQueue_.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelAcosh<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue_.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueue_.template AllocTensor<T>();

    if constexpr (IS_CAST_PATH) {
        // fp16/bf16 回退路径：Cast ->fp32 -> Acosh(fp32) -> Cast ->fp16/bf16
        // fp16 大值（65504）直接调用 Acosh 会因 x^2 溢出返回 0，必须走此路径
        AscendC::LocalTensor<float> tmpFp32 = tmpBuf_.Get<float>();
        AscendC::Cast(tmpFp32, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Acosh(tmpFp32, tmpFp32, currentNum);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            // fp32->bf16 降精度：使用 CAST_RINT（就近取整）
            AscendC::Cast(yLocal, tmpFp32, AscendC::RoundMode::CAST_RINT, currentNum);
        } else {
            // fp32->fp16 降精度：使用 CAST_ROUND
            AscendC::Cast(yLocal, tmpFp32, AscendC::RoundMode::CAST_ROUND, currentNum);
        }
    } else {
        // fp32 路径：直接调用 Acosh 原语，无溢出问题
        AscendC::Acosh(yLocal, xLocal, currentNum);
    }

    outputQueue_.template EnQue<T>(yLocal);
    inputQueue_.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelAcosh<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueue_.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(yGm_[progress * ubFactor_], yLocal, copyParams);
    outputQueue_.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void KernelAcosh<T, BUFFER_MODE>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        // 最后一次循环处理余量（可能小于 ubFactor）
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - ubFactor_ * i) : ubFactor_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsAcosh

#endif // ACOSH_ARCH35_H
