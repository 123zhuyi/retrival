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
 * \file atanh.h
 * \brief Atanh 算子 kernel 类定义（arch32 架构）
 *
 * 公式：atanh(x) = 0.5 * ln((1 + x) / (1 - x))
 *
 * 模板参数：
 *   - T: 数据类型（float / half / bfloat16_t）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * bf16 路径：Cast 到 float32 计算再 Cast 回来
 */
#ifndef ATANH_H
#define ATANH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atanh_tiling_data.h"
#include "atanh_tiling_key.h"

namespace NsAtanh {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class Atanh {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Atanh(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AtanhTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> castBuf; // bf16 专用：Cast 到 float32 的中间缓冲

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const AtanhTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        // bf16: temp 和 cast 缓冲用 float32
        pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
    } else {
        pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(T));
        pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(T));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        // bf16 路径：Cast 到 float32 计算再 Cast 回来
        AscendC::LocalTensor<float> castLocal = castBuf.Get<float>();
        AscendC::LocalTensor<float> temp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> temp2 = tmpBuf2.Get<float>();

        AscendC::Cast(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);

        AscendC::Adds(temp1, castLocal, 1.0f, currentNum);
        AscendC::Muls(temp2, castLocal, -1.0f, currentNum);
        AscendC::Adds(temp2, temp2, 1.0f, currentNum);
        AscendC::Div(temp1, temp1, temp2, currentNum);
        AscendC::Ln(temp1, temp1, currentNum);
        AscendC::Muls(castLocal, temp1, 0.5f, currentNum);

        AscendC::Cast(yLocal, castLocal, AscendC::RoundMode::CAST_ROUND, currentNum);
    } else {
        // float/half 路径：直接计算
        AscendC::LocalTensor<T> temp1 = tmpBuf1.Get<T>();
        AscendC::LocalTensor<T> temp2 = tmpBuf2.Get<T>();

        AscendC::Adds(temp1, xLocal, static_cast<T>(1.0f), currentNum);
        AscendC::Muls(temp2, xLocal, static_cast<T>(-1.0f), currentNum);
        AscendC::Adds(temp2, temp2, static_cast<T>(1.0f), currentNum);
        AscendC::Div(temp1, temp1, temp2, currentNum);
        AscendC::Ln(temp1, temp1, currentNum);
        AscendC::Muls(yLocal, temp1, static_cast<T>(0.5f), currentNum);
    }

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Atanh<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsAtanh
#endif // ATANH_H
