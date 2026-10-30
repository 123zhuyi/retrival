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
 * \file atan_grad.h
 * \brief AtanGrad Kernel 类实现（arch35架构，支持 ascend910b/ascend950）
 *
 * 算子功能：dx = dy * (1 / (1 + x*x))
 *
 * 模板参数：
 *   - T:           数据类型（half / float / bfloat16_t）
 *   - BUFFER_MODE: 缓冲策略（0=单缓冲, 1=双缓冲）
 *
 * 数据流（fp32 路径）：
 *   CopyIn:  GM(x, dy) -> UB(xLocal, dyLocal)
 *   Compute: Mul(tmp, x, x) -> Adds(tmp, 1+x*x) -> Div(dx, dy, tmp)
 *   CopyOut: UB(dxLocal) -> GM(dx)
 *
 * 数据流（fp16/bfloat16 升精度路径）：
 *   CopyIn:  GM(x, dy) -> UB(xLocal[T], dyLocal[T])
 *   Compute: Cast→fp32 -> 四步计算(fp32) -> Cast→T
 *   CopyOut: UB(dxLocal[T]) -> GM(dx)
 *
 * 精度方案说明（穿刺验证结论）：
 *   - fp16: 直接 Reciprocal MERE≈1.07e-3 超阈值，必须升精度到 fp32 计算
 *   - fp32: Reciprocal(INTRINSIC模式)精度不足，改用 Div(dy, 1+x*x) MERE降至1.19e-7
 *   - bf16: Cast(CAST_NONE)+fp32计算+Cast(CAST_RINT)，MERE=1.091e-3 达标
 */

#ifndef ATAN_GRAD_H
#define ATAN_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atan_grad_tiling_data.h"
#include "atan_grad_tiling_key.h"

namespace NsAtanGrad {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class AtanGrad {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline AtanGrad() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, const AtanGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    // fp16 / bf16 升精度计算路径（两者均升至 fp32 再计算）
    __aicore__ inline void ComputeUpcast(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueDy;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueDx;

    // 中间计算临时 buffer（非流水，TBuf）
    // fp16/fp32 路径：1 个 tmp（复用 x*x, 1+x*x, 1/(1+x*x)）
    // bf16 路径：xFp32, dyFp32, tmp, dxFp32 共 4 个 fp32 TBuf
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> xFp32Buf;
    TBuf<QuePosition::VECCALC> dyFp32Buf;
    TBuf<QuePosition::VECCALC> dxFp32Buf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMDy;
    GlobalTensor<T> outputGMDx;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void AtanGrad<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR dy, GM_ADDR dx, const AtanGradTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * blockIdx, blockLength_);
    inputGMDy.SetGlobalBuffer((__gm__ T*)dy + tilingData->blockFactor * blockIdx, blockLength_);
    outputGMDx.SetGlobalBuffer((__gm__ T*)dx + tilingData->blockFactor * blockIdx, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueDy, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueDx, BUFFER_NUM, ubLength_ * sizeof(T));

    if constexpr (std::is_same_v<T, float>) {
        // fp32 路径：1 个 tmp TBuf（Div 方案不需要额外 buffer）
        pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
        // xFp32Buf/dyFp32Buf/dxFp32Buf 在 fp32 路径不使用，初始化为 0 大小避免未初始化读
        pipe.InitBuffer(xFp32Buf, 0);
        pipe.InitBuffer(dyFp32Buf, 0);
        pipe.InitBuffer(dxFp32Buf, 0);
    } else {
        // fp16 / bf16 升精度路径：4 个 fp32 TBuf（xFp32, dyFp32, tmp, dxFp32）
        pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(xFp32Buf, ubLength_ * sizeof(float));
        pipe.InitBuffer(dyFp32Buf, ubLength_ * sizeof(float));
        pipe.InitBuffer(dxFp32Buf, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AtanGrad<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    LocalTensor<T> dyLocal = inputQueueDy.template AllocTensor<T>();

    // 使用 DataCopyExtParams（blockLen 为 uint32_t，支持大于 65535 字节的搬运）
    // DataCopyParams.blockLen 仅为 uint16_t（最大 65535），当 ubFactor*sizeof(T) > 65535 时会溢出
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    DataCopyPad(dyLocal, inputGMDy[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});

    inputQueueX.EnQue(xLocal);
    inputQueueDy.EnQue(dyLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AtanGrad<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> dxLocal = outputQueueDx.template DeQue<T>();

    // 使用 DataCopyExtParams（blockLen 为 uint32_t，支持大于 65535 字节的搬运）
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(outputGMDx[progress * ubLength_], dxLocal, copyParams);
    outputQueueDx.FreeTensor(dxLocal);
}

// fp32 直接计算路径（使用 Div 保证精度）
template <typename T, int BUFFER_MODE>
__aicore__ inline void AtanGrad<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> dyLocal = inputQueueDy.template DeQue<T>();
    LocalTensor<T> dxLocal = outputQueueDx.template AllocTensor<T>();
    LocalTensor<T> tmp = tmpBuf.template Get<T>();

    // 步骤一：tmp = x * x
    Mul(tmp, xLocal, xLocal, static_cast<uint64_t>(currentNum));
    // 步骤二：tmp = 1 + x*x（分母，值 ≥ 1）
    Adds(tmp, tmp, static_cast<T>(1.0f), static_cast<uint64_t>(currentNum));
    // 步骤三：dxLocal = dy / (1 + x*x)
    // 使用 Div 替代 Reciprocal+Mul，避免 Reciprocal INTRINSIC 模式精度不足（MERE≈2.8e-3）
    Div(dxLocal, dyLocal, tmp, static_cast<uint64_t>(currentNum));

    outputQueueDx.template EnQue<T>(dxLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueDy.FreeTensor(dyLocal);
}

// fp16 / bfloat16 升精度计算路径（特化）
template <typename T, int BUFFER_MODE>
__aicore__ inline void AtanGrad<T, BUFFER_MODE>::ComputeUpcast(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> dyLocal = inputQueueDy.template DeQue<T>();
    LocalTensor<T> dxLocal = outputQueueDx.template AllocTensor<T>();

    LocalTensor<float> xFp32 = xFp32Buf.template Get<float>();
    LocalTensor<float> dyFp32 = dyFp32Buf.template Get<float>();
    LocalTensor<float> tmp = tmpBuf.template Get<float>();
    LocalTensor<float> dxFp32 = dxFp32Buf.template Get<float>();

    uint64_t count = static_cast<uint64_t>(currentNum);

    // T -> fp32 升精度（fp16/bf16 均用 CAST_NONE，避免舍入误差）
    Cast(xFp32, xLocal, RoundMode::CAST_NONE, count);
    Cast(dyFp32, dyLocal, RoundMode::CAST_NONE, count);

    // 四步计算（fp32 精度）
    Mul(tmp, xFp32, xFp32, count); // tmp = x*x
    Adds(tmp, tmp, 1.0f, count);   // tmp = 1 + x*x
    // 使用 Div 确保精度，避免 Reciprocal+Mul 的累积误差
    Div(dxFp32, dyFp32, tmp, count); // dxFp32 = dy / (1+x*x)

    // fp32 -> T 降精度（CAST_RINT 银行家舍入）
    Cast(dxLocal, dxFp32, RoundMode::CAST_RINT, count);

    outputQueueDx.template EnQue<T>(dxLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueDy.FreeTensor(dyLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void AtanGrad<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        if constexpr (std::is_same_v<T, float>) {
            // fp32 路径：直接使用 Div（精度充分）
            Compute(currentNum);
        } else {
            // fp16 / bf16 路径：升精度到 fp32 计算
            ComputeUpcast(currentNum);
        }
        CopyOut(i, currentNum);
    }
}

} // namespace NsAtanGrad

#endif // ATAN_GRAD_H
