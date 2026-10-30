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
 * \file add_mat_mat_elements.h
 * \brief AddMatMatElements Kernel 实现类（arch35 架构）
 *
 * 计算公式：c_out = c × beta + alpha × a × b
 *
 * 模板参数：
 *   T: 数据类型（half / float / bfloat16_t）
 *
 * 路径分支（通过 if constexpr 编译时选择）：
 *   - T = float:      fp32 直接路径，Mul + Muls(alpha) + Muls(beta) + Add
 *   - T = half:       fp16 Cast 绕行路径，fp16→float→计算→float→fp16（避免 fp16 累积误差）
 *   - T = bfloat16_t: bf16 Cast 绕行路径，bf16→float→计算→float→bf16
 *
 * 输入顺序与 op_def / proto.h 一致：c, a, b, beta, alpha
 *   alpha/beta 为标量 tensor（1-element），Init 时从 GM 单点读取。
 */

#ifndef ADD_MAT_MAT_ELEMENTS_H_
#define ADD_MAT_MAT_ELEMENTS_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_mat_mat_elements_tiling_data.h"
#include "add_mat_mat_elements_tiling_key.h"

namespace NsAddMatMatElements {

using namespace AscendC;

template <typename T>
class KernelAddMatMatElements {
public:
    __aicore__ inline KernelAddMatMatElements() {}

    __aicore__ inline void Init(GM_ADDR c, GM_ADDR a, GM_ADDR b,
                                 GM_ADDR beta, GM_ADDR alpha, GM_ADDR cOut,
                                 const AddMatMatElementsTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t currentLen);
    __aicore__ inline void Compute(uint32_t currentLen);
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t currentLen);

    // fp32 直接计算路径
    __aicore__ inline void ComputeDirect(uint32_t currentLen);

    // fp16 / bf16 Cast 绕行路径
    __aicore__ inline void ComputeUpcast(uint32_t currentLen);

private:
    TPipe pipe;

    // 输入队列（VECIN）
    TQue<QuePosition::VECIN, 1> inputQueueA;
    TQue<QuePosition::VECIN, 1> inputQueueB;
    TQue<QuePosition::VECIN, 1> inputQueueC;

    // 输出队列（VECOUT）
    TQue<QuePosition::VECOUT, 1> outputQueueOut;

    // GM 指针
    GlobalTensor<T> gmA;
    GlobalTensor<T> gmB;
    GlobalTensor<T> gmC;
    GlobalTensor<T> gmCOut;

    // Tiling 参数
    uint32_t blockLength_;      // 本 Core 负责的总元素数
    uint32_t tileLength_;       // 每次 UB 处理的元素数
    float    alphaVal_;         // alpha 标量（float 存储，Init 时从 GM 读出）
    float    betaVal_;          // beta 标量（float 存储，Init 时从 GM 读出）

    // bf16 路径额外中间 buffer（通过 pipe 分配，使用 FIXME 注释标记迭代三优化点）
    // 迭代一：直接静态分配，无复用优化
    // fp16 / bf16 Cast 绕行路径需要 float 中间 buffer
    TBuf<QuePosition::VECCALC> floatBufA;
    TBuf<QuePosition::VECCALC> floatBufB;
    TBuf<QuePosition::VECCALC> floatBufC;
    TBuf<QuePosition::VECCALC> floatBufTmp;
};

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::Init(
    GM_ADDR c, GM_ADDR a, GM_ADDR b, GM_ADDR beta, GM_ADDR alpha, GM_ADDR cOut,
    const AddMatMatElementsTilingData& tilingData)
{
    uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

    // 计算本 Core 的偏移和长度
    // 最后一个 Core 可能处理较少的元素
    // ISSUE-002 修复：GM 偏移必须用 int64_t（TOPN-8 规范），防止大 Tensor 偏移溢出
    int64_t offset = static_cast<int64_t>(blockIdx) * static_cast<int64_t>(tilingData.blockLength);
    if (blockIdx < tilingData.blockNum - 1) {
        blockLength_ = tilingData.blockLength;
    } else {
        blockLength_ = tilingData.lastBlockLength;
    }
    tileLength_ = tilingData.tileLength;

    // 从 GM 读取 alpha/beta 标量值（1-element tensor）
    // 与 proto.h 一致：beta/alpha 为输入 tensor，dtype 与 a/b/c 同
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        // bisheng 后端不支持 scalar bf16→float Cast，改用 bit-shift 手动转换：
        // bf16 在数值布局上就是 fp32 的高 16 bits（指数位相同，尾数被截断），
        // 因此 bf16→float 即 (uint32_t(bits) << 16) 然后 bit-cast 成 float。
        GlobalTensor<uint16_t> gmAlphaBits;
        gmAlphaBits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(alpha), 1);
        GlobalTensor<uint16_t> gmBetaBits;
        gmBetaBits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(beta), 1);

        union { uint32_t u; float f; } conv;
        conv.u = static_cast<uint32_t>(gmAlphaBits.GetValue(0)) << 16U;
        alphaVal_ = conv.f;
        conv.u = static_cast<uint32_t>(gmBetaBits.GetValue(0)) << 16U;
        betaVal_ = conv.f;
    } else {
        GlobalTensor<T> gmAlpha;
        gmAlpha.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha), 1);
        alphaVal_ = static_cast<float>(gmAlpha.GetValue(0));

        GlobalTensor<T> gmBeta;
        gmBeta.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta), 1);
        betaVal_ = static_cast<float>(gmBeta.GetValue(0));
    }

    // 设置 GM buffer（从本 Core 负责的偏移处开始）
    gmA.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a) + offset, blockLength_);
    gmB.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b) + offset, blockLength_);
    gmC.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(c) + offset, blockLength_);
    gmCOut.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cOut) + offset, blockLength_);

    // 初始化 UB buffer
    pipe.InitBuffer(inputQueueA, 1, tileLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueB, 1, tileLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueC, 1, tileLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueOut, 1, tileLength_ * sizeof(T));

    // fp16 / bf16 路径：额外分配 float 中间 buffer
    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        pipe.InitBuffer(floatBufA,   tileLength_ * sizeof(float));
        pipe.InitBuffer(floatBufB,   tileLength_ * sizeof(float));
        pipe.InitBuffer(floatBufC,   tileLength_ * sizeof(float));
        pipe.InitBuffer(floatBufTmp, tileLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::CopyIn(uint32_t progress, uint32_t currentLen)
{
    LocalTensor<T> aLocal = inputQueueA.AllocTensor<T>();
    LocalTensor<T> bLocal = inputQueueB.AllocTensor<T>();
    LocalTensor<T> cLocal = inputQueueC.AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    // ISSUE-006 修复：tileLength <= TILE_LENGTH_FP16=1024，sizeof(T) <= 4，
    // 最大字节数 = 1024*4 = 4096 << UINT16_MAX=65535；静态断言对编译时常量兜底验证
    static_assert(sizeof(T) <= 4U, "element size must be <= 4 bytes");
    copyParams.blockLen   = static_cast<uint16_t>(currentLen * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;

    DataCopyPadParams padParams{false, 0, 0, 0};

    // ISSUE-002 修复：gmOffset 使用 int64_t，防止大 Tensor 时偏移溢出（TOPN-8）
    int64_t gmOffset = static_cast<int64_t>(progress) * static_cast<int64_t>(tileLength_);
    DataCopyPad(aLocal, gmA[gmOffset], copyParams, padParams);
    DataCopyPad(bLocal, gmB[gmOffset], copyParams, padParams);
    DataCopyPad(cLocal, gmC[gmOffset], copyParams, padParams);

    inputQueueA.EnQue(aLocal);
    inputQueueB.EnQue(bLocal);
    inputQueueC.EnQue(cLocal);
}

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::ComputeDirect(uint32_t currentLen)
{
    // fp16 / fp32 直接路径
    // 计算步骤：
    //   1. tmp = a * b
    //   2. tmp = alpha * tmp
    //   3. c_scaled = beta * c
    //   4. c_out = c_scaled + tmp

    LocalTensor<T> aLocal   = inputQueueA.DeQue<T>();
    LocalTensor<T> bLocal   = inputQueueB.DeQue<T>();
    LocalTensor<T> cLocal   = inputQueueC.DeQue<T>();
    LocalTensor<T> outLocal = outputQueueOut.AllocTensor<T>();

    // step1: tmp = a * b  （复用 outLocal 作为 tmpBuf）
    Mul(outLocal, aLocal, bLocal, currentLen);

    // step2: tmp = alpha * tmp
    T alphaT = static_cast<T>(alphaVal_);
    Muls(outLocal, outLocal, alphaT, currentLen);

    // step3: c_scaled = beta * c  （in-place 修改 cLocal）
    T betaT = static_cast<T>(betaVal_);
    Muls(cLocal, cLocal, betaT, currentLen);

    // step4: c_out = c_scaled + tmp
    Add(outLocal, cLocal, outLocal, currentLen);

    outputQueueOut.EnQue<T>(outLocal);
    inputQueueA.FreeTensor(aLocal);
    inputQueueB.FreeTensor(bLocal);
    inputQueueC.FreeTensor(cLocal);
}

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::ComputeUpcast(uint32_t currentLen)
{
    // fp16 / bf16 Cast 绕行路径：T → float → 计算 → float → T
    // 计算步骤：
    //   1. floatA = Cast(aLocal, T→float)
    //   2. floatB = Cast(bLocal, T→float)
    //   3. floatC = Cast(cLocal, T→float)
    //   4. tmpFloat = floatA * floatB
    //   5. tmpFloat = alpha * tmpFloat
    //   6. floatC = beta * floatC
    //   7. floatOut = floatC + tmpFloat
    //   8. outT = Cast(floatOut, float→T)

    LocalTensor<T>     aLocal   = inputQueueA.DeQue<T>();
    LocalTensor<T>     bLocal   = inputQueueB.DeQue<T>();
    LocalTensor<T>     cLocal   = inputQueueC.DeQue<T>();
    LocalTensor<T>     outLocal = outputQueueOut.AllocTensor<T>();

    LocalTensor<float> floatA   = floatBufA.Get<float>();
    LocalTensor<float> floatB   = floatBufB.Get<float>();
    LocalTensor<float> floatC   = floatBufC.Get<float>();
    LocalTensor<float> floatTmp = floatBufTmp.Get<float>();

    // step1-3: Cast T → float
    Cast(floatA, aLocal, RoundMode::CAST_NONE, currentLen);
    Cast(floatB, bLocal, RoundMode::CAST_NONE, currentLen);
    Cast(floatC, cLocal, RoundMode::CAST_NONE, currentLen);

    // step4: tmp = a * b
    Mul(floatTmp, floatA, floatB, currentLen);

    // step5: tmp = alpha * tmp
    Muls(floatTmp, floatTmp, alphaVal_, currentLen);

    // step6: c_scaled = beta * c
    Muls(floatC, floatC, betaVal_, currentLen);

    // step7: floatOut = c_scaled + tmp
    Add(floatTmp, floatC, floatTmp, currentLen);

    // step8: Cast float → T
    Cast(outLocal, floatTmp, RoundMode::CAST_RINT, currentLen);

    outputQueueOut.EnQue<T>(outLocal);
    inputQueueA.FreeTensor(aLocal);
    inputQueueB.FreeTensor(bLocal);
    inputQueueC.FreeTensor(cLocal);
}

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::Compute(uint32_t currentLen)
{
    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        ComputeUpcast(currentLen);
    } else {
        // float 走直接路径
        ComputeDirect(currentLen);
    }
}

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::CopyOut(uint32_t progress, uint32_t currentLen)
{
    LocalTensor<T> outLocal = outputQueueOut.DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    // ISSUE-006 修复：tileLength <= TILE_LENGTH_FP16=1024，sizeof(T) <= 4，
    // 最大字节数 = 1024*4 = 4096 << UINT16_MAX=65535；静态断言对编译时常量兜底验证
    static_assert(sizeof(T) <= 4U, "element size must be <= 4 bytes");
    copyParams.blockLen   = static_cast<uint16_t>(currentLen * sizeof(T));
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;

    // ISSUE-002 修复：gmOffset 使用 int64_t，防止大 Tensor 时偏移溢出（TOPN-8）
    int64_t gmOffset = static_cast<int64_t>(progress) * static_cast<int64_t>(tileLength_);
    DataCopyPad(gmCOut[gmOffset], outLocal, copyParams);

    outputQueueOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void KernelAddMatMatElements<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }

    uint32_t loopCount = (blockLength_ + tileLength_ - 1) / tileLength_;
    for (uint32_t i = 0; i < loopCount; i++) {
        uint32_t currentLen;
        if (i == loopCount - 1) {
            currentLen = blockLength_ - tileLength_ * i;
        } else {
            currentLen = tileLength_;
        }
        CopyIn(i, currentLen);
        Compute(currentLen);
        CopyOut(i, currentLen);
    }
}

}  // namespace NsAddMatMatElements

#endif  // ADD_MAT_MAT_ELEMENTS_H_
