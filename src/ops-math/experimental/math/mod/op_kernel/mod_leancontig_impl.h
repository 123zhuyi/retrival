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
 * \file mod_leancontig_impl.h
 * \brief Mod<T,ST,OT> lean contiguous compute member definitions (arch22-only).
 *
 * same-dtype fp32/fp16/bf16 的连续 (isInput2Scalar || isInput2SameShape) 计算核精简路径。
 *   动机：上游继承的 inf/nan/zero `Abs→Compare→Select ×2` 收尾 (6 op/tile) 叠在 naive 4-op 上拖慢连续路；
 *   精简核绕过 ComputeCore/ComputeFPCore (inf/nan 收尾 + 常驻常量 + tmp)，直接跑 RemainderAdaptive + cast
 *   收尾 (fp16/bf16 CAST_RINT) -> vec 收敛到 ~4 op；host UB_DIVIDER 69/65 -> 48 -> tile 更宽 -> 摊薄探针同步。
 *   inf/nan 由 naive/AlgoA 自然传播 (x1=inf -> nan；x2=inf -> nan，匹配内置；上游 Select 收尾把 fmod(x,inf)
 *   映到 x，与内置 nan 语义相反 —— 精简核回避该收尾)。
 *
 * 隔离：全 `#if MOD_ENH_ARCH22` 守卫；USE_LEAN_CONTIG = fp32|fp16|bf16，编译期择路 ->
 *   int16 / int32
 * 走
 * ProcessContigPipeline<false> -> ComputeCore 原路
 *   (那些 lane 的 kernel .o 不变)。general broadcast
 * (非融合) 走 ComputeCore -> 需常驻常量 -> InitConstants 不跳过、host 保持 69/65 divider -> 零回归。mod 仅注册 arch22
 * (ascend910b/ascend910_93 均 DAV_2201) -> 无非 arch22 kernel 消费精简 tiling。
 *
 * 数值等价：ComputeContigLean = fp32-native VIEW 输入 + rOut 直写 output slot；fp16/bf16 Cast(CAST_NONE) ->
 *   fp32 -> RemainderAdaptive -> Cast(CAST_RINT) 回 T。RemainderAdaptive/AlgoA 不改写 aIn/bIn -> 复用 flat slot 安全。
 *
 * 与 mod_flat_impl.h / mod_bcast_impl.h 同惯例：本文件不自带 `namespace ModNs`，从 mod.h 的 namespace 块内
 *   #include -> 定义附着到 ModNs::Mod<T,ST,OT>。
 */
#ifndef MOD_LEANCONTIG_IMPL_H
#define MOD_LEANCONTIG_IMPL_H

#if MOD_ENH_ARCH22

// 精简连续核 buffer 集 (在 InitFlatBuffers 分配的 flat self/other/out 双缓之外追加)：RemainderAdaptive 的
//   6 个 disjoint fp32 工作块 (w0..w5)。fp32-native: rOut 直写 output slot (VIEW) -> 无独立结果块；fp16/bf16:
//   需 selfF32/otherF32 (cast 目标) + rF32 (结果块, cast 回 T 前的 fp32)。
//   perElem: fp32 = flat(self 2*4 + other 2*4 + out 2*4)=24 + w0..w5 24 = 48；fp16/bf16 = flat(2*2*3)=12 +
//     w0..w5 24 + rF32 4 + selfF32 4 + otherF32 4 = 48。与 host UB_DIVIDER_*_LEAN=48 lockstep。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::InitLeanWorkBuffers()
{
    pipe.InitBuffer(ResQuotTensorBuff, maxDataCount * sizeof(float)); // w0
    pipe.InitBuffer(ResRemTensorBuff, maxDataCount * sizeof(float));  // w1
    pipe.InitBuffer(A1Buff, maxDataCount * sizeof(float));            // w2
    pipe.InitBuffer(A2Buff, maxDataCount * sizeof(float));            // w3
    pipe.InitBuffer(A3Buff, maxDataCount * sizeof(float));            // w4
    pipe.InitBuffer(A4Buff, maxDataCount * sizeof(float));            // w5
    if constexpr (NEED_FP32_IO_BUF) {                          // fp16/bf16: cast self/other 到 fp32 + rF32 结果块
        pipe.InitBuffer(A5Buff, maxDataCount * sizeof(float)); // rF32 (RemainderAdaptive 结果, cast 前)
        pipe.InitBuffer(x1TensorFP32Buff, maxDataCount * sizeof(float)); // selfF32 (cast target)
        pipe.InitBuffer(x2TensorFP32Buff, maxDataCount * sizeof(float)); // otherF32 (cast target)
    }
}

// 精简 fp32 域核。x1Tensor/x2Tensor/dstTensor = flat slot (ST==OT==T same-dtype)。self/other 物化到 fp32
//   (fp16/bf16 Cast(CAST_NONE) 精确 widen；fp32-native VIEW)，RemainderAdaptive 直算 (naive|AlgoA per-tile
//   max|a| 路由，与 ComputeFPCore 的 arch22 分支同一真值源)，结果 fp32-native 直写 output slot / cast
//   (fp16/bf16) CAST_RINT 回 T。无 inf/nan 收尾 (自然传播)，无 tmp。
template <typename T, typename ST, typename OT>
__aicore__ inline void Mod<T, ST, OT>::ComputeContigLean(int32_t calCount, LocalTensor<ST>& x1Tensor,
                                                         LocalTensor<OT>& x2Tensor, LocalTensor<T>& dstTensor)
{
    const int32_t cnt = calCount;
    LocalTensor<float> selfF32;
    LocalTensor<float> otherF32;
    if constexpr (NEED_FP32_IO_BUF) {
        selfF32 = x1TensorFP32Buff.Get<float>();
        otherF32 = x2TensorFP32Buff.Get<float>();
        Cast(selfF32, x1Tensor, AscendC::RoundMode::CAST_NONE, cnt); // fp16/bf16 -> fp32 (精确 widen)
        Cast(otherF32, x2Tensor, AscendC::RoundMode::CAST_NONE, cnt);
    } else {
        selfF32 = x1Tensor.template ReinterpretCast<float>(); // fp32-native: VIEW 输入 slot
        otherF32 = x2Tensor.template ReinterpretCast<float>();
    }
    // RemainderAdaptive 6 个 disjoint fp32 工作块 (均 != selfF32/otherF32/rOut)。
    LocalTensor<float> w0 = ResQuotTensorBuff.Get<float>();
    LocalTensor<float> w1 = ResRemTensorBuff.Get<float>();
    LocalTensor<float> w2 = A1Buff.Get<float>();
    LocalTensor<float> w3 = A2Buff.Get<float>();
    LocalTensor<float> w4 = A3Buff.Get<float>();
    LocalTensor<float> w5 = A4Buff.Get<float>();
    if constexpr (NEED_FP32_IO_BUF) {
        LocalTensor<float> rF32 = A5Buff.Get<float>();
        RemainderRouted(rF32, selfF32, otherF32, w0, w1, w2, w3, w4, w5, cnt);
        // fp32 -> fp16/bf16
        Cast(dstTensor, rF32, AscendC::RoundMode::CAST_RINT, cnt);
    } else {
        LocalTensor<float> outF32 = dstTensor.template ReinterpretCast<float>(); // fp32-native: r 直写 output slot
        RemainderRouted(outF32, selfF32, otherF32, w0, w1, w2, w3, w4, w5, cnt);
    }
}

// lean Process 循环本身在 mod_flat_impl.h (ProcessContigPipeline<LEAN=true>)：flat (ComputeCore) 与 lean
//   (ComputeContigLean) 连续派发共用同一 FLAT_SLOTS ping-pong 骨架，每 tile 计算步由编译期 LEAN flag 选择。
//   本文件只放 lean 专属的 buffer init + compute (InitLeanWorkBuffers / ComputeContigLean)。

#endif // MOD_ENH_ARCH22

#endif // MOD_LEANCONTIG_IMPL_H
