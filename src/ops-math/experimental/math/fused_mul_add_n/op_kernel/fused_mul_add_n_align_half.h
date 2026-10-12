/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_mul_add_n_align_half.h
 * \brief A2 (DAV_2201) Cast-domain kernel template for FusedMulAddN.
 *        y = x1 * x3[0] + x2 computed in fp32 (T in {half, bfloat16_t}).
 *        Shares Init/Process/CopyIn/CopyOut with FusedMulAddNAlign via the CRTP base
 *        FusedMulAddNAlignBase; only ComputeImpl (cast-domain math) and the fp32 x3[0]
 *        scalar read + cast scratch buffers differ.
 *
 *        两条 cast-域 路径（编译期按 T 分发）：
 *        - bf16 (T == bfloat16_t)：5 op + 2 块 fp32 scratch（每元素 20 字节）。
 *            Cast(x1F,x1,NONE); Cast(x2F,x2,NONE); Muls(x1F,x1F,x3); Add(x1F,x1F,x2F);
 *            Cast(y,x1F,RINT)。A2 不支持 Axpy<float,bf16>，故保持原 5-op 路径不动。
 *        - fp16 (T == half)：Axpy 融合，3 op + 1 块 fp32 scratch（每元素 16 字节）。
 *            Cast(x2F,x2,NONE); Axpy<float,half>(x2F,x1,(half)x3); Cast(y,x2F,RINT)。
 *            Axpy 语义 dst = src*scalar + dst => x2F = x1*x3 + x2（fp32 累加），
 *            一次同时增大 tile（bytesPerElem 20->16）并减 VEC 级数（5->3）。
 *            ⚠ Axpy 的 scalar 形参为 half，x3 先 round 到 fp16 再累加（精度由 ST 门控）。
 */
#ifndef FUSED_MUL_ADD_N_ALIGN_HALF_H
#define FUSED_MUL_ADD_N_ALIGN_HALF_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fused_mul_add_n_align_base.h"

namespace FusedMulAddNNs {

using namespace AscendC;

template <typename T>
class FusedMulAddNAlignHalf : public FusedMulAddNAlignBase<FusedMulAddNAlignHalf<T>, T> {
public:
    __aicore__ inline FusedMulAddNAlignHalf() {}

    // x3 标量升 fp32（bf16 用 ToFloat 非模板重载，half 用 static_cast）+ fp32 cast 中间 buffer。
    //   fp16（Axpy 融合）：仅需 1 块 fp32 scratch（x2F），不再需要 tmpBufX1F_。
    //   bf16（原 5-op 路径）：需 2 块 fp32 scratch（x1F + x2F）。
    __aicore__ inline void InitScalarAndExtraBuffers(const GlobalTensor<T>& inputGmX3, int64_t ubFormer)
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            x3ValueF_ = ToFloat(inputGmX3.GetValue(0));
            this->pipe_->InitBuffer(tmpBufX1F_, ubFormer * sizeof(float));
            this->pipe_->InitBuffer(tmpBufX2F_, ubFormer * sizeof(float));
        } else { // half
            x3ValueF_ = static_cast<float>(inputGmX3.GetValue(0));
            this->pipe_->InitBuffer(tmpBufX2F_, ubFormer * sizeof(float));
        }
    }

    // 中间逐元素计算（cast 域算）。fp16 与 bf16 编译期分发到各自路径。
    __aicore__ inline void ComputeImpl(
        const LocalTensor<T>& x1Local, const LocalTensor<T>& x2Local, const LocalTensor<T>& yLocal, int64_t curNum)
    {
        int32_t count = static_cast<int32_t>(curNum);
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            // bf16：5 op（A2 不支持 Axpy<float,bf16>）。bf16 -> fp32，fp32 Muls/Add，再 fp32 -> bf16。
            LocalTensor<float> x1F = tmpBufX1F_.template Get<float>();
            LocalTensor<float> x2F = tmpBufX2F_.template Get<float>();
            Cast(x1F, x1Local, RoundMode::CAST_NONE, count); // bf16 -> fp32
            Cast(x2F, x2Local, RoundMode::CAST_NONE, count);
            Muls(x1F, x1F, x3ValueF_, count);                // fp32: x1F = x1F * x3[0]
            Add(x1F, x1F, x2F, count);                       // fp32: x1F = x1F + x2F
            Cast(yLocal, x1F, RoundMode::CAST_RINT, count);  // fp32 -> bf16
        } else { // half — Axpy 融合：3 op + 1 块 fp32 scratch。
            LocalTensor<float> x2F = tmpBufX2F_.template Get<float>();
            Cast(x2F, x2Local, RoundMode::CAST_NONE, count); // half -> fp32 (x2F = x2)
            // Axpy: dst = src*scalar + dst => x2F = x1*x3 + x2（fp32 累加，src 为 half 输入）。
            Axpy<float, half>(x2F, x1Local, static_cast<half>(x3ValueF_), count);
            Cast(yLocal, x2F, RoundMode::CAST_RINT, count);  // fp32 -> half
        }
    }

private:
    TBuf<TPosition::VECCALC> tmpBufX1F_; // 仅 bf16 路径使用（fp16 Axpy 融合后不再初始化）
    TBuf<TPosition::VECCALC> tmpBufX2F_;
    float x3ValueF_ = 0.0f;
};

} // namespace FusedMulAddNNs

#endif // FUSED_MUL_ADD_N_ALIGN_HALF_H
