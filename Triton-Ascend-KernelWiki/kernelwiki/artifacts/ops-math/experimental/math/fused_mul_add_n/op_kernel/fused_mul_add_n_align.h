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
 * \file fused_mul_add_n_align.h
 * \brief A2 (DAV_2201) direct-compute kernel template for FusedMulAddN.
 *        y = x1 * x3[0] + x2, native dtype (T in {float, int32_t, int16_t}).
 *        Shares Init/Process/CopyIn/CopyOut with FusedMulAddNAlignHalf via the
 *        CRTP base FusedMulAddNAlignBase; only ComputeImpl (native Muls/Add) and
 *        the x3[0] scalar read differ. x3[0] is read once as a scalar register.
 */
#ifndef FUSED_MUL_ADD_N_ALIGN_H
#define FUSED_MUL_ADD_N_ALIGN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fused_mul_add_n_align_base.h"

namespace FusedMulAddNNs {

using namespace AscendC;

template <typename T>
class FusedMulAddNAlign : public FusedMulAddNAlignBase<FusedMulAddNAlign<T>, T> {
public:
    __aicore__ inline FusedMulAddNAlign() {}

    // x3 标量：Init 一次性读取，不进 UB queue；直算路径无需额外 cast buffer。
    __aicore__ inline void InitScalarAndExtraBuffers(const GlobalTensor<T>& inputGmX3, int64_t ubFormer)
    {
        (void)ubFormer;
        x3Value_ = inputGmX3.GetValue(0);
    }

    // 中间逐元素计算（原生 dtype 直算）：y = x1 * x3[0]; y = y + x2 (in-place)。
    __aicore__ inline void ComputeImpl(
        const LocalTensor<T>& x1Local, const LocalTensor<T>& x2Local, const LocalTensor<T>& yLocal, int64_t curNum)
    {
        Muls(yLocal, x1Local, x3Value_, static_cast<int32_t>(curNum));
        Add(yLocal, yLocal, x2Local, static_cast<int32_t>(curNum));
    }

private:
    T x3Value_{};
};

} // namespace FusedMulAddNNs

#endif // FUSED_MUL_ADD_N_ALIGN_H
