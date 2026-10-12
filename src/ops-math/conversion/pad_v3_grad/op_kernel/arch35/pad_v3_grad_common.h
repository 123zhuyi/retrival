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
 * \file pad_v3_grad_common.h
 * \brief pad v3 grad common utilities
 */

#ifndef ASCENDC_PAD_V3_GRAD_COMMON_H
#define ASCENDC_PAD_V3_GRAD_COMMON_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

namespace PadV3Grad {

constexpr uint32_t BUFFER_NUM = 2;                     ///< 队列 buffer 数量
constexpr uint32_t VL_SIZE = Ops::Base::GetVRegSize(); ///< 向量寄存器大小（256B）
constexpr uint32_t UB_BLOCK = Ops::Base::GetUbBlockSize();

constexpr static int64_t CONST2 = 2;
constexpr static int64_t CONST3 = 3;
constexpr static int64_t CONST4 = 4;
constexpr static int32_t CONST5 = 5;

using PromoteDataT = float; ///< 计算过程使用 float32，避免精度损失

constexpr static AscendC::Reg::CastTrait CAST_TRAIT_0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait CAST_TRAIT_1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};

} // namespace PadV3Grad

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};

#endif
