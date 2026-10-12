/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_OPS_H
#define CAST_OPS_H

#include "kernel_operator.h"

namespace AscendC {
namespace cast_ops {

template <typename U>
__aicore__ inline void PackInt32ToByte(const LocalTensor<U>& dst, const LocalTensor<int32_t>& src,
                                       const LocalTensor<int32_t>& scratch, int32_t length)
{
    static_assert(std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>,
                  "PackInt32ToByte: U must be int8_t or uint8_t");

    auto srcI16 = src.template ReinterpretCast<int16_t>();
    auto maskI16 = scratch.template ReinterpretCast<int16_t>();

    Duplicate(scratch, (int)0x000000FF, 8);

    if constexpr (std::is_same_v<U, int8_t>) {
        Adds(src, src, (int32_t)128, length);
        SetMaskCount();
        SetVectorMask<int16_t, MaskMode::COUNTER>(length * 2);
        And<int16_t, false>(srcI16, srcI16, maskI16, MASK_PLACEHOLDER, 1, {1, 1, 0, 8, 8, 0});
        SetMaskNorm();
        ResetMask();
        Adds(src, src, (int32_t)-128, length);
    } else {
        SetMaskCount();
        SetVectorMask<int16_t, MaskMode::COUNTER>(length * 2);
        And<int16_t, false>(srcI16, srcI16, maskI16, MASK_PLACEHOLDER, 1, {1, 1, 0, 8, 8, 0});
        SetMaskNorm();
        ResetMask();
    }

    SetDeqScale((half)1.0);
    auto halfView = src.template ReinterpretCast<half>();
    Cast(halfView, src, RoundMode::CAST_NONE, length);
    Cast(dst, halfView, RoundMode::CAST_NONE, length);
}

__aicore__ inline void PackInt32ToInt16(const LocalTensor<int16_t>& dst, const LocalTensor<int32_t>& src,
                                        int32_t length)
{
    uint64_t rsvdCnt = 0;
    GatherMask(dst, src.template ReinterpretCast<int16_t>(), (uint8_t)1, true, length * 2, {1, 0, 8, 0}, rsvdCnt);
}

__aicore__ inline void CastToBool(const LocalTensor<int8_t>& dst, const LocalTensor<half>& src, int32_t length)
{
    auto maskU8 = dst.template ReinterpretCast<uint8_t>();
    uint32_t alignedLen = AlignUp(length, 256 / sizeof(half));
    CompareScalar(maskU8, src, (half)0.0f, CMPMODE::NE, alignedLen);
    Duplicate(src, (half)1.0, length);
    Select(src, maskU8, src, (half)0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLen);
    Cast(dst, src, RoundMode::CAST_NONE, length);
}

} // namespace cast_ops
} // namespace AscendC

#endif // CAST_OPS_H
