/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_GENERIC_H
#define CAST_GENERIC_H

#include "cast_base.h"
#include "cast_ops.h"

namespace AscendC {

template <typename T, typename U>
class CastGeneric : public CastBase<T, U> {
    using Base = CastBase<T, U>;
    using Base::pipe;
    using Base::ubProcessNum;
    using Base::xGm;
    using Base::xLocal;
    using Base::xQue;
    using Base::yGm;
    using Base::yLocal;
    using Base::yQue;
    TBuf<TPosition::VECCALC> scratchI32Buf;

public:
    __aicore__ inline CastGeneric() {}
    __aicore__ inline CastGeneric(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const CastTilingData& tiling)
    {
        this->InitParams(tiling);
        this->SetGmAddr(x, y, workspace);
        this->InitIoBuffers();
        if constexpr (std::is_same_v<T, half> &&
                      (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t> || std::is_same_v<U, int16_t>)) {
            pipe.InitBuffer(scratchI32Buf, ubProcessNum * sizeof(int32_t));
        }
        if constexpr (std::is_same_v<T, int32_t> && (std::is_same_v<U, half> || std::is_same_v<U, bool>)) {
            SetDeqScale((half)1.0);
        }
    }

    __aicore__ inline void Process() { this->RunProcess(this); }

    __aicore__ inline void Compute(int32_t length)
    {
        if constexpr (std::is_same_v<T, float>) {
            ComputeFromFP32(length);
        } else if constexpr (std::is_same_v<T, half>) {
            ComputeFromFP16(length);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            ComputeFromInt32(length);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            ComputeFromInt64(length);
        }
    }

private:
    __aicore__ inline void ComputeFromFP32(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);
        if constexpr (std::is_same_v<U, half>) {
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<U, int32_t>) {
            Cast(yLocal, xLocal, RoundMode::CAST_TRUNC, length);
        } else if constexpr (std::is_same_v<U, int16_t>) {
            Cast(xLocal.template ReinterpretCast<int32_t>(), xLocal, RoundMode::CAST_TRUNC, length);
            cast_ops::PackInt32ToInt16(yLocal, xLocal.template ReinterpretCast<int32_t>(), length);
        } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
            Cast(xLocal.template ReinterpretCast<int32_t>(), xLocal, RoundMode::CAST_TRUNC, length);
            cast_ops::PackInt32ToByte<U>(yLocal, xLocal.template ReinterpretCast<int32_t>(),
                                         yLocal.template ReinterpretCast<int32_t>(), length);
        } else if constexpr (std::is_same_v<U, bool>) {
            Cast(xLocal.template ReinterpretCast<half>(), xLocal, RoundMode::CAST_NONE, length);
            cast_ops::CastToBool(yLocal.template ReinterpretCast<int8_t>(), xLocal.template ReinterpretCast<half>(),
                                 length);
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

    __aicore__ inline void ComputeFromFP16(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);
        if constexpr (std::is_same_v<U, float>) {
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<U, int32_t>) {
            Cast(yLocal, xLocal, RoundMode::CAST_TRUNC, length);
        } else if constexpr (std::is_same_v<U, int16_t>) {
            auto i32Scratch = scratchI32Buf.Get<int32_t>();
            Cast(i32Scratch, xLocal, RoundMode::CAST_TRUNC, length);
            cast_ops::PackInt32ToInt16(yLocal, i32Scratch, length);
        } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
            auto i32Scratch = scratchI32Buf.Get<int32_t>();
            Cast(i32Scratch, xLocal, RoundMode::CAST_TRUNC, length);
            cast_ops::PackInt32ToByte<U>(yLocal, i32Scratch, yLocal.template ReinterpretCast<int32_t>(), length);
        } else if constexpr (std::is_same_v<U, bool>) {
            cast_ops::CastToBool(yLocal.template ReinterpretCast<int8_t>(), xLocal, length);
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

    __aicore__ inline void ComputeFromInt32(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);
        if constexpr (std::is_same_v<U, float>) {
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<U, half>) {
            Cast(yLocal, xLocal, RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<U, int16_t>) {
            cast_ops::PackInt32ToInt16(yLocal, xLocal, length);
        } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
            cast_ops::PackInt32ToByte<U>(yLocal, xLocal, yLocal.template ReinterpretCast<int32_t>(), length);
        } else if constexpr (std::is_same_v<U, bool>) {
            Cast(xLocal.template ReinterpretCast<half>(), xLocal, RoundMode::CAST_NONE, length);
            cast_ops::CastToBool(yLocal.template ReinterpretCast<int8_t>(), xLocal.template ReinterpretCast<half>(),
                                 length);
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

    __aicore__ inline void ComputeFromInt64(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);
        if constexpr (std::is_same_v<U, int32_t>) {
            uint64_t rsvdCnt = 0;
            GatherMask(yLocal, xLocal.template ReinterpretCast<int32_t>(), (uint8_t)1, true, length * 2, {1, 0, 8, 0},
                       rsvdCnt);
        } else if constexpr (std::is_same_v<U, int16_t>) {
            uint64_t rsvdCnt = 0;
            GatherMask(yLocal, xLocal.template ReinterpretCast<int16_t>(), (uint8_t)3, true, length * 4, {1, 0, 8, 0},
                       rsvdCnt);
        } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
            uint64_t rsvdCnt = 0;
            GatherMask(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(),
                       (uint8_t)3, true, length * 4, {1, 0, 8, 0}, rsvdCnt);
            auto int32Scratch = yLocal.template ReinterpretCast<int32_t>();
            Duplicate(int32Scratch, (int)(0x00FF00FF), 8);
            SetMaskCount();
            SetVectorMask<int16_t, MaskMode::COUNTER>(length * 2);
            And<int16_t, false>(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(),
                                int32Scratch.template ReinterpretCast<int16_t>(), MASK_PLACEHOLDER, 1,
                                {1, 1, 0, 8, 8, 0});
            SetMaskNorm();
            ResetMask();
            if constexpr (std::is_same_v<U, int8_t>) {
                Adds(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(),
                     (int16_t)128, length);
                SetMaskCount();
                SetVectorMask<int16_t, MaskMode::COUNTER>(length);
                And<int16_t, false>(
                    xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(),
                    int32Scratch.template ReinterpretCast<int16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 0, 8, 8, 0});
                SetMaskNorm();
                ResetMask();
                Adds(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(),
                     (int16_t)-128, length);
            }
            Cast(xLocal.template ReinterpretCast<half>(), xLocal.template ReinterpretCast<int16_t>(),
                 RoundMode::CAST_NONE, length);
            Cast(yLocal, xLocal.template ReinterpretCast<half>(), RoundMode::CAST_NONE, length);
        } else if constexpr (std::is_same_v<U, bool>) {
            uint64_t rsvdCnt = 0;
            GatherMask(xLocal.template ReinterpretCast<int32_t>(), xLocal.template ReinterpretCast<int32_t>(),
                       (uint8_t)1, true, length * 2, {1, 0, 8, 0}, rsvdCnt);
            SetDeqScale((half)1.0);
            Cast(xLocal.template ReinterpretCast<half>(), xLocal.template ReinterpretCast<int32_t>(),
                 RoundMode::CAST_NONE, length);
            cast_ops::CastToBool(yLocal.template ReinterpretCast<int8_t>(), xLocal.template ReinterpretCast<half>(),
                                 length);
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }
};

} // namespace AscendC

#endif // CAST_GENERIC_H
