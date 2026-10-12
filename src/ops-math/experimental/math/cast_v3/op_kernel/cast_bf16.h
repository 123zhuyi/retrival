/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_BF16_H
#define CAST_BF16_H

#include "cast_base.h"
#include "cast_ops.h"

namespace AscendC {

// 16-bit-centric cast: at least one side is bf16/int16/half of bf16-path.
// Owns extra scratch buffers temp1/temp2 for per-bit / gather tricks.
template <typename T, typename U>
class CastBf16 : public CastBase<T, U> {
    using Base = CastBase<T, U>;
    using Base::pipe;
    using Base::ubProcessNum;
    using Base::xGm;
    using Base::xLocal;
    using Base::xQue;
    using Base::yGm;
    using Base::yLocal;
    using Base::yQue;

public:
    __aicore__ inline CastBf16() {}
    __aicore__ inline CastBf16(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const CastTilingData& tiling)
    {
        this->InitParams(tiling);
        this->SetGmAddr(x, y, workspace);
        this->InitIoBuffers();
        pipe.InitBuffer(floatBuf, ubProcessNum * sizeof(float));
        pipe.InitBuffer(int32Buf, ubProcessNum * sizeof(uint32_t));
        floatLocal = floatBuf.template Get<float>();
        int32Local = int32Buf.template Get<int32_t>();

        uint16_t oneBits = 0x3F80;
        bf16One = *reinterpret_cast<half*>(&oneBits);
        uint16_t negBits = 0x007F;
        int8Neg = *reinterpret_cast<half*>(&negBits);
    }

    __aicore__ inline void Process() { this->RunProcess(this); }

    __aicore__ inline void Compute(int32_t length)
    {
        if constexpr (std::is_same_v<T, half> && std::is_same_v<U, int16_t>) {
            ComputeFromFP16(length);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            ComputeFromInt16(length);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            ComputeFromBF16(length);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            ComputeFromInt64(length);
        } else {
            ComputeToBF16(length);
        }
    }

private:
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> int32Buf;
    LocalTensor<float> floatLocal;
    LocalTensor<int32_t> int32Local;
    half bf16One;
    half int8Neg;

    __aicore__ inline void ComputeFromInt16(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);

        if constexpr (std::is_same_v<U, bool>) {
            Abs(xLocal.template ReinterpretCast<half>(), xLocal.template ReinterpretCast<half>(), length);
            Mins(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(), (int16_t)1,
                 length);
            Cast(yLocal.template ReinterpretCast<uint8_t>(), xLocal.template ReinterpretCast<half>(),
                 RoundMode::CAST_CEIL, length);
        } else if constexpr (std::is_same_v<U, int32_t> || std::is_same_v<U, float> || std::is_same_v<U, half>) {
            auto x_low = int32Local.template ReinterpretCast<int16_t>();
            auto x_high = floatLocal.template ReinterpretCast<int16_t>();
            auto maskLocal = yLocal.template ReinterpretCast<int16_t>();
            Duplicate(maskLocal, (int16_t)0x00FF, length);
            And(x_low, xLocal, maskLocal, length);
            Sub(x_high, xLocal, x_low, length);

            auto half_temp = yLocal.template ReinterpretCast<half>();
            auto float_high = floatLocal.template ReinterpretCast<float>();
            auto float_low = int32Local.template ReinterpretCast<float>();

            SetDeqScale((half)1.0);
            Cast(half_temp, x_low, RoundMode::CAST_NONE, length);
            Cast(float_low, half_temp, RoundMode::CAST_NONE, length);
            Cast(half_temp, x_high, RoundMode::CAST_NONE, length);
            Cast(float_high, half_temp, RoundMode::CAST_NONE, length);

            if constexpr (std::is_same_v<U, float>) {
                Add(yLocal, float_high, float_low, length);
            } else {
                Add(float_high, float_high, float_low, length);
                if constexpr (std::is_same_v<U, int32_t>) {
                    Cast(yLocal, float_high, RoundMode::CAST_RINT, length);
                } else {
                    Cast(yLocal, float_high, RoundMode::CAST_NONE, length);
                }
            }
        } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
            auto int32Scratch = yLocal.template ReinterpretCast<int32_t>();
            Duplicate(int32Scratch, (int)(0x00FF00FF), 8);
            And(xLocal, xLocal, int32Scratch.template ReinterpretCast<int16_t>(), (uint64_t)128,
                AlignUp(length * sizeof(int16_t), 256) / 256, {1, 1, 0, 8, 8, 0});
            if constexpr (std::is_same_v<U, int8_t>) {
                Duplicate(int32Scratch, (int)(0xFF00FF00), 8);
                auto negLocal = floatLocal.template ReinterpretCast<int16_t>();
                Or(negLocal, xLocal, int32Scratch.template ReinterpretCast<int16_t>(), (uint64_t)128,
                   AlignUp(length * sizeof(int16_t), 256) / 256, {1, 1, 0, 8, 8, 0});
                CompareScalar(yLocal.template ReinterpretCast<uint8_t>(), xLocal.template ReinterpretCast<half>(),
                              int8Neg, CMPMODE::GT, AlignUp(length, 256 / sizeof(half)));
                Select(xLocal.template ReinterpretCast<half>(), yLocal.template ReinterpretCast<uint8_t>(),
                       negLocal.template ReinterpretCast<half>(), xLocal.template ReinterpretCast<half>(),
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, AlignUp(length, 256 / sizeof(half)));
            }
            Cast(xLocal.template ReinterpretCast<half>(), xLocal, RoundMode::CAST_NONE, length);
            Cast(yLocal, xLocal.template ReinterpretCast<half>(), RoundMode::CAST_NONE, length);
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

    __aicore__ inline void ComputeFromBF16(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);

        if constexpr (std::is_same_v<U, bool>) {
            Abs(xLocal.template ReinterpretCast<half>(), xLocal.template ReinterpretCast<half>(), length);
            Mins(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(), (int16_t)1,
                 length);
            Cast(yLocal.template ReinterpretCast<uint8_t>(), xLocal.template ReinterpretCast<half>(),
                 RoundMode::CAST_CEIL, length);
        } else {
            auto x_int16 = xLocal.template ReinterpretCast<int16_t>();
            auto x_low = int32Local.template ReinterpretCast<int16_t>();
            auto x_high = floatLocal.template ReinterpretCast<int16_t>();
            auto mask_buf = int32Local.template ReinterpretCast<int16_t>()[length];
            Duplicate(mask_buf, (int16_t)0x00FF, length);
            And(x_low, x_int16, mask_buf, length);
            Sub(x_high, x_int16, x_low, length);

            auto half_high = floatLocal.template ReinterpretCast<half>()[length];
            auto half_low = int32Local.template ReinterpretCast<half>()[length];
            SetDeqScale((half)1.0);
            Cast(half_high, x_high, RoundMode::CAST_NONE, length);
            Cast(half_low, x_low, RoundMode::CAST_NONE, length);

            auto float_val = floatLocal.template ReinterpretCast<float>();
            auto float_low = int32Local.template ReinterpretCast<float>();
            Cast(float_val, half_high, RoundMode::CAST_NONE, length);
            Cast(float_low, half_low, RoundMode::CAST_NONE, length);
            Add(float_val, float_val, float_low, length);
            Muls(float_val, float_val, 65536.0f, length);

            if constexpr (std::is_same_v<U, float>) {
                auto y_int32 = yLocal.template ReinterpretCast<int32_t>();
                Cast(y_int32, float_val, RoundMode::CAST_RINT, length);
            } else {
                auto float_bits = int32Local.template ReinterpretCast<int32_t>();
                Cast(float_bits, float_val, RoundMode::CAST_RINT, length);
                auto bf16_as_float = float_bits.template ReinterpretCast<float>();

                if constexpr (std::is_same_v<U, int32_t>) {
                    Cast(yLocal, bf16_as_float, RoundMode::CAST_TRUNC, length);
                } else if constexpr (std::is_same_v<U, half>) {
                    Cast(yLocal, bf16_as_float, RoundMode::CAST_NONE, length);
                } else if constexpr (std::is_same_v<U, int16_t>) {
                    auto int32_temp = floatLocal.template ReinterpretCast<int32_t>();
                    Cast(int32_temp, bf16_as_float, RoundMode::CAST_TRUNC, length);
                    Cast(yLocal, int32_temp, RoundMode::CAST_NONE, length);
                } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
                    auto int32_trunc = floatLocal.template ReinterpretCast<int32_t>();
                    Cast(int32_trunc, bf16_as_float, RoundMode::CAST_TRUNC, length);
                    cast_ops::PackInt32ToByte<U>(yLocal, int32_trunc, yLocal.template ReinterpretCast<int32_t>(),
                                                 length);
                }
            }
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

    __aicore__ inline void ComputeToBF16(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);

        if constexpr (std::is_same_v<T, bool>) {
            auto halfLocal = int32Local.template ReinterpretCast<half>();
            auto align16Len = AlignUp(length, 16);
            auto uint8Local = halfLocal[align16Len].template ReinterpretCast<uint8_t>();
            Cast(halfLocal, xLocal.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, length);
            CompareScalar(uint8Local, halfLocal, (half)0.0f, CMPMODE::EQ, AlignUp(length, 256 / sizeof(half)));
            Duplicate(yLocal, (U)0, length);
            Select(yLocal.template ReinterpretCast<half>(), uint8Local, yLocal.template ReinterpretCast<half>(),
                   bf16One, SELMODE::VSEL_TENSOR_SCALAR_MODE, AlignUp(length, 256 / sizeof(half)));
        } else {
            if constexpr (std::is_same_v<T, float>) {
                CreateVecIndex(int32Local, 0, length);
                Muls(int32Local, int32Local, 4, length);
                Adds(int32Local, int32Local, 2, length);
                Mins(xLocal.template ReinterpretCast<int16_t>(), xLocal.template ReinterpretCast<int16_t>(),
                     (int16_t)0x7F7E, length);
                Adds(xLocal.template ReinterpretCast<int32_t>(), xLocal.template ReinterpretCast<int32_t>(), 0x00008000,
                     length);
                Gather(yLocal, xLocal.template ReinterpretCast<uint16_t>(),
                       int32Local.template ReinterpretCast<uint32_t>(), 0, length);
            } else {
                if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
                    Cast(floatLocal, xLocal, RoundMode::CAST_NONE, length);
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    CreateVecIndex(int32Local, 0, length);
                    Muls(int32Local, int32Local, 8, length);
                    Gather(floatLocal.template ReinterpretCast<uint32_t>(), xLocal.template ReinterpretCast<uint32_t>(),
                           int32Local.template ReinterpretCast<uint32_t>(), 0, length);
                    Cast(floatLocal, floatLocal.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, length);
                } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                    Cast(yLocal.template ReinterpretCast<half>(), xLocal, RoundMode::CAST_NONE, length);
                    Cast(floatLocal, yLocal.template ReinterpretCast<half>(), RoundMode::CAST_NONE, length);
                }
                CreateVecIndex(int32Local, 0, length);
                Muls(int32Local, int32Local, 4, length);
                Adds(int32Local, int32Local, 2, length);
                Mins(floatLocal.template ReinterpretCast<int16_t>(), floatLocal.template ReinterpretCast<int16_t>(),
                     (int16_t)0x7F7E, length);
                Adds(floatLocal.template ReinterpretCast<int32_t>(), floatLocal.template ReinterpretCast<int32_t>(),
                     0x00008000, length);
                Gather(yLocal, floatLocal.template ReinterpretCast<uint16_t>(),
                       int32Local.template ReinterpretCast<uint32_t>(), 0, length);
            }
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }

    __aicore__ inline void ComputeFromFP16(int32_t length)
    {
        xQue.template DeQue<T>(xLocal);
        yQue.template AllocTensor<U>(yLocal);
        if constexpr (std::is_same_v<U, int16_t>) {
            auto i32Scratch = floatLocal.template ReinterpretCast<int32_t>();
            Cast(i32Scratch, xLocal, RoundMode::CAST_TRUNC, length);
            cast_ops::PackInt32ToInt16(yLocal, i32Scratch, length);
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
            GatherMask(yLocal.template ReinterpretCast<uint32_t>(), xLocal.template ReinterpretCast<uint32_t>(),
                       (uint8_t)1, true, length * 2, {1, 0, 8, 0}, rsvdCnt);
        } else if constexpr (std::is_same_v<U, float> || std::is_same_v<U, half>) {
            auto int32_x = xLocal.template ReinterpretCast<int32_t>();
            auto v_low = floatLocal.template ReinterpretCast<int32_t>();
            auto v_high = int32Local.template ReinterpretCast<int32_t>();
            uint64_t rsvdCnt = 0;
            GatherMask(v_low, int32_x, (uint8_t)1, true, length * 2, {1, 0, 8, 0}, rsvdCnt);
            GatherMask(v_high, int32_x, (uint8_t)2, true, length * 2, {1, 0, 8, 0}, rsvdCnt);
            auto f_low = floatLocal.template ReinterpretCast<float>();
            auto f_high = int32Local.template ReinterpretCast<float>();
            Cast(f_low, v_low, RoundMode::CAST_NONE, length);
            Cast(f_high, v_high, RoundMode::CAST_NONE, length);
            uint32_t alignedLenF = AlignUp(length, 256 / sizeof(float));
            if constexpr (std::is_same_v<U, float>) {
                auto cmp_mask = yLocal.template ReinterpretCast<uint8_t>();
                auto f_comp = xLocal.template ReinterpretCast<float>();
                CompareScalar(cmp_mask, f_low, 0.0f, CMPMODE::LT, alignedLenF);
                Duplicate(f_comp, 1.0f, length);
                Select(f_comp, cmp_mask, f_comp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLenF);
                Add(f_high, f_high, f_comp, length);
                Muls(f_high, f_high, 4294967296.0f, length);
                Add(yLocal, f_high, f_low, length);
            } else {
                auto cmp_mask = yLocal.template ReinterpretCast<uint8_t>();
                auto f_comp = xLocal.template ReinterpretCast<float>();
                CompareScalar(cmp_mask, f_low, 0.0f, CMPMODE::LT, alignedLenF);
                Duplicate(f_comp, 1.0f, length);
                Select(f_comp, cmp_mask, f_comp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedLenF);
                Add(f_high, f_high, f_comp, length);
                Muls(f_high, f_high, 4294967296.0f, length);
                auto f_result = floatLocal.template ReinterpretCast<float>();
                Add(f_result, f_high, f_low, length);
                Cast(yLocal, f_result, RoundMode::CAST_NONE, length);
            }
        } else if constexpr (std::is_same_v<U, bool>) {
            auto hi_lo_or = floatLocal.template ReinterpretCast<uint32_t>();
            auto v_low = int32Local.template ReinterpretCast<uint32_t>();
            uint64_t rsvdCnt = 0;
            GatherMask(hi_lo_or, xLocal.template ReinterpretCast<uint32_t>(), (uint8_t)1, true, length * 2,
                       {1, 0, 8, 0}, rsvdCnt);
            GatherMask(v_low, xLocal.template ReinterpretCast<uint32_t>(), (uint8_t)2, true, length * 2, {1, 0, 8, 0},
                       rsvdCnt);
            auto hi_lo_or_u16 = hi_lo_or.template ReinterpretCast<uint16_t>();
            auto v_low_u16 = v_low.template ReinterpretCast<uint16_t>();
            Or(hi_lo_or_u16, hi_lo_or_u16, v_low_u16, length * 2);
            auto or_as_int = hi_lo_or.template ReinterpretCast<int32_t>();
            auto or_as_float = xLocal.template ReinterpretCast<float>();
            Cast(or_as_float, or_as_int, RoundMode::CAST_NONE, length);
            auto mask_u8 = yLocal.template ReinterpretCast<uint8_t>();
            uint32_t alignedLenF = AlignUp(length, 256 / sizeof(float));
            CompareScalar(mask_u8, or_as_float, 0.0f, CMPMODE::NE, alignedLenF);
            auto r_half = int32Local.template ReinterpretCast<half>();
            Duplicate(r_half, (half)1.0f, length);
            Select(r_half, mask_u8, r_half, (half)0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   AlignUp(length, 256 / sizeof(half)));
            Cast(yLocal.template ReinterpretCast<uint8_t>(), r_half, RoundMode::CAST_NONE, length);
        }
        xQue.template FreeTensor<T>(xLocal);
        yQue.template EnQue<U>(yLocal);
    }
};

} // namespace AscendC

#endif // CAST_BF16_H
