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
 * \file mul_dag.h
 * \brief mul dag
 */

#ifndef MUL_DAG_H
#define MUL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#endif

namespace MulDag {
using namespace AscendC;
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
constexpr int CAST_MODE_FLOOR = 2;
constexpr int CAST_MODE_ROUND = 4;
constexpr uint32_t COUNT_DOUBLE = 2;
constexpr std::int32_t FF = 255;

constexpr int SIGNMASK = 0x80000000; // 符号位
constexpr int EXPMASK = 0x7FF << 20; // 11位指数
constexpr int MMASK = 0xFFFFF;       // 20位尾数
constexpr int NANI = 0x7FF80000;
constexpr int MAX_EXP = 2048 - 1;
constexpr int INT32_MMAXMASK = 0x7FFFFFFF;
constexpr int SUBNORMAL_MASK = 0xFFFF0000;
constexpr int EXP_OFFSET = 20;
constexpr int EXP_BIAS = 1024 - 1;
constexpr int BIT_ALIGN = 8;
constexpr int OFFSET_16 = 16;
constexpr int OFFSET_32 = 32;
constexpr int OFFSET_64 = 64;
constexpr int NOSIGN_OFFSET_32 = 32 - 1;
constexpr int SUBN_BOUND_52 = -53;
constexpr int SUBN_BOUND_32 = -33;
constexpr int SUBN_BOUND_16 = -17;
constexpr int MAX_SAFE_LSHIFT_20 = 20 - 1;
constexpr int HIGH_BIT_OFFSET_32 = 30;

union U {
    unsigned int i[2];
    double d;
    unsigned long long u;
};

union ll {
    unsigned int i[2];
    unsigned long long il;
};

template <class T1, class T2> // complex64, complex32
struct CastComplex32ToComplex64 : public Vec::ElemwiseUnaryOP<T1, T2> {
    __aicore__ inline CastComplex32ToComplex64(LocalTensor<T1>& dst, LocalTensor<T2>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::Cast(
            dst.template ReinterpretCast<float>(), src.template ReinterpretCast<half>(), RoundMode::CAST_NONE,
            count * COUNT_DOUBLE);
#endif
    }
};
template <class T1, class T2> // complex32, complex64
struct CastComplex64ToComplex32 : public Vec::ElemwiseUnaryOP<T1, T2> {
    __aicore__ inline CastComplex64ToComplex32(LocalTensor<T1>& dst, LocalTensor<T2>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::Cast(
            dst.template ReinterpretCast<half>(), src.template ReinterpretCast<float>(), RoundMode::CAST_RINT,
            count * COUNT_DOUBLE);
#endif
    }
};

template <class T>
struct AndFF : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline AndFF(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::Duplicate(dst, static_cast<T>(FF), count);
        AscendC::And(dst, src, dst, count);
#endif
    }
};

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__
    LAUNCH_BOUND(1024) inline void MulDouble_vf(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2, int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        U a;
        U b;
        a.d = src1[index];
        b.d = src2[index];

        // 输入subnormal处理  --> 核心 s3可能是0 --> done
        // 输出subnormal处理
        int sa = a.i[1] & SIGNMASK;
        int sb = b.i[1] & SIGNMASK;
        int ea = (a.i[1] & EXPMASK) >> EXP_OFFSET;
        int eb = (b.i[1] & EXPMASK) >> EXP_OFFSET;

        U c; // result

        // inf nan处理
        if (ea == MAX_EXP && ((a.i[1] & MMASK) > 0 || a.i[0] > 0)) {
            c.i[1] = NANI;
            dst[index] = c.d;
            continue;
        }
        if (eb == MAX_EXP && ((b.i[1] & MMASK) > 0 || b.i[0] > 0)) {
            c.i[1] = NANI;
            dst[index] = c.d;
            continue;
        }

        if (ea == MAX_EXP) {
            // inf * 0
            if (eb == 0 && (b.i[1] & MMASK) == 0 && b.i[0] == 0) {
                c.i[1] = NANI;
                dst[index] = c.d;
                continue;
            }
            c = a;
            c.i[1] ^= (b.i[1] & SIGNMASK); // inf * b
            dst[index] = c.d;
            continue;
        }
        if (eb == MAX_EXP) {
            // inf * 0
            if (ea == 0 && (a.i[1] & MMASK) == 0 && a.i[0] == 0) {
                c.i[1] = NANI;
                dst[index] = c.d;
                continue;
            }
            c = b;
            c.i[1] ^= (a.i[1] & SIGNMASK); // inf * b
            dst[index] = c.d;
            continue;
        }

        // 0的处理
        if (((a.i[1] & INT32_MMAXMASK) == 0) && (a.i[0] == 0)) {
            c.d = 0;
            dst[index] = c.d;
            continue;
        }
        if (((b.i[1] & INT32_MMAXMASK) == 0) && (b.i[0] == 0)) {
            c.d = 0;
            dst[index] = c.d;
            continue;
        }

        // 输入subnormal
        bool aflag = (ea > 0);
        bool bflag = (eb > 0);
        ea = ea - EXP_BIAS + 1 - aflag;
        eb = eb - EXP_BIAS + 1 - bflag;

        unsigned int ma_high = (a.i[1] & MMASK) + (aflag << EXP_OFFSET);
        unsigned int ma_low = a.i[0];
        unsigned int mb_high = (b.i[1] & MMASK) + (bflag << EXP_OFFSET);
        unsigned int mb_low = b.i[0];

        ll res[3];

        res[0].il = (unsigned long long)ma_low * (unsigned long long)mb_low;

        res[1].i[0] = res[0].i[1];
        res[1].i[1] = 0;
         // 利用FMA指令处理进位（参考ISA的IMAD指令）
        res[1].il += (unsigned long long)ma_low * (unsigned long long)mb_high;
        // 即采用了 unsigned int * unsigned int + unsigned long long -> 的指令
        res[1].il += (unsigned long long)ma_high * (unsigned long long)mb_low;

        res[2].i[0] = res[1].i[1];
        res[2].i[1] = 0;
        res[2].il += (unsigned long long)ma_high * (unsigned long long)mb_high;

        // 换个符号
        unsigned int s3 = res[2].i[1];
        unsigned int s2 = res[2].i[0];
        unsigned int s1 = res[1].i[0];
        unsigned int s0 = res[0].i[0];

        // 寻找s3最高位，正常来说是8或9； subnormal情况可能会不一样
        // 处理输入subnormal

        // total_shift 记录需要左移的总位数 （32位的倍数）
        int total_shift = 0;

        if (s3 != 0) {
            total_shift = 0; // MSB 在 s3，无需移动
        } else if ((s2 & SUBNORMAL_MASK) != 0) {
            total_shift = OFFSET_16; // MSB 在 s2 的高 16 位 (需要移动 16 位)
        } else if (s2 != 0) {
            total_shift = OFFSET_32; // MSB 在 s2 的低 16 位
        } else if ((s1 & SUBNORMAL_MASK) != 0) {
            total_shift = OFFSET_32 + OFFSET_16; // MSB 在 s1 的高 16 位 (需要移动 48 位)
        } else if (s1 != 0) {
            total_shift = OFFSET_64; // MSB 在 s1 的低 16 位
        } else if ((s0 & SUBNORMAL_MASK) != 0) {
            total_shift = OFFSET_64 + OFFSET_16; // MSB 在 s0 的高 16 位 (需要移动 80 位)
        } else if (s0 != 0) {
            total_shift = OFFSET_64 + OFFSET_32; // MSB 在 s0 的低 16 位
        } else {
            total_shift = 0; // 乘积为0
            s3 = 0;
        }

        if (total_shift == OFFSET_16) {
            // 左移 16 位
            s3 = (s3 << OFFSET_16) | (s2 >> OFFSET_16);
            s2 = (s2 << OFFSET_16) | (s1 >> OFFSET_16);
            s1 = (s1 << OFFSET_16) | (s0 >> OFFSET_16);
            s0 = (s0 << OFFSET_16);
        } else if (total_shift == OFFSET_32) {
            // 左移 32 位
            s3 = s2;
            s2 = s1;
            s1 = s0;
            s0 = 0;
        } else if (total_shift == (OFFSET_32 + OFFSET_16)) {
            // 左移 48 位 （32 + 16）
            s3 = (s1 >> OFFSET_16);
            s2 = (s1 << OFFSET_16) | (s0 >> OFFSET_16);
            s1 = (s0 << OFFSET_16);
            s0 = 0;
        } else if (total_shift == OFFSET_64) {
            // 左移 64 位
            s3 = s1;
            s2 = s0;
            s1 = 0;
            s0 = 0;
        } else if (total_shift == (OFFSET_64 + OFFSET_16)) {
            // 左移 80 位 （64 + 16）
            s3 = (s0 >> OFFSET_16);
            s2 = (s0 << OFFSET_16);
            s1 = 0;
            s0 = 0;
        } else if (total_shift == (OFFSET_64 + OFFSET_32)) {
            // 左移 96 位
            s3 = s0;
            s2 = 0;
            s1 = 0;
            s0 = 0;
        }
        int bias = -total_shift;

        // 3. CLZ 计算块内单比特对齐 n
        int n = NOSIGN_OFFSET_32 - __builtin_clz(s3);
        unsigned int mc_high = 0, mc_low = 0;
        int sc = sa ^ sb;
        int ec = ea + (n - BIT_ALIGN) + eb + EXP_BIAS + bias;

        if (ec >= MAX_EXP) { // 处理输出inf
            ec = MAX_EXP;
        } else if (ec <= 0) { // subnormal
            // exp = -1023 + ec
            if (ec <= SUBN_BOUND_52) {
                c.d = 0;
                dst[index] = c.d;
                continue;
            }
            // subnormal
            if (ec <= SUBN_BOUND_32) { // 意味着 ec 位于 [-52, -33]，需要右移 32 位
                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = 0;
                ec += OFFSET_32;
            } else if (ec <= SUBN_BOUND_16) { // 意味着 ec 位于 [-32,-17],需要右移 16 位
                // 仅进行一次 16 位右移
                s0 = (s0 >> OFFSET_16) | (s1 << OFFSET_16);
                s1 = (s1 >> OFFSET_16) | (s2 << OFFSET_16);
                s2 = (s2 >> OFFSET_16) | (s3 << OFFSET_16);
                s3 = s3 >> OFFSET_16;
                ec += OFFSET_16; // ec 现在位于 [-16, 0]
            }
            // 原来是n， 现在是 20 + ec
            // 0 <= n < 16
            int lshift = MAX_SAFE_LSHIFT_20 + ec - n; // -16 <= ec < 0 --> -13 < lshift < 19
            if (lshift > 0) {                         // 0 < lshift < 19
                mc_high = s3 << lshift;
                mc_high += s2 >> (OFFSET_32 - lshift);
                unsigned int roundvalue = 1 << (HIGH_BIT_OFFSET_32 - lshift);
                mc_low = (s2 << lshift);
                mc_low += ((s1 >> 1) + roundvalue) >> (NOSIGN_OFFSET_32 - lshift); // 如此处理，是为了担心溢出
            } else if (lshift < 0) {                                               // 0 < rshift < 13
                int rshift = -lshift;
                mc_high = s3 >> rshift;
                mc_low = s3 << (OFFSET_32 - rshift);
                unsigned int roundvalue = 1 << (rshift - 1);
                mc_low += (s2 + roundvalue) >> (rshift); // 这里不会溢出
            } else {
                mc_high = s3;
                mc_low = s2;
                mc_low += ((s1 >> 1) + (1 << HIGH_BIT_OFFSET_32)) >> NOSIGN_OFFSET_32; // 同样，担心溢出
            }
            ec = 0;
        } else {
            // 处理规格化数
            int lshift = EXP_OFFSET - n;
            if (lshift > 0) {
                mc_high = (s3 << lshift) - (1 << EXP_OFFSET);
                mc_high += s2 >> (OFFSET_32 - lshift);
                mc_low = s2 << lshift;
                unsigned int roundvalue = 1 << (HIGH_BIT_OFFSET_32 - lshift);

                unsigned int s1value = (((s1 >> 1) + roundvalue) >> (NOSIGN_OFFSET_32 - lshift));
                int p = (~mc_low < s1value);
                mc_low += s1value;
                mc_high += p;
            } else if (lshift < 0) {
                int rshift = -lshift;
                mc_high = (s3 >> rshift) - (1 << EXP_OFFSET);
                mc_low = s3 << (OFFSET_32 - rshift);
                unsigned int roundvalue = 1 << (rshift - 1);

                unsigned int s2value = (s2 + roundvalue) >> (rshift);
                int p = (~mc_low < s2value);
                mc_low += s2value;
                mc_high += p;
            } else {
                mc_high = s3 - (1 << EXP_OFFSET);
                mc_low = s2;
                mc_low += ((s1 >> 1) + (1 << HIGH_BIT_OFFSET_32)) >> NOSIGN_OFFSET_32;

                unsigned int s2value = ((s1 >> 1) + (1 << HIGH_BIT_OFFSET_32)) >> NOSIGN_OFFSET_32;
                int p = (~mc_low < s2value);
                mc_low += s2value;
                mc_high += p;
            }
        }
        c.i[1] = sc + (ec << EXP_OFFSET) + mc_high;
        c.i[0] = mc_low;
        dst[index] = c.d;
    }
}
#endif

template <class T>
struct MulDouble : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline MulDouble(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        asc_vf_call<MulDouble_vf<T>>(dim3(1024), dst_1, src1_1, src2_1, count);
#endif
    }
};

// 支持float, int32, int64, int16, complex64
template <typename T>
struct MulOp {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using Y = Bind<Vec::Mul<T>, InputX1, InputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持float16, bfloat16 升精度计算
template <typename T>
struct MulXfp16Op {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using CastX1 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<float>, CastX1, CastX2>;
    using YB16 = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YB16>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持complex32升精度计算
template <typename T, typename PromoteT>
struct MulComplex32Op {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using CastX1 = Bind<CastComplex32ToComplex64<PromoteT, T>, InputX1>;
    using CastX2 = Bind<CastComplex32ToComplex64<PromoteT, T>, InputX2>;

    using Y = Bind<Vec::Mul<PromoteT>, CastX1, CastX2>;
    using YCast = Bind<CastComplex64ToComplex32<T, PromoteT>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持float16/bfloat16/float 混合数据类型
template <typename T1, typename T2, typename PromoteT>
struct MulMixFpOp {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T2>, Placeholder::In1<T2>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T1, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T2, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<PromoteT>, CastX1, CastX2>;

    using OpCopyOut = Bind<Vec::CopyOut<PromoteT>, Placeholder::Out0<PromoteT>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持int8
struct MulInt8Op {
    using InputX1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using InputX2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using CastX1 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<int32_t>, CastX1, CastX2>;
    using Y1 = Bind<AndFF<int32_t>, Y>;

    // cast成uint8_t 避免转int8_t溢出(饱和模式)
    using Y2 = Bind<Vec::Cast<uint8_t, int32_t, CAST_MODE_NONE>, Y1>;

    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, Y2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持uint8
struct MulUint8Op {
    using InputX1 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In0<uint8_t>>;
    using InputX2 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In1<uint8_t>>;
    using CastX1 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<uint32_t>, CastX1, CastX2>;
    using Y1 = Bind<AndFF<int32_t>, Y>;

    // using Y1 = Bind<AndFF<uint32_t>, Y>;
    using Y2 = Bind<Vec::Cast<uint8_t, uint32_t, CAST_MODE_NONE>, Y1>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, Y2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持bool
struct MulBoolOp {
    using InputX1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using InputX2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using CastX1 = Bind<Vec::Cast<half, int8_t, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<half, int8_t, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<half>, CastX1, CastX2>;
    using YCast = Bind<Vec::Cast<int8_t, half, CAST_MODE_ROUND>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, YCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 支持Double
template <typename T>
struct MulDoubleOp {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using Y = Bind<MulDouble<T>, InputX1, InputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace MulDag
#endif // MUL_DAG_H
