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
 * \file arg_max_with_value_base.h
 * \brief Shared base for the ArgWithValue pattern kernels.
 *
 * Holds the GM tensors, this core's output range, and the small set of helpers every pattern needs:
 * dtype sentinels, fp32 cast in/out (the reduce math runs in fp32 for half/bf16/int16 so the int32
 * index stays exact), a point-to-point sync, and the contiguous output store. The L2 op_api copies the
 * kernel's *contiguous* result into the user's (possibly strided) tensors via ViewCopy, so the kernel
 * never deals with output strides.
 */
#ifndef ARG_MAX_WITH_VALUE_BASE_H
#define ARG_MAX_WITH_VALUE_BASE_H

#include "kernel_operator.h"
#include "arg_max_with_value_tiling_data.h"

namespace ArgWithValueNs {
using namespace AscendC;

template <typename T, bool IS_MIN>
class ArgBase {
public:
    __aicore__ inline ArgBase() {}

protected:
    // Non-fp32 inputs are reduced in fp32 (exact for int16 and for the represented result index).
    static constexpr bool USE_CAST = !IsSameType<T, float>::value;

    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR indice, GM_ADDR values,
                                    __tiling_data_ptr__ ArgMaxWithValueTilingData* t)
    {
        axis_ = t->axisSize;
        lastDim_ = t->lastDim;
        outSize_ = t->outSize;
        uint32_t perCore = t->perCore;
        uint32_t bc = t->bigCores; // uneven all-core split: first bc cores get perCore+aln, rest get perCore
        uint32_t blk = GetBlockIdx();
        uint32_t myLen;
        if (bc == 0) { // uniform split (current behaviour): COPY / splitAxis / NLAST-batch
            oStart_ = blk * perCore;
            myLen = perCore;
        } else {
            constexpr uint32_t aln = 32u /
                                     sizeof(T); // value-output 32B align (>= int32 8-align), matches host outAlign
            if (blk < bc) {
                oStart_ = blk * (perCore + aln);
                myLen = perCore + aln;
            } else {
                oStart_ = bc * (perCore + aln) + (blk - bc) * perCore;
                myLen = perCore;
            }
        }
        oLen_ = (oStart_ >= outSize_) ? 0 : ((outSize_ - oStart_) < myLen ? (outSize_ - oStart_) : myLen);
        xGm_ = reinterpret_cast<__gm__ T*>(x);
        valuesGm_ = reinterpret_cast<__gm__ T*>(values);
        indiceGm_ = reinterpret_cast<__gm__ int32_t*>(indice);
    }

    __aicore__ inline uint32_t RoundUp(uint32_t a, uint32_t b) { return (a + b - 1) / b * b; }

    template <typename U>
    __aicore__ inline void RawLoad(const LocalTensor<U>& dst, __gm__ U* src, uint64_t bytes)
    {
        static_assert(sizeof(U) == 2u || sizeof(U) == 4u, "raw MTE supports 16/32-bit elements");
        __ubuf__ U* ub = reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr());
        if constexpr (sizeof(U) == 2u)
            copy_gm_to_ubuf_align_b16(ub, src, 0, 1, bytes, 0, 0, 0, 0);
        else
            copy_gm_to_ubuf_align_b32(ub, src, 0, 1, bytes, 0, 0, 0, 0);
    }

    template <typename U>
    __aicore__ inline void RawStore(__gm__ U* dst, const LocalTensor<U>& src, uint64_t bytes)
    {
        static_assert(sizeof(U) == 2u || sizeof(U) == 4u, "raw MTE supports 16/32-bit elements");
        __ubuf__ U* ub = reinterpret_cast<__ubuf__ U*>(src.GetPhyAddr());
        if constexpr (sizeof(U) == 2u)
            copy_ubuf_to_gm_align_b16(dst, ub, 0, 1, bytes, 0, 0, 0, 0);
        else
            copy_ubuf_to_gm_align_b32(dst, ub, 0, 1, bytes, 0, 0, 0, 0);
    }

    template <typename U>
    __aicore__ inline void RawMove(const LocalTensor<U>& dst, const LocalTensor<U>& src, uint32_t count)
    {
        static_assert(sizeof(U) == 2u || sizeof(U) == 4u, "raw UB copy supports 16/32-bit elements");
        copy_ubuf_to_ubuf(reinterpret_cast<__ubuf__ void*>(dst.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ void*>(src.GetPhyAddr()), 0, 1, count * sizeof(U) / 32u, 0, 0);
    }

    template <typename U>
    __aicore__ inline void RawDup(const LocalTensor<U>& dst, U value, uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vector_dup(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), value, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <RoundMode MODE, typename D, typename S>
    __aicore__ inline void RawCast(const LocalTensor<D>& dst, const LocalTensor<S>& src, uint32_t count)
    {
        __ubuf__ D* d = reinterpret_cast<__ubuf__ D*>(dst.GetPhyAddr());
        __ubuf__ S* s = reinterpret_cast<__ubuf__ S*>(src.GetPhyAddr());
        constexpr uint8_t dr = sizeof(D) > sizeof(S) ? 8 : (sizeof(D) < sizeof(S) ? 4 : 8);
        constexpr uint8_t sr = sizeof(D) > sizeof(S) ? 4 : (sizeof(D) < sizeof(S) ? 8 : 8);
        set_mask_count();
        set_vector_mask(0, count);
        if constexpr (IsSameType<D, float>::value && IsSameType<S, int32_t>::value && MODE == RoundMode::CAST_NONE)
            vconv_s322f32(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, float>::value && IsSameType<S, half>::value && MODE == RoundMode::CAST_NONE)
            vconv_f162f32(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, float>::value && IsSameType<S, bfloat16_t>::value &&
                           MODE == RoundMode::CAST_NONE)
            vconv_bf162f32(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, float>::value && IsSameType<S, int16_t>::value && MODE == RoundMode::CAST_NONE)
            vconv_s162f32(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, int32_t>::value && IsSameType<S, float>::value && MODE == RoundMode::CAST_RINT)
            vconv_f322s32r(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, int32_t>::value && IsSameType<S, float>::value &&
                           MODE == RoundMode::CAST_FLOOR)
            vconv_f322s32f(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, int32_t>::value && IsSameType<S, half>::value && MODE == RoundMode::CAST_RINT)
            vconv_f162s32r(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, half>::value && IsSameType<S, float>::value && MODE == RoundMode::CAST_RINT)
            vconv_f322f16r(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, bfloat16_t>::value && IsSameType<S, float>::value &&
                           MODE == RoundMode::CAST_RINT)
            vconv_f322bf16r(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, int16_t>::value && IsSameType<S, float>::value && MODE == RoundMode::CAST_RINT)
            vconv_f322s16r(d, s, 1, 1, 1, dr, sr);
        else if constexpr (IsSameType<D, half>::value && IsSameType<S, int16_t>::value && MODE == RoundMode::CAST_NONE)
            vconv_s162f16(d, s, 1, 1, 1, dr, sr);
        else
            static_assert(sizeof(D) == 0, "unsupported RawCast dtype/round-mode combination");
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawAdds(const LocalTensor<U>& dst, const LocalTensor<U>& src, U value, uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vadds(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src.GetPhyAddr()), value,
              1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawMuls(const LocalTensor<U>& dst, const LocalTensor<U>& src, U value, uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vmuls(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src.GetPhyAddr()), value,
              1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawAdd(const LocalTensor<U>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1,
                                  uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vadd(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src0.GetPhyAddr()),
             reinterpret_cast<__ubuf__ U*>(src1.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawSub(const LocalTensor<U>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1,
                                  uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vsub(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src0.GetPhyAddr()),
             reinterpret_cast<__ubuf__ U*>(src1.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawMin(const LocalTensor<U>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1,
                                  uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vmin(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src0.GetPhyAddr()),
             reinterpret_cast<__ubuf__ U*>(src1.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawMax(const LocalTensor<U>& dst, const LocalTensor<U>& src0, const LocalTensor<U>& src1,
                                  uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vmax(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src0.GetPhyAddr()),
             reinterpret_cast<__ubuf__ U*>(src1.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <Order_t ORDER, bool BARRIER = true, typename U>
    __aicore__ inline void RawWholeReduce(const LocalTensor<U>& dst, const LocalTensor<U>& src, uint8_t repeat,
                                          uint16_t dstRepStride, uint16_t srcBlkStride, uint16_t srcRepStride)
    {
        static_assert(IsSameType<U, half>::value || IsSameType<U, float>::value,
                      "A2 vcmin/vcmax supports half and float");
        if constexpr (IS_MIN)
            vcmin(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src.GetPhyAddr()),
                  repeat, dstRepStride, srcBlkStride, srcRepStride, ORDER);
        else
            vcmax(reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ U*>(src.GetPhyAddr()),
                  repeat, dstRepStride, srcBlkStride, srcRepStride, ORDER);
        if constexpr (BARRIER)
            pipe_barrier(PIPE_V);
    }

    template <typename U>
    __aicore__ inline void RawCompare(const LocalTensor<uint8_t>& dst, const LocalTensor<U>& src0,
                                      const LocalTensor<U>& src1, CMPMODE mode, uint32_t count)
    {
        constexpr uint32_t lanes = 256u / sizeof(U);
        constexpr uint32_t chunkRepeats = 252u;
        uint32_t repeats = count / lanes;
        __ubuf__ uint8_t* d = reinterpret_cast<__ubuf__ uint8_t*>(dst.GetPhyAddr());
        __ubuf__ U* s0 = reinterpret_cast<__ubuf__ U*>(src0.GetPhyAddr());
        __ubuf__ U* s1 = reinterpret_cast<__ubuf__ U*>(src1.GetPhyAddr());
        for (uint32_t done = 0; done < repeats; done += chunkRepeats) {
            uint8_t rep = static_cast<uint8_t>((repeats - done) < chunkRepeats ? repeats - done : chunkRepeats);
            __ubuf__ uint8_t* dc = d + done * lanes / 8u;
            __ubuf__ U* a = s0 + done * lanes;
            __ubuf__ U* b = s1 + done * lanes;
            if constexpr (IsSameType<U, int32_t>::value) {
                vcmpv_eq(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
            } else {
                if (mode == CMPMODE::LT)
                    vcmpv_lt(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
                else if (mode == CMPMODE::GT)
                    vcmpv_gt(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
                else if (mode == CMPMODE::EQ)
                    vcmpv_eq(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
                else if (mode == CMPMODE::LE)
                    vcmpv_le(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
                else if (mode == CMPMODE::GE)
                    vcmpv_ge(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
                else
                    vcmpv_ne(dc, a, b, rep, 1, 1, 1, 8, 8, 8);
            }
        }
        pipe_barrier(PIPE_V);
    }

    template <typename U>
    __aicore__ inline void RawCompareScalar(const LocalTensor<uint8_t>& dst, const LocalTensor<U>& src, U value,
                                            CMPMODE mode, uint32_t count)
    {
        constexpr uint32_t lanes = 256u / sizeof(U);
        constexpr uint32_t chunkRepeats = 252u;
        uint32_t repeats = count / lanes;
        __ubuf__ uint8_t* d = reinterpret_cast<__ubuf__ uint8_t*>(dst.GetPhyAddr());
        __ubuf__ U* s = reinterpret_cast<__ubuf__ U*>(src.GetPhyAddr());
        for (uint32_t done = 0; done < repeats; done += chunkRepeats) {
            uint8_t rep = static_cast<uint8_t>((repeats - done) < chunkRepeats ? repeats - done : chunkRepeats);
            __ubuf__ uint8_t* dc = d + done * lanes / 8u;
            __ubuf__ U* a = s + done * lanes;
            if constexpr (IsSameType<U, int32_t>::value) {
                vcmpvs_eq(dc, a, value, rep, 1, 1, 8, 8);
            } else {
                if (mode == CMPMODE::LT)
                    vcmpvs_lt(dc, a, value, rep, 1, 1, 8, 8);
                else if (mode == CMPMODE::GT)
                    vcmpvs_gt(dc, a, value, rep, 1, 1, 8, 8);
                else if (mode == CMPMODE::EQ)
                    vcmpvs_eq(dc, a, value, rep, 1, 1, 8, 8);
                else if (mode == CMPMODE::LE)
                    vcmpvs_le(dc, a, value, rep, 1, 1, 8, 8);
                else if (mode == CMPMODE::GE)
                    vcmpvs_ge(dc, a, value, rep, 1, 1, 8, 8);
                else
                    vcmpvs_ne(dc, a, value, rep, 1, 1, 8, 8);
            }
        }
        pipe_barrier(PIPE_V);
    }

    template <typename U>
    __aicore__ inline void RawIota(const LocalTensor<U>& dst, U first, uint32_t count)
    {
        static_assert(sizeof(U) == 4u, "RawIota is used only by fp32/int32 index paths");
        __ubuf__ U* d = reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr());
        set_mask_norm();
        set_vector_mask(0, 0x01ULL);
        vector_dup(d, first + static_cast<U>(0), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x02ULL);
        vector_dup(d, first + static_cast<U>(1), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x04ULL);
        vector_dup(d, first + static_cast<U>(2), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x08ULL);
        vector_dup(d, first + static_cast<U>(3), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x10ULL);
        vector_dup(d, first + static_cast<U>(4), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x20ULL);
        vector_dup(d, first + static_cast<U>(5), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x40ULL);
        vector_dup(d, first + static_cast<U>(6), 1, 1, 1, 8, 0);
        set_vector_mask(0, 0x80ULL);
        vector_dup(d, first + static_cast<U>(7), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        for (uint32_t done = 8u; done < count && done < 64u; done += 8u) {
            uint32_t n = (count - done) < 8u ? count - done : 8u;
            set_mask_count();
            set_vector_mask(0, n);
            vadds(d + done, d + done - 8u, static_cast<U>(8), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        for (uint32_t done = 64u; done < count; done += 64u) {
            uint32_t n = (count - done) < 64u ? count - done : 64u;
            set_mask_count();
            set_vector_mask(0, n);
            vadds(d + done, d + done - 64u, static_cast<U>(64), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U>
    __aicore__ inline void RawGatherMask(const LocalTensor<U>& dst, const LocalTensor<U>& src, uint8_t pattern, bool,
                                         uint32_t count, const GatherMaskParams& p, uint64_t& rsvd)
    {
        set_mask_count();
        set_vector_mask(0, count);
        if constexpr (sizeof(U) == 2u) {
            __ubuf__ uint16_t* s = reinterpret_cast<__ubuf__ uint16_t*>(src.GetPhyAddr());
            vreducev2(reinterpret_cast<__ubuf__ uint16_t*>(dst.GetPhyAddr()), s, s, p.repeatTimes, p.src0BlockStride,
                      pattern, p.src0RepeatStride, p.src1RepeatStride);
        } else {
            __ubuf__ uint32_t* s = reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr());
            vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(dst.GetPhyAddr()), s, s, p.repeatTimes, p.src0BlockStride,
                      pattern, p.src0RepeatStride, p.src1RepeatStride);
        }
        rsvd = get_rsvd_cnt();
        pipe_barrier(PIPE_V);
        set_mask_norm();
    }

    template <typename U>
    __aicore__ inline void RawGatherMask(const LocalTensor<U>& dst, const LocalTensor<U>& src,
                                         const LocalTensor<uint32_t>& pattern, bool, uint32_t count,
                                         const GatherMaskParams& p, uint64_t& rsvd)
    {
        static_assert(sizeof(U) == 4u, "uint32 pattern GatherMask requires 32-bit data");
        set_mask_count();
        set_vector_mask(0, count);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(dst.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(pattern.GetPhyAddr()), p.repeatTimes, p.src0BlockStride, 0,
                  p.src0RepeatStride, p.src1RepeatStride);
        rsvd = get_rsvd_cnt();
        pipe_barrier(PIPE_V);
        set_mask_norm();
    }

    template <typename U>
    __aicore__ inline void RawGather(const LocalTensor<U>& dst, const LocalTensor<U>& src,
                                     const LocalTensor<uint32_t>& offsets, uint32_t base, uint32_t count)
    {
        constexpr uint32_t lanes = 256u / sizeof(U);
        uint32_t full = count / lanes;
        uint32_t tail = count % lanes;
        uint32_t srcAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(src.GetPhyAddr())) + base;
        set_mask_norm();
        if (full) {
            set_vector_mask(sizeof(U) == 2u ? ~0ULL : 0ULL, ~0ULL);
            if constexpr (sizeof(U) == 2u)
                vgather(reinterpret_cast<__ubuf__ uint16_t*>(dst.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(offsets.GetPhyAddr()), srcAddr, 8, full);
            else
                vgather(reinterpret_cast<__ubuf__ uint32_t*>(dst.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(offsets.GetPhyAddr()), srcAddr, 8, full);
        }
        if (tail) {
            uint64_t low = tail >= 64u ? ~0ULL : ((1ULL << tail) - 1ULL);
            uint64_t high = tail <= 64u ? 0ULL : ((1ULL << (tail - 64u)) - 1ULL);
            set_vector_mask(high, low);
            if constexpr (sizeof(U) == 2u)
                vgather(reinterpret_cast<__ubuf__ uint16_t*>(dst[full * lanes].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(offsets[full * lanes].GetPhyAddr()), srcAddr, 8, 1);
            else
                vgather(reinterpret_cast<__ubuf__ uint32_t*>(dst[full * lanes].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(offsets[full * lanes].GetPhyAddr()), srcAddr, 8, 1);
        }
        pipe_barrier(PIPE_V);
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <CMPMODE MODE, typename C, typename U>
    __aicore__ inline void RawCompareSelect(const LocalTensor<U>& dst, const LocalTensor<C>& cmp0,
                                            const LocalTensor<C>& cmp1, const LocalTensor<U>& onTrue,
                                            const LocalTensor<U>& onFalse, uint32_t count)
    {
        static_assert(sizeof(C) == sizeof(U), "vcmp/vsel fused path requires equal lane widths");
        constexpr uint32_t lanes = 256u / sizeof(C);
        constexpr uint32_t chunkRepeats = 252u;
        constexpr uint32_t maskRegionBytes = 4096u;
        __ubuf__ C* a = reinterpret_cast<__ubuf__ C*>(cmp0.GetPhyAddr());
        __ubuf__ C* b = reinterpret_cast<__ubuf__ C*>(cmp1.GetPhyAddr());
        __ubuf__ U* d = reinterpret_cast<__ubuf__ U*>(dst.GetPhyAddr());
        __ubuf__ U* t = reinterpret_cast<__ubuf__ U*>(onTrue.GetPhyAddr());
        __ubuf__ U* f = reinterpret_cast<__ubuf__ U*>(onFalse.GetPhyAddr());
        uint32_t fullRepeats = count / lanes;
        uint32_t done = 0;
        if (fullRepeats >= (IsSameType<C, int32_t>::value ? 1u : 3u)) {
            __ubuf__ uint8_t* mask = reinterpret_cast<__ubuf__ uint8_t*>(TMP_UB_OFFSET);
            __ubuf__ uint32_t* descriptor = reinterpret_cast<__ubuf__ uint32_t*>(TMP_UB_OFFSET + maskRegionBytes);
            set_mask_count();
            set_vector_mask(0, 8);
            vector_dup(descriptor, static_cast<uint32_t>(TMP_UB_OFFSET), 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_cmpmask(descriptor);
            pipe_barrier(PIPE_V);
            while (fullRepeats > 0u) {
                uint8_t rep = static_cast<uint8_t>(fullRepeats < chunkRepeats ? fullRepeats : chunkRepeats);
                if constexpr (IsSameType<C, int32_t>::value || MODE == CMPMODE::EQ)
                    vcmpv_eq(mask, a + done, b + done, rep, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::LT)
                    vcmpv_lt(mask, a + done, b + done, rep, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::GT)
                    vcmpv_gt(mask, a + done, b + done, rep, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::LE)
                    vcmpv_le(mask, a + done, b + done, rep, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::GE)
                    vcmpv_ge(mask, a + done, b + done, rep, 1, 1, 1, 8, 8, 8);
                else
                    vcmpv_ne(mask, a + done, b + done, rep, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                uint32_t chunkCount = static_cast<uint32_t>(rep) * lanes;
                set_vector_mask(0, chunkCount);
                vsel(d + done, t + done, f + done, 1, 1, 1, 1, 8, 8, 8,
                     static_cast<uint8_t>(SELMODE::VSEL_TENSOR_TENSOR_MODE));
                pipe_barrier(PIPE_V);
                done += chunkCount;
                fullRepeats -= rep;
            }
            set_mask_norm();
            set_vector_mask(~0ULL, ~0ULL);
            if constexpr (IsSameType<C, int32_t>::value)
                return; // all current bitwise-EQ callers provide a 64-lane-aligned count
        }
        if constexpr (!IsSameType<C, int32_t>::value) {
            set_mask_norm();
            for (; done < count; done += lanes) {
                uint32_t n = (count - done) < lanes ? count - done : lanes;
                uint64_t low = n >= 64u ? ~0ULL : ((1ULL << n) - 1ULL);
                uint64_t high = n <= 64u ? 0ULL : (n == 128u ? ~0ULL : ((1ULL << (n - 64u)) - 1ULL));
                set_vector_mask(high, low);
                if constexpr (MODE == CMPMODE::EQ)
                    vcmp_eq(a + done, b + done, 1, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::LT)
                    vcmp_lt(a + done, b + done, 1, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::GT)
                    vcmp_gt(a + done, b + done, 1, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::LE)
                    vcmp_le(a + done, b + done, 1, 1, 1, 1, 8, 8, 8);
                else if constexpr (MODE == CMPMODE::GE)
                    vcmp_ge(a + done, b + done, 1, 1, 1, 1, 8, 8, 8);
                else
                    vcmp_ne(a + done, b + done, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vsel(d + done, t + done, f + done, 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
        }
        set_vector_mask(~0ULL, ~0ULL);
    }

    template <typename U, typename M, typename V>
    __aicore__ inline void RawSelectExtremum(const LocalTensor<U>& indexDst, const LocalTensor<M>& mask,
                                             const LocalTensor<U>& indexSrc, U scalar, const LocalTensor<V>& valueDst,
                                             const LocalTensor<V>& valueSrc0, const LocalTensor<V>& valueSrc1,
                                             uint32_t count)
    {
        static_assert(sizeof(U) == sizeof(V), "select and extremum must share one counter-mask lane width");
        __ubuf__ U* scalarBuf = reinterpret_cast<__ubuf__ U*>(TMP_UB_OFFSET);
        set_mask_count();
        set_vector_mask(0, 32);
        vector_dup(scalarBuf, scalar, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_cmpmask(scalarBuf);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, count);
        if constexpr (IsSameType<U, int32_t>::value) {
            // A2 vsel has no int32 payload overload.  Select the exact 32-bit index bits through the supported
            // fp32 lane type; this is a type-only reinterpret and emits no conversion instruction.
            vsel(reinterpret_cast<__ubuf__ float*>(indexDst.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ float*>(indexSrc.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ M*>(mask.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8,
                 static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
        } else {
            vsel(reinterpret_cast<__ubuf__ U*>(indexDst.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ U*>(indexSrc.GetPhyAddr()), reinterpret_cast<__ubuf__ M*>(mask.GetPhyAddr()),
                 1, 1, 1, 1, 8, 8, 8, static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
        }
        if constexpr (IS_MIN)
            vmin(reinterpret_cast<__ubuf__ V*>(valueDst.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ V*>(valueSrc0.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ V*>(valueSrc1.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        else
            vmax(reinterpret_cast<__ubuf__ V*>(valueDst.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ V*>(valueSrc0.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ V*>(valueSrc1.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(~0ULL, ~0ULL);
    }

    // Padding / running-init sentinel: strictly worse than any real value, so padded lanes never win.
    __aicore__ inline T PadVal()
    {
        if constexpr (IsSameType<T, half>::value)
            return IS_MIN ? (half)__builtin_inff() : (half)-__builtin_inff();
        else if constexpr (IsSameType<T, int16_t>::value)
            return IS_MIN ? (int16_t)32767 : (int16_t)-32768;
        else
            return IS_MIN ? (T)__builtin_inff() : (T)-__builtin_inff();
    }
    __aicore__ inline void RawDupInitF(const LocalTensor<float>& dst, uint32_t count)
    {
        // Write the IEEE-754 bits directly: the A2 float scalar-immediate path can saturate infinity to 65504.
        RawDup(dst.template ReinterpretCast<uint32_t>(), IS_MIN ? 0x7f800000u : 0xff800000u, count);
    }

    __aicore__ inline void RawDupPad(const LocalTensor<T>& dst, uint32_t count)
    {
        if constexpr (IsSameType<T, float>::value) {
            RawDupInitF(dst.template ReinterpretCast<float>(), count);
        } else if constexpr (IsSameType<T, half>::value) {
            RawDup(dst.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(IS_MIN ? 0x7c00u : 0xfc00u), count);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            RawDup(dst.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(IS_MIN ? 0x7f80u : 0xff80u), count);
        } else {
            RawDup(dst, PadVal(), count);
        }
    }

    template <HardEvent EV>
    __aicore__ inline void Sync()
    {
        if constexpr (EV == HardEvent::MTE2_V) {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        } else if constexpr (EV == HardEvent::V_MTE2) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        } else if constexpr (EV == HardEvent::V_MTE3) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        } else if constexpr (EV == HardEvent::MTE3_V) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        } else if constexpr (EV == HardEvent::MTE2_MTE3) {
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        } else {
            static_assert(EV == HardEvent::MTE3_MTE2, "unsupported hard event");
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    template <HardEvent EV>
    __aicore__ inline void SyncMte3Complete()
    {
        static_assert(EV == HardEvent::MTE3_V || EV == HardEvent::MTE3_MTE2,
                      "MTE3 completion can only release a dependent V or MTE2 pipeline");
        pipe_barrier(PIPE_MTE3);
        Sync<EV>();
    }

    __aicore__ inline LocalTensor<float> ToF(const LocalTensor<T>& x, const LocalTensor<float>& scratch, uint32_t n)
    {
        if constexpr (IsSameType<T, float>::value) {
            return x.template ReinterpretCast<float>();
        } else {
            RawCast<RoundMode::CAST_NONE>(scratch, x, n);
            return scratch;
        }
    }

    // Store a fp32 result back into a non-fp32 output tile. FP32 paths compute in the output tile directly.
    __aicore__ inline void StoreValF(const LocalTensor<T>& oval, const LocalTensor<float>& v, uint32_t n)
    {
        static_assert(!IsSameType<T, float>::value, "fp32 result must already reside in the output tile");
        RawCast<RoundMode::CAST_RINT>(oval, v, n); // exact round-trip for the values we produce
    }

    // Load R rows of `len` elements (rows `gmStride` apart in GM) into a [R][rowW] tile. Each row is
    // right-padded to rowW with a worse-than-any sentinel so padded lanes are inert. One raw aligned MTE2.
    __aicore__ inline void LoadRows(const LocalTensor<T>& dst, uint64_t base, uint32_t R, uint32_t len, uint32_t rowW,
                                    uint32_t gmStride)
    {
        constexpr uint32_t BLK = 32 / sizeof(T);
        if (gmStride == len && rowW == len) { // fully contiguous source and destination: use one large MTE2 transfer
            __ubuf__ T* dstPtr = reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr());
            if constexpr (sizeof(T) == 2u) {
                copy_gm_to_ubuf_align_b16(dstPtr, xGm_ + base, 0, 1, (uint64_t)R * len * sizeof(T), 0, 0, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b32(dstPtr, xGm_ + base, 0, 1, (uint64_t)R * len * sizeof(T), 0, 0, 0, 0);
            }
            return;
        }
        uint32_t rp = (BLK - len % BLK) % BLK;    // right-pad the final partial 32B block
        uint32_t dstDb = (rowW - len - rp) / BLK; // remaining datablock gap up to rowW
        // MTE2 rightPad fills the entire row when dstDb is zero; only a remaining datablock gap needs prefill.
        if (dstDb > 0) {
            RawDupPad(dst, R * rowW);
            Sync<HardEvent::V_MTE2>(); // prefill (V) must finish before the load (MTE2)
        }
        __ubuf__ T* dstPtr = reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr());
        set_mov_pad_val(GetScalarBitcodeValue(PadVal()));
        if constexpr (sizeof(T) == 2u) {
            copy_gm_to_ubuf_align_b16(dstPtr, xGm_ + base, 0, static_cast<uint16_t>(R), len * sizeof(T), 0,
                                      static_cast<uint8_t>(rp), (gmStride - len) * sizeof(T), dstDb);
        } else {
            copy_gm_to_ubuf_align_b32(dstPtr, xGm_ + base, 0, static_cast<uint16_t>(R), len * sizeof(T), 0,
                                      static_cast<uint8_t>(rp), (gmStride - len) * sizeof(T), dstDb);
        }
    }

    // Load only real elements. The tail of the final 32B block and the dstDb gaps are dummy UB: callers must
    // guarantee that padded lanes are masked out or belong to independent output columns that are never stored.
    // This removes the padding fill, its vector instruction, and the V->MTE2 dependency from those paths.
    __aicore__ inline void LoadRowsNoFill(const LocalTensor<T>& dst, uint64_t base, uint32_t R, uint32_t len,
                                          uint32_t rowW, uint32_t gmStride)
    {
        constexpr uint32_t BLK = 32 / sizeof(T);
        if (gmStride == len && rowW == len) { // contiguous fast path: one >=block copy, nothing to skip
            __ubuf__ T* dstPtr = reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr());
            if constexpr (sizeof(T) == 2u) {
                copy_gm_to_ubuf_align_b16(dstPtr, xGm_ + base, 0, 1, (uint64_t)R * len * sizeof(T), 0, 0, 0, 0);
            } else {
                copy_gm_to_ubuf_align_b32(dstPtr, xGm_ + base, 0, 1, (uint64_t)R * len * sizeof(T), 0, 0, 0, 0);
            }
            return;
        }
        uint32_t occupied = this->RoundUp(len, BLK);
        uint32_t dstDb = (rowW - occupied) / BLK;
        __ubuf__ T* dstPtr = reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr());
        if constexpr (sizeof(T) == 2u) {
            copy_gm_to_ubuf_align_b16(dstPtr, xGm_ + base, 0, static_cast<uint16_t>(R), len * sizeof(T), 0, 0,
                                      (gmStride - len) * sizeof(T), dstDb);
        } else {
            copy_gm_to_ubuf_align_b32(dstPtr, xGm_ + base, 0, static_cast<uint16_t>(R), len * sizeof(T), 0, 0,
                                      (gmStride - len) * sizeof(T), dstDb);
        }
    }

    // Contiguous output store of n value + n index elements (output stride is always 1 — see file header).
    __aicore__ inline void StoreOut(uint32_t outOff, uint32_t n, const LocalTensor<T>& oval,
                                    const LocalTensor<int32_t>& oidx)
    {
        __ubuf__ T* valueUb = reinterpret_cast<__ubuf__ T*>(oval.GetPhyAddr());
        __ubuf__ int32_t* indexUb = reinterpret_cast<__ubuf__ int32_t*>(oidx.GetPhyAddr());
        if constexpr (sizeof(T) == 2u) {
            copy_ubuf_to_gm_align_b16(valuesGm_ + outOff, valueUb, 0, 1, n * sizeof(T), 0, 0, 0, 0);
        } else {
            copy_ubuf_to_gm_align_b32(valuesGm_ + outOff, valueUb, 0, 1, n * sizeof(T), 0, 0, 0, 0);
        }
        copy_ubuf_to_gm_align_b32(indiceGm_ + outOff, indexUb, 0, 1, n * sizeof(int32_t), 0, 0, 0, 0);
    }

    __gm__ T* xGm_;
    __gm__ T* valuesGm_;
    __gm__ int32_t* indiceGm_;
    uint32_t axis_, lastDim_, outSize_;
    uint32_t oStart_, oLen_; // this core's output range [oStart_, oStart_ + oLen_)
};
} // namespace ArgWithValueNs
#endif // ARG_MAX_WITH_VALUE_BASE_H
