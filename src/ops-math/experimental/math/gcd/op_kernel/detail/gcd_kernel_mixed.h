/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_MIXED_H_
#define DETAIL_GCD_KERNEL_MIXED_H_

template <bool BF16_LHS>
class GcdKernelUint8Bf16ToUint8 {
public:
    __aicore__ inline GcdKernelUint8Bf16ToUint8() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        if constexpr (BF16_LHS) {
            bf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x1));
            byteWordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x2));
        } else {
            byteWordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x1));
            bf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x2));
        }
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 3) / 4));
    }

    __aicore__ inline void Process()
    {
        ProcessMixedPackedKernel<GcdKernelUint8Bf16ToUint8<BF16_LHS>, 4, 8>(*this, tiling_, yGm_);
    }

    __aicore__ inline uint8_t ComputeMixedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        int64_t byteOffset = BF16_LHS ? x2Offset : x1Offset;
        int64_t bf16Offset = BF16_LHS ? x1Offset : x2Offset;
        uint64_t lhs = BF16_LHS ? GcdOp::AbsBf16BitsToU64(ReadBf16(bf16Offset)) :
                                  static_cast<uint64_t>(ReadPackedByte(byteWordGm_, byteOffset));
        uint64_t rhs = BF16_LHS ? static_cast<uint64_t>(ReadPackedByte(byteWordGm_, byteOffset)) :
                                  GcdOp::AbsBf16BitsToU64(ReadBf16(bf16Offset));
        uint64_t result = GcdOp::GcdUnsigned64(lhs, rhs);
        uint64_t promoted = GcdOp::AbsBf16BitsToU64(GcdOp::U64ToBf16Bits(result));
        return static_cast<uint8_t>(promoted);
    }

private:
    __aicore__ inline uint16_t ReadBf16(int64_t elementOffset)
    {
        return bf16Gm_.GetValue(static_cast<uint64_t>(elementOffset));
    }

    GcdTilingData tiling_;
    GlobalTensor<uint32_t> byteWordGm_;
    GlobalTensor<uint16_t> bf16Gm_;
    GlobalTensor<uint32_t> yGm_;
};

template <bool FLOAT_LHS>
class GcdKernelInt8FloatToInt8 {
public:
    __aicore__ inline GcdKernelInt8FloatToInt8() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        if constexpr (FLOAT_LHS) {
            floatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1));
            byteWordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x2));
        } else {
            byteWordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x1));
            floatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2));
        }
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 3) / 4));
    }

    __aicore__ inline void Process()
    {
        ProcessMixedPackedKernel<GcdKernelInt8FloatToInt8<FLOAT_LHS>, 4, 8>(*this, tiling_, yGm_);
    }

    __aicore__ inline uint8_t ComputeMixedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        int64_t byteOffset = FLOAT_LHS ? x2Offset : x1Offset;
        int64_t floatOffset = FLOAT_LHS ? x1Offset : x2Offset;
        uint64_t lhs = FLOAT_LHS ? GcdOp::AbsToU64<float>(ReadFloat(floatOffset)) :
                                   GcdOp::AbsI8BitsToU64(ReadPackedByte(byteWordGm_, byteOffset));
        uint64_t rhs = FLOAT_LHS ? GcdOp::AbsI8BitsToU64(ReadPackedByte(byteWordGm_, byteOffset)) :
                                   GcdOp::AbsToU64<float>(ReadFloat(floatOffset));
        return static_cast<uint8_t>(GcdOp::GcdUnsigned64(lhs, rhs));
    }

private:
    __aicore__ inline float ReadFloat(int64_t elementOffset)
    {
        return floatGm_.GetValue(static_cast<uint64_t>(elementOffset));
    }

    GcdTilingData tiling_;
    GlobalTensor<uint32_t> byteWordGm_;
    GlobalTensor<float> floatGm_;
    GlobalTensor<uint32_t> yGm_;
};

template <bool FP16_LHS>
class GcdKernelInt16Fp16ToInt16 {
public:
    __aicore__ inline GcdKernelInt16Fp16ToInt16() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        if constexpr (FP16_LHS) {
            fp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x1));
            int16WordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x2));
        } else {
            int16WordGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x1));
            fp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x2));
        }
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 1) / 2));
    }

    __aicore__ inline void Process()
    {
        ProcessMixedPackedKernel<GcdKernelInt16Fp16ToInt16<FP16_LHS>, 2, 16>(*this, tiling_, yGm_);
    }

    __aicore__ inline uint16_t ComputeMixedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        int64_t int16Offset = FP16_LHS ? x2Offset : x1Offset;
        int64_t fp16Offset = FP16_LHS ? x1Offset : x2Offset;
        uint64_t lhs = FP16_LHS ? GcdOp::AbsFp16BitsToU64(ReadFp16(fp16Offset)) : ReadInt16PromotedFp16Abs(int16Offset);
        uint64_t rhs = FP16_LHS ? ReadInt16PromotedFp16Abs(int16Offset) : GcdOp::AbsFp16BitsToU64(ReadFp16(fp16Offset));
        uint64_t result = GcdOp::GcdUnsigned64(lhs, rhs);
        uint64_t promoted = GcdOp::AbsFp16BitsToU64(GcdOp::U64ToFp16Bits(result));
        return static_cast<uint16_t>(promoted);
    }

private:
    __aicore__ inline uint16_t ReadFp16(int64_t elementOffset)
    {
        return fp16Gm_.GetValue(static_cast<uint64_t>(elementOffset));
    }

    __aicore__ inline uint64_t ReadInt16PromotedFp16Abs(int64_t elementOffset)
    {
        uint16_t int16Bits = ReadPackedHalf(int16WordGm_, elementOffset);
        uint16_t fp16Bits = GcdOp::U64ToFp16Bits(GcdOp::AbsI16BitsToU64(int16Bits));
        return GcdOp::AbsFp16BitsToU64(fp16Bits);
    }

    GcdTilingData tiling_;
    GlobalTensor<uint32_t> int16WordGm_;
    GlobalTensor<uint16_t> fp16Gm_;
    GlobalTensor<uint32_t> yGm_;
};

#endif // DETAIL_GCD_KERNEL_MIXED_H_
