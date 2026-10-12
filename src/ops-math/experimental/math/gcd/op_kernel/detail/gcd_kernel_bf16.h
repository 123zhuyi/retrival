/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_BF16_H_
#define DETAIL_GCD_KERNEL_BF16_H_

template <>
class GcdKernel<bfloat16_t> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 1) / 2));
        x1Bf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x1));
        x2Bf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x2));
        yBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
        vectorEnabled_ = IsContiguousElementwise(tiling_) && tiling_.totalNum >= GCD_BF16_VECTOR_MIN_ELEMENTS;
        if (vectorEnabled_) {
            InitGcdVectorWorkBuffers(buffers_, GCD_BF16_VECTOR_TILE, sizeof(bfloat16_t), sizeof(bfloat16_t));
        }
    }

    __aicore__ inline void Process()
    {
        ProcessPackedKernel<GcdKernel<bfloat16_t>, 2, 16>(*this, tiling_, IsContiguousElementwise(tiling_),
                                                          vectorEnabled_, GCD_BF16_VECTOR_ALIGN_ELEMENTS,
                                                          GCD_BF16_VECTOR_TILE, yGm_);
    }

    __aicore__ inline uint16_t ComputeContiguousLaneBits(int64_t word, int64_t lane)
    {
        return ComputeBf16Bits(word * 2 + lane, word * 2 + lane);
    }

    __aicore__ inline uint16_t ComputeStridedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        return ComputeBf16Bits(x1Offset, x2Offset);
    }

    __aicore__ inline bool ComputePackedVectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        return ComputeBf16VectorTile(linear, count, syncBeforeScalarTail);
    }

private:
    __aicore__ inline uint16_t ComputeBf16Bits(int64_t x1Offset, int64_t x2Offset)
    {
        uint64_t lhs = GcdOp::AbsBf16BitsToU64(x1Gm_.GetValue(static_cast<uint64_t>(x1Offset)));
        uint64_t rhs = GcdOp::AbsBf16BitsToU64(x2Gm_.GetValue(static_cast<uint64_t>(x2Offset)));
        return GcdOp::U64ToBf16Bits(GcdOp::GcdUnsigned64(lhs, rhs));
    }

    __aicore__ inline bool ComputeBf16VectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        LocalTensor<bfloat16_t> x1Local = buffers_.x1Buf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> x2Local = buffers_.x2Buf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = buffers_.yBuf.Get<bfloat16_t>();
        LocalTensor<float> aLocal = buffers_.aBuf.Get<float>();
        LocalTensor<float> bLocal = buffers_.bBuf.Get<float>();
        LocalTensor<float> t1 = buffers_.t1Buf.Get<float>();
        LocalTensor<float> t2 = buffers_.t2Buf.Get<float>();
        LocalTensor<float> t3 = buffers_.t3Buf.Get<float>();

        PrepareBf16Vector(linear, count, x1Local, x2Local, aLocal, bLocal);
        if (!IsBf16VectorSafe(aLocal, bLocal, t1, t2, t3, count)) {
            return false;
        }
        RoundBf16VectorInputs(aLocal, bLocal, t1, count);
        RunFloatEuclid(aLocal, bLocal, t1, t2, t3, count, GCD_BF16_VECTOR_MAX_ITER);
        StoreBf16Vector(linear, count, syncBeforeScalarTail, yLocal, aLocal);
        return true;
    }

    __aicore__ inline void PrepareBf16Vector(int64_t linear, int32_t count, LocalTensor<bfloat16_t> x1Local,
                                             LocalTensor<bfloat16_t> x2Local, LocalTensor<float> aLocal,
                                             LocalTensor<float> bLocal)
    {
        DataCopy(x1Local, x1Bf16Gm_[static_cast<uint64_t>(linear)], count);
        DataCopy(x2Local, x2Bf16Gm_[static_cast<uint64_t>(linear)], count);
        SyncPipe<HardEvent::MTE2_V>();
        Cast(aLocal, x1Local, RoundMode::CAST_NONE, count);
        Cast(bLocal, x2Local, RoundMode::CAST_NONE, count);
        Abs(aLocal, aLocal, count);
        Abs(bLocal, bLocal, count);
    }

    __aicore__ inline bool IsBf16VectorSafe(LocalTensor<float> aLocal, LocalTensor<float> bLocal, LocalTensor<float> t1,
                                            LocalTensor<float> t2, LocalTensor<float> t3, int32_t count)
    {
        ReduceMax<float>(t1, aLocal, t3, count, false);
        ReduceMax<float>(t2, bLocal, t3, count, false);
        SyncPipe<HardEvent::V_S>();
        float lhsMax = t1.GetValue(0);
        float rhsMax = t2.GetValue(0);
        return lhsMax <= GCD_FLOAT_VECTOR_SAFE_ABS_F && rhsMax <= GCD_FLOAT_VECTOR_SAFE_ABS_F;
    }

    __aicore__ inline void RoundBf16VectorInputs(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
                                                 LocalTensor<float> t1, int32_t count)
    {
        FloorFloatVectorInputs(aLocal, bLocal, t1, count);
    }

    __aicore__ inline void StoreBf16Vector(int64_t linear, int32_t count, bool syncBeforeScalarTail,
                                           LocalTensor<bfloat16_t> yLocal, LocalTensor<float> aLocal)
    {
        Cast(yLocal, aLocal, RoundMode::CAST_RINT, count);
        SyncPipe<HardEvent::V_MTE3>();
        DataCopy(yBf16Gm_[static_cast<uint64_t>(linear)], yLocal, count);
        SyncMte3ToScalarIf(syncBeforeScalarTail);
    }

    GcdTilingData tiling_;
    bool vectorEnabled_ = false;
    GcdVectorWorkBuffers buffers_;
    GlobalTensor<uint16_t> x1Gm_;
    GlobalTensor<uint16_t> x2Gm_;
    GlobalTensor<uint32_t> yGm_;
    GlobalTensor<bfloat16_t> x1Bf16Gm_;
    GlobalTensor<bfloat16_t> x2Bf16Gm_;
    GlobalTensor<bfloat16_t> yBf16Gm_;
};

#endif // DETAIL_GCD_KERNEL_BF16_H_
