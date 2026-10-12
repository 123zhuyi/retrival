/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_INT8_H_
#define DETAIL_GCD_KERNEL_INT8_H_

template <>
class GcdKernel<int8_t> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 3) / 4));
        x1ByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1));
        x2ByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2));
        yByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        vectorEnabled_ = IsContiguousElementwise(tiling_) && tiling_.totalNum >= GCD_UINT8_VECTOR_MIN_ELEMENTS;
        if (vectorEnabled_) {
            InitGcdByteVectorWorkBuffers(buffers_, GCD_UINT8_VECTOR_TILE, sizeof(int8_t));
        }
    }

    __aicore__ inline void Process()
    {
        ProcessPackedKernel<GcdKernel<int8_t>, 4, 8>(*this, tiling_, IsContiguousElementwise(tiling_), vectorEnabled_,
                                                     GCD_UINT8_VECTOR_ALIGN_ELEMENTS, GCD_UINT8_VECTOR_TILE, yGm_);
    }

    __aicore__ inline uint8_t ComputeContiguousLaneBits(int64_t word, int64_t lane)
    {
        uint32_t lhsWord = x1Gm_.GetValue(static_cast<uint64_t>(word));
        uint32_t rhsWord = x2Gm_.GetValue(static_cast<uint64_t>(word));
        uint8_t lhsBits = static_cast<uint8_t>((lhsWord >> (lane * 8)) & 0xffU);
        uint8_t rhsBits = static_cast<uint8_t>((rhsWord >> (lane * 8)) & 0xffU);
        return GcdOp::GcdInt8RawBits(lhsBits, rhsBits);
    }

    __aicore__ inline uint8_t ComputeStridedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        return GcdOp::GcdInt8RawBits(ReadPackedByte(x1Gm_, x1Offset), ReadPackedByte(x2Gm_, x2Offset));
    }

    __aicore__ inline bool ComputePackedVectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        ComputeByteVectorTile<int8_t, true>(buffers_, x1ByteGm_, x2ByteGm_, yByteGm_, linear, count,
                                            GCD_UINT8_VECTOR_MAX_ITER, syncBeforeScalarTail);
        return true;
    }

private:
    GcdTilingData tiling_;
    bool vectorEnabled_ = false;
    GcdByteVectorWorkBuffers buffers_;
    GlobalTensor<uint32_t> x1Gm_;
    GlobalTensor<uint32_t> x2Gm_;
    GlobalTensor<uint32_t> yGm_;
    GlobalTensor<int8_t> x1ByteGm_;
    GlobalTensor<int8_t> x2ByteGm_;
    GlobalTensor<uint8_t> yByteGm_;
};

#endif // DETAIL_GCD_KERNEL_INT8_H_
