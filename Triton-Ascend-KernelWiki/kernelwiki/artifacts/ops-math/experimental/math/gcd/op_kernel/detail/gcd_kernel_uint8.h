/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_UINT8_H_
#define DETAIL_GCD_KERNEL_UINT8_H_

template <>
class GcdKernel<uint8_t> {
public:
    __aicore__ inline GcdKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y), static_cast<uint64_t>((tiling_.totalNum + 3) / 4));
        x1ByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x1));
        x2ByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x2));
        yByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        vectorEnabled_ = IsContiguousElementwise(tiling_) && tiling_.totalNum >= GCD_UINT8_VECTOR_MIN_ELEMENTS;
        if (vectorEnabled_) {
            InitGcdByteVectorWorkBuffers(buffers_, GCD_UINT8_VECTOR_TILE, sizeof(uint8_t));
        }
    }

    __aicore__ inline void Process()
    {
        if (IsContiguousElementwise(tiling_)) {
            ProcessPackedKernel<GcdKernel<uint8_t>, 4, 8>(*this, tiling_, true, vectorEnabled_,
                                                          GCD_UINT8_VECTOR_ALIGN_ELEMENTS, GCD_UINT8_VECTOR_TILE, yGm_);
            return;
        }
        if (IsSmallRank3ComplementaryBroadcast()) {
            ProcessSmallRank3ComplementaryBroadcast();
            return;
        }
        ProcessPackedKernel<GcdKernel<uint8_t>, 4, 8>(*this, tiling_, false, vectorEnabled_,
                                                      GCD_UINT8_VECTOR_ALIGN_ELEMENTS, GCD_UINT8_VECTOR_TILE, yGm_);
    }

    __aicore__ inline uint8_t ComputeContiguousLaneBits(int64_t word, int64_t lane)
    {
        uint32_t lhsWord = x1Gm_.GetValue(static_cast<uint64_t>(word));
        uint32_t rhsWord = x2Gm_.GetValue(static_cast<uint64_t>(word));
        uint8_t lhsBits = static_cast<uint8_t>((lhsWord >> (lane * 8)) & 0xffU);
        uint8_t rhsBits = static_cast<uint8_t>((rhsWord >> (lane * 8)) & 0xffU);
        return GcdOp::GcdUint8Bits(lhsBits, rhsBits);
    }

    __aicore__ inline uint8_t ComputeStridedLaneBits(int64_t x1Offset, int64_t x2Offset)
    {
        return GcdOp::GcdUint8Bits(ReadPackedByte(x1Gm_, x1Offset), ReadPackedByte(x2Gm_, x2Offset));
    }

    __aicore__ inline bool ComputePackedVectorTile(int64_t linear, int32_t count, bool syncBeforeScalarTail)
    {
        ComputeByteVectorTile<uint8_t, false>(buffers_, x1ByteGm_, x2ByteGm_, yByteGm_, linear, count,
                                              GCD_UINT8_VECTOR_MAX_ITER, syncBeforeScalarTail);
        return true;
    }

private:
    __aicore__ inline bool IsSmallRank3ComplementaryBroadcast() const
    {
        if (tiling_.rank != 3 || tiling_.totalNum > 1024) {
            return false;
        }
        bool x1BroadcastMidX2BroadcastLast = tiling_.x1Strides[0] == tiling_.outputDims[2] &&
                                             tiling_.x1Strides[1] == 0 && tiling_.x1Strides[2] == 1 &&
                                             tiling_.x2Strides[0] == tiling_.outputDims[1] &&
                                             tiling_.x2Strides[1] == 1 && tiling_.x2Strides[2] == 0;
        bool x1BroadcastLastX2BroadcastMid = tiling_.x1Strides[0] == tiling_.outputDims[1] &&
                                             tiling_.x1Strides[1] == 1 && tiling_.x1Strides[2] == 0 &&
                                             tiling_.x2Strides[0] == tiling_.outputDims[2] &&
                                             tiling_.x2Strides[1] == 0 && tiling_.x2Strides[2] == 1;
        return x1BroadcastMidX2BroadcastLast || x1BroadcastLastX2BroadcastMid;
    }

    __aicore__ inline void ProcessSmallRank3ComplementaryBroadcast()
    {
        int64_t d0 = tiling_.outputDims[0];
        int64_t d1 = tiling_.outputDims[1];
        int64_t d2 = tiling_.outputDims[2];
        uint32_t packed = 0;
        int64_t lane = 0;
        int64_t wordIndex = 0;
        for (int64_t i = 0; i < d0; ++i) {
            int64_t x1Outer = i * tiling_.x1Strides[0];
            int64_t x2Outer = i * tiling_.x2Strides[0];
            for (int64_t j = 0; j < d1; ++j) {
                int64_t x1Mid = x1Outer + j * tiling_.x1Strides[1];
                int64_t x2Mid = x2Outer + j * tiling_.x2Strides[1];
                if (tiling_.x2Strides[2] == 0) {
                    uint8_t rhs = ReadPackedByte(x2Gm_, x2Mid);
                    for (int64_t k = 0; k < d2; ++k) {
                        uint8_t bits = GcdOp::GcdUint8Bits(ReadPackedByte(x1Gm_, x1Mid + k * tiling_.x1Strides[2]),
                                                           rhs);
                        WritePackedLane(bits, packed, lane, wordIndex);
                    }
                } else {
                    uint8_t lhs = ReadPackedByte(x1Gm_, x1Mid);
                    for (int64_t k = 0; k < d2; ++k) {
                        uint8_t bits = GcdOp::GcdUint8Bits(lhs,
                                                           ReadPackedByte(x2Gm_, x2Mid + k * tiling_.x2Strides[2]));
                        WritePackedLane(bits, packed, lane, wordIndex);
                    }
                }
            }
        }
        if (lane != 0) {
            yGm_.SetValue(static_cast<uint64_t>(wordIndex), packed);
        }
    }

    __aicore__ inline void WritePackedLane(uint8_t bits, uint32_t& packed, int64_t& lane, int64_t& wordIndex)
    {
        packed |= static_cast<uint32_t>(bits) << (lane * 8);
        ++lane;
        if (lane == 4) {
            yGm_.SetValue(static_cast<uint64_t>(wordIndex), packed);
            ++wordIndex;
            packed = 0;
            lane = 0;
        }
    }

    GcdTilingData tiling_;
    bool vectorEnabled_ = false;
    GcdByteVectorWorkBuffers buffers_;
    GlobalTensor<uint32_t> x1Gm_;
    GlobalTensor<uint32_t> x2Gm_;
    GlobalTensor<uint32_t> yGm_;
    GlobalTensor<uint8_t> x1ByteGm_;
    GlobalTensor<uint8_t> x2ByteGm_;
    GlobalTensor<uint8_t> yByteGm_;
};

#endif // DETAIL_GCD_KERNEL_UINT8_H_
