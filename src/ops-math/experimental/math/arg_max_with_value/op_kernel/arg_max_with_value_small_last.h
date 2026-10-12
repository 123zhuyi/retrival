/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARG_MAX_WITH_VALUE_SMALL_LAST_H
#define ARG_MAX_WITH_VALUE_SMALL_LAST_H

#include "arg_max_with_value_base.h"

namespace ArgWithValueNs {
using namespace AscendC;

// Low-overhead contiguous LAST kernel. The host admits exactly two algorithmic domains:
//   1. one row, axis <= DIRECT_AXIS_CAP;
//   2. a 32B-output-aligned multi-core row batch fitting one native vcmin mask and the fixed UB layout.
// Fixed UB addresses and local scalars keep unrelated LAST state out of the generated kernel.
template <typename T, bool IS_MIN, bool LONG_ONLY = false, bool PACKED_ONLY = false>
class ArgLastDirect : public ArgBase<T, IS_MIN> {
    static constexpr bool NEEDS_FP32 = IsSameType<T, bfloat16_t>::value || IsSameType<T, int16_t>::value;
    static constexpr uint32_t RAW_ADDR = 0u;
    static constexpr uint32_t FLOAT_ADDR = 32u * 1024u;
    static constexpr uint32_t CHUNK_VALUE_ADDR = 64u * 1024u;
    static constexpr uint32_t CHUNK_INDEX_ADDR = 68u * 1024u;
    static constexpr uint32_t VALUE_ADDR = 72u * 1024u;
    static constexpr uint32_t WIN_CHUNK_ADDR = 73u * 1024u;
    static constexpr uint32_t OFFSET_ADDR = 74u * 1024u;
    static constexpr uint32_t LOCAL_INDEX_ADDR = 75u * 1024u;
    static constexpr uint32_t INDEX_ADDR = 76u * 1024u;
    static constexpr uint32_t OUTPUT_ADDR = 77u * 1024u;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values,
                                __tiling_data_ptr__ ArgMaxWithValueTilingData* t)
    {
        this->InitBase(x, indice, values, t);
        this->xGm_ += static_cast<uint64_t>(this->oStart_) * this->axis_;
        this->valuesGm_ += this->oStart_;
        this->indiceGm_ += this->oStart_;
    }

    __aicore__ inline void Process()
    {
        static_assert(!(LONG_ONLY && PACKED_ONLY), "direct LAST schedule must select one long-axis algorithm");
        if constexpr (PACKED_ONLY && !IsSameType<T, half>::value) {
            ReduceOneLong<true>();
        } else if constexpr (LONG_ONLY) {
            ReduceOneLong<false>();
        } else {
            ReduceRows();
        }
    }

private:
    __aicore__ inline void FillPad(__ubuf__ T* dst, uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        if constexpr (IsSameType<T, float>::value) {
            vector_dup(reinterpret_cast<__ubuf__ uint32_t*>(dst), IS_MIN ? 0x7f800000u : 0xff800000u, 1, 1, 1, 8, 0);
        } else if constexpr (IsSameType<T, half>::value) {
            vector_dup(reinterpret_cast<__ubuf__ uint16_t*>(dst), static_cast<uint16_t>(IS_MIN ? 0x7c00u : 0xfc00u), 1,
                       1, 1, 8, 0);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            vector_dup(reinterpret_cast<__ubuf__ uint16_t*>(dst), static_cast<uint16_t>(IS_MIN ? 0x7f80u : 0xff80u), 1,
                       1, 1, 8, 0);
        } else {
            vector_dup(dst, static_cast<T>(IS_MIN ? 32767 : -32768), 1, 1, 1, 8, 0);
        }
    }

    template <typename U>
    __aicore__ inline void ReduceNative(const LocalTensor<U>& value, const LocalTensor<int32_t>& index,
                                        const LocalTensor<U>& input, uint32_t rows, uint32_t width, uint32_t axis)
    {
        uint64_t low = axis >= 64u ? ~0ULL : ((1ULL << axis) - 1ULL);
        uint64_t high = axis <= 64u ? 0ULL : (axis == 128u ? ~0ULL : ((1ULL << (axis - 64u)) - 1ULL));
        set_mask_norm();
        set_vector_mask(high, low);
        __ubuf__ U* src = reinterpret_cast<__ubuf__ U*>(input.GetPhyAddr());
        const uint8_t repeat = static_cast<uint8_t>(rows);
        const uint16_t srcRepStride = width / (32u / sizeof(U));
        if constexpr (IS_MIN) {
            vcmin(reinterpret_cast<__ubuf__ U*>(value.GetPhyAddr()), src, repeat, 1, 1, srcRepStride,
                  Order_t::ONLY_VALUE);
            vcmin(reinterpret_cast<__ubuf__ U*>(index.GetPhyAddr()), src, repeat, 1, 1, srcRepStride,
                  Order_t::ONLY_INDEX);
        } else {
            vcmax(reinterpret_cast<__ubuf__ U*>(value.GetPhyAddr()), src, repeat, 1, 1, srcRepStride,
                  Order_t::ONLY_VALUE);
            vcmax(reinterpret_cast<__ubuf__ U*>(index.GetPhyAddr()), src, repeat, 1, 1, srcRepStride,
                  Order_t::ONLY_INDEX);
        }
    }

    __aicore__ inline void ReduceRows()
    {
        constexpr uint32_t blockElems = 32u / sizeof(T);
        const uint32_t rows = this->oLen_;
        const uint32_t width = this->RoundUp(this->axis_, blockElems);
        const uint32_t elems = rows * width;
        LocalTensor<T> input(TPosition::VECCALC, RAW_ADDR, elems);
        LocalTensor<int32_t> index(TPosition::VECCALC, INDEX_ADDR, this->RoundUp(rows, 8u));
        LocalTensor<T> output(TPosition::VECCALC, OUTPUT_ADDR, this->RoundUp(rows, blockElems));
        __ubuf__ T* inputPtr = reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr());
        // The exact-axis reduction mask excludes the hardware-filled dummy tail.
        if constexpr (sizeof(T) == 2u) {
            copy_gm_to_ubuf_align_b16(inputPtr, this->xGm_, 0, static_cast<uint16_t>(rows), this->axis_ * sizeof(T), 0,
                                      0, 0, 0);
        } else {
            copy_gm_to_ubuf_align_b32(inputPtr, this->xGm_, 0, static_cast<uint16_t>(rows), this->axis_ * sizeof(T), 0,
                                      0, 0, 0);
        }
        this->template Sync<HardEvent::MTE2_V>();

        if constexpr (NEEDS_FP32) {
            LocalTensor<float> inputF(TPosition::VECCALC, FLOAT_ADDR, elems);
            LocalTensor<float> valueF(TPosition::VECCALC, VALUE_ADDR, this->RoundUp(rows, 8u));
            set_mask_count();
            set_vector_mask(0, elems);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                vconv_bf162f32(reinterpret_cast<__ubuf__ float*>(inputF.GetPhyAddr()), inputPtr, 1, 1, 1, 8, 4);
            } else {
                vconv_s162f32(reinterpret_cast<__ubuf__ float*>(inputF.GetPhyAddr()), inputPtr, 1, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            ReduceNative(valueF, index, inputF, rows, width, this->axis_);
            pipe_barrier(PIPE_V);
            set_mask_count();
            set_vector_mask(0, rows);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                vconv_f322bf16r(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(valueF.GetPhyAddr()), 1, 1, 1, 4, 8);
            } else {
                vconv_f322s16r(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(valueF.GetPhyAddr()), 1, 1, 1, 4, 8);
            }
        } else {
            ReduceNative(output, index, input, rows, width, this->axis_);
        }
        Store(output, index, rows);
    }

    template <bool PACKED>
    __aicore__ inline void ReduceOneLong()
    {
        using R = typename Conditional<IsSameType<T, half>::value, half, float>::type;
        constexpr uint32_t lanes = 256u / sizeof(R);
        const uint32_t width = this->RoundUp(this->axis_, lanes);
        constexpr uint32_t blockElems = 32u / sizeof(T);
        const uint32_t inputAligned = this->RoundUp(this->axis_, blockElems);
        const uint32_t chunks = width / lanes;
        LocalTensor<T> raw(TPosition::VECCALC, RAW_ADDR, width);
        __ubuf__ T* rawPtr = reinterpret_cast<__ubuf__ T*>(raw.GetPhyAddr());
        set_mov_pad_val(GetScalarBitcodeValue(this->PadVal()));
        const uint8_t rightPad = static_cast<uint8_t>(inputAligned - this->axis_);
        if constexpr (sizeof(T) == 2u) {
            copy_gm_to_ubuf_align_b16(rawPtr, this->xGm_, 0, 1, this->axis_ * sizeof(T), 0, rightPad, 0, 0);
        } else {
            copy_gm_to_ubuf_align_b32(rawPtr, this->xGm_, 0, 1, this->axis_ * sizeof(T), 0, rightPad, 0, 0);
        }
        this->template Sync<HardEvent::MTE2_V>();
        if (inputAligned < width) {
            FillPad(rawPtr + inputAligned, width - inputAligned);
            pipe_barrier(PIPE_V);
        }

        LocalTensor<R> input = raw.template ReinterpretCast<R>();
        if constexpr (NEEDS_FP32) {
            input = LocalTensor<R>(TPosition::VECCALC, FLOAT_ADDR, width);
            set_mask_count();
            set_vector_mask(0, width);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                vconv_bf162f32(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()), rawPtr, 1, 1, 1, 8, 4);
            } else {
                vconv_s162f32(reinterpret_cast<__ubuf__ float*>(input.GetPhyAddr()), rawPtr, 1, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
        }

        LocalTensor<R> chunkValue(TPosition::VECCALC, CHUNK_VALUE_ADDR, PACKED ? 2u * chunks : chunks);
        LocalTensor<int32_t> chunkIndex(TPosition::VECCALC, CHUNK_INDEX_ADDR, chunks);
        LocalTensor<R> value(TPosition::VECCALC, VALUE_ADDR, 16u);
        LocalTensor<int32_t> winChunk(TPosition::VECCALC, WIN_CHUNK_ADDR, 8u);
        LocalTensor<int32_t> offset(TPosition::VECCALC, OFFSET_ADDR, 8u);
        LocalTensor<int32_t> localIndex(TPosition::VECCALC, LOCAL_INDEX_ADDR, 8u);
        LocalTensor<int32_t> index(TPosition::VECCALC, INDEX_ADDR, 8u);
        LocalTensor<R> reducedValue = value;
        if constexpr (PACKED) {
            static_assert(IsSameType<R, float>::value, "packed long reduction requires fp32 compute lanes");
            set_mask_norm();
            set_vector_mask(0, ~0ULL);
            this->template RawWholeReduce<Order_t::VALUE_INDEX>(chunkValue, input, static_cast<uint8_t>(chunks), 1, 1,
                                                                8);
            const uint32_t pairWords = 2u * chunks;
            const uint64_t valid = pairWords == 64u ? ~0ULL : ((1ULL << pairWords) - 1ULL);
            set_vector_mask(0, 0x5555555555555555ULL & valid);
            this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(value, chunkValue, 1, 1, 1, 8);
            this->template RawWholeReduce<Order_t::ONLY_INDEX>(winChunk.template ReinterpretCast<R>(), chunkValue, 1, 1,
                                                               1, 8);

            set_mask_count();
            set_vector_mask(0, 1);
            vmuls(reinterpret_cast<__ubuf__ int32_t*>(offset.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()), static_cast<int32_t>(sizeof(R)), 1, 1, 1,
                  8, 8);
            vmuls(reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()), static_cast<int32_t>(lanes / 2u), 1, 1, 1,
                  8, 8);
            pipe_barrier(PIPE_V);
            vgather(reinterpret_cast<__ubuf__ uint32_t*>(localIndex.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(offset.GetPhyAddr()),
                    static_cast<uint32_t>(reinterpret_cast<uint64_t>(chunkValue.GetPhyAddr())) + sizeof(R), 8, 1);
            pipe_barrier(PIPE_V);
            vadd(reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        } else {
            if (chunks > lanes && chunks < 2u * lanes) {
                // The second level consumes two full repeats.  Make the unused tail of repeat 1 inert;
                // otherwise it reduces uninitialized UB when only 65..127 chunk results are valid.
                constexpr uint32_t blockElems = 32u / sizeof(R);
                const uint32_t tailStart = chunks / blockElems * blockElems;
                set_mask_count();
                set_vector_mask(0, 2u * lanes - tailStart);
                if constexpr (IsSameType<R, float>::value) {
                    vector_dup(reinterpret_cast<__ubuf__ uint32_t*>(chunkValue[tailStart].GetPhyAddr()),
                               IS_MIN ? 0x7f800000u : 0xff800000u, 1, 1, 1, 8, 0);
                } else {
                    vector_dup(reinterpret_cast<__ubuf__ uint16_t*>(chunkValue[tailStart].GetPhyAddr()),
                               static_cast<uint16_t>(IS_MIN ? 0x7c00u : 0xfc00u), 1, 1, 1, 8, 0);
                }
                pipe_barrier(PIPE_V);
            }
            set_mask_norm();
            set_vector_mask(sizeof(R) == 2u ? ~0ULL : 0ULL, ~0ULL);
            this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(chunkValue, input, static_cast<uint8_t>(chunks),
                                                                      1, 1, 8);
            this->template RawWholeReduce<Order_t::ONLY_INDEX>(chunkIndex.template ReinterpretCast<R>(), input,
                                                               static_cast<uint8_t>(chunks), 1, 1, 8);
            if (chunks > lanes) {
                // A float repeat covers 64 values. Reduce 65..128 chunk extrema in two groups, then fold the groups.
                this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(value, chunkValue, 2, 1, 1, 8);
                this->template RawWholeReduce<Order_t::ONLY_INDEX>(winChunk.template ReinterpretCast<R>(), chunkValue,
                                                                   2, 1, 1, 8);
                set_vector_mask(0, 3u);
                reducedValue = value[8];
                this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(reducedValue, value, 1, 1, 1, 8);
                this->template RawWholeReduce<Order_t::ONLY_INDEX>(localIndex.template ReinterpretCast<R>(), value, 1,
                                                                   1, 1, 8);
                set_mask_count();
                set_vector_mask(0, 1);
                vmuls(reinterpret_cast<__ubuf__ int32_t*>(offset.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr()),
                      static_cast<int32_t>(sizeof(int32_t)), 1, 1, 1, 8, 8);
                vmuls(reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr()), static_cast<int32_t>(lanes), 1, 1,
                      1, 8, 8);
                pipe_barrier(PIPE_V);
                vgather(reinterpret_cast<__ubuf__ uint32_t*>(index.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(offset.GetPhyAddr()),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(winChunk.GetPhyAddr())), 8, 1);
                pipe_barrier(PIPE_V);
                vadd(reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr()),
                     reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            } else {
                set_vector_mask(0, chunks == 64u ? ~0ULL : ((1ULL << chunks) - 1ULL));
                this->template RawWholeReduce<Order_t::ONLY_VALUE, false>(value, chunkValue, 1, 1, 1, 8);
                this->template RawWholeReduce<Order_t::ONLY_INDEX>(winChunk.template ReinterpretCast<R>(), chunkValue,
                                                                   1, 1, 1, 8);
            }
            set_mask_count();
            set_vector_mask(0, 1);
            vmuls(reinterpret_cast<__ubuf__ int32_t*>(offset.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()), static_cast<int32_t>(sizeof(int32_t)), 1,
                  1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vgather(reinterpret_cast<__ubuf__ uint32_t*>(localIndex.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(offset.GetPhyAddr()),
                    static_cast<uint32_t>(reinterpret_cast<uint64_t>(chunkIndex.GetPhyAddr())), 8, 1);
            vmuls(reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()), static_cast<int32_t>(lanes), 1, 1, 1, 8,
                  8);
            pipe_barrier(PIPE_V);
            vadd(reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ int32_t*>(winChunk.GetPhyAddr()),
                 reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr()), 1, 1, 1, 1, 8, 8, 8);
        }

        LocalTensor<T> output = reducedValue.template ReinterpretCast<T>();
        if constexpr (NEEDS_FP32) {
            output = LocalTensor<T>(TPosition::VECCALC, OUTPUT_ADDR, 16u);
            set_mask_count();
            set_vector_mask(0, 1);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                vconv_f322bf16r(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(reducedValue.GetPhyAddr()), 1, 1, 1, 4, 8);
            } else {
                vconv_f322s16r(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(reducedValue.GetPhyAddr()), 1, 1, 1, 4, 8);
            }
        }
        Store(output, index, 1u);
    }

    __aicore__ inline void ReduceHalfLongRows()
    {
        constexpr uint32_t chunks = 8u;
        constexpr uint32_t blockElems = 16u;
        const uint32_t rows = this->oLen_;
        const uint32_t chunk = this->RoundUp((this->axis_ + chunks - 1u) / chunks, blockElems);
        const uint32_t width = chunk * chunks;
        const uint32_t inputAligned = this->RoundUp(this->axis_, blockElems);
        LocalTensor<half> input(TPosition::VECCALC, RAW_ADDR, rows * width);
        LocalTensor<half> pairs(TPosition::VECCALC, CHUNK_VALUE_ADDR, rows * chunks * 2u);
        LocalTensor<half> value(TPosition::VECCALC, VALUE_ADDR, rows * blockElems);
        LocalTensor<int32_t> winner(TPosition::VECCALC, WIN_CHUNK_ADDR, rows * 8u);
        LocalTensor<int32_t> offsets(TPosition::VECCALC, OFFSET_ADDR, rows * 8u);
        LocalTensor<int32_t> localIndex(TPosition::VECCALC, LOCAL_INDEX_ADDR, rows * 8u);
        LocalTensor<int32_t> index(TPosition::VECCALC, INDEX_ADDR, rows * 8u);
        __ubuf__ half* inputPtr = reinterpret_cast<__ubuf__ half*>(input.GetPhyAddr());
        __ubuf__ half* pairsPtr = reinterpret_cast<__ubuf__ half*>(pairs.GetPhyAddr());
        __ubuf__ half* valuePtr = reinterpret_cast<__ubuf__ half*>(value.GetPhyAddr());
        __ubuf__ int32_t* winnerPtr = reinterpret_cast<__ubuf__ int32_t*>(winner.GetPhyAddr());
        __ubuf__ int32_t* offsetPtr = reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr());
        __ubuf__ int32_t* localPtr = reinterpret_cast<__ubuf__ int32_t*>(localIndex.GetPhyAddr());
        __ubuf__ int32_t* indexPtr = reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr());

        set_mov_pad_val(static_cast<uint16_t>(IS_MIN ? 0x7c00u : 0xfc00u));
        copy_gm_to_ubuf_align_b16(inputPtr, this->xGm_, 0, static_cast<uint16_t>(rows), this->axis_ * sizeof(half), 0,
                                  static_cast<uint8_t>(inputAligned - this->axis_), 0,
                                  (width - inputAligned) / blockElems);
        this->template Sync<HardEvent::MTE2_V>();
        if (inputAligned < width) {
            const uint32_t tail = width - inputAligned;
            set_mask_norm();
            set_vector_mask(tail <= 64u ? 0ULL : ((1ULL << (tail - 64u)) - 1ULL),
                            tail >= 64u ? ~0ULL : ((1ULL << tail) - 1ULL));
            vector_dup(reinterpret_cast<__ubuf__ uint16_t*>(inputPtr + inputAligned),
                       static_cast<uint16_t>(IS_MIN ? 0x7c00u : 0xfc00u), static_cast<uint8_t>(rows), 1, 1,
                       width / blockElems, 0);
            pipe_barrier(PIPE_V);
        }

        set_mask_norm();
        set_vector_mask(chunk <= 64u ? 0ULL : ((1ULL << (chunk - 64u)) - 1ULL),
                        chunk >= 64u ? ~0ULL : ((1ULL << chunk) - 1ULL));
        if constexpr (IS_MIN) {
            vcmin(pairsPtr, inputPtr, static_cast<uint8_t>(rows * chunks), 1, 1, chunk / blockElems,
                  Order_t::VALUE_INDEX);
        } else {
            vcmax(pairsPtr, inputPtr, static_cast<uint8_t>(rows * chunks), 1, 1, chunk / blockElems,
                  Order_t::VALUE_INDEX);
        }
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0x5555ULL);
        if constexpr (IS_MIN) {
            vcmin(valuePtr, pairsPtr, static_cast<uint8_t>(rows), 1, 1, 1, Order_t::ONLY_VALUE);
            vcmin(reinterpret_cast<__ubuf__ half*>(winnerPtr), pairsPtr, static_cast<uint8_t>(rows), 8, 1, 1,
                  Order_t::ONLY_INDEX);
        } else {
            vcmax(valuePtr, pairsPtr, static_cast<uint8_t>(rows), 1, 1, 1, Order_t::ONLY_VALUE);
            vcmax(reinterpret_cast<__ubuf__ half*>(winnerPtr), pairsPtr, static_cast<uint8_t>(rows), 8, 1, 1,
                  Order_t::ONLY_INDEX);
        }
        pipe_barrier(PIPE_V);

        set_mask_norm();
        set_vector_mask(0, 1u);
        if (rows >= 8u) {
            // The winner for each row is one half-word in the row-major pair tile.  Build one
            // contiguous byte-offset vector and gather all rows with one instruction.  The old
            // row loop emitted one vgather per row (up to 31); the extra address arithmetic is
            // fixed and is amortized once the tile has eight rows.
            this->RawIota(offsets, static_cast<int32_t>(0), rows);
            this->RawMuls(offsets, offsets, static_cast<int32_t>(chunks * 2u * sizeof(half)), rows);
            this->RawMuls(localIndex, winner, static_cast<int32_t>(2u * sizeof(half)), rows);
            this->RawAdd(offsets, offsets, localIndex, rows);
            this->RawAdds(offsets, offsets, static_cast<int32_t>(sizeof(half)), rows);
            set_mask_count();
            set_vector_mask(0, 1u);
            vgather(reinterpret_cast<__ubuf__ uint16_t*>(localPtr), reinterpret_cast<__ubuf__ uint32_t*>(offsetPtr),
                    static_cast<uint32_t>(reinterpret_cast<uint64_t>(pairsPtr)), 1, static_cast<uint8_t>(rows));
            this->RawMuls(winner, winner, static_cast<int32_t>(chunk / 2u), rows);
        } else {
            vector_dup(localPtr, static_cast<int32_t>(0), static_cast<uint8_t>(rows), 1, 1, 1, 0);
            vmuls(offsetPtr, winnerPtr, static_cast<int32_t>(sizeof(half)), static_cast<uint8_t>(rows), 1, 1, 1, 1);
            vmuls(winnerPtr, winnerPtr, static_cast<int32_t>(chunk / 2u), static_cast<uint8_t>(rows), 1, 1, 1, 1);
            pipe_barrier(PIPE_V);
            for (uint32_t row = 0; row < rows; ++row) {
                vgather(reinterpret_cast<__ubuf__ uint16_t*>(localPtr + row * 8u),
                        reinterpret_cast<__ubuf__ uint32_t*>(offsetPtr + row * 8u),
                        static_cast<uint32_t>(reinterpret_cast<uint64_t>(pairsPtr + row * chunks * 2u)) + sizeof(half),
                        8, 1);
            }
        }
        pipe_barrier(PIPE_V);
        vadd(indexPtr, winnerPtr, localPtr, static_cast<uint8_t>(rows), 1, 1, 1, 1, 1, 1);
        this->template Sync<HardEvent::V_MTE3>();
        copy_ubuf_to_gm_align_b16(this->valuesGm_, valuePtr, 0, 1, rows * sizeof(half), 0, 0, 0, 0);
        copy_ubuf_to_gm_align_b32(this->indiceGm_, indexPtr, 0, static_cast<uint16_t>(rows), sizeof(int32_t), 0, 0, 0,
                                  0);
    }

    __aicore__ inline void Store(const LocalTensor<T>& value, const LocalTensor<int32_t>& index, uint32_t count)
    {
        this->template Sync<HardEvent::V_MTE3>();
        if constexpr (sizeof(T) == 2u) {
            copy_ubuf_to_gm_align_b16(this->valuesGm_, reinterpret_cast<__ubuf__ T*>(value.GetPhyAddr()), 0, 1,
                                      count * sizeof(T), 0, 0, 0, 0);
        } else {
            copy_ubuf_to_gm_align_b32(this->valuesGm_, reinterpret_cast<__ubuf__ T*>(value.GetPhyAddr()), 0, 1,
                                      count * sizeof(T), 0, 0, 0, 0);
        }
        copy_ubuf_to_gm_align_b32(this->indiceGm_, reinterpret_cast<__ubuf__ int32_t*>(index.GetPhyAddr()), 0, 1,
                                  count * sizeof(int32_t), 0, 0, 0, 0);
    }
};
} // namespace ArgWithValueNs
#endif // ARG_MAX_WITH_VALUE_SMALL_LAST_H
