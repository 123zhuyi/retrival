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
 * \file ragged_bin_count_simt.h
 * \brief SIMT implementation for RaggedBinCount on DAV_3510.
 */
#ifndef RAGGED_BIN_COUNT_SIMT_H
#define RAGGED_BIN_COUNT_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "ragged_bin_count_precision_policy.h"
#include "ragged_bin_count_tiling_data.h"

namespace NsRaggedBinCount {

using namespace AscendC;

constexpr uint32_t THREAD_NUM_U32 = 1024U;
constexpr uint32_t THREAD_NUM_U64 = 512U;
constexpr uint32_t MAPPING_MODE_ROW = 0U;
constexpr uint32_t MAPPING_MODE_VALUE = 1U;
constexpr uint32_t UB_BLOCK_BYTES = 32U;
// The compensated ROW path scans a short row once per bin. Keep the total
// comparison work bounded so long or wide histograms retain the original
// value-parallel scatter schedule.
constexpr int64_t PRECISE_ROW_MAX_VALUES = 256;
constexpr int64_t PRECISE_ROW_MAX_BINS = 64;
constexpr int64_t PRECISE_ROW_MAX_WORK = 4096;
// VALUE mapping is selected for a few long, highly contended rows. When the
// histogram has only one or two bins, thousands of threads otherwise race on
// the same FP32 atomic and the last bit can vary between launches. The shared
// precision-policy header bounds a deterministic multi-core reduction to this
// subdomain; wider VALUE histograms retain the original scatter schedule.

template <bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterWrite(__gm__ float* output, int64_t outputOffset, uint64_t valueIndex,
                                         __gm__ float* weights)
{
    if constexpr (BINARY_OUTPUT) {
        // The bin only has to be set once and every writer stores the same
        // value, so reading first is safe and skips most of the atomics: a
        // stale "not set" costs one redundant exchange, and "already set" can
        // only be observed after some thread wrote 1.0F.  Output stays bitwise
        // in {0.0F, 1.0F}.
        //
        // Four variants were measured on the 74 binary cases of the performance
        // set (median NPU time, and how many fall below folded G/N 0.1):
        //
        //   asc_atomic_exch            correct   424 us   51/74 below   <- was here
        //   read + asc_atomic_exch     correct   1.27x faster, 47/74    <- is here
        //   plain store                WRONG     75x faster,   0/32
        //   asc_atomic_or(0x3F800000)  correct   0.6x, i.e. SLOWER, 56/74
        //
        // The plain store is not merely risky, it is measurably wrong: cores
        // write back whole cache lines, so a 16-row x 1-bin output (64 B, 16
        // cores) kept only one core's writes -- 6.25% correct.  The corruption
        // fraction tracks output size exactly, which is the false-sharing
        // signature, and it is why the atomic cannot simply be dropped.
        //
        // So every GM atomic serialises under contention and swapping which
        // atomic is used does not help; the read above is the only cheap win
        // available without restructuring.  ScatterWriteLocal below is the
        // restructuring: when the output fits in UB the host switches the
        // scatter over to it and this path only runs for outputs too large to
        // privatise, where the hits are spread thin enough not to serialise.
        if (output[outputOffset] != 1.0F) {
            (void)asc_atomic_exch(output + outputOffset, 1.0F);
        }
    } else if constexpr (HAS_WEIGHTS) {
        (void)asc_atomic_add(output + outputOffset, weights[valueIndex]);
    } else {
        (void)asc_atomic_add(output + outputOffset, 1.0F);
    }
}

template <bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterWriteLocal(__ubuf__ float* histogram, int64_t outputOffset, uint64_t valueIndex,
                                              __gm__ float* weights)
{
    // Same three cases as ScatterWrite, but against the core's private UB copy of the output. The
    // atomics stay -- the 1024 threads of one core still collide on a bin -- yet they now resolve in
    // core-local SRAM instead of crossing to GM, and no cache line is shared with another core.
    if constexpr (BINARY_OUTPUT) {
        if (histogram[outputOffset] != 1.0F) {
            (void)asc_atomic_exch(histogram + outputOffset, 1.0F);
        }
    } else if constexpr (HAS_WEIGHTS) {
        (void)asc_atomic_add(histogram + outputOffset, weights[valueIndex]);
    } else {
        (void)asc_atomic_add(histogram + outputOffset, 1.0F);
    }
}

template <bool PRIVATE, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterDispatch(__gm__ float* output, __ubuf__ float* histogram, int64_t outputOffset,
                                            uint64_t valueIndex, __gm__ float* weights)
{
    if constexpr (PRIVATE) {
        ScatterWriteLocal<BINARY_OUTPUT, HAS_WEIGHTS>(histogram, outputOffset, valueIndex, weights);
    } else {
        ScatterWrite<BINARY_OUTPUT, HAS_WEIGHTS>(output, outputOffset, valueIndex, weights);
    }
}

// Knuth TwoSum: sum is the rounded high part and error recovers the low part
// discarded by that addition. A non-finite high part is already the required
// IEEE result; only a NaN compensation is cleared so it cannot poison a later
// finite add, matching the established RegBase reduction convention.
__simt_callee__ inline void CompensatedTwoSum(float lhs, float rhs, float& sum, float& error)
{
    sum = lhs + rhs;
    const float rhsVirtual = sum - lhs;
    error = (lhs - (sum - rhsVirtual)) + (rhs - rhsVirtual);
    if (error != error) {
        error = 0.0F;
    }
}

__simt_callee__ inline bool IsFiniteFloat(float value)
{
    union FloatBits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return (bits.u & 0x7F800000U) != 0x7F800000U;
}

__simt_callee__ inline bool RequiresExactAddition(float lhs, float rhs)
{
    union FloatBits {
        float f;
        uint32_t u;
    } lhsBits, rhsBits;
    lhsBits.f = lhs;
    rhsBits.f = rhs;
    const uint32_t lhsAbs = lhsBits.u & 0x7FFFFFFFU;
    const uint32_t rhsAbs = rhsBits.u & 0x7FFFFFFFU;
    const uint32_t lhsExponent = lhsAbs >> 23U;
    const uint32_t rhsExponent = rhsAbs >> 23U;

    // The exact accumulator consumes IEEE-754 fields directly. Route
    // non-finite values and non-zero subnormal inputs through it so this path
    // does not depend on the SIMT arithmetic mode's NaN or denormal handling.
    //
    // A normal in one of the lowest 23 raw exponent bands has a subnormal ULP.
    // Route it before arithmetic as well: even when paired with a much larger
    // value, TwoSum's discarded residual can be subnormal and FTZ can erase it
    // before a post-addition check observes it. Raw exponent 24 is the first
    // band whose ULP is the minimum normal; its exp-23 boundary neighbour is
    // already selected by the other operand.
    const bool lhsRequiresExact = lhsExponent == 0xFFU || (lhsExponent == 0U && lhsAbs != 0U) ||
                                  (lhsExponent > 0U && lhsExponent <= 23U);
    const bool rhsRequiresExact = rhsExponent == 0xFFU || (rhsExponent == 0U && rhsAbs != 0U) ||
                                  (rhsExponent > 0U && rhsExponent <= 23U);
    return lhsRequiresExact || rhsRequiresExact;
}

__simt_callee__ inline bool CompensatedAccumulate(float value, float& high, float& low)
{
    float sum = 0.0F;
    float error = 0.0F;
    bool requiresExact = RequiresExactAddition(high, value);
    CompensatedTwoSum(high, value, sum, error);
    requiresExact = requiresExact || !IsFiniteFloat(sum);

    // The high/low expansion can represent two FP32 components exactly. If
    // adding the new error to the old low part needs a third component, the
    // final high + low may round on the wrong side of a halfway point. Detect
    // that residual and let the caller rescan with the exact accumulator.
    float correction = 0.0F;
    float correctionResidual = 0.0F;
    requiresExact = RequiresExactAddition(low, error) || requiresExact;
    CompensatedTwoSum(low, error, correction, correctionResidual);
    requiresExact = requiresExact || !IsFiniteFloat(correction) || correctionResidual != 0.0F;
    requiresExact = RequiresExactAddition(sum, correction) || requiresExact;
    CompensatedTwoSum(sum, correction, high, low);
    requiresExact = requiresExact || !IsFiniteFloat(high);
    return requiresExact;
}

// A finite FP32 value is an integer multiple of 2^-149. The largest finite
// value needs bit 276 in that unit and summing at most 262144 values needs at
// most bit 294, so five 64-bit limbs hold the exact signed sum without overflow.
// This slow path is entered when the lightweight TwoSum path cannot guarantee
// mode-independent correct rounding: non-finite/subnormal/low-band inputs,
// finite intermediate overflow, or a third component needed to retain the
// exact low-order residual.
struct ExactFloatAccumulator {
    uint64_t limb0;
    uint64_t limb1;
    uint64_t limb2;
    uint64_t limb3;
    uint64_t limb4;
    bool hasNan;
    bool hasPositiveInf;
    bool hasNegativeInf;
};

__simt_callee__ inline uint64_t AddWithCarry(uint64_t& target, uint64_t value)
{
    const uint64_t previous = target;
    target += value;
    return target < previous ? 1ULL : 0ULL;
}

__simt_callee__ inline uint64_t SubWithBorrow(uint64_t& target, uint64_t value)
{
    const uint64_t previous = target;
    target -= value;
    return previous < value ? 1ULL : 0ULL;
}

__simt_callee__ inline void AddAccumulatorWord(ExactFloatAccumulator& accumulator, uint32_t wordIndex, uint64_t value)
{
    uint64_t carry = 0ULL;
    if (wordIndex == 0U) {
        carry = AddWithCarry(accumulator.limb0, value);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb1, carry);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb2, carry);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb3, carry);
        if (carry != 0ULL) {
            (void)AddWithCarry(accumulator.limb4, carry);
        }
    } else if (wordIndex == 1U) {
        carry = AddWithCarry(accumulator.limb1, value);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb2, carry);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb3, carry);
        if (carry != 0ULL) {
            (void)AddWithCarry(accumulator.limb4, carry);
        }
    } else if (wordIndex == 2U) {
        carry = AddWithCarry(accumulator.limb2, value);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb3, carry);
        if (carry != 0ULL) {
            (void)AddWithCarry(accumulator.limb4, carry);
        }
    } else if (wordIndex == 3U) {
        carry = AddWithCarry(accumulator.limb3, value);
        if (carry != 0ULL) {
            (void)AddWithCarry(accumulator.limb4, carry);
        }
    } else {
        (void)AddWithCarry(accumulator.limb4, value);
    }
}

__simt_callee__ inline void SubAccumulatorWord(ExactFloatAccumulator& accumulator, uint32_t wordIndex, uint64_t value)
{
    uint64_t borrow = 0ULL;
    if (wordIndex == 0U) {
        borrow = SubWithBorrow(accumulator.limb0, value);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb1, borrow);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb2, borrow);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb3, borrow);
        if (borrow != 0ULL) {
            (void)SubWithBorrow(accumulator.limb4, borrow);
        }
    } else if (wordIndex == 1U) {
        borrow = SubWithBorrow(accumulator.limb1, value);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb2, borrow);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb3, borrow);
        if (borrow != 0ULL) {
            (void)SubWithBorrow(accumulator.limb4, borrow);
        }
    } else if (wordIndex == 2U) {
        borrow = SubWithBorrow(accumulator.limb2, value);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb3, borrow);
        if (borrow != 0ULL) {
            (void)SubWithBorrow(accumulator.limb4, borrow);
        }
    } else if (wordIndex == 3U) {
        borrow = SubWithBorrow(accumulator.limb3, value);
        if (borrow != 0ULL) {
            (void)SubWithBorrow(accumulator.limb4, borrow);
        }
    } else {
        (void)SubWithBorrow(accumulator.limb4, value);
    }
}

__simt_callee__ inline void AccumulateExactFloat(float value, ExactFloatAccumulator& accumulator)
{
    union FloatBits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    const uint32_t sign = bits.u >> 31U;
    const uint32_t exponent = (bits.u >> 23U) & 0xFFU;
    uint32_t mantissa = bits.u & 0x007FFFFFU;
    if (exponent == 0xFFU) {
        if (mantissa != 0U) {
            accumulator.hasNan = true;
        } else if (sign == 0U) {
            accumulator.hasPositiveInf = true;
        } else {
            accumulator.hasNegativeInf = true;
        }
        return;
    }
    if (exponent == 0U) {
        if (mantissa == 0U) {
            return;
        }
    } else {
        mantissa |= 0x00800000U;
    }

    const uint32_t shift = exponent == 0U ? 0U : exponent - 1U;
    const uint32_t wordIndex = shift >> 6U;
    const uint32_t bitOffset = shift & 63U;
    const uint64_t lowPart = static_cast<uint64_t>(mantissa) << bitOffset;
    const uint64_t highPart = bitOffset == 0U ? 0ULL : static_cast<uint64_t>(mantissa) >> (64U - bitOffset);
    if (sign == 0U) {
        AddAccumulatorWord(accumulator, wordIndex, lowPart);
        if (highPart != 0ULL) {
            AddAccumulatorWord(accumulator, wordIndex + 1U, highPart);
        }
    } else {
        SubAccumulatorWord(accumulator, wordIndex, lowPart);
        if (highPart != 0ULL) {
            SubAccumulatorWord(accumulator, wordIndex + 1U, highPart);
        }
    }
}

__simt_callee__ inline uint64_t GetAccumulatorWord(const ExactFloatAccumulator& accumulator, uint32_t wordIndex)
{
    if (wordIndex == 0U) {
        return accumulator.limb0;
    }
    if (wordIndex == 1U) {
        return accumulator.limb1;
    }
    if (wordIndex == 2U) {
        return accumulator.limb2;
    }
    if (wordIndex == 3U) {
        return accumulator.limb3;
    }
    return wordIndex == 4U ? accumulator.limb4 : 0ULL;
}

__simt_callee__ inline void MakeAccumulatorMagnitude(ExactFloatAccumulator& accumulator)
{
    accumulator.limb0 = ~accumulator.limb0;
    accumulator.limb1 = ~accumulator.limb1;
    accumulator.limb2 = ~accumulator.limb2;
    accumulator.limb3 = ~accumulator.limb3;
    accumulator.limb4 = ~accumulator.limb4;
    AddAccumulatorWord(accumulator, 0U, 1ULL);
}

__simt_callee__ inline int32_t HighestBitInWord(uint64_t value)
{
    int32_t bit = 0;
    if (value >= (1ULL << 32U)) {
        value >>= 32U;
        bit += 32;
    }
    if (value >= (1ULL << 16U)) {
        value >>= 16U;
        bit += 16;
    }
    if (value >= (1ULL << 8U)) {
        value >>= 8U;
        bit += 8;
    }
    if (value >= (1ULL << 4U)) {
        value >>= 4U;
        bit += 4;
    }
    if (value >= (1ULL << 2U)) {
        value >>= 2U;
        bit += 2;
    }
    if (value >= (1ULL << 1U)) {
        ++bit;
    }
    return bit;
}

__simt_callee__ inline int32_t HighestAccumulatorBit(const ExactFloatAccumulator& accumulator)
{
    if (accumulator.limb4 != 0ULL) {
        return 256 + HighestBitInWord(accumulator.limb4);
    }
    if (accumulator.limb3 != 0ULL) {
        return 192 + HighestBitInWord(accumulator.limb3);
    }
    if (accumulator.limb2 != 0ULL) {
        return 128 + HighestBitInWord(accumulator.limb2);
    }
    if (accumulator.limb1 != 0ULL) {
        return 64 + HighestBitInWord(accumulator.limb1);
    }
    if (accumulator.limb0 != 0ULL) {
        return HighestBitInWord(accumulator.limb0);
    }
    return -1;
}

__simt_callee__ inline uint64_t ShiftAccumulatorRight(const ExactFloatAccumulator& accumulator, uint32_t shift)
{
    const uint32_t wordIndex = shift >> 6U;
    const uint32_t bitOffset = shift & 63U;
    uint64_t value = GetAccumulatorWord(accumulator, wordIndex) >> bitOffset;
    if (bitOffset != 0U) {
        value |= GetAccumulatorWord(accumulator, wordIndex + 1U) << (64U - bitOffset);
    }
    return value;
}

__simt_callee__ inline bool AnyAccumulatorBitsBelow(const ExactFloatAccumulator& accumulator, uint32_t bitCount)
{
    const uint32_t completeWords = bitCount >> 6U;
    for (uint32_t index = 0U; index < completeWords; ++index) {
        if (GetAccumulatorWord(accumulator, index) != 0ULL) {
            return true;
        }
    }
    const uint32_t remainingBits = bitCount & 63U;
    if (remainingBits == 0U) {
        return false;
    }
    const uint64_t mask = (1ULL << remainingBits) - 1ULL;
    return (GetAccumulatorWord(accumulator, completeWords) & mask) != 0ULL;
}

__simt_callee__ inline float FinalizeExactFloat(ExactFloatAccumulator accumulator)
{
    union FloatBits {
        float f;
        uint32_t u;
    } result;
    if (accumulator.hasNan || (accumulator.hasPositiveInf && accumulator.hasNegativeInf)) {
        result.u = 0x7FC00000U;
        return result.f;
    }
    if (accumulator.hasPositiveInf) {
        result.u = 0x7F800000U;
        return result.f;
    }
    if (accumulator.hasNegativeInf) {
        result.u = 0xFF800000U;
        return result.f;
    }

    const bool negative = (accumulator.limb4 >> 63U) != 0ULL;
    if (negative) {
        MakeAccumulatorMagnitude(accumulator);
    }
    int32_t highestBit = HighestAccumulatorBit(accumulator);
    if (highestBit < 0) {
        result.u = 0U;
        return result.f;
    }
    if (highestBit <= 22) {
        result.u = (negative ? 0x80000000U : 0U) | static_cast<uint32_t>(accumulator.limb0);
        return result.f;
    }

    const uint32_t rightShift = static_cast<uint32_t>(highestBit - 23);
    uint64_t significand = ShiftAccumulatorRight(accumulator, rightShift) & 0x00FFFFFFULL;
    if (rightShift != 0U) {
        const bool roundBit = ((GetAccumulatorWord(accumulator, (rightShift - 1U) >> 6U) >> ((rightShift - 1U) & 63U)) &
                               1ULL) != 0ULL;
        const bool sticky = AnyAccumulatorBitsBelow(accumulator, rightShift - 1U);
        if (roundBit && (sticky || ((significand & 1ULL) != 0ULL))) {
            ++significand;
            if (significand == 0x01000000ULL) {
                significand >>= 1U;
                ++highestBit;
            }
        }
    }

    const int32_t exponent = highestBit - 22;
    if (exponent >= 255) {
        result.u = negative ? 0xFF800000U : 0x7F800000U;
        return result.f;
    }
    result.u = (negative ? 0x80000000U : 0U) | (static_cast<uint32_t>(exponent) << 23U) |
               (static_cast<uint32_t>(significand) & 0x007FFFFFU);
    return result.f;
}

template <bool PRIVATE>
__simt_callee__ inline void WritePreciseRowResult(__gm__ float* output, __ubuf__ float* histogram, int64_t outputOffset,
                                                  float value)
{
    if constexpr (PRIVATE) {
        // The precise paths give one core exclusive final ownership of each
        // output element; all other private histograms retain +0.0 there.
        histogram[outputOffset] = value;
    } else {
        // Keep the GM write atomic: different rows owned by different cores can
        // share a cache line even though no two threads own the same element.
        (void)asc_atomic_add(output + outputOffset, value);
    }
}

constexpr uint32_t PRECISE_PARTIAL_HAS_HIT = 1U;
constexpr uint32_t PRECISE_PARTIAL_REQUIRES_EXACT = 2U;

struct PreciseValuePartial {
    float high;
    float low;
    uint32_t flags;
    uint32_t reserved;
};

static_assert(sizeof(PreciseValuePartial) == PRECISE_VALUE_PARTIAL_BYTES,
              "host and kernel disagree on the precise VALUE partial size");

template <typename VALUE_TYPE, typename INDEX_TYPE>
__simt_callee__ inline uint32_t AccumulatePreciseRange(INDEX_TYPE rangeBegin, INDEX_TYPE rangeEnd, int64_t bin,
                                                       __gm__ VALUE_TYPE* values, __gm__ float* weights, float& high,
                                                       float& low)
{
    bool hasHit = false;
    bool requiresExact = false;
    high = 0.0F;
    low = 0.0F;
    for (INDEX_TYPE valueIndex = rangeBegin; valueIndex < rangeEnd; ++valueIndex) {
        if (static_cast<int64_t>(values[valueIndex]) == bin) {
            requiresExact = CompensatedAccumulate(weights[valueIndex], high, low) || requiresExact;
            hasHit = true;
        }
    }
    return (hasHit ? PRECISE_PARTIAL_HAS_HIT : 0U) | (requiresExact ? PRECISE_PARTIAL_REQUIRES_EXACT : 0U);
}

template <typename VALUE_TYPE, typename INDEX_TYPE>
__simt_callee__ inline float AccumulateExactBin(INDEX_TYPE rowBegin, INDEX_TYPE rowEnd, int64_t bin,
                                                __gm__ VALUE_TYPE* values, __gm__ float* weights)
{
    ExactFloatAccumulator accumulator{};
    for (INDEX_TYPE valueIndex = rowBegin; valueIndex < rowEnd; ++valueIndex) {
        if (static_cast<int64_t>(values[valueIndex]) == bin) {
            AccumulateExactFloat(weights[valueIndex], accumulator);
        }
    }
    return FinalizeExactFloat(accumulator);
}

template <typename VALUE_TYPE, typename INDEX_TYPE>
__simt_callee__ inline bool AccumulatePreciseBin(INDEX_TYPE rowBegin, INDEX_TYPE rowEnd, int64_t bin,
                                                 __gm__ VALUE_TYPE* values, __gm__ float* weights, float& rowResult)
{
    float high = 0.0F;
    float low = 0.0F;
    const uint32_t flags = AccumulatePreciseRange<VALUE_TYPE, INDEX_TYPE>(rowBegin, rowEnd, bin, values, weights, high,
                                                                          low);
    if ((flags & PRECISE_PARTIAL_HAS_HIT) == 0U) {
        return false;
    }
    rowResult = (flags & PRECISE_PARTIAL_REQUIRES_EXACT) != 0U ?
                    AccumulateExactBin<VALUE_TYPE, INDEX_TYPE>(rowBegin, rowEnd, bin, values, weights) :
                    high + low;
    return true;
}

__simt_callee__ inline bool UsePreciseRowPath(int64_t rowLength, int64_t numBins)
{
    return rowLength > 0 && rowLength <= PRECISE_ROW_MAX_VALUES && numBins > 0 && numBins <= PRECISE_ROW_MAX_BINS &&
           rowLength * numBins <= PRECISE_ROW_MAX_WORK;
}

__simt_callee__ inline int64_t FindRaggedRow(__gm__ int64_t* splits, int64_t numRows, int64_t valueIndex)
{
    // Find the first row whose end split is greater than valueIndex. Using
    // upper-bound semantics correctly skips empty rows represented by repeated splits.
    int64_t low = 0;
    int64_t high = numRows;
    while (low < high) {
        const int64_t middle = low + ((high - low) >> 1);
        if (splits[middle + 1] <= valueIndex) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low >= numRows) {
        return -1;
    }
    const int64_t rowBegin = splits[low];
    const int64_t rowEnd = splits[low + 1];
    return (rowBegin <= valueIndex && valueIndex < rowEnd) ? low : -1;
}

template <typename INDEX_TYPE, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void InitializeOutputAndFlag(
    uint32_t coreIdx, uint32_t coreNum, int64_t outputElements, __gm__ float* output, __gm__ uint32_t* invalidFlag)
{
    const INDEX_TYPE first = static_cast<INDEX_TYPE>(coreIdx) * static_cast<INDEX_TYPE>(blockDim.x) +
                             static_cast<INDEX_TYPE>(threadIdx.x);
    const INDEX_TYPE stride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(coreNum);
    if (coreIdx == 0U && threadIdx.x == 0U) {
        *invalidFlag = 0U;
    }
    for (INDEX_TYPE index = first; index < static_cast<INDEX_TYPE>(outputElements); index += stride) {
        output[index] = 0.0F;
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ValidateInputs(uint32_t coreIdx, uint32_t coreNum,
                                                                                int64_t numSplits, int64_t numValues,
                                                                                __gm__ int64_t* splits,
                                                                                __gm__ VALUE_TYPE* values,
                                                                                __gm__ uint32_t* invalidFlag)
{
    const INDEX_TYPE first = static_cast<INDEX_TYPE>(coreIdx) * static_cast<INDEX_TYPE>(blockDim.x) +
                             static_cast<INDEX_TYPE>(threadIdx.x);
    const INDEX_TYPE stride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(coreNum);

    for (INDEX_TYPE splitIndex = first; splitIndex < static_cast<INDEX_TYPE>(numSplits); splitIndex += stride) {
        const int64_t split = splits[splitIndex];
        bool invalid = split < 0 || split > numValues;
        invalid = invalid || (splitIndex == 0U && split != 0);
        invalid = invalid || (splitIndex + 1U == static_cast<INDEX_TYPE>(numSplits) && split != numValues);
        if (splitIndex + 1U < static_cast<INDEX_TYPE>(numSplits)) {
            invalid = invalid || split > splits[splitIndex + 1U];
        }
        if (invalid) {
            (void)asc_atomic_or(invalidFlag, 1U);
        }
    }

    for (INDEX_TYPE valueIndex = first; valueIndex < static_cast<INDEX_TYPE>(numValues); valueIndex += stride) {
        if (values[valueIndex] < static_cast<VALUE_TYPE>(0)) {
            (void)asc_atomic_or(invalidFlag, 1U);
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, bool PRIVATE, bool BINARY_OUTPUT,
          bool HAS_WEIGHTS>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ScatterByRow(
    uint32_t coreIdx, uint32_t coreNum, int64_t numRows, int64_t numValues, int64_t numBins, __gm__ int64_t* splits,
    __gm__ VALUE_TYPE* values, __gm__ float* weights, __gm__ float* output, __ubuf__ float* histogram)
{
    if constexpr (HAS_WEIGHTS && !BINARY_OUTPUT) {
        // Flatten (row, bin) across the whole launch. The former row-first
        // mapping used only numBins threads while each core serially visited
        // its rows; for a 64x32 output that left 992 of 1024 threads idle.
        // Each flattened task still owns exactly one output element and scans
        // that row in fixed input order, so the precision contract is
        // unchanged while all cores and threads can participate.
        const INDEX_TYPE firstOutput = static_cast<INDEX_TYPE>(coreIdx) * static_cast<INDEX_TYPE>(blockDim.x) +
                                       static_cast<INDEX_TYPE>(threadIdx.x);
        const INDEX_TYPE outputStride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(coreNum);
        const INDEX_TYPE outputElements = static_cast<INDEX_TYPE>(numRows * numBins);
        for (INDEX_TYPE outputIndex = firstOutput; outputIndex < outputElements; outputIndex += outputStride) {
            const INDEX_TYPE row = outputIndex / static_cast<INDEX_TYPE>(numBins);
            const int64_t bin = static_cast<int64_t>(outputIndex % static_cast<INDEX_TYPE>(numBins));
            const int64_t rowBegin = splits[row];
            const int64_t rowEnd = splits[row + 1U];
            if (rowBegin < 0 || rowEnd < rowBegin || rowEnd > numValues ||
                !UsePreciseRowPath(rowEnd - rowBegin, numBins)) {
                continue;
            }
            float rowResult = 0.0F;
            if (AccumulatePreciseBin<VALUE_TYPE, INDEX_TYPE>(static_cast<INDEX_TYPE>(rowBegin),
                                                             static_cast<INDEX_TYPE>(rowEnd), bin, values, weights,
                                                             rowResult)) {
                WritePreciseRowResult<PRIVATE>(output, histogram, static_cast<int64_t>(outputIndex), rowResult);
            }
        }
    }

    for (INDEX_TYPE row = static_cast<INDEX_TYPE>(coreIdx); row < static_cast<INDEX_TYPE>(numRows);
         row += static_cast<INDEX_TYPE>(coreNum)) {
        const int64_t rowBegin = splits[row];
        const int64_t rowEnd = splits[row + 1];
        if (rowBegin < 0 || rowEnd < rowBegin || rowEnd > numValues) {
            continue;
        }
        const int64_t rowLength = rowEnd - rowBegin;
        if constexpr (HAS_WEIGHTS && !BINARY_OUTPUT) {
            if (UsePreciseRowPath(rowLength, numBins)) {
                continue;
            }
        }
        const INDEX_TYPE first = static_cast<INDEX_TYPE>(rowBegin) + static_cast<INDEX_TYPE>(threadIdx.x);
        for (INDEX_TYPE valueIndex = first; valueIndex < static_cast<INDEX_TYPE>(rowEnd);
             valueIndex += static_cast<INDEX_TYPE>(blockDim.x)) {
            const int64_t bin = static_cast<int64_t>(values[valueIndex]);
            if (bin >= 0 && bin < numBins) {
                const int64_t outputOffset = static_cast<int64_t>(row) * numBins + bin;
                ScatterDispatch<PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>(output, histogram, outputOffset,
                                                                     static_cast<uint64_t>(valueIndex), weights);
            }
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void BuildPreciseValuePartials(
    uint32_t coreIdx, uint32_t coreNum, int64_t outputElements, int64_t numBins, __gm__ int64_t* splits,
    __gm__ VALUE_TYPE* values, __gm__ float* weights, __gm__ PreciseValuePartial* partials)
{
    const int64_t logicalTasks = outputElements * PRECISE_VALUE_PARTITIONS_PER_CORE;
    const int64_t totalPartitions = static_cast<int64_t>(coreNum) * PRECISE_VALUE_PARTITIONS_PER_CORE;
    for (int64_t logicalTask = static_cast<int64_t>(threadIdx.x); logicalTask < logicalTasks;
         logicalTask += static_cast<int64_t>(blockDim.x)) {
        const int64_t outputIndex = logicalTask / PRECISE_VALUE_PARTITIONS_PER_CORE;
        const int64_t localPartition = logicalTask - outputIndex * PRECISE_VALUE_PARTITIONS_PER_CORE;
        const int64_t globalPartition = static_cast<int64_t>(coreIdx) * PRECISE_VALUE_PARTITIONS_PER_CORE +
                                        localPartition;
        const int64_t row = outputIndex / numBins;
        const int64_t bin = outputIndex - row * numBins;
        const int64_t rowBegin = splits[row];
        const int64_t rowLength = splits[row + 1] - rowBegin;
        const int64_t baseLength = rowLength / totalPartitions;
        const int64_t extraPartitions = rowLength - baseLength * totalPartitions;
        const int64_t rangeBegin = rowBegin + globalPartition * baseLength +
                                   (globalPartition < extraPartitions ? globalPartition : extraPartitions);
        const int64_t rangeLength = baseLength + (globalPartition < extraPartitions ? 1 : 0);
        float high = 0.0F;
        float low = 0.0F;
        const uint32_t flags = AccumulatePreciseRange<VALUE_TYPE, INDEX_TYPE>(
            static_cast<INDEX_TYPE>(rangeBegin), static_cast<INDEX_TYPE>(rangeBegin + rangeLength), bin, values,
            weights, high, low);
        const int64_t slotIndex = (static_cast<int64_t>(coreIdx) * outputElements + outputIndex) *
                                      PRECISE_VALUE_PARTITIONS_PER_CORE +
                                  localPartition;
        // SIMT scalar GM stores from different cores can evict whole cache
        // lines over one another. Atomic exchange is uncontended here (every
        // slot has one producer) and makes each field globally visible before
        // the cross-core barrier without a separate DCache clean protocol.
        (void)asc_atomic_exch(&partials[slotIndex].high, high);
        (void)asc_atomic_exch(&partials[slotIndex].low, low);
        (void)asc_atomic_exch(&partials[slotIndex].flags, flags);
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void FinalizePreciseValuePartials(
    uint32_t coreIdx, uint32_t coreNum, int64_t outputElements, int64_t numBins, __gm__ int64_t* splits,
    __gm__ VALUE_TYPE* values, __gm__ float* weights, __gm__ PreciseValuePartial* partials, __gm__ float* output)
{
    if (coreIdx != 0U) {
        return;
    }
    for (int64_t outputIndex = static_cast<int64_t>(threadIdx.x); outputIndex < outputElements;
         outputIndex += static_cast<int64_t>(blockDim.x)) {
        float high = 0.0F;
        float low = 0.0F;
        bool hasHit = false;
        bool requiresExact = false;
        for (int64_t core = 0; core < static_cast<int64_t>(coreNum); ++core) {
            for (int64_t partition = 0; partition < PRECISE_VALUE_PARTITIONS_PER_CORE; ++partition) {
                const int64_t slotIndex = (core * outputElements + outputIndex) * PRECISE_VALUE_PARTITIONS_PER_CORE +
                                          partition;
                const uint32_t partialFlags = partials[slotIndex].flags;
                if ((partialFlags & PRECISE_PARTIAL_HAS_HIT) == 0U) {
                    continue;
                }
                const float partialHigh = partials[slotIndex].high;
                const float partialLow = partials[slotIndex].low;
                hasHit = true;
                requiresExact = requiresExact || ((partialFlags & PRECISE_PARTIAL_REQUIRES_EXACT) != 0U);
                requiresExact = CompensatedAccumulate(partialHigh, high, low) || requiresExact;
                requiresExact = CompensatedAccumulate(partialLow, high, low) || requiresExact;
            }
        }
        if (hasHit) {
            const int64_t row = outputIndex / numBins;
            const int64_t bin = outputIndex - row * numBins;
            const float result = requiresExact ? AccumulateExactBin<VALUE_TYPE, INDEX_TYPE>(
                                                     static_cast<INDEX_TYPE>(splits[row]),
                                                     static_cast<INDEX_TYPE>(splits[row + 1]), bin, values, weights) :
                                                 high + low;
            // Only block zero reaches this point, so distinct threads can
            // store their exclusively owned output elements without a GM
            // atomic or the cross-core cache-line race of VALUE scatter.
            (void)asc_atomic_exch(output + outputIndex, result);
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, bool PRIVATE, bool BINARY_OUTPUT,
          bool HAS_WEIGHTS>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void ScatterByValue(
    uint32_t coreIdx, uint32_t coreNum, int64_t numRows, int64_t numValues, int64_t numBins, __gm__ int64_t* splits,
    __gm__ VALUE_TYPE* values, __gm__ float* weights, __gm__ float* output, __ubuf__ float* histogram)
{
    const INDEX_TYPE first = static_cast<INDEX_TYPE>(coreIdx) * static_cast<INDEX_TYPE>(blockDim.x) +
                             static_cast<INDEX_TYPE>(threadIdx.x);
    const INDEX_TYPE stride = static_cast<INDEX_TYPE>(blockDim.x) * static_cast<INDEX_TYPE>(coreNum);
    for (INDEX_TYPE valueIndex = first; valueIndex < static_cast<INDEX_TYPE>(numValues); valueIndex += stride) {
        const int64_t bin = static_cast<int64_t>(values[valueIndex]);
        if (bin < 0 || bin >= numBins) {
            continue;
        }
        const int64_t row = FindRaggedRow(splits, numRows, static_cast<int64_t>(valueIndex));
        if (row >= 0) {
            const int64_t outputOffset = row * numBins + bin;
            ScatterDispatch<PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>(output, histogram, outputOffset,
                                                                 static_cast<uint64_t>(valueIndex), weights);
        }
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, uint32_t MAPPING_MODE, bool PRIVATE,
          bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__aicore__ inline void LaunchScatter(const RaggedBinCountTilingData* tilingData, __gm__ int64_t* splitsGm,
                                     __gm__ VALUE_TYPE* valuesGm, __gm__ float* weightsGm, __gm__ float* outputGm,
                                     __ubuf__ float* histogram, uint32_t coreIdx, uint32_t coreNum)
{
    if constexpr (MAPPING_MODE == MAPPING_MODE_ROW) {
        asc_vf_call<ScatterByRow<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>>(
            dim3(THREAD_NUM), coreIdx, coreNum, tilingData->numRows, tilingData->numValues, tilingData->numBins,
            splitsGm, valuesGm, weightsGm, outputGm, histogram);
    } else {
        asc_vf_call<ScatterByValue<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>>(
            dim3(THREAD_NUM), coreIdx, coreNum, tilingData->numRows, tilingData->numValues, tilingData->numBins,
            splitsGm, valuesGm, weightsGm, outputGm, histogram);
    }
}

template <typename VALUE_TYPE, typename INDEX_TYPE, uint32_t THREAD_NUM, uint32_t MAPPING_MODE, bool BINARY_OUTPUT,
          bool HAS_WEIGHTS>
__aicore__ inline void ProcessWithIndexType(GM_ADDR splits, GM_ADDR values, GM_ADDR weights, GM_ADDR output,
                                            GM_ADDR userWorkspace, const RaggedBinCountTilingData* tilingData,
                                            TPipe* pipe)
{
    __gm__ int64_t* splitsGm = reinterpret_cast<__gm__ int64_t*>(splits);
    __gm__ VALUE_TYPE* valuesGm = reinterpret_cast<__gm__ VALUE_TYPE*>(values);
    __gm__ float* weightsGm = reinterpret_cast<__gm__ float*>(weights);
    __gm__ float* outputGm = reinterpret_cast<__gm__ float*>(output);
    __gm__ uint32_t* invalidFlag = reinterpret_cast<__gm__ uint32_t*>(userWorkspace);
    __gm__ PreciseValuePartial* preciseValuePartials = reinterpret_cast<__gm__ PreciseValuePartial*>(
        userWorkspace + USER_WORKSPACE_HEADER_BYTES);

    // Read the logical block identity in outer __aicore__ code and pass it into
    // the SIMT VF explicitly: blockIdx/gridDim inside a VF describe the VF grid,
    // not the kernel's outer work partition.
    const uint32_t launchCoreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t launchCoreNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t usedCoreNum = tilingData->usedCoreNum < launchCoreNum ? tilingData->usedCoreNum : launchCoreNum;
    const bool participates = launchCoreIdx < usedCoreNum;

    if (participates) {
        asc_vf_call<InitializeOutputAndFlag<INDEX_TYPE, THREAD_NUM>>(dim3(THREAD_NUM), launchCoreIdx, usedCoreNum,
                                                                     tilingData->outputElements, outputGm, invalidFlag);
    }
    SyncAll();

    if (participates) {
        asc_vf_call<ValidateInputs<VALUE_TYPE, INDEX_TYPE, THREAD_NUM>>(dim3(THREAD_NUM), launchCoreIdx, usedCoreNum,
                                                                        tilingData->numSplits, tilingData->numValues,
                                                                        splitsGm, valuesGm, invalidFlag);
    }
    SyncAll();
    if (*invalidFlag != 0U) {
        return;
    }

    if constexpr (MAPPING_MODE == MAPPING_MODE_VALUE && HAS_WEIGHTS && !BINARY_OUTPUT) {
        if (UsePreciseValuePath(tilingData->numValues, tilingData->numBins)) {
            if (participates) {
                asc_vf_call<BuildPreciseValuePartials<VALUE_TYPE, INDEX_TYPE, THREAD_NUM>>(
                    dim3(THREAD_NUM), launchCoreIdx, usedCoreNum, tilingData->outputElements, tilingData->numBins,
                    splitsGm, valuesGm, weightsGm, preciseValuePartials);
            }
            SyncAll();
            if (participates) {
                asc_vf_call<FinalizePreciseValuePartials<VALUE_TYPE, INDEX_TYPE, THREAD_NUM>>(
                    dim3(THREAD_NUM), launchCoreIdx, usedCoreNum, tilingData->outputElements, tilingData->numBins,
                    splitsGm, valuesGm, weightsGm, preciseValuePartials, outputGm);
            }
            return;
        }
    }

    if (!participates) {
        return;
    }

    // The host privatises whenever the whole output fits in the dynamic UB budget and the extra
    // write-back is cheaper than the global atomics it removes. Both mapping modes qualify: ROW owns
    // each row outright, VALUE has every core touching every row, and a full-output private copy
    // serves both without the kernel having to know which.
    const int32_t privateHistElems = static_cast<int32_t>(tilingData->privateHistElems);
    if (privateHistElems > 0) {
        const uint32_t histogramBytes = static_cast<uint32_t>(privateHistElems) * static_cast<uint32_t>(sizeof(float));
        // Reserve whole 32-byte blocks so Duplicate's vectorised tail cannot run past the allocation;
        // the write-back below still copies exactly histogramBytes, and the host checked the same
        // rounded-up figure against the UB budget.
        const uint32_t histogramBufferBytes = ((histogramBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES) *
                                              UB_BLOCK_BYTES;
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe->InitBuffer(histogramQueue, 1, histogramBufferBytes);
        LocalTensor<float> histogram = histogramQueue.template AllocTensor<float>();
        Duplicate(histogram, 0.0F, privateHistElems);

        LaunchScatter<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, MAPPING_MODE, true, BINARY_OUTPUT, HAS_WEIGHTS>(
            tilingData, splitsGm, valuesGm, weightsGm, outputGm, (__ubuf__ float*)histogram.GetPhyAddr(), launchCoreIdx,
            usedCoreNum);

        // EnQue/DeQue is what orders the SIMT writes into UB against the DMA that reads them back out.
        histogramQueue.EnQue(histogram);
        histogram = histogramQueue.template DeQue<float>();

        // Every core folds its private copy into the already-zeroed output. Counting and weighting use
        // atomic add to implement sum reduction; weighted FP32 results can still depend on the order in
        // which cores merge. The binary path only ever stores 0.0F/1.0F, and atomic max is the OR of
        // those -- adding would yield 2.0F whenever two cores both saw a bin, which VALUE mapping makes
        // routine.
        if constexpr (BINARY_OUTPUT) {
            SetAtomicMax<float>();
        } else {
            SetAtomicAdd<float>();
        }
        GlobalTensor<float> outputGlobal;
        outputGlobal.SetGlobalBuffer(outputGm);
        const DataCopyExtParams copyParams{static_cast<uint16_t>(1), histogramBytes, 0U, 0U, 0U};
        DataCopyPad(outputGlobal, histogram, copyParams);
        SetAtomicNone();

        histogramQueue.FreeTensor(histogram);
    } else {
        LaunchScatter<VALUE_TYPE, INDEX_TYPE, THREAD_NUM, MAPPING_MODE, false, BINARY_OUTPUT, HAS_WEIGHTS>(
            tilingData, splitsGm, valuesGm, weightsGm, outputGm, (__ubuf__ float*)nullptr, launchCoreIdx, usedCoreNum);
    }
}

template <typename VALUE_TYPE, uint32_t MAPPING_MODE, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__aicore__ inline void Process(GM_ADDR splits, GM_ADDR values, GM_ADDR weights, GM_ADDR output, GM_ADDR userWorkspace,
                               const RaggedBinCountTilingData* tilingData, TPipe* pipe)
{
    // Every uint32 index path either steps by `blockDim.x * usedCoreNum` (InitializeOutputAndFlag,
    // ValidateInputs, ScatterByValue) or offsets by `threadIdx.x` off a split boundary (ScatterByRow),
    // so the bound must leave a full stride of headroom below 2^32 rather than running up to the type
    // limit. Without it, a count within one stride of 2^32 makes the accumulator wrap: the loop
    // condition becomes true again at a low index, so ScatterByRow re-attributes the leading values to
    // whichever row sits at the end. INDEX_HEADROOM is that stride's upper bound; anything above the
    // reduced limit falls to the uint64 path, which is correct at any size.
    constexpr uint64_t U32_MAX_VALUE = 0xFFFFFFFFULL;
    constexpr uint64_t MAX_SUPPORTED_CORES = 128ULL;
    constexpr uint64_t INDEX_HEADROOM = static_cast<uint64_t>(THREAD_NUM_U32) * MAX_SUPPORTED_CORES;
    constexpr uint64_t U32_INDEX_LIMIT = U32_MAX_VALUE - INDEX_HEADROOM;
    const bool useUint32Index = static_cast<uint64_t>(tilingData->numSplits) <= U32_INDEX_LIMIT &&
                                static_cast<uint64_t>(tilingData->numValues) <= U32_INDEX_LIMIT &&
                                static_cast<uint64_t>(tilingData->outputElements) <= U32_INDEX_LIMIT;

    if (useUint32Index) {
        ProcessWithIndexType<VALUE_TYPE, uint32_t, THREAD_NUM_U32, MAPPING_MODE, BINARY_OUTPUT, HAS_WEIGHTS>(
            splits, values, weights, output, userWorkspace, tilingData, pipe);
    } else {
        ProcessWithIndexType<VALUE_TYPE, uint64_t, THREAD_NUM_U64, MAPPING_MODE, BINARY_OUTPUT, HAS_WEIGHTS>(
            splits, values, weights, output, userWorkspace, tilingData, pipe);
    }
}

} // namespace NsRaggedBinCount

#endif // RAGGED_BIN_COUNT_SIMT_H
