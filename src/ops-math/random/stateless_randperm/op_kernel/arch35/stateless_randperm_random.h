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
 * \file stateless_randperm_random.h
 * \brief
 */

#ifndef STATELESS_RANDPERM_RANDOM_H
#define STATELESS_RANDPERM_RANDOM_H

#include "kernel_operator.h"

namespace StatelessRandperm {
using namespace AscendC;

constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr uint16_t CONVERT_COUNTER_SIZE = 2;
constexpr uint16_t MAX_DEPTH = 8;
constexpr uint16_t ALG_RGIHT_BIT = 32;
constexpr uint64_t RESERVED_SAMPLES_PER_OUTPUT = 256;
constexpr uint32_t PHILOX_W32_A = 0x9E3779B9;
constexpr uint32_t PHILOX_W32_B = 0xBB67AE85;
constexpr uint32_t PHILOX_M4X32_A = 0xD2511F53;
constexpr uint32_t PHILOX_M4X32_B = 0xCD9E8D57;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr float RANDOM_THREAD_R = 2.0f;
constexpr float RANDOM_THREAD_L = -2.0f;
constexpr uint32_t MANTISSA_BIT = 23;
constexpr uint32_t LOOP_CNT = 10;

constexpr uint32_t RESULT_ELEMENT_CNT = 4;
constexpr uint32_t UNROLL_FACTOR = 2;
constexpr uint32_t MAX_DIM_X = 624;
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t THREAD_LAUNCH = 512;
constexpr uint32_t PHILOX_USED_THREAD = 256;
constexpr uint32_t PHILOX_THREAD_LAUNCH = 256;
constexpr uint32_t DATACOPY_THREAD_LAUNCH = 2048;

__simt_callee__ __aicore__ inline void CopyArray4(uint32_t* dst, const uint32_t* src)
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[IDX_2] = src[IDX_2];
    dst[IDX_3] = src[IDX_3];
}

template <typename T>
__simt_callee__ __aicore__ inline void CopyArray2(T* dst, const T* src)
{
    dst[0] = src[0];
    dst[1] = src[1];
}

__simt_callee__ __aicore__ inline void StateIncr(uint32_t* counter)
{
    if(++counter[0]) return;
    if(++counter[1]) return;
    if(++counter[2]) return;
    ++counter[3];
}

__simt_callee__ __aicore__ inline void StateIncr(uint32_t* counter, const uint64_t n) 
{
    uint32_t nlo = static_cast<uint32_t>(n);
    uint32_t nhi = static_cast<uint32_t>(n >> 32);

    counter[0] += nlo;
    if (counter[0] < nlo) {
        nhi++;
    }
    counter[1] += nhi;
    if (nhi <= counter[1]) 
        return;
    if (++counter[2]) return;
    ++counter[3];
}

__simt_callee__ __aicore__ inline void StateIncrHi(uint32_t* counter, uint64_t n)
{
    const uint32_t countLo = static_cast<uint32_t>(n);
    uint32_t countHi = static_cast<uint32_t>(n >> 32);

    counter[2] += countLo;
    if (counter[2] < countLo) {
        countHi++;
    }
    counter[3] += countHi;
        
}

/*
 * Helper function to return the lower and higher 32-bits from two 32-bit
 * integer multiplications.
 */
__simt_callee__ __aicore__ inline void MultiplyHighLow(uint32_t a, uint32_t b, uint32_t* result_low, uint32_t* result_high)
{
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> ALG_RGIHT_BIT);
}

// Helper function for a single round of the underlying Philox algorithm.
__simt_callee__ __aicore__ inline void ComputeSingleRound(uint32_t* counter, const uint32_t* key)
{
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(PHILOX_M4X32_A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(PHILOX_M4X32_B, counter[IDX_2], &lo1, &hi1);

    uint32_t result[ALG_COUNTER_SIZE];
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[IDX_2] = hi0 ^ counter[IDX_3] ^ key[1];
    result[IDX_3] = lo0;

    CopyArray4(counter, result);
}

__simt_callee__ __aicore__ inline void RaiseKey(uint32_t* key)
{
    key[0] += PHILOX_W32_A;
    key[1] += PHILOX_W32_B;
}

/*
 * Returns counter: a group of four random numbers using the underlying Philox
 * algorithm.
 */
__simt_callee__ __aicore__ inline void PhiloxRandom(const uint32_t* keyCst, uint32_t* counter, uint32_t* results)
{
    uint32_t key[ALG_KEY_SIZE];
    uint32_t counterTmp[ALG_COUNTER_SIZE];
    CopyArray4(key, keyCst);
    CopyArray4(counterTmp, counter);

#pragma unroll    
    for (int32_t k = 0; k < LOOP_CNT; k++) {
        ComputeSingleRound(counterTmp, key);
        RaiseKey(key);
    }
    CopyArray4(results, counterTmp);
}

__simt_callee__ __aicore__ inline void SkipAhead(uint64_t n, uint32_t& state, const uint32_t* keyCst, 
                                uint32_t* counter, uint32_t* results)
{
    // counter is result
    state += (n & 3);
    n /= 4;
    if (state > 3) {
        n += 1;
        state -= 4;
    }
    StateIncr(counter, n);
    PhiloxRandom(keyCst, counter, results);
}

__simt_callee__ __aicore__ inline void SkipAhead_Sequence(uint64_t n, uint32_t& state, const uint32_t* keyCst, 
                                        uint32_t* counter, uint32_t* results)
{
    StateIncrHi(counter, n);
    PhiloxRandom(keyCst, counter, results);
}

__simt_callee__ __aicore__ inline void RandInit(uint32_t& state, uint64_t subsequence, uint64_t offset,
                    uint32_t* key, uint32_t* counter, uint32_t* results)
{
    uint32_t counterZero[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    CopyArray4(counter, counterZero);
    state = 0;
    
    SkipAhead_Sequence(subsequence, state, key, counter, results);
    SkipAhead(offset, state, key, counter, results);
}

__simt_callee__ __aicore__ inline void Rand4(uint32_t& state, uint32_t* key, uint32_t* counter, uint32_t* results, uint32_t* last)
{
    uint32_t tmp[ALG_COUNTER_SIZE];
    uint32_t r[ALG_COUNTER_SIZE];
    CopyArray4(tmp, last);
    StateIncr(counter);
    PhiloxRandom(key, counter, r);
    CopyArray4(last, r);
    switch(state) {
        case 0:
            CopyArray4(results, tmp);
            return;
        case 1:
            results[0] = tmp[1];
            results[1] = tmp[2];
            results[2] = tmp[3];
            results[3] = r[0];
            break;
        case 2:
            results[0] = tmp[2];
            results[1] = tmp[3];
            results[2] = r[0];
            results[3] = r[1];
            break;
        case 3:
            results[0] = tmp[3];
            results[1] = r[0];
            results[2] = r[1];
            results[3] = r[2];
            break;
        default:
            CopyArray4(results, tmp);
            return;
    }
    return;
}

__simt_callee__ __aicore__ inline void Rand1(uint32_t& state, uint32_t* key, uint32_t* counter, uint32_t& results, uint32_t* last)
{
    uint32_t curRes[ALG_COUNTER_SIZE];
    switch(state++) {
        default:
            results = last[0];
            break;
        case 1:
            results = last[1];
            break;
        case 2:
            results = last[2];
            break;
        case 3:
            results = last[3];
            break;
    }
    if (state == 4) {
        StateIncr(counter);
        PhiloxRandom(key, counter, curRes);
        CopyArray4(last, curRes);
        state -= 4;
    }
    return;
}

template <typename T, typename V>
__simt_callee__ __aicore__ inline T uniform_int_from_to(V val, uint64_t range, int64_t base)
{
    return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

template <typename T>
__simt_callee__ __aicore__ inline void ConvertToResult(T* output, const uint32_t* results)
{
    int64_t from;
    int64_t to;
    if (IsSameType<T, int32_t>::value) {
        from = INT32_MIN;
        to = INT32_MAX;
    } else{
        from = INT64_MIN;
        to = INT64_MAX;
    }
    uint64_t range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    int64_t base = from;

    uint64_t philox[CONVERT_COUNTER_SIZE];
    T converted[CONVERT_COUNTER_SIZE];
    philox[0] = (static_cast<uint64_t>(results[0])) << 32 | static_cast<uint64_t>(results[1]);
    converted[0] = uniform_int_from_to<T>(philox[0], range, base);
    philox[1] = (static_cast<uint64_t>(results[2])) << 32 | static_cast<uint64_t>(results[3]);
    converted[1] = uniform_int_from_to<T>(philox[1], range, base);
    CopyArray2(output, converted);
}

} // namespace StatelessRandperm
#endif
