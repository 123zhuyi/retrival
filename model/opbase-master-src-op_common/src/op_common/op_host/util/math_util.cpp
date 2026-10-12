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
 * \file math_util.cpp
 * \brief common math operator functions
 */

#include "op_common/op_host/util/math_util.h"

namespace Ops {
namespace Base {
constexpr int32_t SHIFT_2_BITS = 2;
constexpr int32_t SHIFT_4_BITS = 4;
constexpr int32_t SHIFT_8_BITS = 8;
constexpr int32_t SHIFT_16_BITS = 16;
constexpr int32_t SHIFT_32_BITS = 32;

constexpr uint32_t kRolScrambleLeft = 15;
constexpr uint32_t kRolScrambleRight = 17;
constexpr uint32_t kRolBodyLeft = 13;
constexpr uint32_t kRolBodyRight = 19;
constexpr uint32_t kChunkOffset = 5;
constexpr uint32_t kRolTailLeft = 8;
constexpr uint32_t kRolTailRight = 13;
constexpr uint32_t kReadSize = 16;

uint64_t LastPow2(uint64_t n)
{
    n |= (n >> 1);
    n |= (n >> SHIFT_2_BITS);
    n |= (n >> SHIFT_4_BITS);
    n |= (n >> SHIFT_8_BITS);
    n |= (n >> SHIFT_16_BITS);
    n |= (n >> SHIFT_32_BITS);

    return std::max(static_cast<uint64_t>(1), n - (n >> 1));
}

static inline uint32_t MurmurScramble(uint32_t key)
{
    key *= static_cast<uint32_t>(0xcc9e2d51);
    key = (key << kRolScrambleLeft) | (key >> kRolScrambleRight);
    key *= static_cast<uint32_t>(0x1b873593);
    return key;
}

uint32_t MurmurHash(const void* src, uint32_t srcLen, uint32_t seed)
{
    const uint32_t* key = static_cast<const uint32_t*>(src);
    uint32_t hash_key = seed;
    uint32_t tmp_key;
    // Read in blocks of 4
    for (uint32_t i = srcLen >> 2; i > 0U; i--) {
        // Get a source of differing results across endiannesses.
        tmp_key = *key;
        key++;
        hash_key ^= MurmurScramble(tmp_key);
        hash_key = (hash_key << kRolBodyLeft) | (hash_key >> kRolBodyRight);
        hash_key = hash_key * kChunkOffset + 0xe6546b64;
    }
    // Process the rest
    const uint8_t* rest_key = static_cast<const uint8_t*>(src);
    tmp_key = 0U;
    for (uint32_t i = srcLen & 3U; i != 0; i--) {
        tmp_key <<= kRolTailLeft;
        tmp_key |= rest_key[i - 1U];
    }
    hash_key ^= MurmurScramble(tmp_key);
    // Finalize
    hash_key ^= srcLen;
    hash_key ^= hash_key >> kReadSize;
    hash_key *= 0x85ebca6b;
    hash_key ^= hash_key >> kRolTailRight;
    hash_key *= 0xc2b2ae35;
    hash_key ^= hash_key >> kReadSize;
    return hash_key;
}

/**
 * @brief   将总长度splitLen尽可能平均划分为parts个子块。
 * @param [in] splitLen  要划分的总长度，入参大于0。
 * @param [in] parts  期望的划分块数，入参大于0。
 * @param [out] splitResult  划分结果，包含：
                            - splitCount: 实际划分块数
                            - splitFactor：除最后一块外每个子块的长度
                            - splitTailFactor：最后一块的长度

 * @return  返回true表示划分成功，false表示划分失败（例如输入非法）
 */
bool SplitIntoEqualByParts(int64_t splitLen, int32_t parts, SplitResult& splitResult)
{
    if (splitLen <= 0 || parts <= 0) {
        return false;
    }
    splitResult.splitCount = splitLen < parts ? splitLen : parts;
    splitResult.splitFactor = CeilDiv(splitLen, static_cast<int64_t>(splitResult.splitCount));
    splitResult.splitCount = CeilDiv(splitLen, splitResult.splitFactor);
    splitResult.splitTailFactor = (splitLen % splitResult.splitFactor == 0) ? splitResult.splitFactor :
                                                                              splitLen % splitResult.splitFactor;
    return true;
}

/**
 * @brief   给定长度因子factor，尽可能等长地划分总长度splitLen。
 * @param [in] splitLen  要划分的总长度，入参大于0。
 * @param [in] factor   每块的理想长度，入参大于0。
 * @param [out] splitResult  划分结果，包含：
                            - splitCount: 实际划分块数
                            - splitFactor：除最后一块外每个子块的长度
                            - splitTailFactor：最后一块的长度

 * @return  返回true表示划分成功，false表示划分失败（例如输入非法）
 */
bool SplitIntoEqualByFactor(int64_t splitLen, int32_t factor, SplitResult& splitResult)
{
    if (splitLen <= 0 || factor <= 0) {
        return false;
    }
    splitResult.splitFactor = splitLen < factor ? splitLen : factor;
    splitResult.splitCount = CeilDiv(splitLen, static_cast<int64_t>(splitResult.splitFactor));
    splitResult.splitTailFactor = (splitLen % splitResult.splitFactor == 0) ? splitResult.splitFactor :
                                                                              splitLen % splitResult.splitFactor;
    splitResult.splitFactor = (splitResult.splitCount == 1) ? 0 : splitResult.splitFactor;
    return true;
}
} // namespace Base
} // namespace Ops
