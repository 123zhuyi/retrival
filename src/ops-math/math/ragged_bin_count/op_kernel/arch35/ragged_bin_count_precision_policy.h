/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAGGED_BIN_COUNT_PRECISION_POLICY_H
#define RAGGED_BIN_COUNT_PRECISION_POLICY_H

#include <cstdint>

namespace NsRaggedBinCount {

// A VALUE-mapped weighted launch enters the deterministic partial-reduction
// path only for this high-contention subdomain.  Keep the policy shared by the
// host workspace calculation and the kernel dispatch so they cannot drift.
constexpr int64_t PRECISE_VALUE_MAX_BINS = 2;
constexpr int64_t PRECISE_VALUE_MAX_WORK = 262144;

// Each core emits this many fixed-order partials per output element.  A slot
// stores high, low, flags and padding (16 bytes); the first 32 user-workspace
// bytes remain reserved for the whole-input validation flag.
constexpr int64_t PRECISE_VALUE_PARTITIONS_PER_CORE = 16;
constexpr uint64_t PRECISE_VALUE_PARTIAL_BYTES = 16U;
constexpr uint64_t USER_WORKSPACE_HEADER_BYTES = 32U;

#ifdef __CCE_AICORE__
__aicore__ inline bool UsePreciseValuePath(int64_t numValues, int64_t numBins)
#else
constexpr bool UsePreciseValuePath(int64_t numValues, int64_t numBins)
#endif
{
    return numValues > 0 && numBins > 0 && numBins <= PRECISE_VALUE_MAX_BINS &&
           numValues <= PRECISE_VALUE_MAX_WORK / numBins;
}

static_assert(PRECISE_VALUE_MAX_WORK <= (1LL << 42),
              "the five-limb exact accumulator cannot cover this VALUE work bound");

} // namespace NsRaggedBinCount

#endif // RAGGED_BIN_COUNT_PRECISION_POLICY_H
