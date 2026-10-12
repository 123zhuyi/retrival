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
 * \file broadcast_tiling_base.cpp
 * \brief atvoss broadcast template tiling
 */
#include "op_common/atvoss/broadcast/broadcast_tiling_base.h"

namespace Ops {
namespace Base {
static constexpr int64_t BROADCAST_REPEAT_BYTES = 256;
static constexpr uint64_t BROADCAST_COMPUTE_KEY = 1;
static constexpr uint64_t MAX_NDDMA_DIM = 5;
static constexpr uint64_t SCHEDULE_KEY_1 = 100;
static constexpr uint64_t SCHEDULE_KEY_2 = 1100;
static constexpr int64_t DEFAULT_SCHEMODE = 1;

uint64_t BroadcastGetComputeKey() { return BROADCAST_COMPUTE_KEY; }

uint64_t BroadcastGetScheduleKey(uint32_t axisInsideUB)
{
    if (axisInsideUB <= MAX_NDDMA_DIM) {
        return SCHEDULE_KEY_1;
    }
    return SCHEDULE_KEY_2;
}

int64_t BroadcastGetMaxElemNum(int64_t ubSize, const BroadcastComputeParams& computeParams)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t extraSize = computeParams.extraSize[0];
    int64_t bufferDivisor = computeParams.bufferDivisor[0];
    int64_t maxElemNums = (ubSize - extraSize) * BROADCAST_BITS_NUM / bufferDivisor;
    int64_t alignFactor = BROADCAST_REPEAT_BYTES * BROADCAST_BITS_NUM / minDtypeBits;
    int64_t maxElemNumAlign = maxElemNums / alignFactor * alignFactor;
    return maxElemNumAlign;
}
} // namespace Base
} // namespace Ops
