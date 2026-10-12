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
 * \file exp_segsum_grad_tiling_data.h
 * \brief POD tiling data struct shared by arch35 host tiling and kernel (Ascend950).
 *        Field layout mirrors the A2 ExpSegsumGradTilingData so the compute logic is preserved.
 */

#ifndef EXP_SEGSUM_GRAD_TILING_DATA_ARCH35_H
#define EXP_SEGSUM_GRAD_TILING_DATA_ARCH35_H

#include <cstdint>

// Ascend950 has 64 AIV cores; needCoreNum can reach coreNumPlatform, so the
// per-core batch bound arrays must be >= 64 (A2 used 50, which is too small here).
constexpr uint16_t EXP_SEGSUM_GRAD_MAX_CORE_ARCH35 = 64;

struct ExpSegsumGradTilingDataArch35 {
    int64_t needCoreNum = 0;
    int64_t batches = 0;
    int64_t tailDimLength = 0;
    int64_t slideSize = 0;
    int32_t batchStart[EXP_SEGSUM_GRAD_MAX_CORE_ARCH35] = {0};
    int32_t batchEnd[EXP_SEGSUM_GRAD_MAX_CORE_ARCH35] = {0};
};

#endif // EXP_SEGSUM_GRAD_TILING_DATA_ARCH35_H
