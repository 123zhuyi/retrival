/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HISTOGRAM_FIXED_WIDTH_SIMT_COMMON_H
#define HISTOGRAM_FIXED_WIDTH_SIMT_COMMON_H

#include "simt_api/asc_simt.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace HistogramFixedWidthSIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE, typename CORE_NUM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCleanY(__gm__ int32_t* yGmAddr, const int32_t blockIdx,
                                                                       const CORE_NUM_T clearYCoreNum,
                                                                       const int64_t clearYIndexBase,
                                                                       const int32_t clearYDataLength)
{
    if (blockIdx >= static_cast<int32_t>(clearYCoreNum)) {
        return;
    }

    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < clearYDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t yIndex = clearYIndexBase + index;
        yGmAddr[yIndex] = static_cast<int32_t>(0);
    }
}

constexpr int64_t STRIDE_OFFSET_2 = 2;
constexpr int64_t STRIDE_OFFSET_3 = 3;
constexpr int64_t STRIDE_OFFSET_4 = 4;
constexpr int64_t STRIDE_OFFSET_5 = 5;

} // namespace HistogramFixedWidthSIMT

#endif // HISTOGRAM_FIXED_WIDTH_SIMT_COMMON_H
