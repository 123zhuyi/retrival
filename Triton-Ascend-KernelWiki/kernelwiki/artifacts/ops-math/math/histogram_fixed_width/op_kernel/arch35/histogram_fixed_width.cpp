/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

#ifndef HISTOGRAM_FIXED_WIDTH_THREAD_NUM
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif
#define HISTOGRAM_FIXED_WIDTH_THREAD_NUM
#endif

#include "histogram_fixed_width_tilingdata.h"
#include "histogram_fixed_width_tilingkey.h"
#include "histogram_fixed_width_simt_full_load.h"
#include "histogram_fixed_width_simt_not_full_load.h"
#include "histogram_fixed_width_simt_not_full_load_simt.h"

using namespace HistogramFixedWidthSIMT;

template <typename X>
struct ComputeTypeOf {
    using Type = float;
};
template <>
struct ComputeTypeOf<int32_t> {
    using Type = int64_t;
};
template <>
struct ComputeTypeOf<int64_t> {
    using Type = int64_t;
};

template <uint64_t loadMode>
__aicore__ inline void HistogramFixedWidthCompute(GM_ADDR x, GM_ADDR range, GM_ADDR y, GM_ADDR tiling)
{
    using X_TYPE = DTYPE_X;
    using COMPUTE_TYPE = typename ComputeTypeOf<X_TYPE>::Type;

    AscendC::TPipe tpipe;
    if constexpr (loadMode == TPL_LOAD_MODE_UB_FULL) {
        GET_TILING_DATA_WITH_STRUCT(HistogramFixedWidthSimtTilingData, tilingData, tiling);
        HistogramFixedWidthSIMT::HistogramFixedWidthSimtFullLoad<X_TYPE, COMPUTE_TYPE> op;
        op.Init(x, range, y, &tilingData, &tpipe);
        op.Process();
    } else if constexpr (loadMode == TPL_LOAD_MODE_UB_NOT_FULL) {
        GET_TILING_DATA_WITH_STRUCT(HistogramFixedWidthSimtTilingData, tilingData, tiling);
        HistogramFixedWidthSIMT::HistogramFixedWidthSimtNotFullLoad<X_TYPE, COMPUTE_TYPE> op;
        op.Init(x, range, y, &tilingData, &tpipe);
        op.Process();
    } else if constexpr (loadMode == TPL_LOAD_MODE_UB_NOT_FULL_SIMT) {
        GET_TILING_DATA_WITH_STRUCT(HistogramFixedWidthSimtTilingData, tilingData, tiling);
        HistogramFixedWidthSIMT::HistogramFixedWidthSimtNotFullLoadGmAtomicAdd<X_TYPE, COMPUTE_TYPE> op;
        op.Init(x, range, y, &tilingData);
        op.Process();
    }
}

template <uint64_t loadMode>
__global__ __aicore__ void histogram_fixed_width(GM_ADDR x, GM_ADDR range, GM_ADDR nbins, GM_ADDR y, GM_ADDR workspace,
                                                 GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(HistogramFixedWidthSimtTilingData);

    HistogramFixedWidthCompute<loadMode>(x, range, y, tiling);
}
