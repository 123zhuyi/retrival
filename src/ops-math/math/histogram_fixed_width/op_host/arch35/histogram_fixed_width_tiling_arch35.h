/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HISTOGRAM_FIXED_WIDTH_TILING_ARCH35_H
#define HISTOGRAM_FIXED_WIDTH_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "math/histogram_fixed_width/op_kernel/arch35/histogram_fixed_width_tilingdata.h"
#include "math/histogram_fixed_width/op_kernel/arch35/histogram_fixed_width_tilingkey.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "op_host/util/fp16.h"
#include "log/log.h"

namespace optiling {

struct HistogramFixedWidthCompileInfo {};

static constexpr int64_t HFW_INPUT_IDX_X = 0;
static constexpr int64_t HFW_INPUT_IDX_RANGE = 1;
static constexpr int64_t HFW_INPUT_IDX_NBINS = 2;
static constexpr int64_t HFW_OUTPUT_IDX = 0;
static constexpr int64_t HFW_RANGE_LENGTH = 2;
static constexpr int64_t HFW_SIZE_OF_INT32 = 4;
static constexpr int64_t HFW_GM_ATOMIC_ADD_FACTOR = 100;
static constexpr uint64_t HFW_SIMT_DCACHE_SIZE = 32 * 1024;

static constexpr uint64_t HFW_TPL_LOAD_MODE_UB_FULL = 0;
static constexpr uint64_t HFW_TPL_LOAD_MODE_UB_NOT_FULL = 1;
static constexpr uint64_t HFW_TPL_LOAD_MODE_UB_NOT_FULL_SIMT = 2;

class HistogramFixedWidthTiling {
public:
    explicit HistogramFixedWidthTiling(gert::TilingContext* context) : context_(context) {};
    ~HistogramFixedWidthTiling() = default;

    ge::graphStatus DoTiling();

private:
    ge::graphStatus ParamCheck();
    ge::graphStatus GetSocInfo();
    ge::graphStatus CalcTiling();
    ge::graphStatus ValidateRange(const ge::DataType xDtype);
    ge::graphStatus ReadRangeMinMax(const ge::DataType xDtype, float& minVal, float& maxVal);

    gert::TilingContext* context_;

    // soc info
    uint32_t coreNum_{0};
    uint64_t ubSize_{0};

    // input info
    int64_t totalLength_{0};
    int64_t bins_{0};

    // tiling result
    uint64_t loadMode_{0};
};

} // namespace optiling

#endif // HISTOGRAM_FIXED_WIDTH_TILING_ARCH35_H
