/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_TILING_ARCH35_H
#define GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_TILING_ARCH35_H

#include "graph/types.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/grouped_dynamic_mx_quant_with_dual_axis_tilingdata.h"

namespace optiling {
struct GroupedDynamicMxQuantWithDualAxisCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct GroupedDynamicMxQuantWithDualAxisTilingParam {
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t groupNum = 0;
    int64_t rowBlockSize = 0;
    int64_t colBlockSize = 0;
    int64_t colBlocksNum = 0;
    int64_t scale1ColPairs = 0;
    int64_t scaleAlg = 1;
    int64_t dstDtype = 0;
    int64_t inputDtype = 0;
    int64_t ubSize = 0;
    int64_t workspaceSize = 0;
};
} // namespace optiling

#endif // GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_TILING_ARCH35_H
