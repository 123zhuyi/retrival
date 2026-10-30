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
 * \file adjacent_difference_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADJACENT_DIFFERENCE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADJACENT_DIFFERENCE_H_

#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdjacentDifferenceTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalSize);
TILING_DATA_FIELD_DEF(int64_t, tilingNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdjacentDifference, AdjacentDifferenceTilingData)

struct AdjacentDifferenceCompileInfo
{
    uint32_t aivCoreNum_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    uint32_t ubSize_ = 0;
    uint32_t vecRegSize_ = 0;
    uint32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_ADJACENT_DIFFERENCE_H_
