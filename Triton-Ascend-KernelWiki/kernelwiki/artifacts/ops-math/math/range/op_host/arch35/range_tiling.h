/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RANGE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RANGE_H_

#include <cstdint>

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
// B) compileInfo结构体保留，做简单修改，删除其中非ascendc的部分
struct RangeCompileInfo {
    uint32_t running_core_num;
    // 简化后的compileInfo，只保留ascendc相关的核心信息
    int64_t totalCoreNum; // 物理总核数（保留）
    int64_t ubSize;       // UB大小
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

template <typename T>
ge::graphStatus RangeGetConstValue(gert::TilingContext* context, const gert::Tensor* tensor, T& value);

template <typename T>
ge::graphStatus CalculateOutputSize(
    gert::TilingContext* context, const gert::Tensor* tensorStart, const gert::Tensor* tensorLimit,
    const gert::Tensor* tensorDelta, uint64_t& outputSize);

ge::graphStatus OpTilingCalculateOutputSize(
    gert::TilingContext* context, const gert::Tensor* start, const gert::Tensor* tensorLimit,
    const gert::Tensor* tensorDelta, const ge::DataType dtypeOutput);
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_RANGE_H_
