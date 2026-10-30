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
 * \file rfft1d_tiling_base.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "platform/platform_info.h"
#include "rfft1_d_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

ge::graphStatus Rfft1DBaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const Rfft1DCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;

        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNum();
        if (coreNum < 1) {
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Rfft1DBaseTiling::GetShapeAttrsInfo()
{
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    length = *runtimeAttrs->GetAttrPointer<int64_t>(0);
    normal = *runtimeAttrs->GetAttrPointer<int64_t>(1);

    auto inputXDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    dtype = context_->GetInputDesc(0)->GetDataType();
    if (dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(), "Rfft1D support float32");
        return ge::GRAPH_FAILED;
    }

    if (normal != static_cast<int32_t>(NORM_VALUES::BACKWARD) && normal !=  static_cast<int32_t>(NORM_VALUES::FORWARD) && normal !=  static_cast<int32_t>(NORM_VALUES::ORTHO)) {
        OP_LOGE(context_->GetNodeName(), "Incorrect norm parameter value");
        return ge::GRAPH_FAILED;
    }
    if (length < 1) {
        OP_LOGE(context_->GetNodeName(), "n parameter should be > 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
