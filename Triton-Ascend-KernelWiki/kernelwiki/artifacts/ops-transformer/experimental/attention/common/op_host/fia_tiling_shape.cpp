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
 * \file fia_tiling_shape.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "fia_tiling_shape.h"


namespace optiling {
static const std::map<FiaLayout, std::vector<FiaAxis>> FIA_LAYOUT_AXIS_MAP = {
    {FiaLayout::BSND, {FiaAxis::B, FiaAxis::S, FiaAxis::N, FiaAxis::D}},
    {FiaLayout::BNSD, {FiaAxis::B, FiaAxis::N, FiaAxis::S, FiaAxis::D}},
};

static ge::graphStatus GetLayoutAxes(std::vector<FiaAxis> &layoutAxes, const FiaLayout &layout,
    const std::string &opName, const std::string &funcName)
{
    auto it = FIA_LAYOUT_AXIS_MAP.find(layout);
    if (it == FIA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(opName, "[%s] compare layout %s is unsupported.",
            funcName.c_str(), LayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    layoutAxes = it->second;
    return ge::GRAPH_SUCCESS;
}

bool FiaTilingShape::HasAxis(const FiaAxis &axis) const
{   
    const auto& layoutIt = FIA_LAYOUT_AXIS_MAP.find(layout_);
    if (layoutIt == FIA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<FiaAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    return true;
}

size_t FiaTilingShape::GetAxisIdx(const FiaAxis &axis) const
{
    if (HasAxis(axis)) {
        const std::vector<FiaAxis>& axes = FIA_LAYOUT_AXIS_MAP.find(layout_)->second;
        const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
        return std::distance(axes.begin(), axisIt);
    }
    return 0;
}

int64_t FiaTilingShape::GetAxisNum(const FiaAxis &axis) const
{
    return HasAxis(axis) ? shape_.GetDim(GetAxisIdx(axis)) : invalidDimValue_;
}
} // namespace optiling