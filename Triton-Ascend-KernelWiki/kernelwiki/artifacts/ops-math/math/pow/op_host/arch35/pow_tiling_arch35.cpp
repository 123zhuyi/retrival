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
 * \file pow_tiling_arch35.cpp
 * \brief
 */

#include "pow_tiling_arch35.h"

namespace optiling
{
using namespace Ops::Base;
constexpr size_t INPUT_BASE_IDX = 0;
constexpr size_t INPUT_EXP_IDX = 1;
constexpr size_t OUTPUT_POW_IDX = 0;

constexpr int64_t INPUT_DTYPE_B8 = 1;
constexpr int64_t INPUT_DTYPE_B16 = 2;
constexpr int64_t INPUT_DTYPE_B32 = 4;
static constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

bool PowTilingBase::IsCapable()
{
    return true;
}

ge::graphStatus PowTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t PowTilingBase::GetTilingKey() const
{
    return 0;
}

ge::graphStatus PowTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const PowCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                    OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
    params_.coreNum = compileInfo->coreNum;
    params_.ubSize = compileInfo->ubSize;
    params_.blockSize = compileInfo->blockSize;
    params_.isRegBase = compileInfo->isRegBase;
    params_.vectorLength = compileInfo->vectorLength;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowTilingBase::GetShapeAttrsInfo()
{
    auto inputBase = context_->GetInputShape(INPUT_BASE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputBase);
    params_.baseShape = inputBase->GetStorageShape();

    auto baseDesc = context_->GetInputDesc(INPUT_BASE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, baseDesc);
    params_.baseDtype = baseDesc->GetDataType();

    if (params_.baseDtype == ge::DT_FLOAT || params_.baseDtype == ge::DT_INT32) {
        params_.baseDtypeSize = INPUT_DTYPE_B32;
        params_.computeDtypeSize = INPUT_DTYPE_B32;
    } else if ((params_.baseDtype == ge::DT_FLOAT16) || (params_.baseDtype == ge::DT_BF16)) {
        params_.baseDtypeSize = INPUT_DTYPE_B16;
        params_.computeDtypeSize = INPUT_DTYPE_B32;
    } else if ((params_.baseDtype == ge::DT_INT16)) {
        params_.baseDtypeSize = INPUT_DTYPE_B16;
        params_.computeDtypeSize = INPUT_DTYPE_B16;
    } else if ((params_.baseDtype == ge::DT_INT8) || (params_.baseDtype == ge::DT_UINT8)) {
        params_.baseDtypeSize = INPUT_DTYPE_B8;
        params_.computeDtypeSize = INPUT_DTYPE_B8;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE("Pow", "x1",
            ge::TypeUtils::DataTypeToSerialString(params_.baseDtype).c_str(),
            "float32, float16, bfloat16, uint8, int8, int16 or int32");
        return ge::GRAPH_FAILED;
    }

    auto inputExp = context_->GetInputShape(INPUT_EXP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputExp);
    params_.expShape = inputExp->GetStorageShape();

    auto expDesc = context_->GetInputDesc(INPUT_EXP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expDesc);
    params_.expDtype = expDesc->GetDataType();

    auto outputPow = context_->GetOutputShape(OUTPUT_POW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputPow);
    params_.powShape = outputPow->GetStorageShape();

    auto powDesc = context_->GetOutputDesc(OUTPUT_POW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, powDesc);
    params_.powDtype = powDesc->GetDataType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE("Pow", "failed to get workspace size"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
