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
 * \file transpose_v2_tiling.cpp
 * \brief
 */

#include "transpose021_tiling.h"
#include "transpose102_tiling.h"

namespace optiling {

template <typename T>
static ge::graphStatus DoOpTiling(gert::TilingContext* context)
{
    T transposeV2Tiling(context);
    auto ret = transposeV2Tiling.GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetPlatformInfo failed");
        return ret;
    }
    ret = transposeV2Tiling.DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "DoTiling failed");
        return ret;
    }
    transposeV2Tiling.ComputeTilingKey();
    transposeV2Tiling.SetTiling();
    transposeV2Tiling.PrintTilingData();
    return ret;
}

template <typename T>
void GetPerm(gert::TilingContext* context, std::vector<int64_t>& perm)
{
    const gert::Tensor* permTensor = context->GetInputTensor(1);
    const T* permValue = permTensor->GetData<T>();
    const gert::StorageShape* permShape = context->GetInputShape(1);
    int32_t permNum = permShape->GetStorageShape().GetDim(0);
    for (int32_t i = 0; i < permNum; i++) {
        perm.push_back(static_cast<int64_t>(permValue[i]));
    }
}

static ge::graphStatus Tiling4TransposeV2(gert::TilingContext* context)
{
    ge::DataType permDatatype = context->GetInputDesc(1)->GetDataType();
    std::vector<int64_t> perm;
    if (permDatatype == ge::DT_INT32) {
        GetPerm<int32_t>(context, perm);
    } else if (permDatatype == ge::DT_INT64) {
        GetPerm<int64_t>(context, perm);
    }
    ge::graphStatus ret;
    if (perm == std::vector<int64_t>{0, 2, 1}) {
        ret = DoOpTiling<Transpose021Tiling>(context);
    } else if (perm == std::vector<int64_t>{1, 0, 2} || perm == std::vector<int64_t>{0, 2, 1, 3}) {
        ret = DoOpTiling<Transpose102Tiling>(context);
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupport perm.");
        return ge::GRAPH_FAILED;
    }
    return ret;
}

static ge::graphStatus TilingPrepare4TransposeV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4TransposeV2 tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransposeV2)
    .Tiling(Tiling4TransposeV2)
    .TilingParse<Tiling4TransposeV2CompileInfo>(TilingPrepare4TransposeV2)
    .TilingInputsDataDependency({1});
} // namespace optiling
