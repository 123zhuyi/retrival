/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_v2_tiling_arch35.cpp
 * \brief add_v2 tiling for ascend950 (arch35)
 */

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base_util.h"
#include "infershape_broadcast_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/arch35/add_v2_dag.h"
#include "../../op_kernel/arch35/add_v2_struct_arch35.h"
#include "add_v2_tiling_arch35.h"

using namespace ge;
using namespace AddV2Op;
using namespace Ops::Base;

namespace optiling {

constexpr int64_t ASCEND_WORKSPACE = 16777216; // 16M

// 自定义模板分支：空 Tensor 走 schMode 999 + userDef 1，见 add_v2_struct_arch35.h
constexpr uint64_t ADD_V2_SCH_MODE_EMPTY = 999;
constexpr uint64_t ADD_V2_USER_DEF_NORMAL = 0;
constexpr uint64_t ADD_V2_USER_DEF_EMPTY = 1;

class AddV2TilingArch35 {
public:
    explicit AddV2TilingArch35(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcDtype();
    ge::graphStatus CheckShape() const;
    ge::graphStatus CheckDtype() const;
    ge::graphStatus SetWorkspace() const;
    ge::graphStatus HandleEmptyTensor() const;

private:
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext;
};

ge::graphStatus AddV2TilingArch35::SetWorkspace() const
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = static_cast<uint64_t>(ASCEND_WORKSPACE);
    return ge::GRAPH_SUCCESS;
}

// 空 Tensor 早返回。ATVOSS 的 BroadcastBaseTiling 在合轴之后会显式拒绝 0 元素
// （broadcast_tiling.h: "tensor check is empty, check failed"），不能落到 DoTiling，
// 因此这里自己出一份 tiling：blockDim = 1，tilingKey 选自定义分支，kernel 侧直接返回。
ge::graphStatus AddV2TilingArch35::HandleEmptyTensor() const
{
    OP_LOGD(tilingContext, "AddV2: empty tensor, skip kernel computation.");
    OP_CHECK_IF(SetWorkspace() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext, "AddV2: set workspace failed (empty tensor)."), return ge::GRAPH_FAILED);

    auto* tilingData = tilingContext->GetTilingData<AddV2EmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingData);
    tilingData->numel = 0;

    tilingContext->SetBlockDim(1);
    tilingContext->SetTilingKey(GET_TPL_TILING_KEY(ADD_V2_SCH_MODE_EMPTY, ADD_V2_USER_DEF_EMPTY));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddV2TilingArch35::CalcDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddV2TilingArch35::CheckShape() const
{
    auto inputX1 = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputX1);
    auto inputX2 = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputX2);
    auto outputY = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputY);
    return ge::GRAPH_SUCCESS;
}

// 仅注册同 dtype 组合（canonical AddV2 Verifier 亦要求 x1/x2 同 dtype）
ge::graphStatus AddV2TilingArch35::CheckDtype() const
{
    auto input1Desc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    OP_CHECK_IF(input1Dtype != this->inputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x1 and x2",
                                                       (ge::TypeUtils::DataTypeToSerialString(this->inputDtype) +
                                                        " and " + ge::TypeUtils::DataTypeToSerialString(input1Dtype))
                                                           .c_str(),
                                                       "x1 and x2 must have the same dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddV2TilingArch35::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "AddV2TilingArch35 RunTiling enter.");
    OP_CHECK_IF(
        CalcDtype() == ge::GRAPH_FAILED,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "x1", "unknown", "calc dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "input shape", "invalid",
                                                      "input shape check failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() == ge::GRAPH_FAILED,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "input dtype", "invalid",
                                                      "input dtype check failed"),
                return ge::GRAPH_FAILED);

    // y 的元素个数为 0 即空 Tensor（广播规则下 0 只能与 0 或 1 相配，输出空 <=> 有输入为空）。
    auto outputY = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputY);
    int64_t numel = outputY->GetStorageShape().GetShapeSize();
    // GetShapeSize() 在维度乘积溢出 int64_t 时返回 kInvalidDimValue（负数）而不报错，
    // 必须在 numel == 0 判断之前拦掉，否则负的 numel 会被当成非空穿透到 DoTiling。
    OP_CHECK_IF(numel < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "y",
                                                      Ops::Base::ToString(outputY->GetStorageShape()).c_str(),
                                                      "the product of all dims overflows int64_t"),
                return ge::GRAPH_FAILED);
    if (numel == 0) {
        return HandleEmptyTensor();
    }

    ge::graphStatus ret = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    if (this->inputDtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<AddWithCastCompute<half>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_BF16) {
        BroadcastBaseTiling<AddWithCastCompute<bfloat16_t>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<AddWithCastCompute<float>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_INT64 || this->inputDtype == ge::DT_COMPLEX64) {
        BroadcastBaseTiling<AddWithoutCastCompute<int64_t>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_UINT8) {
        BroadcastBaseTiling<AddWithoutCastCompute<uint8_t>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_INT8) {
        BroadcastBaseTiling<AddWithoutCastCompute<int8_t>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_INT32) {
        BroadcastBaseTiling<AddWithoutCastCompute<int32_t>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else if (this->inputDtype == ge::DT_INT16) {
        BroadcastBaseTiling<AddWithoutCastCompute<int16_t>::OpDag> brcBaseTiling(tilingContext);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), ADD_V2_USER_DEF_NORMAL);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x1",
                                  ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
                                  "fp16, bf16, fp32, int64, int32, int16, uint8, int8, complex64");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret == ge::GRAPH_FAILED,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "tiling", "failed",
                                                      "broadcastBaseTiling failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetWorkspace() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "AddV2: set workspace failed."),
                return ge::GRAPH_FAILED);

    OP_LOGD(tilingContext, "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AddV2Arch35(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter Tiling4AddV2Arch35");
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_VALUE("AddV2", "tiling_context", "nullptr", "not nullptr"),
                return ge::GRAPH_FAILED);
    AddV2TilingArch35 addV2Tiling(context);
    return addV2Tiling.RunTiling();
}

static ge::graphStatus TilingPrepare4AddV2Arch35(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AddV2CompileInfoArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_CHECK_IF(compileInfo->totalCoreNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "totalCoreNum,ubSize",
                    std::to_string(compileInfo->totalCoreNum) + ", " + std::to_string(compileInfo->ubSize),
                    "The values of totalCoreNum and ubSize must be greater than 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddV2).Tiling(Tiling4AddV2Arch35).TilingParse<AddV2CompileInfoArch35>(TilingPrepare4AddV2Arch35);
} // namespace optiling
