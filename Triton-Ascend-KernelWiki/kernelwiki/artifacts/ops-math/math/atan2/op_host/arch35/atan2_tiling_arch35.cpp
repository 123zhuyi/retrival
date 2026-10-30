/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file atan2_tiling_arch35.cpp
 * \brief atan2_tiling source file
 */

#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/atan2/op_kernel/arch35/atan2_dag.h"
#include "math/atan2/op_kernel/arch35/atan2_struct.h"
#include "atan2_tiling_arch35.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;
namespace optiling {

constexpr static uint64_t ATAN2_COMMON_TILING_PRIORITY = 0;

ge::graphStatus Atan2Tiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool Atan2Tiling::IsCapable()
{
    return true;
}


bool Atan2Tiling::CheckDtype(
    const ge::DataType& input0Dtype, const ge::DataType& input1Dtype, const ge::DataType& outputDtype) const
{
    if (input0Dtype != input1Dtype || input0Dtype != outputDtype) {
        OP_LOGE(
            context_->GetNodeName(), "Dtype of input0[%s] should be equal to dtype of input1[%s] and output[%s].",
            ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(input1Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str());
        return false;
    }
    return true;
}

ge::graphStatus Atan2Tiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (!CheckDtype(input0Dtype, input1Dtype, outputDtype)) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<Atan2Op::Atan2Dag<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<Atan2Op::Atan2Dag<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_BF16) {
        BroadcastBaseTiling<Atan2Op::Atan2Dag<bfloat16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE(context_->GetNodeName(), "Input dtype is only support fp16, bf16, fp32, while got %s!",
            ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus Atan2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t Atan2Tiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus Atan2Tiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Atan2Tiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Atan2Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAtan2(gert::TilingContext* context)
{
    OP_LOGD("Atan2Tiling", "Enter TilingForAtan2");
    if (context == nullptr) {
        OP_LOGE("Atan2Tiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc Atan2Tiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForAtan2([[maybe_unused]] gert::TilingParseContext* context){
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Atan2).Tiling(TilingForAtan2).TilingParse<BroadcastCompileInfo>(TilingPrepareForAtan2);

REGISTER_OPS_TILING_TEMPLATE(Atan2, Atan2Tiling, ATAN2_COMMON_TILING_PRIORITY);
} // namespace optiling
