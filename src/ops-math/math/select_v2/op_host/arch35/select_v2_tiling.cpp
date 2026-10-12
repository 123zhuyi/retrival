/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file    select_v2_tiling.cpp
 * \brief   select_v2_tiling source file
 */

 #include "select_v2_tiling.h"
 #include <graph/utils/type_utils.h>
 #include "atvoss/broadcast/broadcast_tiling.h"
 #include "math/select_v2/op_kernel/arch35/select_v2_dag.h"
 #include "math/select_v2/op_kernel/arch35/select_v2_struct.h"
 #include "op_host/math_tiling_templates_registry.h"

 using namespace AscendC;
 using namespace ge;
 
 namespace optiling {
 static constexpr uint64_t SELECT_V2_COMMON_TILING_PRIORITY = 0;
 
 ge::graphStatus SelectV2Tiling::GetShapeAttrsInfo()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 bool SelectV2Tiling::IsCapable()
 {
     return true;
 }
 
 ge::graphStatus SelectV2Tiling::DoOpTiling()
 {
     auto inputDesc = context_->GetInputDesc(0);
     OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
     ge::DataType input0DType = inputDesc->GetDataType();
 
     auto input1Desc = context_->GetInputDesc(1);
     OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
     ge::DataType input1DType = input1Desc->GetDataType();

     auto input2Desc = context_->GetInputDesc(2);
     OP_CHECK_NULL_WITH_CONTEXT(context_, input2Desc);
     ge::DataType input2DType = input2Desc->GetDataType();

     auto outputYDesc = context_->GetOutputDesc(0);
     OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
     ge::DataType outputDtype = outputYDesc->GetDataType();
     if ((input0DType != ge::DT_BOOL) || (input1DType != input2DType) || (outputDtype != input1DType)) {
         std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                                 ge::TypeUtils::DataTypeToSerialString(input1DType) + ", " +
                                 ge::TypeUtils::DataTypeToSerialString(input2DType) + " and " +
                                 ge::TypeUtils::DataTypeToSerialString(outputDtype);
         OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "condition, x1, x2 and y",
             dtypesStr.c_str(), "The dtype of condition must be bool, and the dtypes of x1, x2 and y must be the same");
         return ge::GRAPH_FAILED;
     }
 
     ge::graphStatus ret = ge::GRAPH_SUCCESS;
     if (input1DType == ge::DT_INT64) {
         Ops::Base::BroadcastBaseTiling<SelectV2Op::SelectV2<int64_t>::OpDag> brcBaseTiling(context_);
         ret = brcBaseTiling.DoTiling();
         tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
     } else if (input1DType == ge::DT_INT8) {
         Ops::Base::BroadcastBaseTiling<SelectV2Op::SelectV2<int8_t>::OpDag> brcBaseTiling(context_);
         ret = brcBaseTiling.DoTiling();
         tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
     } else if (input1DType == ge::DT_UINT8) {
         Ops::Base::BroadcastBaseTiling<SelectV2Op::SelectV2<uint8_t>::OpDag> brcBaseTiling(context_);
         ret = brcBaseTiling.DoTiling();
         tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
     } else if (input1DType == ge::DT_INT32) {
         Ops::Base::BroadcastBaseTiling<SelectV2Op::SelectV2<int32_t>::OpDag> brcBaseTiling(context_);
         ret = brcBaseTiling.DoTiling();
         tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
     } else if (input1DType == ge::DT_FLOAT16 || input1DType == ge::DT_BF16) {
         Ops::Base::BroadcastBaseTiling<SelectV2Op::SelectV2<Ops::Base::half>::OpDag> brcBaseTiling(context_);
         ret = brcBaseTiling.DoTiling();
         tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
     } else if (input1DType == ge::DT_FLOAT) {
         Ops::Base::BroadcastBaseTiling<SelectV2Op::SelectV2<float>::OpDag> brcBaseTiling(context_);
         ret = brcBaseTiling.DoTiling();
         tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
     } else {
         std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input1DType);
         OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
             "int64, int32, int8, uint8, float16, bf16 or fp32");
         return ge::GRAPH_FAILED;
     }
 
     return ret;
 }
 
 ge::graphStatus SelectV2Tiling::DoLibApiTiling()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 uint64_t SelectV2Tiling::GetTilingKey() const
 {
     return tilingKey;
 }
 
 ge::graphStatus SelectV2Tiling::GetWorkspaceSize()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus SelectV2Tiling::PostTiling()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus SelectV2Tiling::GetPlatformInfo()
 {
     return ge::GRAPH_SUCCESS;
 }
 
 ge::graphStatus TilingForSelectV2(gert::TilingContext* context)
 {
     OP_LOGD("SelectV2Tiling", "Enter TilingForSelectV2");
     if (context == nullptr) {
         OP_LOGE("SelectV2Tiling", "Tiling context is nullptr");
         return ge::GRAPH_FAILED;
     }
 
     OP_LOGD(context, "Enter ascendc SelectV2Tiling");
     return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
 }

 ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}
 
 IMPL_OP_OPTILING(SelectV2)
     .Tiling(TilingForSelectV2)
     .TilingParse<Ops::Base::BroadcastCompileInfo>(TilingPrepareForBroadcast);
 REGISTER_OPS_TILING_TEMPLATE(SelectV2, SelectV2Tiling, SELECT_V2_COMMON_TILING_PRIORITY);
 }   // namespace optiling 