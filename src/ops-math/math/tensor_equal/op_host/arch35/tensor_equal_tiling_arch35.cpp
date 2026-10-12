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
 * \file tensor_equal_tiling_arch35.cpp
 * \brief
 */

#include "tensor_equal_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "platform/platform_ascendc.h"

namespace optiling {
const static size_t INPUT_X = 0;
const static size_t INPUT_Y = 1;
const static size_t OUTPUT_Z = 0;

const static int64_t SINGLE_CORE = 1;
const static int64_t MIN_CORE_PROCESS = 2048;
const static int64_t DOUBLE_BUFFER = 2;
const static int64_t DOUBLE_INPUT = 2;
const static int64_t RESERVED_UB_SIZE = 512;
const static int64_t WORKSPACE_SAVE_SIZE = 32;
const static int64_t EMPTY_SHAPE_TILINGKEY = 101;
const static int64_t DIFFER_SHAPE_TILINGKEY = 111;
const static int64_t NORMAL_SHAPE_TILINGKEY = 121;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16,  ge::DT_BOOL,
                                                        ge::DT_INT8,    ge::DT_UINT8,  ge::DT_INT16, ge::DT_UINT16,
                                                        ge::DT_INT32,   ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus TensorEqualTiling::GetPlatformInfo()
{
    OP_LOGD(opName_, "TensorEqualTiling GetPlatformInfo.");
    auto compileInfo = reinterpret_cast<const TensorEqualCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = compileInfo->totalCoreNum;
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0),
                    OP_LOGE(opName_, "ub size is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TensorEqualTiling::CheckDType()
{
    auto inputXDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    inputXDType_ = inputXDesc->GetDataType();
    int64_t inputXDTypeSize = ge::GetSizeByDataType(inputXDType_);
    if (inputXDTypeSize <= 0) {
        std::string valueStr = Ops::Base::ToString(inputXDType_);
        std::string reasonMsg = "input x dtype size is invalid, got size " + std::to_string(inputXDTypeSize);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto inputYDesc = context_->GetInputDesc(INPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputYDesc);
    inputYDType_ = inputYDesc->GetDataType();
    int64_t inputYDTypeSize = ge::GetSizeByDataType(inputYDType_);
    if (inputYDTypeSize <= 0) {
        std::string valueStr = Ops::Base::ToString(inputYDType_);
        std::string reasonMsg = "input y dtype size is invalid, got size " + std::to_string(inputYDTypeSize);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (inputXDType_ != inputYDType_) {
        std::string valueStr = Ops::Base::ToString(inputXDType_) + " and " + Ops::Base::ToString(inputYDType_);
        std::string reasonMsg = "The dtype of input x and input y must be the same";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "input tensor x and input tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (INPUT_SUPPORT_DTYPE_SET.count(inputXDType_) == 0) {
        std::string valueStr = Ops::Base::ToString(inputXDType_);
        std::string reasonMsg =
            "Input dtype only support float16, float32, bfloat16, int8, uint8, int32, uint32, int16, uint16, int64, uint64, bool currently";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, "input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }    
    
    auto outputDesc = context_->GetOutputDesc(OUTPUT_Z);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    outputDType_ = outputDesc->GetDataType();
    if (outputDType_ != ge::DataType::DT_BOOL) {
        std::string valueStr = Ops::Base::ToString(outputDType_);
        std::string reasonMsg = "output z dtype should be BOOL";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "output tensor z", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus TensorEqualTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "TensorEqualTiling GetShapeAttrsInfo.");

    auto inputXShapePtr = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXShapePtr);
    auto inputXShape = inputXShapePtr->GetStorageShape();
    auto inputYShapePtr = context_->GetInputShape(INPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputYShapePtr);
    auto inputYShape = inputYShapePtr->GetStorageShape();

    if (CheckDType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t inputXDTypeSize = ge::GetSizeByDataType(inputXDType_);
    inputDtypeSize_ = inputXDTypeSize;
    inputShapeSize_ = inputXShape.GetShapeSize();
    vRegSize_ = Ops::Base::GetVRegSize(context_);
    
    tilingKey_ = NORMAL_SHAPE_TILINGKEY;
    if (inputXShape.GetShapeSize() == 0 || inputYShape.GetShapeSize() == 0) {
        tilingKey_ = EMPTY_SHAPE_TILINGKEY;
    }
    tilingKey_ = inputXShape != inputYShape ? DIFFER_SHAPE_TILINGKEY : tilingKey_;

    return ge::GRAPH_SUCCESS;
}

bool TensorEqualTiling::IsCapable()
{
    return true;
}

// 3、计算数据切分TilingData
ge::graphStatus TensorEqualTiling::DoOpTiling()
{
    OP_LOGD(opName_, "TensorEqualTiling DoOpTiling.");

    int64_t vRegFactor = Ops::Base::CeilDiv(vRegSize_, inputDtypeSize_);
    // split core
    int64_t perCoreProcessTemp = Ops::Base::CeilDiv(inputShapeSize_, totalCoreNum_);
    int64_t perCoreProcess = std::max(perCoreProcessTemp, MIN_CORE_PROCESS);
    perCoreProcess = Ops::Base::CeilAlign(perCoreProcess, vRegFactor);
    usedCoreNum_ = Ops::Base::CeilDiv(inputShapeSize_, perCoreProcess);
    if (tilingKey_ == DIFFER_SHAPE_TILINGKEY || tilingKey_ == EMPTY_SHAPE_TILINGKEY) {
        usedCoreNum_ = SINGLE_CORE;
    }
    int64_t perCoreUbFactor = Ops::Base::CeilDiv(inputShapeSize_, usedCoreNum_);
    perCoreProcess = Ops::Base::CeilAlign(perCoreUbFactor, vRegFactor);
    int64_t tailCoreUbFactor = inputShapeSize_ - (usedCoreNum_ - 1) * perCoreUbFactor;

    platformUbFactor_ = ubSize_ / DOUBLE_BUFFER;
    platformUbFactor_ = platformUbFactor_ - RESERVED_UB_SIZE * DOUBLE_INPUT;
    int64_t maxUbAvailable = platformUbFactor_ / DOUBLE_INPUT / inputDtypeSize_;

    // split ub
    ubFactor_ = perCoreUbFactor >= maxUbAvailable ? maxUbAvailable : perCoreUbFactor;
    perCoreLoopTimes_ = Ops::Base::CeilDiv(perCoreUbFactor, ubFactor_);
    perCoreTailFactor_ = perCoreUbFactor - (perCoreLoopTimes_ - 1) * ubFactor_ ;

    tailCoreLoopTimes_ = Ops::Base::CeilDiv(tailCoreUbFactor, ubFactor_);
    tailCoreTailFactor_ = tailCoreUbFactor - (tailCoreLoopTimes_ - 1) * ubFactor_ ;

    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus TensorEqualTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t TensorEqualTiling::GetTilingKey() const
{
    OP_LOGD(opName_, "TensorEqualTiling GetTilingKey .");
    return tilingKey_;
}

// 6、计算Workspace 大小
ge::graphStatus TensorEqualTiling::GetWorkspaceSize()
{
    OP_LOGD(opName_, "TensorEqualTiling GetWorkspaceSize .");
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SAVE_SIZE;

    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus TensorEqualTiling::PostTiling()
{
    OP_LOGD(opName_, "TensorEqualTiling PostTiling.");

    auto res = context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS),
        OP_LOGE(opName_, "SetBlockDim failed."),
        return ge::GRAPH_FAILED);
    context_->SetScheduleMode(1); // SyncAll need set
    context_->SetTilingKey(tilingKey_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void TensorEqualTiling::SetTilingData()
{
    OP_LOGD(opName_, "TensorEqualTiling SetTilingData.");
    tilingData_.set_inputShapeSize(inputShapeSize_);
    tilingData_.set_inputDtypeSize(inputDtypeSize_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_perCoreLoopTimes(perCoreLoopTimes_);
    tilingData_.set_tailCoreLoopTimes(tailCoreLoopTimes_);
    tilingData_.set_perCoreTailFactor(perCoreTailFactor_);
    tilingData_.set_tailCoreTailFactor(tailCoreTailFactor_);
    tilingData_.set_tilingKey(tilingKey_);
}

void TensorEqualTiling::DumpTilingInfo()
{
    std::ostringstream info;

    info << "inputShapeSize: " << tilingData_.get_inputShapeSize();
    info << ", inputDtypeSize: " << tilingData_.get_inputDtypeSize();
    info << ", usedCoreNum: " << tilingData_.get_usedCoreNum();
    info << ", ubSize: " << tilingData_.get_ubSize();
    info << ", ubFactor: " << tilingData_.get_ubFactor();
    info << ", perCoreLoopTimes: " << tilingData_.get_perCoreLoopTimes();
    info << ", tailCoreLoopTimes: " << tilingData_.get_tailCoreLoopTimes();
    info << ", perCoreTailFactor: " << tilingData_.get_perCoreTailFactor();
    info << ", tailCoreTailFactor: " << tilingData_.get_tailCoreTailFactor();
    info << ", tilingKey: " << tilingData_.get_tilingKey();
    
    OP_LOGI(opName_, "%s", info.str().c_str());
}

ge::graphStatus Tiling4TensorEqual(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TensorEqual", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const TensorEqualCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context->GetNodeName(), "Tiling TensorEqual start");
    TensorEqualTiling tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4TensorEqual(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TensorEqual", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Tiling Prepare For TensorEqual start");

    auto compileInfo = context->GetCompiledInfo<TensorEqualCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
        OP_LOGE(context->GetNodeName(), "Failed to get ub size"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the TensorEqual op.
IMPL_OP_OPTILING(TensorEqual)
    .Tiling(Tiling4TensorEqual)
    .TilingParse<TensorEqualCompileInfo>(TilingPrepare4TensorEqual);
} // namespace optiling