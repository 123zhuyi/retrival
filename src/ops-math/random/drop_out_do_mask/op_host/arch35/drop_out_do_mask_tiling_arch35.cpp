/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file drop_out_do_mask_tiling_arch35.cpp
 * \brief
 */

#include "drop_out_do_mask_tiling_arch35.h"
#include <limits>
#include <string>
#include "tiling/tiling_api.h"
#include "drop_out_do_mask_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"

namespace optiling {
using namespace Ops::Base;

constexpr int64_t X_IDX = 0;
constexpr int64_t MASK_IDX = 1;
constexpr int64_t KEEP_PROB_IDX = 2;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t BIT_ALIGN = 128;
constexpr int64_t ALIGN256 = 256;
constexpr int64_t UINT8_BIT_NUMBER = 8;
constexpr int64_t CACHE_LINE = 128;
constexpr int64_t RESERVED_UB_SIZE = 8192;            // 8K
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16M
constexpr int64_t UB_MIN_FACTOR = 2048;

static const std::set<ge::DataType> DROP_SUPPORTED_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

bool DropOutDoMaskTiling::IsCapable()
{
    return true;
}

ge::graphStatus DropOutDoMaskTiling::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<DropOutDoMaskCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutDoMaskTiling::CheckInputShape()
{
    auto xShapePtr = context_->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    allAxis_ = xShape.GetShapeSize();

    auto keepProbShapePtr = context_->GetInputShape(KEEP_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keepProbShapePtr);
    auto& keepProbShape = Ops::Base::EnsureNotScalar(keepProbShapePtr->GetStorageShape());
    int64_t keepProbAxis = keepProbShape.GetShapeSize();
    if (keepProbAxis != 1) {
        std::string valueStr = std::to_string(keepProbAxis);
        std::string reasonMsg = "size of keep_prob has to be 1";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "input keep_prob", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutDoMaskTiling::GetShapeAttrsInfo()
{
    auto xPtr = context_->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
    dType_ = xPtr->GetDataType();
    if (DROP_SUPPORTED_DTYPE.find(dType_) == DROP_SUPPORTED_DTYPE.end()) {
        std::string valueStr = ToString(dType_);
        std::string reasonMsg = "x dtype only support float32, float16, bfloat16";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto maskPtr = context_->GetInputDesc(MASK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskPtr);
    auto maskDtype = maskPtr->GetDataType();
    bool dtypeInValid = (maskDtype != ge::DT_UINT8 && maskDtype != ge::DT_UINT1);
    if (dtypeInValid) {
        std::string valueStr = ToString(maskDtype);
        std::string reasonMsg = "mask dtype only support uint8, uint1 currently";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "input tensor mask", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto probPtr = context_->GetInputDesc(KEEP_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probPtr);
    auto probPtrDtype = probPtr->GetDataType();
    if (probPtrDtype != dType_) {
        std::string valueStr = ToString(dType_) + " and " + ToString(probPtrDtype);
        std::string reasonMsg = "keep_prob dtype must be equal to x dtype";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "input keep_prob and input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "input shape check failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutDoMaskTiling::DoOpTiling()
{
    typeSize_ = ge::GetSizeByDataType(dType_);
    OP_CHECK_IF(typeSize_ <= 0, OP_LOGE(context_->GetNodeName(), "get dataType size fail."), return ge::GRAPH_FAILED);
    // total: ub/db
    // used: x*typesize (x), x/8 (mask), x*typesize(out)
    int64_t ubBlock = GetUbBlockSize(context_);
    ubFactor_ = (ubSize_ - RESERVED_UB_SIZE - ubBlock) / DB_BUFFER * UINT8_BIT_NUMBER /
                (UINT8_BIT_NUMBER * typeSize_ + 1 + UINT8_BIT_NUMBER * typeSize_);
    ubFactor_ = Ops::Base::FloorAlign(ubFactor_, ALIGN256);
    normBlockData_ = Ops::Base::CeilDiv(allAxis_, totalCoreNum_);
    normBlockData_ = Ops::Base::CeilAlign(normBlockData_, ALIGN256);
    normBlockData_ = std::max(normBlockData_, UB_MIN_FACTOR);
    usedCoreNum_ = Ops::Base::CeilDiv(allAxis_, normBlockData_);
    tailBlockData_ = allAxis_ - (usedCoreNum_ - 1) * normBlockData_;
    int64_t normBlockLoop = Ops::Base::CeilDiv(normBlockData_, ubFactor_);
    int64_t normBlockTail = normBlockData_ - (normBlockLoop - 1) * ubFactor_;
    int64_t tailBlockLoop = Ops::Base::CeilDiv(tailBlockData_, ubFactor_);
    int64_t tailBlockTail = tailBlockData_ - (tailBlockLoop - 1) * ubFactor_;

    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_normBlockData(normBlockData_);
    tilingData_.set_tailBlockData(tailBlockData_);
    tilingData_.set_normBlockLoop(normBlockLoop);
    tilingData_.set_normBlockTail(normBlockTail);
    tilingData_.set_tailBlockLoop(tailBlockLoop);
    tilingData_.set_tailBlockTail(tailBlockTail);
    tilingData_.set_epsilon(std::numeric_limits<double>::epsilon());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutDoMaskTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t DropOutDoMaskTiling::GetTilingKey() const
{
    uint64_t tilingKey = 100;
    return tilingKey;
}

ge::graphStatus DropOutDoMaskTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutDoMaskTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void DropOutDoMaskTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "usedCoreNum: " << tilingData_.get_usedCoreNum();
    info << ", normBlockData: " << tilingData_.get_normBlockData();
    info << ", tailBlockData: " << tilingData_.get_tailBlockData();
    info << ", ubFactor: " << tilingData_.get_ubFactor();
    info << ", normBlockLoop:" << tilingData_.get_normBlockLoop();
    info << ", normBlockTail: " << tilingData_.get_normBlockTail();
    info << ", tailBlockLoop: " << tilingData_.get_tailBlockLoop();
    info << ", tailBlockTail: " << tilingData_.get_tailBlockTail();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus DropOutDoMaskTilingForAscendC(gert::TilingContext* context)
{
    DropOutDoMaskTiling tiling(context);
    return tiling.DoTiling();
}

REGISTER_OPS_TILING_TEMPLATE(DropOutDoMask, DropOutDoMaskTiling, 1000);
} // namespace optiling
