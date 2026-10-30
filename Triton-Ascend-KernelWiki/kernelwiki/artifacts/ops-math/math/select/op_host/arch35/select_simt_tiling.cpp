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
 * \file    select_simt_tiling.cpp
 * \brief   select_simt_tiling source file
 */

#include "select_simt_tiling.h"
#include <graph/utils/type_utils.h>
#include "select_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/select/op_kernel/arch35/select_struct.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint64_t SELECT_COMMON_TILING_PRIORITY = 0;
static constexpr int64_t NUM_TWO = 2;
static constexpr int32_t CONDITION_IDX = 0;
#ifdef DAVID_FPGA
const static int64_t SMALL_CASE_THREAD_NUM = 64;
#else
const static int64_t SMALL_CASE_THREAD_NUM = 128;
#endif
static constexpr int32_t X1_IDX = 1;
static constexpr int32_t X2_IDX = 2;
static constexpr int32_t MIN_SIZE = 512;
static constexpr int32_t DCACHE_SIZE = 128 * 1024;
static constexpr uint64_t WORKSPACE_SIZE = 32;
static constexpr uint64_t BLOCK_NUM = 1;
static constexpr int64_t INPUT_DTYPE_B64 = 8;
static constexpr int64_t INPUT_DTYPE_B32 = 4;
static constexpr int64_t INPUT_DTYPE_B16 = 2;
static constexpr int64_t TILING_KEY = 999;
static constexpr int64_t USER_DEF = 1;


ge::graphStatus SelectSimtTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(SelectCheckInputDtype(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "SelectCheckInputDtype error!"),
        return ge::GRAPH_FAILED);

    vector<gert::Shape> inputShapes;
    OP_CHECK_IF(InferSelectShape(context_, inputShapes) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "InferSelectShape error!"),
        return ge::GRAPH_FAILED);

    conditionShape_ = inputShapes[CONDITION_IDX];
    x1Shape_ = inputShapes[X1_IDX];
    x2Shape_ = inputShapes[X2_IDX];
    
    return ge::GRAPH_SUCCESS;
}

bool SelectSimtTiling::IsMatchAB()
{
    int64_t pos = -1;
    int64_t conditionDimNum = conditionShape_.GetDimNum();
    for (int64_t i = 0; i < conditionDimNum; i++) {
        if (conditionShape_.GetDim(i) == 1){
            pos = i;
            break;
        }
        aSize_ *= conditionShape_.GetDim(i);
    }
    if (pos < 0 || pos >= conditionDimNum){
        return false;
    }
    for (int64_t i = pos; i < conditionDimNum; i++) {
        if (conditionShape_.GetDim(i) != 1){
            return false;
        }
        bSize_ *= x1Shape_.GetDim(i);
    }
    return true;
}

bool SelectSimtTiling::XDtypeImprove()
{
    auto x1Dtype = context_->GetInputDesc(X1_IDX)->GetDataType();
    int64_t xDtypeSize = ge::GetSizeByDataType(x1Dtype);
    if (bSize_ * xDtypeSize < MIN_SIZE) {
        return false;
    }
    if (xDtypeSize == INPUT_DTYPE_B64) {
        return true;
    }
    if ((xDtypeSize < INPUT_DTYPE_B64) && (bSize_ * xDtypeSize % INPUT_DTYPE_B64) == 0) {
        OP_LOGD(context_->GetNodeName(), "XDtypeImprove lastAxisBytes %ld, improve to INPUT_DTYPE_B64", bSize_);  
        bSize_ /= (INPUT_DTYPE_B64 / xDtypeSize);
        return true;
    }
    
    return false;
}

bool SelectSimtTiling::IsCapable()
{
    if (!IsMatchAB()){
        return false;
    }
    if (!XDtypeImprove()){
        return false;
    }
    return true;
}

ge::graphStatus SelectSimtTiling::DoOpTiling()
{
    ySize_ = aSize_ * bSize_;
    SelectSimtTilingData* tilingData = context_->GetTilingData<SelectSimtTilingData>();
    tilingData->aSize = aSize_;
    tilingData->bSize = bSize_;
    while ((threadNum_ >= NUM_TWO * SMALL_CASE_THREAD_NUM) && (Ops::Base::CeilDiv(ySize_, threadNum_) < (aivNum_ / NUM_TWO))) {
        threadNum_ = threadNum_ / NUM_TWO;
    }
    tilingData->threadNum = threadNum_;
    int64_t perCoreElements = Ops::Base::CeilDiv(ySize_, aivNum_);
    if (ySize_ < threadNum_) {
        tilingData->needCoreNum = 1;
        tilingData->perCoreElements = ySize_;
        tilingData->lastCoreElements = ySize_;
        needCoreNum_ = 1;
    } else {
        perCoreElements = (perCoreElements + threadNum_ - 1) / threadNum_ * threadNum_;  // 对齐到threadNum_的倍数
        needCoreNum_ = Ops::Base::CeilDiv(ySize_, perCoreElements);
        int64_t lastCoreElements = ySize_ - perCoreElements * (needCoreNum_ - 1);
        tilingData->needCoreNum = needCoreNum_;
        tilingData->perCoreElements = perCoreElements;
        tilingData->lastCoreElements = lastCoreElements;
    }

    context_->SetBlockDim(needCoreNum_);
    
    tilingKey = GET_TPL_TILING_KEY(TILING_KEY, USER_DEF);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SelectSimtTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus SelectSimtTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectSimtTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SelectSimtTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SelectCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        aivNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(context_->GetNodeName(), "Get ubSize form compileInfo is: %ld", ubSize_);
        OP_LOGD(context_->GetNodeName(), "Get aivNum form compileInfo is: %ld", aivNum_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
        OP_LOGD(context_->GetNodeName(), "Get ubSize form ascendcPlatform is: %ld", ubSize_);
        OP_LOGD(context_->GetNodeName(), "Get aivNum form ascendcPlatform is: %ld", aivNum_);
    }
    aicoreParams_.numBlocks = aivNum_;
    return ge::GRAPH_SUCCESS;
}
 

 REGISTER_OPS_TILING_TEMPLATE(Select, SelectSimtTiling, SELECT_COMMON_TILING_PRIORITY);
 }   // namespace optiling 