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
 * \file bincount_tiling.cc
 * \brief bincount_tiling file
 */

#include <iostream>
#include <cstring>
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"
#include "bincount_tiling.h"
#include "util/math_util.h"
#include "util/const_util.h"

namespace optiling {
ge::graphStatus BincountTiling::Init()
{
    OP_LOGD(context_->GetNodeName(), "BincountTiling init enter.");
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<BincountTilingData>();
        OP_CHECK_IF(
            tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        (memset_s(tilingData_, sizeof(BincountTilingData), 0, sizeof(BincountTilingData)) != EOK),
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed"), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "BincountTiling init exit.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BincountTiling::BincountGetPlatformData(const AscendCBincountCompileInfo* compileInfo)
{
    coreNum_ = compileInfo->totalCoreNum;
    ubSize_ = static_cast<int64_t>(compileInfo->totalUbSize - SIMD_SIMT_DCACHE_SIZE);
    isDetermine_ = context_->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGI(
        context_->GetNodeName(), "BincountGetPlatformData ubSize is %ld, coreNum_ is %ld, isDetermine is %ld", ubSize_,
        coreNum_, isDetermine_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BincountTiling::CheckShape()
{
    auto arrShape = context_->GetInputShape(INPUT_IDX_ARRAY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, arrShape);
    OP_CHECK_IF(
        arrShape->GetStorageShape().GetDimNum() != DIM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "array", std::to_string(arrShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);

    auto weightsShape = context_->GetInputShape(INPUT_IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsShape);
    OP_CHECK_IF(
        weightsShape->GetStorageShape().GetDimNum() != DIM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "weights", std::to_string(weightsShape->GetStorageShape().GetDimNum()).c_str(),
            "1D"),
        return ge::GRAPH_FAILED);

    auto binsShape = context_->GetOutputShape(OUTPUT_IDX_BINS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, binsShape);
    OP_CHECK_IF(
        binsShape->GetStorageShape().GetDimNum() != DIM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "bins", std::to_string(binsShape->GetStorageShape().GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BincountTiling::CheckDtype()
{
    auto inputDesc = context_->GetInputDesc(INPUT_IDX_ARRAY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto arrayDataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        arrayDataType != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "array", ge::TypeUtils::DataTypeToSerialString(arrayDataType).c_str(), "INT32"),
        return ge::GRAPH_FAILED);

    auto sizeDesc = context_->GetInputDesc(INPUT_IDX_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sizeDesc);
    auto sizeDataType = sizeDesc->GetDataType();
    OP_CHECK_IF(
        sizeDataType != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "size", ge::TypeUtils::DataTypeToSerialString(sizeDataType).c_str(), "INT32"),
        return ge::GRAPH_FAILED);

    auto weightsDesc = context_->GetInputDesc(INPUT_IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsDesc);
    auto weightsDataType = weightsDesc->GetDataType();
    OP_CHECK_IF(
        !(weightsDataType == ge::DT_FLOAT || weightsDataType == ge::DT_INT32 || weightsDataType == ge::DT_INT64),
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "weights", ge::TypeUtils::DataTypeToSerialString(weightsDataType).c_str(),
            "FLOAT, INT32 or INT64"),
        return ge::GRAPH_FAILED);

    auto outputBinsDesc = context_->GetOutputDesc(OUTPUT_IDX_BINS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputBinsDesc);
    auto outputBinsDataType = outputBinsDesc->GetDataType();
    OP_CHECK_IF(
        !(outputBinsDataType == ge::DT_FLOAT || outputBinsDataType == ge::DT_INT32 ||
          outputBinsDataType == ge::DT_INT64),
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "bins", ge::TypeUtils::DataTypeToSerialString(outputBinsDataType).c_str(),
            "FLOAT, INT32 or INT64"),
        return ge::GRAPH_FAILED);

    if (outputBinsDataType != weightsDataType) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(weightsDataType) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(outputBinsDataType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "weights and bins", dtypeMsg.c_str(),
            "The dtypes of output parameter bins and input parameter weights must be the same");
        return ge::GRAPH_FAILED;
    }
    binsDataType_ = outputBinsDataType;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BincountTiling::CheckInputParams()
{
    auto inputArrShape = context_->GetInputShape(INPUT_IDX_ARRAY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputArrShape);
    arrayShapeSize_ = inputArrShape->GetStorageShape().GetShapeSize();

    auto inputSizeShape = context_->GetInputShape(INPUT_IDX_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSizeShape);
    gert::Shape sizeShape = Ops::Base::EnsureNotScalar(inputSizeShape->GetStorageShape());
    int32_t sizeDims = sizeShape.GetDimNum();
    OP_CHECK_IF(
        sizeDims != DIM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "size", std::to_string(sizeDims).c_str(), "1D"),
        return ge::GRAPH_FAILED);
    int64_t sizeContent = 0;
    OP_CHECK_IF(
        (!Ops::Base::GetConstInt(context_, DIM_1, sizeContent)), OP_LOGE(context_->GetNodeName(), "get size failed"),
        return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "sizeContent is %ld", sizeContent);
    OP_CHECK_IF(
        sizeContent < 0,
        OP_LOGE_FOR_INVALID_VALUE(
            context_->GetNodeName(), "size", std::to_string(sizeContent).c_str(), "greater than or equal to 0"),
        return ge::GRAPH_FAILED);
    inputSize_ = sizeContent;

    auto inputWeightsShape = context_->GetInputShape(INPUT_IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputWeightsShape);
    weightsShapeSize_ = inputWeightsShape->GetStorageShape().GetShapeSize();
    if (weightsShapeSize_ != 0) {
        isWeight_ = WEIGHT;
    }
    OP_LOGI(context_->GetNodeName(), "weightsShapeSize_ is %ld, isWeight_ is %ld", weightsShapeSize_, isWeight_);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_BINS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    int64_t outputBinsLength = outputShape->GetStorageShape().GetShapeSize();
    OP_LOGI(context_->GetNodeName(), "outputBinsLength is %ld", sizeContent);
    if (outputBinsLength != inputSize_) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "bins", std::to_string(outputBinsLength).c_str(),
            ("The shape size of parameter bins must be equal to the shape size of parameter size " + std::to_string(inputSize_)).c_str());
        return ge::GRAPH_FAILED;
    }
    binsShapeSize_ = outputBinsLength;

    return ge::GRAPH_SUCCESS;
}

inline bool BincountTiling::IsMatchSimtBatchLoadMode()
{
    return (weightsShapeSize_ + arrayShapeSize_) > binsShapeSize_ / GM_ATOMIC_ADD_FACTOR;
}

ge::graphStatus BincountTiling::ComputeTilingStrategy()
{
    OP_LOGD(context_->GetNodeName(), "ComputeTilingStrategy enter.");
    switch (binsDataType_) {
        case ge::DT_FLOAT:
            outputDtype_ = OUTPUT_DTYPE_FLOAT;
            ubNumCanUse_ = static_cast<int64_t>(ubSize_ / SIZE_DTYPE_FLOAT); // numbers UB can processed once
            break;
        case ge::DT_INT32:
            outputDtype_ = OUTPUT_DTYPE_INT32;
            ubNumCanUse_ = static_cast<int64_t>(ubSize_ / SIZE_DTYPE_INT32);
            break;
        case ge::DT_INT64:
            outputDtype_ = OUTPUT_DTYPE_INT64;
            ubNumCanUse_ = static_cast<int64_t>(ubSize_ / SIZE_DTYPE_INT64);
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    if (isDetermine_) {
        schId_ = SCH_ID_SIMT_DETERMIN;
        return ComputeTilingSimtDetermine();
    }

    if (binsDataType_ == ge::DT_INT64) {
        schId_ = SCH_ID_SIMT_NOT_FULL_LOAD;
        return ComputeTilingSimtNotDetermine();
    }

    if (binsShapeSize_ < ubNumCanUse_) {
        schId_ = SCH_ID_SIMT_FULL_LOAD;
    } else if (IsMatchSimtBatchLoadMode()) {
        int64_t ubLoopNum = Ops::Base::CeilDiv(binsShapeSize_, ubNumCanUse_);
        if (ubLoopNum <= MAX_UB_LOOP_FOR_BATH) {
            schId_ = SCH_ID_SIMT_BATCH_LOAD; // also atomicAdd on ub
        } else {
            schId_ = SCH_ID_SIMT_NOT_FULL_LOAD;
        }
    } else {
        schId_ = SCH_ID_SIMT_NOT_FULL_LOAD;
    }

    return ComputeTilingSimtNotDetermine();
}

ge::graphStatus BincountTiling::ComputeTilingSimtNotDetermine()
{
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtNotDetermine enter.");
    ubLoopNum_ = Ops::Base::CeilDiv(binsShapeSize_, ubNumCanUse_);
    formerLength_ = Ops::Base::CeilDiv(arrayShapeSize_, coreNum_); // split core by input array
    OP_CHECK_IF(
        formerLength_ == 0, OP_LOGE(context_->GetNodeName(), "formerLength_ must not be 0."), return ge::GRAPH_FAILED);
    needXCoreNum_ = Ops::Base::CeilDiv(arrayShapeSize_, formerLength_); // we need X core number to process input
    tailLength_ = arrayShapeSize_ - (needXCoreNum_ - 1) * formerLength_; // remained data

    clearYFactor_ = Ops::Base::CeilDiv(binsShapeSize_, coreNum_);
    OP_CHECK_IF(
        clearYFactor_ == 0, OP_LOGE(context_->GetNodeName(), "clearYFactor_ must not be 0."), return ge::GRAPH_FAILED);
    clearYCoreNum_ = Ops::Base::CeilDiv(binsShapeSize_, clearYFactor_);
    clearYTail_ = binsShapeSize_ - (clearYCoreNum_ - 1) * clearYFactor_;
    needCoreNum_ = std::max(needXCoreNum_, clearYCoreNum_);
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtNotDetermine end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BincountTiling::ComputeTilingSimtDetermine()
{
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtDetermine enter.");
    // deterministic
    binsFormerLength_ = Ops::Base::CeilDiv(binsShapeSize_, coreNum_); // split core by output bins
    OP_CHECK_IF(
        binsFormerLength_ == 0, OP_LOGE(context_->GetNodeName(), "binsFormerLength_ must not be 0."),
        return ge::GRAPH_FAILED);
    needBinsCoreNum_ = Ops::Base::CeilDiv(binsShapeSize_, binsFormerLength_); // we need core number to process output
    binsTailLength_ = binsShapeSize_ - (needBinsCoreNum_ - 1) * binsFormerLength_; // remained data
    needCoreNum_ = needBinsCoreNum_;
    OP_LOGD(context_->GetNodeName(), "ComputeTilingSimtDetermine end.");
    return ge::GRAPH_SUCCESS;
}

void BincountTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "Bincount tilingData needCoreNum_ is %ld, size is %ld,"
        "ubNumCanUse is %ld, ubLoopNum is %ld, needXCoreNum is %ld, formerLength is %ld, tailLength is %ld,"
        "clearYCoreNum is %ld, clearYFactor is %ld, clearYTail is %ld, binsFormerLength is %ld"
        "needBinsCoreNum is %ld, binsTailLength is %ld",
        needCoreNum_, tilingData_->size, tilingData_->ubNumCanUse, tilingData_->ubLoopNum, tilingData_->needXCoreNum,
        tilingData_->formerLength, tilingData_->tailLength, tilingData_->clearYCoreNum, tilingData_->clearYFactor,
        tilingData_->clearYTail, tilingData_->binsFormerLength, tilingData_->needBinsCoreNum,
        tilingData_->binsTailLength);
    return;
}

ge::graphStatus BincountTiling::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "SetTilingData enter.");
    tilingData_->size = inputSize_;
    tilingData_->ubNumCanUse = ubNumCanUse_;
    tilingData_->ubLoopNum = ubLoopNum_;
    tilingData_->needXCoreNum = needXCoreNum_;
    tilingData_->formerLength = formerLength_;
    tilingData_->tailLength = tailLength_;
    tilingData_->clearYCoreNum = clearYCoreNum_;
    tilingData_->clearYFactor = clearYFactor_;
    tilingData_->clearYTail = clearYTail_;
    tilingData_->binsFormerLength = binsFormerLength_;
    tilingData_->needBinsCoreNum = needBinsCoreNum_;
    tilingData_->binsTailLength = binsTailLength_;
    tilingData_->arraySize = arrayShapeSize_;
    OP_LOGI(
        context_->GetNodeName(), "schId is %ld, outputDtype is %ld, isWeight is %ld", schId_, outputDtype_, isWeight_);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId_, outputDtype_, isWeight_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is %ld", tilingKey);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(needCoreNum_); // double check
    context_->SetLocalMemorySize(ubSize_);
    context_->SetScheduleMode(1); // SyncAll need set
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Bincount(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const AscendCBincountCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    BincountTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.BincountGetPlatformData(compileInfo) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "BincountGetPlatformData return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.CheckShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CheckShape return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.CheckDtype() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CheckDtype return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.CheckInputParams() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CheckInputParams return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.ComputeTilingStrategy() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "ComputeTilingStrategy return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.SetTilingData() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "SetTilingData return failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Bincount(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4Bincount running.");
    auto compileInfo = context->GetCompiledInfo<AscendCBincountCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "coreNum is invalid, must greater than zero"),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->totalUbSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalUbSize <= 0), OP_LOGE(context->GetNodeName(), "ubSize is invalid, must greater than zero"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "totalUbSize is %lu.", compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Bincount).Tiling(Tiling4Bincount).TilingParse<AscendCBincountCompileInfo>(TilingPrepare4Bincount);

} // namespace optiling