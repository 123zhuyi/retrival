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
 * \file random_uniform_int_v2_tiling_arch35.cpp
 * \brief
 */

#include <random>
#include <iostream>
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "random_uniform_int_v2_tiling_arch35.h"
#include  "random/random_common/op_host/arch35/random_tiling_base.h"
#include "op_host/math_tiling_templates_registry.h"
#include "register/op_def_registry.h"

namespace optiling {

template <typename T>
ge::graphStatus RandomUniformIntV2Tiling::GetIntValue(const gert::Tensor *constTensor, gert::Shape &constShape)
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetIntValue begin.");
    const T *constValue = constTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, constValue);
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetIntValue constNum=%zu", constNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformIntV2Tiling::GetIntValueByDtype(const gert::Tensor *constTensor, gert::Shape &constShape,
                                                             ge::DataType dType)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (dType == ge::DataType::DT_INT32) {
        ret = GetIntValue<int32_t>(constTensor, constShape);
    } else {
        ret = GetIntValue<int64_t>(constTensor, constShape);
    }
    return ret;
}

ge::graphStatus RandomUniformIntV2Tiling::GetMinAndMaxValue()
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetMinAndMaxValue begin.");
    auto minDesc = context_->GetRequiredInputDesc(IN_MIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minDesc);
    minDtype_ = minDesc->GetDataType();
    if ((minDtype_ != ge::DataType::DT_INT32) && (minDtype_ != ge::DataType::DT_INT64)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "min",
            Ops::Base::ToString(minDtype_).c_str(), "dtype must be in [DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }

    auto minTensor = context_->GetRequiredInputTensor(IN_MIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minTensor);
    auto minTensorSize = static_cast<int64_t>(minTensor->GetShapeSize());
    if (minTensorSize != 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "min data shape_size",
            std::to_string(minTensorSize).c_str(), "shape_size must be 1");
        return ge::GRAPH_FAILED;
    }
    gert::Shape minShape;
    auto ret = GetIntValueByDtype(minTensor, minShape, minDtype_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "min GetIntValueByDtype failed."), return ge::GRAPH_FAILED);
    lo_ = static_cast<int64_t>(minShape.GetDim((0)));

    auto maxDesc = context_->GetRequiredInputDesc(IN_MAX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxDesc);
    auto maxDtype = maxDesc->GetDataType();
    if (maxDtype != minDtype_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "max, min",
            (Ops::Base::ToString(maxDtype) + ", " + Ops::Base::ToString(minDtype_)).c_str(),
            "max dtype must be same as min dtype");
        return ge::GRAPH_FAILED;
    }

    auto maxTensor = context_->GetRequiredInputTensor(IN_MAX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxTensor);
    auto maxTensorSize = static_cast<int64_t>(maxTensor->GetShapeSize());
    if (maxTensorSize != 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "max data shape_size",
            std::to_string(maxTensorSize).c_str(), "shape_size must be 1");
        return ge::GRAPH_FAILED;
    }
    gert::Shape maxShape;
    ret = GetIntValueByDtype(maxTensor, maxShape, maxDtype);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "max GetIntValueByDtype failed."), return ge::GRAPH_FAILED);
    const int64_t maxTensorValue = static_cast<int64_t>(maxShape.GetDim((0)));
    if (maxTensorValue <= lo_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "max, min",
            (std::to_string(maxTensorValue) + ", " + std::to_string(lo_)).c_str(),
            "max value must be greater than min value");
        return ge::GRAPH_FAILED;
    }
    range_ = static_cast<uint64_t>(maxTensorValue) - static_cast<uint64_t>(lo_);

    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetMinAndMaxValue end.");
    return ge::GRAPH_SUCCESS;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus RandomUniformIntV2Tiling::GetPlatformInfo()
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling GetPlatformInfo.");
    auto compileInfo = static_cast<const RandomUniformIntV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetPlatformInfo ubSize_=%d, totalCoreNum_=%d", ubSize_, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus RandomUniformIntV2Tiling::GetShapeAttrsInfo()
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetShapeAttrsInfo begin.");
    OP_CHECK_IF(GetInputInfo(), 
        OP_LOGE(opName_, "GetInputInfo failed!"), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetOutputInfo(), 
        OP_LOGE(opName_, "GetOutputInfo failed!"), return ge::GRAPH_FAILED);
    
    if (shapeSize_ != outputSize_) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName_, "input shape, output",
            (std::to_string(shapeSize_) + ", " + std::to_string(outputSize_)).c_str(),
            "input shape size must be equal to output size");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(GetAttrInfo(), 
        OP_LOGE(opName_, "GetAttrInfo failed!"), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformIntV2Tiling::GetInputInfo()
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetInputInfo begin.");
    auto shapeDesc = context_->GetRequiredInputDesc(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeDesc);
    auto shapeDtype = shapeDesc->GetDataType();
    if ((shapeDtype != ge::DataType::DT_INT32) && (shapeDtype != ge::DataType::DT_INT64)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input shape",
            Ops::Base::ToString(shapeDtype).c_str(), "dtype must be in [DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }

    auto input1Shape = context_->GetInputShape(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Shape);
    uint32_t shapeDimNum = input1Shape->GetStorageShape().GetDimNum();
    if (shapeDimNum != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "input shape",
            std::to_string(shapeDimNum).c_str(), "must be 1D tensor");
        return ge::GRAPH_FAILED;
    }

    auto shapeTensor = context_->GetRequiredInputTensor(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeTensor);
    gert::Shape constShape;
    auto ret = GetIntValueByDtype(shapeTensor, constShape, shapeDtype);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
        OP_LOGE(opName_, "input shape GetIntValueByDtype failed."), return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "RandomUniformIntV2Tiling::GetInputInfo get shapeTensor end.");

    uint32_t shapeRank = constShape.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        shapeSize_ *= static_cast<int64_t>(constShape.GetDim(idx));
    }
    if (shapeSize_ == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "input shape",
            std::to_string(shapeSize_).c_str(), "shape_size must not be 0");
        return ge::GRAPH_FAILED;
    }

    auto offsetDesc = context_->GetInputDesc(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
    auto offsetDtype = offsetDesc->GetDataType();
    if (offsetDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input offset",
            Ops::Base::ToString(offsetDtype).c_str(), "dtype must be DT_INT64");
        return ge::GRAPH_FAILED;
    }
        
    auto offsetTensor = context_->GetInputTensor(IN_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetTensor);
    auto offsetTensorSize = static_cast<int64_t>(offsetTensor->GetShapeSize());   // 验证
    if (offsetTensorSize != 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "input offset shape_size",
            std::to_string(offsetTensorSize).c_str(), "shape_size must be 1");
        return ge::GRAPH_FAILED;
    }

    ret = GetMinAndMaxValue();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "GetMinAndMaxValue failed."), return ge::GRAPH_FAILED);
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetInputInfo end.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformIntV2Tiling::GetOutputInfo()
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetOutputInfo begin.");
    auto outDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    outDtype_ = outDesc->GetDataType();
    if (outDtype_ != minDtype_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "out, min",
            (Ops::Base::ToString(outDtype_) + ", " + Ops::Base::ToString(minDtype_)).c_str(),
            "out dtype must be same as min dtype");
        return ge::GRAPH_FAILED;
    }

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outTensor = outputShape->GetStorageShape();
    outputSize_ = outTensor.GetShapeSize();
    if (outputSize_ == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "output shape_size",
            std::to_string(outputSize_).c_str(), "shape_size must not be 0");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetOutputInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RandomUniformIntV2Tiling::GetAttrInfo()
{
    OP_LOGI(opName_, "RandomUniformIntV2Tiling::GetAttrInfo begin.");
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto* seedAttr = attrs->GetAttrPointer<int64_t>(ATTR_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedAttr);
    const auto* seed2Attr = attrs->GetAttrPointer<int64_t>(ATTR_SEED2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seed2Attr);
    
    seed_ = *seedAttr;
    seed2_ = *seed2Attr;
    if (seed_ == 0 && seed2_ == 0) {
        seed_ = static_cast<int64_t>(New64());
        seed2_ = static_cast<int64_t>(New64());
    }
    OP_LOGI(opName_, "RandomUniformIntV2Tiling seed_ is %ld, seed2_ is %ld", seed_, seed2_);
    return ge::GRAPH_SUCCESS;
}

bool RandomUniformIntV2Tiling::IsCapable()
{
    return true;
}

void RandomUniformIntV2Tiling::SetTilingData()
{
    RandomUniformIntV2TilingData4RegBase* tilingData = context_->GetTilingData<RandomUniformIntV2TilingData4RegBase>();
    tilingData->blockNum = blockNum_;
    tilingData->normalCoreProNum = normalCoreProNum_;
    tilingData->tailCoreProNum = tailCoreProNum_;
    tilingData->singleUbSize = singleUbSize_;
    tilingData->seed = seed_;
    tilingData->seed2 = seed2_;
    tilingData->outputSize = outputSize_;
    tilingData->range = range_;
    tilingData->lo = lo_;
}

void RandomUniformIntV2Tiling::DoBlockTiling() 
{
    outputDtypeSize_ = ge::GetSizeByDataType(outDtype_);
    if (outputDtypeSize_ == 0) {
        return;
    }

    auto coreAlignFactor = CORE_ALIGN_SIZE / outputDtypeSize_;
    auto blockFactor = Ops::Base::CeilDiv(outputSize_, totalCoreNum_);
    auto blockAlignFactor = Ops::Base::CeilDiv(blockFactor, coreAlignFactor) * coreAlignFactor;
    auto minTilingSize = MIN_TILING_SIZE;
    normalCoreProNum_ = std::max(static_cast<uint32_t>(blockAlignFactor), minTilingSize);
    blockNum_ = Ops::Base::CeilDiv(outputSize_, normalCoreProNum_);
    tailCoreProNum_ = outputSize_ - normalCoreProNum_ * (blockNum_ - 1);
    return;
}

void RandomUniformIntV2Tiling::UbTiling() 
{
  // quarterUbSize: 2 for double buffer; coefVal for temp RNG, philox temp buff need uint32 to int32/int64
  int64_t coefVal = DOUBLE_BUFFER;
  auto quarterUbSize = (ubSize_ - DCACHE_SIZE) / (DOUBLE_BUFFER + coefVal);
  auto ubBlockSize = static_cast<int32_t>(Ops::Base::GetUbBlockSize(context_));
  auto alignFactor = ubBlockSize / outputDtypeSize_;
  singleUbSize_ = (quarterUbSize / outputDtypeSize_ / alignFactor) * alignFactor;
}

// 3、计算数据切分TilingData
ge::graphStatus RandomUniformIntV2Tiling::DoOpTiling()
{
    OP_LOGD(opName_, "RandomUniformIntV2Tiling DoOpTiling.");
    DoBlockTiling();
    UbTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus RandomUniformIntV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t RandomUniformIntV2Tiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(RANDOM_UNIFORM_INT_V2_TPL);
    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus RandomUniformIntV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus RandomUniformIntV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(blockNum_);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    context_->SetTilingKey(GetTilingKey());
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void RandomUniformIntV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << " ubSize: " << ubSize_;
    info << " totalCoreNum: " << totalCoreNum_;
    info << " blockNum: " << blockNum_;
    info << " normalCoreProNum: " << normalCoreProNum_;
    info << " tailCoreProNum: " << tailCoreProNum_;
    info << " singleUbSize: " << singleUbSize_;
    info << " seed: " << seed_;
    info << " seed2: " << seed2_;
    info << " outputSize: " << outputSize_;
    info << " range: " << range_;
    info << " lo: " << lo_;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingPrepare4RandomUniformIntV2Tiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<RandomUniformIntV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "RandomUniformIntV2 GetHardwareInfo Failed, vectorCoreNum:%ld, ubSize:%ld.", compileInfo->totalCoreNum,
            compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingRandomUniformIntV2(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingRandomUniformIntV2");
    RandomUniformIntV2Tiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(RandomUniformIntV2)
    .Tiling(TilingRandomUniformIntV2)
    .TilingParse<RandomUniformIntV2CompileInfo>(TilingPrepare4RandomUniformIntV2Tiling)
    .TilingInputsDataDependency({IN_SHAPE_IDX, IN_MIN_IDX, IN_MAX_IDX});

} // namespace optiling