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
 * \file add_n_tiling_arch35.cpp
 * \brief
 */
#include "add_n_tiling_arch35.h"
#include "log/log.h"
#include <iostream>

namespace optiling {
constexpr int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t ADDN_BUFFER_NUM = 6; // db: 2 input + 1 output
constexpr int64_t GROUP_SIZE = 8;
constexpr int64_t ALINE_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t TILINGKEY = 100ULL;
constexpr uint64_t TILINGKEY_FULLLOAD = 101ULL;
constexpr size_t KINPUT = 0U;

class AddNTiling {
public:
    explicit AddNTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    AddNTilingData tiling;

protected:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();
    ge::graphStatus SetTilingKey();
    ge::graphStatus GetCompileInfo();
    ge::graphStatus CalcTiling();
    ge::graphStatus WriteTilingData(const Ops::Base::ElewiseTilingData& elewiseTilingData);
    ge::graphStatus CheckOpParam();

private:
    gert::TilingContext* tilingContext;
    int64_t coreNum_{0};
    int64_t ubSize_{0};

    int64_t dtypeSize_ = 0;
    int64_t sizeN_ = 0;
    int64_t firstN_ = 0;
    int64_t loopN_ = 0;
    gert::Shape shape_;
};

ge::graphStatus AddNTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "AddNTiling SetTilingData enter.");
    if (tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "SetTilingData failed, GetRawTilingData is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::SetTilingKey()
{
    auto input_storage_shape = tilingContext->GetInputShape(KINPUT)->GetStorageShape();
    size_t dim_num = input_storage_shape.GetDimNum();
    int64_t sum2 = 1;
    for (size_t i = 0; i < dim_num; i++) {
        int64_t sum1 = input_storage_shape.GetDim(i);
        sum2 = sum1 * sum2;
    }
    sum2 = (sum2 * dtypeSize_ + ALINE_SIZE - 1) / ALINE_SIZE * ALINE_SIZE / dtypeSize_;
    if ((sizeN_ + 1) * sum2 * dtypeSize_ < ubSize_) {
        tilingContext->SetTilingKey(TILINGKEY_FULLLOAD);
        tilingContext->SetBlockDim(tiling.get_blockNum());
        return ge::GRAPH_SUCCESS;
    }
    tilingContext->SetTilingKey(TILINGKEY);
    tilingContext->SetBlockDim(tiling.get_blockNum());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::GetCompileInfo()
{
    auto compileInfo = tilingContext->GetCompileInfo<AddNCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            tilingContext->GetNodeName(), "coreNum,ubSize", std::to_string(coreNum_)+","+std::to_string(ubSize_),"The values of coreNum,ubSize must be greater than 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::CheckDtype()
{
    auto x1Desc = tilingContext->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, x1Desc);
    auto x1Dtype = x1Desc->GetDataType();

    for (int64_t i = 1; i < sizeN_; i++) {
        auto x2Desc = tilingContext->GetDynamicInputDesc(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, x2Desc);
        auto x2Dtype = x2Desc->GetDataType();

        OP_CHECK_IF(
            x1Dtype != x2Dtype, OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x[0], x[i]",
                                                                        ge::TypeUtils::DataTypeToSerialString(x1Dtype) + ", " +
                                                                        ge::TypeUtils::DataTypeToSerialString(x2Dtype),
                                                                        "The dtypes of x[0] and x[i] must be the same"),
            return ge::GRAPH_FAILED);
    }

    auto yDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(
        x1Dtype != yDtype, OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "x,y",
                                                                  ge::TypeUtils::DataTypeToSerialString(x1Dtype) + ", " +                                                          
                                                                  ge::TypeUtils::DataTypeToSerialString(yDtype),
                                                                  "The dtypes of x and y must be the same"),
        return ge::GRAPH_FAILED);

    dtypeSize_ = ge::GetSizeByDataType(yDtype);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::CheckShape()
{
    // get first input shape
    auto xTensorShape1Ptr = tilingContext->GetDynamicInputShape(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xTensorShape1Ptr);
    auto xTensorShape1 = xTensorShape1Ptr->GetStorageShape();

    // check input shapes
    for (int64_t i = 1; i < sizeN_; i++) {
        auto xTensorShape2Ptr = tilingContext->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xTensorShape2Ptr);
        auto xTensorShape2 = xTensorShape2Ptr->GetStorageShape();
        OP_CHECK_IF(
            xTensorShape2 != xTensorShape1,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x[0], x[i]",
                                                    (Ops::Base::ToString(xTensorShape1) + ", " + Ops::Base::ToString(xTensorShape2)).c_str(),
                                                    "The shapes of x[0] and x[i] must be the same"),
            return ge::GRAPH_FAILED);
    }

    // check output shape
    auto yShapePtr = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(
        xTensorShape1 != yShape,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "x,y", (Ops::Base::ToString(xTensorShape1) + ", " + Ops::Base::ToString(yShape)).c_str(),
                                               "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    shape_ = yShape;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::CalcTiling()
{
    Ops::Base::ComputeParams params;
    params.maxDtypeBits = dtypeSize_ * Ops::Base::BITS_NUM;
    params.minDtypeBits = dtypeSize_ * Ops::Base::BITS_NUM;
    params.extraSize = {0};
    params.bufferDivisor = {ADDN_BUFFER_NUM * params.maxDtypeBits};

    Ops::Base::ElewiseTilingParams elewiseTilingParams;
    elewiseTilingParams.shape = shape_;
    elewiseTilingParams.coreNum = coreNum_;
    elewiseTilingParams.ubSize = ubSize_;
    elewiseTilingParams.computeMap = {{0, params}};
    Ops::Base::ElewiseTilingData elewiseTilingData;
    if (Ops::Base::ElewiseTiling(elewiseTilingParams, elewiseTilingData) != ge::GRAPH_SUCCESS) {
        OP_LOGE(tilingContext, "Do elewise tiling failed.");
        return ge::GRAPH_FAILED;
    }

    if (sizeN_ <= GROUP_SIZE + 1) {
        firstN_ = sizeN_;
        loopN_ = 0;
    } else {
        int64_t firstSize = sizeN_ % GROUP_SIZE;
        firstN_ = (firstSize <= 1) ? firstSize + GROUP_SIZE : firstSize;
        loopN_ = (sizeN_ - firstN_) / GROUP_SIZE;
    }
    OP_CHECK_IF(
        (WriteTilingData(elewiseTilingData) != ge::GRAPH_SUCCESS),
        OP_LOGE(tilingContext->GetNodeName(), "Do AddNTiling WriteTilingData Failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::WriteTilingData(const Ops::Base::ElewiseTilingData& elewiseTilingData)
{
    OP_LOGD(
        tilingContext->GetNodeName(),
        "tilingdata is dim0:%ld, blockFormer:%ld, blockNum:%ld, ubFormer:%ld,"
        "ubLoopOfFormerBlock:%ld, ubLoopOfTailBlock:%ld, ubTailOfFormerBlock:%ld, ubTailOfTailBlock:%ld,"
        "elemNum:%ld, sizeN:%ld, firstN:%ld, loopN:%ld",
        elewiseTilingData.dim0, elewiseTilingData.blockFormer, elewiseTilingData.blockNum, elewiseTilingData.ubFormer,
        elewiseTilingData.ubLoopOfFormerBlock, elewiseTilingData.ubLoopOfTailBlock,
        elewiseTilingData.ubTailOfFormerBlock, elewiseTilingData.ubTailOfTailBlock, elewiseTilingData.elemNum, sizeN_,
        firstN_, loopN_);

    tiling.set_dim0(elewiseTilingData.dim0);
    tiling.set_blockFormer(elewiseTilingData.blockFormer);
    tiling.set_blockNum(elewiseTilingData.blockNum);
    tiling.set_ubFormer(elewiseTilingData.ubFormer);
    tiling.set_ubLoopOfFormerBlock(elewiseTilingData.ubLoopOfFormerBlock);
    tiling.set_ubLoopOfTailBlock(elewiseTilingData.ubLoopOfTailBlock);
    tiling.set_ubTailOfFormerBlock(elewiseTilingData.ubTailOfFormerBlock);
    tiling.set_ubTailOfTailBlock(elewiseTilingData.ubTailOfTailBlock);
    tiling.set_elemNum(elewiseTilingData.elemNum);
    tiling.set_sizeN(sizeN_);
    tiling.set_firstN(firstN_);
    tiling.set_loopN(loopN_);

    OP_CHECK_IF(
        (SetTilingData() != ge::GRAPH_SUCCESS),
        OP_LOGE(tilingContext->GetNodeName(), "Do AddNTiling SetTilingData Failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (SetTilingKey() != ge::GRAPH_SUCCESS),
        OP_LOGE(tilingContext->GetNodeName(), "Do AddNTiling SetTilingKey Failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::CheckOpParam()
{
    // get input num
    auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, anchorInstanceInfo);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    OP_CHECK_IF(
        inputNum == 0, OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "x", "0", ">= 1"),
        return ge::GRAPH_FAILED);

    // get attr N
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const auto* attrN = attrs->GetAttrPointer<int32_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrN);
    sizeN_ = static_cast<int64_t>(*attrN);
    OP_CHECK_IF(
        sizeN_ != static_cast<int64_t>(inputNum),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "N", std::to_string(sizeN_)+","+std::to_string(static_cast<int64_t>(inputNum)), "The value of N must be the tensor number of x"),
        return ge::GRAPH_FAILED);

    // check shape
    OP_CHECK_IF(
        CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"),
        return ge::GRAPH_FAILED);
    // check dtype
    OP_CHECK_IF(
        CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "check dtype failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTiling::RunTiling()
{
    OP_CHECK_IF(
        (GetCompileInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(tilingContext->GetNodeName(), "Do AddNTiling GetCompileInfo Failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (CheckOpParam() != ge::GRAPH_SUCCESS),
        OP_LOGE(tilingContext->GetNodeName(), "Do AddNTiling CheckOpParam Failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CalcTiling() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "calculate tiling failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AddN(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4AddN rt2.0 is running.");

    auto platformInfo = tilingContextGen->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    AddNTiling addNOpTiling(tilingContextGen);
    return addNOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForAddN(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AddNCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddN).Tiling(Tiling4AddN).TilingParse<AddNCompileInfo>(TilingPrepareForAddN);
} // namespace optiling
