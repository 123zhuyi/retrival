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
 * \file moe_re_routing_tiling_base.cpp
 * \brief
 */
#include <iostream>
#include "log/log.h"
#include "moe_re_routing_tiling_base.h"

namespace optiling {

constexpr int64_t IN_TOKEN_INDEX = 0;
constexpr int64_t IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX = 1;
constexpr int64_t IN_TOKEN_SCALE_INDEX = 2;
constexpr int64_t OUTPUT_PERMUTE_TOKENS_INDEX = 0;
constexpr int64_t OUT_SCALE_INDEX = 1;
constexpr int64_t OUT_PERMUTE_TOKEN_IDX_IDNEX = 2;
constexpr int64_t OUTPUT_EXPERT_TOKEN_NUM_INDEX = 3;
constexpr int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 0;
constexpr int64_t ATTR_IDX_TYPE_INDEX = 1;
constexpr int64_t DIM_SIZE_TWO = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

static const std::set<ge::DataType> TOKENS_DTYPE = {
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};
static const std::set<ge::DataType> EXPERT_TOKEN_NUM_DTYPE = {ge::DT_INT32, ge::DT_INT64};
static const std::set<ge::DataType> SCALE_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};

static std::tuple<int64_t, int64_t> GetShapeTuple(const gert::TilingContext *context, const int64_t index = 0)
{
    const gert::StorageShape *shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(
        shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."), return std::make_tuple(0, 0));
    // check shape length is DIM_SIZE_TWO
    OP_CHECK_IF(shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE_TWO,
        OP_LOGE(context->GetNodeName(), "Shape must be 2D, but get %zu.", shapePtr->GetStorageShape().GetDimNum()),
        return std::make_tuple(0, 0));
    return std::make_tuple(shapePtr->GetStorageShape().GetDim(0), shapePtr->GetStorageShape().GetDim(1));
}

ge::graphStatus MoeReRoutingTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MoeReRoutingCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr,
            OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
            return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        int64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ((uint64_t &)ubSize));
        ubSize_ = ubSize;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    OP_CHECK_IF((coreNum_ <= 0),
        OP_LOGE(context_->GetNodeName(), "GetHardwareInfo Failed, coreNum:%ld", coreNum_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0,
        OP_LOGE(context_->GetNodeName(), "GetHardwareInfo Failed, ubSize:%ld.", ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckNullptr() const
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(IN_TOKEN_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(IN_TOKEN_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_PERMUTE_TOKENS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUT_PERMUTE_TOKEN_IDX_IDNEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUM_INDEX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckDtypeAndAttr() const
{
    OP_CHECK_IF(std::find(TOKENS_DTYPE.begin(), TOKENS_DTYPE.end(), tokenDtype_) == TOKENS_DTYPE.end(),
        OP_LOGE(context_->GetNodeName(),
            "tokens support INT8, BF16, F16, FP8_E4M3FN, FP8_E5M2 datetype, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(tokenDtype_).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(std::find(EXPERT_TOKEN_NUM_DTYPE.begin(), EXPERT_TOKEN_NUM_DTYPE.end(), expertDtype_) ==
                        EXPERT_TOKEN_NUM_DTYPE.end(),
        OP_LOGE(context_->GetNodeName(),
            "expert_token_num_per_rank support int32, int64, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(expertDtype_).c_str()),
        return ge::GRAPH_FAILED);
    if (tokenDtype_ == ge::DT_FLOAT8_E4M3FN || tokenDtype_ == ge::DT_FLOAT8_E5M2) {
        OP_CHECK_IF((scaleDtype_ != ge::DT_FLOAT8_E8M0 && scaleDtype_ != ge::DT_FLOAT),
            OP_LOGE(context_->GetNodeName(),
                "tokens is fp8, scale should be float8_e8m0 or float32, actual %s.",
                ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str()),
            return ge::GRAPH_FAILED);
    }
    if (hasScale_) {
        auto outputScaleType = context_->GetOutputDesc(OUT_SCALE_INDEX)->GetDataType();
        OP_CHECK_IF(outputScaleType != scaleDtype_,
            OP_LOGE(context_->GetNodeName(),
                "output permute_per_token_scales should be the same with %s, actual %s.",
                ge::TypeUtils::DataTypeToSerialString(scaleDtype_).c_str(),
                ge::TypeUtils::DataTypeToSerialString(outputScaleType).c_str()),
            return ge::GRAPH_FAILED);
    }
    auto outputTokenType = context_->GetOutputDesc(OUTPUT_PERMUTE_TOKENS_INDEX)->GetDataType();
    OP_CHECK_IF(outputTokenType != tokenDtype_,
        OP_LOGE(context_->GetNodeName(),
            "output permute_tokens should be the same with %s, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(tokenDtype_).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputTokenType).c_str()),
        return ge::GRAPH_FAILED);

    auto outputTokenIdxType = context_->GetOutputDesc(OUT_PERMUTE_TOKEN_IDX_IDNEX)->GetDataType();
    OP_CHECK_IF(outputTokenIdxType != ge::DT_INT32,
        OP_LOGE(context_->GetNodeName(),
            "output permute_tokens_idx dtype should be %s, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(ge::DT_INT32).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputTokenIdxType).c_str()),
        return ge::GRAPH_FAILED);

    auto outputTokenNumType = context_->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUM_INDEX)->GetDataType();
    OP_CHECK_IF(outputTokenNumType != expertDtype_,
        OP_LOGE(context_->GetNodeName(),
            "output expert_token_num dtype should the same with %s, actual %s.",
            ge::TypeUtils::DataTypeToSerialString(expertDtype_).c_str(),
            ge::TypeUtils::DataTypeToSerialString(outputTokenNumType).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((idxType_ != 0 && idxType_ != 1),
        OP_LOGE(context_->GetNodeName(), "idxType should be 0 or 1."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckOutputShape() const
{
    const gert::StorageShape *outTokenShapePtr = context_->GetOutputShape(OUTPUT_PERMUTE_TOKENS_INDEX);
    OP_CHECK_IF(outTokenShapePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "outTokenShapePtr is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outTokenShapePtr->GetStorageShape() != context_->GetInputShape(IN_TOKEN_INDEX)->GetStorageShape(),
        OP_LOGE(context_->GetNodeName(), "Input tokens shape must be same with out pertmute_tokens."),
        return ge::GRAPH_FAILED);
    if (hasScale_) {
        const gert::StorageShape *outScaleShapePtr = context_->GetOutputShape(OUT_SCALE_INDEX);
        OP_CHECK_IF(outScaleShapePtr == nullptr,
            OP_LOGE(context_->GetNodeName(), "outScaleShapePtr is nullptr."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            outScaleShapePtr->GetStorageShape() != context_->GetInputShape(IN_TOKEN_SCALE_INDEX)->GetStorageShape(),
            OP_LOGE(context_->GetNodeName(), "per_tokens_scale shape must same with permute_per_token_scales."),
            return ge::GRAPH_FAILED);
    }
    const gert::StorageShape *tokenIdxShapePtr = context_->GetOutputShape(OUT_PERMUTE_TOKEN_IDX_IDNEX);
    OP_CHECK_IF(tokenIdxShapePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "tokenIdxShapePtr is nullptr."),
        return ge::GRAPH_FAILED);
    auto tokenIdxShape = tokenIdxShapePtr->GetStorageShape();
    OP_CHECK_IF(tokenIdxShape.GetDimNum() != 1,
        OP_LOGE(
            context_->GetNodeName(), "tokenIdxShape shape only support dim 1 actual %zu.", tokenIdxShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenIdxShape.GetDim(0) != tokenSum_,
        OP_LOGE(context_->GetNodeName(),
            "permute_token_idx should equal tokenSum %ld, actual %ld.",
            tokenSum_,
            tokenIdxShape.GetDim(0)),
        return ge::GRAPH_FAILED);
    const gert::StorageShape *tokenNumShapePtr = context_->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUM_INDEX);
    OP_CHECK_IF(tokenNumShapePtr == nullptr,
        OP_LOGE(context_->GetNodeName(), "tokenNumShapePtr is nullptr."),
        return ge::GRAPH_FAILED);
    auto tokenNumShape = tokenNumShapePtr->GetStorageShape();
    OP_CHECK_IF(tokenNumShape.GetDimNum() != 1,
        OP_LOGE(
            context_->GetNodeName(), "tokenNumShape shape only support dim 1 actual %zu.", tokenNumShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenNumShape.GetDim(0) != expertNum_,
        OP_LOGE(context_->GetNodeName(),
            "permute_token_Num should equal expertNum %ld, actual %ld.",
            expertNum_,
            tokenNumShape.GetDim(0)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::CheckParam()
{
    OP_CHECK_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "check dtype and attr failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenSum_ < 0,
        OP_LOGE(context_->GetNodeName(), "tokenSum %ld is less than 0.", tokenSum_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenSize_ <= 0,
        OP_LOGE(context_->GetNodeName(), "tokenSize %ld should be greater than 0.", tokenSize_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(rankNums_ <= 0,
        OP_LOGE(context_->GetNodeName(), "rankNums %ld should be greater than 0.", rankNums_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertNum_ <= 0,
        OP_LOGE(context_->GetNodeName(), "expertNum %ld should be greater than 0.", expertNum_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShape() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "check output shape failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeReRoutingTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        context_ == nullptr, OP_LOGE(context_->GetNodeName(), "context_ can not be nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckNullptr() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "check nullptr fail."),
        return ge::GRAPH_FAILED);
    auto tokenShapeTuple = GetShapeTuple(context_, IN_TOKEN_INDEX);
    auto expertShapeTuple = GetShapeTuple(context_, IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX);
    tokenSum_ = std::get<0>(tokenShapeTuple);
    tokenSize_ = std::get<1>(tokenShapeTuple);
    rankNums_ = std::get<0>(expertShapeTuple);
    expertNum_ = std::get<1>(expertShapeTuple);
    tokenDtype_ = context_->GetInputDesc(IN_TOKEN_INDEX)->GetDataType();
    expertDtype_ = context_->GetInputDesc(IN_EXPERT_TOKEN_NUM_PER_RANK_INDEX)->GetDataType();
    auto scaleShapePtr = context_->GetInputShape(IN_TOKEN_SCALE_INDEX);
    if (scaleShapePtr != nullptr) {
        auto &scaleShape = context_->GetInputShape(IN_TOKEN_SCALE_INDEX)->GetStorageShape();
        if (scaleShape.GetDimNum() == 1) {
            scaleSize_ = 1;
        } else {
            auto scaleShapeTuple = GetShapeTuple(context_, IN_TOKEN_SCALE_INDEX);
            scaleSize_ = std::get<1>(scaleShapeTuple);
            auto scaleSum = std::get<0>(scaleShapeTuple);
            OP_CHECK_IF(scaleSum != tokenSum_,
                OP_LOGE(
                    context_->GetNodeName(), "scale tokenSum: %ld should equal to tokenSum: %ld.", scaleSum, tokenSum_),
                return ge::GRAPH_FAILED);
        }
        auto scalePtr = context_->GetOptionalInputDesc(IN_TOKEN_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, scalePtr);
        scaleDtype_ = scalePtr->GetDataType();
        hasScale_ = true;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto expertTokenNumType = attrs->GetInt(ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokenNumType);
    OP_CHECK_IF(*expertTokenNumType != 1,
        OP_LOGE(context_->GetNodeName(), "expertTokenNumType should == 1."),
        return ge::GRAPH_FAILED);

    auto idxType = attrs->GetInt(ATTR_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxType);
    idxType_ = static_cast<int64_t>(*idxType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling