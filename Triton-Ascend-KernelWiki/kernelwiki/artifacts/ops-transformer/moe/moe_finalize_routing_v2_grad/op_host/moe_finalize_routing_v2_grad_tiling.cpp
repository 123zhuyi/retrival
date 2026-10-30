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
 * \file moe_finalize_routing_v2_grad_tiling.cpp
 * \brief
 */
#include "moe_finalize_routing_v2_grad_tiling.h"

namespace optiling {
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t INPUT_0_IDX = 0;
constexpr int64_t INPUT_1_IDX = 1;
constexpr int64_t INPUT_2_IDX = 2;
constexpr int64_t INPUT_3_IDX = 3;
constexpr int64_t INPUT_4_IDX = 4;
constexpr int64_t INPUT_5_IDX = 5;
constexpr int64_t OUTPUT_0_IDX = 0;
constexpr int64_t OUTPUT_1_IDX = 1;
constexpr int64_t ATTR_0_IDX = 0;
constexpr int64_t ATTR_1_IDX = 1;
constexpr int64_t ATTR_2_IDX = 2;
constexpr int64_t ATTR_3_IDX = 3;

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
     OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
     OP_CHECK_IF(
        (aicoreParams_.numBlocks <= 0), OP_LOGE(nodeName_, "get aiv core num failed."),
        return ge::GRAPH_FAILED);

    uint64_t totalUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
    aicoreParams_.ubSize = totalUbSize;
     OP_CHECK_IF(
        (aicoreParams_.ubSize <= 0), OP_LOGE(nodeName_, "get ub size failed."),
        return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    vlFp32_ = Ops::Base::GetVRegSize(context_) / sizeof(float);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetRequiredTensorInfo()
{
    auto gradYShapePtr = context_->GetInputShape(INPUT_0_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, gradYShapePtr);
    gradYShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(gradYShapePtr->GetOriginShape());
    auto gradYDescPtr = context_->GetInputDesc(INPUT_0_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDescPtr);
    gradYType_ = gradYDescPtr->GetDataType();

    auto expandedRowIdxShapePtr = context_->GetInputShape(INPUT_1_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr);
    expandedRowIdxShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(expandedRowIdxShapePtr->GetOriginShape());
    auto expandedRowIdxDescPtr = context_->GetInputDesc(INPUT_1_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxDescPtr);
    expandedRowIdxType_ = expandedRowIdxDescPtr->GetDataType();

    auto gradExpandedXShapePtr = context_->GetOutputShape(OUTPUT_0_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, gradExpandedXShapePtr);
    gradExpandedXShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(gradExpandedXShapePtr->GetOriginShape());
    auto gradExpandedXDescPtr = context_->GetOutputDesc(OUTPUT_0_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, gradExpandedXDescPtr);
    gradExpandedXType_ = gradExpandedXDescPtr->GetDataType();

    auto gradScalesShapePtr = context_->GetOutputShape(OUTPUT_1_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, gradScalesShapePtr);
    gradScalesShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(gradScalesShapePtr->GetOriginShape());
    auto gradScalesDescPtr = context_->GetOutputDesc(OUTPUT_1_IDX);
     OP_CHECK_NULL_WITH_CONTEXT(context_, gradScalesDescPtr);
    gradScalesType_ = gradScalesDescPtr->GetDataType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetOptionalTensorInfo()
{
    auto scalesInputShape = context_->GetOptionalInputShape(INPUT_3_IDX);
    if (scalesInputShape == nullptr) {
        isScalesExist_ = false;
    } else {
        isScalesExist_ = true;
        scalesShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(scalesInputShape->GetOriginShape());
        auto scalesDesc = context_->GetOptionalInputDesc(INPUT_3_IDX);
         OP_CHECK_NULL_WITH_CONTEXT(context_, scalesDesc);
        scalesType_ = scalesDesc->GetDataType();

        auto expandedXInputShape = context_->GetOptionalInputShape(INPUT_2_IDX);
         OP_CHECK_IF(
            (expandedXInputShape == nullptr),
            OP_LOGE(
                nodeName_,
                "expandedXInputShape is nullptr."
                " when scalesOptional is not null, expandedXOptional can't be null."),
            return ge::GRAPH_FAILED);
        expandedXShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(expandedXInputShape->GetOriginShape());
        auto expandedXDesc = context_->GetOptionalInputDesc(INPUT_2_IDX);
         OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXDesc);
        expandedXType_ = expandedXDesc->GetDataType();

        auto biasInputShape = context_->GetOptionalInputShape(INPUT_5_IDX);
        if (biasInputShape == nullptr) {
            isBiasExist_ = false;
        } else {
            isBiasExist_ = true;
            biasShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(biasInputShape->GetOriginShape());
            auto biasDesc = context_->GetOptionalInputDesc(INPUT_5_IDX);
             OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
            biasType_ = biasDesc->GetDataType();

            auto expertIdxInputShape = context_->GetOptionalInputShape(INPUT_4_IDX);
             OP_CHECK_IF(
                (expertIdxInputShape == nullptr),
                OP_LOGE(
                    nodeName_,
                    "expertIdxInputShape is nullptr."
                    " when biasOptional is not null, expertIdxOptional can't be null."),
                return ge::GRAPH_FAILED);
            expertIdxShape_ = Ops::Transformer::OpTiling::EnsureNotScalar(expertIdxInputShape->GetOriginShape());
            auto expertIdxDesc = context_->GetOptionalInputDesc(INPUT_4_IDX);
             OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxDesc);
            expertIdxType_ = expertIdxDesc->GetDataType();
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
     OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    if (attrs->GetAttrNum() > ATTR_0_IDX) {
        dropPadMode_ = *(attrs->GetAttrPointer<int64_t>(ATTR_0_IDX));
    }
    if (attrs->GetAttrNum() > ATTR_1_IDX) {
        activeNum_ = *(attrs->GetAttrPointer<int64_t>(ATTR_1_IDX));
    }

    if (dropPadMode_ == 1) {
         OP_CHECK_IF(
            (attrs->GetAttrNum() <= ATTR_3_IDX),
            OP_LOGE(nodeName_, "if drop_pad_mod is 1, expert_num and expert_capacity is required."),
            return ge::GRAPH_FAILED);
        expertNum_ = *(attrs->GetAttrPointer<int64_t>(ATTR_2_IDX));
        expertCapacity_ = *(attrs->GetAttrPointer<int64_t>(ATTR_3_IDX));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetShapeAttrsInfo()
{
     OP_CHECK_IF(
        (GetRequiredTensorInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(nodeName_, "GetRequiredTensorInfo failed."), return ge::GRAPH_FAILED);

     OP_CHECK_IF(
        (GetOptionalTensorInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(nodeName_, "GetOptionalTensorInfo failed."), return ge::GRAPH_FAILED);

     OP_CHECK_IF(
        (GetAttrInfo() != ge::GRAPH_SUCCESS), OP_LOGE(nodeName_, "GetAttrInfo failed."),
        return ge::GRAPH_FAILED);

     OP_CHECK_IF(
        (CheckParams() != ge::GRAPH_SUCCESS), OP_LOGE(nodeName_, "CheckParams failed."),
        return ge::GRAPH_FAILED);

    topK_ = isScalesExist_ ? scalesShape_.GetDim(1) : 1;
    hidden_ = gradYShape_.GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckAttr()
{
     OP_CHECK_IF(
        ((dropPadMode_ != 0) && (dropPadMode_ != 1)),
        OP_LOGE(nodeName_, "drop_pad_mode must be 0 or 1, but got %ld.", dropPadMode_), return ge::GRAPH_FAILED);

    if (dropPadMode_ == 1) {
         OP_CHECK_IF(
            ((expertNum_ <= 0) || (expertCapacity_ <= 0)),
            OP_LOGE(nodeName_, "if drop_pad_mod is 1, expert_num and expert_capacity must be greater than 0."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckRequiredInput()
{
     OP_CHECK_IF(
        (gradYShape_.GetDimNum() != NUM_TWO),
        OP_LOGE(nodeName_, "grad_y dimnum must be 2, but got %zu.", gradYShape_.GetDimNum()), return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        (expandedRowIdxShape_.GetDimNum() != 1),
        OP_LOGE(nodeName_, "expanded_row_idx dimnum must be 1, but got %zu.", expandedRowIdxShape_.GetDimNum()),
        return ge::GRAPH_FAILED);

     OP_CHECK_IF(
        ((gradYType_ != ge::DT_FLOAT) && (gradYType_ != ge::DT_BF16) && (gradYType_ != ge::DT_FLOAT16)),
        OP_LOGE(nodeName_, "grad_y dtype must be FLOAT or FLOAT16 or BFLOAT16."), return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        (expandedRowIdxType_ != ge::DT_INT32), OP_LOGE(nodeName_, "expanded_row_idx dtype must be DT_INT32."),
        return ge::GRAPH_FAILED);

    gradYTypeByteSize_ = ge::GetSizeByDataType(gradYType_);
     OP_CHECK_IF(
        (gradYTypeByteSize_ <= 0),
        OP_LOGE(nodeName_, "grad_y dtype byte size must be greater than 0."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckOptionalInputShape()
{
    if (expandedXDimNum_ == NUM_TWO) {
         OP_CHECK_IF(
            (expandedXShape_.GetDim(0) != expandedXDim0_),
            OP_LOGE(
                nodeName_, "expanded_x dim0 must be equal to %ld, but got %ld.", expandedXDim0_,
                expandedXShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
    } else {
         OP_CHECK_IF(
            (expandedXShape_.GetDim(0) * expandedXShape_.GetDim(1) != expandedXDim0_),
            OP_LOGE(
                nodeName_, "expanded_x dim0 * dim1 must be equal to %ld, but got %ld.", expandedXDim0_,
                expandedXShape_.GetDim(0) * expandedXShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    }
     OP_CHECK_IF(
        (expandedXShape_.GetDim(expandedXDimNum_ - 1) != gradYShape_.GetDim(1)),
        OP_LOGE(
            nodeName_, "expanded_x last dim and grad_y dim1 must be same, but got %ld, %ld.",
            expandedXShape_.GetDim(expandedXDimNum_ - 1), gradYShape_.GetDim(1)),
        return ge::GRAPH_FAILED);

     OP_CHECK_IF(
        (scalesShape_.GetDim(0) != gradYShape_.GetDim(0)),
        OP_LOGE(
            nodeName_, "scales and grad_y dim0 must be same, but got %ld, %ld.", scalesShape_.GetDim(0),
            gradYShape_.GetDim(0)),
        return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        (scalesShape_.GetShapeSize() != expandedRowIdxShape_.GetShapeSize()),
        OP_LOGE(
            nodeName_, "scales and expanded_row_idx shape size must be same, but got %ld, %ld.",
            scalesShape_.GetShapeSize(), expandedRowIdxShape_.GetShapeSize()),
        return ge::GRAPH_FAILED);

    if (isBiasExist_) {
         OP_CHECK_IF(
            (expertIdxShape_.GetDim(0) != gradYShape_.GetDim(0)),
            OP_LOGE(
                nodeName_, "expert_idx and grad_y dim0 must be same, but got %ld, %ld.", expertIdxShape_.GetDim(0),
                gradYShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (expertIdxShape_.GetShapeSize() != expandedRowIdxShape_.GetShapeSize()),
            OP_LOGE(
                nodeName_, "expert_idx and expanded_row_idx shape size must be same, but got %ld, %ld.",
                expertIdxShape_.GetShapeSize(), expandedRowIdxShape_.GetShapeSize()),
            return ge::GRAPH_FAILED);

         OP_CHECK_IF(
            (biasShape_.GetDim(0) <= 0),
            OP_LOGE(nodeName_, "bias dim0 must be greater than 0, but got %ld.", biasShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
        if (dropPadMode_ == 1) {
             OP_CHECK_IF(
                (biasShape_.GetDim(0) != expertNum_),
                OP_LOGE(nodeName_, "bias dim0 must be equal to %ld, but got %ld.", expertNum_, biasShape_.GetDim(0)),
                return ge::GRAPH_FAILED);
        }
         OP_CHECK_IF(
            (biasShape_.GetDim(1) != gradYShape_.GetDim(1)),
            OP_LOGE(
                nodeName_, "grad_y and bias dim1 must be same, but got %ld, %ld.", gradYShape_.GetDim(1),
                biasShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckOptionalInputDtype()
{
     OP_CHECK_IF(
        (expandedXType_ != gradYType_), OP_LOGE(nodeName_, "expanded_x and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        (scalesType_ != gradYType_), OP_LOGE(nodeName_, "scales and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);
     OP_CHECK_IF(
        (expandedRowIdxType_ != ge::DataType::DT_INT32),
        OP_LOGE(nodeName_, "expanded_row_idx dtype only support int32."), return ge::GRAPH_FAILED);
    if (isBiasExist_) {
         OP_CHECK_IF(
            (expertIdxType_ != expandedRowIdxType_),
            OP_LOGE(nodeName_, "expert_idx and expanded_row_idx dtype must be same."), return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (biasType_ != gradYType_), OP_LOGE(nodeName_, "bias and grad_y dtype must be same."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckOutput()
{
     OP_CHECK_IF(
        ((gradExpandedXShape_.GetDimNum() != NUM_TWO) && (gradExpandedXShape_.GetDimNum() != NUM_THREE)),
        OP_LOGE(nodeName_, "grad_expanded_x dimnum must be 2 or 3, but got %zu.", gradExpandedXShape_.GetDimNum()),
        return ge::GRAPH_FAILED);

    if (expandedXDimNum_ == NUM_TWO) {
         OP_CHECK_IF(
            (gradExpandedXShape_.GetDim(0) != expandedXDim0_),
            OP_LOGE(
                nodeName_, "grad_expanded_x dim0 must be equal to %ld, but got %ld.", expandedXDim0_,
                gradExpandedXShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (gradExpandedXShape_.GetDim(1) != gradYShape_.GetDim(1)),
            OP_LOGE(
                nodeName_, "grad_expanded_x and grad_y dim1 must be same, but got %ld, %ld.",
                gradExpandedXShape_.GetDim(1), gradYShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    } else {
         OP_CHECK_IF(
            (gradExpandedXShape_.GetDim(0) * gradExpandedXShape_.GetDim(1) != expandedXDim0_),
            OP_LOGE(
                nodeName_, "grad_expanded_x dim0 * dim1 must be equal to %ld, but got %ld.", expandedXDim0_,
                gradExpandedXShape_.GetDim(0) * gradExpandedXShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (gradExpandedXShape_.GetDim(NUM_TWO) != gradYShape_.GetDim(1)),
            OP_LOGE(
                nodeName_, "grad_expanded_x dim2 and grad_y dim1 must be same, but got %ld, %ld.",
                gradExpandedXShape_.GetDim(NUM_TWO), gradYShape_.GetDim(1)),
            return ge::GRAPH_FAILED);
    }

     OP_CHECK_IF(
        (gradExpandedXType_ != gradYType_), OP_LOGE(nodeName_, "grad_expanded_x and grad_y dtype must be same."),
        return ge::GRAPH_FAILED);
    if (isScalesExist_) {
         OP_CHECK_IF(
            (gradScalesShape_.GetDim(0) != gradYShape_.GetDim(0)),
            OP_LOGE(
                nodeName_, "grad_scales and grad_y dim0 must be same, but got %ld, %ld.", gradScalesShape_.GetDim(0),
                gradYShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (gradScalesShape_.GetShapeSize() != expandedRowIdxShape_.GetShapeSize()),
            OP_LOGE(
                nodeName_, "grad_scales and expanded_row_idx shape size must be same, but got %ld, %ld.",
                gradScalesShape_.GetShapeSize(), expandedRowIdxShape_.GetShapeSize()),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (gradScalesShape_.GetDimNum() != NUM_TWO),
            OP_LOGE(nodeName_, "grad_scales dimnum must be 2, but got %zu.", gradScalesShape_.GetDimNum()),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (gradScalesType_ != scalesType_), OP_LOGE(nodeName_, "grad_scales and scale dtype must be same."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::CheckParams()
{
     OP_CHECK_IF(
        (CheckAttr() != ge::GRAPH_SUCCESS), OP_LOGE(nodeName_, "CheckAttr check failed."),
        return ge::GRAPH_FAILED);

     OP_CHECK_IF(
        (CheckRequiredInput() != ge::GRAPH_SUCCESS),
        OP_LOGE(nodeName_, "CheckRequiredInput check failed."), return ge::GRAPH_FAILED);

    expandedXDimNum_ = NUM_TWO;
    expandedXDim0_ = expandedRowIdxShape_.GetDim(0);
    if (dropPadMode_ == 0 && activeNum_ > 0 && activeNum_ < expandedXDim0_) {
        expandedXDim0_ = activeNum_;
    } else if (dropPadMode_ == 1) {
        expandedXDimNum_ = NUM_THREE;
        expandedXDim0_ = expertNum_ * expertCapacity_;
    }
    if (isScalesExist_) {
         OP_CHECK_IF(
            (expandedXShape_.GetDimNum() != static_cast<size_t>(expandedXDimNum_)),
            OP_LOGE(
                nodeName_, "expanded_x dimnum error. dropMode = %ld, got expandedXdim %zu", dropPadMode_,
                expandedXShape_.GetDimNum()),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (scalesShape_.GetDimNum() != NUM_TWO),
            OP_LOGE(nodeName_, "scales dimnum must be 2, but got %zu.", scalesShape_.GetDimNum()),
            return ge::GRAPH_FAILED);
        if (isBiasExist_) {
             OP_CHECK_IF(
                (expertIdxShape_.GetDimNum() != NUM_TWO),
                OP_LOGE(nodeName_, "expert_idx dimnum must be 2, but got %zu.", expertIdxShape_.GetDimNum()),
                return ge::GRAPH_FAILED);
             OP_CHECK_IF(
                (biasShape_.GetDimNum() != NUM_TWO),
                OP_LOGE(nodeName_, "bias dimnum must be 2, but got %zu.", biasShape_.GetDimNum()),
                return ge::GRAPH_FAILED);
        }
         OP_CHECK_IF(
            (CheckOptionalInputShape() != ge::GRAPH_SUCCESS),
            OP_LOGE(nodeName_, "CheckOptionalInputShape check failed."),
            return ge::GRAPH_FAILED);
         OP_CHECK_IF(
            (CheckOptionalInputDtype() != ge::GRAPH_SUCCESS),
            OP_LOGE(nodeName_, "CheckOptionalInputDtype check failed."),
            return ge::GRAPH_FAILED);
    } else {
         OP_CHECK_IF(
            (gradYShape_.GetDim(0) != expandedRowIdxShape_.GetDim(0)),
            OP_LOGE(
                nodeName_, "grad_y and expanded_row_idx dim0 must be same, but got %ld, %ld.", gradYShape_.GetDim(0),
                expandedRowIdxShape_.GetDim(0)),
            return ge::GRAPH_FAILED);
    }

     OP_CHECK_IF(
        (CheckOutput() != ge::GRAPH_SUCCESS), OP_LOGE(nodeName_, "CheckOutput check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeFinalizeRoutingV2GradTiling::SetBinaryAddParams(
    MoeFinalizeRoutingV2GradBinaryAddTilingData& params, int64_t factor)
{
    int64_t quotient = (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(factor)));
    params.set_binaryAddQuotient(quotient);
    int64_t vcaddNum = quotient / vlFp32_;
    if (static_cast<uint64_t>(vcaddNum) <= vlFp32_) {
        params.set_binaryAddk(0);
        params.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32_;
        params.set_binaryAddk(__builtin_ctzl(binaryAddNum));
        params.set_binaryAddLastNum(vlFp32_);
    }
}

void MoeFinalizeRoutingV2GradTiling::CalcBaseInfo()
{
    initOutEachCoreBatchNum_ = expandedXDim0_ / aicoreParams_.numBlocks;
    initOutModCoreNum_ = expandedXDim0_ % aicoreParams_.numBlocks;
    if (initOutEachCoreBatchNum_ == 0) {
        initOutNeedCoreNum_ = initOutModCoreNum_;
    } else {
        initOutNeedCoreNum_ = aicoreParams_.numBlocks;
    }

    int64_t expandedRowIdxDim0 = expandedRowIdxShape_.GetDim(0);
    computeEachCoreBatchNum_ = expandedRowIdxDim0 / aicoreParams_.numBlocks;
    computeModCoreNum_ = expandedRowIdxDim0 % aicoreParams_.numBlocks;
    if (computeEachCoreBatchNum_ == 0) {
        computeNeedCoreNum_ = computeModCoreNum_;
    } else {
        computeNeedCoreNum_ = aicoreParams_.numBlocks;
    }
}

uint64_t MoeFinalizeRoutingV2GradTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::PostTiling()
{
    tilingData_.set_initOutNeedCoreNum(initOutNeedCoreNum_);
    tilingData_.set_initOutEachCoreBatchNum(initOutEachCoreBatchNum_);
    tilingData_.set_initOutModCoreNum(initOutModCoreNum_);
    tilingData_.set_computeNeedCoreNum(computeNeedCoreNum_);
    tilingData_.set_computeEachCoreBatchNum(computeEachCoreBatchNum_);
    tilingData_.set_computeModCoreNum(computeModCoreNum_);
    tilingData_.set_dropPadMode(dropPadMode_);
    tilingData_.set_topK(topK_);
    tilingData_.set_hidden(hidden_);
    tilingData_.set_expandedXDim0(expandedXDim0_);
    tilingData_.set_hiddenPrePart(hiddenPrePart_);
    tilingData_.set_hiddenInnerLoops(hiddenInnerLoops_);
    tilingData_.set_hiddenLastPart(hiddenLastPart_);
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());

    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(initOutNeedCoreNum_);
    if (computeNeedCoreNum_ > initOutNeedCoreNum_) {
        context_->SetBlockDim(computeNeedCoreNum_);
    }

    OP_LOGI(
        nodeName_,
        "MoeFinalizeRoutingV2Grad tilingData is initOutNeedCoreNum:%ld, initOutEachCoreBatchNum:%ld, "
        "initOutModCoreNum:%ld, computeNeedCoreNum:%ld, computeEachCoreBatchNum:%ld, computeModCoreNum:%ld, "
        "dropPadMode:%ld, topK:%ld, hidden:%ld, expandedXDim0:%ld, hiddenPrePart:%ld, hiddenInnerLoops:%ld, "
        "hiddenLastPart:%ld, tilingKey:%ld",
        tilingData_.get_initOutNeedCoreNum(), tilingData_.get_initOutEachCoreBatchNum(),
        tilingData_.get_initOutModCoreNum(), tilingData_.get_computeNeedCoreNum(),
        tilingData_.get_computeEachCoreBatchNum(), tilingData_.get_computeModCoreNum(), tilingData_.get_dropPadMode(),
        tilingData_.get_topK(), tilingData_.get_hidden(), tilingData_.get_expandedXDim0(),
        tilingData_.get_hiddenPrePart(), tilingData_.get_hiddenInnerLoops(), tilingData_.get_hiddenLastPart(),
        tilingData_.get_tilingKey());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::DoOpTiling()
{
    CalcBaseInfo();

     OP_CHECK_IF(
        (CalcTilingKey() != ge::GRAPH_SUCCESS), OP_LOGE(nodeName_, "CalcTilingKey failed."),
        return ge::GRAPH_FAILED);

    if (dropPadMode_ == 1) {
        context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFinalizeRoutingV2GradTiling::GetWorkspaceSize()
{
    size_t* workSpaces = context_->GetWorkspaceSizes(1);
     OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaces);
    workSpaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4MoeFinalizeRoutingV2Grad(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MoeFinalizeRoutingV2Grad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeFinalizeRoutingV2Grad)
    .Tiling(Tiling4MoeFinalizeRoutingV2Grad)
    .TilingParse<MoeFinalizeRoutingV2GradCompileInfo>(TilingPrepare4MoeFinalizeRoutingV2Grad);
} // namespace optiling