/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_grouped_mat_mul_tiling_base.cpp
 * \brief
 */
#include "allto_allv_grouped_mat_mul_tiling_base.h"
#include <numeric>

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace optiling {
// protected
ge::graphStatus AlltoAllvGmmTilingBase::GetCommonPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
        VECTOR_INNER_ERR_REPORT_TILING(context_->GetNodeName(), "can not get platform info."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckCommonPlatformInfo()
{
    OP_TILING_CHECK((aicCoreNum_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, aic num can not be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((aivCoreNum_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, aiv num can not be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l0aSize_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, l0a size can not be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l0bSize_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, l0b size can not be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l0cSize_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, l0c size can not be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((l1Size_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, l1 size can not be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((ubSize_ == 0U),
        OP_LOGE(context_->GetNodeName(), "platform info is invalid, ub size can not be 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetCommonShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetCommonShapeAttrsInfo.");
    auto getAttrsInfoStatus = GetAttrsInfo();
    if (getAttrsInfoStatus != ge::GRAPH_SUCCESS) {
        return getAttrsInfoStatus;
    }
    // get shape info depends attr
    auto getShapeInfoStatus = GetShapeInfo();
    if (getShapeInfoStatus != ge::GRAPH_SUCCESS) {
        return getShapeInfoStatus;
    }
    OP_LOGD(context_->GetNodeName(), "end GetCommonShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckCommonShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckCommonShapeAttrsInfo.");
    auto checkAttrsInfoStatus = CheckAttrsInfo();
    if (checkAttrsInfoStatus != ge::GRAPH_SUCCESS) {
        return checkAttrsInfoStatus;
    }
    auto checkShapeInfoStatus = CheckShapeInfo();
    if (checkShapeInfoStatus != ge::GRAPH_SUCCESS) {
        return checkShapeInfoStatus;
    }
    auto checkFormatStatus = CheckFormat();
    if (checkFormatStatus != ge::GRAPH_SUCCESS) {
        return checkFormatStatus;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckCommonShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

// private

ge::graphStatus AlltoAllvGmmTilingBase::GetAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetAttrsInfo.");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "can not get attrs."), return ge::GRAPH_FAILED);
    // group
    groupPtr_ = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr_ == nullptr, OP_LOGE(context_->GetNodeName(), "group attr can not be null."),
        return ge::GRAPH_FAILED);
    group_ = groupPtr_;
    // epWorldSize
    epWorldSizePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr_ == nullptr, OP_LOGE(context_->GetNodeName(), "epWorldSize attr can not be null."),
        return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr_;
    // sendCounts
    sendCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr_ == nullptr, OP_LOGE(context_->GetNodeName(), "sendCounts attr can not be null."),
        return ge::GRAPH_FAILED);
    sendCounts = static_cast<const int64_t*>(sendCountsPtr_->GetData());
    // recvCounts
    recvCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr_ == nullptr, OP_LOGE(context_->GetNodeName(), "recvCounts attr can not be null."),
        return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t*>(recvCountsPtr_->GetData());
    // transGmmWeight
    transGmmWeightPtr_ = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (transGmmWeightPtr_ != nullptr) {
        transGmmWeight_ = *transGmmWeightPtr_;
    }
    // transMmWeight
    transMmWeightPtr_ = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    if (transMmWeightPtr_ != nullptr) {
        transMmWeight_ = *transMmWeightPtr_;
    }
    // permuteOutFlag
    permuteOutFlagPtr_ = attrs->GetAttrPointer<bool>(ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr_ != nullptr) {
        permuteOutFlag_ = *permuteOutFlagPtr_;
    }
    gmmXQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX);
    gmmWeightQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX);
    mmXQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_MM_X_QUANT_MODE_INDEX);
    mmWeightQuantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_MM_WEIGHT_QUANT_MODE_INDEX);
    OP_LOGD(context_->GetNodeName(), "end GetAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckAttrsInfo.");
    if (CheckEpWorldSizeValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommCountsRange() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommCountsValue() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckEpWorldSizeValue()
{
    // check epWorldSize in socVersion
    std::vector<int64_t> epWorldSizeValueList;
    std::string epWorldSizeValueStr = "";
    auto platformInfo = context_->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        epWorldSizeValueList = { 2, 4, 8, 16, 32, 64, 128, 256 }; // epWorldSize value only support 2, 4, 8, 16, 32, 64, 128, 256
    } else {
        epWorldSizeValueList = { 8, 16, 32, 64, 128 }; // epWorldSize value only support 8, 16, 32, 64, 128
    }
    for (size_t i = 0; i < epWorldSizeValueList.size(); i++) {
        if (i != 0) {
            epWorldSizeValueStr += ", ";
        }
        epWorldSizeValueStr += std::to_string(epWorldSizeValueList[i]);
    }
    OP_TILING_CHECK(std::find(epWorldSizeValueList.begin(), epWorldSizeValueList.end(), epWorldSize_) ==
        epWorldSizeValueList.end(),
        OP_LOGE(context_->GetNodeName(), "epWorldSize[%lu] should be in [%s]!", epWorldSize_,
        epWorldSizeValueStr.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckCommCountsRange()
{
    // check sendCounts/recvCounts size
    uint64_t sendCountsSize = sendCountsPtr_->GetSize();
    uint64_t recvCountsSize = recvCountsPtr_->GetSize();
    OP_TILING_CHECK(sendCountsSize != recvCountsSize,
        OP_LOGE(context_->GetNodeName(),
        "The size of sendCounts(e * epWorldSize) %lu should be equal to recvCounts(e * epWorldSize) %lu !",
        sendCountsSize, recvCountsSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(e_ * epWorldSize_ != sendCountsSize,
        OP_LOGE(context_->GetNodeName(),
        "The first dim of gmmWeight(e, H1, N1) %lu  multi epWorldSize_ %lu shoubl be equal to the size of "
        "sendCounts(e*ep) %lu!",
        e_, epWorldSize_, sendCountsSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((e_ * epWorldSize_ <= EXPERT_MIN_VALUE) || (e_ * epWorldSize_ > EXPERT_MAX_VALUE),
        OP_LOGE(context_->GetNodeName(),
        "The size of send_counts(e*ep) and recv_counts(e*ep) should be in (%lu, %lu], but got %lu!", EXPERT_MIN_VALUE,
        EXPERT_MAX_VALUE, e_ * epWorldSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckCommCountsValue()
{
    // check sendCounts range
    for (uint64_t index = 0U; index < e_ * epWorldSize_; index++) {
        OP_TILING_CHECK((sendCounts[index] < SEND_COUNTS_MIN_VALUE) || (sendCounts[index] > bsk_),
            OP_LOGE(context_->GetNodeName(), "sendCounts[%lu] should be in [0, BSK[%lu]], but get %lu", index, bsk_,
            sendCounts[index]),
            return ge::GRAPH_FAILED);
    }
    // check recvCounts range
    for (uint64_t index = 0U; index < e_ * epWorldSize_; index++) {
        OP_TILING_CHECK((recvCounts[index] < DIM_ZERO) || (recvCounts[index] > a_),
            OP_LOGE(context_->GetNodeName(), "recvCounts[%lu] should be in [0, A[%lu]], but get %lu", index, a_,
            recvCounts[index]),
            return ge::GRAPH_FAILED);
    }
    uint64_t sendCountsSize = sendCountsPtr_->GetSize();
    uint64_t recvCountsSize = recvCountsPtr_->GetSize();
    // check sum(sendCounts) = BSK
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != bsk_,
        OP_LOGE(context_->GetNodeName(), "The sum of sendCounts %lu should be equal to BSK %lu!", sendCountsSum, bsk_),
        return ge::GRAPH_FAILED);
    // check sum(recvCounts) = A
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != a_,
        OP_LOGE(context_->GetNodeName(), "The sum of recvCounts %lu should be equal to A %lu!", recvCountsSum, a_),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeInfo.");
    if (GetGmmXShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetGmmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetCountsTensorShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetMmxShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetMmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetGmmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetMmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetPermuteOutShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckShapeInfo.");
    if (CheckGmmXShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGmmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCountsTensorShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmxShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmWeightShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGmmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMmYShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckPermuteOutShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetGmmXShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetGmmXShapeInfo.");
    // check gmmX not null
    OP_TILING_CHECK(context_->GetInputShape(GMM_X_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmX input shape can not be null."), return ge::GRAPH_FAILED);
    bsk_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    h1_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(context_->GetInputDesc(GMM_X_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "can not get gmmX input desc."), return ge::GRAPH_FAILED);
    gmmXDataType_ = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_LOGD(context_->GetNodeName(), "end GetGmmXShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckGmmXShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmXShapeInfo.");
    // check dim = 2
    if (context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim of gmmX(BSK, H1) should be 2, but got %lu!",
            context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // check bsk range
    OP_TILING_CHECK(bsk_ <= BSK_MIN_VALUE || bsk_ >= BSK_MAX_VALUE,
        OP_LOGE(context_->GetNodeName(), "BSK should be in (%lu, %lu), but got %lu.", BSK_MIN_VALUE, BSK_MAX_VALUE,
        bsk_),
        return ge::GRAPH_FAILED);
    // check h1 range
    OP_TILING_CHECK((h1_ <= H1_MIN_VALUE) || (h1_ >= H1_MAX_VALUE),
        OP_LOGE(context_->GetNodeName(), "H1 should be in (%lu, %lu), but got %lu.", H1_MIN_VALUE, H1_MAX_VALUE, h1_),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckGmmXShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetGmmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetGmmWeightShapeInfo.");
    OP_TILING_CHECK(context_->GetInputShape(GMM_WEIGHT_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmWeight input shape can not be null."), return ge::GRAPH_FAILED);
    e_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(n1DimIndex);
    OP_TILING_CHECK(context_->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "can not get gmmWeight input desc."), return ge::GRAPH_FAILED);
    gmmWeightDataType_ = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_LOGD(context_->GetNodeName(), "end GetGmmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckGmmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmWeightShapeInfo.");
    // check dim
    if (context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != DIM_THREE) {
        OP_LOGE(context_->GetNodeName(), "The dim of gmmWeight(e, H1, N1) should be 3, but got %lu!",
            context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // check H1 equal
    uint64_t gmmWeightH1 = transGmmWeight_ ?
        context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_TWO) :
        context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(h1_ != gmmWeightH1,
        OP_LOGE(context_->GetNodeName(),
        "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of gmmWeight(e, H1, N1) !", h1_, gmmWeightH1),
        return ge::GRAPH_FAILED);
    // check e range
    OP_TILING_CHECK((e_ <= E_MIN_VALUE) || (e_ > E_MAX_VALUE),
        OP_LOGE(context_->GetNodeName(), "e should be in (%lu, %lu], but got %lu.", E_MIN_VALUE, E_MAX_VALUE, e_),
        return ge::GRAPH_FAILED);
    // check N1 range
    OP_TILING_CHECK(n1_ <= N1_MIN_VALUE || n1_ >= N1_MAX_VALUE,
        OP_LOGE(context_->GetNodeName(), "N1 should be in (%lu, %lu), but got %lu!", N1_MIN_VALUE, N1_MAX_VALUE, n1_),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckGmmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetCountsTensorShapeInfo() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckCountsTensorShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckCountsTensorShapeInfo.");
    // sendCountsTensor only support nullptr
    OP_TILING_CHECK(context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_INDEX) != nullptr,
        OP_LOGE(context_->GetNodeName(), "sendCountsTensor should all be null."), return ge::GRAPH_FAILED);
    // recvCountsTensor only support nullptr
    OP_TILING_CHECK(context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_INDEX) != nullptr,
        OP_LOGE(context_->GetNodeName(), "recvCountsTensor should all be null."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckCountsTensorShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetMmxShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetMmxShapeInfo.");
    // optional input mmX
    if (context_->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        hasSharedExpertFlag_ = true;
        bs_ = context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(context_->GetOptionalInputDesc(MM_X_INDEX) == nullptr,
            OP_LOGE(context_->GetNodeName(), "can not get mmX input desc."), return ge::GRAPH_FAILED);
        mmXDataType_ = context_->GetOptionalInputDesc(MM_X_INDEX)->GetDataType();
        h2_ = context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    }
    OP_LOGD(context_->GetNodeName(), "end GetMmxShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckMmxShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmxShapeInfo.");
    if (!hasSharedExpertFlag_) {
        OP_LOGD(context_->GetNodeName(), "end CheckMmxShapeInfo.");
        return ge::GRAPH_SUCCESS;
    }
    // check dim
    if (context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim of mmX(BS, H2) should be 2, but got %lu!",
            context_->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // check BS range
    OP_TILING_CHECK(bs_ < BS_MIN_VALUE,
        OP_LOGE(context_->GetNodeName(), "BS should be larger than %lu, but got %lu!", BS_MIN_VALUE, bs_),
        return ge::GRAPH_FAILED);
    // check BSK divisible by BS
    OP_TILING_CHECK((bsk_ % bs_ != 0),
        OP_LOGE(context_->GetNodeName(), "BSK should be divisible by BS, but got BSK[%lu] and BS[%lu].", bsk_, bs_),
        return ge::GRAPH_FAILED);
    k_ = bsk_ / bs_;
    // check K range
    OP_TILING_CHECK((k_ < K_MIN_VALUE) || (k_ > K_MAX_VALUE),
        OP_LOGE(context_->GetNodeName(), "K should be in (%lu, %lu), but got %lu.", K_MIN_VALUE, K_MAX_VALUE, k_),
        return ge::GRAPH_FAILED);
    // check H2 range
    OP_TILING_CHECK((h2_ <= H2_MIN_VALUE) || (h2_ >= H2_MAX_VALUE),
        OP_LOGE(context_->GetNodeName(), "H2 should be in (%lu, %lu), but got %lu.", H2_MIN_VALUE, H2_MAX_VALUE, h2_),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckMmxShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetMmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetMmWeightShapeInfo.");
    if (context_->GetOptionalInputShape(MM_WEIGHT_INDEX) != nullptr) {
        n2_ = transMmWeight_ ? context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO) :
                               context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    } else {
        OP_TILING_CHECK(transMmWeight_ == true,
            OP_LOGE(context_->GetNodeName(), "The transMmWeight should be false when mmWeight is null!"), return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end GetMmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckMmWeightShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmWeightShapeInfo.");
    if (context_->GetOptionalInputShape(MM_WEIGHT_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // check h2 equal
    uint64_t mmWeightH2 = transMmWeight_ ?
        context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ONE) :
        context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(h2_ != mmWeightH2,
        OP_LOGE(context_->GetNodeName(), "The H2 %lu of mmX(BS, H2) should be equal to the H2 %lu of mmWeight(H2, N2)!",
        h2_, mmWeightH2),
        return ge::GRAPH_FAILED);
    // check dim
    if (context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim of mmWeight(H2, N2) should be 2, but got %lu!",
            context_->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckMmWeightShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetGmmYShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetGmmYShapeInfo.");
    // output gmmY
    OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmY shape can not be null."), return ge::GRAPH_FAILED);
    a_ = context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    OP_LOGD(context_->GetNodeName(), "end GetGmmYShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckGmmYShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckGmmYShapeInfo.");
    // check dim
    if (context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim of gmmY(A, N1) should be 2, but got %lu!",
            context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end CheckGmmYShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetMmYShapeInfo() {
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckMmYShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckMmYShapeInfo.");
    // check mmX, mmWeight and mmY all be nullptr or all be not nullptr
    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_INDEX);
    const gert::StorageShape *outputMmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_INDEX);
    if (!((mmXStorageShape == nullptr) && (mmWeightStorageShape == nullptr) &&
        (outputMmYStorageShape == nullptr || outputMmYStorageShape->GetStorageShape().GetDimNum() == DIM_ZERO)) &&
        !((mmXStorageShape != nullptr) && (mmWeightStorageShape != nullptr) &&
        (outputMmYStorageShape != nullptr && outputMmYStorageShape->GetStorageShape().GetDimNum() != DIM_ZERO))) {
        OP_LOGE(context_->GetNodeName(), "mmX, mmWeight and mmY should all be nullptr or all be not nullptr!");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr && outputMmYStorageShape->GetStorageShape().GetDimNum() != DIM_ZERO) {
        // check dim
        if (context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
            OP_LOGE(context_->GetNodeName(), "The dim of mmY(BS, N2) should be 2, but got %lu!",
                context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        // check BS equal
        uint64_t mmYBS = context_->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        OP_TILING_CHECK(bs_ != mmYBS,
            OP_LOGE(context_->GetNodeName(), "The BS %lu of mmX(BS, H2) should be equal to the BS %lu of mmY(BS, N2)!", bs_, mmYBS), return ge::GRAPH_FAILED);
    }
    OP_LOGD(context_->GetNodeName(), "end CheckMmYShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetPermuteOutShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetPermuteOutShapeInfo.");
    if (!permuteOutFlag_) {
        OP_LOGD(context_->GetNodeName(), "end GetPermuteOutShapeInfo.");
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr,
        OP_LOGE(context_->GetNodeName(), "permuteOut output can not be null."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end GetPermuteOutShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckPermuteOutShapeInfo()
{
    OP_LOGD(context_->GetNodeName(), "start CheckPermuteOutShapeInfo.");
    if (!permuteOutFlag_) {
        if (context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) != nullptr &&
            context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != DIM_ZERO) {
            OP_LOGE(context_->GetNodeName(), "The permuteOut should be null when permuteOutFlag is false!");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(context_->GetNodeName(), "end CheckPermuteOutShapeInfo.");
        return ge::GRAPH_SUCCESS;
    }
    // check not null
    if (context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr ||
        context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() == DIM_ZERO) {
        OP_LOGE(context_->GetNodeName(), "The permuteOut should not be null when permuteOutFlag is true!");
        return ge::GRAPH_FAILED;
    }
    // check dim
    if (context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != DIM_TWO) {
        OP_LOGE(context_->GetNodeName(), "The dim of permuteOut(A, H1) should be 2, but got %lu!",
            context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // check A\H1 equal
    uint64_t permuteA = context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t permuteH1 = context_->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    OP_TILING_CHECK(h1_ != permuteH1,
        OP_LOGE(context_->GetNodeName(),
        "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of permuteOut(A, H1)!", h1_, permuteH1),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(a_ != permuteA,
        OP_LOGE(context_->GetNodeName(), "The A %lu of gmmY(A, H1) should be equal to the A %lu of permuteOut(A, H1)!",
        a_, permuteA),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "end CheckPermuteOutShapeInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::CheckFormat()
{
    OP_LOGD(context_->GetNodeName(), "start CheckFormat.");
    auto gmmXDesc = context_->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXDesc == nullptr, 
        OP_LOGE(context_->GetNodeName(), "gmmX tensor desc can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "gmmX storage format should be ND."), return ge::GRAPH_FAILED);
    auto gmmWeightDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmWeight tensor desc can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "gmmWeight storage format should be ND."), return ge::GRAPH_FAILED);
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_INDEX);
    if (mmXDesc != nullptr) {
        OP_TILING_CHECK(mmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(context_->GetNodeName(), "mmX storage format should be ND."), return ge::GRAPH_FAILED);
    }
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_INDEX);
    if (mmWeightDesc != nullptr) {
        OP_TILING_CHECK(mmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
            OP_LOGE(context_->GetNodeName(), "mmWeight storage format should be ND."), return ge::GRAPH_FAILED);
    }
    auto gmmYDesc = context_->GetOutputDesc(OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK(gmmYDesc == nullptr,
        OP_LOGE(context_->GetNodeName(), "gmmY tensor desc can not be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
        OP_LOGE(context_->GetNodeName(), "gmmY storage format should be ND."), return ge::GRAPH_FAILED);
    if (hasSharedExpertFlag_) {
        auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        if (mmYDesc != nullptr) {
            OP_TILING_CHECK(mmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                OP_LOGE(context_->GetNodeName(), "mmY storage format should be ND."), return ge::GRAPH_FAILED);
        }
    }
    auto permuteOutDesc = context_->GetOutputDesc(OUTPUT_PERMUTE_OUT_INDEX);
    if (permuteOutFlag_) {
        if (permuteOutDesc != nullptr) {
            OP_TILING_CHECK(permuteOutDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                OP_LOGE(context_->GetNodeName(), "permuteOut storage format should be ND."), return ge::GRAPH_FAILED);
        }
    }
    OP_LOGD(context_->GetNodeName(), "end CheckFormat.");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling