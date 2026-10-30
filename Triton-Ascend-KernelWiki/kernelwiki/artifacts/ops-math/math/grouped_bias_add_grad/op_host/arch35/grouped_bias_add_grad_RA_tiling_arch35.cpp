/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_tiling_arch35.cpp
\brief tiling implementation for grouped_bias_add_grad arch35
*/
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "grouped_bias_add_grad_RA_tiling_arch35.h"
#include "math/grouped_bias_add_grad/op_kernel/arch35/grouped_bias_add_grad_tilingkey.h"
#include "op_common/atvoss/reduce/reduce_tiling.h"
namespace optiling {

// Helper: Align up to alignment boundary
int64_t GroupedBiasAddGradTilingArch35::AlignUp(int64_t value, int64_t alignment) const
{
    return (value + alignment - 1) / alignment * alignment;
}

// Helper: Align down to alignment boundary
int64_t GroupedBiasAddGradTilingArch35::AlignDown(int64_t value, int64_t alignment) const
{
    return value / alignment * alignment;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::GetPlatformInfo()
{
    // 在初始化实现，打印
    OP_LOGD(
        context_, "Platform info: coreNum=%u, ubSize=%lu, cacheLineSize=%ld, blockSize=%ld", coreNum_, ubSize_,
        cacheLineSize_, blockSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::GetInputOutputInfo()
{
    // Get grad_y input info
    auto gradYInputDesc = context_->GetInputDesc(GRAD_Y_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYInputDesc);
    gradYDtype_ = gradYInputDesc->GetDataType();
    if (gradYDtype_ != ge::DT_FLOAT && gradYDtype_ != ge::DT_FLOAT16 && gradYDtype_ != ge::DT_BF16) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(gradYDtype_);
        std::string reasonMsg = "the dtype of input grad_y should be one of FP32/FP16/BF16";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "grad_y", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto gradYInputShapePtr = context_->GetInputShape(GRAD_Y_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYInputShapePtr);
    auto gradYInputShape = gradYInputShapePtr->GetStorageShape();
    gradYDimNum_ = gradYInputShape.GetDimNum();

    // Get group_idx optional input info
    auto groupIdxInputShapePtr = context_->GetOptionalInputShape(GROUP_IDX_INPUT_IDX);
    if (gradYDimNum_ != RA_DIM_NUM || groupIdxInputShapePtr->GetStorageShape().GetDimNum() != 1) {
        std::string dimMsg = std::to_string(gradYDimNum_);
        std::string reasonMsg = "the input grad_y should be 2D tensor when group_idx is not null, "
                                "and group_idx must be 1D tensor";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "grad_y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    dimGB_ = gradYInputShape.GetDim(0);
    dimG_ = groupIdxInputShapePtr->GetStorageShape().GetShapeSize();
    dimH_ = gradYInputShape.GetDim(1);

    if (dimG_ > INPUT_MAX_GROUP) {
        std::string valueMsg = std::to_string(dimG_);
        std::string reasonMsg = "group size should be less than or equal to " + std::to_string(INPUT_MAX_GROUP);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "group size", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto groupIdxInputDesc = context_->GetInputDesc(GROUP_IDX_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupIdxInputDesc);
    groupIdxDtype_ = groupIdxInputDesc->GetDataType();
    if (groupIdxDtype_ != ge::DT_INT32 && groupIdxDtype_ != ge::DT_INT64) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(groupIdxDtype_);
        std::string reasonMsg = "the dtype of input group_idx should be INT32 or INT64";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "group_idx", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_, "Input info: dimG=%ld, dimGB=%ld, dimH=%ld", dimG_, dimGB_, dimH_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::GetAttrInfo()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrGroupIdxType = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_IDX_TYPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrGroupIdxType);

    groupIdxType_ = static_cast<int32_t>(*attrGroupIdxType);
    if (groupIdxType_ != 0 && groupIdxType_ != 1) {
        std::string valueMsg = std::to_string(groupIdxType_);
        std::string correctMsg = "0 or 1";
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_idx_type", valueMsg.c_str(), correctMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_, "Attr info: groupIdxType=%d", groupIdxType_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::CheckInputOutput()
{
    // Check output shape
    auto gradBiasOutputShapePtr = context_->GetOutputShape(GRAD_BIAS_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradBiasOutputShapePtr);
    auto gradBiasOutputShape = gradBiasOutputShapePtr->GetStorageShape();
    if (gradBiasOutputShape.GetDimNum() != RA_DIM_NUM) {
        std::string dimMsg = std::to_string(gradBiasOutputShape.GetDimNum());
        std::string correctMsg = "2";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_bias", dimMsg.c_str(), correctMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto gradBiasDim0 = gradBiasOutputShape.GetDim(0);
    auto gradBiasDim1 = gradBiasOutputShape.GetDim(1);
    if ((gradBiasDim0 != dimG_) || (gradBiasDim1 != dimH_)) {
        std::string shapeMsg = "[" + std::to_string(gradBiasDim0) + ", " + std::to_string(gradBiasDim1) + "]";
        std::string correctMsg = "[" + std::to_string(dimG_) + ", " + std::to_string(dimH_) + "]";
        OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "grad_bias", shapeMsg.c_str(), correctMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::GetShapeAttrsInfo()
{
    auto ret = GetInputOutputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetAttrInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputOutput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::DetermineMode()
{
    // 空tensor场景，dimGB_或dimH_为0，无需计算
    if (dimGB_ == 0 || dimH_ == 0 || dimG_ == 0) {
        tilingMode_ = GroupedBiasAddGradTilingModeArch35::EMPTY_TENSOR;
        OP_LOGD(context_, "Tiling mode: EMPTY_TENSOR (dimGB_=%ld, dimH_=%ld)", dimGB_, dimH_);
        return ge::GRAPH_SUCCESS;
    }

    // 2D case: determine by H split count
    auto computeTypeSize = ge::GetSizeByDataType(gradYDtype_);
    int64_t hBlockCount = Ops::Base::CeilDiv(dimH_ * computeTypeSize, cacheLineSize_);
    if (hBlockCount > static_cast<int64_t>(coreNumThreshold_)) {
        // Core count would be > 32, use H-axis split (Template 2)
        tilingMode_ = GroupedBiasAddGradTilingModeArch35::CUT_H_MODE;
        OP_LOGD(context_, "Tiling mode: CUT_H_MODE (hBlockCount=%ld > %ld)", hBlockCount, coreNumThreshold_);
    } else {
        // Core count <= 32, use G*H block split (Template 3)
        tilingMode_ = GroupedBiasAddGradTilingModeArch35::CUT_G_MODE;
        OP_LOGD(context_, "Tiling mode: CUT_G_MODE (hBlockCount=%ld <= %ld)", hBlockCount, coreNumThreshold_);
    }
    return ge::GRAPH_SUCCESS;
}

void GroupedBiasAddGradTilingArch35::OptimizeCoreSplit(
    int64_t totalBlocks, int64_t coreNum, int64_t& blockFactor, int64_t& blockTailFactor, int64_t& normalCoreNum,
    int64_t& tailStartIndex) const
{
    // 优先选择尾核相等的分核方案
    // 例如：优先选择 3322 而非 3331

    int64_t baseBlockFactor = totalBlocks / coreNum;
    int64_t remainder = totalBlocks % coreNum;

    if (remainder == 0) {
        // 刚好整除，所有核处理相同数量
        blockFactor = baseBlockFactor;
        blockTailFactor = baseBlockFactor;
        normalCoreNum = coreNum;
        tailStartIndex = coreNum;
    } else {
        // 有余数，需要优化分配
        // 方案1: (baseBlockFactor + 1) * remainder + baseBlockFactor * (coreNum - remainder)
        // 即 remainder 个核处理 baseBlockFactor+1 块，其余处理 baseBlockFactor 块

        // 检查是否可以使用更均匀的分配
        // 尾核相等的条件：所有尾核处理相同数量
        blockFactor = baseBlockFactor + 1;
        blockTailFactor = baseBlockFactor;
        normalCoreNum = remainder;
        tailStartIndex = remainder;
    }

    OP_LOGD(
        context_,
        "OptimizeCoreSplit: totalBlocks=%ld, coreNum=%ld, blockFactor=%ld, "
        "blockTailFactor=%ld, normalCoreNum=%ld, tailStartIndex=%ld",
        totalBlocks, coreNum, blockFactor, blockTailFactor, normalCoreNum, tailStartIndex);
}

ge::graphStatus GroupedBiasAddGradTilingArch35::DoCutHTiling()
{
    // 模版2：核数>32，切分H轴，按cacheLineSize切分
    auto computeTypeSize = ge::GetSizeByDataType(gradYDtype_);
    int64_t groupIdxTypeSize = static_cast<int64_t>(ge::GetSizeByDataType(groupIdxDtype_));

    // 计算H轴按cacheLineSize切分的块数
    int64_t hFactorCount = Ops::Base::CeilDiv(dimH_ * computeTypeSize, cacheLineSize_);
    int64_t hPerBlock = cacheLineSize_ / computeTypeSize;
    int64_t hTailFactor = dimH_ - (hFactorCount - 1) * hPerBlock;
    if (hTailFactor <= 0) {
        hTailFactor = hPerBlock;
    }

    usedCoreNum_ = hFactorCount > static_cast<int64_t>(coreNum_) ? coreNum_ : hFactorCount;
    int64_t blockFactor = Ops::Base::CeilDiv(hFactorCount, static_cast<int64_t>(coreNum_));
    usedCoreNum_ = Ops::Base::CeilDiv(hFactorCount, blockFactor);
    int64_t blockTailFactor = hFactorCount - (usedCoreNum_ - 1) * blockFactor;

    // 计算UB空间分配
    int64_t availableUb = ubSize_ - cacheLineSize_ * USE_TEMP_CACHELINE_NUM - TEMP_BUF_SIZE;

    // groupIdx空间: G * sizeof(groupIdxType), 32B对齐
    int64_t groupedIdxSize = AlignUp(dimG_ * groupIdxTypeSize, ALIGN_32_BYTE);

    // 限制不超过 MAX_OUT_SIZE(16KB)
    int64_t outputSize = AlignUp(hPerBlock * blockFactor * static_cast<int64_t>(sizeof(float)), ALIGN_32_BYTE);
    outputSize = outputSize < MAX_OUT_SIZE ? outputSize : MAX_OUT_SIZE;
    int64_t maxOutputElements = MAX_OUT_SIZE / static_cast<int64_t>(sizeof(float));
    int64_t useUbSize = (availableUb - groupedIdxSize - outputSize * BUFFER_NUM_ARCH35 * QUE_NUM) / BUFFER_NUM_ARCH35;
    useUbSize = AlignDown(useUbSize, ALIGN_32_BYTE);

    cutHTilingData_.blockFactor = blockFactor;
    cutHTilingData_.blockTailFactor = blockTailFactor;
    cutHTilingData_.groupIdxDim = dimG_;
    cutHTilingData_.inputShape[0] = dimGB_;
    cutHTilingData_.inputShape[1] = dimH_;
    cutHTilingData_.hTailFactor = hTailFactor;
    cutHTilingData_.useUbSize = useUbSize;
    cutHTilingData_.groupedIdxSize = groupedIdxSize;
    cutHTilingData_.outputSize = outputSize;
    cutHTilingData_.maxOutputElements = maxOutputElements;
    cutHTilingData_.useTempBuf = TEMP_BUF_SIZE;
    cutHTilingData_.groupIdxType = static_cast<bool>(groupIdxType_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::DoCutGTiling()
{
    // 模版3：核数<=32，按G*(H块数)均匀分核
    auto computeTypeSize = ge::GetSizeByDataType(gradYDtype_);
    int64_t groupIdxTypeSize = static_cast<int64_t>(ge::GetSizeByDataType(groupIdxDtype_));
    int64_t cutHDim = Ops::Base::CeilDiv(dimH_ * computeTypeSize, cacheLineSize_);
    int64_t cutGDim = dimG_;
    int64_t totalBlocks = cutGDim * cutHDim;

    // 分核策略：使用优化的分核方法，优先使尾核相等（3322 优于 3331）
    int64_t effectiveCoreNum = totalBlocks > static_cast<int64_t>(coreNum_) ? coreNum_ : totalBlocks;
    int64_t blockFactor, blockTailFactor, normalCoreNum, tailStartIndex;
    OptimizeCoreSplit(totalBlocks, effectiveCoreNum, blockFactor, blockTailFactor, normalCoreNum, tailStartIndex);

    usedCoreNum_ = effectiveCoreNum;

    int64_t availableUb = ubSize_ - cacheLineSize_ * USE_TEMP_CACHELINE_NUM - TEMP_BUF_SIZE;
    int64_t hPerBlock = cacheLineSize_ / computeTypeSize;
    int64_t groupedIdxSize = AlignUp(dimG_ * groupIdxTypeSize, ALIGN_32_BYTE);

    // output空间 限制不超过 MAX_OUT_SIZE (16KB)
    int64_t maxHBlocks = std::min(blockFactor, cutHDim);
    int64_t outputSize = AlignUp(maxHBlocks * hPerBlock * sizeof(float), ALIGN_32_BYTE);
    outputSize = outputSize < MAX_OUT_SIZE ? outputSize : MAX_OUT_SIZE;
    int64_t maxOutputElements = MAX_OUT_SIZE / static_cast<int64_t>(sizeof(float));
    int64_t useUbSize = (availableUb - groupedIdxSize - outputSize * BUFFER_NUM_ARCH35 * QUE_NUM) / BUFFER_NUM_ARCH35;
    useUbSize = AlignDown(useUbSize, ALIGN_32_BYTE);

    // 计算UB内H尾部长度
    int64_t ubHTailFactor = dimH_ - (cutHDim - 1) * hPerBlock;
    if (ubHTailFactor <= 0) {
        ubHTailFactor = hPerBlock;
    }

    cutGTilingData_.groupIdxType = static_cast<bool>(groupIdxType_);
    cutGTilingData_.cutGDim = cutGDim;
    cutGTilingData_.cutHDim = cutHDim;
    cutGTilingData_.blockFactor = blockFactor;
    cutGTilingData_.blockTailFactor = blockTailFactor;
    cutGTilingData_.blockTailStartIndex = tailStartIndex;
    cutGTilingData_.inputShape[0] = dimGB_;
    cutGTilingData_.inputShape[1] = dimH_;
    cutGTilingData_.ubHTailFactor = ubHTailFactor;
    cutGTilingData_.useUbSize = useUbSize;
    cutGTilingData_.groupedIdxSize = groupedIdxSize;
    cutGTilingData_.outputSize = outputSize;
    cutGTilingData_.maxOutputElements = maxOutputElements;
    cutGTilingData_.useTempBuf = TEMP_BUF_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::DoOpTiling()
{
    auto ret = DetermineMode();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    switch (tilingMode_) {
        case GroupedBiasAddGradTilingModeArch35::CUT_H_MODE:
            return DoCutHTiling();
        case GroupedBiasAddGradTilingModeArch35::CUT_G_MODE:
            return DoCutGTiling();
        case GroupedBiasAddGradTilingModeArch35::EMPTY_TENSOR: {
            usedCoreNum_ = 1;
            return ge::GRAPH_SUCCESS;
        }
        default:
        {
            std::string modeMsg = std::to_string(static_cast<uint32_t>(tilingMode_));
            std::string reasonMsg = "Unknown tiling mode, valid modes are CUT_H_MODE, CUT_G_MODE or EMPTY_TENSOR";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tilingMode",
                modeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
}

ge::graphStatus GroupedBiasAddGradTilingArch35::GetWorkspaceSize()
{
    auto workSpaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workSpaces);

    workspaceSize_ = EMPTY_WORKSPACE_SIZE;
    workSpaces[0] = workspaceSize_;
    OP_LOGD(context_, "Workspace size: %ld", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTilingArch35::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);

    // 根据不同模式保存不同的tiling数据
    if (tilingMode_ == GroupedBiasAddGradTilingModeArch35::CUT_H_MODE) {
        auto* tilingData = context_->GetTilingData<GroupedBiasAddGradCutHTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
        tilingData->blockFactor = cutHTilingData_.blockFactor;
        tilingData->blockTailFactor = cutHTilingData_.blockTailFactor;
        tilingData->groupIdxDim = cutHTilingData_.groupIdxDim;
        tilingData->inputShape[0] = cutHTilingData_.inputShape[0];
        tilingData->inputShape[1] = cutHTilingData_.inputShape[1];
        tilingData->hTailFactor = cutHTilingData_.hTailFactor;
        tilingData->useUbSize = cutHTilingData_.useUbSize;
        tilingData->groupedIdxSize = cutHTilingData_.groupedIdxSize;
        tilingData->outputSize = cutHTilingData_.outputSize;
        tilingData->maxOutputElements = cutHTilingData_.maxOutputElements;
        tilingData->groupIdxType = cutHTilingData_.groupIdxType;
        tilingData->useTempBuf = cutHTilingData_.useTempBuf;
    } else if (tilingMode_ == GroupedBiasAddGradTilingModeArch35::CUT_G_MODE) {
        auto* tilingData = context_->GetTilingData<GroupedBiasAddGradCutGTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
        tilingData->cutGDim = cutGTilingData_.cutGDim;
        tilingData->cutHDim = cutGTilingData_.cutHDim;
        tilingData->blockFactor = cutGTilingData_.blockFactor;
        tilingData->blockTailFactor = cutGTilingData_.blockTailFactor;
        tilingData->blockTailStartIndex = cutGTilingData_.blockTailStartIndex;
        tilingData->inputShape[0] = cutGTilingData_.inputShape[0];
        tilingData->inputShape[1] = cutGTilingData_.inputShape[1];
        tilingData->ubHTailFactor = cutGTilingData_.ubHTailFactor;
        tilingData->useUbSize = cutGTilingData_.useUbSize;
        tilingData->groupedIdxSize = cutGTilingData_.groupedIdxSize;
        tilingData->outputSize = cutGTilingData_.outputSize;
        tilingData->maxOutputElements = cutGTilingData_.maxOutputElements;
        tilingData->groupIdxType = cutGTilingData_.groupIdxType;
        tilingData->useTempBuf = cutGTilingData_.useTempBuf;
    } else if (tilingMode_ == GroupedBiasAddGradTilingModeArch35::EMPTY_TENSOR) {
        auto* tilingData = context_->GetTilingData<GroupedBiasAddGradEmptyTensorTilingData>();
        tilingData->blockDim = 1;
    }
    // REDUCE_SUM_3D mode: placeholder, no tiling data to save yet

    PrintTilingInfo();
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedBiasAddGradTilingArch35::GetTilingKey() const
{
    uint32_t templateNum = static_cast<uint32_t>(tilingMode_);
    // GroupIdxDtype: 0 = int32, 1 = int64
    uint32_t groupIdxDtypeVal = (groupIdxDtype_ == ge::DT_INT64) ? 1 : 0;
    uint64_t tilingKey;
    Ops::Base::ReduceTilingKey reduceTiling;
    GEN_REDUCE_TILING_KEY(tilingKey, reduceTiling, templateNum, groupIdxDtypeVal);
    return tilingKey;
}

void GroupedBiasAddGradTilingArch35::PrintTilingInfo() const
{
    if (tilingMode_ == GroupedBiasAddGradTilingModeArch35::CUT_H_MODE) {
        OP_LOGI(context_, "blockFactor %ld", cutHTilingData_.blockFactor);
        OP_LOGI(context_, "blockTailFactor %ld", cutHTilingData_.blockTailFactor);
        OP_LOGI(context_, "groupIdxDim %ld", cutHTilingData_.groupIdxDim);
        OP_LOGI(context_, "inputShape[0] %ld", cutHTilingData_.inputShape[0]);
        OP_LOGI(context_, "inputShape[1] %ld", cutHTilingData_.inputShape[1]);
        OP_LOGI(context_, "hTailFactor %ld", cutHTilingData_.hTailFactor);
        OP_LOGI(context_, "useUbSize %ld", cutHTilingData_.useUbSize);
        OP_LOGI(context_, "groupedIdxSize %ld", cutHTilingData_.groupedIdxSize);
        OP_LOGI(context_, "outputSize %ld", cutHTilingData_.outputSize);
        OP_LOGI(context_, "groupIdxType %d", cutHTilingData_.groupIdxType);
        OP_LOGI(context_, "useTempBuf %ld", cutHTilingData_.useTempBuf);
    } else if (tilingMode_ == GroupedBiasAddGradTilingModeArch35::CUT_G_MODE) {
        OP_LOGI(context_, "cutGDim %ld", cutGTilingData_.cutGDim);
        OP_LOGI(context_, "cutHDim %ld", cutGTilingData_.cutHDim);
        OP_LOGI(context_, "blockFactor %ld", cutGTilingData_.blockFactor);
        OP_LOGI(context_, "blockTailFactor %ld", cutGTilingData_.blockTailFactor);
        OP_LOGI(context_, "blockTailStartIndex %ld", cutGTilingData_.blockTailStartIndex);
        OP_LOGI(context_, "inputShape[0] %ld", cutGTilingData_.inputShape[0]);
        OP_LOGI(context_, "inputShape[1] %ld", cutGTilingData_.inputShape[1]);
        OP_LOGI(context_, "ubHTailFactor %ld", cutGTilingData_.ubHTailFactor);
        OP_LOGI(context_, "useUbSize %ld", cutGTilingData_.useUbSize);
        OP_LOGI(context_, "groupedIdxSize %ld", cutGTilingData_.groupedIdxSize);
        OP_LOGI(context_, "outputSize %ld", cutGTilingData_.outputSize);
        OP_LOGI(context_, "groupIdxType %d", cutGTilingData_.groupIdxType);
        OP_LOGI(context_, "useTempBuf %ld", cutGTilingData_.useTempBuf);
    }
}

} // namespace optiling
