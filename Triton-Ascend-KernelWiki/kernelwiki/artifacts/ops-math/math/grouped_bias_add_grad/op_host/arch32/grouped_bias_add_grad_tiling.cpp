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
 * \file grouped_bias_add_grad_tiling.cpp
 * \brief
 */

#include <sstream>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "grouped_bias_add_grad_tiling_base.h"
#include "grouped_bias_add_grad_tiling.h"

namespace optiling {
namespace groupedbiasaddgrad {

static const uint32_t ATTR_GROUP_IDX_TYPE_INDEX = 0;

ge::graphStatus GroupedBiasAddGradTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] GetPlatformInfo start running.");
    auto compileInfo = reinterpret_cast<const GroupedBiasAddGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    baseInfoOp_.vectorCoreNum = compileInfo->coreNum;
    OP_CHECK_IF(
        (baseInfoOp_.vectorCoreNum <= 0),
        OP_LOGE(nodeName_, "GroupedBiasAddGradTiling get num of vector core is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp_.ubSize = compileInfo->ubSize;
    OP_CHECK_IF(
        (baseInfoOp_.ubSize <= 0),
        OP_LOGE(nodeName_, "GroupedBiasAddGradTiling get ub size is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp_.ubSize -= RESERVED_UB_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::GetInputInfo()
{
    // grad_y 处理包含dtype、shape校验，获取G、H、C大小
    // 可选输入group_idx，包含shape校验
    // grad_bias 处理包含shape校验
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] GetInputInfo start running.");
    auto gradYInputDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYInputDesc);
    baseInfoOp_.gradYInputDtype = gradYInputDesc->GetDataType();
    OP_CHECK_IF(
        (baseInfoOp_.gradYInputDtype != ge::DT_FLOAT && baseInfoOp_.gradYInputDtype != ge::DT_FLOAT16 &&
         baseInfoOp_.gradYInputDtype != ge::DT_BF16),
        OP_LOGE(nodeName_, "the dtype of input grad_y should be one of FP32/FP16/BF16."), return ge::GRAPH_FAILED);

    auto gradYInputShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYInputShapePtr);
    auto gradYInputShape = gradYInputShapePtr->GetStorageShape();
    baseInfoOp_.gradYDimNum = gradYInputShape.GetDimNum();

    // group_idx
    auto groupIdxInputShapePtr = context_->GetOptionalInputShape(GROUP_IDX_INPUT_INDEX);
    if (groupIdxInputShapePtr == nullptr) {
        OP_CHECK_IF(
            (baseInfoOp_.gradYDimNum != THREE_NUM),
            OP_LOGE(nodeName_, "the input grad_y should be 3D tensor when group_idx is null."),
            return ge::GRAPH_FAILED);
        baseInfoOp_.dimG = gradYInputShape.GetDim(0);
        baseInfoOp_.dimC = gradYInputShape.GetDim(1);
        baseInfoOp_.dimH = gradYInputShape.GetDim(TWO_NUM);
    } else {
        OP_CHECK_IF(
            (baseInfoOp_.gradYDimNum != TWO_NUM || groupIdxInputShapePtr->GetStorageShape().GetDimNum() != 1),
            OP_LOGE(
                nodeName_,
                "the input grad_y should be 2D tensor when group_idx is not null. and group_idx must be 1D tensor."),
            return ge::GRAPH_FAILED);
        baseInfoOp_.existGroupIdx = 1;
        baseInfoOp_.dimGB = gradYInputShape.GetDim(0);
        baseInfoOp_.dimG = groupIdxInputShapePtr->GetStorageShape().GetShapeSize();
        baseInfoOp_.dimH = gradYInputShape.GetDim(1);
        auto groupIdxInputDesc = context_->GetInputDesc(GROUP_IDX_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIdxInputDesc);
        baseInfoOp_.groupIdxInputDtype = groupIdxInputDesc->GetDataType();
        OP_CHECK_IF(
            (baseInfoOp_.groupIdxInputDtype != ge::DT_INT32 && baseInfoOp_.groupIdxInputDtype != ge::DT_INT64),
            OP_LOGE(nodeName_, "the dtype of input group_idx should be INT32 or INT64."), return ge::GRAPH_FAILED);
    }

    if (baseInfoOp_.existGroupIdx == 1 && baseInfoOp_.dimG > INPUT_MAX_GROUP) {
        OP_LOGE(
            nodeName_, "input group_idx shape not support more than %ld, but got %ld.", INPUT_MAX_GROUP,
            baseInfoOp_.dimG);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::GetAttrInfo()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] GetAttrInfo start running.");
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrGroupIdxType = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrGroupIdxType);

    baseInfoOp_.groupIdxType = static_cast<int32_t>(*attrGroupIdxType);
    OP_CHECK_IF(
        (baseInfoOp_.groupIdxType != 0 && baseInfoOp_.groupIdxType != 1),
        OP_LOGE(nodeName_, "the value of group_idx_type should be 0 or 1, but got %d", baseInfoOp_.groupIdxType),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::CheckOutput()
{
    // grad_bias
    auto gradBiasOutputShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradBiasOutputShapePtr);
    auto gradBiasOutputShape = gradBiasOutputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (gradBiasOutputShape.GetDimNum() != TWO_NUM),
        OP_LOGE(nodeName_, "the dim of grad_bias should be 2, but got %zu.", gradBiasOutputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    auto gradBiasdim0 = gradBiasOutputShape.GetDim(0);
    auto gradBiasdim1 = gradBiasOutputShape.GetDim(1);
    OP_CHECK_IF(
        ((gradBiasdim0 != baseInfoOp_.dimG) || (gradBiasdim1 != baseInfoOp_.dimH)),
        OP_LOGE(
            nodeName_, "the shape of grad_bias should be [%ld, %ld], bug got [%ld, %ld].", baseInfoOp_.dimG,
            baseInfoOp_.dimH, gradBiasdim0, gradBiasdim1),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoSplitTiling()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] DoUnequalCTiling start running.");
    int64_t coreNum = baseInfoOp_.vectorCoreNum;
    int64_t totalSplitNum = Ops::Base::CeilDiv(baseInfoOp_.dimH, splitCoreOp_.baseH) * baseInfoOp_.dimG;
    splitCoreOp_.usedCoreNum = totalSplitNum > coreNum ? coreNum : totalSplitNum;
    splitCoreOp_.normalCoreProcessNum = Ops::Base::CeilDiv(totalSplitNum, splitCoreOp_.usedCoreNum);
    splitCoreOp_.tailCoreProcessNum = splitCoreOp_.normalCoreProcessNum - 1;
    int64_t tailCoreNum = splitCoreOp_.normalCoreProcessNum * splitCoreOp_.usedCoreNum - totalSplitNum;
    splitCoreOp_.normalCoreNum = splitCoreOp_.usedCoreNum - tailCoreNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoUnequalCPerformanceTiling()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] DoUnequalCTiling start running.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoTiling()
{
    // 分别走C等长和不等长模板切分，C不等长情况还可以根据H大小判断走高性能模板
    splitCoreOp_.baseH = H_BASE_SIZE / sizeof(float);
    int64_t avilableUbsize = baseInfoOp_.ubSize - H_BASE_SIZE * BUFFER_NUM - UB_GROUP_SUM_NUM * H_BASE_SIZE;
    splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
    DoSplitTiling();
    bool perfTemp = baseInfoOp_.dimG >= PERF_G_NUM && splitCoreOp_.usedCoreNum >= baseInfoOp_.vectorCoreNum &&
                    baseInfoOp_.dimG % TWO_NUM == 0 &&
                    baseInfoOp_.dimH / splitCoreOp_.baseH >= splitCoreOp_.usedCoreNum / TWO_NUM;
    int64_t unitLength = baseInfoOp_.groupIdxInputDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);
    if (baseInfoOp_.existGroupIdx == 0) {
        splitCoreOp_.loopCNum = Ops::Base::CeilDiv(baseInfoOp_.dimC, splitCoreOp_.baseC);
        splitCoreOp_.wsUnitNum = splitCoreOp_.loopCNum;
    } else if (perfTemp && baseInfoOp_.dimGB < INT32_MAX) {
        baseInfoOp_.performance = 1;
        int64_t groupIdxAlign = Ops::Base::CeilDiv(baseInfoOp_.dimG, BLOCK_SIZE / unitLength) * BLOCK_SIZE / unitLength;
        int64_t sortBuffer =
            Ops::Base::CeilDiv(baseInfoOp_.dimG, BLOCK_SIZE) * BLOCK_SIZE * sizeof(int32_t) * TWO_NUM * TWO_NUM;
        int64_t groupIdxExtraBuffer = groupIdxAlign * unitLength * TWO_NUM + sortBuffer;
        avilableUbsize = avilableUbsize - groupIdxExtraBuffer;
        splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
        splitCoreOp_.wsUnitNum = Ops::Base::CeilDiv(baseInfoOp_.dimGB, splitCoreOp_.baseC);
        splitCoreOp_.normalCoreProcessNum = (baseInfoOp_.dimG + 1) / TWO_NUM;
        splitCoreOp_.tailCoreProcessNum = baseInfoOp_.dimG / TWO_NUM;
        splitCoreOp_.normalCoreNum = splitCoreOp_.usedCoreNum / TWO_NUM;
    } else {
        int64_t groupIdxAlign = (baseInfoOp_.dimG * unitLength + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        avilableUbsize = avilableUbsize - groupIdxAlign - groupIdxAlign;
        splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
        splitCoreOp_.wsUnitNum = Ops::Base::CeilDiv(baseInfoOp_.dimGB, splitCoreOp_.baseC);
    }
    return ge::GRAPH_SUCCESS;
}

void GroupedBiasAddGradTiling::SaveToTilingData()
{
    tilingData_.set_usedCoreNum(splitCoreOp_.usedCoreNum);
    tilingData_.set_normalCoreNum(splitCoreOp_.normalCoreNum);
    tilingData_.set_normalCoreProcessNum(splitCoreOp_.normalCoreProcessNum);
    tilingData_.set_tailCoreProcessNum(splitCoreOp_.tailCoreProcessNum);
    tilingData_.set_wsUnitNum(splitCoreOp_.wsUnitNum);
    tilingData_.set_dimG(baseInfoOp_.dimG);
    tilingData_.set_dimC(baseInfoOp_.dimC);
    tilingData_.set_dimH(baseInfoOp_.dimH);
    tilingData_.set_dimGB(baseInfoOp_.dimGB);
    tilingData_.set_baseH(splitCoreOp_.baseH);
    tilingData_.set_baseC(splitCoreOp_.baseC);
    tilingData_.set_loopCNum(splitCoreOp_.loopCNum);
    tilingData_.set_groupIdxType(baseInfoOp_.groupIdxType);
}

ge::graphStatus GroupedBiasAddGradTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t workspaceSize = WORKSPACE_BASE_CAL;
    workspaces[0] = workspaceSize;
    workspaces[0] += splitCoreOp_.wsUnitNum * H_BASE_SIZE * splitCoreOp_.usedCoreNum;

    SaveToTilingData();

    context_->SetBlockDim(splitCoreOp_.usedCoreNum);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedBiasAddGradTiling::ComputeTiling(const std::vector<uint32_t>& args) const
{
    uint64_t result = 1000000UL;
    uint64_t startValue = 1;
    constexpr uint64_t incrementCoefficient = 10;
    for (auto it = args.rbegin(); it != args.rend(); ++it) {
        result += *it * startValue;
        startValue *= incrementCoefficient;
    }
    return result;
}

uint64_t GroupedBiasAddGradTiling::GetTilingKey() const
{
    // 计算tiling，kernel入口根据不同tiling进入不同模板
    DtypeEnum inDtype = DtypeEnum::FLOAT16;
    if (baseInfoOp_.gradYInputDtype == ge::DT_BF16) {
        inDtype = DtypeEnum::BFLOAT16;
    }
    if (baseInfoOp_.gradYInputDtype == ge::DT_FLOAT) {
        inDtype = DtypeEnum::FLOAT32;
    }

    uint32_t useUBSum = 0;
    if (splitCoreOp_.wsUnitNum <= UB_GROUP_SUM_NUM) {
        useUBSum = 1;
    }

    uint32_t groupIdxDtype = baseInfoOp_.groupIdxInputDtype == ge::DT_INT32 ? 0 : 1;
    auto tilingKey = ComputeTiling(
        {static_cast<uint32_t>(groupIdxDtype), baseInfoOp_.performance, useUBSum, baseInfoOp_.existGroupIdx,
         static_cast<uint32_t>(inDtype)});
    OP_LOGI(nodeName_, "[GroupedBiasAddGrad] GetTilingKey [%lu].", tilingKey);
    return tilingKey;
}

void GroupedBiasAddGradTiling::DumpTilingInfo() const
{
    std::ostringstream info;
    info << "baseInfoOp_.vectorCoreNum: " << baseInfoOp_.vectorCoreNum << std::endl;
    info << "baseInfoOp_.gradYDimNum: " << baseInfoOp_.gradYDimNum << std::endl;
    info << "baseInfoOp_.dimG: " << baseInfoOp_.dimG << std::endl;
    info << "baseInfoOp_.dimC: " << baseInfoOp_.dimC << std::endl;
    info << "baseInfoOp_.dimH: " << baseInfoOp_.dimH << std::endl;
    info << "baseInfoOp_.dimGB: " << baseInfoOp_.dimGB << std::endl;
    info << "baseInfoOp_.gradYInputDtype: " << baseInfoOp_.gradYInputDtype << std::endl;
    info << "baseInfoOp_.existGroupIdx: " << baseInfoOp_.existGroupIdx << std::endl;
    info << "baseInfoOp_.performance: " << baseInfoOp_.performance << std::endl;
    info << "baseInfoOp_.groupIdxType: " << baseInfoOp_.groupIdxType << std::endl;

    info << "splitCoreOp_.usedCoreNum: " << splitCoreOp_.usedCoreNum << std::endl;
    info << "splitCoreOp_.normalCoreNum: " << splitCoreOp_.normalCoreNum << std::endl;
    info << "splitCoreOp_.normalCoreProcessNum: " << splitCoreOp_.normalCoreProcessNum << std::endl;
    info << "splitCoreOp_.tailCoreProcessNum: " << splitCoreOp_.tailCoreProcessNum << std::endl;
    info << "splitCoreOp_.baseH: " << splitCoreOp_.baseH << std::endl;
    info << "splitCoreOp_.baseC: " << splitCoreOp_.baseC << std::endl;
    info << "splitCoreOp_.loopCNum: " << splitCoreOp_.loopCNum << std::endl;
    info << "splitCoreOp_.wsUnitNum: " << splitCoreOp_.wsUnitNum << std::endl;

    OP_LOGI(nodeName_, "%s", info.str().c_str());
}

ge::graphStatus GroupedBiasAddGradTiling::RunGroupedBiasAddGradTiling()
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = GetInputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetAttrInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGroupedBiasAddGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("GroupedBiasAddGrad", "context should not be nullptr."), return ge::GRAPH_FAILED);

    GroupedBiasAddGradTiling tiling(context);
    return tiling.RunGroupedBiasAddGradTiling();
}

ge::graphStatus TilingPrepareForGroupedBiasAddGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("GroupedBiasAddGrad", "context should not be nullptr."), return ge::GRAPH_FAILED);
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepareForGroupedBiasAddGrad start.");

    auto compileInfo = context->GetCompiledInfo<GroupedBiasAddGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(nodeName, "Failed to get core number."), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(nodeName, "Failed to get ub size."), return ge::GRAPH_FAILED);

    OP_LOGD(nodeName, "TilingPrepareForGroupedBiasAddGrad end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedBiasAddGrad)
    .Tiling(TilingForGroupedBiasAddGrad)
    .TilingParse<GroupedBiasAddGradCompileInfo>(TilingPrepareForGroupedBiasAddGrad);
} // namespace groupedbiasaddgrad
} // namespace optiling