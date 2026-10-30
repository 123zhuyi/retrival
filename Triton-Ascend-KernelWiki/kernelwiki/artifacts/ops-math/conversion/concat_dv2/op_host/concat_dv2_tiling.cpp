/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_dv2_tiling.cpp
 * \brief
 */
#include "concat_dv2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include <tuple>

namespace optiling {

static void ConcatDV2PrintParam(gert::TilingContext* context, ConcatDV2TilingData& tiling)
{
    auto nodeName = context;
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ConcatDV2 tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> op [TilingData]: elePerLoop = %ld", tiling.get_elePerLoop());
    OP_LOGD(nodeName, ">>> op [TilingData]: elePercore = %ld", tiling.get_elePercore());
    OP_LOGD(nodeName, ">>> op [TilingData]: ubLoop = %u", tiling.get_ubLoop());
    OP_LOGD(nodeName, ">>> op [TilingData]: eleTailCore = %u", tiling.get_eleTailCore());
    OP_LOGD(nodeName, ">>> op [TilingData]: ubLoopTail = %ld", tiling.get_ubLoopTail());
    OP_LOGD(nodeName, ">>> op [TilingData]: sameDimSize = %ld", tiling.get_sameDimSize());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> End print ConcatDV2 tiling data <<<<<<<<<<<<<<<<");
}
static std::tuple<std::vector<int64_t>, int64_t, int64_t> CalculatePrefixSumAndSizes(gert::TilingContext* context)
{
    // 前序和——用于分核分块
    auto computeNodeInfo = context->GetComputeNodeInfo();
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();

    std::vector<int64_t> prefixSum(inputNum + 1, 0);
    int64_t sameDimSize = 0;
    int64_t outputSize = 0;

    for (uint32_t i = 0; i < inputNum; i++) {
        auto inputTensorShapePtr = context->GetDynamicInputShape(0, i);
        gert::Shape inputTensorShape = inputTensorShapePtr->GetStorageShape();
        int64_t inputSizes = 1;
        size_t inputTensorDimNum = inputTensorShape.GetDimNum();
        for (size_t j = 0; j < inputTensorDimNum; j++) {
            inputSizes *= inputTensorShape.GetDim(j);
        }
        if (i == 0) {
            sameDimSize = inputSizes / inputTensorShape.GetDim(0);
        }
        prefixSum[i + 1] = prefixSum[i] + inputSizes;
    }
    outputSize = prefixSum[inputNum];
    return std::make_tuple(prefixSum, sameDimSize, outputSize);
}
static ge::graphStatus CalculateCoreDistribution(
    gert::TilingContext* context, ConcatDV2Tiling& tilingdata, int64_t outputSize, int32_t coreNum,
    uint32_t elePerUbBlock)
{
    if (coreNum == 0) {
        OP_LOGD(context, "coreNum should not be equal to 0.");
        return ge::GRAPH_FAILED;
    }

    if (elePerUbBlock == 0) {
        OP_LOGD(context, "elePerUbBlock should not be equal to 0.");
        return ge::GRAPH_FAILED;
    }

    tilingdata.elePercore = (outputSize + coreNum - 1) / coreNum;
    int64_t elePercoreAlign = (tilingdata.elePercore + elePerUbBlock - 1) / elePerUbBlock * elePerUbBlock;
    int64_t usedCoreNum = (outputSize + elePercoreAlign - 1) / elePercoreAlign;
    int64_t eleTailCore = outputSize - (usedCoreNum - 1) * elePercoreAlign;
    tilingdata.elePercore = elePercoreAlign;
    tilingdata.usedCoreNum = usedCoreNum;
    tilingdata.eleTailCore = eleTailCore;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus CalculateBlockDistribution(
    gert::TilingContext* context, ConcatDV2Tiling& tilingdata, uint32_t elePerUbBlock, uint64_t ubSize)
{
    if (elePerUbBlock == 0) {
        OP_LOGD(context, "elePerUbBlock should not be equal to 0.");
        return ge::GRAPH_FAILED;
    }
    uint32_t elePerLoop = (ubSize / tilingdata.dtypeSize + elePerUbBlock - 1) / elePerUbBlock * elePerUbBlock;
    int64_t ubLoop = (tilingdata.elePercore + elePerLoop - 1) / elePerLoop;
    uint32_t ubLoopTail = (tilingdata.eleTailCore + elePerLoop - 1) / elePerLoop;
    tilingdata.elePerLoop = elePerLoop;
    tilingdata.ubLoop = ubLoop;
    tilingdata.ubLoopTail = ubLoopTail;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Tiling4ConcatDV2(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto [prefixSum, sameDimSize, outputSize] = CalculatePrefixSumAndSizes(context);
    // 获取数据类型
    auto inputDesc = context->GetDynamicInputDesc(0, 0);
    auto inputDataType = inputDesc->GetDataType();
    ConcatDV2Tiling tilingdata;
    tilingdata.dtypeSize = ge::GetSizeByDataType(inputDataType);
    if (tilingdata.dtypeSize == 0) {
        OP_LOGD(context, "dtypeSize should not be equal to 0."); return ge::GRAPH_FAILED;
    }
    uint32_t elePerUbBlock = UB_BLOCK_SIZE / tilingdata.dtypeSize;
    // 分核分块
    CalculateCoreDistribution(context, tilingdata, outputSize, coreNum, elePerUbBlock);
    CalculateBlockDistribution(context, tilingdata, elePerUbBlock, ubSize);
    // 计算偏移
    uint32_t inputNum = prefixSum.size() - 1;
    int64_t endTensorIdx[TILING_ARRAY_LENGTH];
    int64_t endTensorOffset[TILING_ARRAY_LENGTH];
    for (int64_t i = 1; i < tilingdata.usedCoreNum; i++) {
        for (int64_t j = 1; j < inputNum + 1; j++) {
            if (prefixSum[j] >= i * tilingdata.elePercore) {
                endTensorIdx[i - 1] = j - 1;
                break;
            }
        }
        endTensorOffset[i - 1] = i * tilingdata.elePercore - prefixSum[endTensorIdx[i - 1]];
    }
    endTensorIdx[tilingdata.usedCoreNum - 1] = inputNum - 1;
    endTensorOffset[tilingdata.usedCoreNum - 1] = prefixSum[inputNum] - prefixSum[inputNum - 1];
    ConcatDV2TilingData tiling;
    tiling.set_elePerLoop(tilingdata.elePerLoop);
    tiling.set_elePercore(tilingdata.elePercore);
    tiling.set_ubLoop(tilingdata.ubLoop);
    tiling.set_eleTailCore(tilingdata.eleTailCore);
    tiling.set_ubLoopTail(tilingdata.ubLoopTail);
    tiling.set_sameDimSize(sameDimSize);
    tiling.set_endTensorIdx(endTensorIdx);
    tiling.set_endTensorOffset(endTensorOffset);
    context->SetTilingKey(TILING_KEY);
    ConcatDV2PrintParam(context, tiling);
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    *workspaceSize = sysWorkspaceSize;
    context->SetBlockDim(tilingdata.usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ConcatDV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForConcatDV2 start.");
    OP_LOGD(context->GetNodeName(), "TilingPrepareForConcatDV2 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ConcatDV2).Tiling(Tiling4ConcatDV2).TilingParse<Tiling4ConcatDV2CompileInfo>(TilingPrepare4ConcatDV2);
} // namespace optiling