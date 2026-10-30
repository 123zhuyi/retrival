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
 * \file masked_select_v3_tiling.cpp
 * \brief
 */
#include <cmath>
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"
#include "masked_select_v3_tiling.h"

namespace {
constexpr static uint64_t BLOCK_SIZE = 256;
constexpr static uint32_t DOUBLE_BUFFER = 2;
constexpr static float UB_USAGE = 0.65f;
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
} // namespace

namespace optiling {
class MaskedSelectV3Tiling {
public:
    explicit MaskedSelectV3Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    MaskedSelectV3TilingData tiling;

    gert::TilingContext* tilingContext = nullptr;
    // 总元素个数
    uint64_t totalLength = tilingContext->GetInputShape(0)->GetStorageShape().GetShapeSize();

    // ub对齐后长度
    uint64_t totalLengthAlignedWithBlock = 0;

    uint64_t tilingKey = 1;
    uint64_t formerNum = 0;
    uint64_t formerLength = 0;
    uint64_t formerTileNum = 0;
    uint64_t formerTileLength = 0;
    uint64_t formerLastTileLength = 0;

    uint64_t tailNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailTileNum = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailLastTileLength = 0;

    uint64_t numBlocks = 0;

    // 求单个元素大小
    uint64_t sizeOfDataType = 1;
    uint64_t dataType = tilingContext->GetInputDesc(0)->GetDataType();
};
ge::graphStatus MaskedSelectV3Tiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling init start.");

    auto compileInfo = reinterpret_cast<const MaskedSelectV3CompileInfo*>(tilingContext->GetCompileInfo());
    uint64_t aivNum = compileInfo->aivNum; // Vector核数量
    uint64_t ubSize = compileInfo->ubSize; // ubSize大小
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            sizeOfDataType = sizeof(int32_t);
            break;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            sizeOfDataType = sizeof(int64_t);
            break;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            sizeOfDataType = sizeof(int16_t);
            break;
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            sizeOfDataType = sizeof(int8_t);
            break;
    }

    // 一个block存放的元素
    uint64_t ALIGN_NUM = BLOCK_SIZE / sizeOfDataType; // 256/<8>=32

    // ub对齐后长度
    totalLengthAlignedWithBlock = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

    float total_fact = static_cast<float>(sizeof(uint8_t)) + static_cast<float>(sizeOfDataType) * 3.0f;
    if (sizeOfDataType == sizeof(int64_t)) {
        total_fact += static_cast<float>(sizeof(float)) +
                      (1.0f / static_cast<float>((sizeof(int64_t)))) * static_cast<float>(INT64_LENGTH_IN_INT32);
    } else {
        total_fact += static_cast<float>(sizeof(int16_t)) + (1.0f / static_cast<float>(sizeof(int64_t)));
    }
    if (sizeOfDataType == 1) {
        total_fact += static_cast<float>(sizeof(int32_t));
    }

    // 核内拆分，策略是尽可能的填满ub_size,最后一包单独处理，
    float tmp = (static_cast<float>(sizeOfDataType)) * 1.0f / total_fact;
    // ub_block在ascend C中不能全部被用来作为输入输出，给了13/20系数。计算出x 一次最多能处理多少block数量
    uint64_t ubBlockNum = static_cast<uint64_t>((ubSize * UB_USAGE * tmp) / BLOCK_SIZE);
    OP_LOGD(tilingContext->GetNodeName(), "ubBlockNum: %u.", ubBlockNum);
    if (ubBlockNum % DOUBLE_BUFFER != 0) {
        ubBlockNum--;
    }

    OP_LOGD(tilingContext->GetNodeName(), "totalLength: %lu.", totalLength);
    // ub能放的元素个数
    uint64_t ubLength = ubBlockNum * ALIGN_NUM;
    // block数量
    uint64_t ubNum = (totalLengthAlignedWithBlock + ubLength - 1) / ubLength;

    // 运行核数
    numBlocks = (ubNum > aivNum) ? aivNum : ubNum;
    tilingContext->SetBlockDim(numBlocks);

    tilingKey = sizeOfDataType;
    tilingContext->SetTilingKey(tilingKey);

    // 切分流程
    formerNum = totalLength % numBlocks;
    if (formerNum == 0u) {
        formerNum = numBlocks;
    }
    tailNum = numBlocks - formerNum;

    formerLength = (totalLength + numBlocks - 1) / numBlocks;
    formerTileNum = (formerLength + ubLength - 1) / ubLength;
    formerTileLength = ubLength;
    formerLastTileLength = formerLength % ubLength;
    if (formerLastTileLength == 0u) {
        formerLastTileLength = ubLength;
    }

    if (tailNum > 0u) {
        tailLength = (totalLength - formerLength * formerNum) / tailNum; // 一定可能整出
        tailTileNum = (tailLength + ubLength - 1) / ubLength;
        tailTileLength = ubLength;
        tailLastTileLength = tailLength % ubLength;
        if (tailLastTileLength == 0u) {
            tailLastTileLength = ubLength;
        }
    }

    OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSelectV3Tiling::RunKernelTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");
    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_formertileNum(formerTileNum);
    tiling.set_formertileLength(formerTileLength);
    tiling.set_formerlasttileLength(formerLastTileLength);

    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tailtileNum(tailTileNum);
    tiling.set_tailtileLength(tailTileLength);
    tiling.set_taillasttileLength(tailLastTileLength);
    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto compileInfo = reinterpret_cast<const MaskedSelectV3CompileInfo*>(tilingContext->GetCompileInfo());
    uint32_t sysWorkspaceSize = compileInfo->workSpaceSize;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaces入参所需workspace的块数。当前限制使用一块。
    size_t usrSize = totalLengthAlignedWithBlock * sizeOfDataType + numBlocks * 64u;
    OP_LOGD(tilingContext->GetNodeName(), "usrWorkspaceSize: %lu.", usrSize);
    currentWorkspace[0] =
        usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间框架来申请并管理。
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void MaskedSelectV3Tiling::TilingDataPrint()
{
    OP_LOGD(tilingContext->GetNodeName(), "sizeOfDataType: %lu.", sizeOfDataType);
    OP_LOGD(tilingContext->GetNodeName(), "formerNum: %lu.", tiling.get_formerNum());
    OP_LOGD(tilingContext->GetNodeName(), "formerLength: %lu.", tiling.get_formerLength());
    OP_LOGD(tilingContext->GetNodeName(), "formerTileNum: %lu.", tiling.get_formertileNum());
    OP_LOGD(tilingContext->GetNodeName(), "formerTileLength: %lu.", tiling.get_formertileLength());
    OP_LOGD(tilingContext->GetNodeName(), "formerLastTileLength: %lu.", tiling.get_formerlasttileLength());
    OP_LOGD(tilingContext->GetNodeName(), "tailNum: %lu.", tiling.get_tailNum());
    OP_LOGD(tilingContext->GetNodeName(), "tailLength: %lu.", tiling.get_tailLength());
    OP_LOGD(tilingContext->GetNodeName(), "tailTileNum: %lu.", tiling.get_tailtileNum());
    OP_LOGD(tilingContext->GetNodeName(), "tailTileLength: %lu.", tiling.get_tailtileLength());
    OP_LOGD(tilingContext->GetNodeName(), "tailLastTileLength: %lu.", tiling.get_taillasttileLength());
}

ge::graphStatus TilingForMaskedSelectV3(gert::TilingContext* context)
{
    OP_LOGI("TilingForMaskedSelectV3 tiling start");
    auto compileInfo = reinterpret_cast<const MaskedSelectV3CompileInfo*>(context->GetCompileInfo());
    if (compileInfo->isRegbase) {
        OP_LOGI("TilingForMaskedSelectV3IsRegbaseSocVersion tiling start");
        return TilingForMaskedSelectV3IsRegbaseSocVersion(context);
    }
    MaskedSelectV3Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init tiling object return failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForMaskedSelectV3(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForMaskedSelectV3", "Tiling context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForMaskedSelectV3 start.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 将aivNum、workSpaveSize、ubSize的变量放到compileInfo中
    auto compileInfo = context->GetCompiledInfo<MaskedSelectV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "compileInfo->aivNum is %lu.", compileInfo->aivNum);

    compileInfo->workSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context, "compileInfo->workSpaceSize is %lu.", compileInfo->workSpaceSize);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_LOGD(context, "compileInfo->ubSize is %lu.", compileInfo->ubSize);
    if (compileInfo->aivNum == 0UL || compileInfo->workSpaceSize == 0UL || compileInfo->ubSize == 0UL) {
        OP_LOGE(context, "Get compile info failed.");
        return ge::GRAPH_FAILED;
    }

    compileInfo->isRegbase = Ops::Base::IsRegbaseSocVersion(context);
    OP_LOGD(context, "compileInfo->isRegbase is %d.", compileInfo->isRegbase);

    OP_LOGD(context, "TilingPrepareForMaskedSelectV3 end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaskedSelectV3)
    .Tiling(TilingForMaskedSelectV3)
    .TilingParse<MaskedSelectV3CompileInfo>(TilingPrepareForMaskedSelectV3);
} // namespace optiling