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
 * \file sinkhorn.cc
 * \brief
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "platform/platform_ascendc.h"
#include "sinkhorn_tiling.h"
#include <log/log.h>
#include <tiling/platform/platform_ascendc.h>
#include <platform/platform_infos_def.h>


namespace {
constexpr static float UB_USAGE = 0.85f;
constexpr static uint32_t BLOCK_SIZE = 256;
constexpr static uint32_t ROW_BLOCK_SIZE = 32;
constexpr static uint32_t MAX_TILE_ROW = 127 * 32;    // 数据复制时，blockCount最大为4095，向下对齐到 127 * 32
constexpr static uint32_t WORKSPACE_HEADER_SIZE = 64; // 64B对齐
} // namespace

namespace optiling {
ge::graphStatus TilingPrepareForSinkhorn(gert::TilingParseContext* context);
ge::graphStatus TilingForSinkhorn(gert::TilingContext* context);

class SinkhornTiling
{
public:
    explicit SinkhornTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint();

private:
    inline ge::graphStatus InitWS();
    inline ge::graphStatus InitTiling();

    SinkhornTilingData tiling;

    gert::TilingContext* tilingContext = nullptr;

    // ub对齐后长度
    uint64_t totalLengthAlignedWithBlock;

    // context传递
    uint64_t tilingKey = 1;
    uint64_t numBlocks = 0;

    // tiling data传递
    uint64_t formerNum = 0;    // former 数量
    uint64_t formerRow = 0;    // former cost行数
    uint64_t formerLength = 0; // former cost总长

    uint64_t formerTileNum = 0;        // former Tile数量
    uint64_t formerLastTileRow = 0;    // fomer last Tile行数
    uint64_t formerLastTileLength = 0; // fomer last Tile长度

    uint64_t tailNum = 0;    // tail 数量
    uint64_t tailRow = 0;    // tail cost行数
    uint64_t tailLength = 0; // tail cost总长

    uint64_t tailTileNum = 0;        // tail Tile数量
    uint64_t tailLastTileRow = 0;    // tail last Tile行数
    uint64_t tailLastTileLength = 0; // tail last Tile长度

    uint64_t tileRow = 0;    // Tile行数(非Last)
    uint64_t tileLength = 0; // Tile长度(非Last)

    uint64_t totalRow = 0;        // 总行数
    uint64_t totalCol = 0;        // 总列数
    uint64_t totalColAligned = 0; // 对齐后的总列数

    float tol = 0.0001f; // 误差

    size_t userWorkspaceSize; // workspace大小

    // 求单个元素大小
    uint64_t sizeOfDataType = 1;
};

ge::graphStatus SinkhornTiling::Init()
{
    OP_LOGD(tilingContext, "Tiling init start.");

    auto costShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, costShape);
    auto costStorageShape = costShape->GetStorageShape();
    totalRow = costStorageShape.GetDim(0);
    totalCol = costStorageShape.GetDim(1);

    auto attrs = tilingContext->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() != 0) {
        const float* tolPtr = attrs->GetAttrPointer<float>(0);
        if (tolPtr != nullptr) {
            tol = *tolPtr;
        }
    }

    auto compileInfo = reinterpret_cast<const SinkhornCompileInfo*>(tilingContext->GetCompileInfo());
    uint64_t aivNum = compileInfo->aivNum; // Vector核数量
    uint64_t ubSize = compileInfo->ubSize; // ubSize大小

    uint64_t dataType = tilingContext->GetInputDesc(0)->GetDataType();
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_BF16:
            sizeOfDataType = sizeof(float);
            break;
        case ge::DT_FLOAT16:
            sizeOfDataType = sizeof(uint16_t);
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    // 一个block存放的元素
    uint32_t rowAlignNum = ROW_BLOCK_SIZE / sizeOfDataType; // 32B对齐 32/<4|2> = 8|16
    totalColAligned = ((totalCol + rowAlignNum - 1) / rowAlignNum) * rowAlignNum;

    uint32_t usableUbSize = static_cast<uint32_t>(ubSize * UB_USAGE);

    // UB可以容纳的行                costIn         costOut       4 * d0
    tileRow = usableUbSize / ((totalColAligned + totalColAligned + 4) * sizeOfDataType);

    // 去除4*d1空间
    tileRow -= 4;

    // 保护处理， 数据复制时，blockCount最大为4095，向下对齐到 127 * 32
    if (tileRow > MAX_TILE_ROW) {
        tileRow = MAX_TILE_ROW;
    }

    tileLength = tileRow * totalCol;

    // block数量
    uint32_t ubNum = (totalRow + tileRow - 1) / tileRow;

    // 运行核数
    numBlocks = (ubNum > aivNum) ? aivNum : ubNum;

    tilingKey = dataType;

    InitTiling();
    InitWS();

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SinkhornTiling::InitTiling()
{
    // 切分流程
    formerNum = totalRow % numBlocks;
    if (formerNum == 0) {
        formerNum = numBlocks;
    }
    tailNum = numBlocks - formerNum;

    formerRow = (totalRow + numBlocks - 1) / numBlocks;
    formerLength = formerRow * totalCol;
    formerTileNum = (formerRow + tileRow - 1) / tileRow;
    formerLastTileRow = formerRow % tileRow;
    if (formerLastTileRow == 0) {
        formerLastTileRow = tileRow;
    }
    formerLastTileLength = formerLastTileRow * totalCol;

    if (tailNum > 0) {
        tailRow = (totalRow - formerRow * formerNum) / tailNum; // 一定能整除
        tailLength = tailRow * totalCol;
        tailTileNum = (tailRow + tileRow - 1) / tileRow;
        tailLastTileRow = tailRow % tileRow;
        if (tailLastTileRow == 0) {
            tailLastTileRow = tileRow;
        }
        tailLastTileLength = tailLastTileRow * totalCol;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SinkhornTiling::InitWS()
{
    // 头(64B)
    userWorkspaceSize = WORKSPACE_HEADER_SIZE;
    // d0
    userWorkspaceSize += totalRow * sizeof(float);

    // d1 block
    userWorkspaceSize += numBlocks * totalCol * sizeof(float);

    // d1/d1 new global
    userWorkspaceSize += (totalCol + totalCol) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinkhornTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");

    // context传递
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(numBlocks);

    // tiling data传递
    tiling.set_formerNum(formerNum);
    tiling.set_formerRow(formerRow);
    tiling.set_formerLength(formerLength);

    tiling.set_formerTileNum(formerTileNum);
    tiling.set_formerLastTileRow(formerLastTileRow);
    tiling.set_formerLastTileLength(formerLastTileLength);

    tiling.set_tailNum(tailNum);
    tiling.set_tailRow(tailRow);
    tiling.set_tailLength(tailLength);

    tiling.set_tailTileNum(tailTileNum);
    tiling.set_tailLastTileRow(tailLastTileRow);
    tiling.set_tailLastTileLength(tailLastTileLength);

    tiling.set_tileRow(tileRow);
    tiling.set_tileLength(tileLength);

    tiling.set_totalRow(totalRow);
    tiling.set_totalCol(totalCol);
    tiling.set_totalColAligned(totalColAligned);

    tiling.set_tol(tol);

    tiling.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaces入参所需workspace的块数。当前限制使用一块。
    auto compileInfo = reinterpret_cast<const SinkhornCompileInfo*>(tilingContext->GetCompileInfo());
    currentWorkspace[0] =
        userWorkspaceSize +
        compileInfo->sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间框架来申请并管理。

    OP_LOGD(tilingContext, "userWorkspaceSize: %lu.", userWorkspaceSize);
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void SinkhornTiling::TilingDataPrint()
{
    OP_LOGD(tilingContext, "            tilingKey: %lu", tilingContext->GetTilingKey());
    OP_LOGD(tilingContext, "             numBlocks: %u", tilingContext->GetBlockDim());

    OP_LOGD(tilingContext, "            formerNum: %lu", tiling.get_formerNum());
    OP_LOGD(tilingContext, "            formerRow: %lu", tiling.get_formerRow());
    OP_LOGD(tilingContext, "         formerLength: %lu", tiling.get_formerLength());

    OP_LOGD(tilingContext, "        formerTileNum: %lu", tiling.get_formerTileNum());
    OP_LOGD(tilingContext, "    formerLastTileRow: %lu", tiling.get_formerLastTileRow());
    OP_LOGD(tilingContext, " formerLastTileLength: %lu", tiling.get_formerLastTileLength());

    OP_LOGD(tilingContext, "              tailNum: %lu", tiling.get_tailNum());
    OP_LOGD(tilingContext, "              tailRow: %lu", tiling.get_tailRow());
    OP_LOGD(tilingContext, "           tailLength: %lu", tiling.get_tailLength());

    OP_LOGD(tilingContext, "          tailTileNum: %lu", tiling.get_tailTileNum());
    OP_LOGD(tilingContext, "      tailLastTileRow: %lu", tiling.get_tailLastTileRow());
    OP_LOGD(tilingContext, "   tailLastTileLength: %lu", tiling.get_tailLastTileLength());

    OP_LOGD(tilingContext, "              tileRow: %lu", tiling.get_tileRow());
    OP_LOGD(tilingContext, "           tileLength: %lu", tiling.get_tileLength());

    OP_LOGD(tilingContext, "             totalRow: %lu", tiling.get_totalRow());
    OP_LOGD(tilingContext, "             totalCol: %lu", tiling.get_totalCol());
    OP_LOGD(tilingContext, "      totalColAligned: %lu", tiling.get_totalColAligned());

    OP_LOGD(tilingContext, "                  tol: %f", tiling.get_tol());

    OP_LOGD(tilingContext, "    userWorkspaceSize: %lu", userWorkspaceSize);
}

ge::graphStatus TilingForSinkhorn(gert::TilingContext* context)
{
    SinkhornTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForSinkhorn(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForSinkhorn start.");
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 将aivNum、workSpaveSize、ubSize的变量放到compileInfo中
    auto compileInfo = context->GetCompiledInfo<SinkhornCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "compileInfo->aivNum is %lu.", compileInfo->aivNum);

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context, "compileInfo->sysWorkspaceSize is %lu.", compileInfo->sysWorkspaceSize);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_LOGD(context, "compileInfo->ubSize is %lu.", compileInfo->ubSize);

    OP_LOGD(context, "TilingPrepareForSinkhorn end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sinkhorn).Tiling(TilingForSinkhorn).TilingParse<SinkhornCompileInfo>(TilingPrepareForSinkhorn);
} // namespace optiling
