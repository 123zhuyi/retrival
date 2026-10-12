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
 * \file triangulator_tiling.cpp
 * \brief
 */
#include <cmath>
#include <tuple>
#include "triangulator_tiling.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"

using namespace Ops::Base;

namespace optiling {
constexpr int64_t WORKSPACE_SIZE = 16777216; // 16M
constexpr int64_t RESERVED_UB_SIZE = 8192;   // 8k
constexpr int64_t FOUR_BUFFER = 4;
constexpr int64_t TWO_BUFFER = 2;
constexpr int64_t SEVEN_BUFFER = 8;
constexpr int64_t UB_BLOCK_SIZE = 32;

constexpr int64_t MIN_ROW_NUMBER = 128;
constexpr int64_t MIN_CORE_COE = 1;

constexpr int64_t OUTPUT_ZERO_MODE = 0;
constexpr int64_t OUTPUT_INPUT_MODE = 1;
constexpr int64_t OUTPUT_NORMAL_MODE = 2;
constexpr int64_t TINY_SHAPE_MODE = 3;
constexpr int64_t MEDIUM_SHAPE_MODE = 4;

constexpr size_t MASK_LENGTH = 4;
constexpr uint8_t MASK_BIT_WIDTH = 64;

constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t LARGE_BLOCK_FOR_MOVE = 1024;

constexpr int64_t TUPLE_INDEX_0 = 0;
constexpr int64_t TUPLE_INDEX_1 = 1;
constexpr int64_t TUPLE_INDEX_2 = 2;

int64_t static GetMod(const int64_t l_value, const int64_t r_value)
{
    if (r_value == 0) {
        return l_value;
    }
    return l_value % r_value;
}

int64_t static GetFloorDiv(int64_t u_value, int64_t d_value)
{
    int64_t res_value = 0;
    if (d_value == 0) {
        return u_value;
    }
    res_value = u_value / d_value;
    return res_value;
}

void TriangulatorTiling::SaveToTilingData()
{
    switch (splitCoreOp.tilingMode) {
        case OUTPUT_NORMAL_MODE:
            normalTilingData.set_bufferSize(baseInfoOp.bufferSize);
            normalTilingData.set_row(baseInfoOp.row);
            normalTilingData.set_col(baseInfoOp.col);
            normalTilingData.set_diagOffset(baseInfoOp.diagOffset);
            normalTilingData.set_usedCoreNum(splitCoreOp.usedCoreNum);
            normalTilingData.set_normalCoreProcessNum(splitCoreOp.normalCoreProcessNum);
            normalTilingData.set_tailCoreProcessNum(splitCoreOp.tailCoreProcessNum);
            normalTilingData.set_rowInner(splitCoreOp.rowInner);
            normalTilingData.set_colInner(splitCoreOp.colInner);
            break;
        case OUTPUT_INPUT_MODE:
        case OUTPUT_ZERO_MODE:
            tilingData.set_bufferSize(baseInfoOp.bufferSize);
            tilingData.set_usedCoreNum(splitCoreOp.usedCoreNum);
            tilingData.set_normalCoreProcessNum(splitCoreOp.normalCoreProcessNum);
            tilingData.set_tailCoreProcessNum(splitCoreOp.tailCoreProcessNum);
            break;
        case MEDIUM_SHAPE_MODE:
            mediumTilingData.set_bufferSize(baseInfoOp.bufferSize);
            mediumTilingData.set_col(baseInfoOp.col);
            mediumTilingData.set_row(baseInfoOp.row);
            mediumTilingData.set_copyCols(splitCoreOp.copyCols);
            mediumTilingData.set_usedCoreNum(splitCoreOp.usedCoreNum);
            mediumTilingData.set_normalCoreProcessNum(splitCoreOp.normalCoreProcessNum);
            mediumTilingData.set_tailCoreProcessNum(splitCoreOp.tailCoreProcessNum);
            mediumTilingData.set_highInner(splitCoreOp.highInner);
            mediumTilingData.set_highOuter(splitCoreOp.highOuter);
            mediumTilingData.set_highTail(splitCoreOp.highTail);
            mediumTilingData.set_headRows(splitCoreOp.headRows);
            mediumTilingData.set_midRows(splitCoreOp.midRows);
            mediumTilingData.set_tailRows(splitCoreOp.tailRows);
            break;
        case TINY_SHAPE_MODE:
            tinyTilingData.set_bufferSize(baseInfoOp.bufferSize);
            tinyTilingData.set_planeArea(baseInfoOp.col * baseInfoOp.row);
            tinyTilingData.set_usedCoreNum(splitCoreOp.usedCoreNum);
            tinyTilingData.set_normalCoreProcessNum(splitCoreOp.normalCoreProcessNum);
            tinyTilingData.set_tailCoreProcessNum(splitCoreOp.tailCoreProcessNum);
            tinyTilingData.set_highInner(splitCoreOp.highInner);
            tinyTilingData.set_highOuter(splitCoreOp.highOuter);
            tinyTilingData.set_highTail(splitCoreOp.highTail);
            tinyTilingData.set_highMask(splitCoreOp.highMask);
        default:
            break;
    }
}

void TriangulatorTiling::DumpTilingInfo() const
{
    OP_LOGD("Triangulator", "[Triangulator] DumpTilingInfo start running");

    std::ostringstream info;
    info << "operatorType[false:tril  true:triu] : " << operatorType_ << std::endl;

    info << "baseInfoOp.totalCoreNum: " << baseInfoOp.totalCoreNum << std::endl;
    info << "baseInfoOp.ubSize: " << baseInfoOp.ubSize << std::endl;
    info << "baseInfoOp.bufferSize: " << baseInfoOp.bufferSize << std::endl;
    info << "baseInfoOp.row: " << baseInfoOp.row << std::endl;
    info << "baseInfoOp.col: " << baseInfoOp.col << std::endl;
    info << "baseInfoOp.diagOffset: " << baseInfoOp.diagOffset << std::endl;
    info << "baseInfoOp.fusedFrontAxis: " << baseInfoOp.fusedFrontAxis << std::endl;
    info << "baseInfoOp.dtypeBytes: " << baseInfoOp.dtypeBytes << std::endl;
    info << "baseInfoOp.vRegSize: " << baseInfoOp.vRegSize << std::endl;

    info << "splitCoreOp.rowInner: " << splitCoreOp.rowInner << std::endl;
    info << "splitCoreOp.rowOuter: " << splitCoreOp.rowOuter << std::endl;
    info << "splitCoreOp.rowTail: " << splitCoreOp.rowTail << std::endl;
    info << "splitCoreOp.colInner: " << splitCoreOp.colInner << std::endl;
    info << "splitCoreOp.colOuter: " << splitCoreOp.colOuter << std::endl;
    info << "splitCoreOp.colTail: " << splitCoreOp.colTail << std::endl;

    info << "splitCoreOp.highInner: " << splitCoreOp.highInner << std::endl;
    info << "splitCoreOp.highOuter: " << splitCoreOp.highOuter << std::endl;
    info << "splitCoreOp.highTail: " << splitCoreOp.highTail << std::endl;
    for (size_t i = 0; i < MASK_LENGTH; ++i) {
        info << "splitCoreOp.highMask[" << i << "]: " << splitCoreOp.highMask[i] << " ";
    }
    info << std::endl;
    info << "splitCoreOp.baseBlockNum: " << splitCoreOp.baseBlockNum << std::endl;
    info << "splitCoreOp.normalCoreProcessNum: " << splitCoreOp.normalCoreProcessNum << std::endl;
    info << "splitCoreOp.usedCoreNum: " << splitCoreOp.usedCoreNum << std::endl;
    info << "splitCoreOp.tailCoreProcessNum: " << splitCoreOp.tailCoreProcessNum << std::endl;
    info << "splitCoreOp.tilingKey: " << splitCoreOp.tilingKey << std::endl;

    OP_LOGI("Triangulator", "%s", info.str().c_str());
}

ge::graphStatus TriangulatorTiling::GetInputInfo(const TriluTilingParams* params, uint8_t dtypeBytes)
{
    OP_LOGD(nodeName_, "[Triangulator] GetInputInfo start running.");
    OP_CHECK_NULL_WITH_CONTEXT(context_, params);

    baseInfoOp.row = params->row;
    baseInfoOp.col = params->col;
    baseInfoOp.diagOffset = params->diagonal;
    baseInfoOp.fusedFrontAxis = params->matrixNum;
    baseInfoOp.dtypeBytes = dtypeBytes;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriangulatorTiling::GetPlatformInfo(const TriluCompileInfo* compileInfo)
{
    OP_LOGD(nodeName_, "[Triangulator] GetPlatformInfo start running.");
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    baseInfoOp.totalCoreNum = static_cast<int64_t>(compileInfo->availableAICoreNum);

    OP_CHECK_IF(
        (baseInfoOp.totalCoreNum <= 0),
        OP_LOGE(nodeName_, "TriangulatorTiling get num of vector core is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp.ubSize = static_cast<int64_t>(compileInfo->availableUBSize);

    OP_CHECK_IF(
        (baseInfoOp.ubSize <= 0), OP_LOGE(nodeName_, "TriangulatorTiling get ub size is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp.ubSize -= RESERVED_UB_SIZE; // 可用UB空间
    baseInfoOp.vRegSize = GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

void TriangulatorTiling::ComputeTilingMode()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeTilingMode start running.");
    if ((!operatorType_ && baseInfoOp.diagOffset <= -baseInfoOp.row) ||
        (operatorType_ && baseInfoOp.diagOffset >= baseInfoOp.col)) {
        splitCoreOp.tilingMode = OUTPUT_ZERO_MODE;
    } else if (
        (!operatorType_ && baseInfoOp.diagOffset >= baseInfoOp.col - 1) ||
        (operatorType_ && baseInfoOp.diagOffset <= 1 - baseInfoOp.row)) {
        splitCoreOp.tilingMode = OUTPUT_INPUT_MODE;
    } else if (baseInfoOp.col * baseInfoOp.row * baseInfoOp.dtypeBytes <= baseInfoOp.vRegSize) {
        splitCoreOp.tilingMode = TINY_SHAPE_MODE;
    } else if (
        baseInfoOp.row * CeilAlign(baseInfoOp.col * baseInfoOp.dtypeBytes, baseInfoOp.vRegSize) * FOUR_BUFFER <=
        baseInfoOp.ubSize) {
        splitCoreOp.tilingMode = MEDIUM_SHAPE_MODE;
    } else {
        splitCoreOp.tilingMode = OUTPUT_NORMAL_MODE;
    }
}

void TriangulatorTiling::ComputeTinyInfo()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeTinyInfo start running.");
    baseInfoOp.bufferSize = FloorAlign(baseInfoOp.ubSize / FOUR_BUFFER, UB_BLOCK_SIZE);
    baseInfoOp.bufferEleNum = FloorDiv(baseInfoOp.bufferSize, baseInfoOp.dtypeBytes);
    int64_t tmpInner = FloorDiv(
        baseInfoOp.bufferEleNum * baseInfoOp.dtypeBytes,
        baseInfoOp.vRegSize); // pad to vreg size to avoid unaligned access
    int64_t tmpOuter = CeilDiv(baseInfoOp.fusedFrontAxis, tmpInner);
    int64_t minCoreNum = CeilDiv(baseInfoOp.totalCoreNum, MIN_CORE_COE);
    splitCoreOp.highOuter = std::max(minCoreNum, tmpOuter);
    splitCoreOp.highInner = CeilDiv(baseInfoOp.fusedFrontAxis, splitCoreOp.highOuter);
    splitCoreOp.highOuter = CeilDiv(baseInfoOp.fusedFrontAxis, splitCoreOp.highInner);
    splitCoreOp.baseBlockNum = splitCoreOp.highOuter;
    int64_t tmpTail = GetMod(baseInfoOp.fusedFrontAxis, splitCoreOp.highInner);
    splitCoreOp.highTail = tmpTail == 0 ? splitCoreOp.highInner : tmpTail;
    ComputeMask();
}

void TriangulatorTiling::ComputeMask()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeMask start running.");
    uint8_t idx = 0;
    for (auto r = 0; r < baseInfoOp.row; ++r) {
        for (auto c = 0; c < baseInfoOp.col; ++c) {
            if ((!operatorType_ && c <= r + baseInfoOp.diagOffset) ||
                (operatorType_ && c >= r + baseInfoOp.diagOffset)) {
                splitCoreOp.highMask[idx / MASK_BIT_WIDTH] |= 1UL << (idx % MASK_BIT_WIDTH);
            }
            ++idx;
        }
    }
}

void TriangulatorTiling::ComputeTilingKey()
{
    int64_t tenDigit = splitCoreOp.tilingMode;
    int64_t digit = baseInfoOp.dtypeBytes;
    splitCoreOp.tilingKey = tenDigit * DIGIT_TEN + digit;
}

std::tuple<int64_t, int64_t, int64_t> OutputZeroAndInputTiling(int64_t fusedNum, int64_t totalCoreNum, int64_t bitWidth)
{
    int64_t coreNum = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    if (fusedNum * bitWidth < LARGE_BLOCK_FOR_MOVE) { // 大块搬运，按照单次搬运不小于1024Byte计算
        coreNum = 1;
        normalCoreProcessNum = fusedNum;
        tailCoreProcessNum = fusedNum;
    } else {
        normalCoreProcessNum = CeilDiv(fusedNum, totalCoreNum);
        normalCoreProcessNum = (normalCoreProcessNum < GetFloorDiv(LARGE_BLOCK_FOR_MOVE, bitWidth)) ?
                                   GetFloorDiv(LARGE_BLOCK_FOR_MOVE, bitWidth) :
                                   normalCoreProcessNum;
        coreNum = CeilDiv(fusedNum, normalCoreProcessNum);
        tailCoreProcessNum = fusedNum - normalCoreProcessNum * (coreNum - 1);
    }
    return std::make_tuple(coreNum, normalCoreProcessNum, tailCoreProcessNum);
}

std::tuple<int64_t, int64_t, int64_t> OutputNormalTiling(int64_t fusedNum, int64_t totalCoreNum)
{
    int64_t normalCore = CeilDiv(fusedNum, totalCoreNum);
    int64_t coreNum = CeilDiv(fusedNum, normalCore);
    int64_t tailCore = fusedNum - normalCore * (coreNum - 1);
    return std::make_tuple(coreNum, normalCore, tailCore);
}

void TriangulatorTiling::ComputeNormalInfo()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeNormalInfo start running.");
    baseInfoOp.bufferSize = FloorAlign(baseInfoOp.ubSize / SEVEN_BUFFER, UB_BLOCK_SIZE);
    baseInfoOp.bufferEleNum = FloorDiv(baseInfoOp.bufferSize, baseInfoOp.dtypeBytes);
    int64_t tmp = FloorAlign(baseInfoOp.vRegSize, UB_BLOCK_SIZE) / baseInfoOp.dtypeBytes;
    if (baseInfoOp.col <= tmp) { // col can fit in vreg, small col
        tmp = CeilAlign(baseInfoOp.col, UB_BLOCK_SIZE / baseInfoOp.dtypeBytes);
        splitCoreOp.colInner = std::min(FloorAlign(baseInfoOp.bufferEleNum, tmp), CeilAlign(baseInfoOp.col, tmp));
        splitCoreOp.rowInner = std::min(FloorDiv(baseInfoOp.bufferEleNum, splitCoreOp.colInner), baseInfoOp.row);
    } else if (baseInfoOp.row <= MIN_ROW_NUMBER) { // row is small
        splitCoreOp.rowInner = baseInfoOp.row;
        splitCoreOp.colInner = std::min(FloorDiv(baseInfoOp.bufferEleNum, splitCoreOp.rowInner), baseInfoOp.col);
        if (splitCoreOp.colInner < tmp) {
            splitCoreOp.colInner = FloorAlign(splitCoreOp.colInner, UB_BLOCK_SIZE / baseInfoOp.dtypeBytes);
        } else {
            splitCoreOp.colInner = FloorAlign(splitCoreOp.colInner, tmp);
        }
    } else { // normal case
        splitCoreOp.colInner = std::min(FloorAlign(baseInfoOp.bufferEleNum, tmp), CeilAlign(baseInfoOp.col, tmp));
        splitCoreOp.rowInner = std::min(FloorDiv(baseInfoOp.bufferEleNum, splitCoreOp.colInner), baseInfoOp.row);
    }

    splitCoreOp.colOuter = CeilDiv(baseInfoOp.col, splitCoreOp.colInner);
    splitCoreOp.rowOuter = CeilDiv(baseInfoOp.row, splitCoreOp.rowInner);

    int64_t colTailTmp = GetMod(baseInfoOp.col, splitCoreOp.colInner);
    splitCoreOp.colTail = colTailTmp == 0 ? splitCoreOp.colInner : colTailTmp;
    int64_t rowTailTmp = GetMod(baseInfoOp.row, splitCoreOp.rowInner);
    splitCoreOp.rowTail = rowTailTmp == 0 ? splitCoreOp.rowInner : rowTailTmp;
    splitCoreOp.baseBlockNum = baseInfoOp.fusedFrontAxis * splitCoreOp.colOuter * splitCoreOp.rowOuter;
}

void TriangulatorTiling::ComputeMediumInfo()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeMediumInfo start running.");
    baseInfoOp.bufferSize = FloorAlign(baseInfoOp.ubSize / FOUR_BUFFER, UB_BLOCK_SIZE);
    baseInfoOp.bufferEleNum = FloorDiv(baseInfoOp.bufferSize, baseInfoOp.dtypeBytes);
    int64_t tmp = FloorAlign(baseInfoOp.vRegSize, UB_BLOCK_SIZE) / baseInfoOp.dtypeBytes;
    if (baseInfoOp.col <= tmp) {
        tmp = CeilAlign(baseInfoOp.col, UB_BLOCK_SIZE / baseInfoOp.dtypeBytes);
    }
    int64_t tmpNum = baseInfoOp.row * CeilAlign(baseInfoOp.col, tmp);
    int64_t tmpInner = FloorDiv(baseInfoOp.bufferEleNum, tmpNum);
    int64_t tmpOuter = CeilDiv(baseInfoOp.fusedFrontAxis, tmpInner);
    int64_t minCoreNum = CeilDiv(baseInfoOp.totalCoreNum, MIN_CORE_COE);

    splitCoreOp.highOuter = std::max(minCoreNum, tmpOuter);
    splitCoreOp.highInner = CeilDiv(baseInfoOp.fusedFrontAxis, splitCoreOp.highOuter);
    splitCoreOp.highOuter = CeilDiv(baseInfoOp.fusedFrontAxis, splitCoreOp.highInner);
    splitCoreOp.baseBlockNum = splitCoreOp.highOuter;
    int64_t tmpTail = GetMod(baseInfoOp.fusedFrontAxis, splitCoreOp.highInner);
    splitCoreOp.highTail = tmpTail == 0 ? splitCoreOp.highInner : tmpTail;

    splitCoreOp.headRows = -baseInfoOp.diagOffset;
    splitCoreOp.tailRows = baseInfoOp.row - baseInfoOp.col - splitCoreOp.headRows;
    splitCoreOp.copyCols = (splitCoreOp.headRows > 0 ? 1 : -splitCoreOp.headRows + 1) - (operatorType_ ? 1 : 0);
    splitCoreOp.headRows = splitCoreOp.headRows < 0 ? 0 : splitCoreOp.headRows;
    splitCoreOp.tailRows = splitCoreOp.tailRows < 0 ? 0 : splitCoreOp.tailRows;
    splitCoreOp.midRows = baseInfoOp.row - splitCoreOp.headRows - splitCoreOp.tailRows;
}

ge::graphStatus TriangulatorTiling::DoTiling()
{
    OP_LOGD(nodeName_, "[Triangulator] DoTiling start running.");
    ComputeTilingMode();

    if (splitCoreOp.tilingMode == OUTPUT_NORMAL_MODE) {
        ComputeNormalInfo();
    } else if (splitCoreOp.tilingMode == TINY_SHAPE_MODE) {
        ComputeTinyInfo();
    } else if (splitCoreOp.tilingMode == MEDIUM_SHAPE_MODE) {
        ComputeMediumInfo();
    } else {
        baseInfoOp.bufferSize = FloorAlign(baseInfoOp.ubSize / TWO_BUFFER, UB_BLOCK_SIZE);
        baseInfoOp.bufferEleNum = FloorDiv(baseInfoOp.bufferSize, baseInfoOp.dtypeBytes);
        splitCoreOp.baseBlockNum = baseInfoOp.fusedFrontAxis * baseInfoOp.col * baseInfoOp.row;
    }
    splitCoreOp.usedCoreNum = std::min(splitCoreOp.baseBlockNum, baseInfoOp.totalCoreNum);
    splitCoreOp.normalCoreProcessNum = splitCoreOp.baseBlockNum / splitCoreOp.usedCoreNum;
    splitCoreOp.tailCoreProcessNum = splitCoreOp.baseBlockNum % splitCoreOp.usedCoreNum;
    if (splitCoreOp.tilingMode == OUTPUT_INPUT_MODE || splitCoreOp.tilingMode == OUTPUT_ZERO_MODE) {
        std::tie(splitCoreOp.usedCoreNum, splitCoreOp.normalCoreProcessNum, splitCoreOp.tailCoreProcessNum) =
            OutputZeroAndInputTiling(splitCoreOp.baseBlockNum, baseInfoOp.totalCoreNum, baseInfoOp.dtypeBytes);
    }
    ComputeTilingKey();
    OP_LOGD(nodeName_, "[Triangulator] DoTiling run completed.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriangulatorTiling::PostTiling()
{
    OP_LOGD(nodeName_, "[Triangulator] PostTiling start running.");
    size_t* userWorkspaceSize = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, userWorkspaceSize);
    userWorkspaceSize[0] = WORKSPACE_SIZE;

    SaveToTilingData();

    context_->SetBlockDim(splitCoreOp.usedCoreNum);
    if (context_->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    switch (splitCoreOp.tilingMode) {
        case TINY_SHAPE_MODE:
            if (tinyTilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
                return ge::GRAPH_FAILED;
            }
            tinyTilingData.SaveToBuffer(
                context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
            context_->GetRawTilingData()->SetDataSize(tinyTilingData.GetDataSize());
            break;
        case MEDIUM_SHAPE_MODE:
            if (mediumTilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
                return ge::GRAPH_FAILED;
            }
            mediumTilingData.SaveToBuffer(
                context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
            context_->GetRawTilingData()->SetDataSize(mediumTilingData.GetDataSize());
            break;
        case OUTPUT_NORMAL_MODE:
            if (normalTilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
                return ge::GRAPH_FAILED;
            }
            normalTilingData.SaveToBuffer(
                context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
            context_->GetRawTilingData()->SetDataSize(normalTilingData.GetDataSize());
            break;
        default:
            if (tilingData.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
                return ge::GRAPH_FAILED;
            }
            tilingData.SaveToBuffer(
                context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
            context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
            break;
    }
    context_->SetTilingKey(splitCoreOp.tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TriangulatorTiling::RunTriangulatorTilingAscendC(
    const TriluCompileInfo* compileInfo, const TriluTilingParams* params, uint8_t dtypeBytes)
{
    OP_LOGD(nodeName_, "[Triangulator] RunTriangulatorTilingAscendC start running.");
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = GetInputInfo(params, dtypeBytes);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo(compileInfo);
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
} // namespace optiling