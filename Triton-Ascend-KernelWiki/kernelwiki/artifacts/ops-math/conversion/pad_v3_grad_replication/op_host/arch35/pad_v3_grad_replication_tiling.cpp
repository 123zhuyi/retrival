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
 * \file pad_v3_grad_replication_tiling.cpp
 * \brief tiling implementation for pad_v3_grad_replication
 */

#include "pad_v3_grad_replication_tiling.h"
#include "util/platform_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "op_api/op_util.h"

namespace optiling {

static constexpr uint64_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr size_t PADDINGS_IDX = 1;
static constexpr size_t PAIR = 2;
static constexpr uint64_t RESERVED_UB = 2 * 1024; // 保留UB空间
static constexpr uint64_t MAX_DIM_NUM = 8;
static constexpr uint64_t INT16_MAX_VAL = 32767; // int16最大值，用于16位类型数据分块限制
static constexpr uint8_t FP32_SIZE = 4;
static constexpr uint8_t FP16_SIZE = 2;
static constexpr uint8_t BF16_SIZE = 2;
static constexpr uint8_t INT8_BYTE_SIZE = 1;  // int8/uint8数据类型占用字节数
static constexpr uint8_t INT16_BYTE_SIZE = 2; // int16/uint16数据类型占用字节数
static constexpr uint8_t INT32_BYTE_SIZE = 4; // int32/uint32数据类型占用字节数
static constexpr uint8_t INT64_BYTE_SIZE = 8; // int64/uint64数据类型占用字节数
static constexpr uint8_t NUM_TWO = 2;

template <typename T>
std::string PadV3GradReplicationTiling::ToString(const T* value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ",";
    }
    r = r + "]";
    return r;
}

uint64_t PadV3GradReplicationTiling::GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize)
{
    if (alignBlockSize == 0) {
        return 0;
    }
    return (inputSize + alignBlockSize - 1) / alignBlockSize * alignBlockSize;
}

ge::graphStatus PadV3GradReplicationTiling::Init()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    blockSize_ = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    OP_CHECK_IF(blockSize_ == 0, OP_LOGE(context_->GetNodeName(), "Failed to get ub block size."),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "Init: coreNum=%u, ubSize=%lu, blockSize=%lu", coreNum_, ubSize_, blockSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradReplicationTiling::GetShapeAttrsInfo()
{
    const gert::StorageShape* xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);

    dimNum_ = xShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum_ == 0 || dimNum_ > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dimNum_).c_str(),
                                                         "The shape dim of x must be within the range [1, 8]"),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < dimNum_; i++) {
        outputShape_[i] = xShape->GetStorageShape().GetDim(i); // padding后的tensor（含padding）
    }

    const gert::StorageShape* yShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);

    for (size_t i = 0; i < dimNum_; i++) {
        inputShape_[i] = yShape->GetStorageShape().GetDim(i); // 原始tensor（不含padding）
    }

    paramsDtype_ = context_->GetInputDesc(0)->GetDataType();
    switch (paramsDtype_) {
        case ge::DT_FLOAT:
            dataSize_ = FP32_SIZE;
            break;
        case ge::DT_FLOAT16:
            dataSize_ = FP16_SIZE;
            break;
        case ge::DT_BF16:
            dataSize_ = BF16_SIZE;
            break;
        case ge::DT_INT8:
        case ge::DT_UINT8:
            dataSize_ = INT8_BYTE_SIZE;
            break;
        case ge::DT_INT16:
        case ge::DT_UINT16:
            dataSize_ = INT16_BYTE_SIZE;
            break;
        case ge::DT_INT32:
        case ge::DT_UINT32:
            dataSize_ = INT32_BYTE_SIZE;
            break;
        case ge::DT_INT64:
        case ge::DT_UINT64:
            dataSize_ = INT64_BYTE_SIZE;
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                                  Ops::Base::ToString(paramsDtype_).c_str(),
                                                  "The dtype of x must be float, float16, bfloat16 or integer types");
            return ge::GRAPH_FAILED;
    }

    OP_LOGI(context_->GetNodeName(), "GetShapeAttrsInfo: dimNum=%u, dtype=%s, dataSize=%u", dimNum_,
            Ops::Base::ToString(paramsDtype_).c_str(), dataSize_);
    OP_LOGI(context_->GetNodeName(), "inputShape=%s", ToString(inputShape_, dimNum_).c_str());
    OP_LOGI(context_->GetNodeName(), "outputShape=%s", ToString(outputShape_, dimNum_).c_str());

    return ge::GRAPH_SUCCESS;
}

template <typename T>
void PadV3GradReplicationTiling::GetPaddingsToShape(const gert::Tensor* paddingsTensor)
{
    const T* paddingsValue = paddingsTensor->GetData<T>();

    for (size_t i = 0; i < dimNum_; i++) {
        leftPad_[i] = paddingsValue[PAIR * i];
        rightPad_[i] = paddingsValue[PAIR * i + 1];
    }

    OP_LOGI(context_->GetNodeName(), "leftPad=%s", ToString(leftPad_, dimNum_).c_str());
    OP_LOGI(context_->GetNodeName(), "rightPad=%s", ToString(rightPad_, dimNum_).c_str());
}

ge::graphStatus PadV3GradReplicationTiling::GetPaddings()
{
    const gert::Tensor* paddingsTensor = context_->GetInputTensor(PADDINGS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingsTensor);

    if (!ops::IsConstTensor(paddingsTensor)) {
        OP_LOGE(context_->GetNodeName(), "paddings must be const tensor.");
        return ge::GRAPH_FAILED;
    }

    const size_t paddingsNum = static_cast<size_t>(paddingsTensor->GetShapeSize());
    OP_CHECK_IF(paddingsNum != PAIR * dimNum_,
                OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "paddings", std::to_string(paddingsNum).c_str(),
                                             std::to_string(PAIR * dimNum_).c_str()),
                return ge::GRAPH_FAILED);

    ge::DataType paddingsDtype = paddingsTensor->GetDataType();
    switch (paddingsDtype) {
        case ge::DT_INT32:
            GetPaddingsToShape<int32_t>(paddingsTensor);
            break;
        case ge::DT_INT64:
            GetPaddingsToShape<int64_t>(paddingsTensor);
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "paddings",
                                                  Ops::Base::ToString(paddingsDtype).c_str(),
                                                  "The dtype of paddings must be int32 or int64");
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void PadV3GradReplicationTiling::CalcStrideAligned()
{
    // stride计算：从最低维到最高维
    // stride[N-1] = 1（最低维）
    // stride[k] = stride[k+1] × outputShape[k+1]

    strideAligned_[dimNum_ - 1] = 1; // 最低维stride=1

    // 最低维行大小（blockSize对齐）
    uint64_t rowSize = outputShape_[dimNum_ - 1] * dataSize_;
    uint64_t rowSizeAligned = GetSizeOfBlockAlign(rowSize, blockSize_) / dataSize_;

    for (int64_t k = dimNum_ - NUM_TWO; k >= 0; k--) {
        if (k == dimNum_ - NUM_TWO) {
            strideAligned_[k] = rowSizeAligned; // 中间维stride（行stride，blockSize对齐）
        } else {
            strideAligned_[k] = strideAligned_[k + 1] * outputShape_[k + 1];
        }
    }

    OP_LOGI(context_->GetNodeName(), "strideAligned=%s", ToString(strideAligned_, dimNum_).c_str());
}

bool PadV3GradReplicationTiling::IsPaddingDim(uint32_t axis) const
{
    return (dimNum_ <= PAD_GRAD_REPLICATION_MAX_PAD_DIMS_NUM) ||
           (axis + PAD_GRAD_REPLICATION_MAX_PAD_DIMS_NUM >= dimNum_);
}

uint64_t PadV3GradReplicationTiling::CalcWorstFactor(uint32_t axis) const
{
    // corner tile 最坏 UB 放大因子：外层每个可 padding 轴的边缘切片数乘积
    // 外层轴 k 在边缘时（outerCoords[k]==0 或 inputShape[k]-1），需额外搬入
    //   max(paddingLeft[k], paddingRight[k]) + 1 个切片
    uint64_t factor = 1;
    for (uint32_t k = 0; k < axis; k++) {
        if (!IsPaddingDim(k))
            continue;
        uint64_t maxPad = std::max(static_cast<uint64_t>(leftPad_[k]), static_cast<uint64_t>(rightPad_[k]));
        factor *= inputShape_[k] == 1 ? leftPad_[k] + rightPad_[k] + 1 : (maxPad + 1);
    }
    return factor;
}

bool PadV3GradReplicationTiling::TrySplitAxis(uint32_t axis, uint64_t ubAvailable)
{
    // UB 估算必须与 kernel Init 完全一致（pad_v3_grad_replication.h:127-137）：
    //   worstFactor    = ∏_{k<axis} (max(pL_k, pR_k) + 1)            // corner tile 外层放大
    //   innerProdUb    = (axis == N-1) ? 1 : strideAligned[axis]      // dataBuf 内层，尾轴 32B 对齐
    //   innerProdInGm  = (axis == N-1) ? 1 : ∏_{k>axis} inputShape[k] // outputBuf 内层，紧凑
    //   padOverhead    = singleTile ? (pL + pR) : max(pL, pR)         // 单 tile 两端 pad，多 tile 仅一端
    //   maxSliceAxis   = splitSize + padOverhead
    //   dataBufBytes   = CeilAlign(worstFactor × maxSliceAxis × innerProdUb × dataSize, BLOCK_SIZE)
    //   outputBufBytes = CeilAlign(splitSize     × innerProdInGm × dataSize, BLOCK_SIZE)
    //   总 UB = dataBufBytes + outputBufBytes
    //

    const uint64_t pL = static_cast<uint64_t>(leftPad_[axis]);
    const uint64_t pR = static_cast<uint64_t>(rightPad_[axis]);
    const uint64_t worstFactor = CalcWorstFactor(axis);

    uint64_t innerProdUb = 1;
    uint64_t innerProdInGm = 1;
    if (axis != static_cast<uint32_t>(dimNum_) - 1) {
        innerProdUb = strideAligned_[axis];
        for (uint32_t k = axis + 1; k < dimNum_; k++) {
            innerProdInGm *= inputShape_[k];
        }
    }

    // dataBuf 元素索引限制：
    //   - cast 类型 (fp16/bf16): dataBuf 存 F32，gather 用 uint32_t 索引，无 16 位限制
    //   - 非 cast ≤16 位: GatherToOutputBufGatherPath IndexT=uint16_t，限制 INT16_MAX
    //   - > 16 位: uint32_t 索引，无限制
    bool isCastType = (paramsDtype_ == ge::DT_FLOAT16 || paramsDtype_ == ge::DT_BF16);
    uint64_t dataBufSz = isCastType ? FP32_SIZE : dataSize_; // PromoteT 大小 vs T 大小
    uint64_t maxDataBufElements = UINT64_MAX;
    if (!isCastType && dataSize_ <= static_cast<uint32_t>(NUM_TWO)) {
        maxDataBufElements = INT16_MAX_VAL;
    }

    // 每 tile 至多一端 pad（首 tile pL / 尾 tile pR），单 tile 才同时 pL + pR。
    // 先用 max(pL, pR) 估 budget，出单 tile 时再回验 pL + pR。
    const uint64_t maxSinglePad = (pL > pR) ? pL : pR;
    // dataBuf 用 PromoteT 大小，outputBuf 仍用 T 大小
    const uint64_t perUnitBytes = worstFactor * innerProdUb * dataBufSz + innerProdInGm * dataSize_;
    const uint64_t fixedBytes = worstFactor * innerProdUb * maxSinglePad * dataBufSz;
    // 两个 buf 各自 CeilAlign 到 BLOCK_SIZE，最坏多占 2 × (BLOCK_SIZE - 1)
    const uint64_t alignSlack = 2 * blockSize_;

    OP_LOGI(context_->GetNodeName(),
            "TrySplitAxis: axis=%u, worstFactor=%lu, innerProdUb=%lu, innerProdInGm=%lu, pL=%lu, pR=%lu, "
            "perUnitBytes=%lu, fixedBytes=%lu, ubAvailable=%lu",
            axis, worstFactor, innerProdUb, innerProdInGm, pL, pR, perUnitBytes, fixedBytes, ubAvailable);

    if (perUnitBytes == 0) {
        OP_LOGI(context_->GetNodeName(), "axis=%u failed: perUnitBytes=0 (invalid shape)", axis);
        return false;
    }
    // 阶梯式从 ubAvailable 减 fixedBytes / alignSlack，避免任何中间值 uint64 翻转
    if (fixedBytes >= ubAvailable) {
        OP_LOGI(context_->GetNodeName(), "axis=%u failed: fixedBytes=%lu >= ubAvailable=%lu (两端 pad 区已超 UB)", axis,
                fixedBytes, ubAvailable);
        return false;
    }
    uint64_t budget = ubAvailable - fixedBytes;
    if (alignSlack >= budget) {
        OP_LOGI(context_->GetNodeName(), "axis=%u failed: alignSlack=%lu >= remaining budget=%lu", axis, alignSlack,
                budget);
        return false;
    }
    budget -= alignSlack;

    // 解 splitSize：perUnitBytes × splitSize ≤ budget
    uint64_t unitsPerTile = budget / perUnitBytes;
    if (unitsPerTile == 0) {
        OP_LOGI(context_->GetNodeName(), "axis=%u failed: unitsPerTile=0", axis);
        return false;
    }
    splitSize_ = static_cast<uint32_t>(std::min(unitsPerTile, inputShape_[axis]));
    // 单 tile 需同时装 pL + pR：回验预算（之前按 max(pL,pR) 估，仅多 tile 正确）
    if (splitSize_ == inputShape_[axis] && (pL > 0 || pR > 0)) {
        const uint64_t bothFixedBytes = worstFactor * innerProdUb * (pL + pR) * dataBufSz;
        if (bothFixedBytes + alignSlack >= ubAvailable) {
            if (inputShape_[axis] <= 1) {
                OP_LOGI(context_->GetNodeName(), "axis=%u failed: singleTile both pads exceed UB and shape=1", axis);
                return false;
            }
            splitSize_ = static_cast<uint32_t>(inputShape_[axis] - 1);
            OP_LOGI(context_->GetNodeName(), "axis=%u: single-tile both pads exceed UB, force multi-tile splitSize=%u",
                    axis, splitSize_);
        } else {
            uint64_t budgetBoth = ubAvailable - bothFixedBytes - alignSlack;
            uint64_t unitsBoth = budgetBoth / perUnitBytes;
            if (unitsBoth >= inputShape_[axis]) {
                // fits as single tile, keep splitSize_
            } else if (unitsBoth == 0) {
                if (inputShape_[axis] > 1) {
                    // pads fit UB but no room for even 1 data unit → force multi-tile
                    splitSize_ = static_cast<uint32_t>(inputShape_[axis] - 1);
                    OP_LOGI(context_->GetNodeName(),
                            "axis=%u: single-tile pads fit but no room for data, force multi-tile splitSize=%u", axis,
                            splitSize_);
                } else {
                    OP_LOGI(context_->GetNodeName(),
                            "axis=%u failed: single-tile pads fit but no room for data with shape=1", axis);
                    return false;
                }
            } else {
                splitSize_ = static_cast<uint32_t>(unitsBoth);
                OP_LOGI(context_->GetNodeName(), "axis=%u: single-tile both pads reduce to %u (bothFixed=%lu)", axis,
                        splitSize_, bothFixedBytes);
            }
        }
    }

    // 检查 dataBuf 元素索引限制（B16 dtype Gather/Scatter uint16_t 索引）
    // dataBuf 分配与 kernel Init 一致：singleTile 用 pL+pR，multiTile 用 max(pL,pR)
    uint64_t effectivePad = (static_cast<uint64_t>(splitSize_) == inputShape_[axis]) ? (pL + pR) : maxSinglePad;
    const uint64_t indexCoef = worstFactor * innerProdUb;
    if (maxDataBufElements != UINT64_MAX) {
        uint64_t dataBufElems = indexCoef * (static_cast<uint64_t>(splitSize_) + effectivePad);
        if (dataBufElems > maxDataBufElements) {
            // 单 tile 时 effectivePad = pL+pR 可能过大导致 uint16_t 不足，
            // 尝试切到多 tile（effectivePad 降为 maxSinglePad）
            bool forcedMultiTile = false;
            if (static_cast<uint64_t>(splitSize_) == inputShape_[axis] && inputShape_[axis] > 1 &&
                (pL + pR) > maxSinglePad && indexCoef > 0) {
                uint64_t multiEffPad = maxSinglePad;
                if (maxDataBufElements >= indexCoef * multiEffPad) {
                    uint64_t multiAllowedSlice = maxDataBufElements / indexCoef;
                    if (multiAllowedSlice > multiEffPad) {
                        uint64_t multiSplit = multiAllowedSlice - multiEffPad;
                        splitSize_ = static_cast<uint32_t>(std::min(multiSplit, inputShape_[axis] - 1));
                        effectivePad = multiEffPad;
                        forcedMultiTile = true;
                        OP_LOGI(context_->GetNodeName(),
                                "axis=%u: uint16_t limit forced multi-tile, splitSize=%u, effectivePad=%lu", axis,
                                splitSize_, effectivePad);
                    }
                }
            }
            if (!forcedMultiTile) {
                if (indexCoef == 0 || maxDataBufElements < indexCoef * effectivePad) {
                    OP_LOGI(
                        context_->GetNodeName(),
                        "axis=%u failed: pad overhead exceeds uint16_t index limit (indexCoef=%lu, effectivePad=%lu)",
                        axis, indexCoef, effectivePad);
                    return false;
                }
                uint64_t allowedSlice = maxDataBufElements / indexCoef; // splitSize + effectivePad
                if (allowedSlice <= effectivePad) {
                    OP_LOGI(context_->GetNodeName(), "axis=%u failed: allowedSlice=%lu ≤ effectivePad=%lu", axis,
                            allowedSlice, effectivePad);
                    return false;
                }
                uint64_t allowedSplit = allowedSlice - effectivePad;
                splitSize_ = static_cast<uint32_t>(std::min(allowedSplit, static_cast<uint64_t>(splitSize_)));
                OP_LOGI(context_->GetNodeName(), "axis=%u: uint16_t index limit reduces splitSize to %u", axis,
                        splitSize_);
            }
        }
    }

    // 计算 splitCount：每个 tile = 外层(axis<k) 1 个位置 + 当前轴 splitSize 单位 + 内层(axis>k) 全量
    if (splitSize_ == 0) {
        OP_LOGI(context_->GetNodeName(), "axis=%u failed: splitSize=0 after adjustments", axis);
        return false;
    }
    uint64_t outerCombos = 1;
    for (uint32_t k = 0; k < axis; k++) {
        outerCombos *= inputShape_[k];
    }
    uint64_t splitCountAxis = Ops::Base::CeilDiv(inputShape_[axis], static_cast<uint64_t>(splitSize_));
    splitCount_ = static_cast<uint32_t>(outerCombos * splitCountAxis);

    const uint64_t finalDataBufElems = indexCoef * (static_cast<uint64_t>(splitSize_) + effectivePad);
    const uint64_t finalDataBufBytes = finalDataBufElems * dataSize_;
    const uint64_t finalOutputBufBytes = static_cast<uint64_t>(splitSize_) * innerProdInGm * dataSize_;
    OP_LOGI(context_->GetNodeName(),
            "axis=%u success: splitSize=%u, splitCountAxis=%lu, outerCombos=%lu, splitCount=%u, "
            "worstFactor=%lu, dataBufBytes=%lu, outputBufBytes=%lu",
            axis, splitSize_, splitCountAxis, outerCombos, splitCount_, worstFactor, finalDataBufBytes,
            finalOutputBufBytes);

    return true;
}

void PadV3GradReplicationTiling::CalcSplitStrategy()
{
    uint64_t ubAvailable = ubSize_ - RESERVED_UB;

    // 从最高维开始尝试非尾轴（axis=0 → axis=N-2）
    for (uint32_t axis = 0; axis + 1 < dimNum_; axis++) {
        if (TrySplitAxis(axis, ubAvailable)) {
            splitAxis_ = axis;
            OP_LOGI(context_->GetNodeName(), "CalcSplitStrategy: splitAxis=%u, stop trying lower axes.", splitAxis_);
            return;
        }
    }

    // 非尾轴全部失败，切尾轴（axis=N-1）。
    // 尾轴走 edge_simt 纯 SIMT kernel，直接从 GM 读写、不使用 UB buffer，
    // 因此无需 UB 预算检查。splitSize_ 取全轴长度（edge_simt 按核均分 output 空间）。
    splitAxis_ = dimNum_ - 1;
    const uint32_t lastAxis = static_cast<uint32_t>(dimNum_) - 1;
    splitSize_ = static_cast<uint32_t>(inputShape_[lastAxis]);

    uint64_t outerCombos = 1;
    for (uint32_t k = 0; k + 1 < static_cast<uint32_t>(dimNum_); k++) {
        outerCombos *= inputShape_[k];
    }
    splitCount_ = static_cast<uint32_t>(outerCombos);

    OP_LOGI(context_->GetNodeName(),
            "CalcSplitStrategy: non-tail axes failed, force tail axis=%u (edge_simt), "
            "splitSize=%u, splitCount=%u",
            splitAxis_, splitSize_, splitCount_);
}

void PadV3GradReplicationTiling::CalcUsedCore()
{
    // 切尾轴：edge_simt kernel 按 output 空间均分，尽量用满全部核
    if (splitAxis_ == dimNum_ - 1) {
        usedCoreNum_ = coreNum_;
        tilesPerCore_ = 1;
        OP_LOGI(context_->GetNodeName(), "CalcUsedCore: tail axis edge_simt, use all cores=%u", usedCoreNum_);
        return;
    }

    // 非尾轴：按 splitCount 分核
    if (splitCount_ <= coreNum_) {
        usedCoreNum_ = splitCount_;
        tilesPerCore_ = 1;
    } else {
        usedCoreNum_ = coreNum_;
        tilesPerCore_ = Ops::Base::CeilDiv(splitCount_, usedCoreNum_);
    }

    OP_LOGI(context_->GetNodeName(), "CalcUsedCore: splitCount=%u, coreNum=%u, usedCoreNum=%u, tilesPerCore=%u",
            splitCount_, coreNum_, usedCoreNum_, tilesPerCore_);
}

void PadV3GradReplicationTiling::FillTilingData(PadV3GradReplicationTilingData* tilingData)
{
    tilingData->dimNum = dimNum_;
    tilingData->splitAxis = splitAxis_;
    tilingData->splitCount = splitCount_;
    tilingData->splitSize = splitSize_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->tilesPerCore = tilesPerCore_;

    for (size_t i = 0; i < dimNum_; i++) {
        tilingData->inputShape[i] = inputShape_[i];
        tilingData->outputShape[i] = outputShape_[i];
        tilingData->strideAligned[i] = strideAligned_[i];
        tilingData->leftPad[i] = leftPad_[i];
        tilingData->rightPad[i] = rightPad_[i];
    }

    OP_LOGI(context_->GetNodeName(), "FillTilingData done.");
}

ge::graphStatus PadV3GradReplicationTiling::DoTiling()
{
    OP_LOGI(context_->GetNodeName(), "Start PadV3GradReplication tiling.");

    // Step 1: Init
    ge::graphStatus ret = Init();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "Init failed."), return ge::GRAPH_FAILED);

    // Step 2: Get shape and attrs info
    ret = GetShapeAttrsInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfo failed."),
                return ge::GRAPH_FAILED);

    // Step 3: Get paddings
    ret = GetPaddings();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetPaddings failed."),
                return ge::GRAPH_FAILED);

    // Step 4: Calc stride (32B aligned)
    CalcStrideAligned();

    // Step 5: Calc split strategy
    CalcSplitStrategy();

    // Step 6: Calc used core
    CalcUsedCore();

    // Step 7: Fill tiling data
    PadV3GradReplicationTilingData* tilingData = context_->GetTilingData<PadV3GradReplicationTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    FillTilingData(tilingData);

    // Step 8: Set tiling context
    // 用 GET_TPL_TILING_KEY 让框架按 ASCENDC_TPL_ARGS_DECL 自动算位打包（DimNum 占低 4 bit、SplitAxis
    // 占高 4 bit；UINT 字段自动转 vals 数组 index）。参数顺序必须与 tilingkey.h 中 ARGS_DECL 一致。
    const uint64_t tilingKey = GET_TPL_TILING_KEY(dimNum_, splitAxis_);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(usedCoreNum_);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORK_SPACE_SIZE;

    OP_LOGI(context_->GetNodeName(), "PadV3GradReplication tiling done. tilingKey=%lu, blockDim=%u", tilingKey,
            usedCoreNum_);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3GradReplication(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4PadV3GradReplication start.");

    auto compileInfo = context->GetCompiledInfo<PadV3GradReplicationCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(compileInfo->core_num <= 0, OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ub_size);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGI(context->GetNodeName(), "TilingPrepare4PadV3GradReplication done. coreNum=%ld, ubSize=%lu",
            compileInfo->core_num, compileInfo->ub_size);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PadV3GradReplication)
    .Tiling([](gert::TilingContext* context) -> ge::graphStatus {
        PadV3GradReplicationTiling tiling(context);
        return tiling.DoTiling();
    })
    .TilingParse<PadV3GradReplicationCompileInfo>(TilingPrepare4PadV3GradReplication)
    .TilingInputsDataDependency({PADDINGS_IDX});

} // namespace optiling
