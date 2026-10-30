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
 * \file strided_slice_tiling.cc
 * \brief
 */
#include "strided_slice_tiling_arch35.h"

namespace optiling {
static const std::string OP_NAME = "StridedSlice";
constexpr int64_t MAX_GATHER_STRIDE = 16;
constexpr int64_t MOVE_ALIGN_CACHE_LINE_FACTOR = 4;
constexpr int64_t NDDMA_CACHE_LINE_FACTOR = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UNCONST_BEGIN_VALUE = -10; // unconst场景下，用-10标识需要从kernel获取
constexpr int64_t GATHER_MAX_IDX = 65536;

static const int INDEX_X = 0;
static const int INDEX_BEGIN = 1;
static const int INDEX_END = 2;
static const int INDEX_STRIDES = 3;
static const int INDEX_Y = 0;
static const size_t IDX_MASK_BEGIN = 0;
static const size_t IDX_MASK_END = 1;
static const size_t IDX_MASK_ELLIPSIS = 2;
static const size_t IDX_MASK_NEW_AXIS = 3;
static const size_t IDX_MASK_SHRINK_AXIS = 4;

ge::graphStatus StrideSliceTiling::Init(
    int64_t coreNum, int64_t ubSize, int64_t cacheLineSize, SliceParametersRuntime2& sliceParam,
    const ge::DataType& dtype)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init StrideSliceTiling.");
    coreNum_ = coreNum;
    ubSize_ = ubSize;   
    cacheLineSize_ = cacheLineSize;
    if (CheckPlatformParam() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    sliceParam_ = sliceParam;
    dtype_ = dtype;
    xDtypeSize_ = GetSizeByDataType(dtype);
    ubElementNum_ = (ubSize_ - UB_RESERVE_SIZE) / xDtypeSize_ / DOUBLE_BUFFER; // 负Stride会在ub切分时重新赋值
    dimNum_ = sliceParam.inputShape.GetDimNum();
    if (dimNum_ > MAX_AXIS_NUM_FOR_STRIDESLICE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                tilingContext_->GetNodeName(), "input", std::to_string(dimNum_),
                "The shape dim of input must be <= 8.");
        return ge::GRAPH_FAILED;
    } 

    const auto outputShape = sliceParam.outputShape;
    const auto inputShape = sliceParam.inputShape;
    const auto strideList = sliceParam.strideList;
    int32_t shapeSize = dimNum_;
    lastOneOutputDim_ = outputShape.GetDim(shapeSize - 1);
    lastTwoOutputDim_ = shapeSize >= NUMBER_TWO ? outputShape.GetDim(shapeSize - NUMBER_TWO) : 0;
    lastThreeOutputDim_ = shapeSize >= NUMBER_THREE ? outputShape.GetDim(shapeSize - NUMBER_THREE) : 0;
    lastOneInputDim_ = inputShape.GetDim(shapeSize - 1);
    lastTwoInputDim_ = shapeSize >= NUMBER_TWO ? inputShape.GetDim(shapeSize - NUMBER_TWO) : 0;
    lastThreeInputDim_ = shapeSize >= NUMBER_THREE ? inputShape.GetDim(shapeSize - NUMBER_THREE) : 0;
    lastOneStride_ = strideList[shapeSize - 1];
    lastTwoStride_ = shapeSize >= NUMBER_TWO ? strideList[shapeSize - NUMBER_TWO] : 0;
    lastThreeStride_ = shapeSize >= NUMBER_THREE ? strideList[shapeSize - NUMBER_THREE] : 0;
    lastFourStride_ = shapeSize >= NUMBER_FOUR ? strideList[shapeSize - NUMBER_FOUR] : 0;

    totalOutputSize_ = xDtypeSize_;
    for (int32_t i = 0; i < dimNum_; i++) {
        if (outputShape.GetDim(i) <= 0 || inputShape.GetDim(i) <= 0) {
            isEmptyTensor_ = true;
            return ge::GRAPH_SUCCESS;
        }
        totalOutputSize_ = totalOutputSize_ * outputShape.GetDim(i);
        if (strideList[i] < 0) {
            isStrideNeg_ = true;
        }
    }
    CalInputOutputSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StrideSliceTiling::CheckPlatformParam()
{
    if (coreNum_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                tilingContext_->GetNodeName(), "core number", std::to_string(coreNum_),
                "The value of core number cannot <= 0.");
        return ge::GRAPH_FAILED;
    }
    if (ubSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                tilingContext_->GetNodeName(), "ubSize", std::to_string(ubSize_),
                "The value of ubSize cannot <= 0.");
        return ge::GRAPH_FAILED;
    }    
    if (cacheLineSize_ <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                tilingContext_->GetNodeName(), "cacheLineSize", std::to_string(cacheLineSize_),
                "The value of cacheLineSize cannot <= 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void StrideSliceTiling::CalInputOutputSize()
{
    int64_t outputProd = 1;
    int64_t inputProd = 1;
    int64_t shapeSize = std::min((int64_t)dimNum_, MAX_SIMT_UB_SPLIT_AXIS_NUM);
    const auto outputShape = sliceParam_.outputShape;
    const auto inputShape = sliceParam_.inputShape;
    for (int64_t i = shapeSize - 1; i >= 0; i--) {
        outputProd *= outputShape.GetDim(i);
        outputShapeProd_[i] = outputProd;
        inputProd *= inputShape.GetDim(i);
        inputShapeProd_[i] = inputProd;
    }
    if (outputShapeProd_[0] > MAX_UINT32_NUM || inputShapeProd_[0] > MAX_UINT32_NUM) {
        isShapeExceedUint32_ = true;
    }
    if (outputShapeProd_[0] <= SIMT_MIN_OUTPUT_SIZE) {
        isShapeBelowThreshold_ = true;
    }
}

void StrideSliceTiling::SetAllInUbSplitInfo()
{
    const auto& outputShape = sliceParam_.outputShape;
    isAllInUb_ = true;
    blkIndex_ = 0;
    ubIndex_ = 0;
    blkFactor_ = outputShape.GetDim(0);
    ubFactor_ = blkFactor_;
    realCoreNum_ = 1;
    ubTailFactor_ = 0;
    ubTailTailFactor_ = 0;
    blkTailFactor_ = 0;
}

void StrideSliceTiling::SetTwoDimTilingInfo()
{
    // 计算分核 整尾块切分  moveAlignv2参数
    if (lastOneInputDim_ * xDtypeSize_ < VL_SIZE && sliceParam_.isBeginConst == 1 && 
             sliceParam_.beginList.GetDim(1) == 0) {
        outBlockLen_ = static_cast<uint32_t>(lastOneInputDim_ * xDtypeSize_); // 全量搬入
    } else {
        outBlockLen_ = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
    }

    uint32_t perCoreRowNum = ubSize_ / NUMBER_FOUR / Ops::Base::CeilAlign(outBlockLen_, (uint32_t)BLOCK_SIZE);
    uint32_t coreNumByInput = Ops::Base::CeilDiv((uint32_t)lastTwoInputDim_, perCoreRowNum);
    uint32_t coreNumByOutput = Ops::Base::CeilDiv((uint32_t)totalOutputSize_, (uint32_t)TWO_DIM_MIN_OUTPUT_SIZE);
    uint32_t calCoreNum = coreNumByInput > coreNumByOutput ? coreNumByInput : coreNumByOutput;
    realCoreNum_ = static_cast<int64_t>(calCoreNum) >= coreNum_ ? coreNum_ : static_cast<int64_t>(calCoreNum);

    blkTailFactor_ = lastTwoInputDim_ / realCoreNum_;
    mainCoreNum_ = static_cast<uint32_t>(lastTwoInputDim_ % realCoreNum_);
    if (mainCoreNum_ != 0U) {
        blkFactor_ = blkTailFactor_ + 1;
    } else {
        blkFactor_ = blkTailFactor_;
        mainCoreNum_ = static_cast<uint32_t>(realCoreNum_);
    }
}

void StrideSliceTiling::CalcBlockSplitInfo()
{
    const auto& outputShape = sliceParam_.outputShape;
    int64_t minCoreProcessSize = 0;
    if (dimNum_ >= NUMBER_SIX && maxSplitDim_ == MAX_NDDMA_UB_SPLIT_AXIS_NUM) {
        int64_t lastFourDimProd = 1L;
        for (int32_t i = 0; i < NUMBER_FOUR; i++) {
            lastFourDimProd *= outputShape.GetDim(dimNum_ - 1 - i);
        }
        // 每个核最少处理4个轴的数据
        minCoreProcessSize = std::min(lastFourDimProd * xDtypeSize_, LAST_DIM_MIN_DATA_SIZE);
    } else {
        // 每个核最少处理1KB的数据，小shape场景下减少开核个数
        minCoreProcessSize = LAST_DIM_MIN_DATA_SIZE;
    }
    int64_t rightDiv = std::min(coreNum_, Ops::Base::CeilDiv(totalOutputSize_, minCoreProcessSize));
    if (dimNum_ == NUMBER_TWO && ((lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1) * xDtypeSize_ <=
                                     LAST_DIM_MIN_DATA_SIZE * NUMBER_FOUR) {
        rightDiv = std::min(rightDiv, outputShape.GetDim(0));
    }
    for (int32_t i = 0; i < dimNum_; i++) {
        int64_t curDim = outputShape.GetDim(i);
        if (rightDiv >= 1 && rightDiv / curDim <= 1) {
            blkIndex_ = i;
            blkFactor_ = Ops::Base::CeilDiv(curDim, rightDiv);
            break;
        }
        rightDiv = rightDiv / curDim;
    }

    // 最后一维留够1KB
    if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM && blkIndex_ == static_cast<int64_t>(dimNum_ - 1) &&
        blkFactor_ * std::abs(lastOneStride_) < LAST_DIM_MIN_DATA_SIZE / xDtypeSize_) {
        blkFactor_ =
            std::min(LAST_DIM_MIN_DATA_SIZE / xDtypeSize_ / std::abs(lastOneStride_), outputShape.GetDim(blkIndex_));
    }

    // 说明整个shape都很小，blk不切最后一根轴
    if (blkIndex_ == static_cast<int64_t>(-1)) {
        blkIndex_ = static_cast<int64_t>(dimNum_ - 1);
        blkFactor_ = outputShape.GetDim(static_cast<size_t>(dimNum_ - 1));
    }

    blkTailFactor_ = outputShape.GetDim(blkIndex_) % blkFactor_;
    realCoreNum_ = Ops::Base::CeilDiv(outputShape.GetDim(blkIndex_), blkFactor_);
    for (int32_t i = 0; i < blkIndex_; i++) {
        realCoreNum_ *= outputShape.GetDim(i);
    }
}

void StrideSliceTiling::CalcUbSplitInfo()
{
    if (isSliceGather_) { // lastInputDim < 128B  lastOutputDim < 64B
        uint64_t calOutputSize = Ops::Base::CeilAlign(lastTwoOutputDim_ * lastOneOutputDim_ * xDtypeSize_, BLOCK_SIZE);
        uint64_t calInputSize = Ops::Base::CeilAlign(lastTwoOutputDim_ * lastOneInputDim_ * xDtypeSize_, BLOCK_SIZE);
        uint64_t halfUbSize = static_cast<uint64_t>((ubSize_ - INIT_INDEX_SIZE - cacheLineSize_) / DOUBLE_BUFFER);
        if (calOutputSize + calInputSize < halfUbSize) {
            // ub切分轴至少在-3轴
            ubSizeInput_ = static_cast<int64_t>(
                halfUbSize * calInputSize / (calOutputSize + calInputSize) / BLOCK_SIZE * BLOCK_SIZE);
            ubSizeOutput_ = static_cast<int64_t>(
                halfUbSize * calOutputSize / (calOutputSize + calInputSize) / BLOCK_SIZE * BLOCK_SIZE + cacheLineSize_);
        } else {
            // ub切分轴在-2轴
            ubSizeInput_ = static_cast<int64_t>(
                halfUbSize * lastOneInputDim_ / (lastOneInputDim_ + lastOneOutputDim_) / BLOCK_SIZE * BLOCK_SIZE);
            ubSizeOutput_ = static_cast<int64_t>(
                halfUbSize * lastOneOutputDim_ / (lastOneInputDim_ + lastOneOutputDim_) / BLOCK_SIZE * BLOCK_SIZE +
                cacheLineSize_);
        }
        ubElementNum_ = ubSizeInput_ / xDtypeSize_;
    }

    int64_t rightProduct = 1;
    const auto outputShape = sliceParam_.outputShape;
    const auto inputShape = sliceParam_.inputShape;
    int32_t shapeSize = outputShape.GetDimNum();
    int32_t maxLeftIdx = std::max(shapeSize - maxSplitDim_, static_cast<int32_t>(blkIndex_));
    for (int32_t i = shapeSize - 1; i >= 0 && i >= maxLeftIdx; i--) {
        int64_t curDim = outputShape.GetDim(i);
        if (rightProduct * curDim >= ubElementNum_) {
            ubIndex_ = i;
            ubFactor_ = ubElementNum_ / rightProduct;
            ubTailFactor_ = curDim % ubFactor_;
            break;
        }
        if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM && i == shapeSize - NUMBER_TWO) {
            rightProduct = Ops::Base::CeilAlign(curDim * rightProduct * xDtypeSize_, BLOCK_SIZE) / xDtypeSize_;
        } else if (isSliceGather_ && i == shapeSize - 1) {
            rightProduct = inputShape.GetDim(i);
        } else if (
            isSliceGather_ &&
            i == shapeSize - NUMBER_TWO) { // 最后2维连续搬入，倒数2根轴可以合并，否则就是最后一维数据需要block对齐
            rightProduct = Ops::Base::CeilAlign(curDim * rightProduct * xDtypeSize_, BLOCK_SIZE) / xDtypeSize_;
        } else {
            rightProduct = curDim * rightProduct;
        }
    }

    if (ubIndex_ == -1L) {
        ubIndex_ = maxLeftIdx;
        ubFactor_ = outputShape.GetDim(ubIndex_);
        ubTailFactor_ = 0;
    }

    // ub blk split same axis
    if (ubIndex_ == blkIndex_) {
        ubFactor_ = std::min(ubFactor_, blkFactor_);
        ubTailFactor_ = blkFactor_ % ubFactor_;
        ubTailTailFactor_ = blkTailFactor_ % ubFactor_;
    }
}

void StrideSliceTiling::CalcUbSplitInfoNeg()
{
    int32_t ubInFactor = isNddma_ ? 1 : std::abs(lastOneStride_);
    ubInFactor = std::max(ubInFactor, 1);
    // 开启db： 2*UbIn + 2*UbOut = 2*UbOut*ubInFactor + 2*UbOut = ubSize_
    int32_t ubOutBase = static_cast<int32_t>(
        (ubSize_ - UB_RESERVE_SIZE) / DOUBLE_BUFFER / (ubInFactor + 1) / cacheLineSize_ * cacheLineSize_);
    ubOutBase = std::max(ubOutBase, static_cast<int32_t>(cacheLineSize_));
    ubSizeInput_ = static_cast<int64_t>(ubOutBase * ubInFactor);
    // B8/B16使用uint16作为gather的索引，这里限制一下，保证不超过uint16的最大值。限制输入大小不超过64KB
    if (ubSizeInput_ > GATHER_MAX_IDX) {
        ubOutBase = GATHER_MAX_IDX / ubInFactor / cacheLineSize_ * cacheLineSize_;
        ubOutBase = std::max(ubOutBase, static_cast<int32_t>(cacheLineSize_));
        ubSizeInput_ = static_cast<int64_t>(ubOutBase * ubInFactor);
    }
    ubElementNum_ = ubOutBase / xDtypeSize_;
    ubSizeOutput_ = ubOutBase;

    // moveAlign和nddma、gather/ub2ub，都要求最后一维ub内按block对齐
    int64_t rightProduct = 1;
    const auto outputShape = sliceParam_.outputShape;
    int32_t shapeSize = outputShape.GetDimNum();
    int32_t maxLeftIdx = std::max(shapeSize - maxSplitDim_, static_cast<int32_t>(blkIndex_));
    for (int32_t i = shapeSize - 1; i >= 0 && i >= maxLeftIdx; i--) {
        int64_t curDim = outputShape.GetDim(i);
        if (i == shapeSize - 1) {
            // 这里按ubOut作对齐，则ubIn必然也是对齐的
            curDim = Ops::Base::CeilAlign(curDim, BLOCK_SIZE / xDtypeSize_);
        }
        if (rightProduct * curDim >= ubElementNum_) {
            ubIndex_ = i;
            ubFactor_ = ubElementNum_ / rightProduct;
            ubTailFactor_ = outputShape.GetDim(i) % ubFactor_;
            break;
        }
        rightProduct = curDim * rightProduct;
    }

    if (ubIndex_ == -1L) {
        ubIndex_ = maxLeftIdx;
        ubFactor_ = outputShape.GetDim(ubIndex_);
        ubTailFactor_ = 0;
    }

    // ub blk split same axis
    if (ubIndex_ == blkIndex_) {
        ubFactor_ = std::min(ubFactor_, blkFactor_);
        ubTailFactor_ = blkFactor_ % ubFactor_;
        ubTailTailFactor_ = blkTailFactor_ % ubFactor_;
    }
}

ge::graphStatus StrideSliceTiling::RunStrideSliceTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start running Tiling4StrideSlice.");
    // WARNING: 优先判断 isEmptyTensor_，else分支内不需要除0保护
    if (isEmptyTensor_) {
        OP_LOGI(tilingContext_->GetNodeName(), "Input or Output tensor is empty, set SIMT mode.");
        SetSIMTTilingMode();
    } else if (isStrideNeg_) {
        CalMaxSplitDimNeg();
        if (maxSplitDim_ != MAX_SIMT_UB_SPLIT_AXIS_NUM) {
            CalcBlockSplitInfo();
            CalcUbSplitInfoNeg();
        }
        SetTilingModeNeg();
    } else {
        CalMaxSplitDim();
        if (maxSplitDim_ == MAX_TWO_DIM_UB_SPLIT_AXIS_NUM) {
            SetTwoDimTilingInfo();
        } else if (totalOutputSize_ <= LAST_DIM_MIN_DATA_SIZE && dimNum_ <= maxSplitDim_) {
            SetAllInUbSplitInfo();
        } else {
            CalcBlockSplitInfo();
            if (!useGather_) {
                CalcUbSplitInfo();
            } else {
                CalcUbSplitInfoNeg();
            }
        }
        SetTilingMode();
    }

    // fill data
    FillTilingData();
    // print data
    PrintTilingData();
    // set block dim and tilingKey
    SetBlockDimAndTilingKey();

    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;

    OP_LOGD(tilingContext_->GetNodeName(), "Tiling4StrideSlice success.");
    return ge::GRAPH_SUCCESS;
}

int64_t StrideSliceTiling::GetDimStartPosition(int32_t idx)
{
    if (!sliceParam_.isBeginConst) {
        if (sliceParam_.beginList[idx] >= 0) {
            return (sliceParam_.strideList[idx] > 0) ?
                       sliceParam_.beginList[idx] :
                       (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
        }

        return 0;
    }

    return (sliceParam_.strideList[idx] > 0) ?
               sliceParam_.beginList[idx] :
               (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
}

int64_t StrideSliceTiling::GetDimEndPosition(int32_t idx)
{
    if (!sliceParam_.isBeginConst) {
        if (sliceParam_.beginList[idx] >= 0) {
            return (sliceParam_.strideList[idx] < 0) ?
                       sliceParam_.beginList[idx] :
                       (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
        }

        return 1;
    }

    return (sliceParam_.strideList[idx] < 0) ?
               sliceParam_.beginList[idx] :
               (sliceParam_.beginList[idx] + (sliceParam_.outputShape[idx] - 1) * sliceParam_.strideList[idx]);
}

int64_t StrideSliceTiling::GetValidNumInCacheLine()
{
    int64_t cachelineNum = cacheLineSize_ / xDtypeSize_; // cacheline内的数据个数
    int64_t validNum = 1;                                // 起始位置肯定有效，计数为1

    if (lastOneInputDim_ >= cachelineNum) {
        int64_t checkNum = std::min((lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1, cachelineNum);
        validNum = Ops::Base::CeilDiv(checkNum, std::abs(lastOneStride_));
        OP_LOGI(tilingContext_->GetNodeName(), "cachelineNum:%ld validNum:%ld", cachelineNum, validNum);
        return validNum;
    }

    int64_t startOffsetNum = 0;
    int64_t inputShapeStride = 1;
    for (int32_t i = dimNum_ - 1; i >= 0; i--) {
        int64_t startV = GetDimStartPosition(i);
        startOffsetNum += startV * inputShapeStride;
        inputShapeStride *= sliceParam_.inputShape[i];
    }
    OP_LOGD(
        tilingContext_->GetNodeName(), "startOffsetNum:%ld inputShapeProd_[0]:%ld", startOffsetNum, inputShapeProd_[0]);

    int64_t totalInputCnt = std::min(inputShapeProd_[0], startOffsetNum + cachelineNum);
    for (int64_t idx = startOffsetNum + 1; idx < totalInputCnt; idx++) {
        bool curValid = true;
        int64_t shapeIdx = idx;
        for (int32_t i = dimNum_ - 1; i >= 0; i--) {
            int64_t curStart = GetDimStartPosition(i);
            int64_t curEnd = GetDimEndPosition(i);
            int64_t curDimIdx = shapeIdx % sliceParam_.inputShape[i];
            shapeIdx = shapeIdx / sliceParam_.inputShape[i];
            if (curDimIdx < curStart || curDimIdx > curEnd ||
                (curDimIdx - curStart) % std::abs(sliceParam_.strideList[i]) != 0) {
                curValid = false;
                break;
            }
        }
        if (curValid) {
            validNum++;
        }
    }

    OP_LOGI(tilingContext_->GetNodeName(), "cachelineNum:%ld validNum:%ld", cachelineNum, validNum);
    return validNum;
}

void StrideSliceTiling::CalMaxSplitDimLastStrideNegative()
{
    int64_t lastDimInLen = (lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1;
    // MoveAlignV2 + gather
    if (std::abs(lastOneStride_) <= MAX_GATHER_STRIDE && lastDimInLen * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
        useGather_ = true;
        return;
    }

    // nddma + gather
    int64_t validNum = GetValidNumInCacheLine();
    if (validNum * xDtypeSize_ >= cacheLineSize_ / NDDMA_CACHE_LINE_FACTOR &&
        lastOneOutputDim_ * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG;
        useGather_ = true;
        isNddma_ = true;
        return;
    }
    // default simt
    return;
}

void StrideSliceTiling::CalMaxSplitDimLastStrideOne()
{
    int64_t lastDimInLen = (lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1;
    if (lastDimInLen * xDtypeSize_ >= cacheLineSize_ / MOVE_ALIGN_CACHE_LINE_FACTOR) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
        useGather_ = false;
        return;
    }
    // default simt
    return;
}

void StrideSliceTiling::CalMaxSplitDimLastStridePositive()
{
    int64_t lastDimInLen = (lastOneOutputDim_ - 1) * std::abs(lastOneStride_) + 1;
    if (lastOneStride_ <= MAX_GATHER_STRIDE && lastDimInLen * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
        useGather_ = true;
        return;
    }

    int64_t validNum = GetValidNumInCacheLine();
    if (validNum * xDtypeSize_ >= cacheLineSize_ / NDDMA_CACHE_LINE_FACTOR &&
        lastOneOutputDim_ * xDtypeSize_ >= cacheLineSize_) {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG;
        useGather_ = false;
        isNddma_ = true;
        return;
    }
    // default simt
    return;
}

void StrideSliceTiling::CalMaxSplitDimNeg()
{
    if (totalOutputSize_ <= LAST_DIM_MIN_DATA_SIZE) {
        return;
    }

    if (lastOneStride_ < 0) {
        return CalMaxSplitDimLastStrideNegative();
    }

    if (lastOneStride_ == 1) {
        return CalMaxSplitDimLastStrideOne();
    }

    return CalMaxSplitDimLastStridePositive();
}

void StrideSliceTiling::CalMaxSplitDim()
{
    int64_t validNum = GetValidNumInCacheLine();
    if (lastOneStride_ == 1) {
        if (lastOneOutputDim_ * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
            maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
            return;
        }

        if (validNum * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
            maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
            return;
        }
    } else {
        // lastOneStride_ > 1
        int64_t lastDimInLen = (lastOneOutputDim_ - 1) * lastOneStride_ + 1;
        if (lastOneStride_ <= MAX_GATHER_STRIDE && lastDimInLen * xDtypeSize_ >= cacheLineSize_ &&
            totalOutputSize_ > LAST_DIM_MIN_DATA_SIZE) {
            maxSplitDim_ = MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM;
            useGather_ = true;
            return;
        }

        if (validNum * xDtypeSize_ >= RESERVE_LAST_DIM_SIZE) {
            maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
            return;
        }
    }

    if (isShapeExceedUint32_ || isShapeBelowThreshold_) {
        maxSplitDim_ = MAX_NDDMA_UB_SPLIT_AXIS_NUM;
        return;
    }
}

void StrideSliceTiling::SetTilingMode()
{
    if (maxSplitDim_ == MAX_MOV_ALIGN_V2_UB_SPLIT_AXIS_NUM) {
        SetMovAlignV2TilingMode();
    } else if (maxSplitDim_ == MAX_NDDMA_UB_SPLIT_AXIS_NUM) {
        SetNddmaTilingMode();
    } else {
        SetSIMTTilingMode();
    }
}

void StrideSliceTiling::SetTilingModeNeg()
{
    if (maxSplitDim_ == MAX_SIMT_UB_SPLIT_AXIS_NUM) {
        SetSIMTTilingMode();
    } else {
        if (!isNddma_) {
            SetMovAlignV2TilingModeNeg();
        } else {
            SetNddmaTilingModeNeg();
        }
    }
}

void StrideSliceTiling::SetMovAlignV2TilingModeNeg()
{
    tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB;

    // ubfactor切-1轴
    if (ubIndex_ == dimNum_ - 1L) {
        MovAlignV2UbSplitLastOneDim();
        return;
    }

    // ubfactor切-2轴
    if (ubIndex_ == dimNum_ - NUMBER_TWO) {
        return MovAlignV2UbSplitLastTwoDim();
    }

    // ubfactor切-3轴
    if (ubIndex_ == dimNum_ - NUMBER_THREE) {
        return MovAlignV2UbSplitLastThreeDim();
    }

    // ubfactor切-4轴
    if (ubIndex_ == dimNum_ - NUMBER_FOUR) {
        return MovAlignV2UbSplitLastFourDim();
    }
}

void StrideSliceTiling::SetNddmaTilingModeNeg()
{
    tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_NDDMA_GATHER : STRIDED_SLICE_KEY_NDDMA_UB2UB;

    // ubfactor切倒数第i轴
    int ubSplitDimNumber = static_cast<int>(dimNum_ - ubIndex_);
    for (int32_t j = 1; j <= ubSplitDimNumber; j++) {
        if (ubSplitDimNumber == j) {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = ubFactor_;
        } else {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = sliceParam_.outputShape.GetDim(dimNum_ - j);
        }
        nddmaTotalNum_ = nddmaTotalNum_ * nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j];
        nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = std::abs(sliceParam_.strideList[dimNum_ - j]);
        nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] = 1;
        for (int32_t k = 1; k < j; k++) {
            nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] *= sliceParam_.inputShape.GetDim(dimNum_ - k);
            // 尾轴对齐到block
            if (k == 1) {
                nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] *=
                    Ops::Base::CeilAlign(sliceParam_.outputShape.GetDim(dimNum_ - k), BLOCK_SIZE / xDtypeSize_);
            } else {
                nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM_NEG - j] *= sliceParam_.outputShape.GetDim(dimNum_ - k);
            }
        }
    }

    return;
}

/*
input
(S, A, B, C, D)
outputShape:
(s, a, b, c, d)

模板2, NDDMA场景
loop0Size
loop0srcStride
loop0DstStride
单位都是element number
loop0size = loopSizeList[NUMBER_FOUR]
loop0srcStirde = loopsrcStirdeList[NUMBER_FOUR]
loop0dstStride = loopdstStrideList[NUMBER_FOUR]

场景1: ub切d
(a, b, c, d.o, d.i)
loop0Size = d.i
loop0SrcStride = stride_d
loop0dstStride = 1

loop0TailSize = d % d.i

场景2:
ub切c
(a, b, c.o, c.i, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c.i
loop1SrcStride = D * stride_c
loop1dstStride = d

loop1TailSize = c % c.i

场景3:
ub切b
(a, b.o, b.i, c, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c
loop1SrcStride = D * stride_c
loop1dstStride = d

loop2Size = b.i
loop2SrcStride = D * C * stride_b
loop2dstStride = d * c

loop2TailSize = b % b.i

场景4:
ub切a
(a.o, a.i, b, c, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c
loop1SrcStride = D * stride_c
loop1dstStride = d

loop2Size = b
loop2SrcStride = D * C * stride_b
loop2dstStride = d * c

loop3Size = a.i
loop3SrcStride = D * C * B * stride_a
loop3dstStride = d * c * b

loop3TailSize = a % a.i

场景5:
ub切s
(s.o, s.i, a, b, c, d)
loop0Size = d
loop0SrcStride = stride_d
loop0dstStride = 1

loop1Size = c
loop1SrcStride = D * stride_c
loop1dstStride = d

loop2Size = b
loop2SrcStride = D * C * stride_b
loop2dstStride = d * c

loop3Size = a
loop3SrcStride = D * C * B * stride_a
loop3dstStride = d * c * b

loop4Size = s.i
loop4SrcStride = D * C * B * A * stride_s
loop4dstStride = d * c * b * a

loop4TailSize = s % s.i
*/
void StrideSliceTiling::SetNddmaTilingMode()
{
    int32_t shapeSize = dimNum_;
    tilingKey_ = STRIDED_SLICE_KEY_NDDMA;
    // ubfactor切倒数第i轴
    int ubSplitDimNumber = static_cast<int>(shapeSize - ubIndex_);
    if (ubSplitDimNumber == 1) {
        tilingKey_ = STRIDED_SLICE_KEY_NDDMA_LAST_DIM;
    }
    for (int32_t j = 1; j <= ubSplitDimNumber; j++) {
        if (ubSplitDimNumber == j) {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = ubFactor_;
        } else {
            nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = sliceParam_.outputShape.GetDim(dimNum_ - j);
        }
        nddmaTotalNum_ = nddmaTotalNum_ * nddmaLoopSize_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j];
        nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = sliceParam_.strideList[dimNum_ - j];
        nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] = 1;
        for (int32_t k = 1; k < j; k++) {
            nddmaLoopSrcStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] *= sliceParam_.inputShape.GetDim(dimNum_ - k);
            nddmaLoopDstStride_[MAX_NDDMA_UB_SPLIT_AXIS_NUM - j] *= sliceParam_.outputShape.GetDim(dimNum_ - k);
        }
    }
}

void StrideSliceTiling::SetSIMTTilingMode()
{
    tilingKey_ = isShapeExceedUint32_ ? STRIDED_SLICE_KEY_SIMT_BIG_SHAPE : STRIDED_SLICE_KEY_SIMT;
    realCoreNum_ = coreNum_;
}

void StrideSliceTiling::MovAlignV2UbSplitLastOneDim()
{
    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(1);
        // 首尾都是有效的、能取到的数
        mainMoveAlignV2Info_.blockLen =
            static_cast<uint32_t>((ubFactor_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_);
        mainMoveAlignV2Info_.srcStride = 0U;
        mainMoveAlignV2Info_.dstStride = 0U;
    }

    if (!isStrideNeg_) {
        tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM;
    }
}

void StrideSliceTiling::MovAlignV2UbSplitLastTwoDim()
{
    int64_t srcStride = std::abs(lastTwoStride_ * lastOneInputDim_ * xDtypeSize_);
    if (srcStride > MAX_UINT32_NUM || ubFactor_ > MAX_UINT16_NUM) {
        ResetMovAlignV2Para(lastOneOutputDim_);
        MovAlignV2UbSplitLastOneDim();
        return;
    }

    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(ubFactor_);
        // 首尾都是有效的、能取到的数
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(
            (lastOneOutputDim_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_);
        mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
        mainMoveAlignV2Info_.dstStride = 0U;
    }

    if (!isStrideNeg_) {
        if (!useGather_) {
            // 整块参数
            mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(ubFactor_);
            mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
            mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
            mainMoveAlignV2Info_.dstStride = mainMoveAlignV2Info_.blockLen;

            // 二维shape，且blk和ub都切0轴时，使用特殊模板
            tilingKey_ = (dimNum_ == NUMBER_TWO) ? STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM : STRIDED_SLICE_KEY_MOVE_ALIGN;
        } else {
            tilingKey_ = STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER;
        }
    }
}

void StrideSliceTiling::MovAlignV2UbSplitLastThreeDim()
{
    int64_t srcStride = std::abs(lastTwoStride_ * lastOneInputDim_ * xDtypeSize_);
    int64_t loop1SrcStride = std::abs(lastThreeStride_ * lastOneInputDim_ * lastTwoInputDim_ * xDtypeSize_);
    // compact 模式，最后两根轴一起对齐到block
    int64_t loop1DstStride = Ops::Base::CeilAlign(lastOneOutputDim_ * lastTwoOutputDim_ * xDtypeSize_, BLOCK_SIZE);
    int64_t burstLenNeg = 0;
    if (isStrideNeg_ || useGather_) {
        // 首尾都是有效的、能取到的数
        burstLenNeg = (lastOneOutputDim_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_;
        // 非compact 模式，最后一根轴对齐到block
        loop1DstStride = lastTwoOutputDim_ * Ops::Base::CeilAlign(burstLenNeg, BLOCK_SIZE);
    }
    if (loop1SrcStride > MAX_UINT32_NUM || srcStride > MAX_UINT32_NUM || lastTwoOutputDim_ > MAX_UINT16_NUM ||
        loop1DstStride > MAX_UINT16_NUM || ubFactor_ > MAX_UINT16_NUM) {
        ResetMovAlignV2Para(lastTwoOutputDim_);
        MovAlignV2UbSplitLastTwoDim();
        return;
    }

    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(burstLenNeg);
        mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
        mainMoveAlignV2Info_.dstStride = 0;
        mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(ubFactor_);
        mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
        mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
    }

    if (!isStrideNeg_) {
        tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN;
        if (!useGather_) {
            // 整块参数
            mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
            mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
            mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
            mainMoveAlignV2Info_.dstStride = mainMoveAlignV2Info_.blockLen;
            mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(ubFactor_);
            mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
            mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
        }
    }
}

void StrideSliceTiling::MovAlignV2UbSplitLastFourDim()
{
    int64_t srcStride = std::abs(lastTwoStride_ * lastOneInputDim_ * xDtypeSize_);
    int64_t loop1SrcStride = std::abs(lastThreeStride_ * lastOneInputDim_ * lastTwoInputDim_ * xDtypeSize_);
    int64_t loop2SrcStride =
        std::abs(lastFourStride_ * lastOneInputDim_ * lastTwoInputDim_ * lastThreeInputDim_ * xDtypeSize_);
    int64_t loop1DstStride = Ops::Base::CeilAlign(lastOneOutputDim_ * lastTwoOutputDim_ * xDtypeSize_, BLOCK_SIZE);
    int64_t burstLenNeg = 0;
    if (isStrideNeg_ || useGather_) {
        // 首尾都是有效的、能取到的数
        burstLenNeg = (lastOneOutputDim_ * std::abs(lastOneStride_) - std::abs(lastOneStride_) + 1) * xDtypeSize_;
        // 非compact 模式，最后一根轴对齐到block
        loop1DstStride = lastTwoOutputDim_ * Ops::Base::CeilAlign(burstLenNeg, BLOCK_SIZE);
    }
    int64_t loop2DstStride = lastThreeOutputDim_ * loop1DstStride;
    if (loop2SrcStride > MAX_UINT32_NUM || loop1SrcStride > MAX_UINT32_NUM || srcStride > MAX_UINT32_NUM ||
        lastTwoOutputDim_ > MAX_UINT16_NUM || lastThreeOutputDim_ > MAX_UINT16_NUM || ubFactor_ > MAX_UINT16_NUM ||
        loop1DstStride > MAX_UINT16_NUM || loop2DstStride > MAX_UINT16_NUM) {
        ResetMovAlignV2Para(lastThreeOutputDim_);
        MovAlignV2UbSplitLastThreeDim();
        return;
    }

    if (isStrideNeg_ || useGather_) {
        mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
        mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(burstLenNeg);
        mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
        mainMoveAlignV2Info_.dstStride = 0U;
        mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(lastThreeOutputDim_);
        mainMoveAlignV2Info_.loop2Size = static_cast<uint16_t>(ubFactor_);
        mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
        mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
        mainMoveAlignV2Info_.loop2SrcStride = static_cast<uint32_t>(loop2SrcStride);
        mainMoveAlignV2Info_.loop2DstStride = static_cast<uint16_t>(loop2DstStride);
    }

    if (!isStrideNeg_) {
        tilingKey_ = useGather_ ? STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER : STRIDED_SLICE_KEY_MOVE_ALIGN;
        if (!useGather_) {
            // 整块参数
            mainMoveAlignV2Info_.blockCount = static_cast<uint16_t>(lastTwoOutputDim_);
            mainMoveAlignV2Info_.blockLen = static_cast<uint32_t>(lastOneOutputDim_ * xDtypeSize_);
            mainMoveAlignV2Info_.srcStride = static_cast<uint32_t>(srcStride);
            mainMoveAlignV2Info_.dstStride = mainMoveAlignV2Info_.blockLen;
            mainMoveAlignV2Info_.loop1Size = static_cast<uint16_t>(lastThreeOutputDim_);
            mainMoveAlignV2Info_.loop2Size = static_cast<uint16_t>(ubFactor_);
            mainMoveAlignV2Info_.loop1SrcStride = static_cast<uint32_t>(loop1SrcStride);
            mainMoveAlignV2Info_.loop1DstStride = static_cast<uint16_t>(loop1DstStride);
            mainMoveAlignV2Info_.loop2SrcStride = static_cast<uint32_t>(loop2SrcStride);
            mainMoveAlignV2Info_.loop2DstStride = static_cast<uint16_t>(loop2DstStride);
        }
    }
}

void StrideSliceTiling::ResetMovAlignV2Para(int64_t newUbfactor)
{
    ubFactor_ = newUbfactor;
    ubTailFactor_ = 0;
    ubTailTailFactor_ = 0;
    ubIndex_ = ubIndex_ + 1;
    isAllInUb_ = false;
}

/*
input
(A, B, C, D)
outputShape:
(a, b, c, d)

模板1, last_dim>=128B
场景1: ub切d
(a, b, c, d.o, d.i)
整块:
d.i = ubFactor_
step = stride_d * d.i
for i = 0; i * step + d.begin < d.end; i++: // i < d.o
mov2_align_v2(dst, src, nburst = 1, burstlen = d.i * dype, burst_src_stride = d.i * dype,
burst_dst_stride = d.i * dype)

if d % di != 0
尾块：
mov2_align_v2(nburst = 1, burstlen = (d - di * do) * dype, burst_src_stride = 0, burst_dst_stride = 0)

block切a
(a.o, a.i, b, c, d.o, d.i)
for i = 0; i * c.stride + c.begin < c.end; i++:

for i = 0; i * b.stride +b.begin < b.end; i++:

整核:
for i = 0; i < a.i; i++:

尾核:
for i = 0; i < a - a.i * a.o; i++:

场景2:
ub切c
(a, b, c.o, c.i, d)
burstlen = d * dype
nburst = c.i
burst_src_stride = stridec * d * dype

整块:
step = strided * d.i
for i = 0; i < c.o; i++:
mov2_align_v2(dst, src, nburst = c.i, burstlen = d * dype, burst_src_stride = stridec * d * dype,
burst_dst_stride = d * dype)

if c % ci != 0
尾块：
mov2_align_v2(nburst = c - c.o * c.i, burstlen = d * dype, burst_src_stride = stridec * d * dype,
burst_dst_stride = d * dype)

block切a
(a.o, a.i, b, c.o, c.i, d)

for i = 0; i * strideb +b.begin < b.end; i++:

整核:
for i = 0; i < a.i; i++:

尾核:
for i = 0; i < a - a.i * a.o; i++:

场景3:
ub切b
(a, b.o, b.i, c, d)
nburst = c
burstlen = d * dype
burst_src_stride = stridec * d * dype
burst_dst_stride = d * dype

整块:
for i = 0; i < b.o; i++:
loop1_size = b.i
loop1_size_src_stide = strideb * D * C * dtype
loop1_size_dst_stide = d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

尾块：
if b % b.i != 0
loop1_size = b - b.i * b.o
loop1_size_src_stide = strideb * D * C * dtype
loop1_size_dst_stide = d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

场景4:
ub切a
(a.o, a.i, b, c, d)
nburst = c
burstlen = d * dype
burst_src_stride = stridec * D * dype
burst_dst_stride = d * dype
loop1_size = b
loop1_size_src_stide = strideb * D * C * dtype
loop1_size_dst_stide = d * c * dtype

整块:
for i = 0; i < a.o; i++:
loop2_size = a.i
loop2_size_src_stide = stridea * B * D * C * dtype
loop2_size_dst_stide = b * d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

尾块：
if a % a.i != 0
loop2_size = a - a.i * a.o
loop2_size_src_stide = stridea * B * D * C * dtype
loop2_size_dst_stide = b * d * c * dtype
mov2_align_v2(dst, src, nburst, burstlen, burst_src_stride, burst_dst_stride)

*/
void StrideSliceTiling::SetMovAlignV2TilingMode()
{
    int32_t shapeSize = dimNum_;
    // ubfactor切-1轴
    if (ubIndex_ == shapeSize - 1L) {
        MovAlignV2UbSplitLastOneDim();
        return;
    }

    // ubfactor切-2轴
    if (ubIndex_ == shapeSize - NUMBER_TWO) {
        MovAlignV2UbSplitLastTwoDim();
        return;
    }

    // ubfactor切-3轴
    if (ubIndex_ == shapeSize - NUMBER_THREE) {
        MovAlignV2UbSplitLastThreeDim();
        return;
    }

    // ubfactor切-4轴
    if (ubIndex_ == shapeSize - NUMBER_FOUR) {
        MovAlignV2UbSplitLastFourDim();
        return;
    }
}

void StrideSliceTiling::SetMoveAlignParams(StridedSliceMoveAlignParams& params, const MoveAlignV2Info& actInfo)
{
    params.set_blockCount(actInfo.blockCount);
    params.set_blockLen(actInfo.blockLen);
    params.set_srcStride(actInfo.srcStride);
    params.set_dstStride(actInfo.dstStride);
    params.set_loop1Size(actInfo.loop1Size);
    params.set_loop2Size(actInfo.loop2Size);
    params.set_loop1SrcStride(actInfo.loop1SrcStride);
    params.set_loop1DstStride(actInfo.loop1DstStride);
    params.set_loop2SrcStride(actInfo.loop2SrcStride);
    params.set_loop2DstStride(actInfo.loop2DstStride);
}

void StrideSliceTiling::SetShortMoveAlignParams(
    StridedSliceShortMoveAlignParams& params, const MoveAlignV2Info& actInfo)
{
    params.set_blockCount(actInfo.blockCount);
    params.set_blockLen(actInfo.blockLen);
    params.set_srcStride(actInfo.srcStride);
    params.set_dstStride(actInfo.dstStride);
}

void StrideSliceTiling::SetRowsStepsParams(StridedSliceTilingData& tilingData)
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - NUMBER_TWO; i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    int64_t inLoopSteps = 1;
    int64_t outLoopSteps = 1;
    // ub非最后一根轴
    if (ubIndex_ != dimNum_ - 1) {
        inLoopSteps = ubFactor_ * strides_[ubIndex_] * inputSteps_[ubIndex_ + 1];
        outLoopSteps = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps = ubFactor_ * strides_[ubIndex_];
        outLoopSteps = ubFactor_;
    }
    tilingData.set_ubInLoopSteps(inLoopSteps);
    tilingData.set_ubOutLoopSteps(outLoopSteps);

    tilingData.set_rowsOffsetSteps(rowsOffsetSteps_);
    tilingData.set_inputSteps(inputSteps_);
}

void StrideSliceTiling::CalStridedSliceRowsStepsParams()
{
    rowsOffsetSteps_[dimNum_ - 1] = 1;
    inputSteps_[dimNum_ - 1] = inputShape_[dimNum_ - 1];
    outputSteps_[dimNum_ - 1] = outputShape_[dimNum_ - 1];
    for (int32_t i = dimNum_ - NUMBER_TWO; i >= 0; i--) {
        rowsOffsetSteps_[i] = outputShape_[i] * rowsOffsetSteps_[i + 1];
        inputSteps_[i] = inputShape_[i] * inputSteps_[i + 1];
        outputSteps_[i] = outputShape_[i] * outputSteps_[i + 1];
    }

    // ub非最后一根轴
    if (ubIndex_ != dimNum_ - 1) {
        inLoopSteps_ = ubFactor_ * strides_[ubIndex_] * inputSteps_[ubIndex_ + 1];
        outLoopSteps_ = ubFactor_ * outputSteps_[ubIndex_ + 1];
    } else {
        inLoopSteps_ = ubFactor_ * strides_[ubIndex_];
        outLoopSteps_ = ubFactor_;
    }
}

void StrideSliceTiling::FillStridedSliceBaseTilingData(StridedSliceBaseTilingData& tilingData)
{
    if (isStrideNeg_ || useGather_) {
        tilingData.set_ubSize(ubSizeOutput_);
    } else {
        tilingData.set_ubSize(ubSize_);
    }
    tilingData.set_ubIndex(ubIndex_);
    tilingData.set_ubFactor(ubFactor_);
    tilingData.set_ubTailFactor(ubTailFactor_);
    tilingData.set_ubTailTailFactor(ubTailTailFactor_);
    tilingData.set_realCoreNum(realCoreNum_);
    tilingData.set_inputDims(dimNum_);
    tilingData.set_blkIndex(blkIndex_);
    tilingData.set_blkFactor(blkFactor_);
    tilingData.set_blkTailFactor(blkTailFactor_);
    tilingData.set_begin(begin_);
    tilingData.set_strides(strides_);
    tilingData.set_outputShape(outputShape_);

    CalStridedSliceRowsStepsParams();
    tilingData.set_ubInLoopSteps(inLoopSteps_);
    tilingData.set_ubOutLoopSteps(outLoopSteps_);
    tilingData.set_inputSteps(inputSteps_);
    tilingData.set_rowsOffsetSteps(rowsOffsetSteps_);
}

void StrideSliceTiling::FillStridedSliceTilingData100()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData100.");
    FillStridedSliceBaseTilingData(maTilingData_.stridedSliceBaseTilingData);

    SetMoveAlignParams(maTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    maTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(maTilingData_.GetDataSize());
}

void StrideSliceTiling::FillStridedSliceTilingData101()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData101.");
    FillStridedSliceBaseTilingData(maLastDimTilingData_.stridedSliceBaseTilingData);

    maLastDimTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(maLastDimTilingData_.GetDataSize());
}

void StrideSliceTiling::FillStridedSliceTilingDataNDDMA()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingDataNDDMA.");
    FillStridedSliceBaseTilingData(nddmaTilingData_.stridedSliceBaseTilingData);
    nddmaTilingData_.set_ubSizeInput(ubSizeInput_);
    nddmaTilingData_.set_nddmaTotalNum(nddmaTotalNum_);
    nddmaTilingData_.set_nddmaLoopSize(nddmaLoopSize_);
    nddmaTilingData_.set_nddmaLoopSrcStride(nddmaLoopSrcStride_);
    nddmaTilingData_.set_nddmaLoopDstStride(nddmaLoopDstStride_);

    nddmaTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(nddmaTilingData_.GetDataSize());
}

void StrideSliceTiling::SetRowsStepsParamsFor150(StridedSliceMALast2DimTilingData& tilingData)
{
    CalStridedSliceRowsStepsParams();
    tilingData.set_ubInLoopSteps(inLoopSteps_);
    tilingData.set_ubOutLoopSteps(outLoopSteps_);
    int64_t inputSteps[NUMBER_TWO];
    for (size_t i = 0; i < NUMBER_TWO; i++) {
        inputSteps[i] = inputSteps_[i];
    }
    tilingData.set_inputSteps(inputSteps);
}

void StrideSliceTiling::FillStridedSliceTilingData150()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData150.");
    int64_t begin[NUMBER_TWO];
    int64_t strides[NUMBER_TWO];
    int64_t outputShape[NUMBER_TWO];
    for (size_t i = 0; i < NUMBER_TWO; i++) {
        begin[i] = begin_[i];
        strides[i] = strides_[i];
        outputShape[i] = outputShape_[i];
    }
    maLast2DimTilingData_.set_ubSize(ubSize_);
    maLast2DimTilingData_.set_blkFactor(blkFactor_);
    maLast2DimTilingData_.set_blkTailFactor(blkTailFactor_);
    maLast2DimTilingData_.set_ubFactor(ubFactor_);
    maLast2DimTilingData_.set_ubTailFactor(ubTailFactor_);
    maLast2DimTilingData_.set_ubTailTailFactor(ubTailTailFactor_);
    maLast2DimTilingData_.set_realCoreNum(realCoreNum_);

    SetShortMoveAlignParams(maLast2DimTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    maLast2DimTilingData_.set_begin(begin);
    maLast2DimTilingData_.set_strides(strides);
    maLast2DimTilingData_.set_outputShape(outputShape);
    SetRowsStepsParamsFor150(maLast2DimTilingData_);
    maLast2DimTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(maLast2DimTilingData_.GetDataSize());
}

void StrideSliceTiling::FillStridedSliceTilingDataSIMT()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingDataSIMT.");
    simtTilingData_.set_isEmptyTensor(isEmptyTensor_);
    simtTilingData_.set_inputDims(dimNum_);
    simtTilingData_.set_begin(begin_);
    simtTilingData_.set_strides(strides_);
    simtTilingData_.set_outputShapeProd(outputShapeProd_);
    simtTilingData_.set_inputShapeProd(inputShapeProd_);
    simtTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(simtTilingData_.GetDataSize());
}

void StrideSliceTiling::FillStridedSliceTilingData300()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData300.");
    FillStridedSliceBaseTilingData(maGatherTilingData_.stridedSliceBaseTilingData);
    maGatherTilingData_.set_ubSizeInput(ubSizeInput_);
    SetMoveAlignParams(maGatherTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    maGatherTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(maGatherTilingData_.GetDataSize());
}

void StrideSliceTiling::FillStridedSliceTilingData301()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData301.");
    FillStridedSliceBaseTilingData(maUB2UBTilingData_.stridedSliceBaseTilingData);
    maUB2UBTilingData_.set_ubSizeInput(ubSizeInput_);
    SetMoveAlignParams(maUB2UBTilingData_.moveAlignParams, mainMoveAlignV2Info_);
    maUB2UBTilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(maUB2UBTilingData_.GetDataSize());
}

void StrideSliceTiling::FillStridedSliceTilingDataOther()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering other FillTilingData.");
    if (isStrideNeg_ || useGather_) {
        tilingData_.set_ubSize(ubSizeOutput_);
        tilingData_.set_ubSizeInput(ubSizeInput_);
    } else {
        tilingData_.set_ubSize(ubSize_);
    }
    tilingData_.set_coreNum(coreNum_);
    tilingData_.set_ubIndex(ubIndex_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_ubTailFactor(ubTailFactor_);
    tilingData_.set_ubTailTailFactor(ubTailTailFactor_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_inputDims(dimNum_);
    tilingData_.set_blkIndex(blkIndex_);
    tilingData_.set_blkFactor(blkFactor_);
    tilingData_.set_blkTailFactor(blkTailFactor_);
    tilingData_.set_xDtypeSize(xDtypeSize_);
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_nddmaTotalNum(nddmaTotalNum_);
    tilingData_.set_nddmaLoopSize(nddmaLoopSize_);
    tilingData_.set_nddmaLoopSrcStride(nddmaLoopSrcStride_);
    tilingData_.set_nddmaLoopDstStride(nddmaLoopDstStride_);
    tilingData_.set_outputShapeProd(outputShapeProd_);
    tilingData_.set_inputShapeProd(inputShapeProd_);
    tilingData_.set_isShapeExceedUint32(isShapeExceedUint32_);
    tilingData_.set_isEmptyTensor(isEmptyTensor_);

    SetMoveAlignParams(tilingData_.moveAlignParams, mainMoveAlignV2Info_);

    tilingData_.set_begin(begin_);
    tilingData_.set_strides(strides_);
    tilingData_.set_outputShape(outputShape_);

    SetRowsStepsParams(tilingData_);
    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void StrideSliceTiling::FillTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData.");
    for (size_t i = 0; i < sliceParam_.inputShape.GetDimNum(); i++) {
        begin_[i] = sliceParam_.beginList.GetDim(i);
        end_[i] = sliceParam_.endList.GetDim(i);
        strides_[i] = sliceParam_.strideList.GetDim(i);
        inputShape_[i] = sliceParam_.inputShape.GetDim(i);
        outputShape_[i] = sliceParam_.outputShape.GetDim(i);
    }
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            FillStridedSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            FillStridedSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
        case STRIDED_SLICE_KEY_NDDMA_GATHER:
        case STRIDED_SLICE_KEY_NDDMA_UB2UB:
            FillStridedSliceTilingDataNDDMA();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            FillStridedSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_SIMT:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE:
            FillStridedSliceTilingDataSIMT();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
            FillStridedSliceTilingData300();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB:
            FillStridedSliceTilingData301();
            break;
        default:
            FillStridedSliceTilingDataOther();
            break;
    }
}

void StrideSliceTiling::PrintTilingData()
{
    switch (tilingKey_) {
        case STRIDED_SLICE_KEY_MOVE_ALIGN:
            PrintStridedSliceTilingData100();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_LAST_DIM:
            PrintStridedSliceTilingData101();
            break;
        case STRIDED_SLICE_KEY_NDDMA:
        case STRIDED_SLICE_KEY_NDDMA_LAST_DIM:
        case STRIDED_SLICE_KEY_NDDMA_GATHER:
        case STRIDED_SLICE_KEY_NDDMA_UB2UB:
            PrintStridedSliceTilingDataNDDMA();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_TWO_DIM:
            PrintStridedSliceTilingData150();
            break;
        case STRIDED_SLICE_KEY_SIMT:
        case STRIDED_SLICE_KEY_SIMT_BIG_SHAPE:
            PrintStridedSliceTilingDataSIMT();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_GATHER:
            PrintStridedSliceTilingData300();
            break;
        case STRIDED_SLICE_KEY_MOVE_ALIGN_UB2UB:
            PrintStridedSliceTilingData301();
            break;
        default:
            PrintStridedSliceTilingDataOther();
            break;
    }
}

void StrideSliceTiling::PrintStridedSliceBaseTilingData(StridedSliceBaseTilingData& tilingData)
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceBaseTilingData is isStrideNeg_:%d, ubSize:%d, realCoreNum:%d, \
            ubIndex:%d, ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkIndex:%d, blkFactor:%ld, blkTailFactor:%ld, \
            begin:%s, end:%s, stride:%s, inputShape:%s, outputShape:%s, \
            rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld, ubOutLoopSteps:%ld",
        isStrideNeg_, tilingData.get_ubSize(), tilingData.get_realCoreNum(), tilingData.get_ubIndex(),
        tilingData.get_ubFactor(), tilingData.get_ubTailFactor(), tilingData.get_ubTailTailFactor(),
        tilingData.get_blkIndex(), tilingData.get_blkFactor(), tilingData.get_blkTailFactor(),
        ArrayToStr(tilingData.get_begin(), dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(tilingData.get_strides(), dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(tilingData.get_outputShape(), dimNum_).c_str(),
        ArrayToStr(tilingData.get_rowsOffsetSteps(), dimNum_).c_str(),
        ArrayToStr(tilingData.get_inputSteps(), dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str(),
        tilingData.get_ubInLoopSteps(), tilingData.get_ubOutLoopSteps());
}

void StrideSliceTiling::PrintStridedSliceTilingData100()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMATilingData:");
    PrintStridedSliceBaseTilingData(maTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceMATilingData is \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
        mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
        mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride, mainMoveAlignV2Info_.loop2SrcStride,
        mainMoveAlignV2Info_.loop2DstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingData101()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMALastDimTilingData:");
    PrintStridedSliceBaseTilingData(maLastDimTilingData_.stridedSliceBaseTilingData);
}

void StrideSliceTiling::PrintStridedSliceTilingDataNDDMA()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceNDDMATilingData:");
    PrintStridedSliceBaseTilingData(nddmaTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceNDDMATilingData is  ubSizeInput:%d\
    nddmaTotalNum:%ld, nddmaLoopSize:%s, nddmaLoopSrcStride: %s, nddmaLoopDstStride: %s",
        nddmaTilingData_.get_ubSizeInput(), nddmaTilingData_.get_nddmaTotalNum(),
        ArrayToStr(nddmaTilingData_.get_nddmaLoopSize(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(nddmaTilingData_.get_nddmaLoopSrcStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(nddmaTilingData_.get_nddmaLoopDstStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str());
}

void StrideSliceTiling::PrintStridedSliceTilingData150()
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceMALast2DimTilingData is ubSize:%d, realCoreNum:%d, \
            ubFactor:%d, ubTailFactor:%d, ubTailTailFactor:%d, \
            blkFactor:%ld, blkTailFactor:%ld, \
            begin:%s, end:%s, inputShape:%s, outputShape:%s, \
            inputSteps:%s, outputSteps:%s, ubInLoopSteps:%ld, ubOutLoopSteps:%ld, \
            moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u",
        maLast2DimTilingData_.get_ubSize(), maLast2DimTilingData_.get_realCoreNum(),
        maLast2DimTilingData_.get_ubFactor(), maLast2DimTilingData_.get_ubTailFactor(),
        maLast2DimTilingData_.get_ubTailTailFactor(), maLast2DimTilingData_.get_blkFactor(),
        maLast2DimTilingData_.get_blkTailFactor(), ArrayToStr(maLast2DimTilingData_.get_begin(), dimNum_).c_str(),
        ArrayToStr(end_, dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(maLast2DimTilingData_.get_outputShape(), dimNum_).c_str(),
        ArrayToStr(maLast2DimTilingData_.get_inputSteps(), dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str(),
        maLast2DimTilingData_.get_ubInLoopSteps(), maLast2DimTilingData_.get_ubOutLoopSteps(),
        mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
        mainMoveAlignV2Info_.dstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingDataSIMT()
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceSIMTTilingData is isStrideNeg_:%d, isEmptyTensor:%d, begin:%s, stride:%s \
            outputShapeProd:%s, inputShapeProd:%s",
        isStrideNeg_, simtTilingData_.get_isEmptyTensor(), ArrayToStr(simtTilingData_.get_begin(), dimNum_).c_str(),
        ArrayToStr(simtTilingData_.get_strides(), dimNum_).c_str(),
        ArrayToStr(simtTilingData_.get_outputShapeProd(), MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(simtTilingData_.get_inputShapeProd(), MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str());
}

void StrideSliceTiling::PrintStridedSliceTilingData300()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMAGatherTilingData:");
    PrintStridedSliceBaseTilingData(maGatherTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceMAGatherTilingData is ubSizeInput:%d \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        maGatherTilingData_.get_ubSizeInput(), mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen,
        mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size,
        mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
        mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingData301()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Printing StridedSliceMAUB2UBTilingData:");
    PrintStridedSliceBaseTilingData(maUB2UBTilingData_.stridedSliceBaseTilingData);
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "StridedSliceMAUB2UBTilingData is ubSizeInput:%d \
    moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
    loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u",
        maUB2UBTilingData_.get_ubSizeInput(), mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen,
        mainMoveAlignV2Info_.srcStride, mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size,
        mainMoveAlignV2Info_.loop2Size, mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride,
        mainMoveAlignV2Info_.loop2SrcStride, mainMoveAlignV2Info_.loop2DstStride);
}

void StrideSliceTiling::PrintStridedSliceTilingDataOther()
{
    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is isStrideNeg_:%d ubSize:%ld ubSizeInput:%ld, coreNum:%ld, realCoreNum:%ld, \
          ubIndex:%ld, ubFactor:%ld, ubTailFactor:%ld, ubTailTailFactor:%ld, \
          blkIndex:%ld, blkFactor:%ld, blkTailFactor:%ld, xDtypeSize:%ld, tilingKey:%ld, isShapeExceedUint32:%d, \
          isEmptyTensor:%d, begin:%s, end:%s, stride:%s, inputShape:%s, outputShape:%s, \
          rowsOffsetSteps:%s, inputSteps:%s, outputSteps:%s",
        isStrideNeg_, tilingData_.get_ubSize(), ubSizeInput_, coreNum_, tilingData_.get_realCoreNum(),
        tilingData_.get_ubIndex(), tilingData_.get_ubFactor(), tilingData_.get_ubTailFactor(),
        tilingData_.get_ubTailTailFactor(), tilingData_.get_blkIndex(), tilingData_.get_blkFactor(),
        tilingData_.get_blkTailFactor(), tilingData_.get_xDtypeSize(), tilingData_.get_tilingKey(),
        tilingData_.get_isShapeExceedUint32(), tilingData_.get_isEmptyTensor(),
        ArrayToStr(tilingData_.get_begin(), dimNum_).c_str(), ArrayToStr(end_, dimNum_).c_str(),
        ArrayToStr(tilingData_.get_strides(), dimNum_).c_str(), ArrayToStr(inputShape_, dimNum_).c_str(),
        ArrayToStr(tilingData_.get_outputShape(), dimNum_).c_str(),
        ArrayToStr(tilingData_.get_rowsOffsetSteps(), dimNum_).c_str(),
        ArrayToStr(tilingData_.get_inputSteps(), dimNum_).c_str(), ArrayToStr(outputSteps_, dimNum_).c_str());

    OP_LOGI(
        tilingContext_->GetNodeName(),
        "tilingData is nddmaTotalNum:%ld nddmaLoopSize:%s, nddmaLoopSrcStride: %s, \
          nddmaLoopDstStride: %s, moveAlignInfo: blockCount:%u blockLen:%u srcStride:%u dstStride:%u loop1Size:%u \
          loop2Size:%u loop1SrcStride:%u loop1DstStride:%u loop2SrcStride:%u loop2DstStride:%u outputShapeProd: %s \
          inputShapeProd: %s Tiling4StrideSlice ends.",
        tilingData_.get_nddmaTotalNum(),
        ArrayToStr(tilingData_.get_nddmaLoopSize(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData_.get_nddmaLoopSrcStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData_.get_nddmaLoopDstStride(), MAX_NDDMA_UB_SPLIT_AXIS_NUM).c_str(),
        mainMoveAlignV2Info_.blockCount, mainMoveAlignV2Info_.blockLen, mainMoveAlignV2Info_.srcStride,
        mainMoveAlignV2Info_.dstStride, mainMoveAlignV2Info_.loop1Size, mainMoveAlignV2Info_.loop2Size,
        mainMoveAlignV2Info_.loop1SrcStride, mainMoveAlignV2Info_.loop1DstStride, mainMoveAlignV2Info_.loop2SrcStride,
        mainMoveAlignV2Info_.loop2DstStride,
        ArrayToStr(tilingData_.get_outputShapeProd(), MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str(),
        ArrayToStr(tilingData_.get_inputShapeProd(), MAX_SIMT_UB_SPLIT_AXIS_NUM).c_str());
}

void StrideSliceTiling::SetBlockDimAndTilingKey()
{
    tilingContext_->SetBlockDim(realCoreNum_);
    tilingContext_->SetTilingKey(tilingKey_);
}

std::string StrideSliceTiling::ArrayToStr(const int64_t* arr, size_t aSize) const
{
    std::string result;
    for (size_t i = 0; i < aSize; i++) {
        result = result + std::to_string(arr[i]) + " ";
    }
    return result;
}

template <typename T>
static bool AssignInputValueConst(
    const gert::Tensor* tensor, ops::QuickVector& sliceList, bool isAscendc, bool depend, bool& isConst)
{
    int32_t dimNum = tensor->GetShapeSize();
    const T* data = tensor->GetData<T>();
    OP_CHECK_IF(
        (!isAscendc || depend) && data == nullptr,
        OP_LOGE(OP_NAME, "get const value fail, check input is const or not."), return false);
    if (!data) {
        isConst = false;
    } else {
        isConst = true;
        sliceList.SetDimNum(dimNum);
        for (int32_t i = 0; i < dimNum; i++) {
            sliceList[i] = static_cast<int64_t>(data[i]);
        }
    }
    return true;
}

static bool ConstructSliceList(
    const gert::Tensor* tensor, ops::QuickVector& sliceList, bool isAscendc, bool depend, bool& isConst)
{
    if (tensor->GetDataType() == ge::DT_INT32) {
        return AssignInputValueConst<int32_t>(tensor, sliceList, isAscendc, depend, isConst);
    } else if (tensor->GetDataType() == ge::DT_INT64) {
        return AssignInputValueConst<int64_t>(tensor, sliceList, isAscendc, depend, isConst);
    } else {
        OP_LOGD(OP_NAME, "data type is invalid %d", tensor->GetDataType());
        return false;
    }
    return true;
}

static void ConstructSliceShape(const gert::StorageShape* storage, gert::Shape& param)
{
    const gert::Shape& shape = Ops::Base::EnsureNotScalar(storage->GetStorageShape());
    int32_t dimNum = static_cast<int32_t>(shape.GetDimNum());
    param.SetDimNum(dimNum);
    for (int32_t i = 0; i < dimNum; i++) {
        param[i] = shape.GetDim(i);
    }
}

static bool CheckStride(ops::QuickVector& stride, const gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const StridedSliceCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    for (size_t i = 0; i < stride.GetDimNum(); i++) {
        if (stride[i] == 0) {
            OP_LOGD(OP_NAME, "stride is %ld, it must be non-zero.", stride[i]);
            return false;
        }
        if (!compileInfo->isAscendc && stride[i] < 0) {
            OP_LOGD(OP_NAME, "stride is %ld, if not ascendc, it must be greater than zero.", stride[i]);
            return false;
        }
    }
    return true;
}

static ge::graphStatus ConstructSliceParam(
    const gert::TilingContext* context, SliceParametersRuntime2& sliceParam, bool isAscendc)
{
    // construct slice_param.input, slice_param.output_shape
    const gert::StorageShape* xStorage = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::StorageShape* yStorage = context->GetOutputShape(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    ConstructSliceShape(xStorage, sliceParam.inputShape);
    ConstructSliceShape(yStorage, sliceParam.outputShape);

    // construct slice_param.begin_list, slice_param.end_list, slice_param.stride_list
    const gert::Tensor* tensorBegin = context->GetInputTensor(INDEX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorBegin);
    const gert::Tensor* tensorEnd = context->GetInputTensor(INDEX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorEnd);
    const gert::Tensor* tensorStrides = context->GetInputTensor(INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorStrides);
    if (!ConstructSliceList(tensorBegin, sliceParam.beginList, isAscendc, false, sliceParam.isBeginConst)) {
        return ge::GRAPH_FAILED;
    }
    if (!ConstructSliceList(tensorEnd, sliceParam.endList, isAscendc, false, sliceParam.isEndConst)) {
        return ge::GRAPH_FAILED;
    }
    bool isStridesConst = true;
    if (!ConstructSliceList(tensorStrides, sliceParam.strideList, isAscendc, true, isStridesConst)) {
        return ge::GRAPH_FAILED;
    }

    // check slice_param.stride_list valid, only support value==1
    if (!CheckStride(sliceParam.strideList, context)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ProcBeginEndUnconst(
    const gert::TilingContext* context, SliceParametersRuntime2& sliceParam)
{
    if (sliceParam.isBeginConst && sliceParam.isEndConst) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::Tensor* tensorBegin = context->GetInputTensor(INDEX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorBegin);
    const gert::Tensor* tensorEnd = context->GetInputTensor(INDEX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorEnd);

    int64_t beginLen = -1;
    beginLen = std::max(tensorBegin->GetShapeSize(), beginLen);
    beginLen = std::max(tensorEnd->GetShapeSize(), beginLen);
    beginLen = std::max(static_cast<int64_t>(sliceParam.strideList.GetDimNum()), beginLen);

    OP_CHECK_IF(
        beginLen == -1, OP_LOGE(OP_NAME, "beginLen invalid while nonconst"),
        return ge::GRAPH_FAILED);

    // begin or end被mask掉的话，infershape会依据strides的正负重新给begin/end赋值
    // 此处只需要考虑strides为正
    if (!sliceParam.isBeginConst) {
        sliceParam.beginList.SetDimNum(beginLen);
        for (int32_t i = 0; i < beginLen; i++) {
            sliceParam.beginList[i] = 0;
        }
    }

    // 参考rt1.0 infershape
    if (!sliceParam.isEndConst) {
        sliceParam.endList.SetDimNum(beginLen);
        int64_t inputDims = sliceParam.inputShape.GetDimNum();
        int64_t minLen = std::min(beginLen, inputDims);
        for (int32_t i = 0; i < minLen; i++) {
            sliceParam.endList[i] = sliceParam.inputShape.GetDim(i);
        }
        for (int32_t i = minLen; i < beginLen; i++) {
            sliceParam.endList[i] = sliceParam.inputShape.GetDim(inputDims - 1);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static void ReconstructSliceParamByInferShape(
    ops::StridedSliceParams& inputParams, gert::Shape& shapeOutput, SliceParametersRuntime2& sliceParam)
{
    sliceParam.beginList = inputParams.begin;
    sliceParam.endList = inputParams.end;
    sliceParam.strideList = inputParams.strides;
    sliceParam.inputShape.SetDimNum(0);
    for (size_t i = 0; i < inputParams.input_shape.GetDimNum(); i++) {
        sliceParam.inputShape.AppendDim(inputParams.input_shape.GetDim(i));
    }
    for (size_t i = 0; i < shapeOutput.GetDimNum(); i++) {
        sliceParam.outputShape.AppendDim(shapeOutput.GetDim(i));
    }
}

static void MakePerformanceParamsNeg(SliceParametersRuntime2& param)
{
    if (!param.isBeginConst || !param.isEndConst) {
        return;
    }
    OP_LOGI("", "before handle negative perf slice params: %s", param.to_string().c_str());
    SliceParametersRuntime2 perfParams;
    perfParams.isBeginConst = param.isBeginConst;
    perfParams.isEndConst = param.isEndConst;
    size_t perfSize = 0;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        const auto inputShapeI = param.inputShape[i];
        const auto outputShapeI = param.outputShape[i];
        const auto beginI = param.beginList[i];
        const auto endI = param.endList[i];
        const auto stride_i =
            endI > beginI ? std::min(param.strideList[i], endI - beginI) : std::max(param.strideList[i], endI - beginI);
        if (inputShapeI == 1 && outputShapeI == 1 && i != 0) {
            continue;
        }
        // Continuous stride=1 axis fused already. Continuous stride=-1 can fuse too.
        if (i == 0 || inputShapeI != outputShapeI || stride_i != -1 || perfParams.strideList[perfSize - 1] != -1) {
            perfParams.inputShape.AppendDim(inputShapeI);
            perfParams.outputShape.AppendDim(outputShapeI);
            perfParams.beginList.AppendDim(beginI);
            perfParams.endList.AppendDim(endI);
            perfParams.strideList.AppendDim(stride_i);
            perfSize++;
            continue;
        }

        const auto perfIndex = static_cast<size_t>(perfSize - 1);
        perfParams.inputShape[perfIndex] *= inputShapeI;
        perfParams.outputShape[perfIndex] *= outputShapeI;
        perfParams.beginList[perfIndex] = perfParams.beginList[perfIndex] * inputShapeI + inputShapeI - 1;
        perfParams.endList[perfIndex] = perfParams.endList[perfIndex] * inputShapeI + inputShapeI - 1;
        perfParams.strideList[perfIndex] = -1;
    }

    param = perfParams;

    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        if (param.outputShape[i] == 1 && param.strideList[i] < 0) {
            param.strideList[i] = 1;
            param.endList[i] = param.beginList[i] + 1;
        }
    }
}

static void MakePerformanceParams(SliceParametersRuntime2& param, bool isAdjustLastStride)
{
    if (param.outputShape.GetDimNum() == 0) {
        return;
    }

    bool hasNegStride = false;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        if (param.strideList[i] < 0) {
            hasNegStride = true;
            break;
        }
    }

    // Adjust params while input[-1] can be divided by stride[-1]
    if (isAdjustLastStride && !hasNegStride && param.isBeginConst && param.isEndConst) {
        size_t th = param.inputShape.GetDimNum();
        const auto inputLastDim = param.inputShape[th - 1U];
        const auto beginLastDim = param.beginList[th - 1U];
        const auto strideLastDim = param.strideList[th - 1U];
        const auto outputLastDim = param.outputShape[th - 1U];
        if (strideLastDim > 1 && inputLastDim % strideLastDim == 0 && beginLastDim / strideLastDim == 0 &&
            outputLastDim == inputLastDim / strideLastDim) {
            param.inputShape[th - 1] = inputLastDim / strideLastDim;
            param.inputShape.AppendDim(strideLastDim);

            param.beginList[th - 1] = beginLastDim / strideLastDim;
            param.beginList.AppendDim(beginLastDim % strideLastDim);

            param.strideList[th - 1U] = 1U;
            param.strideList.AppendDim(1);

            param.outputShape.AppendDim(1);

            param.endList[th - 1] = param.beginList[th - 1] + outputLastDim;
            param.endList.AppendDim(param.beginList[th] + 1);
        }
    }

    SliceParametersRuntime2 perfParams;
    perfParams.isBeginConst = param.isBeginConst;
    perfParams.isEndConst = param.isEndConst;
    size_t perfSize = 0;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        const auto inputShapeI = param.inputShape[i];
        const auto outputShapeI = param.outputShape[i];
        const auto beginI = param.beginList[i];
        const auto endI = param.endList[i];
        const auto stride_i = endI > beginI ? std::min(param.strideList[i], endI - beginI) : param.strideList[i];
        if (i == 0 || inputShapeI != outputShapeI || stride_i != 1 || perfParams.strideList[perfSize - 1] != 1 ||
            (!param.isBeginConst && (beginI < 0 || (i > 0 && param.beginList[i - 1] < 0)))) {
            int64_t realBeginValue =
                (!param.isBeginConst && beginI < 0) ? (UNCONST_BEGIN_VALUE - static_cast<int64_t>(i)) : beginI;
            perfParams.inputShape.AppendDim(inputShapeI);
            perfParams.outputShape.AppendDim(outputShapeI);
            perfParams.beginList.AppendDim(realBeginValue);
            perfParams.endList.AppendDim(endI);
            perfParams.strideList.AppendDim(stride_i);
            perfSize++;
            continue;
        }

        const auto perfIndex = perfSize - static_cast<size_t>(1);
        perfParams.inputShape[perfIndex] *= inputShapeI;
        perfParams.outputShape[perfIndex] *= outputShapeI;
        perfParams.beginList[perfIndex] *= inputShapeI;
        perfParams.endList[perfIndex] *= inputShapeI;
        perfParams.strideList[perfIndex] = 1;
    }

    param = perfParams;

    if (hasNegStride) {
        MakePerformanceParamsNeg(param);
    }
}

static void MakeSameDims(SliceParametersRuntime2* parametersPtr)
{
    auto& parameters = *parametersPtr;
    bool sameSize = parameters.inputShape.GetDimNum() == parameters.beginList.GetDimNum() &&
                    parameters.inputShape.GetDimNum() == parameters.endList.GetDimNum() &&
                    parameters.inputShape.GetDimNum() == parameters.strideList.GetDimNum();
    if (!sameSize) {
        return;
    }

    parameters.outputShape.SetDimNum(0);
    for (size_t i = 0; i < parameters.inputShape.GetDimNum(); i++) {
        auto interval = parameters.endList[i] - parameters.beginList[i];
        auto strideI = parameters.strideList[i];
        if (strideI == 0) {
            strideI = 1;
        }
        int64_t outputSize = interval / strideI + (interval % strideI != 0 ? 1 : 0);
        parameters.outputShape.AppendDim(outputSize);
    }
}

ge::graphStatus StrideSliceTilingForAscendC(
    gert::TilingContext* tilingContext, int64_t coreNum, int64_t ubSize, int64_t cachelineSize,
    SliceParametersRuntime2& sliceParam, const ge::DataType& dtype)
{
    StrideSliceTiling tilingObject(tilingContext);
    if (tilingObject.Init(coreNum, ubSize, cachelineSize, sliceParam, dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunStrideSliceTiling();
}

ge::graphStatus TilingPrepare4StridedSlice(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init TransposeNddmaTiling.");
    auto ci = context->GetCompiledInfo<StridedSliceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    ci->isAscendc = true;
    OP_CHECK_IF((ci->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);

    ci->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        (ci->cacheLineSize == 0), OP_LOGE(context->GetNodeName(), "Failed to get cacheLineSize."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4StridedSlice(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const StridedSliceCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "compile info: %s.", compileInfo->to_string().c_str());

    // construct slice param
    SliceParametersRuntime2 sliceParam;
    if (ConstructSliceParam(context, sliceParam, compileInfo->isAscendc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // get mask attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* maskBegin = attrs->GetAttrPointer<int64_t>(IDX_MASK_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskBegin);
    const int64_t* maskEnd = attrs->GetAttrPointer<int64_t>(IDX_MASK_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskEnd);
    const int64_t* maskEllipsis = attrs->GetAttrPointer<int64_t>(IDX_MASK_ELLIPSIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskEllipsis);
    const int64_t* maskNewAxis = attrs->GetAttrPointer<int64_t>(IDX_MASK_NEW_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskNewAxis);
    const int64_t* maskShrinkAxis = attrs->GetAttrPointer<int64_t>(IDX_MASK_SHRINK_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShrinkAxis);

    if (ProcBeginEndUnconst(context, sliceParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // infer shape
    const gert::StorageShape* xStorage = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    const gert::Shape& shapeInput = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    ops::StridedSliceParams inputParams = {
        shapeInput,
        sliceParam.beginList,
        sliceParam.endList,
        sliceParam.strideList,
        static_cast<uint64_t>(*maskBegin),
        static_cast<uint64_t>(*maskEnd),
        static_cast<uint64_t>(*maskEllipsis),
        static_cast<uint64_t>(*maskNewAxis),
        static_cast<uint64_t>(*maskShrinkAxis),
        true,
        true,
        true,
        sliceParam.isBeginConst,
        sliceParam.isEndConst,
        shapeInput};
    gert::Shape shapeOutput;
    if (!ops::InferShape(inputParams, &shapeOutput)) {
        return ge::GRAPH_FAILED;
    }

    // reconstruct slice_param by infer shape
    ReconstructSliceParamByInferShape(inputParams, shapeOutput, sliceParam);

    // align slice param dims.
    MakeSameDims(&sliceParam);
    OP_LOGI(context->GetNodeName(), "align slice params: %s", sliceParam.to_string().c_str());
    // optimize formance slice param
    MakePerformanceParams(sliceParam, true);
    OP_LOGI(context->GetNodeName(), "perf slice params: %s", sliceParam.to_string().c_str());

    // infer tiling mode
    auto xDesc = context->GetInputDesc(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);

    if (compileInfo->isAscendc) {
        return StrideSliceTilingForAscendC(
            context, compileInfo->coreNum, compileInfo->ubSize, compileInfo->cacheLineSize, sliceParam,
            xDesc->GetDataType());
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StridedSlice)
    .Tiling(Tiling4StridedSlice)
    .TilingParse<StridedSliceCompileInfo>(TilingPrepare4StridedSlice);

} // namespace optiling