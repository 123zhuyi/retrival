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
 * \file pad_v3_tiling_arch35.cpp
 * \brief ac pad v3 tiling cpp
 */

#include "pad_v3_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "util/platform_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace AscendC;

namespace optiling {
static constexpr uint64_t CONSTANT_SLICE_BRANCH = 10000;
static constexpr uint64_t CONSTANT_SIMT_BRANCH = 20000;
static constexpr uint64_t CONSTANT_SIMT_BIG_SIZE_BRANCH = 20001;
static constexpr uint64_t CONSTANT_CUT_LAST_DIM_BRANCH = 30000;
static constexpr uint64_t CONSTANT_BIG_LAST_DIM_BRANCH = 30001;
static constexpr uint64_t CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH = 30002;
static constexpr uint64_t CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH = 30003;

static constexpr uint64_t REFLECT_SIMT_BRANCH = 21000;
static constexpr uint64_t REFLECT_SIMT_BIG_SIZE_BRANCH = 21001;
static constexpr uint64_t REFLECT_CUT_LAST_DIM_BRANCH = 31000;
static constexpr uint64_t REFLECT_BIG_LAST_DIM_BRANCH = 31001;
static constexpr uint64_t REFLECT_SMALL_LAST_DIM_GATHER_BRANCH = 31002;

static constexpr uint64_t SYMMETRIC_SIMT_BRANCH = 22000;
static constexpr uint64_t SYMMETRIC_SIMT_BIG_SIZE_BRANCH = 22001;
static constexpr uint64_t SYMMETRIC_CUT_LAST_DIM_BRANCH = 32000;
static constexpr uint64_t SYMMETRIC_BIG_LAST_DIM_BRANCH = 32001;
static constexpr uint64_t SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH = 32002;

static constexpr uint64_t EDGE_SIMT_BRANCH = 23000;
static constexpr uint64_t EDGE_SIMT_BIG_SIZE_BRANCH = 23001;
static constexpr uint64_t EDGE_CUT_LAST_DIM_BRANCH = 33000;
static constexpr uint64_t EDGE_BIG_LAST_DIM_BRANCH = 33001;
static constexpr uint64_t EDGE_SMALL_LAST_DIM_GATHER_BRANCH = 33002;

static constexpr uint64_t CIRCULAR_SIMT_BRANCH = 24000;
static constexpr uint64_t CIRCULAR_SIMT_BIG_SIZE_BRANCH = 24001;
static constexpr uint64_t CIRCULAR_CUT_LAST_DIM_BRANCH = 34000;
static constexpr uint64_t CIRCULAR_BIG_LAST_DIM_BRANCH = 34001;
static constexpr uint64_t CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH = 34002;

static constexpr uint64_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr size_t PADDINGS_IDX = 1;
static constexpr size_t PAIR = 2;
static constexpr uint64_t SIMT_BRANCH_SIZE = 4 * 1024;
static constexpr uint64_t UB_MAX_DATA_SIZE_PER_BUFFER = 64 * 1024;
static constexpr uint8_t MAX_DIM_NUM = 8;

static constexpr int64_t MIN_PER_UB_SIZE = 4096;    // Bytes
static constexpr double MIN_USED_CORES_RATIO = 0.8; // 80%
static constexpr uint8_t PAD_DIM_INDEX_SECOND = 2;
static constexpr uint8_t PAD_DIM_INDEX_THIRD = 3;
static constexpr uint8_t PAD_DIM_INDEX_FOURTH = 4;
static constexpr uint64_t EXPANSION_FACTOR = 2;
static constexpr uint64_t HALF_FACTOR = 2;
static constexpr uint64_t DIM_OFFSET_SCALE = 10;
static constexpr uint8_t UB_DIVIDER = 6;
static constexpr uint64_t CONST4 = 4;
static constexpr uint8_t CONST5 = 5;

static const std::unordered_map<ge::DataType, uint16_t> DATA_TYPE_TO_BYTES{
    {ge::DataType::DT_INT8, 1},   {ge::DataType::DT_UINT8, 1},   {ge::DataType::DT_INT16, 2},
    {ge::DataType::DT_UINT16, 2}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 2},
    {ge::DataType::DT_FLOAT, 4},  {ge::DataType::DT_INT32, 4},   {ge::DataType::DT_INT64, 8},
    {ge::DataType::DT_UINT32, 4}, {ge::DataType::DT_UINT64, 8},  {ge::DataType::DT_DOUBLE, 8},
    {ge::DataType::DT_BOOL, 1}};

template <typename T>
std::string PadACTiling::ToString(const T* value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ",";
    }
    r = r + "]";
    return r;
}

uint64_t PadACTiling::GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize)
{
    if (alignBlockSize == 0) {
        return 0;
    }
    return (inputSize + alignBlockSize - 1) / alignBlockSize * alignBlockSize;
}

void PadACTiling::FindSplitAxisLoop(bool isBigLastDim, uint64_t& dimSizeInUb, uint64_t& dimSizeInLast4Axis)
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey FindSplitAxisLoop.");
    dimSizeInUb = dtypeBytes_;
    dimSizeInLast4Axis = dimSizeInUb;
    // 找到切分轴
    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        if (isBigLastDim && i == static_cast<int64_t>(dimNum_ - 1)) {
            dimSizeInUb = GetSizeOfBlockAlign(dimSizeInUb * tilingData_->outShape[i], blockSize_);
        } else {
            dimSizeInUb *= tilingData_->outShape[i];
        }
        // 切分超过4根轴时只切最后4根轴，记录下最后4根轴的大小
        if (i == dimNum_ - PAD_DIM_INDEX_FOURTH) {
            dimSizeInLast4Axis = dimSizeInUb;
        }
        if (dimSizeInUb >= bufferSize_) {
            ubAxis_ = i;
            break;
        }
    }
}

void PadACTiling::DoFindSplitAxisByInput(bool isBigLastDim)
{
    uint64_t dimSizeInUb;
    uint64_t dimSizeInLast4Axis;
    FindSplitAxisLoop(isBigLastDim, dimSizeInUb, dimSizeInLast4Axis);
    // 维度超过4，满载后4个轴
    if (dimNum_ - ubAxis_ > PAD_DIM_INDEX_FOURTH) {
        ubAxis_ = dimNum_ - PAD_DIM_INDEX_FOURTH;
        ubFactor_ = tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        dimSizeInUb = dimSizeInLast4Axis / tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_FOURTH] *
                      tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    } else if (dimSizeInUb > bufferSize_) {
        dimSizeInUb /= tilingData_->outShape[ubAxis_];
        ubFactor_ = dimSizeInUb == 0 ? bufferSize_ : bufferSize_ / dimSizeInUb;
        if (ubFactor_ > tilingData_->inShape[ubAxis_]) {
            ubFactor_ = tilingData_->inShape[ubAxis_];
        }
        dimSizeInUb *= ubFactor_;
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    } else {
        ubFactor_ = tilingData_->inShape[ubAxis_];
        dimSizeInUb = dimSizeInUb / tilingData_->outShape[ubAxis_] * tilingData_->inShape[ubAxis_];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    }
}

void PadACTiling::DoFindSplitAxis(bool isBigLastDim)
{
    uint64_t dimSizeInUb;
    uint64_t dimSizeInLast4Axis;
    FindSplitAxisLoop(isBigLastDim, dimSizeInUb, dimSizeInLast4Axis);

    // 维度超过4，满载后4个轴
    if (dimNum_ - ubAxis_ > PAD_DIM_INDEX_FOURTH) {
        ubAxis_ = dimNum_ - PAD_DIM_INDEX_FOURTH;
        ubFactor_ = tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInLast4Axis, blockSize_);
        return;
    }

    if (dimSizeInUb > bufferSize_) {
        dimSizeInUb /= tilingData_->outShape[ubAxis_];
        if (dimSizeInUb != 0) {
            ubFactor_ = bufferSize_ / dimSizeInUb;
        }
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb * ubFactor_, blockSize_);
        return;
    }

    ubFactor_ = tilingData_->outShape[ubAxis_];
    outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
}

void PadACTiling::CalculateGatherOrScatter()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey CalculateGatherOrScatter.");
    uint8_t vlAxis = dimNum_ - 1;
    uint64_t dimSizeInVL = dtypeBytes_;

    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        dimSizeInVL *= tilingData_->outShape[i];
        if (dimSizeInVL >= vectorSize_) {
            vlAxis = i;
            break;
        }
    }

    if (dimNum_ - 1 - vlAxis == 1 &&
        tilingData_->inShape[dimNum_ - 1] * EXPANSION_FACTOR < tilingData_->outShape[dimNum_ - 1]) {
        tilingKey_ = CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH;
        return;
    }

    if (dimNum_ - 1 - vlAxis == PAD_DIM_INDEX_SECOND &&
        tilingData_->inShape[dimNum_ - 1] * tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_SECOND] * EXPANSION_FACTOR <
            tilingData_->outShape[dimNum_ - 1] * tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_SECOND]) {
        tilingKey_ = CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH;
        return;
    }
    tilingKey_ = CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH;
}

void PadACTiling::CalculateTilingKeyReflect()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKeyReflect.");
    if (outShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > CONST5) {
        tilingKey_ = padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_SIMT_BRANCH : REFLECT_SIMT_BRANCH;
        return;
    }
    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / UB_DIVIDER - vectorSize_, vectorSize_);
    additionTileSize_ = vectorSize_ * EXPANSION_FACTOR;
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER;
    }
    if (lastShapeSizeAlign > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        tilingKey_ = padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_CUT_LAST_DIM_BRANCH : REFLECT_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (lastShapeSizeAlign * EXPANSION_FACTOR > bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->outShape[dimNum_ - 1];
        tilingKey_ = padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_CUT_LAST_DIM_BRANCH : REFLECT_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        if (dtypeBytes_ == 1) {
            additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ * EXPANSION_FACTOR,
                                                    blockSize_);
        }
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - additionTileSize_) / UB_DIVIDER - vectorSize_, vectorSize_);
        DoFindSplitAxisByInput(true);
        tilingKey_ = padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_BIG_LAST_DIM_BRANCH : REFLECT_BIG_LAST_DIM_BRANCH;

        if (lastShapeSizeAlign * EXPANSION_FACTOR > outTileSize_) {
            bufferSize_ = GetSizeOfBlockAlign(ubSize_ / UB_DIVIDER - vectorSize_, vectorSize_);
            ubAxis_ = dimNum_ - 1;
            ubFactor_ = tilingData_->outShape[dimNum_ - 1];
            tilingKey_ = padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_CUT_LAST_DIM_BRANCH : REFLECT_CUT_LAST_DIM_BRANCH;
            outTileSize_ = bufferSize_;
            return TilingInfoTune();
        } else {
            TilingInfoTuneForNormal(lastShapeSizeAlign, padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_CUT_LAST_DIM_BRANCH :
                                                                                         REFLECT_CUT_LAST_DIM_BRANCH);
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign(
            (ubSize_ - vectorSize_ * EXPANSION_FACTOR - blockSize_ * PAD_DIM_INDEX_FOURTH) / UB_DIVIDER - vectorSize_,
            vectorSize_);
        DoFindSplitAxisByInput(false);
        tilingKey_ = padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_SMALL_LAST_DIM_GATHER_BRANCH :
                                                      REFLECT_SMALL_LAST_DIM_GATHER_BRANCH;
        additionTileSize_ = vectorSize_;
        TilingInfoTuneForNormal(lastShapeSizeAlign, padMode_ == ModeNum::SYMMETRIC ? SYMMETRIC_CUT_LAST_DIM_BRANCH :
                                                                                     REFLECT_CUT_LAST_DIM_BRANCH);
    }
}

void PadACTiling::CircularOnlyLastTiling(uint64_t lastShapeSizeAlign)
{
    ubAxis_ = dimNum_ - 1;
    ubFactor_ = tilingData_->inShape[dimNum_ - 1];
    tilingKey_ = CIRCULAR_CUT_LAST_DIM_BRANCH;
    additionTileSize_ = vectorSize_;
    bufferSize_ = lastShapeSizeAlign;
    outTileSize_ = bufferSize_;
    return;
}

bool PadACTiling::CheckTilingInfoSatisfied(PadV3UbTileInfo& tilingInfo)
{
    bool cutOutput = (padMode_ == ModeNum::CONSTANT || padMode_ == ModeNum::EDGE);
    tilingInfo.ubTotalCnt = Ops::Base::CeilDiv(
        cutOutput ? tilingData_->outShape[tilingInfo.ubSplitAxis] : tilingData_->inShape[tilingInfo.ubSplitAxis],
        static_cast<uint64_t>(tilingInfo.ubSplitFactor));
    for (uint64_t i = 0; i < tilingInfo.ubSplitAxis; i++) {
        tilingInfo.ubTotalCnt *= cutOutput ? tilingData_->outShape[i] : tilingData_->inShape[i];
    }

    tilingInfo.ubPerCoreCnt = 1;
    tilingInfo.usedCoreNum = tilingInfo.ubTotalCnt;
    if (tilingInfo.ubTotalCnt > coreNum_) {
        tilingInfo.ubPerCoreCnt = Ops::Base::CeilDiv(tilingInfo.ubTotalCnt, static_cast<int64_t>(coreNum_));
        tilingInfo.usedCoreNum = Ops::Base::CeilDiv(tilingInfo.ubTotalCnt, tilingInfo.ubPerCoreCnt);
    }

    if (static_cast<double>(tilingInfo.usedCoreNum) / static_cast<double>(coreNum_) >= MIN_USED_CORES_RATIO ||
        tilingInfo.ubSplitFactor * tilingData_->inStride[tilingInfo.ubSplitAxis] * dtypeBytes_ <= MIN_PER_UB_SIZE) {
        OP_LOGD(context_, "ubSplitAxis:%d ubSplitFactor:%d is ok, usedCoreNum:%ld dtypeBytes_:%u",
                tilingInfo.ubSplitAxis, tilingInfo.ubSplitFactor, tilingInfo.usedCoreNum, dtypeBytes_);
        return true;
    }

    return false;
}

void PadACTiling::GetOptimizeTiling(const PadV3UbTileInfo& oldTilingInfo, PadV3UbTileInfo& newTilingInfo)
{
    bool cutOutput = (padMode_ == ModeNum::CONSTANT || padMode_ == ModeNum::EDGE);
    int64_t outCount = 1;
    for (uint8_t i = 0; i < oldTilingInfo.ubSplitAxis; i++) {
        outCount *= cutOutput ? tilingData_->outShape[i] : tilingData_->inShape[i];
    }

    // ubPerCoreCnt 不发生变化为前提时，最多循环coreNum+dimNum_次就可以找到最优解, 这里仅做防死循环保护
    uint32_t maxLoop = coreNum_ + dimNum_;
    uint32_t loops = 0;
    bool finded = false;
    for (uint8_t iDim = oldTilingInfo.ubSplitAxis; iDim < dimNum_; iDim++) {
        if (iDim != oldTilingInfo.ubSplitAxis) {
            outCount *= cutOutput ? tilingData_->outShape[iDim - 1] : tilingData_->inShape[iDim - 1];
        }

        int64_t iDimFactor = (iDim == oldTilingInfo.ubSplitAxis) ?
                                 oldTilingInfo.ubSplitFactor :
                                 (cutOutput ? tilingData_->outShape[iDim] : tilingData_->inShape[iDim]);
        for (int64_t factor = iDimFactor; factor > 0;) {
            loops++;
            if (loops > maxLoop) {
                finded = true;
                OP_LOGD(context_, "loops:%u is bigger than maxLoop:%u", loops, maxLoop);
                break;
            }

            // iDimOuter 每次循环会增加1,最多增加 coreNum_ 后，tmpPerCount会增加1
            int64_t iDimOuter = Ops::Base::CeilDiv(cutOutput ? tilingData_->outShape[iDim] : tilingData_->inShape[iDim],
                                                   static_cast<uint64_t>(factor));
            int64_t tmpTotalCount = iDimOuter * outCount;
            int64_t tmpPerCount = tmpTotalCount > coreNum_ ?
                                      Ops::Base::CeilDiv(tmpTotalCount, static_cast<int64_t>(coreNum_)) :
                                      1;
            int64_t tmpCoreNum = tmpTotalCount > coreNum_ ? Ops::Base::CeilDiv(tmpTotalCount, tmpPerCount) :
                                                            tmpTotalCount;
            int64_t tmpFactor = Ops::Base::CeilDiv(cutOutput ? tilingData_->outShape[iDim] : tilingData_->inShape[iDim],
                                                   static_cast<uint64_t>(iDimOuter)); // 切分更均匀

            if (oldTilingInfo.ubPerCoreCnt != tmpPerCount) {
                OP_LOGD(context_, "iDim:%u factor:%ld tmpPerCount:%ld not equal ubPerCoreCnt:%ld", iDim, factor,
                        tmpPerCount, oldTilingInfo.ubPerCoreCnt);
                finded = true;
                break;
            }

            if (factor * tilingData_->inStride[iDim] * dtypeBytes_ < MIN_PER_UB_SIZE ||
                tmpFactor * tilingData_->inStride[iDim] * dtypeBytes_ < MIN_PER_UB_SIZE) {
                OP_LOGD(context_, "iDim:%u factor:%ld tmpFactor:%ld in ubSize is too small", iDim, factor, tmpFactor);
                finded = true;
                break;
            }

            newTilingInfo.ubSplitAxis = iDim;
            newTilingInfo.ubSplitFactor = tmpFactor;
            newTilingInfo.ubTotalCnt = tmpTotalCount;
            newTilingInfo.ubPerCoreCnt = tmpPerCount;
            newTilingInfo.usedCoreNum = tmpCoreNum;

            double usedRate = static_cast<double>(tmpCoreNum) / static_cast<double>(coreNum_);
            OP_LOGD(context_, "current iDim:%u factor:%ld iDimOuter:%ld tmpFactor:%ld tmpCoreNum:%ld usedRate:%f", iDim,
                    factor, iDimOuter, tmpFactor, tmpCoreNum, usedRate);
            if (usedRate >= MIN_USED_CORES_RATIO) {
                finded = true;
                break;
            }
            factor = tmpFactor - 1;
        }

        OP_LOGD(context_, "iDim:%u ubSplitAxis:%u ubSplitFactor:%u loops:%u finded:%d", iDim, newTilingInfo.ubSplitAxis,
                newTilingInfo.ubSplitFactor, loops, finded);

        if (finded) {
            break;
        }
    }
    // 如果切分轴变化，不是尾轴，且该轴满载，那还是切前一个轴且factor=1
    if (newTilingInfo.ubSplitAxis != dimNum_ - 1 && newTilingInfo.ubSplitAxis > oldTilingInfo.ubSplitAxis &&
        newTilingInfo.ubSplitFactor >= (cutOutput ? tilingData_->outShape[newTilingInfo.ubSplitAxis] :
                                                    tilingData_->inShape[newTilingInfo.ubSplitAxis])) {
        newTilingInfo.ubSplitAxis = newTilingInfo.ubSplitAxis - 1;
        newTilingInfo.ubSplitFactor = 1;
        OP_LOGD(context_, "Back to last axis, ubSplitAxis:%u ubSplitFactor:%u", newTilingInfo.ubSplitAxis,
                newTilingInfo.ubSplitFactor);
    }
}

void PadACTiling::TilingInfoTune()
{
    PadV3UbTileInfo oldTilingInfo;
    oldTilingInfo.ubSplitAxis = ubAxis_;
    oldTilingInfo.ubSplitFactor = ubFactor_;
    if (CheckTilingInfoSatisfied(oldTilingInfo)) {
        ubTotalCount_ = oldTilingInfo.ubTotalCnt;
        ubPerCount_ = oldTilingInfo.ubPerCoreCnt;
        coreNum_ = oldTilingInfo.usedCoreNum;
        return;
    }

    OP_LOGI(context_, "before optimize ubSplitAxis:%u ubSplitFactor:%u ubTotalCnt:%ld ubPerCoreCnt:%ld usedCoreNum:%u",
            oldTilingInfo.ubSplitAxis, oldTilingInfo.ubSplitFactor, oldTilingInfo.ubTotalCnt,
            oldTilingInfo.ubPerCoreCnt, oldTilingInfo.usedCoreNum);

    PadV3UbTileInfo newTilingInfo = oldTilingInfo;
    GetOptimizeTiling(oldTilingInfo, newTilingInfo);

    OP_LOGI(context_, "after optimize ubSplitAxis:%u ubSplitFactor:%u ubTotalCnt:%ld ubPerCoreCnt:%ld usedCoreNum:%u",
            newTilingInfo.ubSplitAxis, newTilingInfo.ubSplitFactor, newTilingInfo.ubTotalCnt,
            newTilingInfo.ubPerCoreCnt, newTilingInfo.usedCoreNum);

    ubAxis_ = newTilingInfo.ubSplitAxis;
    ubFactor_ = newTilingInfo.ubSplitFactor;
    ubTotalCount_ = newTilingInfo.ubTotalCnt;
    ubPerCount_ = newTilingInfo.ubPerCoreCnt;
    coreNum_ = newTilingInfo.usedCoreNum;
}

bool PadACTiling::IsCutLastDim()
{
    // 切尾轴模板: (额外空间256B + inputUb) * DB
    additionTileSize_ = vectorSize_;
    bufferSize_ = (ubSize_ - PAIR * vectorSize_) / HALF_FACTOR / vectorSize_ * vectorSize_;
    bufferSize_ = std::min(bufferSize_, UB_MAX_DATA_SIZE_PER_BUFFER);
    bufferSize_ = std::max(bufferSize_, vectorSize_);

    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
    if (lastShapeSizeAlign > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        ubFactor_ = std::min(static_cast<uint64_t>(ubFactor_), tilingData_->inShape[dimNum_ - 1]);
        tilingKey_ = CIRCULAR_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        TilingInfoTune();
        return true;
    }

    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (lastShapeSizeAlign * EXPANSION_FACTOR > bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        CircularOnlyLastTiling(lastShapeSizeAlign);
        TilingInfoTune();
        return true;
    }

    return false;
}

void PadACTiling::TilingInfoTuneForNormal(uint64_t lastShapeSizeAlign, uint64_t tilingBranch)
{
    TilingInfoTune();
    if (ubAxis_ == dimNum_ - 1) {
        tilingKey_ = tilingBranch;
        additionTileSize_ = (padMode_ == ModeNum::SYMMETRIC || padMode_ == ModeNum::REFLECT) ?
                                EXPANSION_FACTOR * vectorSize_ :
                                vectorSize_;
        bufferSize_ = lastShapeSizeAlign;
        outTileSize_ = bufferSize_;
    }
}

void PadACTiling::CalculateTilingKeyCircular()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKeyCircular.");
    if (outShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > CONST5) {
        tilingKey_ = CIRCULAR_SIMT_BRANCH;
        return;
    }

    if (IsCutLastDim()) {
        return;
    }

    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        // 辅助空间: max(尾轴左pad长度32B对齐 * 2, VL/2);   (辅助空间+outUb)*2
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->leftPad[dimNum_ - 1] * dtypeBytes_, blockSize_) * PAIR;
        additionTileSize_ = std::max(static_cast<uint64_t>(additionTileSize_), vectorSize_);
        bufferSize_ = (ubSize_ - PAIR * additionTileSize_) / PAIR / vectorSize_ * vectorSize_;
        bufferSize_ = std::min(bufferSize_, UB_MAX_DATA_SIZE_PER_BUFFER);
        bufferSize_ = std::max(bufferSize_, vectorSize_);
        if (lastShapeSizeAlign * EXPANSION_FACTOR > bufferSize_) {
            CircularOnlyLastTiling(lastShapeSizeAlign);
            return TilingInfoTune();
        }

        DoFindSplitAxisByInput(true);
        tilingKey_ = CIRCULAR_BIG_LAST_DIM_BRANCH;
        // 如果切完后会爆ub，走切W分支
        if ((outTileSize_ + additionTileSize_) * EXPANSION_FACTOR > ubSize_) {
            CircularOnlyLastTiling(lastShapeSizeAlign);
            return TilingInfoTune();
        }

        TilingInfoTuneForNormal(lastShapeSizeAlign, CIRCULAR_CUT_LAST_DIM_BRANCH);
    } else {
        // (inputUb + outputUb + 2*blockSize) * 2 + VL
        additionTileSize_ = vectorSize_;
        bufferSize_ = GetSizeOfBlockAlign(
            (ubSize_ - vectorSize_ * EXPANSION_FACTOR - blockSize_ * CONST4) / CONST4 - vectorSize_, vectorSize_);
        bufferSize_ = std::min(bufferSize_, UB_MAX_DATA_SIZE_PER_BUFFER);
        bufferSize_ = std::max(bufferSize_, vectorSize_);
        DoFindSplitAxisByInput(false);
        tilingKey_ = CIRCULAR_SMALL_LAST_DIM_GATHER_BRANCH;
        TilingInfoTuneForNormal(lastShapeSizeAlign, CIRCULAR_CUT_LAST_DIM_BRANCH);
    }
}

void PadACTiling::CalculateTilingKeyEdge()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKeyEdge.");
    // MovealignV2指令限制，轴较多时走SIMT
    if (outShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > PAD_DIM_INDEX_FOURTH ||
        (dimNum_ == PAD_DIM_INDEX_FOURTH && (tilingData_->leftPad[0] != 0 || rightPad_[0] != 0))) {
        tilingKey_ = EDGE_SIMT_BRANCH;
        return;
    }
    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / PAIR - vectorSize_, vectorSize_);
    additionTileSize_ = vectorSize_;
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER;
    }
    if (lastShapeSizeAlign > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        tilingKey_ = EDGE_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (lastShapeSizeAlign * EXPANSION_FACTOR > bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->outShape[dimNum_ - 1];
        tilingKey_ = EDGE_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        DoFindSplitAxis(true);
        if (tilingKey_ == EDGE_SIMT_BRANCH) {
            return;
        }
        tilingKey_ = EDGE_BIG_LAST_DIM_BRANCH;
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        // 如果切完后会爆ub，走切W分支
        if ((outTileSize_ + additionTileSize_) * EXPANSION_FACTOR > ubSize_) {
            ubAxis_ = dimNum_ - 1;
            ubFactor_ = tilingData_->outShape[dimNum_ - 1] / dtypeBytes_;
            tilingKey_ = EDGE_CUT_LAST_DIM_BRANCH;
            outTileSize_ = lastShapeSizeAlign;
            additionTileSize_ = vectorSize_;
            TilingInfoTune();
        } else {
            TilingInfoTuneForNormal(lastShapeSizeAlign, EDGE_CUT_LAST_DIM_BRANCH);
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - vectorSize_ * PAIR) / PAIR / PAIR, vectorSize_);
        DoFindSplitAxis(false);
        if (tilingKey_ == EDGE_SIMT_BRANCH) {
            return;
        }
        tilingKey_ = EDGE_SMALL_LAST_DIM_GATHER_BRANCH;
        additionTileSize_ = outTileSize_ + vectorSize_;
        TilingInfoTuneForNormal(lastShapeSizeAlign, EDGE_CUT_LAST_DIM_BRANCH);
    }
}

void PadACTiling::CalculateTilingKey()
{
    OP_LOGD(context_, "Start PadACTiling CalculateTilingKey.");
    if (outShapeSize_ <= SIMT_BRANCH_SIZE) {
        tilingKey_ = CONSTANT_SIMT_BRANCH;
        return;
    }
    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / PAIR - vectorSize_, vectorSize_);
    additionTileSize_ = vectorSize_;
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER;
    }
    if (lastShapeSizeAlign > bufferSize_) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_ / dtypeBytes_;
        tilingKey_ = CONSTANT_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w,但是只有一根轴 & w > 128B，也走切w分支
    if (lastShapeSizeAlign * EXPANSION_FACTOR > bufferSize_ ||
        (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->outShape[dimNum_ - 1];
        tilingKey_ = CONSTANT_CUT_LAST_DIM_BRANCH;
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    if (tilingData_->outShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR) {
        DoFindSplitAxis(true);
        if (tilingKey_ == CONSTANT_SIMT_BRANCH) {
            return;
        }
        tilingKey_ = CONSTANT_BIG_LAST_DIM_BRANCH;
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        // 如果切完后会爆ub，走切W分支
        if ((outTileSize_ + additionTileSize_) * EXPANSION_FACTOR > ubSize_) {
            ubAxis_ = dimNum_ - 1;
            ubFactor_ = tilingData_->outShape[dimNum_ - 1] / dtypeBytes_;
            tilingKey_ = CONSTANT_CUT_LAST_DIM_BRANCH;
            outTileSize_ = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1] * dtypeBytes_, vectorSize_);
            additionTileSize_ = vectorSize_;
            TilingInfoTune();
        } else {
            TilingInfoTuneForNormal(lastShapeSizeAlign, CONSTANT_CUT_LAST_DIM_BRANCH);
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - vectorSize_ * PAIR) / PAIR / PAIR, vectorSize_);
        DoFindSplitAxis(false);
        if (tilingKey_ == CONSTANT_SIMT_BRANCH) {
            return;
        }
        // 按vl切分判断是否走scatter
        CalculateGatherOrScatter();
        additionTileSize_ = outTileSize_ + vectorSize_;
        TilingInfoTuneForNormal(lastShapeSizeAlign, CONSTANT_CUT_LAST_DIM_BRANCH);
    }
}
void PadACTiling::DoTilingWithReflect()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithReflect.");
    CalculateTilingKeyReflect();
    if (tilingKey_ == REFLECT_SIMT_BRANCH || tilingKey_ == SYMMETRIC_SIMT_BRANCH) {
        DoTilingWithSIMTReflect();
        return;
    }
    tilingKey_ = tilingKey_ + (dimNum_ - ubAxis_) * DIM_OFFSET_SCALE;
}

void PadACTiling::DoTilingWithCircular()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithCircular.");
    CalculateTilingKeyCircular();
    if (tilingKey_ == CIRCULAR_SIMT_BRANCH) {
        DoTilingWithSIMTCircular();
        return;
    }

    tilingKey_ = tilingKey_ + (dimNum_ - ubAxis_) * DIM_OFFSET_SCALE;
}

void PadACTiling::DoTilingWithEdge()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithEdge.");
    CalculateTilingKeyEdge();
    if (tilingKey_ == EDGE_SIMT_BRANCH) {
        DoTilingWithSIMTEdge();
        return;
    }
    CaculateTilingParams();
}

void PadACTiling::CaculateTilingParams() { tilingKey_ = tilingKey_ + (dimNum_ - ubAxis_) * DIM_OFFSET_SCALE; }

void PadACTiling::DoTilingWithConstant()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithConstant.");
    CalculateTilingKey();
    if (tilingKey_ == CONSTANT_SIMT_BRANCH) {
        DoTilingWithSIMT();
        return;
    }
    CaculateTilingParams();
}

void PadACTiling::DoTilingWithSIMTReflect()
{
    tilingKey_ = REFLECT_SIMT_BRANCH;
    if (padMode_ == ModeNum::SYMMETRIC) {
        tilingKey_ = SYMMETRIC_SIMT_BRANCH;
    }

    if (inShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX || outShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX) {
        tilingKey_ = REFLECT_SIMT_BIG_SIZE_BRANCH;
        if (padMode_ == ModeNum::SYMMETRIC) {
            tilingKey_ = SYMMETRIC_SIMT_BIG_SIZE_BRANCH;
        }
    }
}

ge::graphStatus PadACTiling::DoTilingWithSliceOp()
{
    OP_LOGD(context_, "Start PadACTiling DoTilingWithSliceOp.");
    optiling::SliceParasRuntime2 param;
    for (size_t i = 0; i < dimNum_; i++) {
        const auto input_shape_i = tilingData_->inShape[i];
        const auto output_shape_i = tilingData_->outShape[i];
        const auto begin_i = -tilingData_->leftPad[i];
        const auto end_i = output_shape_i + begin_i;
        const auto stride_i = 1;
        param.input.AppendDim(input_shape_i);
        param.output_shape.AppendDim(output_shape_i);
        param.begin_list.AppendDim(begin_i);
        param.end_list.AppendDim(end_i);
        param.stride_list.AppendDim(stride_i);
    }
    OP_LOGI(context_, "input_shape_ = %s, output_shape_ = %s, begin_ = %s, end_ = %s, stride_ = %s",
            Ops::Base::ToString(param.input).c_str(), Ops::Base::ToString(param.output_shape).c_str(),
            Ops::Base::ToString(param.begin_list).c_str(), Ops::Base::ToString(param.end_list).c_str(),
            Ops::Base::ToString(param.stride_list).c_str());
    ge::graphStatus res = SliceTilingForAscendC(context_, coreNum_, ubSize_, cacheLineSize_, param, paramsDtype_);
    tilingKey_ += context_->GetTilingKey();
    if (res != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "SliceTilingForAscendc failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void PadACTiling::DoTilingWithSIMTEdge()
{
    tilingKey_ = EDGE_SIMT_BRANCH;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        tilingKey_ = EDGE_SIMT_BIG_SIZE_BRANCH;
    }
}

void PadACTiling::DoTilingWithSIMTCircular()
{
    tilingKey_ = CIRCULAR_SIMT_BRANCH;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        tilingKey_ = CIRCULAR_SIMT_BIG_SIZE_BRANCH;
    }
}

void PadACTiling::DoTilingWithSIMT()
{
    tilingKey_ = CONSTANT_SIMT_BRANCH;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        tilingKey_ = CONSTANT_SIMT_BIG_SIZE_BRANCH;
    }
}

void PadACTiling::EmptyTensorCollapse()
{
    OP_LOGD(context_, "Start PadACTiling EmptyTensorCollapse.");
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = outShapeSize_;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
}

void PadACTiling::EmptyTensorCollapseMode()
{
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = 0;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
}

ge::graphStatus PadACTiling::ComputeAfterPaddingsAndStrides()
{
    inShapeSize_ = 1UL;
    outShapeSize_ = 1UL;
    for (int64_t i = dimNum_ - 1; i >= 0; --i) {
        // 校验是否会出现pad为负且比inshape大的情况，为0时会走simt

        if (static_cast<int64_t>(tilingData_->inShape[i]) + tilingData_->leftPad[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) + rightPad_[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) + tilingData_->leftPad[i] + rightPad_[i] < 0) {
            std::string shapeMsg = std::to_string(static_cast<int64_t>(tilingData_->inShape[i]) +
                                                  tilingData_->leftPad[i] + rightPad_[i]);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "outShape", shapeMsg.c_str(),
                                                  "All axes of outShape must be non-negative.");
            return ge::GRAPH_FAILED;
        }
        // compute shape after padding
        tilingData_->outShape[i] = tilingData_->inShape[i] + tilingData_->leftPad[i] + rightPad_[i];
        // compute src stride
        tilingData_->inStride[i] = inShapeSize_;
        inShapeSize_ *= tilingData_->inShape[i];
        // compute dst stride
        tilingData_->outStride[i] = outShapeSize_;
        outShapeSize_ *= tilingData_->outShape[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::CheckModeInputParam(int64_t inShapeV, int64_t padFront, int64_t padBack)
{
    int64_t leftsubin = padFront - inShapeV;
    int64_t rightsubin = padBack - inShapeV;
    if (padMode_ == ModeNum::REFLECT && (0 < (leftsubin + 1) || 0 < (rightsubin + 1))) {
        std::string shapeMsg = std::to_string(padFront) + ", " + std::to_string(padBack) + " and " +
                               std::to_string(inShapeV);
        std::string reasonMsg = "When the mode is reflect, padFront and padBack must be less than inShape";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "padFront, padBack and inShape",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (padMode_ == ModeNum::SYMMETRIC && (0 < leftsubin || 0 < rightsubin)) {
        std::string shapeMsg = std::to_string(padFront) + ", " + std::to_string(padBack) + " and " +
                               std::to_string(inShapeV);
        std::string
            reasonMsg = "When the mode is symmetric, padFront and padBack must be less than or equal to inShape";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "padFront, padBack and inShape",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (padMode_ == ModeNum::CIRCULAR && (0 < leftsubin || 0 < rightsubin)) {
        std::string shapeMsg = std::to_string(padFront) + ", " + std::to_string(padBack) + " and " +
                               std::to_string(inShapeV);
        std::string reasonMsg = "When the mode is circular, padFront and padBack must be less than or equal to inShape";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "padFront, padBack and inShape",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (padMode_ == ModeNum::EDGE && inShapeV == 0 && (padFront != 0 || padBack != 0)) {
        std::string shapeMsg = std::to_string(padFront) + ", " + std::to_string(padBack) + " and " +
                               std::to_string(inShapeV);
        std::string reasonMsg = "When the mode is edge and inShape == 0, padFront and padBack must be 0";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "padFront, padBack and inShape",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DimensionCollapseMode()
{
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(context_,
            "Before collapse paddings, padMode_:%u dimNum is %u, shape is %s, left pad is %s, right pad is %s",
            padMode_, originalRank, ToString(tilingData_->inShape, originalRank).c_str(),
            Ops::Base::ToString(paddings_.padFront).c_str(), Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
        int64_t padFront = paddings_.padFront.GetDim(fastDim);
        int64_t padBack = paddings_.padBack.GetDim(fastDim);

        OP_CHECK_IF(CheckModeInputParam(tilingData_->inShape[fastDim], padFront, padBack) == ge::GRAPH_FAILED,
                    OP_LOGE(context_, "CheckModeInputParam failed."), return ge::GRAPH_FAILED);

        // 消除1轴
        if (tilingData_->inShape[fastDim] == 1 && (tilingData_->inShape[fastDim] + padFront + padBack) == 1) {
            fastDim++;
            dimNum_--;
            continue;
        }
        if (padFront > 0 || padBack > 0) {
            isPadAllNegative_ = false;
        }

        if (padFront < 0 || padBack < 0) {
            isPadAllPositive_ = false;
        }

        uint64_t collapsedShape = tilingData_->inShape[fastDim];
        int64_t collapsedPadFront = paddings_.padFront.GetDim(fastDim);
        int64_t collapsedPadBack = paddings_.padBack.GetDim(fastDim);
        fastDim++;
        if (0 == paddings_.padFront.GetDim(fastDim - 1) && 0 == paddings_.padBack.GetDim(fastDim - 1)) {
            while (fastDim < originalRank &&
                   (0 == paddings_.padFront.GetDim(fastDim) && 0 == paddings_.padBack.GetDim(fastDim))) {
                collapsedShape *= tilingData_->inShape[fastDim];
                collapsedPadFront *= tilingData_->inShape[fastDim];
                collapsedPadBack *= tilingData_->inShape[fastDim];
                fastDim++;
                dimNum_--;
            }
        }
        tilingData_->inShape[slowDim] = collapsedShape;
        tilingData_->leftPad[slowDim] = collapsedPadFront;
        rightPad_[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    // shape全1且不补pad的场景,消除1轴后就为空了
    if (dimNum_ == 0) {
        tilingData_->inShape[0] = 1;
        tilingData_->leftPad[0] = 0;
        rightPad_[0] = 0;
        dimNum_ = 1;
    }
    OP_LOGD(context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
            ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
            ToString(rightPad_, dimNum_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DimensionCollapse()
{
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(context_, "Before collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s",
            originalRank, ToString(tilingData_->inShape, originalRank).c_str(),
            Ops::Base::ToString(paddings_.padFront).c_str(), Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
        int64_t padFront = paddings_.padFront.GetDim(fastDim);
        int64_t padBack = paddings_.padBack.GetDim(fastDim);
        if (padFront < 0 || padBack < 0) {
            isPadAllPositive_ = false;
        }
        if (padFront > 0 || padBack > 0) {
            isPadAllNegative_ = false;
        }
        uint64_t collapsedShape = tilingData_->inShape[fastDim];
        int64_t collapsedPadFront = paddings_.padFront.GetDim(fastDim);
        int64_t collapsedPadBack = paddings_.padBack.GetDim(fastDim);
        fastDim++;
        while (fastDim < originalRank &&
               (0 == paddings_.padFront.GetDim(fastDim) && 0 == paddings_.padBack.GetDim(fastDim))) {
            collapsedShape *= tilingData_->inShape[fastDim];
            collapsedPadFront *= tilingData_->inShape[fastDim];
            collapsedPadBack *= tilingData_->inShape[fastDim];
            fastDim++;
            dimNum_--;
        }

        tilingData_->inShape[slowDim] = collapsedShape;
        tilingData_->leftPad[slowDim] = collapsedPadFront;
        rightPad_[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    OP_LOGD(context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
            ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
            ToString(rightPad_, dimNum_).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void PadACTiling::GetPaddingsToShape(const gert::Tensor* paddingsTensor)
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo GetPaddings GetPaddingsToShape.");
    const T* paddingsValue = paddingsTensor->GetData<T>();
    const size_t paddingsNum = paddingsTensor->GetShapeSize();

    size_t inputDimNum = paddingsNum / PAIR;
    paddings_.padFront.SetDimNum(inputDimNum);
    paddings_.padBack.SetDimNum(inputDimNum);
    size_t indexCof = 1;
    size_t indexOffset = inputDimNum;
    // choose diff padding handle method by paddings_contiguous
    if (paddingContiguous_) {
        indexCof = PAIR;
        indexOffset = 1;
    }
    for (size_t i = 0; i < inputDimNum; ++i) {
        paddings_.padFront.SetDim(i, paddingsValue[i * indexCof]);
        paddings_.padBack.SetDim(i, paddingsValue[i * indexCof + indexOffset]);
    }
}

ge::graphStatus PadACTiling::GetPaddings()
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo GetPaddings.");
    const gert::Tensor* paddingsTensor = context_->GetInputTensor(PADDINGS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingsTensor);
    ge::DataType paddingsDtype = paddingsTensor->GetDataType();
    switch (paddingsDtype) {
        case ge::DT_INT32: {
            GetPaddingsToShape<int32_t>(paddingsTensor);
            return ge::GRAPH_SUCCESS;
        }
        case ge::DT_INT64: {
            GetPaddingsToShape<int64_t>(paddingsTensor);
            return ge::GRAPH_SUCCESS;
        }
        default:
            std::string reasonMsg = "The dtype of paddings must be within the range [int32, int64].";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "paddings",
                                                  Ops::Base::ToString(paddingsDtype).c_str(), reasonMsg.c_str());
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus PadACTiling::GetShapesAndDtypes()
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo GetShapesAndDtypes.");
    // get input shape & input shape's dim num
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    dimNum_ = inShapeVal.GetDimNum();
    if (dimNum_ > MAX_DIM_NUM) {
        std::string reasonMsg = "The shape dim of input must be within the range [0, 8].";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "input", std::to_string(dimNum_).c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    for (uint16_t i = 0; i < dimNum_; ++i) {
        if (inShapeVal.GetDim(i) < 0) {
            std::string reasonMsg = "All dimensions of input must be greater than or equal to 0.";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input",
                                                  std::to_string(inShapeVal.GetDim(i)).c_str(), reasonMsg.c_str());

            return ge::GRAPH_FAILED;
        }
        if (inShapeVal.GetDim(i) == 0) {
            isEmptyTensor_ = true;
        }
        tilingData_->inShape[i] = inShapeVal.GetDim(i);
    }
    // 判断类型
    auto inputTensor = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputTensor);
    paramsDtype_ = inputTensor->GetDataType();
    dtypeBytes_ = GetSizeByDataType(paramsDtype_);
    if (isPadV3_ && padMode_ == ModeNum::CONSTANT) {
        auto inConstantValues = context_->GetInputDesc(PAIR);
        if (inConstantValues && paramsDtype_ != inConstantValues->GetDataType()) {
            std::string paramMsg = Ops::Base::ToString(inConstantValues->GetDataType()) + " and " +
                                   Ops::Base::ToString(paramsDtype_);
            std::string reasonMsg = "The dtype of constant_values must be the same as the dtype of input.";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "constant_values and input",
                                                   paramMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (paramsDtype_ == ge::DT_FLOAT4_E1M2 || paramsDtype_ == ge::DT_FLOAT4_E2M1) {
            dtypeBytes_ = GetSizeByDataType(ge::DT_INT8);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_, "Start PadACTiling GetShapeAttrsInfo isPadV3_:%d.", isPadV3_);
    if (isPadV3_) {
        auto const attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        auto* mode = attrs->GetAttrPointer<char>(0);
        // padv3 mode可选，默认constant; mirrorpad mode必选
        if (mode) {
            if (!strcmp(mode, "edge")) {
                padMode_ = ModeNum::EDGE;
            } else if (!strcmp(mode, "reflect") || (isMirrorPad_ && !strcmp(mode, "REFLECT"))) {
                padMode_ = ModeNum::REFLECT;
            } else if (!strcmp(mode, "symmetric") || (isMirrorPad_ && !strcmp(mode, "SYMMETRIC"))) {
                padMode_ = ModeNum::SYMMETRIC;
            } else if (!strcmp(mode, "circular")) {
                padMode_ = ModeNum::CIRCULAR;
            }
            if (!isMirrorPad_ && strcmp(mode, "constant") != 0 && strcmp(mode, "edge") != 0 &&
                strcmp(mode, "reflect") != 0 && strcmp(mode, "symmetric") != 0 && strcmp(mode, "circular") != 0) {
                std::string reasonMsg = "The value of mode must be in [constant, edge, reflect, symmetric, circular].";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mode", std::string(mode).c_str(),
                                                      reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
            if (isMirrorPad_ && strcmp(mode, "REFLECT") != 0 && strcmp(mode, "SYMMETRIC") != 0) {
                std::string reasonMsg = "The value of mode must be in [REFLECT and SYMMETRIC].";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mode", std::string(mode).c_str(),
                                                      reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            if (isMirrorPad_) {
                std::string reasonMsg = "The value of mode cannot be empty.";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mode", std::string(mode).c_str(),
                                                      reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        if (!isMirrorPad_) {
            auto* paddingContiguous = attrs->GetAttrPointer<bool>(1);
            if (paddingContiguous) {
                paddingContiguous_ = *paddingContiguous;
            }
        }
    }
    OP_CHECK_IF(GetShapesAndDtypes() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo GetShapesAndDtypes error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetPaddings() == ge::GRAPH_FAILED, OP_LOGE(context_, "Get padding info failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void PadACTiling::FillsAndPrintTilingData()
{
    tilingData_->dimNum = dimNum_;
    tilingData_->ubAxis = ubAxis_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->ubPerCount = ubPerCount_;
    tilingData_->ubTotalCount = ubTotalCount_;
    tilingData_->outTileSize = outTileSize_;
    tilingData_->additionTileSize = additionTileSize_;
    OP_LOGI(
        context_,
        "tilingData is dimNum = %u, ubAxis = %u, ubFactor = %u, ubPerCount = %u, ubTotalCount = %u, outTileSize = %u, additionTileSize = %u, \
        inShape: %s, outShape: %s, inStride: %s, outStride: %s, leftPad: %s, rightPad: %s, tilingKey: %lu, coreNum is %u dtype:%s",
        tilingData_->dimNum, tilingData_->ubAxis, tilingData_->ubFactor, tilingData_->ubPerCount,
        tilingData_->ubTotalCount, tilingData_->outTileSize, tilingData_->additionTileSize,
        ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->outShape, dimNum_).c_str(),
        ToString(tilingData_->inStride, dimNum_).c_str(), ToString(tilingData_->outStride, dimNum_).c_str(),
        ToString(tilingData_->leftPad, dimNum_).c_str(), ToString(rightPad_, dimNum_).c_str(), tilingKey_, coreNum_,
        Ops::Base::ToString(paramsDtype_).c_str());
}

ge::graphStatus PadACTiling::Init()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0U, OP_LOGE(context_, "Failed to core num."), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0U, OP_LOGE(context_, "Failed to ub size."), return ge::GRAPH_FAILED);
    blockSize_ = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    OP_CHECK_IF(blockSize_ == 0U, OP_LOGE(context_, "Failed to ub block size."), return ge::GRAPH_FAILED);
    vectorSize_ = static_cast<uint64_t>(Ops::Base::GetVRegSize(context_));
    OP_CHECK_IF(vectorSize_ == 0U, OP_LOGE(context_, "Failed to vector size."), return ge::GRAPH_FAILED);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(cacheLineSize_ == 0U, OP_LOGE(context_, "Failed to cache line size."), return ge::GRAPH_FAILED);
    OP_LOGD(context_,
            "platform info is coreNum_ = %u, ubSize_ = %lu, blockSize_ = %lu, vectorSize_ = %lu, cacheLineSize_ = %u",
            coreNum_, ubSize_, blockSize_, vectorSize_, cacheLineSize_);
    tilingData_ = context_->GetTilingData<PadACTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::Fp8Fp4ValidatePaddings()
{
    OP_LOGD(context_, "Start PadACTiling Fp8Fp4ValidatePaddings.");
    // fp8/fp4 输入数据类型时，pad数组中不能有负数
    size_t frontDimNum = paddings_.padFront.GetDimNum();
    for (size_t i = 0; i < frontDimNum; ++i) {
        int64_t frontValue = paddings_.padFront.GetDim(i);
        if (frontValue < 0) {
            std::string paramMsg = "padFront at index " + std::to_string(i);
            std::string
                reasonMsg = "When the dtype is fp4 or fp8, each value of paddings must be greater than or equal to 0.";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                                                  std::to_string(frontValue).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    size_t backDimNum = paddings_.padBack.GetDimNum();
    for (size_t i = 0; i < backDimNum; ++i) {
        int64_t backValue = paddings_.padBack.GetDim(i);
        if (backValue < 0) {
            std::string paramMsg = "backValue at index " + std::to_string(i);
            std::string
                reasonMsg = "When the dtype is fp4 or fp8, each value of paddings must be greater than or equal to 0.";

            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                                                  std::to_string(backValue).c_str(), reasonMsg.c_str());

            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::Fp4ValidateInShape()
{
    OP_LOGD(context_, "Start PadACTiling Fp4ValidateInShape.");
    // fp4 输入数据类型时，输入数据的最后一维度shape为偶数
    if (tilingData_->inShape[dimNum_ - 1] % HALF_FACTOR != 0) {
        std::string reasonMsg = "When the dtype is fp4, the last axis of input must be an even number.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input",
                                              std::to_string(tilingData_->inShape[dimNum_ - 1]).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::Fp4ValidatePaddings()
{
    OP_LOGD(context_, "Start PadACTiling Fp4ValidatePaddings.");
    // fp4 输入数据类型，左右pad的最后一维均为偶数
    size_t frontDimNum = paddings_.padFront.GetDimNum();
    int64_t frontValue = paddings_.padFront.GetDim(frontDimNum - 1);
    if (frontValue % static_cast<int64_t>(HALF_FACTOR) != 0) {
        std::string paramMsg = "the last axis of padFront";
        std::string reasonMsg = "When the dtype is fp4, the last axis of padFront must be an even number.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                                              std::to_string(frontValue).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    size_t backDimNum = paddings_.padBack.GetDimNum();
    int64_t backValue = paddings_.padBack.GetDim(backDimNum - 1);
    if (backValue % static_cast<int64_t>(HALF_FACTOR) != 0) {
        std::string paramMsg = "the last axis of padBack";
        std::string reasonMsg = "When the dtype is fp4, the last axis of padBack must be an even number.";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), paramMsg.c_str(),
                                              std::to_string(backValue).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void PadACTiling::Fp4TilingData()
{
    OP_LOGD(context_, "Start PadACTiling Fp4TilingData.");
    tilingData_->inShape[dimNum_ - 1] /= HALF_FACTOR;
    tilingData_->leftPad[dimNum_ - 1] /= HALF_FACTOR;
    rightPad_[dimNum_ - 1] /= HALF_FACTOR;
}

ge::graphStatus PadACTiling::DoTilingModeEdge()
{
    OP_CHECK_IF(DimensionCollapseMode() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling Edge Collapse error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadACTiling Edge ComputeAfterPaddingsAndStrides error."), return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapseMode();
        DoTilingWithSIMTEdge();
    } else if (isPadAllPositive_) {
        DoTilingWithEdge();
    } else if (isPadAllNegative_) {
        tilingKey_ = CONSTANT_SLICE_BRANCH;
        OP_CHECK_IF(DoTilingWithSliceOp() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling with slice op error."),
                    return ge::GRAPH_FAILED);
        context_->SetTilingKey(tilingKey_);
        isUseSlice_ = true;
    } else {
        DoTilingWithSIMTEdge();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DoTilingModeMirror()
{
    OP_CHECK_IF(DimensionCollapseMode() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling Mirror Collapse error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadACTiling Mirror ComputeAfterPaddingsAndStrides error."), return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapseMode();
        DoTilingWithSIMTReflect();
    } else if (isPadAllPositive_) {
        DoTilingWithReflect();
    } else if (isPadAllNegative_) {
        tilingKey_ = CONSTANT_SLICE_BRANCH;
        OP_CHECK_IF(DoTilingWithSliceOp() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling with slice op error."),
                    return ge::GRAPH_FAILED);
        context_->SetTilingKey(tilingKey_);
        isUseSlice_ = true;
    } else {
        DoTilingWithSIMTReflect();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DoTilingModeCircular()
{
    OP_CHECK_IF(DimensionCollapseMode() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling Circular Collapse error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadACTiling Circular ComputeAfterPaddingsAndStrides error."),
                return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapseMode();
        DoTilingWithSIMTCircular();
    } else if (isPadAllPositive_) {
        DoTilingWithCircular();
    } else if (isPadAllNegative_) {
        tilingKey_ = CONSTANT_SLICE_BRANCH;
        OP_CHECK_IF(DoTilingWithSliceOp() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling with slice op error."),
                    return ge::GRAPH_FAILED);
        context_->SetTilingKey(tilingKey_);
        isUseSlice_ = true;
    } else {
        DoTilingWithSIMTCircular();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DoTilingModeConstant()
{
    if (paramsDtype_ == ge::DT_HIFLOAT8 || paramsDtype_ == ge::DT_FLOAT8_E5M2 || paramsDtype_ == ge::DT_FLOAT8_E4M3FN ||
        paramsDtype_ == ge::DT_FLOAT8_E8M0 || paramsDtype_ == ge::DT_FLOAT4_E2M1 ||
        paramsDtype_ == ge::DT_FLOAT4_E1M2) {
        // fp8/fp4 输入数据类型时，pad数组中不能有负数
        OP_CHECK_IF(Fp8Fp4ValidatePaddings() == ge::GRAPH_FAILED,
                    OP_LOGE(context_, "PadACTiling Fp8Fp4ValidatePaddings error."), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(DimensionCollapse() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling Constant Collapse error."),
                return ge::GRAPH_FAILED);
    if (paramsDtype_ == ge::DT_FLOAT4_E2M1 || paramsDtype_ == ge::DT_FLOAT4_E1M2) {
        // fp4 输入数据类型时，输入数据的最后一维shape为偶数
        OP_CHECK_IF(Fp4ValidateInShape() == ge::GRAPH_FAILED,
                    OP_LOGE(context_, "PadACTiling Fp4ValidateInShape error."), return ge::GRAPH_FAILED);
        // fp4 时，左右pad的padding num为偶数，最后一维
        OP_CHECK_IF(Fp4ValidatePaddings() == ge::GRAPH_FAILED,
                    OP_LOGE(context_, "PadACTiling Fp4ValidatePaddings error."), return ge::GRAPH_FAILED);
        // 尾轴//2
        Fp4TilingData();
    }
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadACTiling Constant ComputeAfterPaddingsAndStrides error."),
                return ge::GRAPH_FAILED);
    // 空tensor时，合为一根轴
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    if (isPadAllPositive_) {
        DoTilingWithConstant();
    } else if (isPadAllNegative_) {
        tilingKey_ = CONSTANT_SLICE_BRANCH;
        OP_CHECK_IF(DoTilingWithSliceOp() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling with slice op error."),
                    return ge::GRAPH_FAILED);
        context_->SetTilingKey(tilingKey_);
        isUseSlice_ = true;
    } else {
        DoTilingWithSIMT();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadACTiling::DoTiling()
{
    OP_LOGD(context_, "Start PadACTiling DoTiling.");
    OP_CHECK_IF(Init() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling Init error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAttrsInfo() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadACTiling GetShapeAttrsInfo error."),
                return ge::GRAPH_FAILED);
    if (padMode_ == ModeNum::EDGE) {
        OP_CHECK_IF(DoTilingModeEdge() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeEdge failed."),
                    return ge::GRAPH_FAILED);
    } else if (padMode_ == ModeNum::REFLECT || padMode_ == ModeNum::SYMMETRIC) {
        OP_CHECK_IF(DoTilingModeMirror() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeMirror failed."),
                    return ge::GRAPH_FAILED);
    } else if (padMode_ == ModeNum::CIRCULAR) {
        OP_CHECK_IF(DoTilingModeCircular() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeCircular failed."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(DoTilingModeConstant() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeConstant failed."),
                    return ge::GRAPH_FAILED);
    }
    if (isUseSlice_) {
        return ge::GRAPH_SUCCESS;
    }
    FillsAndPrintTilingData();
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORK_SPACE_SIZE;
    OP_LOGD(context_, "Exit PadACTiling DoTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PadV3Tiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "PadV3Tiling running begin");
    const PadV3CompileInfo* compile_info = context->GetCompileInfo<PadV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Tiling4Pad dsl compile_info is Null, running AscendC tiling.");
    PadACTiling tilingObject(context);
    tilingObject.isPadV3_ = true;
    return tilingObject.DoTiling();
}

ge::graphStatus TilingPreparePadv3ForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init PadV3 AscendC Tiling.");
    auto ci = context->GetCompiledInfo<PadV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->core_num <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ub_size <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<PadV3CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4PadV3 Mode success!");
    TilingPreparePadv3ForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the PadV3 op.
IMPL_OP_OPTILING(PadV3).Tiling(PadV3Tiling).TilingParse<PadV3CompileInfo>(TilingPrepare4PadV3);
} // namespace optiling
