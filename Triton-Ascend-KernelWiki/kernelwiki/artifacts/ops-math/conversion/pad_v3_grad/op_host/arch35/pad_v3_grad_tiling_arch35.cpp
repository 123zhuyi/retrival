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
 * \file pad_v3_grad_tiling_arch35.cpp
 * \brief ac pad v3 grad tiling cpp
 */

#include "pad_v3_grad_tiling_arch35.h"
#include "util/platform_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace AscendC;

namespace optiling {

static constexpr uint64_t SYS_WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr size_t PADDINGS_IDX = 1;
static constexpr size_t PAIR = 2;
static constexpr uint8_t MAX_DIM_NUM = 8;
static constexpr uint8_t NON_CONSTANT_MAX_DIM_NUM = 5;
static constexpr uint64_t EXPANSION_FACTOR = 2;
static constexpr uint64_t HALF_FACTOR = 2;

static constexpr uint64_t SIMT_BRANCH_SIZE = 48 * 1024;
static constexpr uint64_t UB_MAX_DATA_SIZE_PER_BUFFER = 64 * 1024;
static constexpr int64_t MIN_PER_UB_SIZE = 4096;    // Bytes
static constexpr double MIN_USED_CORES_RATIO = 0.8; // 80%;

static constexpr uint8_t CONST2 = 2;
static constexpr uint8_t CONST3 = 3;
static constexpr uint8_t CONST4 = 4;
static constexpr uint8_t CONST5 = 5;
static constexpr uint8_t FP32_SIZE = 4;
static constexpr uint8_t PAD_DIM_INDEX_FOURTH = 4;

template <typename T>
std::string PadV3GradACTiling::ToString(const T* value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ",";
    }
    r = r + "]";
    return r;
}

uint64_t PadV3GradACTiling::GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize)
{
    if (alignBlockSize == 0) {
        return 0;
    }
    return (inputSize + alignBlockSize - 1) / alignBlockSize * alignBlockSize;
}

void PadV3GradACTiling::DoFindSplitAxisByInput(bool isBigLastDim)
{
    OP_LOGD(context_, "Start PadV3GradACTiling CalculateTilingKey DoFindSplitAxis.");
    uint64_t dimSizeInUb = dtypeBytes_;
    uint64_t dimSizeInLast4Axis = dimSizeInUb;
    // 找到切分轴
    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        if (isBigLastDim && i == static_cast<int64_t>(dimNum_ - 1)) {
            dimSizeInUb = GetSizeOfBlockAlign(dimSizeInUb * tilingData_->inShape[i], blockSize_);
        } else {
            dimSizeInUb *= tilingData_->inShape[i];
        }
        // 切分超过4根轴时只切最后4根轴，记录下最后4根轴的大小
        if (i == dimNum_ - PAD_DIM_INDEX_FOURTH) {
            dimSizeInLast4Axis = dimSizeInUb;
        }
        if (dimSizeInUb >= bufferSize_ * dtypeBytes_) {
            ubAxis_ = i;
            break;
        }
    }
    // 维度超过4，满载后4个轴
    if (dimNum_ - ubAxis_ > PAD_DIM_INDEX_FOURTH) {
        ubAxis_ = dimNum_ - PAD_DIM_INDEX_FOURTH;
        ubFactor_ = tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        dimSizeInUb = dimSizeInLast4Axis / tilingData_->inShape[dimNum_ - PAD_DIM_INDEX_FOURTH] *
                      tilingData_->outShape[dimNum_ - PAD_DIM_INDEX_FOURTH];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    } else if (dimSizeInUb > bufferSize_ * dtypeBytes_) {
        dimSizeInUb /= tilingData_->inShape[ubAxis_];
        ubFactor_ = dimSizeInUb == 0 ? bufferSize_ : bufferSize_ * dtypeBytes_ / dimSizeInUb;
        if (ubFactor_ > tilingData_->outShape[ubAxis_]) {
            ubFactor_ = tilingData_->outShape[ubAxis_];
        }
        dimSizeInUb *= ubFactor_;
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    } else {
        ubFactor_ = tilingData_->outShape[ubAxis_];
        dimSizeInUb = dimSizeInUb / tilingData_->inShape[ubAxis_] * tilingData_->outShape[ubAxis_];
        outTileSize_ = GetSizeOfBlockAlign(dimSizeInUb, blockSize_);
    }
    outTileSize_ = outTileSize_ / dtypeBytes_;
}

bool PadV3GradACTiling::CheckTilingInfoSatisfied(PadV3GradUbTileInfo& tilingInfo)
{
    tilingInfo.ubTotalCnt = Ops::Base::CeilDiv(tilingData_->outShape[tilingInfo.ubSplitAxis],
                                               static_cast<uint64_t>(tilingInfo.ubSplitFactor));
    for (uint64_t i = 0; i < tilingInfo.ubSplitAxis; i++) {
        tilingInfo.ubTotalCnt *= tilingData_->outShape[i];
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

void PadV3GradACTiling::GetOptimizeTiling(const PadV3GradUbTileInfo& oldTilingInfo, PadV3GradUbTileInfo& newTilingInfo)
{
    int64_t outCount = 1;
    for (uint8_t i = 0; i < oldTilingInfo.ubSplitAxis; i++) {
        outCount *= tilingData_->outShape[i];
    }

    // ubPerCoreCnt 不发生变化为前提时，最多循环coreNum+dimNum_次就可以找到最优解, 这里仅做防死循环保护
    uint32_t maxLoop = coreNum_ + dimNum_;
    uint32_t loops = 0;
    bool finded = false;
    for (uint8_t iDim = oldTilingInfo.ubSplitAxis; iDim < dimNum_; iDim++) {
        if (iDim != oldTilingInfo.ubSplitAxis) {
            outCount *= tilingData_->outShape[iDim - 1];
        }

        int64_t iDimFactor = (iDim == oldTilingInfo.ubSplitAxis) ? oldTilingInfo.ubSplitFactor :
                                                                   tilingData_->outShape[iDim];
        for (int64_t factor = iDimFactor; factor > 0;) {
            loops++;
            if (loops > maxLoop) {
                finded = true;
                OP_LOGD(context_, "loops:%u is bigger than maxLoop:%u", loops, maxLoop);
                break;
            }

            // iDimOuter 每次循环会增加1,最多增加 coreNum_ 后，tmpPerCount会增加1
            int64_t iDimOuter = Ops::Base::CeilDiv(tilingData_->outShape[iDim], static_cast<uint64_t>(factor));
            int64_t tmpTotalCount = iDimOuter * outCount;
            int64_t tmpPerCount = tmpTotalCount > coreNum_ ?
                                      Ops::Base::CeilDiv(tmpTotalCount, static_cast<int64_t>(coreNum_)) :
                                      1;
            int64_t tmpCoreNum = tmpTotalCount > coreNum_ ? Ops::Base::CeilDiv(tmpTotalCount, tmpPerCount) :
                                                            tmpTotalCount;
            int64_t tmpFactor = Ops::Base::CeilDiv(tilingData_->outShape[iDim],
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
        newTilingInfo.ubSplitFactor >= tilingData_->outShape[newTilingInfo.ubSplitAxis]) {
        newTilingInfo.ubSplitAxis = newTilingInfo.ubSplitAxis - 1;
        newTilingInfo.ubSplitFactor = 1;
        OP_LOGD(context_, "Back to last axis, ubSplitAxis:%u ubSplitFactor:%u", newTilingInfo.ubSplitAxis,
                newTilingInfo.ubSplitFactor);
    }
}

void PadV3GradACTiling::TilingInfoTuneForNormal(uint64_t lastShapeSizeAlign)
{
    TilingInfoTune();
    if (ubAxis_ == dimNum_ - 1) {
        cutMode_ = TPL_SIMD_BIG;
        additionTileSize_ = (padMode_ == TPL_MODE_SYMMETRIC || padMode_ == TPL_MODE_REFLECT) ?
                                EXPANSION_FACTOR * vectorSize_ :
                                vectorSize_;
        bufferSize_ = lastShapeSizeAlign;
        outTileSize_ = bufferSize_;
    }
}

void PadV3GradACTiling::TilingInfoTune()
{
    PadV3GradUbTileInfo oldTilingInfo;
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

    PadV3GradUbTileInfo newTilingInfo = oldTilingInfo;
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

void PadV3GradACTiling::CalculateTilingKeyMirror()
{
    OP_LOGD(context_, "Start PadV3GradACTiling CalculateTilingKeyMirror.");
    if (inShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > CONST5) {
        DoTilingWithSIMTMirror();
        return;
    }
    uint64_t alignNum = vectorSize_ / dtypeBytes_;
    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->inShape[dimNum_ - 1], alignNum);

    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / (CONST2 * dtypeBytes_ + CONST4 * FP32_SIZE) - alignNum, alignNum);
    if (ClampBufferSizeAndCheckBigShape(lastShapeSizeAlign)) {
        return;
    }
    // 不切w，但是倒数第二根轴只能切1，此时也走切W分支
    // 不切w, 但是只有一根轴 & w > 128B，也走切w分支
    if (lastShapeSizeAlign * EXPANSION_FACTOR > bufferSize_ ||
        (tilingData_->inShape[dimNum_ - 1] * dtypeBytes_ > vectorSize_ / HALF_FACTOR && dimNum_ == 1)) {
        cutMode_ = TPL_SIMD_BIG;
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = tilingData_->inShape[dimNum_ - 1];
        outTileSize_ = bufferSize_;
        return TilingInfoTune();
    }
    if (tilingData_->inShape[dimNum_ - 1] > vectorSize_ / sizeof(ge::float32_t)) {
        additionTileSize_ = GetSizeOfBlockAlign(tilingData_->inShape[dimNum_ - 1] * dtypeBytes_, blockSize_);
        if (dtypeBytes_ == 1) {
            additionTileSize_ = GetSizeOfBlockAlign(tilingData_->inShape[dimNum_ - 1] * dtypeBytes_ * EXPANSION_FACTOR,
                                                    blockSize_);
        }
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - additionTileSize_) / (CONST4 * dtypeBytes_ + FP32_SIZE) - alignNum,
                                          alignNum);
        DoFindSplitAxisByInput(true);
        cutMode_ = TPL_SIMD_NORMAL;

        if (lastShapeSizeAlign * EXPANSION_FACTOR > outTileSize_) {
            cutMode_ = TPL_SIMD_BIG;
            bufferSize_ = GetSizeOfBlockAlign(ubSize_ / (CONST2 * dtypeBytes_ + CONST4 * FP32_SIZE) - alignNum,
                                              alignNum);
            ubAxis_ = static_cast<uint8_t>(static_cast<int8_t>(dimNum_) - 1);
            ubFactor_ = tilingData_->inShape[dimNum_ - 1];
            outTileSize_ = bufferSize_;
            return TilingInfoTune();
        } else {
            return TilingInfoTuneForNormal(lastShapeSizeAlign);
        }
    } else {
        bufferSize_ = GetSizeOfBlockAlign((ubSize_ - vectorSize_ * CONST3 * CONST2 - blockSize_ * CONST2) /
                                                  (CONST2 * dtypeBytes_ + CONST4 * FP32_SIZE) -
                                              alignNum,
                                          alignNum);
        DoFindSplitAxisByInput(false);
        cutMode_ = TPL_SIMD_SMALL;

        additionTileSize_ = vectorSize_;
        TilingInfoTuneForNormal(lastShapeSizeAlign);
    }
}

bool PadV3GradACTiling::ClampBufferSizeAndCheckBigShape(uint64_t lastShapeSizeAlign)
{
    if (bufferSize_ > UB_MAX_DATA_SIZE_PER_BUFFER / dtypeBytes_) {
        bufferSize_ = UB_MAX_DATA_SIZE_PER_BUFFER / dtypeBytes_;
    }
    if (lastShapeSizeAlign > bufferSize_) {
        cutMode_ = TPL_SIMD_BIG;
        ubAxis_ = dimNum_ - 1;
        ubFactor_ = bufferSize_;
        outTileSize_ = bufferSize_;
        TilingInfoTune();
        return true;
    }
    return false;
}

void PadV3GradACTiling::CalculateTilingKeyCircular()
{
    OP_LOGD(context_, "Start PadV3GradACTiling CalculateTilingKeyCircular.");
    if (inShapeSize_ <= SIMT_BRANCH_SIZE || dimNum_ > CONST5) {
        DoTilingWithSIMTCircular();
        return;
    }

    uint64_t alignNum = blockSize_ / dtypeBytes_;
    uint64_t lastShapeSizeAlign = GetSizeOfBlockAlign(tilingData_->outShape[dimNum_ - 1], alignNum);

    bufferSize_ = GetSizeOfBlockAlign(ubSize_ / (CONST2 * dtypeBytes_ + CONST2 * FP32_SIZE) - alignNum, alignNum);
    if (ClampBufferSizeAndCheckBigShape(lastShapeSizeAlign)) {
        return;
    }
    DoTilingWithSIMTCircular();
}

void PadV3GradACTiling::DoTilingWithSIMTMirror()
{
    isBigShape_ = false;
    isSimt_ = true;
    if (inShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX || outShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX) {
        isBigShape_ = true;
    }
}
void PadV3GradACTiling::DoTilingWithSIMDMirror()
{
    isSimt_ = false;
    CalculateTilingKeyMirror();
}
void PadV3GradACTiling::DoTilingWithSIMTEdge()
{
    isBigShape_ = false;
    isSimt_ = true;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        isBigShape_ = true;
    }
}
void PadV3GradACTiling::DoTilingWithSIMTCircular()
{
    isBigShape_ = false;
    isSimt_ = true;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        isBigShape_ = true;
    }
}
void PadV3GradACTiling::DoTilingWithSIMDCircular()
{
    isSimt_ = false;
    CalculateTilingKeyCircular();
}
void PadV3GradACTiling::DoTilingWithSIMTConstant()
{
    isBigShape_ = false;
    isSimt_ = true;
    if (inShapeSize_ > INT32_MAX || outShapeSize_ > INT32_MAX) {
        isBigShape_ = true;
    }
}

void PadV3GradACTiling::FillsAndPrintTilingData()
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
        "tilingData is dimNum = %u, inShape: %s, outShape: %s, inStride: %s, outStride: %s, leftPad: %s, rightPad: %s, "
        "ubAxis_: %u, ubFactor_: %u, ubPerCount_: %u, ubTotalCount_: %u, outTileSize_: %u, additionTileSize_: %u, "
        "tilingKey: %lu, coreNum is %u dtype:%s",
        tilingData_->dimNum, ToString(tilingData_->inShape, dimNum_).c_str(),
        ToString(tilingData_->outShape, dimNum_).c_str(), ToString(tilingData_->inStride, dimNum_).c_str(),
        ToString(tilingData_->outStride, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
        ToString(tilingData_->rightPad, dimNum_).c_str(), ubAxis_, ubFactor_, ubPerCount_, ubTotalCount_, outTileSize_,
        additionTileSize_, tilingKey_, coreNum_, Ops::Base::ToString(paramsDtype_).c_str());
}

void PadV3GradACTiling::EmptyTensorCollapse()
{
    OP_LOGD(context_, "Start PadV3GradACTiling EmptyTensorCollapse.");
    dimNum_ = 1;
    tilingData_->inShape[0] = 0;
    tilingData_->outShape[0] = outShapeSize_;
    tilingData_->inStride[0] = 1;
    tilingData_->outStride[0] = 1;
    tilingData_->leftPad[0] = 0;
    tilingData_->rightPad[0] = 0;
}

ge::graphStatus PadV3GradACTiling::ComputeAfterPaddingsAndStrides()
{
    inShapeSize_ = 1UL;
    outShapeSize_ = 1UL;

    for (int64_t i = dimNum_ - 1; i >= 0; --i) {
        // 校验是否会出现pad为负且比inshape大的情况

        if (static_cast<int64_t>(tilingData_->inShape[i]) - tilingData_->leftPad[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) - tilingData_->rightPad[i] < 0 ||
            static_cast<int64_t>(tilingData_->inShape[i]) - tilingData_->leftPad[i] - tilingData_->rightPad[i] < 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y",
                                                     std::to_string(static_cast<int64_t>(tilingData_->inShape[i]) -
                                                                    tilingData_->leftPad[i] - tilingData_->rightPad[i])
                                                         .c_str(),
                                                     "The shape dim of y must be >= 0");
            return ge::GRAPH_FAILED;
        }

        // compute shape after padding
        tilingData_->outShape[i] = tilingData_->inShape[i] - tilingData_->leftPad[i] - tilingData_->rightPad[i];
        // compute src stride
        tilingData_->inStride[i] = inShapeSize_;
        inShapeSize_ *= tilingData_->inShape[i];
        // compute dst stride
        tilingData_->outStride[i] = outShapeSize_;
        outShapeSize_ *= tilingData_->outShape[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::CheckModeInputParam(int64_t inShapeV, int64_t padFront, int64_t padBack)
{
    int64_t outShapeV = inShapeV - padFront - padBack;
    int64_t leftsubin = padFront - outShapeV;
    int64_t rightsubin = padBack - outShapeV;
    if (padMode_ == TPL_MODE_REFLECT && (0 < (leftsubin + 1) || 0 < (rightsubin + 1))) {
        std::string incorrectValues = std::to_string(padFront) + " and " + std::to_string(padBack);
        std::string
            reasonStr = "The value of padFront and padBack must be less than InferredOutShape - 1 for reflect mode, "
                        "InferredOutShape: " +
                        std::to_string(outShapeV);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "padFront and padBack", incorrectValues.c_str(),
                                               reasonStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (padMode_ == TPL_MODE_SYMMETRIC && (0 < leftsubin || 0 < rightsubin)) {
        std::string incorrectValues = std::to_string(padFront) + " and " + std::to_string(padBack);
        std::string reasonStr = "The value of padFront and padBack must be no greater than InferredOutShape for "
                                "symmetric mode, "
                                "InferredOutShape: " +
                                std::to_string(outShapeV);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "padFront and padBack", incorrectValues.c_str(),
                                               reasonStr.c_str());
        return ge::GRAPH_FAILED;
    }

    if (padMode_ == TPL_MODE_CIRCULAR && (0 < leftsubin || 0 < rightsubin)) {
        std::string incorrectValues = std::to_string(padFront) + " and " + std::to_string(padBack);
        std::string
            reasonStr = "The value of padFront and padBack must be no greater than InferredOutShape for circular mode, "
                        "InferredOutShape: " +
                        std::to_string(outShapeV);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "padFront and padBack", incorrectValues.c_str(),
                                               reasonStr.c_str());
        return ge::GRAPH_FAILED;
    }

    if (padMode_ == TPL_MODE_EDGE && outShapeV == 0 && (padFront != 0 || padBack != 0)) {
        std::string incorrectValues = std::to_string(padFront) + " and " + std::to_string(padBack);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "padFront and padBack", incorrectValues.c_str(),
            "The value of padFront and padBack must be 0 when InferredOutShape is 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DimensionCollapseMode()
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
        tilingData_->rightPad[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    // shape全1且不补pad的场景,消除1轴后就为空了
    if (dimNum_ == 0) {
        tilingData_->inShape[0] = 1;
        tilingData_->leftPad[0] = 0;
        tilingData_->rightPad[0] = 0;
        dimNum_ = 1;
    }
    OP_LOGD(context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
            ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
            ToString(tilingData_->rightPad, dimNum_).c_str());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DimensionCollapse()
{
    uint16_t fastDim = 0;
    uint16_t slowDim = 0;
    uint8_t originalRank = dimNum_;
    OP_LOGD(context_, "Before collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s",
            originalRank, ToString(tilingData_->inShape, originalRank).c_str(),
            Ops::Base::ToString(paddings_.padFront).c_str(), Ops::Base::ToString(paddings_.padBack).c_str());
    while (fastDim < originalRank) {
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
        tilingData_->rightPad[slowDim] = collapsedPadBack;
        ++slowDim;
    }
    OP_LOGD(context_, "After collapse paddings, dimNum is %u, shape is %s, left pad is %s, right pad is %s", dimNum_,
            ToString(tilingData_->inShape, dimNum_).c_str(), ToString(tilingData_->leftPad, dimNum_).c_str(),
            ToString(tilingData_->rightPad, dimNum_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::DoTilingModeEdge()
{
    OP_CHECK_IF(DimensionCollapseMode() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Edge Collapse error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Edge ComputeAfterPaddingsAndStrides error."),
                return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    DoTilingWithSIMTEdge();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DoTilingModeMirror()
{
    OP_CHECK_IF(DimensionCollapseMode() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Mirror Collapse error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Mirror ComputeAfterPaddingsAndStrides error."),
                return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
        DoTilingWithSIMTMirror();
    } else if (isPadAllPositive_) {
        // simd
        if (inShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX || outShapeSize_ * EXPANSION_FACTOR + 1 > INT32_MAX) {
            // simd normal 目前走simt实现，需要isBigShape_
            isBigShape_ = true;
        }
        DoTilingWithSIMDMirror();
    } else if (isPadAllNegative_) {
        DoTilingWithSIMTMirror();
    } else {
        DoTilingWithSIMTMirror();
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DoTilingModeCircular()
{
    OP_CHECK_IF(DimensionCollapseMode() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Circular Collapse error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Circular ComputeAfterPaddingsAndStrides error."),
                return ge::GRAPH_FAILED);
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
        DoTilingWithSIMTCircular();
    } else if (isPadAllPositive_) {
        // simd
        DoTilingWithSIMDCircular();
    } else if (isPadAllNegative_) {
        DoTilingWithSIMTCircular();
    } else {
        DoTilingWithSIMTCircular();
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PadV3GradACTiling::DoTilingModeConstant()
{
    OP_CHECK_IF(DimensionCollapse() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Constant Collapse error."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAfterPaddingsAndStrides() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling Constant ComputeAfterPaddingsAndStrides error."),
                return ge::GRAPH_FAILED);
    // 空tensor时，合为一根轴
    if (isEmptyTensor_) {
        EmptyTensorCollapse();
    }
    DoTilingWithSIMTConstant();
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void PadV3GradACTiling::GetPaddingsToShape(const gert::Tensor* paddingsTensor)
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo GetPaddings GetPaddingsToShape.");
    const T* paddingsValue = paddingsTensor->GetData<T>();
    const int64_t paddingsNumInt64 = paddingsTensor->GetShapeSize();
    if (paddingsNumInt64 < 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "paddingsTensor", std::to_string(paddingsNumInt64).c_str(),
            "The shape size of paddingsTensor must be within the range representable by the uint32_t type");
        return;
    }
    const size_t paddingsNum = static_cast<size_t>(paddingsNumInt64);
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

ge::graphStatus PadV3GradACTiling::GetPaddings()
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo GetPaddings.");
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
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "paddings", Ops::Base::ToString(paddingsDtype).c_str(),
                                      "int32 or int64");
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus PadV3GradACTiling::GetShapesAndDtypes()
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo GetShapesAndDtypes.");
    // get input shape & input shape's dim num
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    dimNum_ = inShapeVal.GetDimNum(); // 获取维度
    if (dimNum_ > MAX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dimNum_).c_str(),
                                                 "The shape dim of x must be <= 8");
        return ge::GRAPH_FAILED;
    }
    if (padMode_ != TPL_MODE_CONSTANT && dimNum_ > NON_CONSTANT_MAX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dimNum_).c_str(),
                                                 "The shape dim of x must be <= 5 for non-constant mode");
        return ge::GRAPH_FAILED;
    }
    for (uint16_t i = 0; i < dimNum_; ++i) {
        if (inShapeVal.GetDim(i) < 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x",
                                                     std::to_string(inShapeVal.GetDim(i)).c_str(),
                                                     "The shape dim of x must be >= 0");
            return ge::GRAPH_FAILED;
        }
        if (inShapeVal.GetDim(i) == 0) {
            isEmptyTensor_ = true;
        }
        tilingData_->inShape[i] = inShapeVal.GetDim(i); // 获取inShape
    }
    // 判断类型
    auto inputTensor = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputTensor);
    paramsDtype_ = inputTensor->GetDataType();
    dtypeBytes_ = GetSizeByDataType(paramsDtype_);
    // dtypeBytes_ = FP32_SIZE; // fp16/bfp16都需要转化为fp32计算
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_, "Start PadV3GradACTiling GetShapeAttrsInfo");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    if (isCircularPadGrad_) {
        padMode_ = TPL_MODE_CIRCULAR;
        paddingContiguous_ = true;
    } else {
        auto* mode = attrs->GetAttrPointer<char>(0);
        // padv3grad mode可选，默认reflect;
        if (mode) {
            if (!strcmp(mode, "constant")) {
                padMode_ = TPL_MODE_CONSTANT;
            } else if (!strcmp(mode, "edge")) {
                padMode_ = TPL_MODE_EDGE;
            } else if (!strcmp(mode, "symmetric")) {
                padMode_ = TPL_MODE_SYMMETRIC;
            } else if (!strcmp(mode, "circular")) {
                padMode_ = TPL_MODE_CIRCULAR;
            }
            OP_CHECK_IF(strcmp(mode, "constant") != 0 && strcmp(mode, "edge") != 0 && strcmp(mode, "reflect") != 0 &&
                            strcmp(mode, "symmetric") != 0 && strcmp(mode, "circular") != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context_->GetNodeName(), "mode", mode,
                            "The value of mode can only be constant, edge, reflect, symmetric, or circular"),
                        return ge::GRAPH_FAILED);
        }
        auto* paddingContiguous = attrs->GetAttrPointer<bool>(1);
        if (paddingContiguous) {
            paddingContiguous_ = *paddingContiguous;
        }
    }

    OP_CHECK_IF(GetShapesAndDtypes() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling GetShapeAttrsInfo GetShapesAndDtypes error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetPaddings() == ge::GRAPH_FAILED, OP_LOGE(context_, "Get padding info failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::Init()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "core_num", "0",
                                                      "The value of core_num must be > 0"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ub_size", "0",
                                                      "The value of ub_size must be > 0"),
                return ge::GRAPH_FAILED);
    blockSize_ = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    OP_CHECK_IF(blockSize_ == 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ub_block_size", "0",
                                                      "The value of ub_block_size must be > 0"),
                return ge::GRAPH_FAILED);
    vectorSize_ = static_cast<uint64_t>(Ops::Base::GetVRegSize(context_));
    OP_CHECK_IF(vectorSize_ == 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "vector_size", "0",
                                                      "The value of vector_size must be > 0"),
                return ge::GRAPH_FAILED);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(cacheLineSize_ == 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cache_line_size", "0",
                                                      "The value of cache_line_size must be > 0"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_,
            "platform info is coreNum_ = %u, ubSize_ = %lu, blockSize_ = %lu, vectorSize_ = %lu, cacheLineSize_ = %u",
            coreNum_, ubSize_, blockSize_, vectorSize_, cacheLineSize_);
    tilingData_ = context_->GetTilingData<PadV3GradACTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PadV3GradACTiling::DoTiling()
{
    OP_LOGD(context_, "Start PadV3GradACTiling DoTiling.");
    OP_CHECK_IF(Init() == ge::GRAPH_FAILED, OP_LOGE(context_, "PadV3GradACTiling Init error."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAttrsInfo() == ge::GRAPH_FAILED,
                OP_LOGE(context_, "PadV3GradACTiling GetShapeAttrsInfo error."), return ge::GRAPH_FAILED);
    if (padMode_ == TPL_MODE_EDGE) {
        OP_CHECK_IF(DoTilingModeEdge() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeEdge failed."),
                    return ge::GRAPH_FAILED);
    } else if (padMode_ == TPL_MODE_REFLECT || padMode_ == TPL_MODE_SYMMETRIC) {
        OP_CHECK_IF(DoTilingModeMirror() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeMirror failed."),
                    return ge::GRAPH_FAILED);
    } else if (padMode_ == TPL_MODE_CIRCULAR) {
        OP_CHECK_IF(DoTilingModeCircular() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeCircular failed."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(DoTilingModeConstant() == ge::GRAPH_FAILED, OP_LOGE(context_, "DoTilingModeConstant failed."),
                    return ge::GRAPH_FAILED);
    }

    tilingKey_ = GET_TPL_TILING_KEY(padMode_, isBigShape_, isSimt_, cutMode_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is %lu, modeName %d, isBigShape %d, isSimt_ %d, cutMode_ %d",
            tilingKey_, padMode_, isBigShape_, isSimt_, cutMode_);

    FillsAndPrintTilingData();
    context_->SetBlockDim(coreNum_);
    context_->SetTilingKey(tilingKey_);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORK_SPACE_SIZE;

    OP_LOGD(context_, "Exit PadV3GradACTiling DoTiling.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PadV3GradTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "PadV3GradTiling running begin");
    PadV3GradACTiling tilingObject(context);
    return tilingObject.DoTiling();
}

ge::graphStatus TilingPreparePadV3GradForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init PadV3Grad AscendC Tiling.");
    auto ci = context->GetCompiledInfo<PadV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->core_num <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "core_num", std::to_string(ci->core_num).c_str(),
                                              "The value of core_num must be > 0"),
        return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(
        (ci->ub_size <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ub_size", std::to_string(ci->ub_size).c_str(),
                                              "The value of ub_size must be > 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3Grad(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<PadV3GradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4PadV3Grad Mode success!");
    TilingPreparePadV3GradForAscendC(context);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the PadV3Grad op.
IMPL_OP_OPTILING(PadV3Grad).Tiling(PadV3GradTiling).TilingParse<PadV3GradCompileInfo>(TilingPrepare4PadV3Grad);
} // namespace optiling
