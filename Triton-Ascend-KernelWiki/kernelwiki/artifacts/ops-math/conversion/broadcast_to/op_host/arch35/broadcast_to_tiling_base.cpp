/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_tiling_base.cpp
 * \brief calc tiling data for broadcastto AscendC kernel
 */

#include "broadcast_to_tiling_arch35.h"
#include "broadcast_to_tiling_base.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_util.h"
#include "util/const_util.h"
#include "util/math_util.h"

namespace optiling {
namespace brcto {
static constexpr int64_t DIM_NUM_THRESHOLD_FOR_R4_SIZE = 5; // 触发尾轴4D尺寸计算的维度数阈值
static constexpr int64_t TRAILING_DIM_NUM_FOR_R4_SIZE = 4;  // 用于4D尺寸计算的尾轴维度数

ge::graphStatus CheckSameDimNum(const gert::TilingContext* context, const gert::Shape& inShape,
                                const gert::Shape& outShape, size_t& dimNum)
{
    dimNum = inShape.GetDimNum();
    if (dimNum != outShape.GetDimNum()) {
        std::string dimMsg = std::to_string(dimNum) + " and " + std::to_string(outShape.GetDimNum());
        std::string reasonMsg = "The input and output shape dim num should be equal.";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x and y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetABFlag(const gert::TilingContext* context, const gert::Shape& inShape, const gert::Shape& outShape,
                          std::array<bool, MAX_DIM_NUM>& abInfo)
{
    size_t inDimNum;
    if (CheckSameDimNum(context, inShape, outShape, inDimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    for (size_t idx = 0; idx < inDimNum; idx++) {
        abInfo[idx] = (inShape[idx] != outShape[idx]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MergeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    size_t dimNum;
    if (CheckSameDimNum(context, inShape, outShape, dimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (dimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }

    std::array<bool, MAX_DIM_NUM> abInfo{};
    if (GetABFlag(context, inShape, outShape, abInfo) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = "unknown";
        std::string reasonMsg = "Failed to get axes info.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    size_t mIdx = 0;
    for (size_t oIdx = 1; oIdx < dimNum; oIdx++) {
        if (abInfo[oIdx] == abInfo[mIdx]) {
            inShape[mIdx] *= inShape[oIdx];
            outShape[mIdx] *= outShape[oIdx];
        } else {
            mIdx += 1;
            inShape[mIdx] = inShape[oIdx];
            outShape[mIdx] = outShape[oIdx];
            abInfo[mIdx] = abInfo[oIdx];
        }
    }
    inShape.SetDimNum(mIdx + 1);
    outShape.SetDimNum(mIdx + 1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeleteOneSizeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    size_t dimNum;
    if (CheckSameDimNum(context, inShape, outShape, dimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (dimNum == 1) {
        return ge::GRAPH_SUCCESS;
    }

    size_t mIdx = 0;
    for (size_t oIdx = 0; oIdx < dimNum; oIdx++) {
        if (outShape[oIdx] != 1) {
            inShape[mIdx] = inShape[oIdx];
            outShape[mIdx] = outShape[oIdx];
            mIdx += size_t(1);
        }
    }

    if (mIdx == size_t(0)) {
        inShape[0] = 1;
        outShape[0] = 1;
        mIdx += size_t(1);
    }
    inShape.SetDimNum(mIdx);
    outShape.SetDimNum(mIdx);

    return ge::GRAPH_SUCCESS;
}

void AdjustShapesToSameDimNum(gert::Shape& inShape, size_t outDimNum)
{
    auto inDimNum = inShape.GetDimNum();
    if (inDimNum >= outDimNum) {
        return;
    }

    gert::Shape newShape;
    size_t gapSize = outDimNum - inDimNum;
    for (size_t i = 0; i < gapSize; i++) {
        newShape.AppendDim(1);
    }
    for (size_t j = 0; j < inDimNum; j++) {
        newShape.AppendDim(inShape[j]);
    }
    inShape = newShape;
}

static ge::graphStatus CheckBroadcastRule(const gert::TilingContext* context, const gert::Shape& inShape,
                                          const gert::Shape& outShape)
{
    size_t dimNum;
    if (CheckSameDimNum(context, inShape, outShape, dimNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < dimNum; i++) {
        if (inShape[i] != 1 && outShape[i] != inShape[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

ge::graphStatus GetShapeInfo(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape)
{
    auto xStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    inShape = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());

    gert::Shape constShape_{};
    constexpr int64_t constIdx = 1;
    if (Ops::Base::GetConstIntToShape(context, constIdx, constShape_) &&
        outShape != Ops::Base::EnsureNotScalar(constShape_)) {
        OP_LOGW(context->GetNodeName(), "Output shape %s is different from input const shape %s!",
                Shape2String(outShape).c_str(), Shape2String(constShape_).c_str());
    }

    auto outDimNum = outShape.GetDimNum();
    if (inShape.GetDimNum() > outDimNum) {
        std::string dimMsg = std::to_string(inShape.GetDimNum()) + " and " + std::to_string(outDimNum);
        std::string reasonMsg = "The input shape should not have more dimensions than output shape.";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x and y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (outDimNum > BRCTO_MAX_DIM_NUM) {
        std::string dimMsg = std::to_string(outDimNum);
        std::string reasonMsg = "The output dim num should not be greater than " + std::to_string(BRCTO_MAX_DIM_NUM) +
                                ".";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "y", dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    AdjustShapesToSameDimNum(inShape, outDimNum);
    if (inShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "The input or output shape must not be empty.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (CheckBroadcastRule(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "The input and output shapes mismatch the broadcast rule.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "The input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    if (DeleteOneSizeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "Failed to delete one size axes.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "The reshaped input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    if (MergeAxis(context, inShape, outShape) != ge::GRAPH_SUCCESS) {
        std::string shapeMsg = Shape2String(inShape) + " and " + Shape2String(outShape);
        std::string reasonMsg = "Failed to merge axes.";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and y", shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "The merged input and output is: %s and %s", Shape2String(inShape).c_str(),
            Shape2String(outShape).c_str());

    return ge::GRAPH_SUCCESS;
}

void BroadcastToTilingAscendC::GetUAxisInfo()
{
    auto dimNum = outShapePtr_->GetDimNum();
    auto res = outShapePtr_->GetDim(dimNum - 1);

    isLastDimB_ = int32_t(abInfo_[dimNum - 1]);

    if (res >= tensorSize_) {
        uAxis_ = dimNum - 1;
        uLpUnit_ = tensorSize_;
        uAxisLen_ = res;
        uInOffset_ = 1;
        uOutOffset_ = 1;
        isUNotB_ = int32_t(!abInfo_[dimNum - 1]);
        uAxisCnt_ = 1;
        return;
    }

    // to avoid ub overflow
    if (tilingKey_ == TILING_MODE_UB_BRC) {
        res = Ops::Base::CeilAlign(res, blockSize_ / dtypeSize_);
    }

    size_t maxUbAxisCnt = (tilingKey_ != TILING_MODE_NDDMA) ? brctoMaxDMADimNum - 1 : brctoMaxDMADimNum;
    if ((CalcDimSize(outShapePtr_, 0, dimNum - 1) * res <= tensorSize_ && dimNum <= maxUbAxisCnt) || dimNum == 1) {
        uAxis_ = size_t(0);
        uLpUnit_ = outShapePtr_->GetDim(0);
        uAxisLen_ = uLpUnit_;
        uInOffset_ = CalcDimSize(inShapePtr_, 1, dimNum);
        uOutOffset_ = CalcDimSize(outShapePtr_, 1, dimNum);
        isUNotB_ = int32_t(!abInfo_[0]);
        uAxisCnt_ = int8_t(dimNum);
        return;
    }

    int64_t curDim = 0;
    size_t curIdx = 1;
    for (int8_t i = int8_t(dimNum - nTwo); i >= 0; i--) {
        curIdx++;
        curDim = outShapePtr_->GetDim(size_t(i));
        res *= curDim;
        if (res >= tensorSize_ || curIdx >= maxUbAxisCnt) {
            OP_CHECK_IF(curDim == 0 || Ops::Base::FloorDiv(res, curDim) == 0,
                        OP_LOGE(context_->GetNodeName(), "Division by zero in nested division, curDim=%ld, res=%ld",
                                curDim, res),
                        return);
            uAxis_ = size_t(i);
            uLpUnit_ = (tensorSize_ / (res / curDim) > curDim) ? curDim : tensorSize_ / (res / curDim);
            uAxisLen_ = curDim;
            uInOffset_ = CalcDimSize(inShapePtr_, i + 1, dimNum);
            uOutOffset_ = CalcDimSize(outShapePtr_, i + 1, dimNum);
            isUNotB_ = int32_t(!abInfo_[i]);
            uAxisCnt_ = int8_t(dimNum - i);
            return;
        }
    }
}

void BroadcastToTilingAscendC::UpdateTilingKey()
{
    if (uAxisCnt_ == int8_t(brctoMaxDMADimNum)) {
        tilingKey_ = TILING_MODE_FULL_NDDMA;
        return;
    }
    if (tilingKey_ == TILING_MODE_UB_BRC && isLastDimB_ == 0 && uAxisCnt_ > 1) {
        auto dimNum = outShapePtr_->GetDimNum();
        auto lastDimAlign = Ops::Base::CeilAlign(outShapePtr_->GetDim(dimNum - 1) * dtypeSize_, blockSize_);
        if (outShapePtr_->GetDim(dimNum - nTwo) * lastDimAlign <= vlSize_) {
            tilingKey_ = TILING_MODE_LAST_DIM_SMALL_A;
            return;
        }
    }
}

int64_t BroadcastToTilingAscendC::CalcDimSize(const gert::Shape*& shapePtr, size_t begDim, size_t endDim)
{
    int64_t res = 1;
    for (size_t i = begDim; i < endDim; i++) {
        res *= shapePtr->GetDim(i);
    }
    return res;
}

void BroadcastToTilingAscendC::GetDMAAxesParams()
{
    auto dimNum = outShapePtr_->GetDimNum();
    for (size_t i = uAxis_; i < dimNum; i++) {
        if (abInfo_[i]) { // axes B
            xSrcStride_[i - uAxis_] = 0;
        } else { // axes A
            xSrcStride_[i - uAxis_] = CalcDimSize(inShapePtr_, i + 1, dimNum);
        }
        xDstStride_[i - uAxis_] = CalcDimSize(outShapePtr_, i + 1, dimNum);
        if (i == uAxis_) {
            xSize_[i - uAxis_] = uLpUnit_;
        } else {
            xSize_[i - uAxis_] = outShapePtr_->GetDim(i);
        }
    }
}

void BroadcastToTilingAscendC::GetABAxesParams()
{
    if (uAxis_ == size_t(0)) {
        return;
    }

    auto dimNum = inShapePtr_->GetDimNum();
    int64_t tmpDimSize = 0;
    for (size_t i = 0; i < uAxis_; i++) {
        tmpDimSize = outShapePtr_->GetDim(i);
        if (abInfo_[i]) { // axes B
            bAxisLen_ *= tmpDimSize;
            bAxesParams_[bAxesNum_ * bParamUnit] = tmpDimSize;
            bAxesParams_[bAxesNum_ * bParamUnit + 1] = CalcDimSize(outShapePtr_, i + 1, dimNum);
            bAxesNum_++;
        } else { // axes A
            aAxisLen_ *= tmpDimSize;
            aAxesParams_[aAxesNum_ * aParamUnit] = tmpDimSize;
            aAxesParams_[aAxesNum_ * aParamUnit + 1] = CalcDimSize(inShapePtr_, i + 1, dimNum);
            aAxesParams_[aAxesNum_ * aParamUnit + nTwo] = CalcDimSize(outShapePtr_, i + 1, dimNum);
            aAxesNum_++;
        }
    }
}

void BroadcastToTilingAscendC::GetAxesInfo()
{
    GetUAxisInfo();
    GetABAxesParams();
    GetDMAAxesParams();
}

uint32_t BroadcastToTilingAscendC::CalcAxisWeight(int64_t lpCnt)
{
    uint32_t weight = 0;
    weight = Ops::Base::CeilDiv(lpCnt, Ops::Base::CeilDiv(lpCnt, coreNum_));
    if (lpCnt % coreNum_ == 0) {
        weight += uint32_t(coreNum_);
    }
    return weight;
}

void BroadcastToTilingAscendC::OptimizeMCTilingForDoubleMode()
{
    // 优化：当U轴切分导致核数不足时，改为A轴切分，以便利用doubleMode在U轴二次切分
    // 条件：1.核数不足 2.当前是U轴切分 3.A轴可切分 4.U轴有二次切分空间
    if (usedCoreCnt_ < coreNum_ / nTwo && blockAxis_ == nTwo && aAxisLen_ > 1 && uAxisLen_ > uLpUnit_) {
        OP_LOGD("BroadcastTo", "Enter BroadcastToTilingAscendC::GetMCTilingInfo");
        blockAxis_ = 0;
        usedCoreCnt_ = Ops::Base::CeilDiv(aAxisLen_, Ops::Base::CeilDiv(aAxisLen_, coreNum_));
        ntcALen_ = Ops::Base::CeilDiv(aAxisLen_, usedCoreCnt_);
        tcALen_ = aAxisLen_ - ntcALen_ * (usedCoreCnt_ - 1);
        ntcBLen_ = bAxisLen_;
        tcBLen_ = bAxisLen_;
        ntcULen_ = uAxisLen_;
        tcULen_ = uAxisLen_;
    }
}

void BroadcastToTilingAscendC::GetMCTilingInfo()
{
    int64_t aLpCnt = aAxisLen_;
    int64_t bLpCnt = bAxisLen_;
    int64_t uLpCnt = Ops::Base::CeilDiv(uAxisLen_, uLpUnit_);
    uint32_t aAxisWeight = CalcAxisWeight(aLpCnt);
    uint32_t bAxisWeight = CalcAxisWeight(bLpCnt);
    uint32_t uAxisWeight = CalcAxisWeight(uLpCnt);
    uint32_t maxWeight = std::max(aAxisWeight, std::max(bAxisWeight, uAxisWeight));
    if (maxWeight == aAxisWeight) {
        blockAxis_ = 0;
        usedCoreCnt_ = Ops::Base::CeilDiv(aLpCnt, Ops::Base::CeilDiv(aLpCnt, coreNum_));
        ntcALen_ = Ops::Base::CeilDiv(aLpCnt, usedCoreCnt_);
        tcALen_ = aAxisLen_ - ntcALen_ * (usedCoreCnt_ - 1);
        ntcBLen_ = bAxisLen_;
        tcBLen_ = bAxisLen_;
        ntcULen_ = uAxisLen_;
        tcULen_ = uAxisLen_;
    } else if (maxWeight == uAxisWeight) {
        blockAxis_ = nTwo;
        usedCoreCnt_ = Ops::Base::CeilDiv(uLpCnt, Ops::Base::CeilDiv(uLpCnt, coreNum_));
        ntcULen_ = Ops::Base::CeilDiv(uLpCnt, usedCoreCnt_) * uLpUnit_;
        tcULen_ = uAxisLen_ - ntcULen_ * (usedCoreCnt_ - 1);
        ntcALen_ = aAxisLen_;
        tcALen_ = aAxisLen_;
        ntcBLen_ = bAxisLen_;
        tcBLen_ = bAxisLen_;
    } else {
        blockAxis_ = 1;
        usedCoreCnt_ = Ops::Base::CeilDiv(bLpCnt, Ops::Base::CeilDiv(bLpCnt, coreNum_));
        ntcBLen_ = Ops::Base::CeilDiv(bLpCnt, usedCoreCnt_);
        tcBLen_ = bAxisLen_ - ntcBLen_ * (usedCoreCnt_ - 1);
        ntcALen_ = aAxisLen_;
        tcALen_ = aAxisLen_;
        ntcULen_ = uAxisLen_;
        tcULen_ = uAxisLen_;
    }
    OptimizeMCTilingForDoubleMode();
    aLpUnit_ = ntcALen_ > tensorSize_ ? tensorSize_ : ntcALen_;
}

void BroadcastToTilingAscendC::CalcDBMode()
{
    if (tilingKey_ == TILING_MODE_UB_BRC && (ntcALen_ == 1 && (isUNotB_ == 0 || uLpUnit_ >= ntcULen_))) {
        bufferCnt_ = 1;
        return;
    }
    if (tilingKey_ == TILING_MODE_LAST_DIM_LARGE_B && aAxesNum_ == 0) {
        bufferCnt_ = 1;
    }
}

int64_t BroadcastToTilingAscendC::UpdateTensorSize(int64_t tensorSize)
{
    int64_t tmpTensorSize = tensorSize;
    while (tmpTensorSize > maxTensorSize_) {
        tmpTensorSize /= nTwo;
    }
    return tmpTensorSize;
}

void BroadcastToTilingAscendC::UpdateDimSize(int64_t& aDims, int64_t& bDims, int64_t& brwAxis, int64_t& outLastDim)
{
    while (outLastDim < minTensorSize_ && brwAxis > 0) {
        brwAxis--;
        aDims /= inShapePtr_->GetDim(brwAxis);
        if (abInfo_[brwAxis]) {
            bDims /= outShapePtr_->GetDim(brwAxis);
        }
        outLastDim *= outShapePtr_->GetDim(brwAxis);
    }
}

void BroadcastToTilingAscendC::CheckBrwd(int64_t& aDims, int64_t& bDims, int64_t& brwAxis, bool& isBrwd)
{
    while ((aDims >= coreNum_ || bDims >= coreNum_) && brwAxis > 0) {
        brwAxis--;
        aDims /= inShapePtr_->GetDim(brwAxis);
        if (abInfo_[brwAxis]) {
            bDims /= outShapePtr_->GetDim(brwAxis);
        }
        isBrwd = true;
    }
}

void BroadcastToTilingAscendC::AdjustBrwdSize(int64_t& brwSize, int64_t uAxis)
{
    int64_t brwGate = 1;
    int64_t brwLeftSize = Ops::Base::CeilDiv(uAxis, brwSize);
    if (brwSize <= brwGate || brwLeftSize % coreNum_ == 0 || uAxis % coreNum_ != 0) {
        return;
    }

    int64_t coreCnt = Ops::Base::CeilDiv(brwLeftSize, Ops::Base::CeilDiv(brwLeftSize, coreNum_));
    int64_t uLpCnt = Ops::Base::CeilDiv(brwLeftSize, coreCnt);
    int64_t totalULpCnt = uLpCnt * coreNum_;
    OP_CHECK_IF(
        totalULpCnt == 0,
        OP_LOGE(context_->GetNodeName(), "totalULpCnt must not be zero, got totalULpCnt=%ld, uLpCnt=%ld, coreNum=%ld",
                totalULpCnt, uLpCnt, coreNum_),
        return);
    if (uAxis % totalULpCnt == 0) {
        brwSize = uAxis / totalULpCnt;
    }
}

int64_t BroadcastToTilingAscendC::CalcTensorSize4Brwd(int64_t aDims, int64_t bDims, int64_t brwAxis)
{
    int64_t tmpTensorSize = 0;
    auto dimNum = outShapePtr_->GetDimNum();
    auto lastDimSize = outShapePtr_->GetDim(dimNum - 1);
    int64_t rDimSize = CalcDimSize(outShapePtr_, brwAxis + 1, dimNum);
    if (rDimSize >= maxTensorSize_) {
        tmpTensorSize = maxTensorSize_;
    } else if (!abInfo_[dimNum - 1] && lastDimSize > ubSize_ / dtypeSize_ / nTwo / nTwo) {
        tmpTensorSize = lastDimSize;
    } else {
        int64_t brwSize = 1;
        auto uAxis = outShapePtr_->GetDim(brwAxis);
        if (abInfo_[brwAxis]) {
            brwSize = Ops::Base::CeilDiv(bDims * uAxis, coreNum_);
        } else {
            brwSize = Ops::Base::CeilDiv(aDims * uAxis, coreNum_);
        }
        brwSize = std::min(maxTensorSize_ / rDimSize, brwSize);
        AdjustBrwdSize(brwSize, uAxis);
        tmpTensorSize = Ops::Base::CeilAlign(brwSize * rDimSize, minTensorSize_);
    }
    return tmpTensorSize;
}

int64_t BroadcastToTilingAscendC::CalcTensorSize4NBrwd(int64_t aDims, int64_t bDims, int64_t outLastDim)
{
    int64_t tmpTensorSize = 0;
    int64_t dimSize = (outLastDim < maxTensorSize_) ? outShapePtr_->GetShapeSize() : outLastDim;
    int64_t coreGate = int64_t(coreFactor * coreNum_);
    if (aDims >= coreGate || bDims >= coreGate ||
        (aDims >= bDims && aDims * bDims >= coreGate && aDims <= coreNum_ / nTwo)) {
        tmpTensorSize = std::min(outLastDim, maxTensorSize_);
    } else {
        if ((aDims * nTwo <= coreNum_ && bDims * nTwo <= coreNum_)) {
            dimSize = outShapePtr_->GetShapeSize();
        }
        tmpTensorSize = Ops::Base::CeilDiv(Ops::Base::CeilDiv(dimSize, minTensorSize_), coreNum_) * minTensorSize_;
    }
    return tmpTensorSize;
}

void BroadcastToTilingAscendC::CalcTensorSize()
{
    maxTensorSize_ = ubSize_ / blockSize_ * blockSize_ / dtypeSize_;
    minTensorSize_ = cacheLine_ / dtypeSize_;
    int64_t tmpTensorSize = maxTensorSize_;
    auto dimNum = outShapePtr_->GetDimNum();
    auto inLastDim = inShapePtr_->GetDim(dimNum - 1);
    auto outLastDim = outShapePtr_->GetDim(dimNum - 1);
    auto aDims = inShapePtr_->GetShapeSize() / inLastDim;
    OP_CHECK_IF(aDims == 0, OP_LOGE(context_->GetNodeName(), "aDims must not be zero, got %ld", aDims), return);
    auto bDims = outShapePtr_->GetShapeSize() / aDims / outLastDim;
    bool isBrwd = false;

    int64_t ubGate = maxTensorSize_ / nTwo / nTwo;
    int64_t r4DimSize = minTensorSize_;
    if (dimNum > DIM_NUM_THRESHOLD_FOR_R4_SIZE) {
        r4DimSize = CalcDimSize(inShapePtr_, dimNum - TRAILING_DIM_NUM_FOR_R4_SIZE, dimNum);
    }

    isDMABrcA_ = (dimNum > 1 &&
                  (nTwo * outLastDim <= LAST_DIM_GATE ||
                   (outLastDim == LAST_DIM_GATE / nTwo + 1 && outShapePtr_->GetDim(dimNum - nTwo) <= LAST_DIM_GATE) ||
                   ((outLastDim < LAST_DIM_GATE) && (r4DimSize < minTensorSize_))));
    if ((!abInfo_[dimNum - 1] && outLastDim <= ubGate && !isDMABrcA_) ||
        (abInfo_[dimNum - 1] && outLastDim >= LAST_DIM_GATE)) { // UB broadcast
        tmpTensorSize = std::min(ubGate, MAX_TENSOR_SIZE);
    } else if (!abInfo_[dimNum - 1] && outLastDim >= ubGate) { // data copy
        tmpTensorSize = std::min(ubGate * nTwo, MAX_TENSOR_SIZE);
    }
    maxTensorSize_ = tmpTensorSize;

    if (dimNum == 1) {
        tmpTensorSize = Ops::Base::CeilDiv(Ops::Base::CeilDiv(outLastDim, minTensorSize_), coreNum_) * minTensorSize_;
        tensorSize_ = UpdateTensorSize(tmpTensorSize);
        return;
    }

    int64_t brwAxis = int64_t(dimNum) - 1;
    UpdateDimSize(aDims, bDims, brwAxis, outLastDim);
    if (outLastDim >= minTensorSize_) {
        CheckBrwd(aDims, bDims, brwAxis, isBrwd);
        if (isBrwd) {
            tmpTensorSize = CalcTensorSize4Brwd(aDims, bDims, brwAxis);
        } else {
            tmpTensorSize = CalcTensorSize4NBrwd(aDims, bDims, outLastDim);
        }
        tensorSize_ = UpdateTensorSize(tmpTensorSize);
        return;
    }
    tensorSize_ = minTensorSize_;
}

void BroadcastToTilingAscendC::CalcTilingKey()
{
    auto dimNum = outShapePtr_->GetDimNum();
    int64_t lastDim = outShapePtr_->GetDim(dimNum - 1);
    if (lastDim >= tensorSize_ && !abInfo_[dimNum - 1]) {
        tilingKey_ = TILING_MODE_LAST_DIM_LARGE_A;
        return;
    }
    bool ubSizeCheck = tensorSize_ <= ubSize_ / blockSize_ * blockSize_ / dtypeSize_ / nTwo / nTwo;
    if ((lastDim >= tensorSize_ && abInfo_[dimNum - 1]) && ubSizeCheck) {
        tilingKey_ = TILING_MODE_LAST_DIM_LARGE_B;
        return;
    }
    if (((dimNum == 1 && !abInfo_[dimNum - 1]) ||
         (lastDim <= tensorSize_ &&
          ((!abInfo_[dimNum - 1] && !isDMABrcA_) || (abInfo_[dimNum - 1] && lastDim >= LAST_DIM_GATE)))) &&
        ubSizeCheck) {
        tilingKey_ = TILING_MODE_UB_BRC;
        return;
    }
    tilingKey_ = TILING_MODE_NDDMA;
}

void BroadcastToTilingAscendC::CalcTilingData()
{
    GetABFlag(context_, *inShapePtr_, *outShapePtr_, abInfo_);
    CalcTensorSize();
    CalcTilingKey();
    GetAxesInfo();
    UpdateTilingKey();
    GetMCTilingInfo();
    CalcDBMode();
}

ge::graphStatus BroadcastToTilingAscendC::SetBlockCnt()
{
    OP_CHECK_IF(usedCoreCnt_ == 0,
                OP_LOGE(context_->GetNodeName(), "usedCoreCnt must not be zero, got %ld", usedCoreCnt_),
                return ge::GRAPH_FAILED);
    int64_t maxMC = coreNum_ / usedCoreCnt_;
    int64_t uLpCnt = Ops::Base::CeilDiv(ntcULen_, uLpUnit_);
    if (usedCoreCnt_ <= coreNum_ / nTwo &&
        (blockAxis_ != nTwo && ntcULen_ > uLpUnit_ && (uLpCnt >= ntcBLen_ || blockAxis_ == 1))) {
        doubleMode_ = nTwo;
        dFactor_ = Ops::Base::CeilDiv(uLpCnt, Ops::Base::CeilDiv(uLpCnt, std::min(maxMC, uLpCnt)));
    } else if (usedCoreCnt_ <= coreNum_ / nTwo && (blockAxis_ != 1 && ntcBLen_ > 1)) {
        doubleMode_ = 1;
        dFactor_ = Ops::Base::CeilDiv(ntcBLen_, Ops::Base::CeilDiv(ntcBLen_, std::min(maxMC, ntcBLen_)));
    }
    return context_->SetBlockDim(usedCoreCnt_ * dFactor_);
}

ge::graphStatus BroadcastToTilingAscendC::WriteTilingData()
{
    if (SetBlockCnt() != ge::GRAPH_SUCCESS) {
        std::string valueMsg = std::to_string(usedCoreCnt_ * dFactor_);
        std::string reasonMsg = "Failed to set block dim.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "blockDim", valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (context_->SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) {
        std::string valueMsg = std::to_string(tilingKey_);
        std::string reasonMsg = "Failed to set tiling key.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "tilingKey", valueMsg.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_doubleMode(static_cast<uint8_t>(doubleMode_));
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tensorSize(static_cast<uint32_t>(tensorSize_));
    tilingData_.set_usedCoreCnt(usedCoreCnt_);
    tilingData_.set_blockAxis(static_cast<uint8_t>(blockAxis_));
    tilingData_.set_ntcALen(ntcALen_);
    tilingData_.set_tcALen(tcALen_);
    tilingData_.set_ntcBLen(ntcBLen_);
    tilingData_.set_tcBLen(tcBLen_);
    tilingData_.set_ntcULen(ntcULen_);
    tilingData_.set_tcULen(tcULen_);
    tilingData_.set_aLpUnit(aLpUnit_);
    tilingData_.set_uLpUnit(uLpUnit_);
    tilingData_.set_uInOffset(uInOffset_);
    tilingData_.set_uOutOffset(uOutOffset_);
    tilingData_.set_isUNotB(isUNotB_);
    tilingData_.set_isLastDimB(isLastDimB_);
    tilingData_.set_uAxisCnt(static_cast<uint8_t>(uAxisCnt_));
    tilingData_.set_bufferCnt(static_cast<uint8_t>(bufferCnt_));
    tilingData_.set_xSrcStride(xSrcStride_);
    tilingData_.set_xDstStride(xDstStride_);
    tilingData_.set_xSize(xSize_);
    tilingData_.set_aAxesNum(aAxesNum_);
    tilingData_.set_bAxesNum(bAxesNum_);
    tilingData_.set_aAxesParams(aAxesParams_);
    tilingData_.set_bAxesParams(bAxesParams_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = kSyncWorkSpaceSize;

    OP_LOGI("BroadcastToTilingAscendC", "The tiling data is: %s", PrintTilingData().c_str());

    return ge::GRAPH_SUCCESS;
}

std::string BroadcastToTilingAscendC::PrintTilingData()
{
    std::string tilingStr;

    tilingStr += std::to_string(tilingKey_) + ",";
    tilingStr += std::to_string(dFactor_) + ",";
    tilingStr += std::to_string(doubleMode_) + ",";
    tilingStr += std::to_string(uAxisCnt_) + ",";
    tilingStr += std::to_string(bufferCnt_) + ",";
    tilingStr += std::to_string(blockAxis_) + ",";
    tilingStr += std::to_string(tensorSize_) + ",";
    tilingStr += std::to_string(usedCoreCnt_) + ",";
    tilingStr += std::to_string(ntcALen_) + ",";
    tilingStr += std::to_string(tcALen_) + ",";
    tilingStr += std::to_string(ntcBLen_) + ",";
    tilingStr += std::to_string(tcBLen_) + ",";
    tilingStr += std::to_string(ntcULen_) + ",";
    tilingStr += std::to_string(tcULen_) + ",";
    tilingStr += std::to_string(aLpUnit_) + ",";
    tilingStr += std::to_string(uLpUnit_) + ",";
    tilingStr += std::to_string(uInOffset_) + ",";
    tilingStr += std::to_string(uOutOffset_) + ",";
    tilingStr += std::to_string(isUNotB_) + ",";
    tilingStr += std::to_string(isLastDimB_) + ",";
    tilingStr += std::to_string(aAxesNum_) + ",";
    tilingStr += std::to_string(bAxesNum_) + " ";
    tilingStr += ",DMA: ";
    for (size_t i = 0; i < brctoMaxDMADimNum; i++) {
        tilingStr += std::to_string(xSrcStride_[i]) + " ";
        tilingStr += std::to_string(xDstStride_[i]) + " ";
        tilingStr += std::to_string(xSize_[i]) + " ";
    }
    tilingStr += ",aAxes: ";
    for (int32_t j = 0; j < aAxesNum_; j++) {
        tilingStr += std::to_string(aAxesParams_[j * aParamUnit]) + " ";
        tilingStr += std::to_string(aAxesParams_[j * aParamUnit + 1]) + " ";
        tilingStr += std::to_string(aAxesParams_[j * aParamUnit + nTwo]) + " ";
    }
    tilingStr += ",bAxes: ";
    for (int32_t k = 0; k < bAxesNum_; k++) {
        tilingStr += std::to_string(bAxesParams_[k * bParamUnit]) + " ";
        tilingStr += std::to_string(bAxesParams_[k * bParamUnit + 1]) + " ";
    }

    return tilingStr;
}

ge::graphStatus BroadcastToTilingAscendC::DoTiling()
{
    CalcTilingData();
    return WriteTilingData();
}

} // namespace brcto

ge::graphStatus Tiling4BroadcastToAscendC(gert::TilingContext* context, const gert::Shape* inShapePtr,
                                          const gert::Shape* outShapePtr)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    brcto::BroadcastToTilingAscendC brcToTiling(context, inShapePtr, outShapePtr);

    if (brcToTiling.GetHardwareInfo<BroadcastToCompileInfo>() != ge::GRAPH_SUCCESS) {
        std::string valueMsg = "unknown";
        std::string reasonMsg = "BroadcastToTilingAscendC failed to get hardware info.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "hardwareInfo", valueMsg.c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return brcToTiling.DoTiling();
}

} // namespace optiling
