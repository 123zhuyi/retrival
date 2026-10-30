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
 * \file transpose_tiling_arch35.cpp
 * \brief
 */

#include <sstream>
#include "util/platform_util.h"
#include "transpose_tiling_base.h"
#include "transpose_tiling_arch35.h"
#include "transpose_tiling_with_gather_arch35.h"
#include "transpose_tiling_with_nchwconv_arch35.h"
#include "transpose_tiling_with_021vconv_arch35.h"
#include "common/inc/op_host/math_log.h"

namespace optiling {
static constexpr int32_t VCONV_DIM_NUM = 2;
static constexpr int32_t VCONV_DSIZE = 2;

ge::graphStatus TransposeNddmaTiling::Init(const int64_t& coreNum, const int64_t& ubSize)
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init TransposeNddmaTiling.");
    coreNum_ = coreNum;
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    ubSize_ = ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    cacheLineSize_ = Ops::Base::GetCacheLineSize(tilingContext_);
    OP_CHECK_IF((cacheLineSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get cache line size."),
                return ge::GRAPH_FAILED);

    ubBlockSize_ = Ops::Base::GetUbBlockSize(tilingContext_);
    OP_CHECK_IF((ubBlockSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub block size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::RunTranposelTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start running Tiling4Transpose.");
    if (!isReleatedTranspsoe_) {
        OP_CHECK_IF(GetShapeInfo() != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext_->GetNodeName(), "Failed to get shape info!"), return ge::GRAPH_FAILED);
    }

    auto ret = CheckShapeInfo();
    CHECK_RET_SUCC(ret);
    // if axis value is 1, remove it.
    RemoveAxisV2(shapeInfo_);
    // reduce axis
    MergeAxisV2(shapeInfo_);
    // check reduced shape
    ret = CheckReducedShapeInfo();
    CHECK_RET_SUCC(ret);

    CalcTotalVolumeActual();
    OP_CHECK_IF(TryVCONVTiling() == ge::GRAPH_SUCCESS, OP_LOGD(tilingContext_->GetNodeName(), "Do convTiling success"),
                return ge::GRAPH_SUCCESS);

    SetIsLastAxisTranspose();
    if (!isReleatedTranspsoe_ && shapeInfo_.isLastAxisTranspose) {
        TransWithGather::PlatInfo platInfo{coreNum_, ubSize_, cacheLineSize_, ubBlockSize_};
        TransWithGather::TransposeGatherTiling gatherTiling(tilingContext_, platInfo, shapeInfo_);
        OP_CHECK_IF(gatherTiling.DoTiling() == ge::GRAPH_SUCCESS,
                    OP_LOGD(tilingContext_->GetNodeName(), "Do gather tiling done!"), return ge::GRAPH_SUCCESS);
    }

    // ensure tiling template
    EntryTilingTemplate();
    // UB split
    CalcUBSplitInfo();
    // block split
    CalcBlockSplitInfo();
    // dim expand
    NDDMADimExpand();
    // get in ub shape info
    GetInUbShapeInfo();
    // cut twice get interval info
    GetIntervalInfo();
    // fill data
    FillTilingData();
    // print data
    PrintTilingData();
    // set block dim and tilingKey
    tilingContext_->SetBlockDim(tilingData_.transposeOpTiling.get_realCoreNum());
    tilingContext_->SetTilingKey(tilingKey_);
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;
    OP_LOGD(tilingContext_->GetNodeName(), "Tiling4Transpose success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::TryVCONVTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start Try VCONVTiling.");
    auto platformInfo = tilingContext_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto arch = ascendcPlatform.GetCurNpuArch();
    if (!isReleatedTranspsoe_ && arch == NpuArch::DAV_5102) {
        SMALL_SHAPE_BYTES_THRES_HOLD = SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102;
        if (shapeInfo_.reducedPerm[0] == 1 && shapeInfo_.reducedPerm[1] == 0 && shapeInfo_.dim == VCONV_DIM_NUM &&
            shapeInfo_.eleLenInBytes == VCONV_DSIZE && shapeInfo_.reducedInShape[0] > DIM_FIVE) {
            TransposeWithVCONV::PlatInfo platInfo{coreNum_, ubSize_};
            TransposeWithVCONV::TransposeVCONVTiling vconvTiling(tilingContext_, platInfo, shapeInfo_);
            OP_CHECK_IF(vconvTiling.DoTiling() == ge::GRAPH_SUCCESS,
                        OP_LOGD(tilingContext_->GetNodeName(), "Do convTiling done"), return ge::GRAPH_SUCCESS);
        }
        if (Is021VConvValid()) {
            Transpose021WithVCONV::PlatInfo platInfo{coreNum_, ubSize_};
            Transpose021WithVCONV::Transpose021VCONVTiling vconv021Tiling(tilingContext_, platInfo, shapeInfo_);
            OP_CHECK_IF(vconv021Tiling.DoTiling() == ge::GRAPH_SUCCESS,
                        OP_LOGD(tilingContext_->GetNodeName(), "Do 021 convTiling done"), return ge::GRAPH_SUCCESS);
        }
    }
    return ge::GRAPH_FAILED;
}

bool TransposeNddmaTiling::Is021VConvValid()
{
    // check perm: 021 transpose
    if (!(shapeInfo_.reducedPerm[0] == 0 && shapeInfo_.reducedPerm[DIM_ONE] == VCONV_DIM_NUM &&
          shapeInfo_.reducedPerm[DIM_TWO] == 1)) {
        return false;
    }
    // check dim
    if (shapeInfo_.dim != DIM_THREE) {
        return false;
    }
    // check dtype: support B8、B16、B32
    if (!(shapeInfo_.eleLenInBytes == B8_BYTES || shapeInfo_.eleLenInBytes == B16_BYTES ||
          shapeInfo_.eleLenInBytes == B32_BYTES)) {
        return false;
    }
    // check HW shape
    int64_t H = shapeInfo_.reducedInShape[DIM_ONE];
    int64_t W = shapeInfo_.reducedInShape[DIM_TWO];
    if (H <= DIM_EIGHT || W <= DIM_EIGHT) {
        return false;
    }
    if (H * W < HW_MIN_PRODUCT) {
        return false;
    }
    int64_t hAlign = Ops::Base::CeilDiv(H, HW_ALIGN) * HW_ALIGN;
    int64_t wAlign = Ops::Base::CeilDiv(W, HW_ALIGN) * HW_ALIGN;
    if (H * W <= hAlign * wAlign / DIM_TWO) {
        return false;
    }
    // check total volume
    if (shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes < SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_021) {
        return false;
    }
    return true;
}

template <typename T>
bool TransposeNddmaTiling::GetPerm(const gert::Tensor* permTensor)
{
    const T* permValue = permTensor->GetData<T>();
    if (!permValue) {
        OP_LOGE(tilingContext_->GetNodeName(), "Perm GetData is nullptr");
        return false;
    }
    int64_t dims = permTensor->GetShapeSize();
    for (int64_t i = 0; i < dims; i++) {
        shapeInfo_.perm[i] = permValue[i] < 0 ? permValue[i] + dims : permValue[i];
    }
    return true;
}

void TransposeNddmaTiling::SetIsLastAxisTranspose()
{
    int64_t dim = shapeInfo_.dim;
    shapeInfo_.isLastAxisTranspose = shapeInfo_.reducedPerm[dim - 1] != dim - 1 ? true : false;
}

void TransposeNddmaTiling::CalcTotalVolumeActual()
{
    int64_t vol = 1;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        vol = vol * shapeInfo_.reducedInShape[i];
    }
    shapeInfo_.totalVolumeActual = vol;
}

ge::graphStatus TransposeNddmaTiling::GetShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering GetShapeInfo.");

    const gert::Tensor* permTensor = tilingContext_->GetInputTensor(INPUT_IDX_PERM);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, permTensor);
    shapeInfo_.permSize = permTensor->GetShapeSize();

    auto permDtype = tilingContext_->GetInputDesc(INPUT_IDX_PERM)->GetDataType();
    uint64_t permDtypeSize = ge::GetSizeByDataType(permDtype);
    if (permDtypeSize == B32_BYTES) {
        if (!GetPerm<int32_t>(permTensor)) {
            return ge::GRAPH_FAILED;
        }
    } else if (permDtypeSize == B64_BYTES) {
        if (!GetPerm<int64_t>(permTensor)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "perm",
                                  ge::TypeUtils::DataTypeToSerialString(permDtype).c_str(), "int32 or int64");
        return ge::GRAPH_FAILED;
    }

    auto outputY = tilingContext_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, outputY);
    auto yShape = outputY->GetStorageShape();
    auto yDims = yShape.GetDimNum();
    auto inputX = tilingContext_->GetInputTensor(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputX);
    auto xShape = inputX->GetStorageShape();
    auto xDims = xShape.GetDimNum();

    auto xDtype = tilingContext_->GetInputDesc(INPUT_IDX_X)->GetDataType();
    shapeInfo_.eleLenInBytes = ge::GetSizeByDataType(xDtype);
    shapeInfo_.inShapeSize = xDims;
    shapeInfo_.outShapeSize = yDims;
    shapeInfo_.dim = xDims;
    shapeInfo_.origDim = xDims;
    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        shapeInfo_.inShape[i] = xShape[i];
        shapeInfo_.outShape[i] = yShape[i];
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckShapeDims()
{
    int64_t inDims = shapeInfo_.inShapeSize;
    int64_t outDims = shapeInfo_.outShapeSize;
    int64_t permDims = shapeInfo_.permSize;
    OP_CHECK_IF(inDims < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                                         std::to_string(inDims).c_str(), "positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inDims != outDims,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "x and y",
                                                          Ops::Math::Join(inDims, outDims).c_str(),
                                                          "The shape dims of x and y must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inDims != permDims,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    tilingContext_->GetNodeName(), "perm", std::to_string(permDims).c_str(),
                    "The total number of elements of perm must be equal to the shape dim of x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CheckShapeInfo.");
    CHECK_RET_SUCC(CheckShapeDims());

    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.perm[i] >= shapeInfo_.inShapeSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "perm",
                                                  std::to_string(shapeInfo_.perm[i]).c_str(),
                                                  "The value of perm must be less than shape dim of x");
            return ge::GRAPH_FAILED;
        }
        if (shapeInfo_.inShape[shapeInfo_.perm[i]] != shapeInfo_.outShape[i]) {
            std::ostringstream oss;
            oss << "The shape of y must be the same as the shape consisting of the axes of x. "
                << "The " << i << "-th axis is determined by value " << shapeInfo_.perm[i] << " of perm. "
                << "When the value of perm is a negative number, the " << i << "-th axis is equal to the "
                << std::abs(shapeInfo_.perm[i]) << "-th axis couted from the end of shape of x";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                tilingContext_->GetNodeName(), "x and y",
                Ops::Math::Join(shapeInfo_.inShape[shapeInfo_.perm[i]], shapeInfo_.outShape[i]).c_str(),
                oss.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }

    for (int64_t i = 0; i < shapeInfo_.inShapeSize; i++) {
        if (shapeInfo_.inShape[i] <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "x",
                                                  std::to_string(shapeInfo_.inShape[i]).c_str(),
                                                  "All axes of x must be positive numbers");
            return ge::GRAPH_FAILED;
        }
        if (shapeInfo_.outShape[i] <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), "y",
                                                  std::to_string(shapeInfo_.outShape[i]).c_str(),
                                                  "All axes of y must be positive numbers");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeNddmaTiling::CheckReducedShapeInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CheckReducedShapeInfo.");
    auto dim = shapeInfo_.dim;
    if (dim < 1) {
        OP_LOGE(tilingContext_->GetNodeName(), "The dim of reducedShape is invalid, dim = %ld", dim);
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo_.reducedInShape[i] <= 0 || shapeInfo_.reducedOutShape[i] <= 0) {
            OP_LOGE(tilingContext_->GetNodeName(), "Invalid shape, index is %ld, inShape is %ld, outShape is %ld", i,
                    shapeInfo_.reducedInShape[i], shapeInfo_.reducedOutShape[i]);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

int64_t TransposeNddmaTiling::DoSplitUBInput()
{
    int64_t remainingTotalElment = shapeInfo_.totalVolumeActual;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        int64_t currentShapeDim = shapeInfo_.reducedInShape[shapeInfo_.dim - 1 - i];
        if (splitInfo_.inUbElement < currentShapeDim) {
            splitInfo_.inCutIndex = shapeInfo_.dim - 1 - i;
            splitInfo_.inUbFactor = splitInfo_.inUbElement;
            splitInfo_.inTailFactor = currentShapeDim % splitInfo_.inUbFactor;
            splitInfo_.inUbActual *= splitInfo_.inUbElement;
            remainingTotalElment = remainingTotalElment / currentShapeDim *
                                   Ops::Base::CeilDiv(currentShapeDim, splitInfo_.inUbElement);
            break;
        } else {
            splitInfo_.inUbElement /= currentShapeDim;
            splitInfo_.inUbActual *= currentShapeDim;
            remainingTotalElment /= currentShapeDim;
            splitInfo_.inUbFactor = currentShapeDim;
        }
    }
    splitInfo_.outUbElement = splitInfo_.ubElement / splitInfo_.inUbActual;
    return remainingTotalElment;
}

int64_t TransposeNddmaTiling::FindOutIndex(int64_t index)
{
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        if (shapeInfo_.reducedPerm[i] == index) {
            return i;
        }
    }
    return 0;
}

bool TransposeNddmaTiling::UbOutOfBoundCheck(int64_t currentSplitIndex, int64_t currentSplitValue, bool calcIn)
{
    int64_t burstLenBlockAlign = 1;
    for (int64_t i = currentSplitIndex + 1; i < shapeInfo_.dim; i++) {
        burstLenBlockAlign *= shapeInfo_.reducedOutShape[i];
    }
    burstLenBlockAlign *= currentSplitValue;
    if (calcIn && shapeInfo_.reducedPerm[currentSplitIndex] == splitInfo_.inCutIndex) {
        burstLenBlockAlign *= splitInfo_.inUbFactor;
    }
    burstLenBlockAlign = Ops::Base::CeilAlign(burstLenBlockAlign * shapeInfo_.eleLenInBytes, ubBlockSize_) /
                         shapeInfo_.eleLenInBytes;
    int64_t inUbElements = burstLenBlockAlign;
    for (int64_t i = 0; i < currentSplitIndex; i++) {
        if (shapeInfo_.reducedPerm[i] > splitInfo_.inCutIndex) {
            inUbElements *= shapeInfo_.reducedOutShape[i];
        } else if (shapeInfo_.reducedPerm[i] == splitInfo_.inCutIndex) {
            inUbElements *= splitInfo_.inUbFactor;
        }
    }
    if (inUbElements > splitInfo_.ubElement) {
        return true;
    }
    return false;
}

bool TransposeNddmaTiling::UbOutOfBoundCheckNLast(int64_t currentSplitIndex, int64_t currentSplitValue)
{
    int64_t burstLenBlockAlign = 1;
    if (currentSplitIndex == shapeInfo_.dim - 1) {
        burstLenBlockAlign = currentSplitValue;
    } else {
        burstLenBlockAlign = shapeInfo_.reducedInShape[shapeInfo_.dim - 1];
    }
    burstLenBlockAlign = Ops::Base::CeilAlign(burstLenBlockAlign * shapeInfo_.eleLenInBytes, ubBlockSize_) /
                         shapeInfo_.eleLenInBytes;
    int64_t inUbElements = burstLenBlockAlign;
    for (int64_t i = currentSplitIndex; i < shapeInfo_.dim - 1; i++) {
        if (i == currentSplitIndex) {
            inUbElements *= currentSplitValue;
        } else {
            inUbElements *= shapeInfo_.reducedInShape[i];
        }
    }
    if (inUbElements > splitInfo_.ubElement) {
        return true;
    }
    return false;
}

void TransposeNddmaTiling::FindSplitFactorByRateNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                      int64_t remainingTotalElment)
{
    splitInfo_.inCutIndex = currentSplitIndex;
    splitInfo_.inUbFactor = 1;
    splitInfo_.inTailFactor = 0;
    for (int64_t i = splitInfo_.inUbElement; i >= DIM_TWO; i--) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        double rate = static_cast<double>(coreNumNew) / coreNum_;
        if ((rate >= VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            splitInfo_.inUbFactor = i;
            splitInfo_.inTailFactor = currentInShapeDim % i;
            splitInfo_.inUbActual *= i;
            break;
        }
    }
}

void TransposeNddmaTiling::FindSplitFactorByMultiplesLast(int64_t currentSplitIndex, int64_t currentShapeDim,
                                                          int64_t remainingTotalElment, int64_t coreNumMultiples)
{
    splitInfo_.outCutIndex = currentSplitIndex;
    int64_t bestI = 1;
    for (int64_t i = 1; i <= splitInfo_.outUbElement; i++) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentShapeDim, i);
        if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples) &&
            !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
            splitInfo_.outUbFactor = i;
            splitInfo_.outTailFactor = currentShapeDim % i;
            splitInfo_.outUbActual *= i;
            return;
        }
        if (!UbOutOfBoundCheck(currentSplitIndex, i, true)) {
            bestI = i;
        }
    }
    splitInfo_.outUbFactor = bestI;
    splitInfo_.outTailFactor = currentShapeDim % bestI;
    splitInfo_.outUbActual *= bestI;
}

void TransposeNddmaTiling::FindSplitFactorByMultiplesNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                           int64_t remainingTotalElment, int64_t coreNumMultiples)
{
    splitInfo_.inCutIndex = currentSplitIndex;
    for (int64_t i = splitInfo_.inUbElement; i >= 1; i--) {
        if (!UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            splitInfo_.inUbFactor = i;
            splitInfo_.inTailFactor = currentInShapeDim % i;
            splitInfo_.inUbActual *= i;
            break;
        }
    }
    int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, splitInfo_.inUbFactor);
    coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
    for (int64_t i = 1; i < splitInfo_.inUbFactor; i++) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples) &&
            !UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            splitInfo_.inUbFactor = i;
            splitInfo_.inTailFactor = currentInShapeDim % i;
            splitInfo_.inUbActual = splitInfo_.inUbActual / splitInfo_.inUbElement * i;
            break;
        }
    }
}

void TransposeNddmaTiling::DoSplitUB()
{
    int64_t remainingTotalElment = DoSplitUBInput();
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        int64_t currentSplitIndex = shapeInfo_.dim - 1 - i;
        if (shapeInfo_.reducedPerm[currentSplitIndex] > splitInfo_.inCutIndex) { // skip axis full cut by input shape
            continue;
        }
        int64_t currentShapeDim = shapeInfo_.reducedOutShape[currentSplitIndex];
        if (shapeInfo_.reducedPerm[currentSplitIndex] == splitInfo_.inCutIndex) {
            currentShapeDim = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex],
                                                 splitInfo_.inUbFactor);
        }
        remainingTotalElment /= currentShapeDim;
        int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentShapeDim, splitInfo_.outUbElement);
        if (splitInfo_.outUbElement < currentShapeDim) {
            if (coreNumTmp > coreNum_) { // use full coreNum
                int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
                FindSplitFactorByMultiplesLast(currentSplitIndex, currentShapeDim, remainingTotalElment,
                                               coreNumMultiples);
            } else {
                splitInfo_.outCutIndex = currentSplitIndex;
                splitInfo_.outUbFactor = splitInfo_.outUbElement;
                splitInfo_.outTailFactor = currentShapeDim % splitInfo_.outUbElement;
                splitInfo_.outUbActual *= splitInfo_.outUbElement;
            }
            break;
        } else {
            splitInfo_.outUbElement /= currentShapeDim;
            splitInfo_.outUbActual *= currentShapeDim;
            splitInfo_.outCutIndex = currentSplitIndex;
            splitInfo_.outUbFactor = currentShapeDim;
        }
    }
    if (splitInfo_.outCutIndex > FindOutIndex(splitInfo_.inCutIndex)) {
        tilingKey_ = static_cast<int64_t>(SplitMode::CUT_TWICE);
    } else {
        tilingKey_ = static_cast<int64_t>(SplitMode::CUT_ONCE);
    }
}

void TransposeNddmaTiling::DoSplitUBBigDim()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering DoSplitUBBigDim.");
    int64_t dimSize = NDDMA_MAX_DIM_NUM - 1;
    int64_t totalElment = shapeInfo_.totalVolumeActual;
    // search split index and calc base number
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; i--) {
        if (splitInfo_.ubElement < shapeInfo_.reducedOutShape[i]) {
            splitInfo_.outCutIndex = i;
            splitInfo_.outUbFactor = splitInfo_.ubElement;
            splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
            break;
        } else if (dimSize > 0) {
            totalElment = totalElment / shapeInfo_.reducedOutShape[i];
            if (totalElment <= coreNum_) {
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = splitInfo_.ubElement <= shapeInfo_.reducedOutShape[i] ?
                                             splitInfo_.ubElement :
                                             shapeInfo_.reducedOutShape[i];
                splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
                break;
            }
            splitInfo_.ubElement = splitInfo_.ubElement / shapeInfo_.reducedOutShape[i];
            dimSize--;
        } else if (dimSize == 0) {
            splitInfo_.outCutIndex = i;
            splitInfo_.outUbFactor = splitInfo_.ubElement <= shapeInfo_.reducedOutShape[i] ?
                                         splitInfo_.ubElement :
                                         shapeInfo_.reducedOutShape[i];
            splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
            break;
        }
    }
}

void TransposeNddmaTiling::FlushBaseNumForBigDim()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FlushBaseNumForBigDim.");
    int64_t idxNum = NDDMA_MAX_DIM_NUM - 1;
    int64_t baseInNum = 1;
    int64_t baseOutNum = 1;
    int64_t tmpNddmaShape[NDDMA_MAX_DIM_NUM] = {0};
    int64_t oriNddmaIdx[NDDMA_MAX_DIM_NUM] = {-1};
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; i--) {
        baseInShape_[i] = baseInNum;
        baseInNum *= shapeInfo_.reducedInShape[i];
        baseOutNum *= shapeInfo_.reducedOutShape[i];
        if (i > splitInfo_.outCutIndex) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            totalNddmaNum_ *= shapeInfo_.reducedOutShape[i];
            idxNum--;
        } else if (i == splitInfo_.outCutIndex) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            totalNddmaNum_ *= splitInfo_.outUbFactor;
            idxNum--;
        } else if (idxNum >= 0) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            idxNum--;
        }
    }

    std::sort(std::begin(nddmaIdx_), std::end(nddmaIdx_));
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        for (int64_t j = 0; j < NDDMA_MAX_DIM_NUM; j++) {
            if (nddmaIdx_[i] == oriNddmaIdx[j]) {
                baseNddmaShape_[i] = tmpNddmaShape[j];
            }
        }
    }
}

void TransposeNddmaTiling::EntryTilingTemplate()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering EntryTilingTemplate.");
    SetIsLastAxisTranspose();
    splitInfo_.ubElement = ubSize_ / shapeInfo_.eleLenInBytes;
    if (shapeInfo_.dim == 1) {
        // just tensor move
        tilingKey_ = static_cast<int64_t>(SplitMode::TENSOR_MOVE);
        return;
    }

    auto platformInfo = tilingContext_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto arch = ascendcPlatform.GetCurNpuArch();
    if (arch == NpuArch::DAV_5102 && !shapeInfo_.isLastAxisTranspose &&
        shapeInfo_.reducedInShape[shapeInfo_.dim - 1] >= MOVEALIGN_LAST_MIN_ELE) {
        SMALL_SHAPE_BYTES_THRES_HOLD = SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_NLAST;
    }
    if (shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes >= SMALL_SHAPE_BYTES_THRES_HOLD) {
        if (!shapeInfo_.isLastAxisTranspose &&
            shapeInfo_.reducedInShape[shapeInfo_.dim - 1] >= MOVEALIGN_LAST_MIN_ELE) {
            // MoveAlign场景 （大于等于32个元素）
            tilingKey_ = static_cast<int64_t>(SplitMode::N_LAST_TRANSPOSE);
            return;
        }
        if (shapeInfo_.dim <= NDDMA_MAX_DIM_NUM) {
            // nddma场景：尾轴不转置以及尾轴转置且dst[-1]>32（待优化）
            tilingKey_ = static_cast<int64_t>(SplitMode::NDDMA_BASE);
            return;
        } else {
            // BigShape场景,维度超过NDDMA_MAX_DIM_NUM且总元素超过threshold
            tilingKey_ = static_cast<int64_t>(SplitMode::BIG_DIM);
            return;
        }
    }
    // 兜底逻辑 simt
    tilingKey_ = static_cast<int64_t>(SplitMode::SMALL_SHAPE);
}

void TransposeNddmaTiling::CalcUBSplitInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcUBSplitInfo");
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::TENSOR_MOVE):
        case static_cast<int64_t>(SplitMode::N_LAST_TRANSPOSE):
            splitInfo_.ubElement = ubSize_ / BUFFER_NUM / shapeInfo_.eleLenInBytes;
            break;
        case static_cast<int64_t>(SplitMode::NDDMA_BASE):
            splitInfo_.inUbElement = sqrt(splitInfo_.ubElement);
            DoSplitUB();
            break;
        case static_cast<int64_t>(SplitMode::BIG_DIM):
            DoSplitUBBigDim();
            break;
        default:
            break;
    }
}

void TransposeNddmaTiling::CalcBlockSplitInfoForTensorMove()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForTensorMove.");
    if (shapeInfo_.totalVolumeActual < coreNum_) {
        realCoreNum_ = 1;
        blkFactor_ = shapeInfo_.totalVolumeActual;
        blkTailFactor_ = 0;
        splitInfo_.inUbFactor = splitInfo_.ubElement;
    } else {
        realCoreNum_ = coreNum_;
        blkFactor_ = shapeInfo_.totalVolumeActual / coreNum_;
        blkTailFactor_ = shapeInfo_.totalVolumeActual % coreNum_;
        splitInfo_.inUbFactor = splitInfo_.ubElement;
    }
}

int64_t TransposeNddmaTiling::CalcBlockSplitInfoForNoCutForMultiCore(int64_t i, int64_t shapeSizeByte,
                                                                     int64_t& totalElment)
{
    for (int64_t j = 2; j <= shapeInfo_.reducedOutShape[i]; j++) {
        if ((shapeInfo_.reducedOutShape[i] % j == 0) &&
            (shapeSizeByte / shapeInfo_.reducedOutShape[i] * j > cacheLineSize_)) {
            if (j == shapeInfo_.reducedOutShape[i] && i == 0) {
                // 素数且切到了最后，正常切
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = Ops::Base::CeilDiv(cacheLineSize_ + 1,
                                                            shapeSizeByte / shapeInfo_.reducedOutShape[i]);
                splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[i] % splitInfo_.outUbFactor;
                totalElment *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[i], splitInfo_.outUbFactor);
                break;
            } else if (j == shapeInfo_.reducedOutShape[i] && i != 0) {
                // 素数但还没切到最后，全切，剩下的轴开多核
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = shapeInfo_.reducedOutShape[i];
                splitInfo_.outTailFactor = 0;
                break;
            } else {
                // 其他场景，按最小因数切
                splitInfo_.outCutIndex = i;
                splitInfo_.outUbFactor = j;
                splitInfo_.outTailFactor = 0;
                totalElment *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[i], splitInfo_.outUbFactor);
                break;
            }
        }
    }
    return totalElment;
}

void TransposeNddmaTiling::CalcBlockSplitInfoForSmallShape()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForSmallShape.");
    int64_t totalElements = shapeInfo_.totalVolumeActual;
    if (totalElements < coreNum_) {
        realCoreNum_ = totalElements;
        blkFactor_ = 1;
        blkTailFactor_ = 0;
        return;
    }
    // simt every core elemets align to 128Byte
    int64_t blkFactor = totalElements / coreNum_;
    int64_t ceilAlignFactor = Ops::Base::CeilDiv(blkFactor * shapeInfo_.eleLenInBytes,
                                                 SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE) *
                              SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE / shapeInfo_.eleLenInBytes;
    int64_t floorAlignFactor = Ops::Base::FloorDiv(blkFactor * shapeInfo_.eleLenInBytes,
                                                   SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE) *
                               SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE / shapeInfo_.eleLenInBytes;
    if (totalElements - floorAlignFactor * (coreNum_ - 1) <= floorAlignFactor) {
        realCoreNum_ = coreNum_;
        blkFactor_ = floorAlignFactor;
        blkTailFactor_ = totalElements % floorAlignFactor;
    } else {
        realCoreNum_ = Ops::Base::CeilDiv(totalElements, ceilAlignFactor);
        blkFactor_ = ceilAlignFactor;
        blkTailFactor_ = totalElements % ceilAlignFactor;
    }
}

void TransposeNddmaTiling::CalcBlockSplitInfoForNLastTranspose()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForNLastTranspose.");
    splitInfo_.inUbElement = splitInfo_.ubElement;
    int64_t remainingTotalElment = shapeInfo_.totalVolumeActual;
    int64_t solvedTotalElment[MAX_AXIS_NUM_FOR_TRANSPOSE];
    solvedTotalElment[shapeInfo_.dim - 1] = 1;
    int64_t currentSplitIndex;
    int64_t currentInShapeDim;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        currentSplitIndex = shapeInfo_.dim - 1 - i;
        currentInShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
        remainingTotalElment /= currentInShapeDim;
        int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, splitInfo_.inUbElement);
        if (splitInfo_.inUbElement < currentInShapeDim) {
            if (coreNumTmp < coreNum_) { // use at least VEC_CORE_USED_THRES_HOLD * coreNum
                FindSplitFactorByRateNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment);
                break;
            } else { // use full coreNum
                int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
                FindSplitFactorByMultiplesNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment,
                                                coreNumMultiples);
                /* 检查inUbFactor是否合法，不合法则执行退轴逻辑 */
                CheckInUbFactorValid(currentSplitIndex, currentInShapeDim, remainingTotalElment, coreNumMultiples,
                                     solvedTotalElment);
                break;
            }
        } else if (coreNumTmp < coreNum_) { // use at least VEC_CORE_USED_THRES_HOLD * coreNum
            FindSplitFactorByRateNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment);
            break;
        } else {
            splitInfo_.inUbElement /= currentInShapeDim;
            solvedTotalElment[currentSplitIndex] = currentSplitIndex == shapeInfo_.dim - 1 ?
                                                       currentInShapeDim :
                                                       (currentInShapeDim * solvedTotalElment[currentSplitIndex + 1]);
        }
    }
    int64_t coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor) *
                      remainingTotalElment;
    SetRealCoreNumAndBlkFactor(coreNum);
}

void TransposeNddmaTiling::CheckInUbFactorValid(int64_t& currentSplitIndex, int64_t& currentInShapeDim,
                                                int64_t& remainingTotalElment, int64_t& coreNumMultiples,
                                                int64_t* solvedTotalElment)
{
    if (splitInfo_.inUbFactor == 0 && currentSplitIndex < shapeInfo_.dim - 1) {
        while (currentSplitIndex < shapeInfo_.dim - 1) {
            currentSplitIndex++;
            currentInShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
            splitInfo_.inUbElement = shapeInfo_.reducedInShape[currentSplitIndex];
            remainingTotalElment = shapeInfo_.totalVolumeActual / solvedTotalElment[currentSplitIndex];
            coreNumMultiples = remainingTotalElment;
            FindSplitFactorByMultiplesNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment,
                                            coreNumMultiples);
            if (splitInfo_.inUbFactor > 0) {
                break;
            }
        }
    }
}

void TransposeNddmaTiling::SetRealCoreNumAndBlkFactor(int64_t coreNum)
{
    if (coreNum >= coreNum_) {
        realCoreNum_ = coreNum_;
        blkFactor_ = coreNum / coreNum_;
        blkTailFactor_ = coreNum % coreNum_;
    } else {
        realCoreNum_ = coreNum;
        blkFactor_ = 1;
        blkTailFactor_ = 0;
    }
}

void TransposeNddmaTiling::CalcBlockSplitInfoForCutOnce()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForCutOnce.");
    // input and output split the same axis, the split factor is subject to the output
    if (splitInfo_.inCutIndex == shapeInfo_.reducedPerm[splitInfo_.outCutIndex]) {
        splitInfo_.outUbFactor *= splitInfo_.inUbFactor;
        splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex] % splitInfo_.outUbFactor;
    }
    // input and output split different axis, but output split factor is the whole axis
    if (splitInfo_.outUbFactor == shapeInfo_.reducedOutShape[splitInfo_.outCutIndex]) {
        splitInfo_.outCutIndex = FindOutIndex(splitInfo_.inCutIndex);
        splitInfo_.outUbFactor = splitInfo_.inUbFactor;
        splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex] % splitInfo_.outUbFactor;
    }
    int64_t outUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor);
    int64_t outUbAxisExceptSplitAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        if (shapeInfo_.reducedPerm[i] < splitInfo_.inCutIndex) {
            outUbAxisExceptSplitAxis *= shapeInfo_.reducedOutShape[i];
        }
    }
    if (outUbAxis < coreNum_) {
        // use at least VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t currentSplitIndex = splitInfo_.outCutIndex;
        int64_t currentShapeDim = shapeInfo_.reducedOutShape[currentSplitIndex];
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.outUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outUbAxisExceptSplitAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheck(currentSplitIndex, i, false)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (!UbOutOfBoundCheck(currentSplitIndex, i, false) && rate > bestRate) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.outUbFactor = bestI;
            splitInfo_.outTailFactor = currentShapeDim % bestI;
        }
    }
    outUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
                outUbAxisExceptSplitAxis;
    SetRealCoreNumAndBlkFactor(outUbAxis);
}

void TransposeNddmaTiling::CalcBlockSplitInfoForCutTwice()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForCutTwice.");
    int64_t outAxiseExceptSplitInAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        if (shapeInfo_.reducedPerm[i] < splitInfo_.inCutIndex) {
            outAxiseExceptSplitInAxis *= shapeInfo_.reducedOutShape[i];
        }
    }
    outAxiseExceptSplitInAxis *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex],
                                                    splitInfo_.outUbFactor);
    int64_t inUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor);
    if (outAxiseExceptSplitInAxis * inUbAxis < coreNum_) {
        // use at least VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t currentSplitIndex = splitInfo_.inCutIndex;
        int64_t currentShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.inUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outAxiseExceptSplitInAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
                splitInfo_.inUbFactor = i;
                splitInfo_.inTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (rate > bestRate && !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.inUbFactor = bestI;
            splitInfo_.inTailFactor = currentShapeDim % bestI;
        }
    }
    inUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor) *
               outAxiseExceptSplitInAxis;
    SetRealCoreNumAndBlkFactor(inUbAxis);
}

void TransposeNddmaTiling::CalcBlockSplitInfoForBigDim()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfoForBigDim.");
    int64_t outUbAxisExceptSplitAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        outUbAxisExceptSplitAxis *= shapeInfo_.reducedOutShape[i];
    }
    int64_t coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
                      outUbAxisExceptSplitAxis;
    int64_t currentShapeDim = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex];
    if (coreNum < coreNum_) {
        // use at least VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.outUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outUbAxisExceptSplitAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= VEC_CORE_USED_THRES_HOLD)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (rate > bestRate) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.outUbFactor = bestI;
            splitInfo_.outTailFactor = currentShapeDim % bestI;
        }
    } else {
        // use full coreNum
        int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNum, coreNum_);
        for (int64_t i = 1; i <= splitInfo_.outUbFactor; i++) {
            int64_t coreNumNew = outUbAxisExceptSplitAxis * Ops::Base::CeilDiv(currentShapeDim, i);
            if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                break;
            }
        }
    }
    FlushBaseNumForBigDim();
    coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
              outUbAxisExceptSplitAxis;
    SetRealCoreNumAndBlkFactor(coreNum);
}

void TransposeNddmaTiling::CalcBlockSplitInfo()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering CalcBlockSplitInfo.");
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::TENSOR_MOVE):
            CalcBlockSplitInfoForTensorMove();
            break;
        case static_cast<int64_t>(SplitMode::SMALL_SHAPE):
            CalcBlockSplitInfoForSmallShape();
            break;
        case static_cast<int64_t>(SplitMode::CUT_ONCE):
            CalcBlockSplitInfoForCutOnce();
            break;
        case static_cast<int64_t>(SplitMode::CUT_TWICE):
            CalcBlockSplitInfoForCutTwice();
            break;
        case static_cast<int64_t>(SplitMode::BIG_DIM):
            CalcBlockSplitInfoForBigDim();
            break;
        case static_cast<int64_t>(SplitMode::N_LAST_TRANSPOSE):
            CalcBlockSplitInfoForNLastTranspose();
            break;
        default:
            break;
    }
}

void TransposeNddmaTiling::NDDMADimExpand()
{
    int64_t offset = (shapeInfo_.dim < NDDMA_MAX_DIM_NUM) ? (NDDMA_MAX_DIM_NUM - shapeInfo_.dim) : 0;

    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        expandedPerm_[i + offset] = shapeInfo_.reducedPerm[i] + offset;
        expandedInputShape_[i + offset] = shapeInfo_.reducedInShape[i];
        expandedOutputShape_[i + offset] = shapeInfo_.reducedOutShape[i];
    }
}

void TransposeNddmaTiling::GetInUbShapeInfo()
{
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::SMALL_SHAPE):
            CalcInUbShapeInfoForNoNeedCut();
            break;
        case static_cast<int64_t>(SplitMode::CUT_ONCE):
            CalcInUbShapeInfoForCutOnce();
            break;
        case static_cast<int64_t>(SplitMode::CUT_TWICE):
            CalcInUbShapeInfoForCutTwice();
            break;
        default:
            break;
    }
}

void TransposeNddmaTiling::GetIntervalInfo()
{
    switch (tilingKey_) {
        case static_cast<int64_t>(SplitMode::CUT_TWICE):
            GetIntervalInfoForCutTwice();
            break;
        default:
            break;
    }
}

void TransposeNddmaTiling::CalcInUbShapeInfoForNoNeedCut()
{
    int64_t outCutIndexExpand = splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        if (i > outCutIndexExpand) {
            inUbMainDstShape_[i] = expandedOutputShape_[i];
            inUbTailDstShape_[i] = expandedOutputShape_[i];
        } else if (i == outCutIndexExpand) {
            inUbMainDstShape_[i] = splitInfo_.outUbFactor;
            inUbTailDstShape_[i] = splitInfo_.outTailFactor;
        } else {
            inUbMainDstShape_[i] = 1;
            inUbTailDstShape_[i] = 1;
        }
    }
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        inUbMainSrcShape_[expandedPerm_[i]] = inUbMainDstShape_[i];
        inUbTailSrcShape_[expandedPerm_[i]] = inUbTailDstShape_[i];
    }
}

void TransposeNddmaTiling::CalcInUbShapeInfoForCutOnce()
{
    int64_t outCutIndexExpand = splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    int64_t inCutIndexExpand = splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        inUbMainDstShape_[i] = expandedOutputShape_[i];
        inUbTailDstShape_[i] = expandedOutputShape_[i];
        if (i == outCutIndexExpand) {
            inUbMainDstShape_[i] = splitInfo_.outUbFactor;
            inUbTailDstShape_[i] = splitInfo_.outTailFactor;
        } else if (i < outCutIndexExpand && expandedPerm_[i] < inCutIndexExpand) {
            inUbMainDstShape_[i] = 1;
            inUbTailDstShape_[i] = 1;
        }
    }
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        inUbMainSrcShape_[expandedPerm_[i]] = inUbMainDstShape_[i];
        inUbTailSrcShape_[expandedPerm_[i]] = inUbTailDstShape_[i];
    }
}

void TransposeNddmaTiling::CalcInUbShapeInfoForCutTwice()
{
    // 双切分场景下，对于输入，输入切分轴右侧为UB内的轴；对于输出，输出切分轴右侧非由输入确定的UB内的轴也都为UB内的轴
    for (int64_t idx = 0; idx < NDDMA_MAX_DIM_NUM; idx++) {
        inUbMainSrcShape_[idx] = expandedInputShape_[idx];
        inUbInputTailSrcShape_[idx] = expandedInputShape_[idx];
        inUbOutputTailSrcShape_[idx] = expandedInputShape_[idx];
        inUbTailSrcShape_[idx] = expandedInputShape_[idx];
        if (idx < splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            inUbMainSrcShape_[idx] = 1;
            inUbInputTailSrcShape_[idx] = 1;
            inUbTailSrcShape_[idx] = 1;
        } else if (idx == splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            inUbMainSrcShape_[idx] = splitInfo_.inUbFactor;
            inUbInputTailSrcShape_[idx] = splitInfo_.inTailFactor;
            inUbTailSrcShape_[idx] = splitInfo_.inTailFactor;
        }
        inUbOutputTailSrcShape_[idx] = inUbInputTailSrcShape_[idx];
    }
    inUbOutputTailSrcShape_[splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim] = splitInfo_.inUbFactor;
    inUbOutputTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM -
                                          shapeInfo_.dim]] = splitInfo_.outTailFactor;
    inUbMainSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim]] = splitInfo_
                                                                                                        .outUbFactor;
    inUbInputTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM -
                                         shapeInfo_.dim]] = splitInfo_.outUbFactor;
    inUbTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim]] = splitInfo_
                                                                                                        .outTailFactor;
    for (int64_t idx = splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim + 1; idx < NDDMA_MAX_DIM_NUM;
         idx++) {
        if (expandedPerm_[idx] == splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            continue;
        } else {
            inUbMainSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbInputTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbOutputTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
        }
    }
    for (int64_t idx = 0; idx < NDDMA_MAX_DIM_NUM; idx++) {
        inUbMainDstShape_[idx] = inUbMainSrcShape_[expandedPerm_[idx]];
        inUbInputTailDstShape_[idx] = inUbInputTailSrcShape_[expandedPerm_[idx]];
        inUbOutputTailDstShape_[idx] = inUbOutputTailSrcShape_[expandedPerm_[idx]];
        inUbTailDstShape_[idx] = inUbTailSrcShape_[expandedPerm_[idx]];
    }
}

void TransposeNddmaTiling::GetIntervalInfoForCutTwice()
{
    int64_t expandedInputCutIndex = splitInfo_.inCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    int64_t inputOutputCutIndex = expandedPerm_[splitInfo_.outCutIndex + NDDMA_MAX_DIM_NUM - shapeInfo_.dim];

    int64_t outUbLoop = 1;
    for (int64_t i = NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (i != expandedInputCutIndex && i != inputOutputCutIndex) {
            outUbLoop = outUbLoop * (expandedInputShape_[i] / inUbMainSrcShape_[i]);
        }
    }

    offsetRangeMain_.end = (expandedInputShape_[expandedInputCutIndex] / inUbMainSrcShape_[expandedInputCutIndex]) *
                               (expandedInputShape_[inputOutputCutIndex] / inUbMainSrcShape_[inputOutputCutIndex]) *
                               outUbLoop -
                           1;

    if (splitInfo_.inTailFactor != 0 && splitInfo_.outTailFactor != 0) {
        offsetRangeInputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeInputTail_.end = offsetRangeInputTail_.start +
                                    (expandedInputShape_[inputOutputCutIndex] /
                                     inUbMainSrcShape_[inputOutputCutIndex]) *
                                        outUbLoop -
                                    1;
        offsetRangeOutputTail_.start = offsetRangeInputTail_.end + 1;
        offsetRangeOutputTail_.end = offsetRangeOutputTail_.start +
                                     (expandedInputShape_[expandedInputCutIndex] /
                                      inUbMainSrcShape_[expandedInputCutIndex]) *
                                         outUbLoop -
                                     1;
        offsetRangeTail_.start = offsetRangeOutputTail_.end + 1;
        offsetRangeTail_.end = offsetRangeTail_.start + outUbLoop - 1;
    }

    if (splitInfo_.inTailFactor != 0 && splitInfo_.outTailFactor == 0) {
        offsetRangeInputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeInputTail_.end = offsetRangeInputTail_.start +
                                    (expandedInputShape_[inputOutputCutIndex] /
                                     inUbMainSrcShape_[inputOutputCutIndex]) *
                                        outUbLoop -
                                    1;
    }

    if (splitInfo_.inTailFactor == 0 && splitInfo_.outTailFactor != 0) {
        offsetRangeOutputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeOutputTail_.end = offsetRangeOutputTail_.start +
                                     (expandedInputShape_[expandedInputCutIndex] /
                                      inUbMainSrcShape_[expandedInputCutIndex]) *
                                         outUbLoop -
                                     1;
    }
}

void TransposeNddmaTiling::FillTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Entering FillTilingData.");
    tilingData_.transposeOpTiling.set_permSize(shapeInfo_.dim);
    tilingData_.transposeOpTiling.set_inCutIndex(splitInfo_.inCutIndex);
    tilingData_.transposeOpTiling.set_outCutIndex(splitInfo_.outCutIndex);
    tilingData_.transposeOpTiling.set_inUbFactor(splitInfo_.inUbFactor);
    tilingData_.transposeOpTiling.set_outUbFactor(splitInfo_.outUbFactor);
    tilingData_.transposeOpTiling.set_inTailFactor(splitInfo_.inTailFactor);
    tilingData_.transposeOpTiling.set_outTailFactor(splitInfo_.outTailFactor);
    tilingData_.transposeOpTiling.set_realCoreNum(realCoreNum_);
    tilingData_.transposeOpTiling.set_blkFactor(blkFactor_);
    tilingData_.transposeOpTiling.set_blkTailFactor(blkTailFactor_);
    tilingData_.transposeOpTiling.set_ubSize(ubSize_);

    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        inputShape_[i] = shapeInfo_.reducedInShape[i];
        outputShape_[i] = shapeInfo_.reducedOutShape[i];
        perm_[i] = shapeInfo_.reducedPerm[i];
    }
    tilingData_.transposeOpTiling.set_inputShape(inputShape_);
    tilingData_.transposeOpTiling.set_outputShape(outputShape_);
    tilingData_.transposeOpTiling.set_perm(perm_);
    tilingData_.transposeOpTiling.set_baseInShape(baseInShape_);
    tilingData_.transposeOpTiling.set_baseNddmaShape(baseNddmaShape_);
    tilingData_.transposeOpTiling.set_nddmaIdx(nddmaIdx_);
    tilingData_.transposeOpTiling.set_totalNddmaNum(totalNddmaNum_);
    tilingData_.transposeOpTiling.set_rangeMainEnd(offsetRangeMain_.end);
    tilingData_.transposeOpTiling.set_rangeInputTailStart(offsetRangeInputTail_.start);
    tilingData_.transposeOpTiling.set_rangeInputTailEnd(offsetRangeInputTail_.end);
    tilingData_.transposeOpTiling.set_rangeOutputTailStart(offsetRangeOutputTail_.start);
    tilingData_.transposeOpTiling.set_rangeOutputTailEnd(offsetRangeOutputTail_.end);
    tilingData_.transposeOpTiling.set_rangeTailStart(offsetRangeTail_.start);
    tilingData_.transposeOpTiling.set_rangeTailEnd(offsetRangeTail_.end);

    tilingData_.transposeOpTiling.set_expandedPerm(expandedPerm_);
    tilingData_.transposeOpTiling.set_expandedInputShape(expandedInputShape_);
    tilingData_.transposeOpTiling.set_expandedOutputShape(expandedOutputShape_);

    tilingData_.transposeOpTiling.set_inUbMainSrcShape(inUbMainSrcShape_);
    tilingData_.transposeOpTiling.set_inUbMainDstShape(inUbMainDstShape_);
    tilingData_.transposeOpTiling.set_inUbInputTailSrcShape(inUbInputTailSrcShape_);
    tilingData_.transposeOpTiling.set_inUbInputTailDstShape(inUbInputTailDstShape_);
    tilingData_.transposeOpTiling.set_inUbOutputTailSrcShape(inUbOutputTailSrcShape_);
    tilingData_.transposeOpTiling.set_inUbOutputTailDstShape(inUbOutputTailDstShape_);
    tilingData_.transposeOpTiling.set_inUbTailSrcShape(inUbTailSrcShape_);
    tilingData_.transposeOpTiling.set_inUbTailDstShape(inUbTailDstShape_);

    if (!isReleatedTranspsoe_) {
        tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                                 tilingContext_->GetRawTilingData()->GetCapacity());
        tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    }
}

void TransposeNddmaTiling::PrintTilingData()
{
    OP_LOGI(tilingContext_->GetNodeName(), "Entering PrintTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(tilingContext_->GetNodeName(),
                "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld. \
                baseInShape[%ld] is:%ld",
                i, inputShape_[i], i, outputShape_[i], i, perm_[i], i, baseInShape_[i]);
    }
    for (int64_t i = 0; i < NDDMA_MAX_DIM_NUM; i++) {
        OP_LOGI(tilingContext_->GetNodeName(), "baseNddmaShape_[%ld] is:%ld, nddmaIdx_[%ld]:%ld", i, baseNddmaShape_[i],
                i, nddmaIdx_[i]);
    }
    OP_LOGI(tilingContext_->GetNodeName(),
            "tilingData is permSize:%ld, inCutIndex:%ld, outCutIndex:%ld, inUbFactor:%ld, outUbFactor:%ld, \
            inTailFactor:%ld, outTailFactor:%ld, realCoreNum:%ld, blkFactor:%ld, blkTailFactor:%ld, \
            ubSize:%ld, totalNddmaNum:%ld, Tiling4Transpose ends. ",
            tilingData_.transposeOpTiling.get_permSize(), tilingData_.transposeOpTiling.get_inCutIndex(),
            tilingData_.transposeOpTiling.get_outCutIndex(), tilingData_.transposeOpTiling.get_inUbFactor(),
            tilingData_.transposeOpTiling.get_outUbFactor(), tilingData_.transposeOpTiling.get_inTailFactor(),
            tilingData_.transposeOpTiling.get_outTailFactor(), tilingData_.transposeOpTiling.get_realCoreNum(),
            tilingData_.transposeOpTiling.get_blkFactor(), tilingData_.transposeOpTiling.get_blkTailFactor(),
            tilingData_.transposeOpTiling.get_ubSize(), tilingData_.transposeOpTiling.get_totalNddmaNum());
}

ge::graphStatus TransposeNddmaTiling::TilingForReleatedTranspose(gert::TilingContext* context,
                                                                 TransposeOpTilingData* tilingData,
                                                                 TransposeCompilerInfo* compilerInfo,
                                                                 ShapeInfo& opInput)
{
    OP_LOGD(context->GetNodeName(), "Start TilingForReleatedTranspose.");
    TransposeNddmaTiling tilingObject(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingObject.tilingContext_ = context;
    tilingObject.tilingData_.transposeOpTiling = *tilingData;
    tilingObject.shapeInfo_ = opInput;

    tilingObject.isReleatedTranspsoe_ = true;
    if (tilingObject.Init(compilerInfo->coreNum, compilerInfo->ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "tilingObject.isReleatedTranspsoe_: %d", tilingObject.isReleatedTranspsoe_);
    return tilingObject.RunTranposelTiling();
}

static ge::graphStatus TransposeTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "begin to do TilingForTranspose");
    auto compilerInfo = context->GetCompileInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compilerInfo);
    TransposeNddmaTiling tilingObject(context);
    if (tilingObject.Init(compilerInfo->coreNum, compilerInfo->ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunTranposelTiling();
}

ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepareTransposeForAscendC");
    auto ci = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, coreNum:%ld.", ci->coreNum),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "Transpose Op GetHardwareInfo Failed, ubSize:%ld.", ci->ubSize),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Transpose Op get coreNum:%ld, ubSize:%ld.", ci->coreNum, ci->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Transpose)
    .Tiling(TransposeTilingForAscendC)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareTransposeForAscendC);

} // namespace optiling
