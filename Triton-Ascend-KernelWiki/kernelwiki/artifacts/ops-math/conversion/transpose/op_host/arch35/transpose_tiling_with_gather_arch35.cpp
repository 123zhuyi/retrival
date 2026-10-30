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
 * \file transpose_tiling_with_gather_arch35.cpp
 * \brief
 */
#include "transpose_tiling_with_gather_arch35.h"

namespace optiling {
namespace TransWithGather {
static constexpr int8_t NUM_TWO = 2;
static constexpr int8_t NUM_THREE = 3;
static constexpr int64_t NUM_FOUR = 4;
static constexpr int64_t NUM_EIGHT = 8;
static constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;

void TransposeGatherTiling::CalcTensorSize()
{
    // ping pong, input and output
    if (shapeInfo_.eleLenInBytes == 1L) {
        dataTensorSize_ = static_cast<uint32_t>(platInfo_.ubSize / (NUM_TWO * NUM_TWO + NUM_TWO) /
                                                platInfo_.ubBlockSize * platInfo_.ubBlockSize);
        indexTensorSize_ = dataTensorSize_ * static_cast<uint32_t>(NUM_TWO);
    } else if (shapeInfo_.eleLenInBytes == NUM_EIGHT) {
        dataTensorSize_ = static_cast<uint32_t>((platInfo_.ubSize / (NUM_TWO * NUM_TWO * NUM_EIGHT + NUM_FOUR) *
                                                 NUM_EIGHT / platInfo_.ubBlockSize * platInfo_.ubBlockSize));
        indexTensorSize_ = dataTensorSize_ / static_cast<uint32_t>(NUM_TWO);
    } else {
        dataTensorSize_ = static_cast<uint32_t>(platInfo_.ubSize / (NUM_TWO * NUM_TWO + 1) / platInfo_.ubBlockSize *
                                                platInfo_.ubBlockSize);
        indexTensorSize_ = dataTensorSize_;
    }
}

int64_t TransposeGatherTiling::CalcShapeSize(const std::vector<int64_t>& shape, int64_t beg, int64_t end)
{
    int64_t res = 1;
    for (auto idx = beg; idx < end; ++idx) {
        res *= shape[idx];
    }
    return res;
}

void TransposeGatherTiling::CalcInUbPerm(int64_t sqrtedTensor)
{
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; --i) {
        inUbPerm_.perm[inUbPerm_.cnt++] = i;
        inUbPermSet_.insert(i);
        allUbPerm_.insert(i);
        if (inUbPerm_.cnt >= UB_MAX_BRW_NUM ||
            CalcShapeSize(shapeInfo_.reducedInShape, i, shapeInfo_.dim) > sqrtedTensor) {
            break;
        }
    }
}

void TransposeGatherTiling::CalcOutUbPerm(int64_t sqrtedTensor)
{
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; --i) {
        outUbPerm_.perm[outUbPerm_.cnt++] = shapeInfo_.reducedPerm[i];
        allUbPerm_.insert(shapeInfo_.reducedPerm[i]);
        if (outUbPerm_.cnt >= UB_MAX_BRW_NUM ||
            CalcShapeSize(shapeInfo_.reducedOutShape, i, shapeInfo_.dim) > sqrtedTensor) {
            break;
        }
    }
}

void TransposeGatherTiling::AdjustUbCutAxisFactor(int32_t& axisFactor, int8_t axisFlag, int64_t elemInTensor)
{
    int64_t srcInUbAxesSize = 1;
    int64_t srcOutUbAxesSize = 1;
    int64_t dstInUbAxesSize = 1;
    int64_t dstOutUbAxesSize = 1;

    std::set<int8_t> viceUbPerm0(allUbPerm_);
    for (int8_t i = 0; i < outUbPerm_.cnt - 1; ++i) {
        dstOutUbAxesSize *= shapeInfo_.reducedInShape[outUbPerm_.perm[i]];
        viceUbPerm0.erase(outUbPerm_.perm[i]);
    }
    for (int8_t i = 0; i < inUbPerm_.cnt - 1; ++i) {
        if (viceUbPerm0.find(inUbPerm_.perm[i]) != viceUbPerm0.end()) {
            dstInUbAxesSize *= shapeInfo_.reducedInShape[inUbPerm_.perm[i]];
        }
    }
    std::set<int8_t> viceUbPerm1(allUbPerm_);
    for (int8_t i = 0; i < inUbPerm_.cnt - 1; ++i) {
        srcInUbAxesSize *= shapeInfo_.reducedInShape[inUbPerm_.perm[i]];
        viceUbPerm1.erase(inUbPerm_.perm[i]);
    }
    for (int8_t i = 0; i < outUbPerm_.cnt - 1; ++i) {
        if (viceUbPerm1.find(outUbPerm_.perm[i]) != viceUbPerm1.end()) {
            srcOutUbAxesSize *= shapeInfo_.reducedInShape[outUbPerm_.perm[i]];
        }
    }

    int64_t elemPerBlock = platInfo_.ubBlockSize / shapeInfo_.eleLenInBytes;
    int64_t dstFactor = 0;
    int64_t srcFactor = 0;
    // in and out ub cut same axis
    if (axisFlag == 0) {
        bool isDstUbOverflow = (dstInUbAxesSize * Ops::Base::CeilAlign(dstOutUbAxesSize * axisFactor, elemPerBlock) >
                                elemInTensor);
        bool isSrcUbOverflow = (srcOutUbAxesSize * Ops::Base::CeilAlign(srcInUbAxesSize * axisFactor, elemPerBlock) >
                                elemInTensor);
        if (isDstUbOverflow || isSrcUbOverflow) {
            dstFactor = elemInTensor / dstInUbAxesSize / elemPerBlock * elemPerBlock / dstOutUbAxesSize;
            srcFactor = elemInTensor / srcOutUbAxesSize / elemPerBlock * elemPerBlock / srcInUbAxesSize;
            axisFactor = static_cast<int32_t>(std::min(dstFactor, srcFactor));
        }
        // out ub cut axis
    } else if (axisFlag == 1 || axisFlag == NUM_THREE) {
        if (axisFlag == NUM_THREE) {
            dstInUbAxesSize *= ubSplitInfo_.inUbCutAxisFactor;
        }
        if (axisFlag == 1) {
            srcOutUbAxesSize /= ubSplitInfo_.inUbCutAxisFactor;
        }
        bool isDstUbOverflow = (dstInUbAxesSize * Ops::Base::CeilAlign(dstOutUbAxesSize * axisFactor, elemPerBlock) >
                                elemInTensor);
        bool isSrcUbOverflow = (axisFactor * srcOutUbAxesSize *
                                    Ops::Base::CeilAlign(srcInUbAxesSize * ubSplitInfo_.inUbCutAxisFactor,
                                                         elemPerBlock) >
                                elemInTensor);
        if (isDstUbOverflow || isSrcUbOverflow) {
            dstFactor = elemInTensor / dstInUbAxesSize / elemPerBlock * elemPerBlock / dstOutUbAxesSize;
            srcFactor = Ops::Base::FloorDiv(
                elemInTensor / srcOutUbAxesSize,
                Ops::Base::CeilAlign(srcInUbAxesSize * ubSplitInfo_.inUbCutAxisFactor, elemPerBlock));
            axisFactor = static_cast<int32_t>(std::min(dstFactor, srcFactor));
        }
        // in ub cut axis
    } else if (axisFlag == NUM_TWO) {
        dstInUbAxesSize /= ubSplitInfo_.outUbCutAxisFactor;
        bool isDstUbOverflow = (axisFactor * dstInUbAxesSize *
                                    Ops::Base::CeilAlign(dstOutUbAxesSize * ubSplitInfo_.outUbCutAxisFactor,
                                                         elemPerBlock) >
                                elemInTensor);
        bool isSrcUbOverflow = (srcOutUbAxesSize * Ops::Base::CeilAlign(srcInUbAxesSize * axisFactor, elemPerBlock) >
                                elemInTensor);
        if (isDstUbOverflow || isSrcUbOverflow) {
            dstFactor = Ops::Base::FloorDiv(
                elemInTensor / dstInUbAxesSize,
                Ops::Base::CeilAlign(dstOutUbAxesSize * ubSplitInfo_.outUbCutAxisFactor, elemPerBlock));
            srcFactor = elemInTensor / srcOutUbAxesSize / elemPerBlock * elemPerBlock / srcInUbAxesSize;
            axisFactor = static_cast<int32_t>(std::min(dstFactor, srcFactor));
        }
    }
}

void TransposeGatherTiling::CalcUbAxisCutFactor(int64_t elemInTensor, int64_t sqrtedTensor, bool isLastInPermLeft,
                                                bool isLastOutPermLeft, const std::set<int8_t>& viceAllUbPerm)
{
    int64_t allSavedElems = 1;
    for (int8_t idx : allUbPerm_) {
        if (viceAllUbPerm.find(idx) == viceAllUbPerm.end()) {
            allSavedElems *= shapeInfo_.reducedInShape[idx];
        }
    }
    auto dim = shapeInfo_.dim;
    int64_t outSavedElems = CalcShapeSize(shapeInfo_.reducedOutShape, dim - outUbPerm_.cnt + 1, dim);
    int64_t inSavedElems = CalcShapeSize(shapeInfo_.reducedInShape, dim - inUbPerm_.cnt + 1, dim);
    // to save ub for gather index
    int64_t elemPerBlock = platInfo_.ubBlockSize / shapeInfo_.eleLenInBytes;
    int64_t maxOutCutAxisSize = elemInTensor / NUM_FOUR / elemPerBlock * elemPerBlock /
                                Ops::Base::CeilAlign(outSavedElems, elemPerBlock);
    int64_t maxCutAxisSize = elemInTensor / allSavedElems;

    if (isLastInPermLeft && isLastOutPermLeft) {
        if (inUbPerm_.perm[inUbPerm_.cnt - 1] != outUbPerm_.perm[outUbPerm_.cnt - 1]) {
            if (outUbPerm_.cnt + inUbPerm_.cnt == ubSplitInfo_.ubAxesCnt) {
                ubSplitInfo_.inUbCutAxisFactor = std::min(ubSplitInfo_.inUbCutAxisSize, sqrtedTensor / inSavedElems);
                ubSplitInfo_.outUbCutAxisFactor = std::min(ubSplitInfo_.outUbCutAxisSize, sqrtedTensor / outSavedElems);
            } else {
                int64_t comSavedElems = 1;
                for (int8_t idx = 0; idx < outUbPerm_.cnt - 1; ++idx) {
                    if (inUbPermSet_.find(outUbPerm_.perm[idx]) != inUbPermSet_.end()) {
                        comSavedElems *= shapeInfo_.reducedInShape[outUbPerm_.perm[idx]];
                    }
                }
                int64_t newSqrtedTensor = static_cast<int64_t>(
                    std::sqrt(elemInTensor / comSavedElems / elemPerBlock * elemPerBlock));
                int64_t inLeft = inSavedElems / comSavedElems;
                int64_t outLeft = outSavedElems / comSavedElems;
                ubSplitInfo_.inUbCutAxisFactor = std::min(ubSplitInfo_.inUbCutAxisSize, newSqrtedTensor / inLeft);
                ubSplitInfo_.outUbCutAxisFactor = std::min(ubSplitInfo_.outUbCutAxisSize, newSqrtedTensor / outLeft);
                AdjustUbCutAxisFactor(ubSplitInfo_.outUbCutAxisFactor, NUM_THREE, elemInTensor);
            }
        } else {
            ubSplitInfo_.inUbCutAxisFactor = std::min(std::min(ubSplitInfo_.inUbCutAxisSize, maxCutAxisSize),
                                                      maxOutCutAxisSize);
            AdjustUbCutAxisFactor(ubSplitInfo_.inUbCutAxisFactor, 0, elemInTensor);
            ubSplitInfo_.outUbCutAxisFactor = ubSplitInfo_.inUbCutAxisFactor;
        }
    } else if (!isLastInPermLeft && !isLastOutPermLeft) {
        ubSplitInfo_.inUbCutAxisFactor = ubSplitInfo_.inUbCutAxisSize;
        ubSplitInfo_.outUbCutAxisFactor = ubSplitInfo_.outUbCutAxisSize;
    } else {
        if (!isLastInPermLeft) {
            ubSplitInfo_.inUbCutAxisFactor = ubSplitInfo_.inUbCutAxisSize;
            ubSplitInfo_.outUbCutAxisFactor = std::min(std::min(ubSplitInfo_.outUbCutAxisSize, maxCutAxisSize),
                                                       maxOutCutAxisSize);
            AdjustUbCutAxisFactor(ubSplitInfo_.outUbCutAxisFactor, 1, elemInTensor);
        } else {
            ubSplitInfo_.outUbCutAxisFactor = ubSplitInfo_.outUbCutAxisSize;
            ubSplitInfo_.inUbCutAxisFactor = std::min(ubSplitInfo_.inUbCutAxisSize, maxCutAxisSize);
            AdjustUbCutAxisFactor(ubSplitInfo_.inUbCutAxisFactor, NUM_TWO, elemInTensor);
        }
    }
}

ge::graphStatus TransposeGatherTiling::CalcUbAxesInfo(const int64_t (&tmpInAxes)[MAX_TRANS_AXIS_NUM],
                                                      const int64_t (&tmpOutAxes)[MAX_TRANS_AXIS_NUM],
                                                      const int8_t (&tmpOutPerm)[MAX_TRANS_AXIS_NUM])
{
    int8_t inIdx = 0;
    int8_t outIdx = 0;
    for (int8_t j = 0; j < MAX_TRANS_AXIS_NUM; ++j) {
        if (tmpOutAxes[j] != 0) {
            ubSplitInfo_.outUbAxes[outIdx] = static_cast<int32_t>(tmpOutAxes[j]);
            // do like [5,2,3,0] -> [3,1,2,0]
            ubSplitInfo_.ubPerm[outIdx] = static_cast<int8_t>(
                std::distance(allUbPerm_.begin(), allUbPerm_.find(tmpOutPerm[j])));
            ++outIdx;
        }
        if (tmpInAxes[j] != 0) {
            ubSplitInfo_.inUbAxes[inIdx++] = static_cast<int32_t>(tmpInAxes[j]);
        }
    }

    int32_t totalSizeInUb = static_cast<int32_t>(shapeInfo_.eleLenInBytes);
    for (int8_t i = 0; i < ubSplitInfo_.ubAxesCnt; ++i) {
        totalSizeInUb *= ubSplitInfo_.inUbAxes[i];
    }
    int32_t indexStep = 1;
    for (int8_t i = ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] + 1; i < ubSplitInfo_.ubAxesCnt; ++i) {
        indexStep *= ubSplitInfo_.inUbAxes[i];
    }
    // MTE size must be >= gate
    if (totalSizeInUb < MTE_GATE || CheckBC(indexStep)) {
        return ge::GRAPH_FAILED;
    }

    for (int8_t k = 0; k < static_cast<int8_t>(allUbPerm_.size()); ++k) {
        if (ubSplitInfo_.ubPerm[k] == ubSplitInfo_.inUbInCutPos) {
            ubSplitInfo_.outUbInCutPos = k;
        }
        if (ubSplitInfo_.ubPerm[k] == ubSplitInfo_.inUbOutCutPos) {
            ubSplitInfo_.outUbOutCutPos = k;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeGatherTiling::CalcUbSplitInfo4Gather(int64_t elemInTensor, int64_t sqrtedTensor)
{
    int8_t tmpOutPerm[MAX_TRANS_AXIS_NUM] = {0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf};
    int64_t tmpInAxes[MAX_TRANS_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t tmpOutAxes[MAX_TRANS_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::set<int8_t> viceAllUbPerm(allUbPerm_);

    ubSplitInfo_.ubAxesCnt = static_cast<int8_t>(allUbPerm_.size());
    ubSplitInfo_.inUbCutAxisSize = shapeInfo_.reducedInShape[inUbPerm_.perm[inUbPerm_.cnt - 1]];
    ubSplitInfo_.inUbInCutPos = static_cast<int8_t>(
        std::distance(allUbPerm_.begin(), allUbPerm_.find(inUbPerm_.perm[inUbPerm_.cnt - 1])));
    // all axes can be move in ub except last
    for (int8_t i = 0; i < inUbPerm_.cnt - 1; ++i) {
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(), inUbPerm_.perm[i]);
        auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
        tmpOutPerm[idx] = inUbPerm_.perm[i];
        tmpOutAxes[idx] = shapeInfo_.reducedInShape[inUbPerm_.perm[i]];
        tmpInAxes[inUbPerm_.perm[i]] = shapeInfo_.reducedInShape[inUbPerm_.perm[i]];
        viceAllUbPerm.erase(inUbPerm_.perm[i]);
    }
    ubSplitInfo_.outUbCutAxisSize = shapeInfo_.reducedInShape[outUbPerm_.perm[outUbPerm_.cnt - 1]];
    ubSplitInfo_.inUbOutCutPos = static_cast<int8_t>(
        std::distance(allUbPerm_.begin(), allUbPerm_.find(outUbPerm_.perm[outUbPerm_.cnt - 1])));
    for (int8_t i = 0; i < outUbPerm_.cnt - 1; ++i) {
        if (viceAllUbPerm.find(outUbPerm_.perm[i]) != viceAllUbPerm.end()) {
            auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(), outUbPerm_.perm[i]);
            auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
            // pattern like: [6, 5, 4, 0xf, 0, 2, 1, 0xf]
            tmpOutPerm[idx] = outUbPerm_.perm[i];
            // pattern like: [10, 20, 50, 0, 5, 7, 2, 0]
            tmpOutAxes[idx] = shapeInfo_.reducedInShape[outUbPerm_.perm[i]];
            // pattern like: [0, 10, 50, 20, 5, 0, 7, 2]
            tmpInAxes[outUbPerm_.perm[i]] = shapeInfo_.reducedInShape[outUbPerm_.perm[i]];
            viceAllUbPerm.erase(outUbPerm_.perm[i]);
        }
    }

    bool isLastInPermLeft = viceAllUbPerm.find(inUbPerm_.perm[inUbPerm_.cnt - 1]) != viceAllUbPerm.end();
    bool isLastOutPermLeft = viceAllUbPerm.find(outUbPerm_.perm[outUbPerm_.cnt - 1]) != viceAllUbPerm.end();
    CalcUbAxisCutFactor(elemInTensor, sqrtedTensor, isLastInPermLeft, isLastOutPermLeft, viceAllUbPerm);

    if (isLastInPermLeft) {
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(),
                              inUbPerm_.perm[inUbPerm_.cnt - 1]);
        auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
        tmpOutPerm[idx] = inUbPerm_.perm[inUbPerm_.cnt - 1];
        tmpOutAxes[idx] = ubSplitInfo_.inUbCutAxisFactor;
        tmpInAxes[inUbPerm_.perm[inUbPerm_.cnt - 1]] = ubSplitInfo_.inUbCutAxisFactor;
        viceAllUbPerm.erase(inUbPerm_.perm[inUbPerm_.cnt - 1]);
    }
    if (isLastOutPermLeft) {
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(),
                              outUbPerm_.perm[outUbPerm_.cnt - 1]);
        auto idx = std::distance(shapeInfo_.reducedPerm.begin(), iter);
        tmpOutPerm[idx] = outUbPerm_.perm[outUbPerm_.cnt - 1];
        tmpOutAxes[idx] = ubSplitInfo_.outUbCutAxisFactor;
        tmpInAxes[outUbPerm_.perm[outUbPerm_.cnt - 1]] = ubSplitInfo_.outUbCutAxisFactor;
    }

    return CalcUbAxesInfo(tmpInAxes, tmpOutAxes, tmpOutPerm);
}

void TransposeGatherTiling::CalcUbSplitInfo4MTE()
{
    auto dim = shapeInfo_.dim;
    int8_t axisIdx = 0;

    if (outUbPerm_.perm[0] < inUbPerm_.perm[inUbPerm_.cnt - 1]) {
        // to make sure output last dim and move in cube are consecutive
        ubSplitInfo_.axis0InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, outUbPerm_.perm[0] + 1, dim);
        for (int8_t i = inUbPerm_.perm[inUbPerm_.cnt - 1] - 1; i >= 0; --i) {
            if (allUbPerm_.find(i) != allUbPerm_.end() && i != outUbPerm_.perm[0] && axisIdx == 0) {
                ubSplitInfo_.axis1InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
                ++axisIdx;
            } else if (allUbPerm_.find(i) != allUbPerm_.end() && i != outUbPerm_.perm[0] && axisIdx == 1) {
                ubSplitInfo_.axis2InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
            }
        }
    } else {
        for (int8_t i = inUbPerm_.perm[inUbPerm_.cnt - 1] - 1; i >= 0; --i) {
            if (allUbPerm_.find(i) != allUbPerm_.end() && axisIdx == 0) {
                ubSplitInfo_.axis0InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
                ++axisIdx;
            } else if (allUbPerm_.find(i) != allUbPerm_.end() && axisIdx == 1) {
                ubSplitInfo_.axis1InSrcStride = CalcShapeSize(shapeInfo_.reducedInShape, i + 1, dim);
            }
        }
    }

    axisIdx = 0;
    for (int8_t j = dim - outUbPerm_.cnt - 1; j >= 0; --j) {
        if (allUbPerm_.find(shapeInfo_.reducedPerm[j]) != allUbPerm_.end() && axisIdx == 0) {
            ubSplitInfo_.axis0OutDstStride = CalcShapeSize(shapeInfo_.reducedOutShape, j + 1, dim);
            ++axisIdx;
        } else if (allUbPerm_.find(shapeInfo_.reducedPerm[j]) != allUbPerm_.end() && axisIdx == 1) {
            ubSplitInfo_.axis1OutDstStride = CalcShapeSize(shapeInfo_.reducedOutShape, j + 1, dim);
            ++axisIdx;
        } else if (allUbPerm_.find(shapeInfo_.reducedPerm[j]) != allUbPerm_.end() && axisIdx > 1) {
            ubSplitInfo_.axis2OutDstStride = CalcShapeSize(shapeInfo_.reducedOutShape, j + 1, dim);
        }
    }
}

void TransposeGatherTiling::AdjustInUbAxesPosition()
{
    int8_t outLastDimInPos = ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1];
    int8_t axis0Gap = ubSplitInfo_.inUbInCutPos - 1 - outLastDimInPos;
    // only for brorrow axis case, to make output last dim to be the axis0 when move data in
    if (axis0Gap > 0) {
        for (int8_t i = 0; i < axis0Gap; ++i) {
            ubSplitInfo_.inUbAxes[outLastDimInPos + i] = ubSplitInfo_.inUbAxes[outLastDimInPos + i + 1];
        }
        ubSplitInfo_.inUbAxes[outLastDimInPos + axis0Gap] = ubSplitInfo_.outUbAxes[ubSplitInfo_.ubAxesCnt - 1];
        if (outLastDimInPos < ubSplitInfo_.inUbOutCutPos && ubSplitInfo_.inUbOutCutPos < ubSplitInfo_.inUbInCutPos) {
            ubSplitInfo_.inUbOutCutPos -= 1;
        }

        if (outUbPerm_.cnt == NUM_TWO) {
            ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 0;
            ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = 1;
        } else if (outUbPerm_.cnt == NUM_THREE) {
            /*           no overlap: 2 1 0 -> 1 0 2
             *                       2 0 1 -> 1 0 2
             *                       0 2 1 -> 0 1 2
             *                       1 2 0 -> 0 1 2
             *              overlap: x 1 0 -> x 0 1
             *                       1 x 0 -> 0 x 1
             */
            if (inUbPerm_.cnt + outUbPerm_.cnt == ubSplitInfo_.ubAxesCnt) {
                if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] == NUM_TWO) {
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] = 1;
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 0;
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = NUM_TWO;
                } else if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] == NUM_TWO) {
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] = 0;
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 1;
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = NUM_TWO;
                }
            } else {
                if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] >= NUM_TWO) {
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] = 0;
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = 1;
                } else if (ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_TWO] >= NUM_TWO) {
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - NUM_THREE] = 0;
                    ubSplitInfo_.ubPerm[ubSplitInfo_.ubAxesCnt - 1] = 1;
                }
            }
        }
    }
}

bool TransposeGatherTiling::CheckBC(int32_t steps)
{
    int32_t bytesPerSubBank = 8;
    int32_t bytesPerBank = 128;
    int32_t stepBytes = steps * shapeInfo_.eleLenInBytes;
    int32_t stepBytesAlign = Ops::Base::CeilAlign(stepBytes, bytesPerSubBank);
    return (stepBytesAlign % bytesPerBank / bytesPerSubBank % NUM_TWO == 0);
}

int64_t TransposeGatherTiling::CalcSqrtedTensor(int64_t elemInTensor)
{
    int64_t elemPerBlock = platInfo_.ubBlockSize / shapeInfo_.eleLenInBytes;
    int64_t sqrtedTensor = static_cast<int64_t>(std::sqrt(elemInTensor)) / elemPerBlock * elemPerBlock;
    if (sqrtedTensor * shapeInfo_.eleLenInBytes > platInfo_.cacheLineSize) {
        sqrtedTensor = (sqrtedTensor * shapeInfo_.eleLenInBytes / platInfo_.cacheLineSize * platInfo_.cacheLineSize /
                        shapeInfo_.eleLenInBytes);
    }
    int32_t bytesPerSubBank = 8;
    int64_t lastInDim = shapeInfo_.reducedInShape[shapeInfo_.dim - 1];
    if (lastInDim > sqrtedTensor && CheckBC(static_cast<int32_t>(sqrtedTensor))) {
        sqrtedTensor -= (bytesPerSubBank / shapeInfo_.eleLenInBytes);
    }
    return sqrtedTensor;
}

ge::graphStatus TransposeGatherTiling::CalcUbSplitInfo()
{
    int64_t elemInTensor = static_cast<int64_t>(dataTensorSize_ / shapeInfo_.eleLenInBytes);
    int64_t sqrtedTensor = CalcSqrtedTensor(elemInTensor);
    CalcInUbPerm(sqrtedTensor);
    CalcOutUbPerm(sqrtedTensor);
    OP_CHECK_IF(CalcUbSplitInfo4Gather(elemInTensor, sqrtedTensor) != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "MTE size is too small!"), return ge::GRAPH_FAILED);
    CalcUbSplitInfo4MTE();
    AdjustInUbAxesPosition();
    OP_LOGD(context_->GetNodeName(), "UB tiling is done!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeGatherTiling::CalcBlockSplitInfo()
{
    int8_t dim = shapeInfo_.dim;
    int64_t axisFactor = 1;
    int64_t totalElems = 1;
    for (int8_t i = 0; i < dim; ++i) {
        if (allUbPerm_.find(i) == allUbPerm_.end()) {
            axisFactor = 1;
        } else if (i == inUbPerm_.perm[inUbPerm_.cnt - 1] &&
                   ubSplitInfo_.inUbCutAxisSize != ubSplitInfo_.inUbCutAxisFactor) {
            axisFactor = ubSplitInfo_.inUbCutAxisFactor;
            if (ubSplitInfo_.inUbCutAxisSize % ubSplitInfo_.inUbCutAxisFactor != 0) {
                blkSplitInfo_.blkInUbCutPos = blkSplitInfo_.blkAxesCnt;
            }
        } else if (i == outUbPerm_.perm[outUbPerm_.cnt - 1] &&
                   ubSplitInfo_.outUbCutAxisSize != ubSplitInfo_.outUbCutAxisFactor) {
            axisFactor = ubSplitInfo_.outUbCutAxisFactor;
            if (ubSplitInfo_.outUbCutAxisSize % ubSplitInfo_.outUbCutAxisFactor != 0) {
                blkSplitInfo_.blkOutUbCutPos = blkSplitInfo_.blkAxesCnt;
            }
        } else {
            continue;
        }
        int64_t axisLpSize = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[i], axisFactor);
        blkSplitInfo_.blkAxes[blkSplitInfo_.blkAxesCnt] = axisLpSize;
        blkSplitInfo_.blkAxesInAOffset[blkSplitInfo_.blkAxesCnt] = CalcShapeSize(shapeInfo_.reducedInShape, i + 1,
                                                                                 dim) *
                                                                   axisFactor;
        auto iter = std::find(shapeInfo_.reducedPerm.begin(), shapeInfo_.reducedPerm.end(), i);
        int8_t gap = static_cast<int8_t>(std::distance(shapeInfo_.reducedPerm.begin(), iter));
        blkSplitInfo_.blkAxesOutAOffset[blkSplitInfo_.blkAxesCnt] = CalcShapeSize(shapeInfo_.reducedOutShape, gap + 1,
                                                                                  dim) *
                                                                    axisFactor;
        ++blkSplitInfo_.blkAxesCnt;
        totalElems *= axisLpSize;
    }

    blkSplitInfo_.usedCoreCnt = Ops::Base::CeilDiv(totalElems, Ops::Base::CeilDiv(totalElems, platInfo_.coreNum));
    if (blkSplitInfo_.usedCoreCnt < static_cast<uint32_t>(platInfo_.coreNum / NUM_TWO)) {
        return ge::GRAPH_FAILED;
    }
    blkSplitInfo_.blkFactor = Ops::Base::CeilDiv(totalElems, static_cast<int64_t>(blkSplitInfo_.usedCoreCnt));
    blkSplitInfo_.blkTailFactor = totalElems - (blkSplitInfo_.usedCoreCnt - 1) * blkSplitInfo_.blkFactor;
    OP_LOGD(context_->GetNodeName(), "Block tiling is done!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeGatherTiling::SetTilingKeyAndCore()
{
    OP_CHECK_IF(context_->SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set tiling key is failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetBlockDim(blkSplitInfo_.usedCoreCnt) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set used core size is failed!"), return ge::GRAPH_FAILED);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

void TransposeGatherTiling::WriteTilingData()
{
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_dataTensorSize(dataTensorSize_);
    tilingData_.set_indexTensorSize(indexTensorSize_);
    tilingData_.set_usedCoreCnt(blkSplitInfo_.usedCoreCnt);
    tilingData_.set_blkAxesCnt(blkSplitInfo_.blkAxesCnt);
    tilingData_.set_blkInUbCutPos(blkSplitInfo_.blkInUbCutPos);
    tilingData_.set_blkOutUbCutPos(blkSplitInfo_.blkOutUbCutPos);
    tilingData_.set_ubAxesCnt(ubSplitInfo_.ubAxesCnt);
    tilingData_.set_inUbInCutPos(ubSplitInfo_.inUbInCutPos);
    tilingData_.set_inUbOutCutPos(ubSplitInfo_.inUbOutCutPos);
    tilingData_.set_outUbInCutPos(ubSplitInfo_.outUbInCutPos);
    tilingData_.set_outUbOutCutPos(ubSplitInfo_.outUbOutCutPos);
    tilingData_.set_blkFactor(blkSplitInfo_.blkFactor);
    tilingData_.set_blkTailFactor(blkSplitInfo_.blkTailFactor);
    tilingData_.set_inUbCutAxisSize(ubSplitInfo_.inUbCutAxisSize);
    tilingData_.set_outUbCutAxisSize(ubSplitInfo_.outUbCutAxisSize);
    tilingData_.set_inUbCutAxisFactor(ubSplitInfo_.inUbCutAxisFactor);
    tilingData_.set_outUbCutAxisFactor(ubSplitInfo_.outUbCutAxisFactor);
    tilingData_.set_axis0InSrcStride(ubSplitInfo_.axis0InSrcStride);
    tilingData_.set_axis1InSrcStride(ubSplitInfo_.axis1InSrcStride);
    tilingData_.set_axis2InSrcStride(ubSplitInfo_.axis2InSrcStride);
    tilingData_.set_axis0OutDstStride(ubSplitInfo_.axis0OutDstStride);
    tilingData_.set_axis1OutDstStride(ubSplitInfo_.axis1OutDstStride);
    tilingData_.set_axis2OutDstStride(ubSplitInfo_.axis2OutDstStride);

    tilingData_.set_blkAxes(blkSplitInfo_.blkAxes);
    tilingData_.set_blkAxesInAOffset(blkSplitInfo_.blkAxesInAOffset);
    tilingData_.set_blkAxesOutAOffset(blkSplitInfo_.blkAxesOutAOffset);
    tilingData_.set_inUbAxes(ubSplitInfo_.inUbAxes);
    tilingData_.set_outUbAxes(ubSplitInfo_.outUbAxes);
    tilingData_.set_ubPerm(ubSplitInfo_.ubPerm);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

std::string TransposeGatherTiling::PrintTilingData()
{
    std::string tdStr;
    tdStr += std::to_string(static_cast<int64_t>(tilingKey_)) + ",";
    tdStr += std::to_string(static_cast<int32_t>(dataTensorSize_)) + ",";
    tdStr += std::to_string(static_cast<int32_t>(indexTensorSize_)) + ",";
    tdStr += std::to_string(static_cast<int32_t>(blkSplitInfo_.usedCoreCnt)) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkAxesCnt) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkInUbCutPos) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkOutUbCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.ubAxesCnt) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbInCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbOutCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbInCutPos) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbOutCutPos) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkFactor) + ",";
    tdStr += std::to_string(blkSplitInfo_.blkTailFactor) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbCutAxisSize) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbCutAxisSize) + ",";
    tdStr += std::to_string(ubSplitInfo_.inUbCutAxisFactor) + ",";
    tdStr += std::to_string(ubSplitInfo_.outUbCutAxisFactor) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis0InSrcStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis1InSrcStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis2InSrcStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis0OutDstStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis1OutDstStride) + ",";
    tdStr += std::to_string(ubSplitInfo_.axis2OutDstStride) + ",";
    tdStr += "block axes:";
    for (int8_t i = 0; i < MAX_TRANS_AXIS_NUM; ++i) {
        tdStr += std::to_string(blkSplitInfo_.blkAxes[i]) + " ";
    }
    tdStr += ",block axes in offset:";
    for (int8_t i = 0; i < MAX_TRANS_AXIS_NUM; ++i) {
        tdStr += std::to_string(blkSplitInfo_.blkAxesInAOffset[i]) + " ";
    }
    tdStr += ",block axes out offset:";
    for (int8_t i = 0; i < MAX_TRANS_AXIS_NUM; ++i) {
        tdStr += std::to_string(blkSplitInfo_.blkAxesOutAOffset[i]) + " ";
    }
    tdStr += ",ub in axes:";
    for (int8_t i = 0; i < UB_MAX_DIM_NUM; ++i) {
        tdStr += std::to_string(ubSplitInfo_.inUbAxes[i]) + " ";
    }
    tdStr += ",ub out axes:";
    for (int8_t i = 0; i < UB_MAX_DIM_NUM; ++i) {
        tdStr += std::to_string(ubSplitInfo_.outUbAxes[i]) + " ";
    }
    tdStr += ",ub perm:";
    for (int8_t i = 0; i < UB_MAX_DIM_NUM; ++i) {
        tdStr += std::to_string(ubSplitInfo_.ubPerm[i]) + " ";
    }
    return tdStr;
}

ge::graphStatus TransposeGatherTiling::DoTiling()
{
    CalcTensorSize();
    OP_CHECK_IF(CalcUbSplitInfo() != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "Stop to run gather tiling, mte size is too small!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcBlockSplitInfo() != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "Stop to run gather tiling, block count is too small!"),
                return ge::GRAPH_FAILED);
    WriteTilingData();
    OP_LOGI(context_->GetNodeName(), "The tiling data is: %s", PrintTilingData().c_str());
    return SetTilingKeyAndCore();
}

} // namespace TransWithGather
} // namespace optiling
