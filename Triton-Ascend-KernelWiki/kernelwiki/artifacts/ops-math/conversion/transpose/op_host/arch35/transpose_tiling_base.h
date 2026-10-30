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
 * \file transpose_tiling_base.h
 * \brief transpose
 */
#ifndef __TRANSPOSE_RT_V2_H__
#define __TRANSPOSE_RT_V2_H__

#include <array>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <memory>

namespace optiling {
#define TRANSPOSE_MAX_AXIS_NUM 8

struct ShapeInfo {
    int64_t id;
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    std::vector<int64_t> perm;
    std::vector<int64_t> reducedInShape;
    std::vector<int64_t> reducedOutShape;
    std::vector<int64_t> reducedPerm;

    int64_t inShapeSize;
    int64_t outShapeSize;
    int64_t permSize;

    int64_t origDim;
    int64_t dim;
    int64_t totalVolumeActual;
    int64_t identical;
    int64_t lastAxisLen;
    int64_t lastAxisBurstLen;
    int64_t elePerBlock;
    int64_t eleLenInBytes;
    int64_t alignElement; // unit: element number padding.  eg. dtype=float, axis[0]= 11, alignElement=8-(11-8)%8=5
    bool isLastAxisTranspose;
    bool isLastAxisHuge;
    bool isLastTwoAlignedAndTrans;

    ShapeInfo()
    {
        inShape.resize(TRANSPOSE_MAX_AXIS_NUM);
        outShape.resize(TRANSPOSE_MAX_AXIS_NUM);
        perm.resize(TRANSPOSE_MAX_AXIS_NUM);
        reducedInShape.resize(TRANSPOSE_MAX_AXIS_NUM);
        reducedOutShape.resize(TRANSPOSE_MAX_AXIS_NUM);
        reducedPerm.resize(TRANSPOSE_MAX_AXIS_NUM);
        Reset();
    }

    void Reset()
    {
        id = 0;
        inShapeSize = 0;
        outShapeSize = 0;
        permSize = 0;
        origDim = 0;
        dim = 0;
        totalVolumeActual = 0;
        identical = 0;
        lastAxisLen = 0;
        lastAxisBurstLen = 0;
        elePerBlock = 8;
        eleLenInBytes = 0;
        alignElement = 0;
        isLastAxisTranspose = false;
        isLastAxisHuge = false;
        isLastTwoAlignedAndTrans = false;
    }
};

struct TransposeCompilerInfo {
    int64_t coreNum;
    int64_t ubSize; // unit: block

    TransposeCompilerInfo() : coreNum(0), ubSize(0) {}
};

void RemoveAxisV2(ShapeInfo& shapeInfo);

void MergeAxisV2(ShapeInfo& shapeInfo);

} // namespace optiling

#endif // __TRANSPOSE_RT_V2_H__
