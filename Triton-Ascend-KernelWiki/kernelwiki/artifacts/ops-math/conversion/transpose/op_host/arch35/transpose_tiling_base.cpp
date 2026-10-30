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
 * \file transpose_tiling_base.cpp
 * \brief transpose
 */
#include "transpose_tiling_arch35.h"
#include "transpose_tiling_base.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <climits>

using namespace std;
using namespace gert;

namespace optiling {

static void DuplicateArray(const int64_t* src, vector<int64_t>& dst, int64_t len)
{
    for (int64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static void DuplicateArray(const vector<int64_t>& src, vector<int64_t>& dst, int64_t len)
{
    for (int64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static void DuplicateArray(const vector<int64_t>& src, int64_t* dst, int64_t len)
{
    for (int64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

static bool IsAllOne(const ShapeInfo& shapeInfo)
{
    return std::all_of(shapeInfo.inShape.begin(), shapeInfo.inShape.begin() + shapeInfo.dim,
                       [](const int64_t& item) { return item == 1; });
}

static int DecreaseCompare(const void* a, const void* b) { return (*(int64_t*)b - *(int64_t*)a); }

static void CalcOutShape(ShapeInfo& shapeInfo)
{
    const vector<int64_t>& inShape = shapeInfo.reducedInShape;
    const vector<int64_t>& perm = shapeInfo.reducedPerm;
    vector<int64_t>& outShape = shapeInfo.reducedOutShape;
    for (int64_t i = 0; i < shapeInfo.dim; i++) {
        outShape[i] = inShape[perm[i]];
    }
}

void RemoveAxisV2(ShapeInfo& shapeInfo)
{
    int64_t dim = shapeInfo.dim;
    if (dim == 1) {
        DuplicateArray(shapeInfo.inShape, shapeInfo.reducedInShape, dim);
        DuplicateArray(shapeInfo.perm, shapeInfo.reducedPerm, dim);
        DuplicateArray(shapeInfo.outShape, shapeInfo.reducedOutShape, dim);
        return;
    }

    if (IsAllOne(shapeInfo)) {
        shapeInfo.reducedInShape[0] = 1;
        shapeInfo.reducedPerm[0] = 0;
        shapeInfo.reducedOutShape[0] = 1;
        shapeInfo.dim = 1;
        return;
    }

    vector<int64_t>& shape = shapeInfo.reducedInShape;
    int64_t delPerm[TRANSPOSE_MAX_AXIS_NUM];
    int64_t newPerm[TRANSPOSE_MAX_AXIS_NUM];
    int64_t shapeSize = 0;
    int64_t delPermSize = 0;
    int64_t newPermSize = 0;

    for (int64_t i = 0; i < dim; i++) {
        if (shapeInfo.inShape[i] != 1) {
            shape[shapeSize++] = shapeInfo.inShape[i];
        } else {
            for (int64_t j = 0; j < dim; j++) {
                if (shapeInfo.perm[j] == i) {
                    delPerm[delPermSize++] = shapeInfo.perm[j];
                }
            }
        }
    }

    qsort(reinterpret_cast<void*>(&delPerm[0]), delPermSize, sizeof(int64_t), DecreaseCompare);

    for (int64_t i = 0; i < dim; i++) {
        bool delFlag = false;
        for (int64_t j = 0; j < delPermSize; j++) {
            if (shapeInfo.perm[i] == delPerm[j]) {
                delFlag = true;
            }
        }
        if (!delFlag) {
            newPerm[newPermSize++] = shapeInfo.perm[i];
        }
    }

    for (int64_t i = 0; i < delPermSize; i++) {
        for (int64_t j = 0; j < newPermSize; j++) {
            if (newPerm[j] > delPerm[i]) {
                newPerm[j] = newPerm[j] - 1;
            }
        }
    }

    DuplicateArray(newPerm, shapeInfo.reducedPerm, newPermSize);
    shapeInfo.dim = newPermSize;
    CalcOutShape(shapeInfo);
}

void MergeAxisV2(ShapeInfo& shapeInfo)
{
    int64_t dim = shapeInfo.dim;
    if (dim == 1) {
        return;
    }
    int64_t perm[TRANSPOSE_MAX_AXIS_NUM];
    int64_t shape[TRANSPOSE_MAX_AXIS_NUM];
    int64_t newPerm[TRANSPOSE_MAX_AXIS_NUM];
    int64_t newShape[TRANSPOSE_MAX_AXIS_NUM];
    int64_t newDimPosition[TRANSPOSE_MAX_AXIS_NUM];
    int64_t mergedShape[TRANSPOSE_MAX_AXIS_NUM] = {0};
    DuplicateArray(shapeInfo.reducedPerm, perm, dim);
    DuplicateArray(shapeInfo.reducedInShape, shape, dim);
    for (int i = 0; i < TRANSPOSE_MAX_AXIS_NUM; i++) {
        newDimPosition[i] = -1;
    }

    int64_t curHead = shapeInfo.reducedPerm[0];
    newDimPosition[curHead] = 0;
    mergedShape[0] = shape[curHead];
    int dimIndex = 0;
    for (int permIndex = 1; permIndex < dim; ++permIndex) {
        // If two indices in permutation are consecutive numbers, combine their dimensions.
        if (curHead + 1 == perm[permIndex]) {
            curHead = perm[permIndex];
            mergedShape[dimIndex] *= shape[curHead];
        } else {
            // Else start a new dimension.
            curHead = perm[permIndex];
            dimIndex++;
            newDimPosition[curHead] = dimIndex;
            mergedShape[dimIndex] = shape[curHead];
        }
    }

    shapeInfo.dim = dimIndex + 1;

    dimIndex = 0;
    for (int i = 0; i < dim; i++) {
        if (newDimPosition[i] >= 0) {
            newDimPosition[dimIndex++] = newDimPosition[i];
        }
    }

    // Compact the new permutations and dimension sizes.
    dimIndex = 0;
    for (int64_t i = 0; i < dim; ++i) {
        if (newDimPosition[i] >= 0) {
            int64_t newPermIndex = newDimPosition[i];
            for (int64_t j = 0; j < dim; j++) {
                if (newDimPosition[j] == i) {
                    newPerm[dimIndex] = j;
                    break;
                }
            }
            newShape[dimIndex] = mergedShape[newPermIndex];
            dimIndex++;
        }
    }

    DuplicateArray(newShape, shapeInfo.reducedInShape, dimIndex);
    DuplicateArray(newPerm, shapeInfo.reducedPerm, dimIndex);
    shapeInfo.lastAxisLen = shapeInfo.reducedInShape[shapeInfo.dim - 1];
    shapeInfo.lastAxisBurstLen = static_cast<int64_t>(ceil(shapeInfo.lastAxisLen * 1.0 / shapeInfo.elePerBlock));
    CalcOutShape(shapeInfo);
}

} // namespace optiling
