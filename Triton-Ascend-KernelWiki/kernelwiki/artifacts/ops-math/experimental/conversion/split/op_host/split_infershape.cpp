/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file split_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr size_t GMM_INDEX_IN_X = 0;
constexpr size_t GMM_INDEX_IN_INDICES = 1;
constexpr size_t GMM_INDEX_OUT_Y = 0;

constexpr size_t GMM_MIN_FM_DIM = 1;
constexpr size_t GMM_NORM_AXIS = -1;
constexpr size_t ATTRPOS0 = 0;
constexpr size_t ATTRPOS1 = 1;
constexpr int32_t INDICES_LIMIT = 10;

struct SplitSetOutputParams {
    int64_t indicesOrSections[INDICES_LIMIT];
    int64_t axis;
};
static ge::graphStatus UpdateShapeY(gert::InferShapeContext* context, size_t idxY, const std::vector<int64_t>& yDims)
{
    gert::Shape* yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus ComputeSplitGroups(
    gert::InferShapeContext* context, int64_t xAxisLen, int64_t* indicesOrSections, bool& isUniformSplit,
    int64_t& groupNum, int64_t& splitSize)
{
    isUniformSplit = true;
    groupNum = 0;
    splitSize = 0;

    for (int i = 1; (indicesOrSections[i] != 0) && (i < INDICES_LIMIT); ++i) {
        if (indicesOrSections[i] != 0) {
            isUniformSplit = false;
            break;
        }
    }

    if (isUniformSplit) {
        groupNum = indicesOrSections[0];
        if (groupNum == 0) {
            OP_LOGE(context->GetNodeName(), "groupNum must be greater than 0.");
            return GRAPH_FAILED;
        }
        if (xAxisLen % groupNum != 0) {
            OP_LOGE(
                context->GetNodeName(), "When splitting uniformly, xAxisLen %ld must be divisible by groupNum %ld.",
                xAxisLen, groupNum);
            return GRAPH_FAILED;
        }
        splitSize = xAxisLen / groupNum;
    } else {
        int numIndices = 0;
        for (int i = 0; indicesOrSections[i] != 0; ++i) {
            if (indicesOrSections[i] > xAxisLen) {
                OP_LOGE(context->GetNodeName(), "Index %ld exceeds xAxisLen %ld.", indicesOrSections[i], xAxisLen);
                return GRAPH_FAILED;
            }
            numIndices++;
        }
        groupNum = numIndices + 1;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateMultipleShapeYNorm(gert::InferShapeContext* context, int64_t* indicesOrSections)
{
    const gert::Shape* xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t xAxisLen = 1;
    for (size_t dim = 0; dim < xShape->GetDimNum(); ++dim) {
        xAxisLen *= xShape->GetDim(dim);
    }
    int64_t groupNum = 0;
    bool isUniformSplit = true;
    int64_t splitSize = 0;

    OP_CHECK_IF(
        ComputeSplitGroups(context, xAxisLen, indicesOrSections, isUniformSplit, groupNum, splitSize) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ComputeSplitGroups failed."), return GRAPH_FAILED);

    int64_t preOffset = 0;
    for (int idx = 0; idx < groupNum; ++idx) {
        int64_t currentSize = 0;
        if (isUniformSplit) {
            currentSize = splitSize;
        } else {
            if (idx == groupNum - 1) {
                currentSize = xAxisLen - indicesOrSections[idx - 1];
            } else {
                currentSize = indicesOrSections[idx] - preOffset;
                preOffset = indicesOrSections[idx];
            }
        }
        std::vector<int64_t> yDims;
        yDims.push_back(currentSize);
        OP_CHECK_IF(
            UpdateShapeY(context, GMM_INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to update shape of y[%d].", idx), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateMultipleShapeY(gert::InferShapeContext* context, int64_t* indicesOrSections, int64_t axis)
{
    const gert::Shape* xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t xAxisLen = xShape->GetDim(axis);
    int64_t groupNum = 0;
    bool isUniformSplit = true;
    int64_t splitSize = 0;

    OP_CHECK_IF(
        ComputeSplitGroups(context, xAxisLen, indicesOrSections, isUniformSplit, groupNum, splitSize) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ComputeSplitGroups failed."), return GRAPH_FAILED);

    int64_t preOffset = 0;
    for (int idx = 0; idx < groupNum; ++idx) {
        int64_t currentSize = 0;
        if (isUniformSplit) {
            currentSize = splitSize;
        } else {
            if (idx == groupNum - 1) {
                currentSize = xAxisLen - indicesOrSections[idx - 1];
            } else {
                currentSize = indicesOrSections[idx] - preOffset;
                preOffset = indicesOrSections[idx];
            }
        }
        std::vector<int64_t> yDims;
        for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
            if (i == static_cast<size_t>(axis)) {
                yDims.push_back(currentSize);
            } else {
                yDims.push_back(xShape->GetDim(i));
            }
        }
        OP_CHECK_IF(
            UpdateShapeY(context, GMM_INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to update shape of y[%d].", idx), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus XSingleYSeparatedNorm(gert::InferShapeContext* context, int64_t* indicesOrSections)
{
    OP_CHECK_IF(
        UpdateMultipleShapeYNorm(context, indicesOrSections) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
static ge::graphStatus XSingleYSeparated(gert::InferShapeContext* context, int64_t* indicesOrSections, int64_t axis)
{
    OP_CHECK_IF(
        UpdateMultipleShapeY(context, indicesOrSections, axis) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
static ge::graphStatus SplitSetOutputShape(gert::InferShapeContext* context, const SplitSetOutputParams& outputParams)
{
    int64_t indicesOrSections[INDICES_LIMIT];
    for (int i = 0; i < INDICES_LIMIT; i++) {
        indicesOrSections[i] = outputParams.indicesOrSections[i];
    }
    int64_t axis = outputParams.axis;
    if (axis == (int64_t)GMM_NORM_AXIS) {
        OP_CHECK_IF(
            XSingleYSeparatedNorm(context, indicesOrSections) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            XSingleYSeparated(context, indicesOrSections, axis) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferShapeSplit(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    int64_t axis = -1; // 默认值
    int64_t* src_indices = nullptr;
    if (attrs) {
        const int64_t* src_indices_const = nullptr;
        if (attrs->GetInt(ATTRPOS0) != nullptr) {
            src_indices_const = attrs->GetInt(ATTRPOS0);
        }
        src_indices = const_cast<int64_t*>(src_indices_const);
        if (attrs->GetInt(ATTRPOS1) != nullptr) {
            axis = *(attrs->GetInt(ATTRPOS1));
        }
    }
    SplitSetOutputParams outputParams;
    uint32_t len = sizeof(outputParams.indicesOrSections) / sizeof(outputParams.indicesOrSections[0]);
    for (uint32_t i = 0; i < len; i++) {
        outputParams.indicesOrSections[i] = src_indices[i];
    }
    outputParams.axis = axis;

    OP_CHECK_IF(
        SplitSetOutputShape(context, outputParams) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "SplitSetOutputShape failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Split).InferShape(InferShapeSplit);
} // namespace ops