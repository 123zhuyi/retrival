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
 * \file infershape_reduce_util.h
 * \brief reduce infershape2.0 util
 */
#ifndef INFERSHAPE_REDUCE_UTIL_H
#define INFERSHAPE_REDUCE_UTIL_H

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/op_host/util/shape_util.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/opbase_export.h"

using namespace Ops::Base;
namespace Ops {
namespace Math {

template <typename T1, typename T2>
bool CheckAxisBounds(const T1 dimNum, const T2 axis)
{
    if (dimNum == 0) {
        return axis == 0;
    }
    const int64_t minimumNum = static_cast<int64_t>(dimNum) * (-1);
    const int64_t maximumNum = static_cast<int64_t>(dimNum) - 1;

    return static_cast<int64_t>(axis) >= minimumNum && static_cast<int64_t>(axis) <= maximumNum;
}

static ge::graphStatus InferShape4ReduceWithoutAxes(const gert::Shape* inShape, gert::Shape* outShape, bool keepDims,
                                                    int32_t axesDim0)
{
    auto dimNum = inShape->GetDimNum();
    if (dimNum == 0) {
        outShape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    if (keepDims) {
        outShape->SetDimNum(dimNum);
        for (size_t i = 0; i < dimNum; i++) {
            int64_t dim = inShape->GetDim(i);
            // dim == 1: 归约也是1，保持1；dim != 1: 不确定是否被归约，输出-1
            outShape->SetDim(i, (dim == 1) ? 1 : -1);
        }
        return ge::GRAPH_SUCCESS;
    }

    // 归约 K 个轴，输出 dimNum = N - K，不确定哪些轴→全部-1
    outShape->SetDimNum(dimNum - axesDim0);
    for (size_t i = 0; i < dimNum - axesDim0; i++) {
        outShape->SetDim(i, -1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ProcessEmptyAxesWithNoopWithEmptyAxes(gert::InferShapeContext* context,
                                                             const gert::Shape* inShape, gert::Shape* outShape,
                                                             bool keepDims, bool noopWithEmptyAxes)
{
    if (noopWithEmptyAxes) {
        *outShape = *inShape;
        OP_LOGD(context->GetNodeName(),
                "axes is empty tensor and noopWithEmptyAxes is true, set output shape = input shape");
    } else {
        if (keepDims) {
            auto dimNum = inShape->GetDimNum();
            if (dimNum == 0) {
                outShape->SetDimNum(0);
            } else {
                *outShape = *inShape;
                for (size_t i = 0; i < dimNum; i++) {
                    outShape->SetDim(i, 1);
                }
            }
            OP_LOGD(context->GetNodeName(),
                    "axes is empty tensor, noopWithEmptyAxes is false and keepDims is true, reduce all dims to 1");
        } else {
            outShape->SetDimNum(0);
            OP_LOGD(context->GetNodeName(),
                    "axes is empty tensor, noopWithEmptyAxes is false and keepDims is false, output is scalar");
        }
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDimsWithKeepDims(const gert::Shape* xShape, const T* axesDims, int32_t axesShapeSize,
                                       gert::Shape* outputShape)
{
    T dimNum = xShape->GetDimNum();
    const bool isScalar = xShape->GetDimNum() == 0;
    dimNum = isScalar ? 1 : dimNum;
    *outputShape = *xShape;
    for (int32_t i = 0; i < axesShapeSize; i++) {
        OP_CHECK_IF((!CheckAxisBounds<T, T>(dimNum, axesDims[i])), OP_LOGE("reduce", "axesDims is invalid"),
                    return ge::GRAPH_FAILED);
        if (isScalar) {
            // no need to update output shape, when input is scalar
            continue;
        }
        T dim = axesDims[i] < 0 ? axesDims[i] + dimNum : axesDims[i];
        outputShape->SetDim(dim, 1);
    }
    OP_LOGD("ReduceDimsWithKeepDims", "after reduce output shape is %s.", ToString(*outputShape).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDimsWithoutKeepDims(const gert::Shape* xShape, const T* axesDims, int32_t axesShapeSize,
                                          gert::Shape* outputShape)
{
    T dimNum = xShape->GetDimNum();
    outputShape->SetDimNum(0);
    for (T j = 0; j < dimNum; j++) {
        bool reduceFlag = false;
        for (int32_t i = 0; i < axesShapeSize; i++) {
            OP_CHECK_IF((!CheckAxisBounds<T, T>(dimNum, axesDims[i])), OP_LOGE("reduce", "axesDims is invalid"),
                        return ge::GRAPH_FAILED);
            T dim = axesDims[i] < 0 ? axesDims[i] + dimNum : axesDims[i];
            if (dim == j) {
                reduceFlag = true;
                break;
            }
        }
        if (!reduceFlag) {
            outputShape->AppendDim(xShape->GetDim(j));
        }
    }

    OP_LOGD("ReduceDimsWithoutKeepDims", "after reduce output shape is %s.", ToString(*outputShape).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDims(const gert::Shape* xShape, const gert::Tensor* axesTensor, int32_t axesShapeSize,
                           const bool keepDims, gert::Shape* outputShape)
{
    const T* axesDims = axesTensor->GetData<T>();
    if (keepDims) {
        return ReduceDimsWithKeepDims<T>(xShape, axesDims, axesShapeSize, outputShape);
    }
    return ReduceDimsWithoutKeepDims<T>(xShape, axesDims, axesShapeSize, outputShape);
}

static ge::graphStatus DoInferShapeReduce(gert::InferShapeContext* context, const gert::Shape* inShape,
                                          gert::Shape* outShape, const gert::Tensor* axesTensor, bool keepDims,
                                          bool noopWithEmptyAxes)
{
    auto axesShape = axesTensor->GetStorageShape();
    auto axesDimNum = axesShape.GetDimNum();
    auto axesShapeSize = axesShape.GetShapeSize();
    auto axesDim0 = (axesDimNum == 1) ? axesShape.GetDim(0) : ((axesDimNum == 0) ? 1 : 0);

    OP_LOGI(context->GetNodeName(),
            "axesShape = %s, axesDimNum = %zu, axesDim0 = %" PRId64 ", axesShapeSize = %" PRId64,
            ToString(axesShape).c_str(), axesDimNum, axesDim0, axesShapeSize);

    // --- 场景1: axes为空tensor，走noop/allReduce ---
    if ((axesDimNum == 1 && axesDim0 == 0) || axesShapeSize == 0) {
        auto ret = ProcessEmptyAxesWithNoopWithEmptyAxes(context, inShape, outShape, keepDims, noopWithEmptyAxes);
        OP_LOGI(context->GetNodeName(), "outShape = %s", ToString(*outShape).c_str());
        return ret;
    }

    auto axesDtype = axesTensor->GetDataType();
    // --- 场景2: axes非静态常量，保守推导 ---
    if (axesDtype == ge::DT_INT32) {
        const int32_t* axesData = axesTensor->GetData<int32_t>();
        if (axesData == nullptr) {
            OP_LOGW(context->GetNodeName(), "axes is not const, do conservative infer");
            auto ret = InferShape4ReduceWithoutAxes(inShape, outShape, keepDims, axesDim0);
            OP_LOGI(context->GetNodeName(), "outShape = %s", ToString(*outShape).c_str());
            return ret;
        }
    } else if (axesDtype == ge::DT_INT64) {
        const int64_t* axesData = axesTensor->GetData<int64_t>();
        if (axesData == nullptr) {
            OP_LOGW(context->GetNodeName(), "axes is not const, do conservative infer");
            auto ret = InferShape4ReduceWithoutAxes(inShape, outShape, keepDims, axesDim0);
            OP_LOGI(context->GetNodeName(), "outShape = %s", ToString(*outShape).c_str());
            return ret;
        }
    } else {
        OP_LOGW(context->GetNodeName(), "axes dtype not in (int32,int64), is not const, do conservative infer");
        auto ret = InferShape4ReduceWithoutAxes(inShape, outShape, keepDims, axesDim0);
        OP_LOGI(context->GetNodeName(), "outShape = %s", ToString(*outShape).c_str());
        return ret;
    }

    // --- 场景3: axes为静态常量，精确推导 ---
    ge::graphStatus ret;
    if (axesDtype == ge::DT_INT32) {
        ret = ReduceDims<int32_t>(inShape, axesTensor, axesShapeSize, keepDims, outShape);
    } else if (axesDtype == ge::DT_INT64) {
        ret = ReduceDims<int64_t>(inShape, axesTensor, axesShapeSize, keepDims, outShape);
    } else {
        OP_LOGE(context->GetNodeName(), "const axes data type %s must in (int32, int64)", ToString(axesDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "outShape = %s", ToString(*outShape).c_str());
    return ret;
}

inline ge::graphStatus InferShape4ReduceCommon(gert::InferShapeContext* context, const char* opName,
                                               bool hasNoopAttr = true)
{
    OP_LOGI(context->GetNodeName(), "Begin %s.", opName);

    auto inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto axesTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    OP_LOGI(context->GetNodeName(), "inShape = %s", ToString(*inShape).c_str());

    if (IsUnknownRank(*inShape)) {
        OP_LOGI(context->GetNodeName(), "input shape is unknown rank {-2}, set output shape {-2}");
        SetUnknownRank(*outShape);
        return ge::GRAPH_SUCCESS;
    }

    bool keepDimsValue = false;
    const bool* keepDims = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepDims);
    if (keepDims != nullptr) {
        keepDimsValue = *keepDims;
    }
    OP_LOGI(context->GetNodeName(), "keepDims value = %s", keepDimsValue ? "true" : "false");

    bool noopWithEmptyAxesValue = true;
    if (hasNoopAttr) {
        const bool* noopWithEmptyAxes = attrs->GetAttrPointer<bool>(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, noopWithEmptyAxes);
        if (noopWithEmptyAxes != nullptr) {
            noopWithEmptyAxesValue = *noopWithEmptyAxes;
        }
        OP_LOGI(context->GetNodeName(), "noopWithEmptyAxes value = %s", noopWithEmptyAxesValue ? "true" : "false");
    } else {
        // 无noop属性的算子，空axes一律allReduce
        noopWithEmptyAxesValue = false;
    }

    return DoInferShapeReduce(context, inShape, outShape, axesTensor, keepDimsValue, noopWithEmptyAxesValue);
}

inline ge::graphStatus InferShapeRange4ReduceCommon(gert::InferShapeRangeContext* context, const char* opName)
{
    OP_LOGI(context->GetNodeName(), "Begin %s.", opName);

    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferDataType4ReduceCommon(gert::InferDataTypeContext* context, const char* opName)
{
    OP_LOGI(context->GetNodeName(), "Begin %s.", opName);
    auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace Math
} // namespace Ops

#endif // INFERSHAPE_REDUCE_UTIL_H
