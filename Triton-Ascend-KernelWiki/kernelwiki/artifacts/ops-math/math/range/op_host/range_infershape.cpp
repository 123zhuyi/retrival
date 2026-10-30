/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_infershape.cpp
 * \brief
 */
#include <cmath>
#include <type_traits>

#include "log/log.h"
#include "op_host/util/fp16.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
constexpr size_t INT16_BITS_NUM = 16;

static bool IsTensorNull(const gert::InferShapeContext* context, const gert::Tensor* tensor)
{
    auto tensorDataType = tensor->GetDataType();
    auto ret = false;
    switch (tensorDataType) {
        case ge::DT_INT32: {
            const int32_t* constDataPtr = tensor->GetData<int32_t>();
            ret = (constDataPtr == nullptr) ? true : ret;
            break;
        }
        case ge::DT_FLOAT: {
            const float* constDataPtr = tensor->GetData<float>();
            ret = (constDataPtr == nullptr) ? true : ret;
            break;
        }
        case ge::DT_FLOAT16: {
            const Ops::Base::fp16_t* constDataPtr = tensor->GetData<Ops::Base::fp16_t>();
            ret = (constDataPtr == nullptr) ? true : ret;
            break;
        }
        case ge::DT_BF16: {
            const int16_t* constDataPtr = tensor->GetData<int16_t>();
            ret = (constDataPtr == nullptr) ? true : ret;
            break;
        }
        case ge::DT_INT64: {
            const int64_t* constDataPtr = tensor->GetData<int64_t>();
            ret = (constDataPtr == nullptr) ? true : ret;
            break;
        }
        case ge::DT_DOUBLE: {
            const double* constDataPtr = tensor->GetData<double>();
            ret = (constDataPtr == nullptr) ? true : ret;
            break;
        }
        default:
            OP_LOGW(context->GetNodeName(), "aicore datatype not support.");
            break;
    }
    return ret;
}

template <typename T>
static ge::graphStatus RangeGetConstValue(
    gert::InferShapeContext* context, const gert::Tensor* tensor, std::vector<T>& value)
{
    if (tensor->GetDataType() == ge::DT_INT32) {
        const int32_t* constDataPtr = tensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value.push_back(static_cast<T>(*constDataPtr));
        OP_LOGD(context->GetNodeName(), "range get const value:%d", *constDataPtr);
    } else if (tensor->GetDataType() == ge::DT_FLOAT) {
        const float* constDataPtr = tensor->GetData<float>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value.push_back(static_cast<T>(*constDataPtr));
        OP_LOGD(context->GetNodeName(), "range get const value:%f", *constDataPtr);
    } else if (tensor->GetDataType() == ge::DT_FLOAT16) {
        const Ops::Base::fp16_t* constDataPtr = tensor->GetData<Ops::Base::fp16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        float f32 = static_cast<float>(*constDataPtr);
        value.push_back(static_cast<T>(f32));
        OP_LOGD(context->GetNodeName(), "range get const value:%f", static_cast<float>(*constDataPtr));
    } else if (tensor->GetDataType() == ge::DT_BF16) {
        const int16_t* constDataPtr = tensor->GetData<int16_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        int32_t f32hex = (static_cast<int32_t>(*constDataPtr)) << INT16_BITS_NUM;
        float* f32ptr = reinterpret_cast<float*>(&f32hex);
        value.push_back(static_cast<T>(*f32ptr));
        OP_LOGD(context->GetNodeName(), "range get const value:%d", f32hex);
    } else if (tensor->GetDataType() == ge::DT_INT64) {
        const int64_t* constDataPtr = tensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value.push_back(static_cast<T>(*constDataPtr));
        OP_LOGD(context->GetNodeName(), "range get const value:%ld", *constDataPtr);
    } else if (tensor->GetDataType() == ge::DT_DOUBLE) {
        const double* constDataPtr = tensor->GetData<double>();
        OP_CHECK_NULL_WITH_CONTEXT(context, constDataPtr);
        value.push_back(static_cast<T>(*constDataPtr));
        OP_LOGD(context->GetNodeName(), "range get const value:%lf", *constDataPtr);
        // do nothing, impossible
        return ge::GRAPH_SUCCESS;
    } else {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus CheckParam(gert::InferShapeContext* context, T start, T limit, T delta)
{
    OP_CHECK_IF(
        !(delta > (static_cast<T>(0)) || delta < (static_cast<T>(0))),
        OP_LOGE(context->GetNodeName(), "delta is zero."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((limit > start) && (delta < 0)),
        OP_LOGE(context->GetNodeName(), "increate is positive, but delta is negative"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((limit < start) && (delta > 0)),
        OP_LOGE(context->GetNodeName(), "increate is negative, but delta is positive"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus CalculateOutputNum(
    gert::InferShapeContext* context, const gert::Tensor* tensorStart, const gert::Tensor* tensorLimit,
    const gert::Tensor* tensorDelta, uint64_t& totalNum)
{
    std::vector<T> startMultiples;
    std::vector<T> limitMultiples;
    std::vector<T> deltaMultiples;
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorStart, startMultiples) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get start const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorLimit, limitMultiples) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get limit const value fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        RangeGetConstValue<T>(context, tensorDelta, deltaMultiples) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "get delta const value fail."), return ge::GRAPH_FAILED);
    if (startMultiples.empty() || limitMultiples.empty() || deltaMultiples.empty()) {
        totalNum = -1;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        CheckParam(context, startMultiples[0], limitMultiples[0], deltaMultiples[0]) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context->GetNodeName(), "CheckParam fail, start: %lf, limit: %lf, delta: %lf",
            static_cast<double>(startMultiples[0]), static_cast<double>(limitMultiples[0]),
            static_cast<double>(deltaMultiples[0])),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isClosed = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isClosed);
    if (*isClosed) {
        totalNum = static_cast<uint64_t>((limitMultiples[0] - startMultiples[0]) / deltaMultiples[0] + 1);
    } else {
        if (std::is_same<T, int64_t>::value) {
            totalNum = static_cast<uint64_t>(Ops::Base::CeilDiv(
                static_cast<int64_t>(limitMultiples[0]) - static_cast<int64_t>(startMultiples[0]),
                static_cast<int64_t>(deltaMultiples[0])));
        } else {
            std::vector<double> startDouble;
            std::vector<double> limitDouble;
            std::vector<double> deltaDouble;
            OP_CHECK_IF(
                RangeGetConstValue<double>(context, tensorStart, startDouble) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get start const value fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                RangeGetConstValue<double>(context, tensorLimit, limitDouble) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get limit const value fail."), return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                RangeGetConstValue<double>(context, tensorDelta, deltaDouble) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "get delta const value fail."), return ge::GRAPH_FAILED);
            totalNum = static_cast<uint64_t>(std::ceil((limitDouble[0] - startDouble[0]) / deltaDouble[0]));
        }
    }
    OP_LOGD(
        context->GetNodeName(), "CalculateOutputNum: start: %lf, limit: %lf, delta: %lf, total_num: %lu",
        static_cast<double>(startMultiples[0]), static_cast<double>(limitMultiples[0]),
        static_cast<double>(deltaMultiples[0]), totalNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RangeInferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "start running RangeInferShapeFunc...");
    auto startTensor = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, startTensor);
    auto limitTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, limitTensor);
    auto deltaTensor = context->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltaTensor);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    if (IsTensorNull(context, startTensor) || IsTensorNull(context, limitTensor) ||
        IsTensorNull(context, deltaTensor)) {
        out_shape->SetDimNum(1);
        out_shape->SetDim(0, -1);
        OP_LOGD(context->GetNodeName(), "tensor is null, return.");

        return GRAPH_SUCCESS;
    }

    auto outDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    DataType outDtype = outDesc->GetDataType();
    uint64_t totalNum = 0;

    switch (outDtype) {
        case ge::DT_FLOAT:
            OP_CHECK_IF(
                CalculateOutputNum<double>(context, startTensor, limitTensor, deltaTensor, totalNum) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "calculate output_total_num value fail."), return ge::GRAPH_FAILED);
            break;
        case ge::DT_INT32:
        case ge::DT_INT64:;
            OP_CHECK_IF(
                CalculateOutputNum<int64_t>(context, startTensor, limitTensor, deltaTensor, totalNum) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "calculate output_total_num value fail."), return ge::GRAPH_FAILED);
            break;
        default:
            OP_CHECK_IF(
                CalculateOutputNum<float>(context, startTensor, limitTensor, deltaTensor, totalNum) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "calculate output_total_num value fail."), return ge::GRAPH_FAILED);
            break;
    }

    out_shape->SetDimNum(1);
    out_shape->SetDim(0, totalNum);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Range).InputsDataDependency({0, 1, 2}).InferShape(RangeInferShapeFunc);
} // namespace ops
