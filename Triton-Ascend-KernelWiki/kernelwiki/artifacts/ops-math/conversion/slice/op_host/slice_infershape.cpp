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
 * \file slice_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/const_util.h"
#include "util/shape_util.h"
#include "op_util.h"

using namespace ge;
namespace ops {
static constexpr size_t SLICE_IN_X = 0;
static constexpr size_t SLICE_IN_OFFSET = 1;
static constexpr size_t SLICE_IN_SIZE = 2;
static constexpr size_t SLICE_OUT_Y = 0;

struct SliceConstParams {
    gert::Shape offset;
    gert::Shape size;
    int64_t offset_num;
    int64_t size_num;
    bool is_offset_const;
    bool is_size_const;

    std::string to_string() const
    {
        std::string result = "offset_const:" + Ops::Base::ToString(offset);
        result += " size_const:" + Ops::Base::ToString(size);
        result += " offset_num:" + std::to_string(offset_num);
        result += " size_num:" + std::to_string(size_num);
        return result;
    }

    void init()
    {
        offset.SetDimNum(0);
        size.SetDimNum(0);
        offset_num = 0;
        size_num = 0;
        is_size_const = true;
        is_offset_const = true;
    }
};

static int64_t SliceInferRankFn(
    gert::InferShapeContext* context, const gert::Shape* x_shape, const SliceConstParams& slice_infer_info)
{
    int64_t offset_num =
        slice_infer_info.is_offset_const ? slice_infer_info.offset.GetDimNum() : slice_infer_info.offset_num;
    int64_t size_num = slice_infer_info.is_size_const ? slice_infer_info.size.GetDimNum() : slice_infer_info.size_num;
    int64_t x_shape_rank = Ops::Base::IsUnknownRank(*x_shape) ? -1 : x_shape->GetDimNum();

    int64_t output_rank = std::max(offset_num, size_num);
    output_rank = std::max(output_rank, x_shape_rank);
    (void)context;

    return output_rank;
}

static bool CheckSliceInfo(const gert::Shape* x_shape, const SliceConstParams& slice_infer_info)
{
    const bool is_unknown_rank_x = Ops::Base::IsUnknownRank(*x_shape);
    if (is_unknown_rank_x) {
        OP_LOGD("CheckSliceInfo", "input is unknown rank, no nedd check.");
        return true;
    }

    const size_t input_dim = x_shape->GetDimNum();
    if (slice_infer_info.is_offset_const) {
        OP_LOGD("CheckSliceInfo", "will check offset const value.");
        OP_CHECK_IF(
            slice_infer_info.offset.GetDimNum() != input_dim,
            OP_LOGE(
                "CheckSliceInfo", "%s",
                ops::ConcatString(
                    "offset num and input rank must be the same, but offset_value is ",
                    Ops::Base::ToString(slice_infer_info.offset).c_str(), ", input shape is ",
                    Ops::Base::ToString(*x_shape).c_str()).c_str()),
            return false);
    }
    if (slice_infer_info.is_size_const) {
        OP_LOGD("CheckSliceInfo", "will check size const value.");
        OP_CHECK_IF(
            slice_infer_info.size.GetDimNum() != input_dim,
            OP_LOGE(
                "CheckSliceInfo", "%s",
                ops::ConcatString(
                    "size num and input rank must be the same, but offset_value is ",
                    Ops::Base::ToString(slice_infer_info.size).c_str(), ", input shape is ",
                    Ops::Base::ToString(*x_shape).c_str()).c_str()),
            return false);
    }

    return true;
}

static graphStatus SliceInferShapeFnWithConstSize(
    gert::InferShapeContext* context, const gert::Shape* x_shape, const SliceConstParams& slice_infer_info,
    gert::Shape* out_shape)
{
    OP_LOGD(context->GetNodeName(), "start to do SliceInferShapeFnWithConstSize");
    OP_LOGD(context->GetNodeName(), "slice input shape is %s", Ops::Base::ToString(*x_shape).c_str());
    OP_LOGD(context->GetNodeName(), "slice const info is %s", slice_infer_info.to_string().c_str());
    // set output shape as the const input size
    *out_shape = slice_infer_info.size;

    const bool is_unknown_rank_x = Ops::Base::IsUnknownRank(*x_shape);

    // will update output dim value when const size = -1
    const size_t output_rank = out_shape->GetDimNum();
    for (size_t i = 0; i < output_rank; ++i) {
        int64_t size_value = out_shape->GetDim(i);
        OP_CHECK_IF(
            size_value < -1,
            OP_LOGE(
                context->GetNodeName(), "%s",
                ops::ConcatString(
                    "the value of size can not < -1, but is ", Ops::Base::ToString(slice_infer_info.size).c_str()).c_str()),
            return ge::GRAPH_FAILED);
        if (!is_unknown_rank_x && slice_infer_info.is_offset_const) {
            int64_t x_dim_value = x_shape->GetDim(i);
            if (size_value == -1 && x_dim_value != -1) {
                out_shape->SetDim(i, x_dim_value - slice_infer_info.offset.GetDim(i));
            }
        }
    }
    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*out_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do SliceInferShapeFn with const size");

    return ge::GRAPH_SUCCESS;
}

static graphStatus SliceInferShapeFn(
    gert::InferShapeContext* context, const gert::Shape* x_shape, const SliceConstParams& slice_infer_info,
    gert::Shape* out_shape)
{
    OP_LOGD(context->GetNodeName(), "start to do SliceInferShapeFn");
    OP_LOGD(context->GetNodeName(), "slice input shape is %s", Ops::Base::ToString(*x_shape).c_str());
    OP_LOGD(context->GetNodeName(), "slice const info is %s", slice_infer_info.to_string().c_str());
    // check the input info
    OP_CHECK_IF(
        !CheckSliceInfo(x_shape, slice_infer_info), OP_LOGE(context->GetNodeName(), "check input info failed."),
        return ge::GRAPH_FAILED);
    if (slice_infer_info.is_size_const) {
        return SliceInferShapeFnWithConstSize(context, x_shape, slice_infer_info, out_shape);
    }

    // the input size is not const, try infer rank
    OP_LOGD(context->GetNodeName(), "do slice info shape with tensor size.");
    const int64_t output_rank = SliceInferRankFn(context, x_shape, slice_infer_info);
    OP_LOGD(context->GetNodeName(), "try to get the rank of output = %ld.", output_rank);

    if (output_rank <= 0) {
        OP_LOGD(context->GetNodeName(), "set the output shape = -2.");
        Ops::Base::SetUnknownRank(*out_shape);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(context->GetNodeName(), "set the output shape all -1.");
    Ops::Base::SetUnknownShape(output_rank, *out_shape);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShape4Slice(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(SLICE_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape* out_shape = context->GetOutputShape(SLICE_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    //
    SliceConstParams slice_infer_info;
    slice_infer_info.init();
    if (!Ops::Base::GetConstIntToShape(context, SLICE_IN_OFFSET, slice_infer_info.offset)) {
        slice_infer_info.is_offset_const = false;
        OP_LOGD(context->GetNodeName(), "Get const of size unsuccessful, will do dynamic infershape.");
        slice_infer_info.offset.SetDimNum(0);
        const gert::Shape* offset_shape = context->GetInputShape(SLICE_IN_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(context, offset_shape);
        slice_infer_info.offset_num = offset_shape->GetDimNum() == 0 ? 1 : offset_shape->GetDim(0);
    }

    if (!Ops::Base::GetConstIntToShape(context, SLICE_IN_SIZE, slice_infer_info.size)) {
        slice_infer_info.is_size_const = false;
        OP_LOGD(context->GetNodeName(), "Get const of offset unsuccessful, will do dynamic infershape.");
        slice_infer_info.size.SetDimNum(0);
        const gert::Shape* size_shape = context->GetInputShape(SLICE_IN_SIZE);
        OP_CHECK_NULL_WITH_CONTEXT(context, size_shape);
        slice_infer_info.size_num = size_shape->GetDimNum() == 0 ? 1 : size_shape->GetDim(0);
    }

    return SliceInferShapeFn(context, x_shape, slice_infer_info, out_shape);
}

IMPL_OP_INFERSHAPE(Slice).InferShape(InferShape4Slice).InputsDataDependency({SLICE_IN_OFFSET, SLICE_IN_SIZE});
} // namespace ops
