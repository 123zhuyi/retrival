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
 * \file allto_all_matmul_infershape.cpp
 * \brief 图模式（动态图/静态图）走infershape
 */

#include <register/op_impl_registry.h>
#include "util/math_util.h"
#include "mc2_log.h"
#include "op_mc2.h"
#include "mc2_common_infershape.h"

namespace ops {

using Ops::Base::CeilDiv;

constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;
constexpr size_t INDEX_ATTR_ALLTO_ALL_AXES = 2;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_X1_QUANT_MODE = 4;
constexpr size_t INDEX_ATTR_X2_QUANT_MODE = 5;
constexpr size_t INDEX_ATTR_TRANS_X1 = 9;
constexpr size_t INDEX_ATTR_TRANS_X2 = 10;
constexpr size_t INDEX_ATTR_ALLTOALL_OUT_FLAG = 12;
constexpr size_t INDEX_OUT = 0;
constexpr size_t INDEX_ALLTO_ALL_OUT = 1;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t X1_QUANT_NUM = 7;
constexpr uint64_t X2_QUANT_NUM = 2;
constexpr int64_t NUM_MINUS_ONE = -1;
constexpr int64_t NUM_MINUS_TWO = -2;
constexpr int64_t OUTPUT_INFER_SHAPE = 2;
static const char* INNER_DEBUG = "MC2: AlltoAllMatmul InferShape Debug";
const std::set<int> SUPPORT_RANK_NUM{2, 4, 8, 16};

struct AlltoAllMatmulShapeInfo {
    uint64_t output_dim;
    uint64_t rankNum;
    uint64_t m;
    uint64_t n;
    uint64_t k1;
    uint64_t k2;
};

/**
 * @brief 校验AlltoAllMatmul输入shape，并记录输入m，n，k大小
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckShapeForAlltoAllMatmul(const gert::InferShapeContext* context, AlltoAllMatmulShapeInfo& shape)
{
    const auto x1_shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2_shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto alltoAllAxesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_ALLTO_ALL_AXES);
    if (alltoAllAxesPtr != nullptr) {
        OPS_CHECK((alltoAllAxesPtr->GetSize() != DIM_TWO), CUBE_INNER_ERR_REPORT(INNER_DEBUG,
                  "the size of alltoAllAxes should be %ld, but the actual value is %ld.", DIM_TWO, alltoAllAxesPtr->GetSize()),
                  return ge::GRAPH_FAILED);
        const auto alltoAllAxes = static_cast<const int64_t*>(alltoAllAxesPtr->GetData());
        OPS_CHECK((alltoAllAxes[0] != NUM_MINUS_TWO || alltoAllAxes[1] != NUM_MINUS_ONE), CUBE_INNER_ERR_REPORT(INNER_DEBUG,
                  "the value of alltoAllAxes should be [-2, -1], but the actual value is [%ld, %ld].", alltoAllAxes[0], alltoAllAxes[1]),
                  return ge::GRAPH_FAILED);
    }

    const bool* isTransX1 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X1);
    OPS_CHECK(
        isTransX1 == nullptr || *isTransX1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
        "x1 does not support transpose in allto all matmul."), return ge::GRAPH_FAILED);
    const bool* isTransX2 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X2);
    const bool transX2 = ((isTransX2 != nullptr) && (*isTransX2));
    shape.m = x1_shape->GetDim(0U);
    shape.k1 = x1_shape->GetDim(1U);
    shape.n = transX2 ? x2_shape->GetDim(0U) : x2_shape->GetDim(1U);
    shape.k2 = transX2 ? x2_shape->GetDim(1U) : x2_shape->GetDim(0U);
    shape.output_dim = x1_shape->GetDimNum();
    OP_LOGD(INNER_DEBUG, "Matmul m %ld n %ld k1 %ld k2 %ld.", shape.m, shape.n, shape.k1, shape.k2);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取，校验并记录卡数
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckRankDim(gert::InferShapeContext* context, AlltoAllMatmulShapeInfo& shape)
{
    const auto attrs = context->GetAttrs();
    const int* rankDim = attrs->GetAttrPointer<int>(INDEX_ATTR_WORLD_SIZE);
    OPS_CHECK(rankDim == nullptr,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid rank number %zu in allto all matmul.", *rankDim),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SUPPORT_RANK_NUM.find(*rankDim) == SUPPORT_RANK_NUM.end(),
                    OP_LOGE(INNER_DEBUG, "Rank number should be 2 or 4 or 8 or 16, but the actual value is %ld.", *rankDim),
                    return ge::GRAPH_FAILED);
    shape.rankNum = *rankDim;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出shape
 *
 * @param context
 */
static ge::graphStatus InferShapeAlltoAllMatmul(gert::InferShapeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    AlltoAllMatmulShapeInfo shape;
    OPS_CHECK(
        CheckShapeForAlltoAllMatmul(context, shape) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check shape for allto all matmul"),
        return ge::GRAPH_FAILED);
    OPS_CHECK(
        CheckRankDim(context, shape) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check rank dim for allto all matmul."),
        return ge::GRAPH_FAILED);
    auto shape_out = context->GetOutputShape(INDEX_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_out);
    const auto attrs = context->GetAttrs();
    const bool* all2all_out_flag = attrs->GetAttrPointer<bool>(INDEX_ATTR_ALLTOALL_OUT_FLAG);
    OPS_CHECK_NULL_WITH_CONTEXT(context, all2all_out_flag);
    auto all2all_out = context->GetOutputShape(INDEX_ALLTO_ALL_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, all2all_out);
    all2all_out->SetDimNum(OUTPUT_INFER_SHAPE);
    if (all2all_out_flag) {
        if (shape.m != NUM_MINUS_ONE) {
            uint64_t all2all_out_first_dim = CeilDiv(shape.m, shape.rankNum);
            uint64_t all2all_out_second_dim = shape.k1 * shape.rankNum;
            all2all_out->SetDim(0U, all2all_out_first_dim);
            all2all_out->SetDim(1U, all2all_out_second_dim);
            OP_LOGI(
            INNER_DEBUG, "Allto all matmul alltoallout shape after infer shape, output_dim: %ld, m: %ld n: %ld.",
            OUTPUT_INFER_SHAPE, all2all_out_first_dim, all2all_out_second_dim);
        }
    }
    shape_out->SetDimNum(OUTPUT_INFER_SHAPE);
    if (shape.m == NUM_MINUS_ONE) {
        shape_out->SetDim(0U, shape.m);
        shape_out->SetDim(1U, shape.n);
    } else {
        uint64_t out_first_dim = CeilDiv(shape.m, shape.rankNum);
        uint64_t out_second_dim = shape.n;
        shape_out->SetDim(0U, out_first_dim);
        shape_out->SetDim(1U, out_second_dim);
        OP_LOGI(INNER_DEBUG, "allto all matmul output shape after infer shape, m: %ld n: %ld.", out_first_dim, out_second_dim);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出数据类型
 *
 * @param context
 */
static ge::graphStatus InferDataTypeAlltoAllMatmul(gert::InferDataTypeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE(INNER_DEBUG, "Context is null."), return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "Start to infer datatype of allto all matmul.");
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int* x1_quant_mode = attrs->GetAttrPointer<int>(INDEX_ATTR_X1_QUANT_MODE);
    const int* x2_quant_mode = attrs->GetAttrPointer<int>(INDEX_ATTR_X2_QUANT_MODE);
    const int64_t* y_dtypes_ptr = attrs->GetInt(INDEX_ATTR_Y_DTYPE);
    auto y_type = ge::DataType::DT_UNDEFINED;
    ge::DataType x1_type = context->GetInputDataType(INDEX_IN_X1);
    if (*x1_quant_mode == 0 && *x2_quant_mode == 0) {
        if ((y_dtypes_ptr != nullptr && *y_dtypes_ptr != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED))) {
            y_type = static_cast<ge::DataType>(*y_dtypes_ptr);
        } else {
            return ge::GRAPH_FAILED;
        }
    } else if (*x1_quant_mode == X1_QUANT_NUM && *x2_quant_mode == X2_QUANT_NUM) {
        if ((y_dtypes_ptr != nullptr && *y_dtypes_ptr != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED))) {
            y_type = static_cast<ge::DataType>(*y_dtypes_ptr);
        } else {
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, y_type);
    context->SetOutputDataType(1, x1_type);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllMatmul).InferShape(InferShapeAlltoAllMatmul).InferDataType(InferDataTypeAlltoAllMatmul);
} // namespace ops