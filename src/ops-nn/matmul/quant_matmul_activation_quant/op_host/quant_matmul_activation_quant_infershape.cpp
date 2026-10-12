/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_activation_quant_infershape.cpp
 * \brief InferShape and InferDataType for QuantMatmulActivationQuant
 */
#include <string>

#include "common/op_host/matmul_common_infershape.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "runtime/infer_datatype_context.h"

namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t X1_SCALE_INDEX = 3;
constexpr uint32_t X2_SCALE_INDEX = 4;

constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;

constexpr uint32_t Y_DTYPE_INDEX = 4;

constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
constexpr size_t QUANT_MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t QUANT_MATMUL_MAX_SHAPE_SIZE = 6;
constexpr const char* OP_NAME = "QuantMatmulActivationQuant";

const char* GetValidOpNameForInferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return OP_NAME;
    }
    const char* nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    return OP_NAME;
}

const char* GetValidOpNameForInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return OP_NAME;
    }
    const char* nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    return OP_NAME;
}

static ge::graphStatus InferShape4QuantMatmulActivationQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin to do InferShape4QuantMatmulActivationQuant");

    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* opName = GetValidOpNameForInferShape(context);

    auto shape_x1 = context->GetInputShape(X1_INDEX);
    auto shape_x2 = context->GetInputShape(X2_INDEX);
    auto shape_x1_scale = context->GetOptionalInputShape(X1_SCALE_INDEX);
    auto shape_x2_scale = context->GetOptionalInputShape(X2_SCALE_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x1_scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x2_scale);

    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    bool any_unknown_rank = Ops::NN::CheckIsUnknownDimNum(*shape_x1) || Ops::NN::CheckIsUnknownDimNum(*shape_x2);

    auto out_shape_y = context->GetOutputShape(Y_INDEX);
    auto out_shape_y_scale = context->GetOutputShape(Y_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_y);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_y_scale);

    if (!any_unknown_rank && (dim_a < QUANT_MATMUL_MIN_SHAPE_SIZE || dim_a > QUANT_MATMUL_MAX_SHAPE_SIZE ||
                              dim_b < QUANT_MATMUL_MIN_SHAPE_SIZE || dim_b > QUANT_MATMUL_MAX_SHAPE_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "x1, x2",
                                                  Ops::NN::FormatString("%zuD, %zuD", dim_a, dim_b).c_str(),
                                                  "the shape dims of x1 and x2 must be in the range of 2 to 6");
        return ge::GRAPH_FAILED;
    }

    if (any_unknown_rank) {
        out_shape_y->SetDimNum(1);
        out_shape_y->SetDim(0, UNKNOWN_DIM_NUM);
        out_shape_y_scale->SetDimNum(1);
        out_shape_y_scale->SetDim(0, UNKNOWN_DIM_NUM);
        OP_LOGD(opName, "Unknown Rank: y=[-2], y_scale=[-2]");
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(opName, "x1_shape: %s, x2_shape: %s", Ops::Base::ToString(*shape_x1).c_str(),
            Ops::Base::ToString(*shape_x2).c_str());

    auto attrs = context->GetAttrs();
    bool trans_x1 = false;
    bool trans_x2 = false;
    if (attrs != nullptr) {
        const bool* trans_x1_ptr = attrs->GetAttrPointer<bool>(0);
        const bool* trans_x2_ptr = attrs->GetAttrPointer<bool>(1);
        if (trans_x1_ptr != nullptr) {
            trans_x1 = *trans_x1_ptr;
        }
        if (trans_x2_ptr != nullptr) {
            trans_x2 = *trans_x2_ptr;
        }
    }

    int64_t M = trans_x1 ? shape_x1->GetDim(dim_a - 1) : shape_x1->GetDim(dim_a - 2);
    int64_t K = trans_x1 ? shape_x1->GetDim(dim_a - 2) : shape_x1->GetDim(dim_a - 1);
    int64_t K_x2 = trans_x2 ? shape_x2->GetDim(dim_b - 1) : shape_x2->GetDim(dim_b - 2);
    int64_t N = trans_x2 ? shape_x2->GetDim(dim_b - 2) : shape_x2->GetDim(dim_b - 1);

    if (K != UNKNOWN_DIM && K_x2 != UNKNOWN_DIM && K != K_x2) {
        OP_LOGE(opName, "K dimension mismatch: x1.K=%ld, x2.K=%ld", K, K_x2);
        return ge::GRAPH_FAILED;
    }

    size_t num_dim = std::max(dim_a, dim_b);
    out_shape_y->SetDimNum(num_dim);

    // batch 维从右对齐，x1/x2 的 batch 维度数各自不同，不足的补 1
    size_t x1BatchCount = (dim_a >= 2) ? (dim_a - 2) : 0;
    size_t x2BatchCount = (dim_b >= 2) ? (dim_b - 2) : 0;
    size_t batchDimNum = std::max(x1BatchCount, x2BatchCount);
    for (size_t i = 0; i < num_dim - 2; ++i) {
        // 从右侧对齐的 batch 维计算：逻辑输出位置 i 对应 x1/x2 中的右对齐索引
        int64_t x1Idx = static_cast<int64_t>(i) - static_cast<int64_t>(batchDimNum - x1BatchCount);
        int64_t x2Idx = static_cast<int64_t>(i) - static_cast<int64_t>(batchDimNum - x2BatchCount);
        int64_t dim_a_val = (x1Idx >= 0) ? shape_x1->GetDim(static_cast<size_t>(x1Idx)) : 1;
        int64_t dim_b_val = (x2Idx >= 0) ? shape_x2->GetDim(static_cast<size_t>(x2Idx)) : 1;

        if (dim_a_val != UNKNOWN_DIM && dim_b_val != UNKNOWN_DIM && dim_a_val != dim_b_val && dim_a_val != 1 &&
            dim_b_val != 1) {
            OP_LOGE(opName, "Batch dimension broadcast mismatch at dim %zu: x1=%ld, x2=%ld", i, dim_a_val, dim_b_val);
            return ge::GRAPH_FAILED;
        }

        // 动态维广播：一侧未知时输出仍应保持未知，除非另一侧为已知且大于 1
        int64_t broadcast_dim;
        if (dim_a_val == UNKNOWN_DIM || dim_b_val == UNKNOWN_DIM) {
            int64_t known = (dim_a_val != UNKNOWN_DIM) ? dim_a_val :
                            (dim_b_val != UNKNOWN_DIM) ? dim_b_val :
                                                         UNKNOWN_DIM;
            broadcast_dim = (known > 1) ? known : UNKNOWN_DIM;
        } else {
            broadcast_dim = std::max(dim_a_val, dim_b_val);
        }
        out_shape_y->SetDim(i, broadcast_dim);
    }

    out_shape_y->SetDim(num_dim - 2, M);
    out_shape_y->SetDim(num_dim - 1, N);

    OP_LOGD(opName, "output y shape: %s", Ops::Base::ToString(*out_shape_y).c_str());

    int64_t scale_n = UNKNOWN_DIM;
    if (N != UNKNOWN_DIM && N != UNKNOWN_DIM_NUM) {
        scale_n = (N + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
        scale_n = (scale_n + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    size_t y_scale_dim = num_dim + 1;
    out_shape_y_scale->SetDimNum(y_scale_dim);

    for (size_t i = 0; i < num_dim - 2; ++i) {
        out_shape_y_scale->SetDim(i, out_shape_y->GetDim(i));
    }

    out_shape_y_scale->SetDim(num_dim - 2, M);
    out_shape_y_scale->SetDim(num_dim - 1, scale_n);
    out_shape_y_scale->SetDim(num_dim, ALIGN_NUM);

    OP_LOGD(opName, "output y_scale shape: %s", Ops::Base::ToString(*out_shape_y_scale).c_str());

    OP_LOGI(context, "End InferShape4QuantMatmulActivationQuant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4QuantMatmulActivationQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4QuantMatmulActivationQuant");

    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* opName = GetValidOpNameForInferDataType(context);

    auto attrs = context->GetAttrs();
    int64_t y_dtype = -1;
    if (attrs != nullptr) {
        const int64_t* y_dtype_ptr = attrs->GetAttrPointer<int64_t>(Y_DTYPE_INDEX);
        if (y_dtype_ptr != nullptr) {
            y_dtype = *y_dtype_ptr;
        }
    }

    if (y_dtype < 0) {
        OP_LOGE(opName, "Failed to get y_dtype attr");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(Y_INDEX, static_cast<ge::DataType>(y_dtype));
    context->SetOutputDataType(Y_SCALE_INDEX, ge::DT_FLOAT8_E8M0);

    OP_LOGD(context, "End InferDataType4QuantMatmulActivationQuant");
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(QuantMatmulActivationQuant)
    .InferShape(InferShape4QuantMatmulActivationQuant)
    .InferDataType(InferDataType4QuantMatmulActivationQuant);
}
