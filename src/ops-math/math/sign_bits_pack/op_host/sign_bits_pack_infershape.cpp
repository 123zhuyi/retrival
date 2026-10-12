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
 * \file sign_bits_pack_infershape.cpp
 * \brief SignBitsPack 算子 InferShape 实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "graph/ge_error_codes.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShape4SignBitsPack(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* size_ptr = attrs->GetAttrPointer<int64_t>(0);
    if (size_ptr == nullptr) {
        OP_LOGE(context->GetNodeName(), "size attr is null.");
        return ge::GRAPH_FAILED;
    }
    int64_t size = *size_ptr;

    size_t dim_num = input_shape->GetDimNum();
    bool is_unknown_rank = (dim_num == 0) || (dim_num == 1 && input_shape->GetDim(0) == -2);
    if (is_unknown_rank) {
        output_shape->SetDimNum(0);
        output_shape->AppendDim(size);
        output_shape->AppendDim(-1);
        return ge::GRAPH_SUCCESS;
    }

    if (dim_num != 1 || size < 1) {
        OP_LOGE(context->GetNodeName(), "input dim_num must be 1 and size must be >= 1, but dim_num=%zu, size=%ld.",
                dim_num, size);
        return ge::GRAPH_FAILED;
    }

    int64_t n = input_shape->GetDim(0);
    if (n == -1) {
        output_shape->SetDimNum(0);
        output_shape->AppendDim(size);
        output_shape->AppendDim(-1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t packed_len = (n + 7) / 8;

    if (packed_len % size != 0) {
        OP_LOGE(context->GetNodeName(), "packed_len %ld is not divisible by size %ld, input n=%ld.", packed_len, size,
                n);
        return ge::GRAPH_FAILED;
    }

    output_shape->SetDimNum(0);
    output_shape->AppendDim(size);
    output_shape->AppendDim(packed_len / size);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SignBitsPack).InferShape(InferShape4SignBitsPack);

} // namespace ops
