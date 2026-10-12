/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_PADDINGS = 1;
static constexpr size_t OUTPUT_IDX_Z = 0;
static constexpr size_t PADDING_PAIR_SIZE = 2;
static constexpr int64_t UNKNOWN_DIM = -1;

static ge::graphStatus InferShape4PadV3GradReplication(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* zShape = context->GetOutputShape(OUTPUT_IDX_Z);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*zShape);
        OP_LOGI(context->GetNodeName(), "PadV3GradReplication output shape: %s.", Ops::Base::ToString(*zShape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    gert::Shape paddings;
    if (!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context, INPUT_IDX_PADDINGS, paddings)) {
        zShape->SetDimNum(xShape->GetDimNum());
        for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
            zShape->SetDim(i, UNKNOWN_DIM);
        }
        OP_LOGI(context->GetNodeName(), "PadV3GradReplication output shape: %s.", Ops::Base::ToString(*zShape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(paddings.GetDimNum() != xShape->GetDimNum() * PADDING_PAIR_SIZE,
                OP_LOGE(context->GetNodeName(), "The paddings element count must be twice the x rank."),
                return ge::GRAPH_FAILED);
    *zShape = *xShape;
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        const int64_t xDim = xShape->GetDim(i);
        if (xDim != UNKNOWN_DIM) {
            const int64_t zDim = xDim - paddings.GetDim(PADDING_PAIR_SIZE * i) -
                                 paddings.GetDim(PADDING_PAIR_SIZE * i + 1);
            OP_CHECK_IF(
                zDim < 0,
                OP_LOGE(context->GetNodeName(), "The output dimension must not be negative, but got %ld.", zDim),
                return ge::GRAPH_FAILED);
            zShape->SetDim(i, zDim);
        }
    }
    OP_LOGI(context->GetNodeName(), "PadV3GradReplication output shape: %s.", Ops::Base::ToString(*zShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4PadV3GradReplication(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_IDX_Z, context->GetInputDataType(INPUT_IDX_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PadV3GradReplication)
    .InferShape(InferShape4PadV3GradReplication)
    .InferDataType(InferDataType4PadV3GradReplication)
    .InputsDataDependency({INPUT_IDX_PADDINGS});
} // namespace ops
