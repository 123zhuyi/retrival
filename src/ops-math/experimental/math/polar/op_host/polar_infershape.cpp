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
 * \file polar_infershape.cpp
 * \brief Polar InferShape / InferDataType。
 *        InferShape：numpy 广播规则推导 out shape（右对齐各轴取 max），
 *        对齐参考 cann/ops-math math/complex/op_host/op_api/aclnn_polar.cpp 的
 *        OP_CHECK_BROADCAST_AND_INFER_SHAPE。
 *        InferDataType：out 恒置 complex64（与 input dtype 无关）。
 */
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Polar(gert::InferShapeContext* context)
{
    const gert::Shape* in = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in);
    const gert::Shape* an = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, an);
    gert::Shape* out = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);
    size_t inRank = in->GetDimNum();
    size_t anRank = an->GetDimNum();
    size_t outRank = inRank > anRank ? inRank : anRank;
    out->SetDimNum(outRank);
    for (size_t i = 0; i < outRank; i++) {
        int64_t di = (i + inRank >= outRank) ? in->GetDim(i + inRank - outRank) : 1;
        int64_t da = (i + anRank >= outRank) ? an->GetDim(i + anRank - outRank) : 1;
        out->SetDim(i, di > da ? di : da);
    }
    return GRAPH_SUCCESS;
}

// 输出恒为 complex64（对齐参考 OUTPUT_DTYPE = {DT_COMPLEX64}）
static ge::graphStatus InferDataType4Polar(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_COMPLEX64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Polar).InferShape(InferShape4Polar).InferDataType(InferDataType4Polar);
} // namespace ops
