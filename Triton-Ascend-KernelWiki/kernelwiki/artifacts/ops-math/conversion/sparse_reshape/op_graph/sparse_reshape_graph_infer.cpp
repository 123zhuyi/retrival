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
 * \file sparse_reshape_graph_infer.cpp
 * \brief Data type inference for SparseReshape operator
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferDataTypeForSparseReshape(gert::InferDataTypeContext* context)
{
    OP_LOGI("Begin InferDataTypeForSparseReshape");
    const ge::DataType indicesDataType = context->GetInputDataType(0);
    const ge::DataType shapeDataType = context->GetInputDataType(1);
    context->SetOutputDataType(0, indicesDataType);
    context->SetOutputDataType(1, shapeDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseReshape).InferDataType(InferDataTypeForSparseReshape);
} // namespace ops
