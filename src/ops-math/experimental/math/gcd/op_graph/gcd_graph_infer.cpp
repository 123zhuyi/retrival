/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static bool IsGcdFloatingType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static int GcdIntegerRank(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_UINT8:
            return 0;
        case ge::DT_INT8:
            return 1;
        case ge::DT_INT16:
            return 2;
        case ge::DT_INT32:
            return 3;
        case ge::DT_INT64:
            return 4;
        default:
            return -1;
    }
}

static ge::DataType PromoteIntegerTypes(ge::DataType x1, ge::DataType x2)
{
    const int x1Rank = GcdIntegerRank(x1);
    const int x2Rank = GcdIntegerRank(x2);
    if (x1Rank < 0 || x2Rank < 0) {
        return ge::DT_UNDEFINED;
    }
    if (x1 == x2) {
        return x1;
    }
    if ((x1 == ge::DT_UINT8 && x2 == ge::DT_INT8) || (x1 == ge::DT_INT8 && x2 == ge::DT_UINT8) ||
        (x1 == ge::DT_UINT8 && x2 == ge::DT_INT16) || (x1 == ge::DT_INT16 && x2 == ge::DT_UINT8)) {
        return ge::DT_INT16;
    }
    if (x1 == ge::DT_UINT8) {
        return x2;
    }
    if (x2 == ge::DT_UINT8) {
        return x1;
    }
    return x1Rank >= x2Rank ? x1 : x2;
}

static ge::DataType PromoteGcdDataType(ge::DataType x1, ge::DataType x2)
{
    if (x1 == x2) {
        return x1;
    }
    if (x1 == ge::DT_FLOAT || x2 == ge::DT_FLOAT) {
        return ge::DT_FLOAT;
    }
    if ((x1 == ge::DT_FLOAT16 && x2 == ge::DT_BF16) || (x1 == ge::DT_BF16 && x2 == ge::DT_FLOAT16)) {
        return ge::DT_FLOAT;
    }
    if (x1 == ge::DT_FLOAT16 || x2 == ge::DT_FLOAT16) {
        return ge::DT_FLOAT16;
    }
    if (x1 == ge::DT_BF16 || x2 == ge::DT_BF16) {
        return ge::DT_BF16;
    }
    if (IsGcdFloatingType(x1) || IsGcdFloatingType(x2)) {
        return ge::DT_UNDEFINED;
    }
    return PromoteIntegerTypes(x1, x2);
}

static ge::graphStatus InferDataType4Gcd(gert::InferDataTypeContext* context)
{
    OP_LOGI(context, "Enter InferDataType4Gcd");
    const ge::DataType x1DataType = context->GetInputDataType(0);
    const ge::DataType x2DataType = context->GetInputDataType(1);
    context->SetOutputDataType(0, PromoteGcdDataType(x1DataType, x2DataType));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Gcd).InferDataType(InferDataType4Gcd);
} // namespace ops
