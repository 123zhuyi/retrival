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
 * \file reduce_all_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "common/inc/op_host/infershape_reduce_util.h"

using namespace ge;
using namespace Ops::Math;
namespace ops {
static ge::graphStatus InferShape4ReduceAll(gert::InferShapeContext* context)
{
    return InferShape4ReduceCommon(context, "InferShape4ReduceAll", false);
}

static ge::graphStatus InferShapeRange4ReduceAll(gert::InferShapeRangeContext* context)
{
    return InferShapeRange4ReduceCommon(context, "InferShapeRange4ReduceAll");
}

IMPL_OP_INFERSHAPE(ReduceAll)
    .InferShape(InferShape4ReduceAll)
    .InferShapeRange(InferShapeRange4ReduceAll)
    .InputsDataDependency({1});
} // namespace ops
