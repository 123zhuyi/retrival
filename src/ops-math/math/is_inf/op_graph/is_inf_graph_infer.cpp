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
 * \file is_inf_graph_infer.cpp
 * \brief is_inf operater graph infer resource
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferDataType4SingleOutBool2(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP(IsInf).InferDataType(InferDataType4SingleOutBool2);

}; // namespace ops