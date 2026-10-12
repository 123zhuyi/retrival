/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/*!
 * \file ndtri_infer_datatype.cpp
 * \brief Ndtri 算子输出数据类型推导实现
 *
 * 输出 dtype 推导规则：out dtype = self dtype（逐元素算子，dtype 不变）
 *
 * 注册方式：CANN 9.0.0 的 register/op_impl_registry.h 未暴露 IMPL_OP_INFERDATATYPE，
 * 使用通用 IMPL_OP(op_type) 宏（基于 __COUNTER__ 生成唯一符号名），
 * 与 ndtri_infershape.cpp 中的 IMPL_OP_INFERSHAPE 并存。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static constexpr size_t IDX_SELF = 0;
static constexpr size_t OUT_OUT = 0;

static ge::graphStatus InferDataType4Ndtri(gert::InferDataTypeContext* context)
{
    const ge::DataType selfDtype = context->GetInputDataType(IDX_SELF);
    context->SetOutputDataType(OUT_OUT, selfDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Ndtri).InferDataType(InferDataType4Ndtri);

} // namespace ops
