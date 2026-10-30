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
 * \file confusion_transpose_d_graph_infer.cpp
 * \brief confusion_transpose_d operater graph infer resource
 */
//todo:这些头文件都没有啊
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/infershape_broadcast_util.h"

using namespace ge;
namespace ops{
static ge::graphStatus InferDataType4ConfusionTransposeD(gert::InferDataTypeContext* context){
    OP_LOGI("Begin to do infer dataType for ConfusionTransposeD.");
    const ge::DataType xDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDataType);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP(ConfusionTransposeD).InferDataType(InferDataType4ConfusionTransposeD);
}; // namespace ops