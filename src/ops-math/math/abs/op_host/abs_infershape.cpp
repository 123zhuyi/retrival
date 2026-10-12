/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file abs_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "infershape_elewise_util.h"
#include "log/log.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
    static graphStatus InferDataType4Abs(gert::InferDataTypeContext* context){
        OP_LOGD(context->GetNodeName(), "InferDataType4Abs enter");
        auto input_x_dtype =context->GetInputDataType(0);
        DataType output_dtype =input_x_dtype;
        if(input_x_dtype==DT_COMPLEX64){
            output_dtype = DT_FLOAT;
        } else if (input_x_dtype == DT_COMPLEX32){
            output_dtype = DT_FLOAT16;
        }
        context->SetOutputDataType(0, output_dtype);
        OP_LOGD(context->GetNodeName(), "InferDataType4Abs end");
        return GRAPH_SUCCESS;
    }
IMPL_OP_INFERSHAPE(Abs).InferShape(InferShape4Elewise).InferDataType(InferDataType4Abs);
} // namespace ops
