/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"

namespace domi {

static Status ParseParamsReshape(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    int32_t allow_zero = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "allowzero" && attr.type() == ge::onnx::AttributeProto::INT) {
            allow_zero = attr.i();
        }
    }
    op_dest.SetAttr("allowzero", allow_zero);
    return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Reshape")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Reshape"),
                   ge::AscendString("ai.onnx::9::Reshape"),
                   ge::AscendString("ai.onnx::10::Reshape"),
                   ge::AscendString("ai.onnx::11::Reshape"),
                   ge::AscendString("ai.onnx::12::Reshape"),
                   ge::AscendString("ai.onnx::13::Reshape"),
                   ge::AscendString("ai.onnx::14::Reshape"),
                   ge::AscendString("ai.onnx::15::Reshape"),
                   ge::AscendString("ai.onnx::16::Reshape"),
                   ge::AscendString("ai.onnx::17::Reshape"),
                   ge::AscendString("ai.onnx::18::Reshape")})
    .ParseParamsFn(ParseParamsReshape)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
