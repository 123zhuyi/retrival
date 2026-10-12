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
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsEinsum(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    std::string equation_value;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "equation" && attr.type() == ge::onnx::AttributeProto::STRING) {
            equation_value = attr.s();
        }
    }
    if (equation_value.empty()) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Attr of equation must be not null.");
        return FAILED;
    }
    int input_size = node->input_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.SetAttr("equation", equation_value);
    op_dest.SetAttr("N", input_size);
    return SUCCESS;
}
// register Einsum op info to GE
REGISTER_CUSTOM_OP("Einsum")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Einsum"), ge::AscendString("ai.onnx::9::Einsum"),
         ge::AscendString("ai.onnx::10::Einsum"), ge::AscendString("ai.onnx::11::Einsum"),
         ge::AscendString("ai.onnx::12::Einsum"), ge::AscendString("ai.onnx::13::Einsum"),
         ge::AscendString("ai.onnx::14::Einsum"), ge::AscendString("ai.onnx::15::Einsum"),
         ge::AscendString("ai.onnx::16::Einsum"), ge::AscendString("ai.onnx::17::Einsum"),
         ge::AscendString("ai.onnx::18::Einsum")})
    .ParseParamsFn(ParseParamsEinsum)
    .ImplyType(ImplyType::TVM);
} // namespace domi
