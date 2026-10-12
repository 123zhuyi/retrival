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
static Status parseParamsRepeatWithPad(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic RepeatWithPad op_src to NodeProto failed.");
        return FAILED;
    }

    int axis = 0;
    int capacity = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
        }
        if (attr.name() == "capacity") {
            capacity = attr.i();
        }
    }
    op_dest.SetAttr("capacity", capacity);
    op_dest.SetAttr("axis", axis);
    return SUCCESS;
}

// register RepeatWithPad op info to GE
REGISTER_CUSTOM_OP("RepeatWithPad")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::RepeatWithPad"), ge::AscendString("ai.onnx::9::RepeatWithPad"),
         ge::AscendString("ai.onnx::10::RepeatWithPad"), ge::AscendString("ai.onnx::11::RepeatWithPad"),
         ge::AscendString("ai.onnx::12::RepeatWithPad"), ge::AscendString("ai.onnx::13::RepeatWithPad"),
         ge::AscendString("ai.onnx::14::RepeatWithPad"), ge::AscendString("ai.onnx::15::RepeatWithPad"),
         ge::AscendString("ai.onnx::16::RepeatWithPad"), ge::AscendString("ai.onnx::17::RepeatWithPad"),
         ge::AscendString("ai.onnx::18::RepeatWithPad")})
    .ParseParamsFn(parseParamsRepeatWithPad)
    .ImplyType(ImplyType::TVM);
} // namespace domi
