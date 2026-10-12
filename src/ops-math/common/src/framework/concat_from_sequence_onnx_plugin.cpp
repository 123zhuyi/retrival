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

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamConcatFromSequence(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    // required
    int axis = 0;
    // optional
    int new_axis = 0;
    for (auto attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
        } else if (attr.name() == "new_axis") {
            new_axis = attr.i();
        }
    }
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("axis", axis);
    op_dest.SetAttr("new_axis", new_axis);
    op_dest.SetAttr("original_type", "ai.onnx::11::ConcatFromSequence");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("ConcatFromSequence")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::ConcatFromSequence"), ge::AscendString("ai.onnx::12::ConcatFromSequence"),
         ge::AscendString("ai.onnx::13::ConcatFromSequence"), ge::AscendString("ai.onnx::14::ConcatFromSequence"),
         ge::AscendString("ai.onnx::15::ConcatFromSequence"), ge::AscendString("ai.onnx::16::ConcatFromSequence"),
         ge::AscendString("ai.onnx::17::ConcatFromSequence"), ge::AscendString("ai.onnx::18::ConcatFromSequence")})
    .ParseParamsFn(ParseParamConcatFromSequence)
    .ImplyType(ImplyType::TVM);
} // namespace domi