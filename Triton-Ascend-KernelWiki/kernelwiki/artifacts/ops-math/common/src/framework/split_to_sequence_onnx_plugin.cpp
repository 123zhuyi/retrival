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

static Status ParseParamSplitToSequence(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    // required
    int axis = 0;
    // optional
    int keep_dims = 1;
    for (auto attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
        } else if (attr.name() == "keepdims") {
            keep_dims = attr.i();
        }
    }
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("axis", axis);
    op_dest.SetAttr("keepdims", keep_dims);
    op_dest.SetAttr("original_type", "ai.onnx::11::SplitToSequence");
    return SUCCESS;
}

REGISTER_CUSTOM_OP("SplitToSequence")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::SplitToSequence"), ge::AscendString("ai.onnx::12::SplitToSequence"),
         ge::AscendString("ai.onnx::13::SplitToSequence"), ge::AscendString("ai.onnx::14::SplitToSequence"),
         ge::AscendString("ai.onnx::15::SplitToSequence"), ge::AscendString("ai.onnx::16::SplitToSequence"),
         ge::AscendString("ai.onnx::17::SplitToSequence"), ge::AscendString("ai.onnx::18::SplitToSequence")})
    .ParseParamsFn(ParseParamSplitToSequence)
    .ImplyType(ImplyType::TVM);
} // namespace domi