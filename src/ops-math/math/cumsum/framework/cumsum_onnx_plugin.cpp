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
#include "stub_ops.h"
#include "math/cumsum/op_graph/cumsum_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsCumSum(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool exclusive = false;
    bool reverse = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "exclusive" && attr.type() == ge::onnx::AttributeProto::INT) {
            exclusive = (attr.i() == 1);
        } else if (attr.name() == "reverse" && attr.type() == ge::onnx::AttributeProto::INT) {
            reverse = (attr.i() == 1);
        }
    }
    op_dest.SetAttr("exclusive", exclusive);
    op_dest.SetAttr("reverse", reverse);

    return SUCCESS;
}

// register CumSum op info to GE
REGISTER_CUSTOM_OP("Cumsum")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::CumSum"),
                 ge::AscendString("ai.onnx::9::CumSum"),
                 ge::AscendString("ai.onnx::10::CumSum"),
                 ge::AscendString("ai.onnx::11::CumSum"),
                 ge::AscendString("ai.onnx::12::CumSum"),
                 ge::AscendString("ai.onnx::13::CumSum"),
                 ge::AscendString("ai.onnx::14::CumSum"),
                 ge::AscendString("ai.onnx::15::CumSum"),
                 ge::AscendString("ai.onnx::16::CumSum"),
                 ge::AscendString("ai.onnx::17::CumSum"),
                 ge::AscendString("ai.onnx::18::CumSum")})
  .ParseParamsFn(ParseParamsCumSum)
  .ImplyType(ImplyType::TVM);
}  // namespace domi