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
#include "math/cast/op_graph/cast_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsCast(const Message* op_src, ge::Operator& op_dst)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool bfind_to = false;
    for (auto attr : node->attribute()) {
        if (attr.name() == "to") {
            bfind_to = true;
            int onnx = attr.i();
            int32_t om = GetOmDtypeFromOnnxDtype(onnx);
            if (om == -1) {
                OP_LOGE(GetOpName(op_dst).c_str(), "TransDataTypeFromOnnxToOm failed, onnx = %d", onnx);
                return FAILED;
            }
            op_dst.SetAttr("dst_type", om);
            break;
        }
    }

    if (!bfind_to) {
        OP_LOGE(GetOpName(op_dst).c_str(), "Message op_src do not have attribute to.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("Cast")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::8::Cast"),
                 ge::AscendString("ai.onnx::9::Cast"),
                 ge::AscendString("ai.onnx::10::Cast"),
                 ge::AscendString("ai.onnx::11::Cast"),
                 ge::AscendString("ai.onnx::12::Cast"),
                 ge::AscendString("ai.onnx::13::Cast"),
                 ge::AscendString("ai.onnx::14::Cast"),
                 ge::AscendString("ai.onnx::15::Cast"),
                 ge::AscendString("ai.onnx::16::Cast"),
                 ge::AscendString("ai.onnx::17::Cast"),
                 ge::AscendString("ai.onnx::18::Cast")})
  .ParseParamsFn(ParseParamsCast)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
