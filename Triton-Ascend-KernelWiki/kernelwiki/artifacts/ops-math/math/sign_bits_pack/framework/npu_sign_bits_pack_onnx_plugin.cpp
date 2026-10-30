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
// Required attr judge
static const int REQ_ATTR_NUM = 1;
static Status ParseParamsNpuSignBitsPack(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int size = 0;
    // The initialization of required attribute count
    int req_attr_count = 0;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "size" && attr.type() == ge::onnx::AttributeProto::INT) {
            size = attr.i();
            op_dest.SetAttr("size", size);
            ++req_attr_count;
        }
    }

    // Node must have required attribute size.
    if (req_attr_count != REQ_ATTR_NUM) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attr size");
        return FAILED;
    }

    return SUCCESS;
}

// register SignBitsPack op info to GE
REGISTER_CUSTOM_OP("SignBitsPack")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::11::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::12::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::13::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::14::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::15::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::16::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::17::NPUSignBitsPack"),
                   ge::AscendString("ai.onnx::18::NPUSignBitsPack")})
    .ParseParamsFn(ParseParamsNpuSignBitsPack)
    .ImplyType(ImplyType::TVM);
}  // namespace domi