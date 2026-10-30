/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

// Required attr judge
static const int REQ_ATTR_NUM = 2;

static Status ParseParamsNpuSignBitsUnpack(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    ge::AscendString op_name;
    (void)op_dest.GetName(op_name);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    // The initialization of required attribute count
    int req_attr_count = 0;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "size" && attr.type() == ge::onnx::AttributeProto::INT) {
            int size = attr.i();
            op_dest.SetAttr("size", size);
            ++req_attr_count;
        } else if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
            int dtype = attr.i();
            op_dest.SetAttr("dtype", dtype);
            ++req_attr_count;
        }
    }

    // Node must have required attribute size and dtype.
    if (req_attr_count != REQ_ATTR_NUM) {
        OP_LOGE(op_name.GetString(), "Node must have attr size and dtype.");
        return FAILED;
    }

    return SUCCESS;
}

// register SignBitsUnpack op info to GE
REGISTER_CUSTOM_OP("SignBitsUnpack")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUSignBitsUnpack"), 
                   ge::AscendString("ai.onnx::11::NPUSignBitsUnpack"), 
                   ge::AscendString("ai.onnx::12::NPUSignBitsUnpack"),
                   ge::AscendString("ai.onnx::13::NPUSignBitsUnpack"), 
                   ge::AscendString("ai.onnx::14::NPUSignBitsUnpack"), 
                   ge::AscendString("ai.onnx::15::NPUSignBitsUnpack"),
                   ge::AscendString("ai.onnx::16::NPUSignBitsUnpack"), 
                   ge::AscendString("ai.onnx::17::NPUSignBitsUnpack"),
                   ge::AscendString("ai.onnx::18::NPUSignBitsUnpack")})
    .ParseParamsFn(ParseParamsNpuSignBitsUnpack)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
