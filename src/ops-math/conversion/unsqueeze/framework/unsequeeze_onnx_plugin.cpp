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

using domi::ONNX;

namespace domi {

static Status ParseParamsUnsqueeze(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int64_t> axes = {};
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                axes.push_back(attr.ints(i));
            }
        }
    }
    if (axes.size() == 0) {
        OP_LOGE(GetOpName(op_dest).c_str(), "axes is a required argument, please check it!");
        return PARAM_INVALID;
    }
    op_dest.SetAttr("axes", axes);
    return SUCCESS;
}

static Status ParseParamsUnsqueezeV3(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        ge::AscendString op_name;
        (void)op_dest.GetName(op_name);
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    return SUCCESS;
}

// register Add op info to GE
REGISTER_CUSTOM_OP("Unsqueeze")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Unsqueeze"),
                   ge::AscendString("ai.onnx::9::Unsqueeze"),
                   ge::AscendString("ai.onnx::10::Unsqueeze"),
                   ge::AscendString("ai.onnx::11::Unsqueeze"),
                   ge::AscendString("ai.onnx::12::Unsqueeze")})
    .ParseParamsFn(ParseParamsUnsqueeze)
    .ImplyType(ImplyType::GELOCAL);

REGISTER_CUSTOM_OP("UnsqueezeV3")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::13::Unsqueeze"),
                   ge::AscendString("ai.onnx::14::Unsqueeze"),
                   ge::AscendString("ai.onnx::15::Unsqueeze"),
                   ge::AscendString("ai.onnx::16::Unsqueeze"),
                   ge::AscendString("ai.onnx::17::Unsqueeze"),
                   ge::AscendString("ai.onnx::18::Unsqueeze")})
    .ParseParamsFn(ParseParamsUnsqueezeV3)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi