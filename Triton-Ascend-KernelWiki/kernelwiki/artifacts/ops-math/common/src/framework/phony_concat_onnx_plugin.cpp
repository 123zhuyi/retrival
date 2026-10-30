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

static Status ParseParamsPhonyConcat(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int64_t> concat_dim = {0};
    std::vector<int64_t> N = {node->input_size()};
    bool keep_input_offset = true;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "concat_dim" && attr.type() == ge::onnx::AttributeProto::INTS) {
            int num = attr.ints_size();
            if (num > 0) {
                concat_dim.clear();
            }
            for (int i = 0; i < num; ++i) {
                concat_dim.push_back(attr.ints(i));
            }
        } else if (attr.name() == "N" && attr.type() == ge::onnx::AttributeProto::INTS) {
            int num = attr.ints_size();
            if (num > 0) {
                N.clear();
            }
            for (int i = 0; i < num; ++i) {
                N.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keep_input_offset" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() == 0) {
                keep_input_offset = false;
            }
        }
    }

    op_dest.SetAttr("concat_dim", concat_dim);
    op_dest.SetAttr("N", N);
    op_dest.SetAttr("keep_input_offset", keep_input_offset);
    op_dest.SetAttr("original_type", "ai.onnx::12::PhonyConcat");
    op_dest.DynamicInputRegister("x", node->input_size());

    return SUCCESS;
}

REGISTER_CUSTOM_OP("PhonyConcat")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::PhonyConcat"), ge::AscendString("ai.onnx::9::PhonyConcat"),
         ge::AscendString("ai.onnx::10::PhonyConcat"), ge::AscendString("ai.onnx::11::PhonyConcat"),
         ge::AscendString("ai.onnx::12::PhonyConcat"), ge::AscendString("ai.onnx::13::PhonyConcat"),
         ge::AscendString("ai.onnx::14::PhonyConcat"), ge::AscendString("ai.onnx::15::PhonyConcat"),
         ge::AscendString("ai.onnx::16::PhonyConcat"), ge::AscendString("ai.onnx::17::PhonyConcat"),
         ge::AscendString("ai.onnx::18::PhonyConcat")})
    .ParseParamsFn(ParseParamsPhonyConcat)
    .ImplyType(ImplyType::TVM);
} // namespace domi
