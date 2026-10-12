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

static Status ParseParamsPhonySplit(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        ge::AscendString op_name;
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int64_t> split_dim = {0};
    std::vector<int64_t> num_split = {node->output_size()};
    bool keep_output_offset = true;

    for (auto attr : node->attribute()) {
        if (attr.name() == "split_dim" && attr.type() == ge::onnx::AttributeProto::INTS) {
            int num = attr.ints_size();
            if (num > 0) {
                split_dim.clear();
            }
            for (int i = 0; i < num; ++i) {
                split_dim.push_back(attr.ints(i));
            }
        } else if (attr.name() == "num_split" && attr.type() == ge::onnx::AttributeProto::INTS) {
            int num = attr.ints_size();
            if (num > 0) {
                num_split.clear();
            }
            for (int i = 0; i < num; ++i) {
                num_split.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keep_output_offset" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() == 0) {
                keep_output_offset = false;
            }
        }
    }

    op_dest.SetAttr("split_dim", split_dim);
    op_dest.SetAttr("num_split", num_split);
    op_dest.SetAttr("keep_output_offset", keep_output_offset);
    op_dest.SetAttr("original_type", "ai.onnx::12::PhonySplit");
    op_dest.DynamicOutputRegister("y", node->output_size());

    return SUCCESS;
}

REGISTER_CUSTOM_OP("PhonySplit")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::PhonySplit"), ge::AscendString("ai.onnx::9::PhonySplit"),
         ge::AscendString("ai.onnx::10::PhonySplit"), ge::AscendString("ai.onnx::11::PhonySplit"),
         ge::AscendString("ai.onnx::12::PhonySplit"), ge::AscendString("ai.onnx::13::PhonySplit"),
         ge::AscendString("ai.onnx::14::PhonySplit"), ge::AscendString("ai.onnx::15::PhonySplit"),
         ge::AscendString("ai.onnx::16::PhonySplit"), ge::AscendString("ai.onnx::17::PhonySplit"),
         ge::AscendString("ai.onnx::18::PhonySplit")})
    .ParseParamsFn(ParseParamsPhonySplit)
    .ImplyType(ImplyType::TVM);
} // namespace domi
