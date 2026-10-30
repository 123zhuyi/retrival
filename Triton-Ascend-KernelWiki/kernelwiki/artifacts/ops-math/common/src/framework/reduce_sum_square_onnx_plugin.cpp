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

static Status parse_params_reduce_sum_square(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int32_t> v_axes;
    bool keep_dims = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                v_axes.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            if (attr.i() != 0) {
                keep_dims = true;
            }
        }
        if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            OP_LOGW(GetOpName(op_dest).c_str(), "Only support noop_with_empty_axes=0, but 1 is obtained now");
        }
    }

    op_dest.SetAttr("axis", v_axes);
    op_dest.SetAttr("keep_dims", keep_dims);
    return SUCCESS;
}

// register ReduceSumSquare op info to GE
REGISTER_CUSTOM_OP("SquareSumV1")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::1::ReduceSumSquare"), ge::AscendString("ai.onnx::8::ReduceSumSquare"),
         ge::AscendString("ai.onnx::9::ReduceSumSquare"), ge::AscendString("ai.onnx::10::ReduceSumSquare"),
         ge::AscendString("ai.onnx::11::ReduceSumSquare"), ge::AscendString("ai.onnx::12::ReduceSumSquare"),
         ge::AscendString("ai.onnx::13::ReduceSumSquare"), ge::AscendString("ai.onnx::14::ReduceSumSquare"),
         ge::AscendString("ai.onnx::15::ReduceSumSquare"), ge::AscendString("ai.onnx::16::ReduceSumSquare"),
         ge::AscendString("ai.onnx::17::ReduceSumSquare"), ge::AscendString("ai.onnx::18::ReduceSumSquare")})
    .ParseParamsFn(parse_params_reduce_sum_square)
    .ImplyType(ImplyType::TVM);
} // namespace domi
