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
#include "op_math_proto_extend.h"
#include "math/right_shift/op_graph/right_shift_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsBitShift(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    const int input = 2;
    const int output = 1;
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);
    // 2.set original_type
    op_dest.SetAttr("original_type", "ai.onnx::11::BitShift");

    std::string direction = "";
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "direction" && attr.type() == ge::onnx::AttributeProto::STRING) {
            direction = attr.s();
        } else {
            OP_LOGE(GetOpName(op_dest).c_str(), "direction attr is empty");
            return FAILED;
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("direction", direction);
    return SUCCESS;
}

static Status ParseOpToGraphBitShift(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
    std::string direction = "RIGHT";
    if (op.GetAttr("direction", direction) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get value from op failed");
        return FAILED;
    }

    std::vector<Operator> inputs{data0, data1};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    if (direction == "RIGHT") {
        auto bitshift_r = op::RightShift((ori_name + "_RightShift").c_str()).set_input_x(data0).set_input_y(data1);
        output_indexs.emplace_back(bitshift_r, vector<std::size_t>{0});
    } else if (direction == "LEFT") {
        auto bitshift_l = op::LeftShift((ori_name + "_LeftShift").c_str()).set_input_x(data0).set_input_y(data1);
        output_indexs.emplace_back(bitshift_l, vector<std::size_t>{0});
    } else {
        OP_LOGE(GetOpName(op).c_str(), "attr direction is error");
        return FAILED;
    }

    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register BitShift op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("ai.onnx::11::BitShift"),
                 ge::AscendString("ai.onnx::12::BitShift"),
                 ge::AscendString("ai.onnx::13::BitShift"),
                 ge::AscendString("ai.onnx::14::BitShift"),
                 ge::AscendString("ai.onnx::15::BitShift"),
                 ge::AscendString("ai.onnx::16::BitShift"),
                 ge::AscendString("ai.onnx::17::BitShift"),
                 ge::AscendString("ai.onnx::18::BitShift")})
  .ParseParamsFn(ParseParamsBitShift)
  .ParseOpToGraphFn(ParseOpToGraphBitShift)
  .ImplyType(ImplyType::TVM);
}  // namespace domi
