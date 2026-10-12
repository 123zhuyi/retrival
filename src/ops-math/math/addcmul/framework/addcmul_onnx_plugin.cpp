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
#include "math/add/op_graph/add_proto.h"
#include "math/mul/op_graph/mul_proto.h"
#include "math/muls/op_graph/muls_proto.h"

namespace domi {

static Status ParseParamsAddcmul(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    float value = 0.0;
    auto attrs = node->attribute();
    auto it = std::find_if(attrs.begin(), attrs.end(), [](const ge::onnx::AttributeProto& attr) -> bool {
        return attr.name() == "value";
    });
    if (it != attrs.end()) {
        value = it->f();
    } else {
        OP_LOGE(GetOpName(op_dest).c_str(), "get attr value failed.");
        return FAILED;
    }
    const int input = 3;
    const int output = 1;
    op_dest.DynamicInputRegister("args", input);
    op_dest.DynamicOutputRegister("output", output);
    op_dest.SetAttr("original_type", "torch.onnx.symbolic_opset11.addcmul");
    op_dest.SetAttr("value", value);
    return SUCCESS;
}

static Status ParseOpToGraph(const ge::Operator& op, ge::Graph& graph)
{
    auto data_0 = ge::op::Data().set_attr_index(0);
    auto data_1 = ge::op::Data().set_attr_index(1);
    auto data_2 = ge::op::Data().set_attr_index(2);
    float value = 0.0;
    if (op.GetAttr("value", value) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr value failed.");
        return FAILED;
    }
    auto mul = ge::op::Mul("Mul");
    mul.set_input_x1(data_1);
    mul.set_input_x2(data_2);
    auto muls = ge::op::Muls("Muls");
    muls.set_input_x(mul);
    muls.set_attr_value(value);
    auto add = ge::op::Add("Add");
    add.set_input_x1(data_0);
    add.set_input_x2(muls);

    std::vector<ge::Operator> inputs{data_0, data_1, data_2};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(add, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(std::vector<ge::AscendString>{ge::AscendString("torch.onnx.symbolic_opset11.addcmul")})
    .ParseParamsFn(ParseParamsAddcmul)
    .ParseOpToGraphFn(ParseOpToGraph)
    .ImplyType(ImplyType::GELOCAL);
}  // namespace domi
