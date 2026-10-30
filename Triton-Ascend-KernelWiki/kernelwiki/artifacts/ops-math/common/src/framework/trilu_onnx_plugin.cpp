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
#include "op_math_proto_extend.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace {
constexpr int64_t DIGIT_TWO = 2;
} // namespace

namespace domi {
static Status ParseParamsTrilu(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int upper = 1;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "upper" && attr.type() == ge::onnx::AttributeProto::INT) {
            upper = attr.i();
        }
    }
    int node_input_size = node->input_size();
    if (!(node_input_size == 1 || node_input_size == DIGIT_TWO)) {
        OP_LOGE(GetOpName(op_dest).c_str(), "now only support one or two input");
    }
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("upper", upper);
    op_dest.SetAttr("original_type", "ai.onnx::14::Trilu");
    op_dest.DynamicInputRegister("x", node_input_size);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static Status ParseOpToGraphTrilu(const ge::Operator& op_dest, ge::Graph& graph)
{
    std::string ori_name;
    if (op_dest.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op_dest).c_str(), "get name from op_dest failed.");
        return FAILED;
    }
    auto input_node_num = op_dest.GetInputsSize();
    int upper = 1;
    op_dest.GetAttr("upper", upper);
    ge::Operator trilu;

    if (input_node_num == 1) {
        auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
        trilu = op::Trilu(ori_name.c_str()).set_input_x(data0).set_attr_upper(upper);
        std::vector<Operator> inputs{data0};
        std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
        output_indexs.emplace_back(trilu, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else if (input_node_num == DIGIT_TWO) {
        auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
        auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
        trilu = op::Trilu(ori_name.c_str()).set_input_x(data0).set_input_k(data1).set_attr_upper(upper);
        std::vector<Operator> inputs{data0, data1};
        std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
        output_indexs.emplace_back(trilu, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else {
        OP_LOGE(GetOpName(op_dest).c_str(), "input node number can only be 1 or 2.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::14::Trilu"), ge::AscendString("ai.onnx::15::Trilu"),
         ge::AscendString("ai.onnx::16::Trilu"), ge::AscendString("ai.onnx::17::Trilu"),
         ge::AscendString("ai.onnx::18::Trilu")})
    .ParseParamsFn(ParseParamsTrilu)
    .ParseOpToGraphFn(ParseOpToGraphTrilu)
    .ImplyType(ImplyType::TVM);
} // namespace domi
