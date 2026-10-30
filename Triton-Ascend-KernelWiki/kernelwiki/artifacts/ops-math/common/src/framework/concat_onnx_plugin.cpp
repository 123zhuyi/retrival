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
#include "conversion/identity/op_graph/identity_proto.h"
#include "conversion/concat_d/op_graph/concat_d_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
static const int DEFAULT_CONCAT_DIM = 0;

static Status OpConcatUpdateInfo(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int concat_dim = 0;
    bool set_axis_flag = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axis" && attr.type() == ge::onnx::AttributeProto::INT) {
            concat_dim = attr.i();
            set_axis_flag = true;
            break;
        }
    }

    if (!set_axis_flag) {
        OP_LOGE(GetOpName(op_dest).c_str(), "onnx Concat op has no axis attr.");
        concat_dim = DEFAULT_CONCAT_DIM;
        return PARAM_INVALID;
    }
    op_dest.SetAttr("concat_dim", concat_dim);
    op_dest.SetAttr("name", node->name());

    int n = node->input_size();
    op_dest.SetAttr("N", n);
    op_dest.DynamicInputRegister("x", n);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static Status ParseParamsConcatCall(const Message* op_src, ge::Operator& op_dest)
{
    if (OpConcatUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::Concat");
    return SUCCESS;
}

static Status ParseOpToGraphConcat(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int input_size = 0;
    op.GetAttr("N", input_size);
    std::vector<ge::Operator> inputs;
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;

    if (input_size == 0) {
        OP_LOGE(GetOpName(op).c_str(), "input_size must ge 1");
        return FAILED;
    } else if (input_size == 1) {
        auto data_op = op::Data((ori_name + "_data").c_str()).set_attr_index(0);
        auto identity_op = op::Identity((ori_name + "_identity").c_str()).set_input_x(data_op);
        inputs.push_back(data_op);
        output_indexs.emplace_back(identity_op, std::vector<size_t>{0});
    } else {
        int concat_dim = DEFAULT_CONCAT_DIM;
        op.GetAttr("concat_dim", concat_dim);

        auto concat_op = op::ConcatD((ori_name + "_ConcatD").c_str())
                             .create_dynamic_input_x(input_size)
                             .set_attr_concat_dim(concat_dim)
                             .set_attr_N(input_size);

        std::string input_name = "";
        for (int i = 0; i < input_size; ++i) {
            input_name = "data_concat_" + to_string(i);
            auto data_op = op::Data((ori_name + input_name).c_str()).set_attr_index(i);
            concat_op.set_dynamic_input_x(i, data_op);
            inputs.push_back(data_op);
        }
        output_indexs.emplace_back(concat_op, std::vector<size_t>{0});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Concat"), ge::AscendString("ai.onnx::9::Concat"),
         ge::AscendString("ai.onnx::10::Concat"), ge::AscendString("ai.onnx::11::Concat"),
         ge::AscendString("ai.onnx::12::Concat"), ge::AscendString("ai.onnx::13::Concat"),
         ge::AscendString("ai.onnx::14::Concat"), ge::AscendString("ai.onnx::15::Concat"),
         ge::AscendString("ai.onnx::16::Concat"), ge::AscendString("ai.onnx::17::Concat"),
         ge::AscendString("ai.onnx::18::Concat")})
    .ParseParamsFn(ParseParamsConcatCall)
    .ParseOpToGraphFn(ParseOpToGraphConcat)
    .ImplyType(ImplyType::TVM);
} // namespace domi
