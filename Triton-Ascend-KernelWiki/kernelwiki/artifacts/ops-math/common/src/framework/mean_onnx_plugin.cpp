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
#include "math/accumulate_nv2/op_graph/accumulate_nv2_proto.h"
#include "math/muls/op_graph/muls_proto.h"

using ge::Operator;
using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsMean(const Message* op_src, ge::Operator& op_dest)
{
    op_dest.SetAttr("original_type", "ai.onnx::9::Mean");
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed");
        return FAILED;
    }
    int n_num = node->input_size();

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("N", n_num);
    op_dest.DynamicInputRegister("x", n_num);
    op_dest.DynamicOutputRegister("y", 1);

    return SUCCESS;
}

static Status ParseOpToGraphMean(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int n_num = 0;
    if (op.GetAttr("N", n_num) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attribute N failed");
        return FAILED;
    }
    std::vector<ge::Operator> inputs;
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexs;
    if (n_num == 0) {
        OP_LOGE("ParseOpToGraphMean", "input size must greater than 1");
        return FAILED;
    } else {
        auto acc =
            op::AccumulateNV2((ori_name + "_AccumulateNV2").c_str()).create_dynamic_input_x(n_num).set_attr_N(n_num);
        std::string input_name = "";
        for (int i = 0; i < n_num; ++i) {
            input_name = "data_mean_" + to_string(i);
            auto data_op = op::Data((ori_name + input_name).c_str()).set_attr_index(i);
            acc.set_dynamic_input_x(i, data_op);
            inputs.push_back(data_op);
        }

        float num = 1.0 / n_num;
        auto div_op = op::Muls((ori_name + "_Muls").c_str()).set_input_x(acc).set_attr_value(num);
        output_indexs.emplace_back(div_op, std::vector<size_t>{0});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Mean"), ge::AscendString("ai.onnx::9::Mean"),
         ge::AscendString("ai.onnx::10::Mean"), ge::AscendString("ai.onnx::11::Mean"),
         ge::AscendString("ai.onnx::12::Mean"), ge::AscendString("ai.onnx::13::Mean"),
         ge::AscendString("ai.onnx::14::Mean"), ge::AscendString("ai.onnx::15::Mean"),
         ge::AscendString("ai.onnx::16::Mean"), ge::AscendString("ai.onnx::17::Mean"),
         ge::AscendString("ai.onnx::18::Mean")})
    .ParseParamsFn(ParseParamsMean)
    .ParseOpToGraphFn(ParseOpToGraphMean)
    .ImplyType(ImplyType::TVM);
} // namespace domi
