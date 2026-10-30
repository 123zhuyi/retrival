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
#include "conversion/identity/op_graph/identity_proto.h"
#include "math/minimum/op_graph/minimum_proto.h"

using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status OpMinUpdateInfo(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int n = node->input_size();
    op_dest.SetAttr("N", n);
    op_dest.SetAttr("name", node->name());
    op_dest.DynamicInputRegister("x", n);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static Status ParseParamsMinCall(const Message* op_src, ge::Operator& op_dest)
{
    if (OpMinUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::11::Min");
    return SUCCESS;
}

static Status ParseParamsMinCallV9(const Message* op_src, ge::Operator& op_dest)
{
    if (OpMinUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::9::Min");
    return SUCCESS;
}

static Status ParseParamsMinCallV12(const Message* op_src, ge::Operator& op_dest)
{
    if (OpMinUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::12::Min");
    return SUCCESS;
}

static Status ParseParamsMinCallV13(const Message* op_src, ge::Operator& op_dest)
{
    if (OpMinUpdateInfo(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::13::Min");
    return SUCCESS;
}

static Status ParseOpToGraphMin(const ge::Operator& op, Graph& graph)
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
        auto data_op = op::Data((ori_name + "_data_0").c_str()).set_attr_index(0);
        auto identity_op = op::Identity((ori_name + "_identity").c_str()).set_input_x(data_op);
        inputs.push_back(data_op);
        output_indexs.emplace_back(identity_op, std::vector<size_t>{0});
    } else {
        for (int i = 0; i < input_size; ++i) {
            auto data_op = op::Data((ori_name + "_data_" + to_string(i)).c_str()).set_attr_index(i);
            inputs.push_back(data_op);
        }
        ge::Operator& prefix_op = inputs[0];
        for (int i = 1; i < input_size; ++i) {
            auto cur_op = op::Minimum((ori_name + "_Minimum_" + to_string(i)).c_str())
                              .set_input_x1(prefix_op)
                              .set_input_x2(inputs[i]);
            prefix_op = cur_op;
        }
        output_indexs.emplace_back(prefix_op, std::vector<size_t>{0});
    }
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(std::vector<ge::AscendString>{ge::AscendString("ai.onnx::11::Min")})
    .ParseParamsFn(ParseParamsMinCall)
    .ParseOpToGraphFn(ParseOpToGraphMin)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Min"),
                   ge::AscendString("ai.onnx::9::Min"),
                   ge::AscendString("ai.onnx::10::Min")})
    .ParseParamsFn(ParseParamsMinCallV9)
    .ParseOpToGraphFn(ParseOpToGraphMin)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(std::vector<ge::AscendString>{ge::AscendString("ai.onnx::12::Min")})
    .ParseParamsFn(ParseParamsMinCallV12)
    .ParseOpToGraphFn(ParseOpToGraphMin)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::13::Min"),
                   ge::AscendString("ai.onnx::14::Min"),
                   ge::AscendString("ai.onnx::15::Min"),
                   ge::AscendString("ai.onnx::16::Min"),
                   ge::AscendString("ai.onnx::17::Min"),
                   ge::AscendString("ai.onnx::18::Min")})
    .ParseParamsFn(ParseParamsMinCallV13)
    .ParseOpToGraphFn(ParseOpToGraphMin)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
