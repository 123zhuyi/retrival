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
#include "math/square_sum_v1/op_graph/square_sum_v1_proto.h"

using namespace ge;
using ge::Operator;

namespace domi {
static Status ParseParamsReduceSumSquare(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int input_size = node->input_size();
    int output_size = node->output_size();
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("output", output_size);
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceSumSquare");

    std::vector<int64_t> axis_attr = {};
    bool keep_dims_attr = true;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                axis_attr.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims_attr = (attr.i() == 1);
        }
    }
    op_dest.SetAttr("axis", axis_attr);
    op_dest.SetAttr("keep_dims", keep_dims_attr);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphReduceSumSquare(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto reduceSquareSum = op::SquareSumV1((ori_name + "_SquareSumV1").c_str()).set_input_x(data0);

    std::vector<int64_t> axis = {};
    if (op.GetAttr("axis", axis) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get axis from op failed");
        return FAILED;
    }
    reduceSquareSum.set_attr_axis(axis);

    bool keep_dims = false;
    if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }
    reduceSquareSum.set_attr_keep_dims(keep_dims);
    reduceSquareSum.set_attr_noop_with_empty_axes(false);

    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(reduceSquareSum, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

static Status ParseParamsReduceSumSquare13(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("ReduceSumSquare13", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::13::ReduceSumSquare");

    int input_size = node->input_size();
    std::vector<int64_t> axis = {};
    bool keep_dims = true;
    int noop_with_empty_axes = 0;
    for (const auto& attr : node->attribute()) {
        // 兼容版本13后，任然会有将axes作为属性传入的情况
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                axis.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        } else if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT) {
            noop_with_empty_axes = attr.i();
        }
    }
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("input_size", input_size);
    op_dest.SetAttr("axis", axis);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.SetAttr("noop_with_empty_axes", noop_with_empty_axes);
    return SUCCESS;
}

namespace {
struct ReduceSumSquare13Prop {
    std::string ori_name;
    int input_num = 1;
    std::vector<int64_t> axis = {};
    bool keep_dims = false;
    int empty_axes = 0;
};

Status GetProperty(const Operator& op, ReduceSumSquare13Prop& prop)
{
    if (op.GetAttr("name", prop.ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    if (op.GetAttr("input_size", prop.input_num) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get input_num from op failed");
        return FAILED;
    }

    if (op.GetAttr("axis", prop.axis) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get axis from op failed");
        return FAILED;
    }

    if (op.GetAttr("keep_dims", prop.keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }

    if (op.GetAttr("noop_with_empty_axes", prop.empty_axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attribute noop_with_empty_axes failed");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

static Status ParseOpToGraphReduceSumSquare13(const Operator& op, Graph& graph)
{
    ReduceSumSquare13Prop prop;
    if (GetProperty(op, prop) != SUCCESS) {
        return FAILED;
    }
    auto data0 = op::Data((prop.ori_name + "_data0").c_str()).set_attr_index(0);
    auto reduceSquareSum = op::SquareSumV1((prop.ori_name + "_SquareSumV1").c_str())
                               .set_input_x(data0)
                               .set_attr_axis(prop.axis)
                               .set_attr_keep_dims(prop.keep_dims)
                               .set_attr_noop_with_empty_axes(prop.empty_axes);
    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    output_indexs.emplace_back(reduceSquareSum, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);

    return SUCCESS;
}

// register SquareSumV1 op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::1::ReduceSumSquare"), ge::AscendString("ai.onnx::8::ReduceSumSquare"),
                   ge::AscendString("ai.onnx::9::ReduceSumSquare"), ge::AscendString("ai.onnx::10::ReduceSumSquare"),
                   ge::AscendString("ai.onnx::11::ReduceSumSquare"), ge::AscendString("ai.onnx::12::ReduceSumSquare")})
    .ParseParamsFn(ParseParamsReduceSumSquare)
    .ParseOpToGraphFn(ParseOpToGraphReduceSumSquare)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("SquareSumV1")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::13::ReduceSumSquare"), ge::AscendString("ai.onnx::14::ReduceSumSquare"),
                   ge::AscendString("ai.onnx::15::ReduceSumSquare"), ge::AscendString("ai.onnx::16::ReduceSumSquare"),
                   ge::AscendString("ai.onnx::17::ReduceSumSquare"), ge::AscendString("ai.onnx::18::ReduceSumSquare")})
    .ParseParamsFn(ParseParamsReduceSumSquare13)
    .ParseOpToGraphFn(ParseOpToGraphReduceSumSquare13)
    .ImplyType(ImplyType::TVM);
} // namespace domi
