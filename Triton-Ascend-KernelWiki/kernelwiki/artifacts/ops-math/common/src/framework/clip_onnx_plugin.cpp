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
#include "math/cast/op_graph/cast_proto.h"
#include "math/minimum/op_graph/minimum_proto.h"
#include "math/maximum/op_graph/maximum_proto.h"
#include "conversion/clip_by_value/op_graph/clip_by_value_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace {
constexpr int ATTR_NUM = 2;
constexpr int NO_MIN_IDX = 1;
constexpr int NO_MAX_IDX = 2;
} // namespace

namespace domi {

static Status ParseParamsClipV9(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    // 1.add dynamic input and out
    const int input = 3;
    const int output = 1;
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("output", output);

    // 2.set original_type
    op_dest.SetAttr("original_type", "ai.onnx::9::Clip");

    // attr max and min default value
    float max = 3.402823e+38;
    float min = -3.402823e+38;

    for (const auto& attr : node->attribute()) {
        if (attr.name() == "max" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            max = attr.f();
        } else if (attr.name() == "min" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            min = attr.f();
        }
    }

    ge::Tensor scalar_max = CreateScalar(max, ge::DT_FLOAT);
    op_dest.SetAttr("max", scalar_max);
    ge::Tensor scalar_min = CreateScalar(min, ge::DT_FLOAT);
    op_dest.SetAttr("min", scalar_min);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphClipV9(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    ge::Tensor value1;
    op.GetAttr("max", value1);
    ge::Tensor value2;
    op.GetAttr("min", value2);

    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(value1);
    auto data2 = op::Const((ori_name + "_data2").c_str()).set_attr_value(value2);
    auto cast_op = op::Cast((ori_name + "_cast").c_str()).set_input_x(data0).set_attr_dst_type(DT_FLOAT);
    auto clip_by_value = op::ClipByValue((ori_name + "_ClipByValue").c_str())
                             .set_input_x(cast_op)
                             .set_input_clip_value_min(data2)
                             .set_input_clip_value_max(data1);

    std::vector<Operator> inputs{data0};
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    output_indexs.emplace_back(clip_by_value, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

static Status ParseParamsClipV11(const Message* op_src, ge::Operator& op_dest)
{
    // 1.add dynamic input and out
    const int input = 3;
    const int output = 1;
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("output", output);

    // 2.set original_type
    op_dest.SetAttr("original_type", "ai.onnx::11::Clip");

    // 3.set attr if needed
    const ge::onnx::NodeProto* node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    bool no_max = false;
    bool no_min = false;
    int num = node->input_size();
    bool check_input_bad = false;
    if (num < ATTR_NUM) {
        check_input_bad = true;
    } else if (num == ATTR_NUM) {
        no_min = (node->input(NO_MIN_IDX) == "");
        no_max = true;
    } else {
        no_min = (node->input(NO_MIN_IDX) == "");
        no_max = (node->input(NO_MAX_IDX) == "");
    }
    if (no_min && no_max) {
        check_input_bad = true;
    }
    if (check_input_bad) {
        OP_LOGE(GetOpName(op_dest).c_str(), "At least 'min' or 'max' must not be None or Empty");
        return FAILED;
    }

    op_dest.SetAttr("no_max", no_max);
    op_dest.SetAttr("no_min", no_min);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphClipV11(const Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }
    auto data0 = op::Data((ori_name + "_x").c_str()).set_attr_index(0);
    bool no_max = true;
    op.GetAttr("no_max", no_max);
    bool no_min = true;
    op.GetAttr("no_min", no_min);

    Operator input_op1 = data0;
    Operator output_op;
    std::vector<Operator> inputs;
    std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
    if (!no_min && !no_max) { // go to ClipByValue branch
        Operator input_op2 = op::Data((ori_name + "_min").c_str()).set_attr_index(1);
        Operator input_op3 = op::Data((ori_name + "_max").c_str()).set_attr_index(2);
        output_op = op::ClipByValue((ori_name + "_ClipByValue").c_str())
                        .set_input_x(input_op1)
                        .set_input_clip_value_min(input_op2)
                        .set_input_clip_value_max(input_op3);
        inputs = {input_op1, input_op2, input_op3};
    } else if (no_min && !no_max) { // go to minimum branch
        Operator input_op3 = op::Data((ori_name + "_max").c_str()).set_attr_index(2);
        output_op = op::Minimum((ori_name + "_Minimum").c_str()).set_input_x1(input_op1).set_input_x2(input_op3);
        inputs = {input_op1, input_op3};
    } else if (!no_min && no_max) { // go to maximum branch
        Operator input_op2 = op::Data((ori_name + "_min").c_str()).set_attr_index(1);
        output_op = op::Maximum((ori_name + "_Maximum").c_str()).set_input_x1(input_op1).set_input_x2(input_op2);
        inputs = {input_op1, input_op2};
    }
    output_indexs.emplace_back(output_op, vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

// register Clip op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Clip"), ge::AscendString("ai.onnx::9::Clip"),
         ge::AscendString("ai.onnx::10::Clip")})
    .ParseParamsFn(ParseParamsClipV9)
    .ParseOpToGraphFn(ParseOpToGraphClipV9)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::Clip"), ge::AscendString("ai.onnx::12::Clip"),
         ge::AscendString("ai.onnx::13::Clip"), ge::AscendString("ai.onnx::14::Clip"),
         ge::AscendString("ai.onnx::15::Clip"), ge::AscendString("ai.onnx::16::Clip"),
         ge::AscendString("ai.onnx::17::Clip"), ge::AscendString("ai.onnx::18::Clip")})
    .ParseParamsFn(ParseParamsClipV11)
    .ParseOpToGraphFn(ParseOpToGraphClipV11)
    .ImplyType(ImplyType::TVM);
} // namespace domi
