/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "math/add/op_graph/add_proto.h"
#include "math/mod/op_graph/mod_proto.h"
#include "math/one_hot/op_graph/one_hot_proto.h"
#include "math/cast/op_graph/cast_proto.h"
#include "conversion/identity/op_graph/identity_proto.h"
#include "conversion/split/op_graph/split_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;

static Status ParseParamsOnehotCall(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int axis = -1;
    for (auto& attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
            break;
        }
    }
    const int input = 3;
    const int output = 1;
    op_dest.SetAttr("axis", axis);
    op_dest.SetAttr("name", node->name());
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);
    op_dest.SetAttr("original_type", "ai.onnx::11::OneHot");
    return SUCCESS;
}

// OneHot算子不支持bool类型的on_value/off_value，values为bool时前后插Cast转为int8处理
// values为图输入时从入参desc获取类型；values为常量(Initializer)时desc未填充，从常量数据获取类型
static bool IsValuesBool(const ge::Operator& op)
{
    ge::TensorDesc values_desc = op.GetDynamicInputDesc("x", 2);
    if (values_desc.GetDataType() == ge::DT_BOOL) {
        return true;
    }
    ge::Tensor values_const_data;
    return op.GetInputConstData("x2", values_const_data) == ge::GRAPH_SUCCESS &&
           values_const_data.GetTensorDesc().GetDataType() == ge::DT_BOOL;
}

static ge::Operator BuildValuesOp(const std::string& ori_name, ge::Operator values_data, bool is_values_bool)
{
    if (!is_values_bool) {
        return values_data;
    }
    return op::Cast((ori_name + "_Cast_values").c_str()).set_input_x(values_data).set_attr_dst_type(ge::DT_INT8);
}

static ge::Operator BuildOutputOp(const std::string& ori_name, ge::Operator onehot_op, bool is_values_bool)
{
    if (!is_values_bool) {
        return onehot_op;
    }
    return op::Cast((ori_name + "_Cast_output").c_str()).set_input_x(onehot_op, 0).set_attr_dst_type(ge::DT_BOOL);
}

static Status ParseOpToGraphOnehot(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int axis = -1;
    op.GetAttr("axis", axis);

    auto data1 = op::Data((ori_name + "_indicate").c_str()).set_attr_index(0);
    auto data2 = op::Data((ori_name + "_depth").c_str()).set_attr_index(1);
    auto data3 = op::Data((ori_name + "_values").c_str()).set_attr_index(2);

    bool is_values_bool = IsValuesBool(op);
    ge::Operator values_op = BuildValuesOp(ori_name, data3, is_values_bool);

    int32_t split_dim = 0;
    ge::Tensor scalar_split_dim = CreateScalar(split_dim, ge::DT_INT32);
    auto split_const_op = op::Const((ori_name + "_split_dim").c_str()).set_attr_value(scalar_split_dim);
    // In order to solve the problem that the input is negative insert add and mod, becase tbe onehot not support but
    // onnx support
    auto split_d_op = op::Split((ori_name + "_Split").c_str())
                          .create_dynamic_output_y(2)
                          .set_input_x(values_op)
                          .set_input_split_dim(split_const_op)
                          .set_attr_num_split(2);

    auto identity_data1 = op::Identity((ori_name + "_identity_1").c_str()).set_input_x(data1);
    auto identity_data2 = op::Identity((ori_name + "_identity_2").c_str()).set_input_x(data2);
    auto add_op = op::Add((ori_name + "_Add").c_str()).set_input_x1(identity_data1, 0).set_input_x2(identity_data2, 0);
    auto mod_op = op::Mod((ori_name + "_Mod").c_str()).set_input_x1(add_op, 0).set_input_x2(identity_data2, 0);

    auto onehot_op = op::OneHot((ori_name + "_OneHot").c_str())
                         .set_input_x(mod_op, 0)
                         .set_input_depth(identity_data2, 0)
                         .set_input_on_value(split_d_op, 1)
                         .set_input_off_value(split_d_op, 0)
                         .set_attr_axis(axis);
    ge::Operator output_op = BuildOutputOp(ori_name, onehot_op, is_values_bool);
    std::vector<ge::Operator> inputs{data1, data2, data3};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(output_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::9::OneHot"), ge::AscendString("ai.onnx::10::OneHot"),
                   ge::AscendString("ai.onnx::11::OneHot"), ge::AscendString("ai.onnx::12::OneHot"),
                   ge::AscendString("ai.onnx::13::OneHot"), ge::AscendString("ai.onnx::14::OneHot"),
                   ge::AscendString("ai.onnx::15::OneHot"), ge::AscendString("ai.onnx::16::OneHot"),
                   ge::AscendString("ai.onnx::17::OneHot"), ge::AscendString("ai.onnx::18::OneHot")})
    .ParseParamsFn(ParseParamsOnehotCall)
    .ParseOpToGraphFn(ParseOpToGraphOnehot)
    .ImplyType(ImplyType::TVM);
} // namespace domi
