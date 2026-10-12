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
#include "math/reduce_prod/op_graph/reduce_prod_proto.h"

using namespace ge;
using ge::Operator;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsReduceProd(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    std::vector<int> axes = {};
    bool keep_dims = true;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                axes.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        }
        if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
            OP_LOGW(GetOpName(op_dest).c_str(), "Only support noop_with_empty_axes=0, but 1 is obtained now");
        }
    }
    int num = axes.size();
    std::vector<int64_t> dims = {};
    if (num != 0) {
        dims.push_back(num);
    } else {
        dims.push_back(0);
    }
    ge::Tensor tensor = Vec2Tensor(axes, dims, ge::DT_INT32, ge::FORMAT_NCHW);

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("axes", tensor);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.SetAttr("noop_with_empty_axes", 0);
    const int input = 2;
    const int output = 1;
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceProd");

    return SUCCESS;
}

static Status ParseOpToGraphReduceProd(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

    ge::Tensor axes;
    if (op.GetAttr("axes", axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get axes from op failed");
        return FAILED;
    }

    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(axes);
    auto reduceprod = op::ReduceProd((ori_name + "_ReduceProd").c_str()).set_input_x(data0).set_input_axes(data1);

    bool keep_dims = false;
    if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }
    reduceprod.set_attr_keep_dims(keep_dims);

    int noop_with_empty_axes = 0;
    if (op.GetAttr("noop_with_empty_axes", noop_with_empty_axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get noop_with_empty_axes from op failed");
        return FAILED;
    }
    reduceprod.set_attr_noop_with_empty_axes(noop_with_empty_axes);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(reduceprod, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

static Status ParseParamsReduceProd13(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE("ReduceProd13", "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::13::ReduceProd");

    int input_size = node->input_size();
    std::vector<int> axes = {};
    bool keep_dims = true;
    int noop_with_empty_axes = 0;
    for (const auto& attr : node->attribute()) {
        // 兼容版本13后，任然会有将axes作为属性传入的情况
        if (attr.name() == "axes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                axes.push_back(attr.ints(i));
            }
        } else if (attr.name() == "keepdims" && attr.type() == ge::onnx::AttributeProto::INT) {
            keep_dims = (attr.i() == 1);
        } else if (attr.name() == "noop_with_empty_axes" && attr.type() == ge::onnx::AttributeProto::INT) {
            noop_with_empty_axes = attr.i();
        }
    }

    // opset 13+: axes changed from attribute to input.
    // When input_size == 1, there is no axes input; store an empty axes tensor
    // so ParseOpToGraph can retrieve it via GetAttr and pass to Const.
    int num = axes.size();
    std::vector<int64_t> dims = {};
    if (num != 0) {
        dims.push_back(num);
    } else {
        dims.push_back(0);
    }
    ge::Tensor axes_tensor = Vec2Tensor(axes, dims, ge::DT_INT32, ge::FORMAT_NCHW);
    op_dest.SetAttr("axes", axes_tensor);
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("input_size", input_size);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.SetAttr("noop_with_empty_axes", noop_with_empty_axes);
    return SUCCESS;
}

namespace {
struct ReduceProd13Prop {
    std::string ori_name;
    bool keep_dims = false;
    int input_num = 1;
    int empty_axes = 0;
};

Status GetProperty(const Operator& op, ReduceProd13Prop& prop)
{
    if (op.GetAttr("name", prop.ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    if (op.GetAttr("keep_dims", prop.keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }

    if (op.GetAttr("input_size", prop.input_num) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get input_num from op failed");
        return FAILED;
    }

    if (op.GetAttr("noop_with_empty_axes", prop.empty_axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attribute noop_with_empty_axes failed");
        return FAILED;
    }
    return SUCCESS;
}

} // namespace

static Status ParseOpToGraphReduceProd13(const Operator& op, Graph& graph)
{
    ReduceProd13Prop prop;
    if (GetProperty(op, prop) != SUCCESS) {
        return FAILED;
    }
    auto data0 = op::Data((prop.ori_name + "_data0").c_str()).set_attr_index(0);
    int num_input = 2;
    if (prop.input_num == 1) {
        ge::Tensor axes;
        if (op.GetAttr("axes", axes) != SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "get axes from op failed");
            return FAILED;
        }
        auto data1 = op::Const((prop.ori_name + "_data1").c_str()).set_attr_value(axes);
        auto reduceprod = op::ReduceProd((prop.ori_name + "_ReduceProd").c_str())
                              .set_input_x(data0)
                              .set_input_axes(data1)
                              .set_attr_keep_dims(prop.keep_dims)
                              .set_attr_noop_with_empty_axes(prop.empty_axes);
        std::vector<Operator> inputs{data0};
        std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
        output_indexs.emplace_back(reduceprod, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else if (prop.input_num == num_input) {
        auto data1 = op::Data((prop.ori_name + "_data1").c_str()).set_attr_index(1);
        auto reduceprod13 = op::ReduceProd((prop.ori_name + "_ReduceProd").c_str())
                                .set_input_x(data0)
                                .set_input_axes(data1)
                                .set_attr_keep_dims(prop.keep_dims)
                                .set_attr_noop_with_empty_axes(prop.empty_axes);
        std::vector<Operator> inputs{data0, data1};
        std::vector<std::pair<Operator, std::vector<size_t> > > output_indexs;
        output_indexs.emplace_back(reduceprod13, vector<std::size_t>{0});
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    } else {
        OP_LOGE(GetOpName(op).c_str(), "Input num or set attr is invalid");
        return FAILED;
    }
    return SUCCESS;
}

// register ReduceProd op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ReduceProd"), ge::AscendString("ai.onnx::9::ReduceProd"),
                   ge::AscendString("ai.onnx::10::ReduceProd"), ge::AscendString("ai.onnx::11::ReduceProd"),
                   ge::AscendString("ai.onnx::12::ReduceProd")})
    .ParseParamsFn(ParseParamsReduceProd)
    .ParseOpToGraphFn(ParseOpToGraphReduceProd)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("ReduceProd")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::13::ReduceProd"), ge::AscendString("ai.onnx::14::ReduceProd"),
                   ge::AscendString("ai.onnx::15::ReduceProd"), ge::AscendString("ai.onnx::16::ReduceProd"),
                   ge::AscendString("ai.onnx::17::ReduceProd"), ge::AscendString("ai.onnx::18::ReduceProd")})
    .ParseParamsFn(ParseParamsReduceProd13)
    .ParseOpToGraphFn(ParseOpToGraphReduceProd13)
    .ImplyType(ImplyType::TVM);
} // namespace domi
