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
#include "math/reduce_mean/op_graph/reduce_mean_proto.h"

using namespace ge;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamsReduceMean(const Message* op_src, ge::Operator& op_dest)
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

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("axes", axes);
    op_dest.SetAttr("keep_dims", keep_dims);
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::ReduceMean");

    return SUCCESS;
}

static Status ParseOpToGraphReduceMean(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

    std::vector<int> axes = {};
    if (op.GetAttr("axes", axes) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get axes from op failed");
        return FAILED;
    }

    int num = axes.size();
    std::vector<int64_t> dims = {};
    if (num != 0) {
        dims.push_back(num);
    } else {
        dims.push_back(0);
    }
    ge::Tensor axes_tensor = Vec2Tensor(axes, dims, ge::DT_INT32, ge::FORMAT_ND);

    auto data1 = op::Const((ori_name + "_data1").c_str()).set_attr_value(axes_tensor);
    auto Reducemean = op::ReduceMean((ori_name + "_ReduceMean").c_str()).set_input_x(data0).set_input_axes(data1);

    bool keep_dims = false;
    if (op.GetAttr("keep_dims", keep_dims) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get keep_dims from op failed");
        return FAILED;
    }
    Reducemean.set_attr_keep_dims(keep_dims);
    Reducemean.set_attr_noop_with_empty_axes(false);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(Reducemean, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

// register ReduceMean op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::ReduceMean"),
                   ge::AscendString("ai.onnx::9::ReduceMean"),
                   ge::AscendString("ai.onnx::10::ReduceMean"),
                   ge::AscendString("ai.onnx::11::ReduceMean"),
                   ge::AscendString("ai.onnx::12::ReduceMean"),
                   ge::AscendString("ai.onnx::13::ReduceMean"),
                   ge::AscendString("ai.onnx::14::ReduceMean"),
                   ge::AscendString("ai.onnx::15::ReduceMean"),
                   ge::AscendString("ai.onnx::16::ReduceMean"),
                   ge::AscendString("ai.onnx::17::ReduceMean"),
                   ge::AscendString("ai.onnx::18::ReduceMean")})
    .ParseParamsFn(ParseParamsReduceMean)
    .ParseOpToGraphFn(ParseOpToGraphReduceMean)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
