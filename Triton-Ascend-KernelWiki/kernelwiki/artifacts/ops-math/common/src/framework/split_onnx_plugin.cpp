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
#include "conversion/split_v/op_graph/split_v_proto.h"
#include "conversion/split/op_graph/split_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;

namespace domi {
static Status ParseParamsSplitAttribute(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    int axis = 0;
    int num_outputs = 0;
    std::vector<int64_t> split;
    for (auto attr : node->attribute()) {
        if (attr.name() == "axis") {
            axis = attr.i();
        } else if (attr.name() == "split") {
            for (auto dim : attr.ints()) {
                split.push_back(dim);
            }
        }
        if (attr.name() == "num_outputs") {
            num_outputs = attr.i();
        }
    }
    int input_size = node->input_size();
    int output_size = node->output_size();
    int split_size = (int)split.size();
    if (split_size != 0 && split_size != output_size) {
        OP_LOGE(GetOpName(op_dest).c_str(), "output size[%d] is not equal split size[%d].", output_size, split_size);
        return FAILED;
    }
    if (num_outputs != 0 && num_outputs != output_size) {
        OP_LOGE(GetOpName(op_dest).c_str(), "output size[%d] is not equal num_outputs[%d].", output_size, num_outputs);
        return FAILED;
    }
    op_dest.SetAttr("split_dim", axis);
    op_dest.SetAttr("size_splits", split);
    op_dest.SetAttr("input_size", input_size);
    op_dest.SetAttr("num_outputs", num_outputs);
    op_dest.SetAttr("num_split", output_size);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseParamsSplitNew(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    int input_size = node->input_size();
    int output_size = node->output_size();
    if (ParseParamsSplitAttribute(op_src, op_dest) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("original_type", "ai.onnx::13::Split");
    op_dest.DynamicInputRegister("x", input_size);
    op_dest.DynamicOutputRegister("y", output_size);
    return SUCCESS;
}

namespace {
struct SplitNewProp {
    std::string ori_name;
    int32_t split_dim = 0;
    int num_split = 0;
    int num_outputs = 0;
    int input_size = 1;
};

Status GetProperty(const ge::Operator& op, SplitNewProp& prop)
{
    std::string ori_name;
    if (op.GetAttr("name", prop.ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    if (op.GetAttr("split_dim", prop.split_dim) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr split_dim from op failed!.");
        return FAILED;
    }

    if (op.GetAttr("num_split", prop.num_split) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr num_split from op failed!.");
        return FAILED;
    }

    if (op.GetAttr("num_outputs", prop.num_outputs) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr num_outputs from op failed!.");
        return FAILED;
    }

    if (op.GetAttr("input_size", prop.input_size) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr num_split from op failed!.");
        return FAILED;
    }
    return SUCCESS;
}

Status SingleSizeParse(const ge::Operator& op, const SplitNewProp& prop, ge::Operator& const_split_dim_2, Graph& graph)
{
    std::vector<int64_t> size_splits;
    ge::Operator split_op;
    if (op.GetAttr("size_splits", size_splits) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get attr size_splits from op failed!.");
        return FAILED;
    }
    auto input_x_0 = op::Data((prop.ori_name + "_x").c_str()).set_attr_index(0);
    if (prop.num_outputs == 0 && size_splits.empty()) {
        split_op = op::Split((prop.ori_name + "_Split").c_str())
                       .create_dynamic_output_y(prop.num_split)
                       .set_input_x(input_x_0)
                       .set_input_split_dim(const_split_dim_2)
                       .set_attr_num_split(prop.num_split);
    } else {
        ge::Tensor size_splits_tensor;
        if (size_splits.empty()) {
            std::vector<int64_t> input_dim = op.GetInputDesc(0).GetShape().GetDims();
            int64_t dim_size = input_dim[prop.split_dim];
            int64_t split_size = dim_size / prop.num_split;
            std::vector<int64_t> size_splits_num(prop.num_split, split_size);
            if (dim_size % prop.num_split != 0) {
                OP_LOGE(
                    GetOpName(op).c_str(), "dim_size[%ld] can not be evenly divided by split_size[%ld].", dim_size,
                    split_size);
                return FAILED;
            }
            std::vector<int64_t> dims = {(int64_t)size_splits_num.size()};
            size_splits_tensor = Vec2Tensor(size_splits_num, dims, ge::DT_INT64);
        } else {
            std::vector<int64_t> dims = {(int64_t)size_splits.size()};
            size_splits_tensor = Vec2Tensor(size_splits, dims, ge::DT_INT64);
        }
        auto const_size_splits = op::Const((prop.ori_name + "_size_splits").c_str()).set_attr_value(size_splits_tensor);
        split_op = op::SplitV((prop.ori_name + "_SplitV").c_str())
                       .create_dynamic_output_y(prop.num_split)
                       .set_input_x(input_x_0)
                       .set_input_size_splits(const_size_splits)
                       .set_input_split_dim(const_split_dim_2)
                       .set_attr_num_split(prop.num_split);
    }
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < (std::size_t)prop.num_split; i++) {
        idx.push_back(i);
    }
    std::vector<ge::Operator> inputs = {input_x_0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(split_op, idx);
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

} // namespace

static Status ParseOpToGraphSplitNew(const ge::Operator& op, Graph& graph)
{
    SplitNewProp prop;
    if (GetProperty(op, prop) != SUCCESS) {
        return FAILED;
    }

    ge::Tensor scalar_split_dim = CreateScalar(prop.split_dim, ge::DT_INT32);
    auto const_split_dim_2 = op::Const((prop.ori_name + "_split_dim").c_str()).set_attr_value(scalar_split_dim);

    if (prop.input_size == 1) {
        if (SingleSizeParse(op, prop, const_split_dim_2, graph) != SUCCESS) {
            return FAILED;
        }
    } else {
        auto input_x_0 = op::Data((prop.ori_name + "_x").c_str()).set_attr_index(0);
        auto input_size_splits_1 = op::Data((prop.ori_name + "_size_splits").c_str()).set_attr_index(1);
        auto split_v = op::SplitV((prop.ori_name + "_SplitV").c_str())
                           .create_dynamic_output_y(prop.num_split)
                           .set_input_x(input_x_0)
                           .set_input_size_splits(input_size_splits_1)
                           .set_input_split_dim(const_split_dim_2)
                           .set_attr_num_split(prop.num_split);
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < (std::size_t)prop.num_split; i++) {
            idx.push_back(i);
        }

        std::vector<ge::Operator> inputs{input_x_0, input_size_splits_1};
        std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
        output_indexs.emplace_back(split_v, idx);
        graph.SetInputs(inputs).SetOutputs(output_indexs);
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Split"), ge::AscendString("ai.onnx::9::Split"),
         ge::AscendString("ai.onnx::10::Split"), ge::AscendString("ai.onnx::11::Split"),
         ge::AscendString("ai.onnx::12::Split"), ge::AscendString("ai.onnx::13::Split"),
         ge::AscendString("ai.onnx::14::Split"), ge::AscendString("ai.onnx::15::Split"),
         ge::AscendString("ai.onnx::16::Split"), ge::AscendString("ai.onnx::17::Split"),
         ge::AscendString("ai.onnx::18::Split")})
    .ParseParamsFn(ParseParamsSplitNew)
    .ParseOpToGraphFn(ParseOpToGraphSplitNew)
    .ImplyType(ImplyType::TVM);
} // namespace domi