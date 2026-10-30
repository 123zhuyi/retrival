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
#include "conversion/slice/op_graph/slice_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsSlice(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    int op_input_size = node->input_size();
    int op_output_size = node->output_size();
    op_dest.DynamicInputRegister("x", op_input_size);
    op_dest.DynamicOutputRegister("y", op_output_size);

    op_dest.SetAttr("original_type", "npu::1::NPUSlice");

    std::vector<int> offsets = {};
    std::vector<int> sizes = {};
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "offsetss" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                offsets.push_back(attr.ints(i));
            }
        } else if (attr.name() == "sizes" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int i = 0; i < attr.ints_size(); i++) {
                sizes.push_back(attr.ints(i));
            }
        }
    }

    int64_t len_offsets = offsets.size();
    int64_t len_sizes = sizes.size();
    std::vector<int64_t> dims_offsets = {};
    dims_offsets.push_back(len_offsets);
    std::vector<int64_t> dims_sizes = {};
    dims_sizes.push_back(len_sizes);

    ge::Tensor tensor1 = Vec2Tensor(offsets, dims_offsets, ge::DT_INT32, ge::FORMAT_ND);
    ge::Tensor tensor2 = Vec2Tensor(sizes, dims_sizes, ge::DT_INT32, ge::FORMAT_ND);
    op_dest.SetAttr("offsetss", tensor1);
    op_dest.SetAttr("sizes", tensor2);
    op_dest.SetAttr("name", node->name());

    return SUCCESS;
}

static Status ParseOpToGraphSlice(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

    ge::Tensor const_value_1;
    if (op.GetAttr("offsetss", const_value_1) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get offsetss from op failed");
        return FAILED;
    }

    ge::Tensor const_value_2;
    if (op.GetAttr("sizes", const_value_2) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get sizes from op failed");
        return FAILED;
    }

    auto const_op_1 = ge::op::Const((ori_name + "_const_data_1").c_str()).set_attr_value(const_value_1);
    auto const_op_2 = ge::op::Const((ori_name + "_const_data_2").c_str()).set_attr_value(const_value_2);

    auto Slice = ge::op::Slice((ori_name + "_Slice").c_str())
                     .set_input_x(data0)
                     .set_input_offsets(const_op_1)
                     .set_input_size(const_op_2);

    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t> > > outputs;
    outputs.emplace_back(Slice, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);

    return SUCCESS;
}

// register StrideAdd op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUSlice"),
                 ge::AscendString("ai.onnx::11::NPUSlice"),
                 ge::AscendString("ai.onnx::12::NPUSlice"),
                 ge::AscendString("ai.onnx::13::NPUSlice"),
                 ge::AscendString("ai.onnx::14::NPUSlice"),
                 ge::AscendString("ai.onnx::15::NPUSlice"),
                 ge::AscendString("ai.onnx::16::NPUSlice"),
                 ge::AscendString("ai.onnx::17::NPUSlice"),
                 ge::AscendString("ai.onnx::18::NPUSlice")})
  .ParseParamsFn(ParseParamsSlice)
  .ParseOpToGraphFn(ParseOpToGraphSlice)
  .ImplyType(ImplyType::TVM);
} // namespace domi