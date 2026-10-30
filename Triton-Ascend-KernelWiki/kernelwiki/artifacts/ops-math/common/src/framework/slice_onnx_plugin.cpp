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
#include "conversion/strided_slice_v2/op_graph/strided_slice_v2_proto.h"

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
static Status ParseParamSlice(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    return SUCCESS;
}

static Status ParseParamSliceCall(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    std::vector<int64_t> ends = {};
    std::vector<int64_t> starts = {};
    std::vector<int64_t> axes = {};

    bool is_have_axes = false;
    for (auto attr : node->attribute()) {
        if (attr.name() == "ends") {
            int num = attr.ints_size();
            for (int i = 0; i < num; ++i) {
                ends.push_back(attr.ints(i));
            }
        } else if (attr.name() == "starts") {
            int num = attr.ints_size();
            for (int i = 0; i < num; ++i) {
                starts.push_back(attr.ints(i));
            }
        } else if (attr.name() == "axes") {
            is_have_axes = true;
            int num = attr.ints_size();
            for (int i = 0; i < num; ++i) {
                axes.push_back(attr.ints(i));
            }
            op_dest.SetAttr("axes", axes);
        }
    }
    if (ends.empty() || starts.empty()) {
        OP_LOGE(GetOpName(op_dest).c_str(), "attr must have ends and starts");
        return FAILED;
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("end", ends);
    op_dest.SetAttr("begin", starts);
    op_dest.SetAttr("is_have_axes", is_have_axes);
    op_dest.SetAttr("original_type", "ai.onnx::9::Slice");
    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    return SUCCESS;
}

static ge::Operator MakeConstOp(const ge::Operator& op, const std::string& attr_name)
{
    std::vector<int64_t> val = {};
    op.GetAttr(attr_name.c_str(), val);
    std::vector<int64_t> dims(1, val.size());
    ge::Tensor tensor = Vec2Tensor(val, dims, ge::DT_INT64);
    ge::Operator const_op = op::Const(attr_name.c_str()).set_attr_value(tensor);
    return const_op;
}

static Status ParseOpToGraphSlice(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data_op = op::Data((ori_name + "_input1").c_str()).set_attr_index(0);
    auto const_op = MakeConstOp(op, "begin");
    auto const_op1 = MakeConstOp(op, "end");
    auto slice_op = op::StridedSliceV2((ori_name + "_StridedSliceV2").c_str())
                        .set_input_x(data_op)
                        .set_input_begin(const_op)
                        .set_input_end(const_op1);

    bool is_have_axes = false;
    op.GetAttr("is_have_axes", is_have_axes);
    if (is_have_axes) {
        auto const_op2 = MakeConstOp(op, "axes");
        slice_op.set_input_axes(const_op2);
    }

    std::vector<ge::Operator> inputs{data_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(slice_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        std::vector<ge::AscendString>{ge::AscendString("ai.onnx::8::Slice"), ge::AscendString("ai.onnx::9::Slice")})
    .ParseParamsFn(ParseParamSliceCall)
    .ParseOpToGraphFn(ParseOpToGraphSlice)
    .ImplyType(ImplyType::TVM);

REGISTER_CUSTOM_OP("StridedSliceV2")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::11::Slice"), ge::AscendString("ai.onnx::10::Slice"),
         ge::AscendString("ai.onnx::12::Slice"), ge::AscendString("ai.onnx::13::Slice"),
         ge::AscendString("ai.onnx::14::Slice"), ge::AscendString("ai.onnx::15::Slice"),
         ge::AscendString("ai.onnx::16::Slice"), ge::AscendString("ai.onnx::17::Slice"),
         ge::AscendString("ai.onnx::18::Slice")})
    .ParseParamsFn(ParseParamSlice)
    .ImplyType(ImplyType::TVM);
} // namespace domi
