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
#include "math/cast/op_graph/cast_proto.h"
#include "math/bitwise_xor/op_graph/bitwise_xor_proto.h"

using namespace ge;
namespace domi {

static Status parseParamsXor(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (nullptr == node) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }
    const int input = 2;
    const int output = 1;
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("original_type", "ai.onnx::11::Xor");
    op_dest.DynamicInputRegister("x", input);
    op_dest.DynamicOutputRegister("y", output);

    return SUCCESS;
}

static Status ParseOpToGraphXor(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto data1 = op::Data((ori_name + "_data1").c_str()).set_attr_index(1);

    auto cast_a = op::Cast((ori_name + "_Cast_a").c_str()).set_input_x(data0).set_attr_dst_type(ge::DT_INT32);
    auto cast_b = op::Cast((ori_name + "_Cast_b").c_str()).set_input_x(data1).set_attr_dst_type(ge::DT_INT32);
    auto bit_wise_xor = op::BitwiseXor((ori_name + "_Xor").c_str()).set_input_x1(cast_a).set_input_x2(cast_b);
    auto cast_out = op::Cast((ori_name + "_Cast_out").c_str()).set_input_x(bit_wise_xor).set_attr_dst_type(ge::DT_BOOL);

    std::vector<ge::Operator> inputs = {data0, data1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(cast_out, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::Xor"),
                   ge::AscendString("ai.onnx::9::Xor"),
                   ge::AscendString("ai.onnx::10::Xor"),
                   ge::AscendString("ai.onnx::11::Xor"),
                   ge::AscendString("ai.onnx::12::Xor"),
                   ge::AscendString("ai.onnx::13::Xor"),
                   ge::AscendString("ai.onnx::14::Xor"),
                   ge::AscendString("ai.onnx::15::Xor"),
                   ge::AscendString("ai.onnx::16::Xor"),
                   ge::AscendString("ai.onnx::17::Xor"),
                   ge::AscendString("ai.onnx::18::Xor")})
    .ParseParamsFn(parseParamsXor)
    .ParseOpToGraphFn(ParseOpToGraphXor)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
