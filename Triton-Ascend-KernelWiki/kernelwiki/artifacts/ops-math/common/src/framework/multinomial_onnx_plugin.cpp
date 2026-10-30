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

using namespace std;
using namespace ge;
using ge::Operator;
namespace domi {
using NodeProto = ge::onnx::NodeProto;
using OpDesc = std::shared_ptr<ge::OpDesc>;
static const int DATA_TYPE_INT32 = 6;
static Status ParseParamsMultinomialCall(const Message* op_src, ge::Operator& op_dest)
{
    const NodeProto* node = reinterpret_cast<const NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    int data_type = DATA_TYPE_INT32;
    float seed = 0;
    int sample_size = 1;
    for (auto attr : node->attribute()) {
        if (attr.name() == "dtype") {
            data_type = attr.i();
        } else if (attr.name() == "sample_size") {
            sample_size = attr.i();
        } else if (attr.name() == "seed") {
            seed = attr.f();
        }
    }
    op_dest.SetAttr("dtype", data_type);
    op_dest.SetAttr("sample_size", sample_size);
    op_dest.SetAttr("seed", seed);
    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("original_type", "ai.onnx::11::Multinomial");
    return SUCCESS;
}

static Status ParseOpToGraphMultinomial(const ge::Operator& op, Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    int sample_size = 1;
    int data_type = DATA_TYPE_INT32;
    float seed = 0;
    ge::Operator& new_op = const_cast<ge::Operator&>(op);
    new_op.SetAttr("sample_size", sample_size);
    new_op.SetAttr("dtype", data_type);
    new_op.SetAttr("seed", seed);
    ge::DataType dtype_om = GetOmDtypeFromOnnxDtype(data_type);
    if (dtype_om == ge::DT_UNDEFINED) {
        OP_LOGE(GetOpName(op).c_str(), "dtype[%d] is wrong,please select right dtype", data_type);
        return FAILED;
    }
    int int_seed = static_cast<int>(seed);
    ge::Tensor scalar_const_value = CreateScalar(sample_size, ge::DT_INT32);
    auto data_op = op::Data((ori_name + "_input1").c_str()).set_attr_index(0);
    auto const_op = op::Const((ori_name + "_data1").c_str()).set_attr_value(scalar_const_value);
    auto muti_op = op::Multinomial((ori_name + "_Muti").c_str())
                       .set_input_logits(data_op)
                       .set_input_num_samples(const_op)
                       .set_attr_dtype(dtype_om)
                       .set_attr_seed(int_seed);
    std::vector<ge::Operator> inputs{data_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
    output_indexs.emplace_back(muti_op, std::vector<size_t>{0});
    graph.SetInputs(inputs).SetOutputs(output_indexs);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::Multinomial"), ge::AscendString("ai.onnx::9::Multinomial"),
         ge::AscendString("ai.onnx::10::Multinomial"), ge::AscendString("ai.onnx::11::Multinomial"),
         ge::AscendString("ai.onnx::12::Multinomial"), ge::AscendString("ai.onnx::13::Multinomial"),
         ge::AscendString("ai.onnx::14::Multinomial"), ge::AscendString("ai.onnx::15::Multinomial"),
         ge::AscendString("ai.onnx::16::Multinomial"), ge::AscendString("ai.onnx::17::Multinomial"),
         ge::AscendString("ai.onnx::18::Multinomial")})
    .ParseParamsFn(ParseParamsMultinomialCall)
    .ParseOpToGraphFn(ParseOpToGraphMultinomial)
    .ImplyType(ImplyType::TVM);
} // namespace domi
