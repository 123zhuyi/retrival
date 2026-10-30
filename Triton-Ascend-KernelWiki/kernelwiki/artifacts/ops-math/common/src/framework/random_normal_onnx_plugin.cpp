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
#include "math/muls/op_graph/muls_proto.h"
#include "math/shape/op_graph/shape_proto.h"

using namespace ge;
namespace domi {
static Status ParseParamsRandomNormal(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::RandomNormal");

    int dtype = 1;
    float mean = 0.0f;
    float scale = 1.0f;
    int seed = 0;
    std::vector<int> shape_list;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "dtype" && attr.type() == ge::onnx::AttributeProto::INT) {
            dtype = attr.i();
        } else if (attr.name() == "mean" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            mean = attr.f();
        } else if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            scale = attr.f();
        } else if (attr.name() == "seed" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            seed = (int)attr.f();
        } else if (attr.name() == "shape" && attr.type() == ge::onnx::AttributeProto::INTS) {
            for (int dim : attr.ints()) {
                shape_list.push_back(dim);
            }
        };
    }
    if (shape_list.empty()) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Attr of shape must be not null.");
        return FAILED;
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("mean", mean);
    op_dest.SetAttr("scale", scale);
    op_dest.SetAttr("seed", seed);
    op_dest.SetAttr("dtype", dtype);
    op_dest.SetAttr("shape", shape_list);
    return SUCCESS;
}

static Status ParseOpToGraphRandomNormal(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
        return FAILED;
    }

    float mean = 0.0f;
    op.GetAttr("mean", mean);

    float scale = 1.0f;
    op.GetAttr("scale", scale);

    int dtype = 1;
    op.GetAttr("dtype", dtype);

    std::vector<int> shape;
    op.GetAttr("shape", shape);

    if (shape.empty()) {
        OP_LOGE(GetOpName(op).c_str(), "Attr of shape must be not null.");
        return FAILED;
    }

    // cast from onnx dtype to tbe dtype
    std::map<int, ge::DataType> kvlist = {{1, ge::DT_FLOAT}, {10, ge::DT_FLOAT16}, {11, ge::DT_DOUBLE}};
    if (kvlist.find(dtype) == kvlist.end()) {
        OP_LOGE(GetOpName(op).c_str(), "only support float32/half/double, but got %d", dtype);
        return FAILED;
    }

    int seed = 0;
    op.GetAttr("seed", seed);

    ge::TensorDesc tensorDescInput;
    int64_t shapeLen = shape.size();
    std::vector<int64_t> dimsInput = {shapeLen};
    ge::Shape shapeInput(dimsInput);
    tensorDescInput.SetShape(shapeInput);
    tensorDescInput.SetDataType(ge::DT_INT32);

    ge::Tensor tensor_shape(tensorDescInput, reinterpret_cast<uint8_t*>(shape.data()), shapeLen * sizeof(int));

    auto shape_op = op::Const((ori_name + "_input_shape").c_str()).set_attr_value(tensor_shape);
    auto random_op = op::RandomStandardNormal((ori_name + "_randomNormal").c_str())
                         .set_input_shape(shape_op)
                         .set_attr_dtype(kvlist[dtype])
                         .set_attr_seed(seed)
                         .set_attr_seed2(seed);

    auto mul_op = op::Muls((ori_name + "_mul").c_str()).set_input_x(random_op).set_attr_value(scale);
    auto add_op = op::Adds((ori_name + "_add").c_str()).set_input_x(mul_op).set_attr_value(mean);
    std::vector<ge::Operator> inputs{shape_op};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(add_op, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::RandomNormal"), ge::AscendString("ai.onnx::9::RandomNormal"),
         ge::AscendString("ai.onnx::10::RandomNormal"), ge::AscendString("ai.onnx::11::RandomNormal"),
         ge::AscendString("ai.onnx::12::RandomNormal"), ge::AscendString("ai.onnx::13::RandomNormal"),
         ge::AscendString("ai.onnx::14::RandomNormal"), ge::AscendString("ai.onnx::15::RandomNormal"),
         ge::AscendString("ai.onnx::16::RandomNormal"), ge::AscendString("ai.onnx::17::RandomNormal"),
         ge::AscendString("ai.onnx::18::RandomNormal")})
    .ParseParamsFn(ParseParamsRandomNormal)
    .ParseOpToGraphFn(ParseOpToGraphRandomNormal)
    .ImplyType(ImplyType::TVM);
} // namespace domi
