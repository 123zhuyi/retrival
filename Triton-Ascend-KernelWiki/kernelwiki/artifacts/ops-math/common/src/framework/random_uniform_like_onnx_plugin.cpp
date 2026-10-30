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
#include "math/mul/op_graph/mul_proto.h"
#include "math/add/op_graph/add_proto.h"
#include "math/cast/op_graph/cast_proto.h"
#include "math/shape/op_graph/shape_proto.h"

using namespace ge;
namespace domi {
static constexpr int KDtypeNotProvided = -1;
static Status ParseParamsRandomUniformLike(const Message* op_src, ge::Operator& op_dest)
{
    const ge::onnx::NodeProto* node = dynamic_cast<const ge::onnx::NodeProto*>(op_src);
    if (node == nullptr) {
        OP_LOGE(GetOpName(op_dest).c_str(), "Failed to dynamically cast op source to NodeProto.");
        return FAILED;
    }

    op_dest.DynamicInputRegister("x", 1);
    op_dest.DynamicOutputRegister("y", 1);
    op_dest.SetAttr("original_type", "ai.onnx::11::RandomUniformLike");

    int dtype = KDtypeNotProvided;
    float low = 0.0f;
    float high = 1.0f;
    int seed = 0;
    bool has_dtype_attr = false;
    for (const auto& attr : node->attribute()) {
        if (attr.name() == "dtype") {
            dtype = attr.i();
            has_dtype_attr = true;
        } else if (attr.name() == "high") {
            high = attr.f();
        } else if (attr.name() == "low") {
            low = attr.f();
        } else if (attr.name() == "seed") {
            seed = (int)attr.f();
        }
    }

    op_dest.SetAttr("name", node->name());
    op_dest.SetAttr("high", high);
    op_dest.SetAttr("low", low);
    op_dest.SetAttr("seed", seed);
    op_dest.SetAttr("dtype", dtype);
    op_dest.SetAttr("has_dtype_attr", has_dtype_attr);
    return SUCCESS;
}

static ge::DataType ResolveOutputDtype(const ge::Operator& op)
{
    int dtype = KDtypeNotProvided;
    op.GetAttr("dtype", dtype);
    bool has_dtype_attr = false;
    op.GetAttr("has_dtype_attr", has_dtype_attr);
    if (has_dtype_attr) {
        return GetOmDtypeFromOnnxDtype(dtype);
    }
    ge::TensorDesc input_desc = op.GetInputDesc(0);
    ge::DataType in_dtype = input_desc.GetDataType();
    if (in_dtype != ge::DT_UNDEFINED) {
        return in_dtype;
    }
    OP_LOGW(GetOpName(op).c_str(), "RandomUniformLike has no dtype attr.");
    return ge::DT_FLOAT;
}

static Status ParseOpToGraphRandomUniformLike(const ge::Operator& op, ge::Graph& graph)
{
    std::string ori_name;
    if (op.GetAttr("name", ori_name) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "Unable to retrieve identifier from operator.");
        return FAILED;
    }

    float low = 0.0f;
    op.GetAttr("low", low);

    float high = 1.0f;
    op.GetAttr("high", high);

    ge::DataType out_dtype = ResolveOutputDtype(op);
    const std::set<ge::DataType> supported_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE};
    if (supported_dtypes.find(out_dtype) == supported_dtypes.end()) {
        OP_LOGE(GetOpName(op).c_str(),
                "RandomUniformLike output dtype[%d] not supported, only support float/float16/double.",
                static_cast<int>(out_dtype));
        return FAILED;
    }

    int seed = 0;
    op.GetAttr("seed", seed);

    float delta = high - low;
    ge::Tensor scalar_mean = CreateScalar(low, ge::DT_FLOAT);
    ge::Tensor scalar_scale = CreateScalar(delta, ge::DT_FLOAT);

    auto data0 = op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
    auto shape_op = op::Shape((ori_name + "_shape").c_str()).set_input_x(data0).set_attr_dtype(ge::DT_INT32);
    auto random_op = op::RandomUniform((ori_name + "_random_uniform").c_str())
                         .set_input_shape(shape_op)
                         .set_attr_dtype(out_dtype)
                         .set_attr_seed(seed)
                         .set_attr_seed2(seed);
    auto const_scale = op::Const((ori_name + "_const_scale").c_str()).set_attr_value(scalar_scale);
    auto const_mean = op::Const((ori_name + "_const_mean").c_str()).set_attr_value(scalar_mean);
    auto cast_mean = op::Cast((ori_name + "_cast_mean").c_str()).set_input_x(const_mean).set_attr_dst_type(out_dtype);
    auto
        cast_scale = op::Cast((ori_name + "_cast_scale").c_str()).set_input_x(const_scale).set_attr_dst_type(out_dtype);

    auto mul_op = op::Mul((ori_name + "_mul").c_str()).set_input_x1(random_op).set_input_x2(cast_scale);
    auto add_op = op::Add((ori_name + "_add").c_str()).set_input_x1(mul_op).set_input_x2(cast_mean);
    std::vector<ge::Operator> inputs{data0};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(add_op, std::vector<std::size_t>{0});
    graph.SetInputs(inputs).SetOutputs(outputs);
    return SUCCESS;
}

// register Addcmul op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(ONNX)
    .OriginOpType(
        {ge::AscendString("ai.onnx::8::RandomUniformLike"), ge::AscendString("ai.onnx::9::RandomUniformLike"),
         ge::AscendString("ai.onnx::10::RandomUniformLike"), ge::AscendString("ai.onnx::11::RandomUniformLike"),
         ge::AscendString("ai.onnx::12::RandomUniformLike"), ge::AscendString("ai.onnx::13::RandomUniformLike"),
         ge::AscendString("ai.onnx::14::RandomUniformLike"), ge::AscendString("ai.onnx::15::RandomUniformLike"),
         ge::AscendString("ai.onnx::16::RandomUniformLike"), ge::AscendString("ai.onnx::17::RandomUniformLike"),
         ge::AscendString("ai.onnx::18::RandomUniformLike")})
    .ParseParamsFn(ParseParamsRandomUniformLike)
    .ParseOpToGraphFn(ParseOpToGraphRandomUniformLike)
    .ImplyType(ImplyType::TVM);
} // namespace domi
